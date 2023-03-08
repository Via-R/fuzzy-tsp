import json
import matplotlib.pyplot as plt

from time import time
from random import randint
from typing import List, Tuple, Dict, Any
from multiprocessing import Pool
from random import shuffle
from ftsp import (
    Weights,
    CrispWeights,
    estimate_average_route_distance,
    annealing,
    Route,
)

BestRoutesByFuzzinessMethods = Dict[str, Route]
AllRoutesByFuzzinessMethods = Dict[str, List[Route]]
DEFAULT_PROCESS_AMOUNT = 4


class Lambda(object):
    """Custom implementation of lambda to work with multiprocessing Pool's map."""

    def __init__(self, f, *args):
        self.f = f
        self.args = args

    def __call__(self, src):
        return self.f(src, *self.args)


def argmin(l: List[Any]) -> int:
    """Return idx of the min element of input list."""

    return min(range(len(l)), key=lambda x: l[x])


def _get_weights_from_data(data) -> Weights:
    """Extract all weights from selected tsplib95 data."""

    cities = list(data.get_nodes())
    weights = [[None] * len(cities) for _ in range(len(cities))]

    for city_from in cities:
        for city_to in cities:
            weights[city_to - 1][city_from - 1] = data.get_weight(city_from, city_to)

    return weights


def _save_weights(weights: Weights, weights_filename: str) -> None:
    """Save weights to file."""

    with open(weights_filename, "w") as f:
        json.dump(weights, f, indent=4)


def load_weights(weights_filename: str) -> Weights:
    """Load weights from file."""

    with open(weights_filename, "r") as f:
        return json.load(f)


def _generate_fuzzy_weights(
    cities: List[int],
    deviation: Tuple[int, int],
    crisp_weights: CrispWeights,
) -> Tuple[Weights, Weights]:
    """Convert regular tsplib95 weights to randomly generated fuzzy weights."""

    fuzzy_weights: Weights = [[None] * len(cities) for _ in range(max(cities))]
    fuzzy_crisp_weights: Weights = [[None] * len(cities) for _ in range(max(cities))]
    dev_left_percent, dev_right_percent = deviation
    for left_city in cities:
        for right_city in cities:
            if right_city == left_city:
                fuzzy_weights[left_city - 1][right_city - 1] = [0, 0, 0]
                fuzzy_crisp_weights[left_city - 1][right_city - 1] = [0, 0, 0]
                continue
            dev_left, dev_right = (
                randint(0, dev_left_percent * 100) / 100,
                randint(0, dev_right_percent * 100) / 100,
            )
            precise_weight = crisp_weights[left_city - 1][right_city - 1]
            fuzzy_weights[left_city - 1][right_city - 1] = [
                precise_weight * (100 - dev_left) / 100,
                precise_weight,
                precise_weight * (100 + dev_right) / 100,
            ]
            fuzzy_crisp_weights[left_city - 1][right_city - 1] = [precise_weight] * 3

    return fuzzy_weights, fuzzy_crisp_weights


def create_fuzzy_weights(
    data,
    deviation: Tuple[int, int],
    fuzzy_weights_filename: str,
    crisp_weights_filename: str,
) -> Tuple[List[int], Weights]:
    """Generate fuzzy weights from tsplib95 data."""

    cities = list(data.get_nodes())
    min_city = min(cities)
    cities = [city - min_city + 1 for city in cities]

    crisp_weights = _get_weights_from_data(data)
    _save_weights(crisp_weights, crisp_weights_filename)
    fuzzy_weights, fuzzy_crisp_weights = _generate_fuzzy_weights(
        cities, deviation, crisp_weights
    )
    _save_weights(fuzzy_weights, fuzzy_weights_filename)

    return cities, fuzzy_weights


def _route_calculator(
    it_idx: int,
    iterations: int,
    initial_route: Route,
    weights: Weights,
    fuzziness_type: str,
    shuffle_initial_route: bool,
) -> Route:
    """Calculate one solution, used for multiprocessing."""

    print(f"{fuzziness_type=}, it: {it_idx + 1}/{iterations} | start")

    if shuffle_initial_route:
        shuffle(initial_route)

    start_ts = time()

    result = annealing(initial_route, weights, fuzziness_type, "rank")

    print(
        f"{fuzziness_type=}, it: {it_idx + 1}/{iterations} | end {time() - start_ts:.4f}s"
    )

    return result


def calculate_all_routes(
    initial_route: Route,
    weights: Weights,
    iterations: int,
    shuffle_initial_route: bool,
    threads: int,
) -> Tuple[BestRoutesByFuzzinessMethods, AllRoutesByFuzzinessMethods]:
    """Calculate specified amount of routes per each method and choose the best.

    Return both the best routes and all routes as a tuple."""

    fuzziness_types = ["crisp", "triangular", "parabolic"]
    routes_by_fuzziness_types = dict()
    all_routes = dict()
    for fuzziness_type in fuzziness_types:
        with Pool(threads or DEFAULT_PROCESS_AMOUNT) as p:
            routes: List[Route] = p.map(
                Lambda(
                    _route_calculator,
                    iterations,
                    initial_route,
                    weights,
                    fuzziness_type,
                    shuffle_initial_route,
                ),
                range(iterations),
            )

            all_routes[fuzziness_type] = routes
            best_result_idx = argmin(
                p.map(
                    Lambda(estimate_average_route_distance, weights, fuzziness_type),
                    routes,
                )
            )
        routes_by_fuzziness_types[fuzziness_type] = routes[best_result_idx]

    return routes_by_fuzziness_types, all_routes


def print_results(
    data,
    weights: Weights,
    routes_by_fuzziness_types: BestRoutesByFuzzinessMethods,
    city_shift: int,
    evaluations: int,
) -> None:
    """Print out calculation results and draw graphs if coordinates were available."""

    fig, axs = plt.subplots(2, 2)
    fig.canvas.manager.set_window_title("FTSP Solutions")

    for ax in axs.flat:
        ax.set(xlabel="x-label", ylabel="y-label")

    for ax in axs.flat:
        ax.label_outer()
    colors = ["blue", "orange", "red"]
    plot_coords = [(0, 1), (1, 0), (1, 1)]
    all_cities_xs = [data.node_coords[i - city_shift][0] for i in data.get_nodes()]
    all_cities_ys = [data.node_coords[i - city_shift][1] for i in data.get_nodes()]
    axs[0, 0].set_title("cities")
    axs[0, 0].scatter(all_cities_xs, all_cities_ys, color="grey")
    for idx, (fuzziness_type, route) in enumerate(routes_by_fuzziness_types.items()):
        crisp_sampling = estimate_average_route_distance(
            route, weights, "crisp", evaluations
        )
        realistic_sampling = estimate_average_route_distance(
            route, weights, "parabolic", evaluations
        )

        print(f"For fuzziness type '{fuzziness_type}':")
        print(f"Naive objective function value = {crisp_sampling}")
        print(f"Realistic objective function value = {realistic_sampling}", end="\n\n")

        if data.node_coords == {}:
            continue

        xs = [data.node_coords[i - city_shift][0] for i in route + [route[0]]]
        ys = [data.node_coords[i - city_shift][1] for i in route + [route[0]]]
        axs[plot_coords[idx][0], plot_coords[idx][1]].plot(xs, ys, color=colors[idx])
        axs[plot_coords[idx][0], plot_coords[idx][1]].scatter(
            xs[1:-2], ys[1:-2], color="grey"
        )
        axs[plot_coords[idx][0], plot_coords[idx][1]].scatter(
            xs[0], ys[0], s=120, color="green"
        )
        axs[plot_coords[idx][0], plot_coords[idx][1]].scatter(
            xs[-2], ys[-2], s=120, color="red"
        )
        axs[plot_coords[idx][0], plot_coords[idx][1]].set_title(fuzziness_type)

    if data.node_coords == {}:
        return

    for ax in axs.flat:
        ax.set(xlabel="x coordinates", ylabel="y coordinates")

    for ax in axs.flat:
        ax.label_outer()

    plt.show()
