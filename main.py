import copy
import json
import os
import tsplib95
import matplotlib.pyplot as plt

from random import shuffle
from helpers import load_weights, create_fuzzy_weights
from tsp import annealing, estimate_average_route_distance, get_distance

# tsp_problem_name = 'rd400'
# tsp_problem_name = 'rd100'
tsp_problem_name = 'pr76'
# tsp_problem_name = "ulysses16"
# tsp_problem_name = 'gr48'
data = tsplib95.load(f"problems/{tsp_problem_name}.tsp")
fuzzy_weights_filename = f"weights/fuzzy-weights-{tsp_problem_name}.json"
all_routes_filename = f"solutions/all-routes-{tsp_problem_name}.json"
best_routes_filename = f"solutions/best-routes-{tsp_problem_name}.json"


def main(load_all_routes=False, load_best_route=False, ignore_weights=False):
    if ignore_weights or not os.path.exists(fuzzy_weights_filename):
        cities, weights = create_fuzzy_weights(data, tsp_problem_name, (5, 35))
    else:
        weights = load_weights(fuzzy_weights_filename)
        min_city = min(data.get_nodes())
        cities = [city - min_city + 1 for city in data.get_nodes()]

    min_city = min(data.get_nodes())
    city_shift = cities[0] - min_city

    shuffled_cities = copy.deepcopy(cities)

    if load_best_route:
        with open(f"routes-{tsp_problem_name}.json", "r") as rf:
            routes_by_methods = json.load(rf)
    elif load_all_routes:
        with open(f"all-routes-{tsp_problem_name}.json", "r") as rf:
            all_routes_by_methods = json.load(rf)
            routes_by_methods = dict()
            for method, routes in all_routes_by_methods.items():
                if method == "crisp":
                    apprx_method = "crisp"
                elif method == "triangular_rank":
                    apprx_method = "triangular_approximation"
                else:
                    apprx_method = "parabolic_approximation"
                best_result = min(
                    routes,
                    key=lambda r: estimate_average_route_distance(
                        r, weights, apprx_method
                    ),
                )
                routes_by_methods[method] = best_result
    else:
        iterations = 100
        methods = ["crisp", "triangular_rank", "parabolic_rank"]
        methods_results = dict()
        all_routes = dict()
        for method in methods:
            routes = []
            for it in range(iterations):
                # shuffle(shuffled_cities)
                print(f"{method=}, it: {it + 1}/{iterations}")
                routes.append(annealing(shuffled_cities, weights, method))

            if method == "crisp":
                apprx_method = "crisp"
            elif method == "triangular_rank":
                apprx_method = "triangular_approximation"
            else:
                apprx_method = "parabolic_approximation"
            all_routes[method] = routes
            best_result = min(
                routes,
                key=lambda r: estimate_average_route_distance(r, weights, apprx_method),
            )
            methods_results[method] = best_result

        with open(all_routes_filename, "w") as rf:
            json.dump(all_routes, rf, indent=4)

        with open(best_routes_filename, "w") as rf:
            json.dump(methods_results, rf, indent=4)

        routes_by_methods = methods_results

    fig, axs = plt.subplots(2, 2)

    for ax in axs.flat:
        ax.set(xlabel="x-label", ylabel="y-label")

    for ax in axs.flat:
        ax.label_outer()
    colors = ["blue", "orange", "red"]
    plot_coords = [(0, 0), (0, 1), (1, 0)]
    sampling_size = 100_000
    for idx, (method, route) in enumerate(routes_by_methods.items()):
        crisp_sampling = estimate_average_route_distance(
            route, weights, "crisp", sampling_size
        )
        realistic_sampling = estimate_average_route_distance(
            route, weights, "parabolic_approximation", sampling_size
        )

        print(f"For {method=} {crisp_sampling=}, {realistic_sampling=}")

        if data.node_coords == {}:
            continue

        xs = [data.node_coords[i - city_shift][0] for i in route + [route[0]]]
        ys = [data.node_coords[i - city_shift][1] for i in route + [route[0]]]
        axs[plot_coords[idx][0], plot_coords[idx][1]].plot(xs, ys, color=colors[idx])
        axs[plot_coords[idx][0], plot_coords[idx][1]].scatter(
            xs[1:-2], ys[1:-2], color="grey"
        )
        axs[plot_coords[idx][0], plot_coords[idx][1]].scatter(
            xs[-2], ys[-2], s=120, color="green"
        )
        axs[plot_coords[idx][0], plot_coords[idx][1]].scatter(
            xs[-1], ys[-1], s=120, color="red"
        )
        axs[plot_coords[idx][0], plot_coords[idx][1]].set_title(method)

    if data.node_coords == {}:
        return

    for ax in axs.flat:
        ax.set(xlabel="x coordinates", ylabel="y coordinates")

    for ax in axs.flat:
        ax.label_outer()

    plt.show()


main(load_all_routes=False, load_best_route=False, ignore_weights=False)
