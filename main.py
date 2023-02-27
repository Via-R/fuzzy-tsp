import argparse
import copy
import json
import os
import tsplib95
import matplotlib.pyplot as plt

from random import shuffle
from helpers import load_weights, create_fuzzy_weights
from ftsp import annealing, estimate_average_route_distance

available_problems = ["ulysses16", "gr48", "pr76", "rd100", "rd400"]

command_parser = argparse.ArgumentParser(
    prog="FTSP solution analyser",
    description="""This program solves FTSP, considering weights (subjective/objective time, distance, etc.)
        between cities as fuzzy numbers of one of three available types:
        crisp, triangular, parabolic. 
        
        It builds multiple solutions per one of the fuzzy number types,
        and then chooses the best one. After that, it performs random
        evaluations of objective function (sum of estimated weights)
        and outputs comparison of expected and actual values of objective
        function, along with graphs depicting found routes if possible.
        
        When generating new weights, they are based on well-known TSP, and
        fuzzy weights are randomly generated using default crisp weights as
        x values for peak fuzzy values with 5% max divergence to the left
        and 35% max divergence to the right.""".replace(
        "    ", " "
    ),
    epilog="Author: Vadym Rets (vadym.rets@gmail.com)",
)
command_parser.add_argument(
    "-p",
    "--problem",
    default=available_problems[0],
    choices=available_problems,
    help="Name of the problem to solve (show solution for)",
)
command_parser.add_argument(
    "-s",
    "--solutions",
    default=30,
    type=int,
    help="Amount of solutions to calculate for each fuzzy number type (in the end the best one is selected)",
)
command_parser.add_argument(
    "-e",
    "--evaluations",
    default=100_000,
    type=int,
    help="Amount of random evaluations to do when approximating objective function on selected route",
)
command_parser.add_argument(
    "--load-all-routes",
    action="store_true",
    default=False,
    help="Preload available solutions and evaluate the best one instead of creating new ones",
)
command_parser.add_argument(
    "--load-best-route",
    action="store_true",
    default=False,
    help="Preload the best solutions instead of creating new ones",
)
command_parser.add_argument(
    "--ignore-weights",
    action="store_true",
    default=False,
    help="Ignore existing fuzzy weights and generate new ones for selected problem",
)
command_parser.add_argument(
    "--random-initial-route",
    action="store_true",
    default=False,
    help="Use randomly shuffled initial route instead of a consecutive one",
)


def main() -> None:
    args = command_parser.parse_args()

    tsp_problem_name = args.problem
    data = tsplib95.load(f"problems/{tsp_problem_name}.tsp")
    fuzzy_weights_filename = f"weights/fuzzy-weights-{tsp_problem_name}.json"
    all_routes_filename = f"solutions/all-routes-{tsp_problem_name}.json"
    best_routes_filename = f"solutions/best-routes-{tsp_problem_name}.json"

    if args.ignore_weights or not os.path.exists(fuzzy_weights_filename):
        cities, weights = create_fuzzy_weights(data, tsp_problem_name, (5, 35))
    else:
        weights = load_weights(fuzzy_weights_filename)
        min_city = min(data.get_nodes())
        cities = [city - min_city + 1 for city in data.get_nodes()]

    min_city = min(data.get_nodes())
    city_shift = cities[0] - min_city

    shuffled_cities = copy.deepcopy(cities)

    if args.load_best_route:
        with open(best_routes_filename, "r") as rf:
            routes_by_fuzziness_types = json.load(rf)
    elif args.load_all_routes:
        with open(all_routes_filename, "r") as rf:
            all_routes_by_methods = json.load(rf)
            routes_by_fuzziness_types = dict()
            for fuzziness_type, routes in all_routes_by_methods.items():
                best_result = min(
                    routes,
                    key=lambda r: estimate_average_route_distance(
                        r, weights, fuzziness_type
                    ),
                )
                routes_by_fuzziness_types[fuzziness_type] = best_result
    else:
        iterations = 100
        fuzziness_types = ["crisp", "triangular", "parabolic"]
        methods_results = dict()
        all_routes = dict()
        for fuzziness_type in fuzziness_types:
            routes = []
            for it in range(iterations):
                if args.random_initial_route:
                    shuffle(shuffled_cities)
                print(f"{fuzziness_type=}, it: {it + 1}/{iterations}")
                routes.append(
                    annealing(shuffled_cities, weights, fuzziness_type, "rank")
                )

            all_routes[fuzziness_type] = routes
            best_result = min(
                routes,
                key=lambda r: estimate_average_route_distance(
                    r, weights, fuzziness_type
                ),
            )
            methods_results[fuzziness_type] = best_result

        with open(all_routes_filename, "w") as rf:
            json.dump(all_routes, rf, indent=4)

        with open(best_routes_filename, "w") as rf:
            json.dump(methods_results, rf, indent=4)

        routes_by_fuzziness_types = methods_results

    fig, axs = plt.subplots(2, 2)
    fig.canvas.manager.set_window_title("FTSP Solutions")

    for ax in axs.flat:
        ax.set(xlabel="x-label", ylabel="y-label")

    for ax in axs.flat:
        ax.label_outer()
    colors = ["blue", "orange", "red"]
    plot_coords = [(0, 0), (0, 1), (1, 0)]
    for idx, (fuzziness_type, route) in enumerate(routes_by_fuzziness_types.items()):
        crisp_sampling = estimate_average_route_distance(
            route, weights, "crisp", args.evaluations
        )
        realistic_sampling = estimate_average_route_distance(
            route, weights, "parabolic", args.evaluations
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


if __name__ == "__main__":
    main()
