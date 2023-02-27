import argparse
import copy
import json
import os
from multiprocessing import Pool

import tsplib95

from time import time
from helpers import (
    load_weights,
    create_fuzzy_weights,
    print_results,
    calculate_all_routes, Lambda, argmin, DEFAULT_PROCESS_AMOUNT,
)
from ftsp import estimate_average_route_distance

available_problems = ["ulysses16", "gr48", "st70", "pr76", "rd100", "rd400"]

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
    "-t",
    "--threads",
    default=0,
    type=int,
    help="Amount of threads to enable for routes calculations",
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
    """Run the project."""

    start_ts = time()
    args = command_parser.parse_args()

    tsp_problem_name = args.problem
    data = tsplib95.load(f"problems/{tsp_problem_name}.tsp")
    fuzzy_weights_filename = f"weights/fuzzy-weights-{tsp_problem_name}.json"
    crisp_weights_filename = f"weights/fuzzy-weights-{tsp_problem_name}.json"
    all_routes_filename = f"solutions/all-routes-{tsp_problem_name}.json"
    best_routes_filename = f"solutions/best-routes-{tsp_problem_name}.json"

    if args.ignore_weights or not os.path.exists(fuzzy_weights_filename):
        cities, weights = create_fuzzy_weights(
            data, (5, 35), fuzzy_weights_filename, crisp_weights_filename
        )
    else:
        weights = load_weights(fuzzy_weights_filename)
        min_city = min(data.get_nodes())
        cities = [city - min_city + 1 for city in data.get_nodes()]

    min_city = min(data.get_nodes())
    city_shift = cities[0] - min_city

    if args.load_best_route:
        with open(best_routes_filename, "r") as rf:
            routes_by_fuzziness_types = json.load(rf)
    elif args.load_all_routes:
        with open(all_routes_filename, "r") as rf:
            all_routes_by_methods = json.load(rf)
            routes_by_fuzziness_types = dict()
            for fuzziness_type, routes in all_routes_by_methods.items():
                with Pool(args.threads or DEFAULT_PROCESS_AMOUNT) as p:
                    best_result_idx = argmin(
                        p.map(
                            Lambda(estimate_average_route_distance, weights, fuzziness_type),
                            routes,
                        )
                    )
                routes_by_fuzziness_types[fuzziness_type] = routes[best_result_idx]
    else:
        routes_by_fuzziness_types, all_routes = calculate_all_routes(
            copy.deepcopy(cities),
            weights,
            args.solutions,
            args.random_initial_route,
            args.threads,
        )
        with open(all_routes_filename, "w") as rf:
            json.dump(all_routes, rf, indent=4)

        with open(best_routes_filename, "w") as rf:
            json.dump(routes_by_fuzziness_types, rf, indent=4)

    print_results(
        data, weights, routes_by_fuzziness_types, city_shift, args.evaluations
    )

    print(f"Time elapsed: {time() - start_ts:.2f}s")


if __name__ == "__main__":
    main()
