import random
import math
import time
import copy
from cmath import sqrt

from helpers import (
    Route,
    Weights,
    FuzzyNumber,
)


# Methods for BOA method of TFNs
def left_triangular_rank(a: float, b: float, c: float) -> float:
    return 1 / 2 * (2 * a + math.sqrt(2) * math.sqrt((a - b) * (a - c)))


def right_triangular_rank(a: float, b: float, c: float) -> float:
    return 1 / 2 * (2 * c - math.sqrt(2) * math.sqrt((c - a) * (c - b)))


# Methods for BOA method of PFNs
def left_parabolic_rank(a: float, b: float, c: float) -> float:
    x = (
        -(
            (1 + 1j * sqrt(3))
            * (
                -27 * a**3
                + 108 * a**2 * b
                - 27 * a**2 * c
                + sqrt(
                    4 * (-9 * a**2 + 18 * a * b - 9 * b**2) ** 3
                    + (
                        -27 * a**3
                        + 108 * a**2 * b
                        - 27 * a**2 * c
                        - 135 * a * b**2
                        + 54 * a * b * c
                        + 54 * b**3
                        - 27 * b**2 * c
                    )
                    ** 2
                )
                - 135 * a * b**2
                + 54 * a * b * c
                + 54 * b**3
                - 27 * b**2 * c
            )
            ** (1 / 3)
        )
        / (6 * 2 ** (1 / 3))
        + ((1 - 1j * sqrt(3)) * (-9 * a**2 + 18 * a * b - 9 * b**2))
        / (
            3
            * 2 ** (2 / 3)
            * (
                -27 * a**3
                + 108 * a**2 * b
                - 27 * a**2 * c
                + sqrt(
                    4 * (-9 * a**2 + 18 * a * b - 9 * b**2) ** 3
                    + (
                        -27 * a**3
                        + 108 * a**2 * b
                        - 27 * a**2 * c
                        - 135 * a * b**2
                        + 54 * a * b * c
                        + 54 * b**3
                        - 27 * b**2 * c
                    )
                    ** 2
                )
                - 135 * a * b**2
                + 54 * a * b * c
                + 54 * b**3
                - 27 * b**2 * c
            )
            ** (1 / 3)
        )
        + b
    )

    if x.imag > 1e-10:
        raise Exception(f"Parabolic approximation is not precise enough, {x.imag=}")

    return x.real


def right_parabolic_rank(a: float, b: float, c: float) -> float:
    x = (
        -(
            (1 + 1j * sqrt(3))
            * (
                -27 * a * b**2
                + sqrt(
                    (
                        -27 * a * b**2
                        + 54 * a * b * c
                        - 27 * a * c**2
                        + 54 * b**3
                        - 135 * b**2 * c
                        + 108 * b * c**2
                        - 27 * c**3
                    )
                    ** 2
                    + 4 * (-9 * b**2 + 18 * b * c - 9 * c**2) ** 3
                )
                + 54 * a * b * c
                - 27 * a * c**2
                + 54 * b**3
                - 135 * b**2 * c
                + 108 * b * c**2
                - 27 * c**3
            )
            ** (1 / 3)
        )
        / (6 * 2 ** (1 / 3))
        + ((1 - 1j * sqrt(3)) * (-9 * b**2 + 18 * b * c - 9 * c**2))
        / (
            3
            * 2 ** (2 / 3)
            * (
                -27 * a * b**2
                + sqrt(
                    (
                        -27 * a * b**2
                        + 54 * a * b * c
                        - 27 * a * c**2
                        + 54 * b**3
                        - 135 * b**2 * c
                        + 108 * b * c**2
                        - 27 * c**3
                    )
                    ** 2
                    + 4 * (-9 * b**2 + 18 * b * c - 9 * c**2) ** 3
                )
                + 54 * a * b * c
                - 27 * a * c**2
                + 54 * b**3
                - 135 * b**2 * c
                + 108 * b * c**2
                - 27 * c**3
            )
            ** (1 / 3)
        )
        + b
    )

    if x.imag > 1e-10:
        raise Exception(f"Parabolic approximation is not precise enough, {x.imag=}")

    return x.real


def approximate_fuzzy_number(fuzzy_number: FuzzyNumber, method: str) -> float:
    a, b, c = fuzzy_number

    if a == b and b == c or method == "crisp":
        return b

    if method == "triangular_rank":
        # Centroids circumference method
        # return (a + 4 * b + c) / 6

        # Bisector of Area (BOA) method
        # if b - a > c - b:
        #     return left_triangular_rank(a, b, c)
        # if b - a < c - b:
        #     return right_triangular_rank(a, b, c)
        #
        # return b

        # Center of Gravity (COG) method
        return (a + b + c) / 3

    # generate random number in [a;c]x[0;1] until it's under membership function
    if method == "triangular_approximation":
        while True:
            rand_x, rand_y = random.random() * (c - a) + a, random.random()
            sample_y = (rand_x - a) / (b - a) if rand_x <= b else (rand_x - c) / (b - c)
            if rand_y <= sample_y:
                break

        return rand_x

    if method == "parabolic_rank":
        # Center of Gravity (COG) method
        return (3 * a + 2 * b + 3 * c) / 8

        # Bisector of Area (BOA) method
        # if b - a > c - b:
        #     return left_parabolic_rank(a, b, c)
        #
        # elif b - a < c - b:
        #     return right_parabolic_rank(a, b, c)
        #
        # return b

    # create random number in [a;c]x[0;1] until it's under membership function
    if method == "parabolic_approximation":
        while True:
            rand_x, rand_y = random.random() * (c - a) + a, random.random()
            sample_y = (
                -(((rand_x - b) / (b - a)) ** 2) + 1
                if rand_x < b
                else -(((rand_x - b) / (c - b)) ** 2) + 1
            )
            if rand_y <= sample_y:
                break

        return rand_x

    raise Exception("Wrong weight_approximation type")


def fuzzy_sum(a: FuzzyNumber, b: FuzzyNumber) -> FuzzyNumber:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def get_cost(
    route: Route,
    weights: Weights,
    weight_approximation: str,
    partial_approximation=True,
) -> float:
    """Calculate fitness for the route by given weight approximation method."""

    distance: FuzzyNumber = [0, 0, 0]

    for i in range(len(route)):
        from_city = route[i]
        if i + 1 < len(route):
            to_city = route[i + 1]
        else:
            to_city = route[0]

        path_length = (
            [
                approximate_fuzzy_number(
                    weights[from_city - 1][to_city - 1], weight_approximation
                )
            ]
            * 3
            if partial_approximation
            else weights[from_city - 1][to_city - 1]
        )
        distance = fuzzy_sum(distance, path_length)

    fitness = 1 / float(approximate_fuzzy_number(distance, weight_approximation))

    return fitness


def get_distance(
    route: Route,
    weights: Weights,
    weight_approximation: str,
    partial_approximation=False,
) -> float:
    """Get distance by given weight approximation method."""

    return 1 / get_cost(route, weights, weight_approximation, partial_approximation)


def inverse(route: Route) -> Route:
    """Inverse the order of cities in a route between city a and city b."""

    city_a = random.choice(route)
    city_b = random.choice([city for city in route if city != city_a])
    route[min(city_a, city_b) : max(city_a, city_b)] = route[
        min(city_a, city_b) : max(city_a, city_b)
    ][::-1]

    return route


def insert(route: Route) -> Route:
    """Move city a before city b."""

    city_a = random.choice(route)
    route.remove(city_a)
    city_b = random.choice(route)
    index = route.index(city_b)
    route.insert(index, city_a)

    return route


def swap_cities(route: Route) -> Route:
    """Swap cities at positions i and j."""

    city_a = random.choice(route)
    i = route.index(city_a)
    j = route.index(random.choice([city for city in route if city != city_a]))
    route[i], route[j] = route[j], route[i]

    return route


def swap_routes(route: Route) -> Route:
    """Select a subroute from city a to city b and insert it at another position in the route."""

    city_a = random.choice(route)
    city_b = random.choice([city for city in route if city != city_a])
    subroute = route[min(city_a, city_b) : max(city_a, city_b)]
    del route[min(city_a, city_b) : max(city_a, city_b)]

    insert_position = random.choice(range(len(route)))
    route = route[:insert_position] + subroute + route[insert_position:]

    return route


def get_neighboring_route(route: Route) -> Route:
    """Return neighbor of given solution."""

    neighbor = copy.deepcopy(route)

    return random.choice([inverse, insert, swap_cities, swap_routes])(neighbor)


def annealing(
    initial_state: Route, weights: Weights, weight_approximation="precise"
) -> Route:
    """Perform simulated annealing to find a solution."""

    start = time.time()
    initial_temp: float = 5000.0
    alpha = 0.99
    current_temp = initial_temp
    solution = initial_state
    same_solution = 0
    same_cost_diff = 0

    while same_solution < 1500 and same_cost_diff < 50000:  # originally 150k
        neighbor = get_neighboring_route(solution)

        cost_diff = get_cost(neighbor, weights, weight_approximation) - get_cost(
            solution, weights, weight_approximation
        )

        # if the new solution has (almost) the same cost, accept it
        if abs(cost_diff) < 1e-8:
            solution = neighbor
            same_solution = 0
            same_cost_diff += 1

        # if the new solution is better, accept it
        elif cost_diff > 0:
            solution = neighbor
            same_solution = 0
            same_cost_diff = 0

        # if the new solution is not better, accept it with a set probability
        else:
            if random.uniform(0, 1) <= math.exp(cost_diff / current_temp):
                solution = neighbor
                same_solution = 0
                same_cost_diff = 0
            else:
                same_solution += 1
                same_cost_diff += 1

        # print(f"{get_cost(solution, weights, weight_approximation)=}")
        # print(f"{same_solution=}, {same_cost_diff=}")

        current_temp = current_temp * alpha

    print(f"{weight_approximation=}, {time.time() - start:.4f}s")

    return solution


def estimate_average_route_distance(
    route: Route, weights, method: str, sampling_size: int = 100_000
):
    return (
        sum(
            [
                get_distance(route, weights, method, partial_approximation=False)
                for _ in range(sampling_size)
            ]
        )
        / sampling_size
    )
