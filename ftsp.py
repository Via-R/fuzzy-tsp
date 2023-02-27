import random
import math
import copy
from cmath import sqrt
from typing import List, Optional

Route = List[int]
FuzzyNumber = List[float]
CrispWeights = List[List[Optional[float]]]
Weights = List[List[Optional[FuzzyNumber]]]


def triangular_boa_rank(f: FuzzyNumber) -> float:
    """Calculate Bisector of Area rank of given TFN."""

    a, b, c = f

    def left_triangular_rank() -> float:
        """Calculate left BOA rank (a < x <= b)."""

        return 1 / 2 * (2 * a + math.sqrt(2) * math.sqrt((a - b) * (a - c)))

    def right_triangular_rank() -> float:
        """Calculate right BOA rank (b < x < c)."""

        return 1 / 2 * (2 * c - math.sqrt(2) * math.sqrt((c - a) * (c - b)))

    if b - a > c - b:
        return left_triangular_rank()
    if b - a < c - b:
        return right_triangular_rank()

    return b


def triangular_cog_rank(f: FuzzyNumber) -> float:
    """Calculate Center of Gravity rank of given TFN."""

    return sum(f) / 3


def parabolic_boa_rank(f: FuzzyNumber) -> float:
    """Calculate Bisector of Area rank of given PFN."""

    a, b, c = f

    def left_parabolic_rank() -> float:
        """Calculate left BOA rank (a < x <= b)."""

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

    def right_parabolic_rank() -> float:
        """Calculate right BOA rank (b < x < c)."""

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

    if b - a > c - b:
        return left_parabolic_rank()
    if b - a < c - b:
        return right_parabolic_rank()

    return b


def parabolic_cog_rank(f: FuzzyNumber) -> float:
    """Calculate Center of Gravity rank of given PFN."""

    a, b, c = f

    return (3 * a + 2 * b + 3 * c) / 8


def crisp_rank(f: FuzzyNumber) -> float:
    """Calculate crisp rank of given TFN/PFN."""

    return f[1]


def triangular_approximation(f: FuzzyNumber) -> float:
    """Get random TFN rank using normalized fuzzy number form as density function."""

    a, b, c = f

    while True:
        rand_x, rand_y = random.random() * (c - a) + a, random.random()
        sample_y = (rand_x - a) / (b - a) if rand_x <= b else (rand_x - c) / (b - c)
        if rand_y <= sample_y:
            return rand_x


def parabolic_approximation(f: FuzzyNumber) -> float:
    """Get random PFN rank using normalized fuzzy number form as density function."""

    a, b, c = f

    while True:
        rand_x, rand_y = random.random() * (c - a) + a, random.random()
        sample_y = (
            -(((rand_x - b) / (b - a)) ** 2) + 1
            if rand_x < b
            else -(((rand_x - b) / (c - b)) ** 2) + 1
        )
        if rand_y <= sample_y:
            return rand_x


DEFUZZIFICATION_METHODS = {
    "crisp": {"approximation": crisp_rank, "rank": crisp_rank},
    "triangular": {
        "approximation": triangular_approximation,
        "rank": triangular_cog_rank,
    },
    "parabolic": {"approximation": parabolic_approximation, "rank": parabolic_cog_rank},
}
DEFAULT_FUZZINESS_TYPE = "parabolic"


def _get_defuzzification_method(fuzziness_type: str, defuzzification_type: str):
    if defuzzification_type not in ["rank", "approximation"]:
        raise Exception("Wrong defuzzification method name")

    return (
        DEFUZZIFICATION_METHODS.get(fuzziness_type)
        or DEFUZZIFICATION_METHODS[DEFAULT_FUZZINESS_TYPE]
    )[defuzzification_type]


def approximate_fuzzy_number(
    fuzzy_number: FuzzyNumber, fuzziness_type: str, defuzzification_type: str
) -> float:
    """Calculate rank or approximation of a fuzzy number."""

    a, b, c = fuzzy_number

    if a == b and b == c:
        return b

    return _get_defuzzification_method(fuzziness_type, defuzzification_type)(
        fuzzy_number
    )


def fuzzy_sum(a: FuzzyNumber, b: FuzzyNumber) -> FuzzyNumber:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def get_cost(
    route: Route,
    weights: Weights,
    fuzziness_type: str,
    defuzzification_type: str,
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
                    weights[from_city - 1][to_city - 1],
                    fuzziness_type,
                    defuzzification_type,
                )
            ]
            * 3
            if partial_approximation
            else weights[from_city - 1][to_city - 1]
        )
        distance = fuzzy_sum(distance, path_length)

    fitness = 1 / float(
        approximate_fuzzy_number(distance, fuzziness_type, defuzzification_type)
    )

    return fitness


def get_distance(
    route: Route,
    weights: Weights,
    fuzziness_type: str,
    defuzzification_type: str,
    partial_approximation=False,
) -> float:
    """Get distance by given weight approximation method."""

    return 1 / get_cost(
        route, weights, fuzziness_type, defuzzification_type, partial_approximation
    )


def _inverse(route: Route) -> Route:
    """Inverse the order of cities in a route between city a and city b."""

    city_a = random.choice(route)
    city_b = random.choice([city for city in route if city != city_a])
    route[min(city_a, city_b) : max(city_a, city_b)] = route[
        min(city_a, city_b) : max(city_a, city_b)
    ][::-1]

    return route


def _insert(route: Route) -> Route:
    """Move city a before city b."""

    city_a = random.choice(route)
    route.remove(city_a)
    city_b = random.choice(route)
    index = route.index(city_b)
    route.insert(index, city_a)

    return route


def _swap_cities(route: Route) -> Route:
    """Swap cities at positions i and j."""

    city_a = random.choice(route)
    i = route.index(city_a)
    j = route.index(random.choice([city for city in route if city != city_a]))
    route[i], route[j] = route[j], route[i]

    return route


def _swap_routes(route: Route) -> Route:
    """Select a subroute from city a to city b and insert it at another position in the route."""

    city_a = random.choice(route)
    city_b = random.choice([city for city in route if city != city_a])
    subroute = route[min(city_a, city_b) : max(city_a, city_b)]
    del route[min(city_a, city_b) : max(city_a, city_b)]

    insert_position = random.choice(range(len(route)))
    route = route[:insert_position] + subroute + route[insert_position:]

    return route


def _get_neighboring_route(route: Route) -> Route:
    """Return neighbor of given solution."""

    neighbor = copy.deepcopy(route)

    return random.choice([_inverse, _insert, _swap_cities, _swap_routes])(neighbor)


def annealing(
    initial_state: Route,
    weights: Weights,
    fuzziness_type: str,
    defuzzification_type: str,
) -> Route:
    """Perform simulated annealing to find a solution."""

    initial_temp: float = 5000.0
    alpha = 0.99
    current_temp = initial_temp
    solution = initial_state
    same_solution = 0
    same_cost_diff = 0

    while same_solution < 1500 and same_cost_diff < 50000:  # originally 150k
        neighbor = _get_neighboring_route(solution)

        cost_diff = get_cost(
            neighbor, weights, fuzziness_type, defuzzification_type
        ) - get_cost(solution, weights, fuzziness_type, defuzzification_type)

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

        current_temp = current_temp * alpha

    return solution


def estimate_average_route_distance(
    route: Route, weights, fuzziness_type: str, sampling_size: int = 100_000
):
    return (
        sum(
            [
                get_distance(
                    route,
                    weights,
                    fuzziness_type,
                    "approximation",
                    partial_approximation=False,
                )
                for _ in range(sampling_size)
            ]
        )
        / sampling_size
    )
