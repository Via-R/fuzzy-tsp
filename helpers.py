from random import randint
from typing import List, Optional, Tuple
import json

Route = List[int]
FuzzyNumber = List[float]
Weights = List[List[Optional[FuzzyNumber]]]


def get_weights_from_data(data):
    cities = list(data.get_nodes())
    weights = [[None] * len(cities) for _ in range(len(cities))]

    for city_from in cities:
        for city_to in cities:
            weights[city_to - 1][city_from - 1] = data.get_weight(city_from, city_to)

    return weights


def save_weights(weights, weights_filename: str) -> None:
    with open(weights_filename, "w") as f:
        json.dump(weights, f, indent=4)


def load_weights(weights_filename: str) -> Weights:
    with open(weights_filename, "r") as f:
        return json.load(f)


def generate_fuzzy_weights(
    cities: List[int],
    deviation: Tuple[int, int],
    crisp_weights,
) -> Tuple[Weights, Weights]:
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


def create_fuzzy_weights(data, problem_name, deviation: Tuple[int, int]):
    cities = list(data.get_nodes())
    min_city = min(cities)
    cities = [city - min_city + 1 for city in cities]

    crisp_weights = get_weights_from_data(data)
    save_weights(crisp_weights, f"weights/weights-{problem_name}.json")
    fuzzy_weights, fuzzy_crisp_weights = generate_fuzzy_weights(
        cities, deviation, crisp_weights
    )
    save_weights(fuzzy_weights, f"weights/fuzzy-weights-{problem_name}.json")

    return cities, fuzzy_weights
