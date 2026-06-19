from typing import List
from collections import Counter
from Vector import sum_of_squares, dot
import math

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs) // 2]

def median_even(xs: List[float]) -> float:
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs)
    return (sorted_xs[hi_midpoint-1] + sorted_xs[hi_midpoint]) / 2

def median(xs: List[float]) -> float:
    return median_even(xs) if len(xs) % 2 == 0 else median_odd(xs)

def quantile(xs: List[float], p: float) -> float:
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

def mode(x: List[float]) -> List[float]:
    counter = Counter(x)
    max_count = max(counter.values())
    return [x_i for x_i, count in counter.items() if count == max_count]

def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

def de_mean(xs: List[float]) -> float:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    assert len(xs) >= 2, "La variance nécessite au moins deux élément"
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n-1)

def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))

def interquartile_range(xs: List[float]) -> float:
    return quantile(xs, 0.75) - quantile(xs, 0.25) 

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs et ys doivent être de la meme taille"
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

def corelation(xs: List[float], ys: List[float]) -> float:
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) /stdev_x / stdev_y
    else:
        return 0