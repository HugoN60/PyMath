from typing import Tuple
from Vector import Vector
from statistics import correlation, standard_deviation, mean, de_mean
import random
import tqdm
from decsenteDeGradient import gradient_step

def predict(alpha: float, beta: float, x_i: float) -> float:
    return alpha * x_i + beta

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    alpha = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    beta = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y)) / total_sum_of_squares(y)

num_epoch = 1000
guess = [random.random(), random.random()]
learning_rate = 0.00001
with tqdm.trange(num_epoch):
    for _ in t:
        alpha, beta = guess
        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                     for x_i, y_i )