from typing import Callable, List
from Vector import distances, add, scalar_multiply
import random

Vector = List[float]

def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h

def partial_difference_quotient(f: Callable[[Vector], float], v: Vector, i: int, h: float) -> float:
    """Compute de i partial difference quotient of f in v"""
    w = v.copy()
    w[i] += h
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float], v: Vector, h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def function(v: Vector) -> float:
    return (v[0] - 3)** 2 + (v[1] + 2) ** 2 + 5
 
v = [random.uniform(-10, 10) for i in range(2)]
for epoch in range(1000):
    grad = estimate_gradient(function, v)
    v = gradient_step(v, grad, -0.01)

print(v)
print(function(v))