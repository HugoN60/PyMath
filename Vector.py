from typing import List
import math

Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    """Add two vectors"""
    assert len(v) == len(w), "The vectors must have the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def substract(v: Vector, w: Vector) -> Vector:
    """Substract two vectors"""
    assert len(v) == len(w), "The vectors must have the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Add all the vector in the list"""
    assert vectors, "No vector"
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "All vectors must have the same length"
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multply each elements by c"""
    return [v_i * c for v_i in v]

def vector_mean(v: List[Vector]) -> Vector:
    """Compute the average element by element"""
    return scalar_multiply(1/len(v), vector_sum(v))

def dot(v: Vector, w: Vector) -> float:
    """Compute v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "The vectors must have the same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
    """Compute v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def magnitude(v: Vector) -> float:
    """Compute v magnitude"""
    return math.sqrt(sum_of_squares(v))

def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(substract(v, w))

def distances(v: Vector, w: Vector) -> float:
    return magnitude(substract(v, w))

