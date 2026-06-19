#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:48:44 2026

@author: hugonoel
"""

import random
from typing import TypeVar, List, Tuple

X = TypeVar('X')
Y = TypeVar('Y')

def split_data(data: List[X], prob: float) -> Tuple(List[X], List[X]):
    data = data.copy()
    random.shuffle(data)
    cut = int(len(data) * prob)
    return (data[:cut], data[cut:])

def train_test_split(xs: List[X], ys: List[Y], pct_test: float) -> Tuple[List[X], List[X],
                                                                         List[Y], List[Y]]:
    idx = [i for i in range(len(xs))]
    train_idx, test_idx = split_data(idx, pct_test)
    return ([xs[i] for i in train_idx],
            [xs[i] for i in test_idx],
            [ys[i] for i in train_idx],
            [ys[i] for i in test_idx])