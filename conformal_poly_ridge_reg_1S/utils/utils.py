"""
Function that could be used
"""

import numpy as np


def func_exp(x, a, b):
    return -np.exp(np.exp(x * b)) + a
