"""
Testing the class : custom_dataset
"""

# %%
import sys

import matplotlib.pyplot as plt

sys.path.append("..")

from conformal_poly_ridge_reg_1S.data import custom_dataset
from conformal_poly_ridge_reg_1S.utils import utils

# %%
N = 57
x_min = -3
x_max = 3
u_N = 8
# Noise
sigma0 = 0.1
noise_a = -0.06
# Function
intercept = 13
slope = 0.3

data_gen = custom_dataset.test_sample(N, x_min, x_max, u_N, sigma0, noise_a)

# Visualize
Xs, Ys = data_gen.get_dataset(utils.func_exp, intercept, slope)
plt.figure()
plt.scatter(Xs, Ys, marker="+")
plt.grid()
plt.show()

# %%
