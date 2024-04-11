"""Testing the class : dataLoader"""

# %%
import sys
import numpy as np
sys.path.append("..")
from conformal_poly_ridge_reg_1S.data import data_prepare

# %%
input_size = 3
output_size = 1
N = 100

# %%
inputs = np.random.random((N, input_size))
trueParam = np.random.random((input_size, output_size))
outputs = inputs @ trueParam + np.random.normal(0, 0.1, (N, output_size))

# %%
dt = data_prepare.dataLoader(inputs, outputs)