"""Testing the class : conformal_ridge_regression_1S
    on generated data"""

# %%
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
from conformal_poly_ridge_reg_1S.data import data_prepare
from conformal_poly_ridge_reg_1S.model import \
    conf_pred_poly_ridge_reg_1S as predictor

random.seed(42)
np.random.seed(42)


# %%
# Generate data
def bellFunc(x, mu, var, amp):
    return amp * np.exp(-np.divide(np.power(x - mu, 2), var))


def expFunc(x, freq, amp):
    return amp * np.exp(-freq * x)


def scale(x, x0, s0):
    return s0 * (x - x0)


mus = [1.8, 0.7]
vars = [0.28, 0.13]
freq = 2.8
amps = [3, 1.6, 2]
x0 = -0.8
s0 = 1.8

N = 100
nCov = 1
inputShape = (N, nCov)
inputs = np.random.uniform(-1, 1, inputShape)

scaledInputs = scale(inputs, x0, s0)
outputs = (
    bellFunc(scaledInputs, mus[0], vars[0], amps[0])
    + bellFunc(scaledInputs, mus[1], vars[1], amps[1])
    + expFunc(scaledInputs, freq, amps[2])
    + np.random.normal(0, 0.2, (N, 1))
)

iDs = list(range(N))
random.shuffle(iDs)
inputs = inputs[iDs, :]
outputs = outputs[iDs, :]
dtldr_train = data_prepare.dataLoader(inputs, outputs)

xs = np.arange(-1, 1, 0.05).reshape(-1, 1)
dtldr_test = data_prepare.dataLoader(xs, xs)

# %%
degree = 5
sigma = 1
alpha = 0.01
region_predictor = predictor.conformal_ridge_regression_1S(degree)
region_predictor.fit(dtldr_train)
prediction = region_predictor.forward(dtldr_test)
lws, ubs = region_predictor.predictive_region(
    dtldr_train, N, dtldr_test, sigma**2, alpha
)

# %%
plt.figure()

plt.scatter(
    dtldr_train.get_input(),
    dtldr_train.get_output(),
    marker="+",
    color="k",
    label="ground truth",
)

plt.plot(
    dtldr_test.get_input().flatten(),
    prediction.flatten(),
    color="r",
    label="model response",
)

plt.plot(dtldr_test.get_input().flatten(), lws, color="k", label="lower quantile")

plt.legend()
plt.grid()
plt.show()

# %%
