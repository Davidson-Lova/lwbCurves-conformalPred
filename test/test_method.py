"""
Illustration of the proposed method on artificial data
Step 0 : Generate 1000 "measurement" points.
Step 1 : With polynomial conformal ridge regression,
         predict a median curve and a lower bound curve
         with risk 10^{-3} as a baseline.
Step 2 : Take 57 "measurement" points at random from the 1000
         for training data.
Step 3 : Construct median curve and choose the degree that
         minimizes the l2 error between the training median
         curve and the baseline median curve.
Step 4 : Predict a lower bound curve
Step 5 : Compare the proposed and the baseline median curve.
         Compare the proposed and the baseline lower bound curve
"""

# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import random

sys.path.append("..")

from conformal_poly_ridge_reg_1S.data import custom_dataset, data_prepare
from conformal_poly_ridge_reg_1S.model import \
    conf_pred_poly_ridge_reg_1S as predictors
from conformal_poly_ridge_reg_1S.utils import utils

random.seed(42)
np.random.seed(42)

# %%
# Generate ample dataset
N_full = 1000

# xlim
x_min = -3
x_max = 3

# 
u_N = 8

# Noise
sigma0 = 0.1
noise_a = -0.05

# Function
intercept = 13
slope = 0.3

data_gen_full = custom_dataset.test_sample(N_full, x_min, x_max, u_N, sigma0, noise_a)

# %%
# Visualize
X_full, Y_full = data_gen_full.get_dataset(utils.func_exp, intercept, slope)
plt.figure()
plt.scatter(X_full, Y_full, marker="+")
plt.xlim(x_min, x_max)
plt.ylim(0, intercept)
plt.grid()
plt.show()

# Format full data
data_full = data_prepare.dataLoader(X_full.reshape(-1, 1), Y_full.reshape(-1, 1))


# %%
# Evaluation points - regular mesh of [x_min, x_max]
n_evals = 100
X_evals = np.linspace(x_min, x_max, n_evals).reshape(-1, 1)
data_evals = data_prepare.dataLoader(X_evals, X_evals)


# %%
# Predicting median curve using full data
maxDegree = 5
alpha = 1e-3
alpha = max(alpha, 1 / (N_full + 1))
alpha_med = 0.5
sigma = 0.9

# %%
mean_line_style = "dashed"
lwb_line_style = "solid"

l_full_med = []
l_full_lwb = []
for degree in range(maxDegree + 1):
    # Instantiate predictor
    rp = predictors.conformal_ridge_regression_1S(degree)
    rp.fit(data_full)
    med, _ = rp.predictive_region(data_full, N_full, data_evals, sigma**2, alpha_med)
    lwb, _ = rp.predictive_region(data_full, N_full, data_evals, sigma**2, alpha)
    l_full_med.append(med)
    l_full_lwb.append(lwb)
    # Show the result
    plt.figure()
    plt.plot(
        data_evals.get_input(),
        med,
        color="darkorange",
        linestyle=mean_line_style,
        label="Baseline median",
    )

    plt.plot(
        data_evals.get_input(),
        lwb,
        color="green",
        linestyle=lwb_line_style,
        label="Baseline lower bound",
    )

    plt.scatter(
        data_full.get_input(),
        data_full.get_output(),
        marker="d",
        label="Measurements",
        color="k",
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("degree : {}".format(degree))
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(0, intercept)

    plt.show()

# %%
# Set baseline median response
full_opt_deg = 5
bsln_med = l_full_med[full_opt_deg]
bsln_lwb = l_full_lwb[full_opt_deg]


# %%
N_train = 57
iDs_train = random.sample(list(range(N_full)), N_train)
X_train = X_full[iDs_train]
Y_train = Y_full[iDs_train]

# Visualize
plt.figure()
plt.scatter(X_train, Y_train, marker="+")
plt.xlim(x_min, x_max)
plt.ylim(0, intercept)
plt.grid()
plt.show()

# Format training data
data_train = data_prepare.dataLoader(X_train.reshape(-1, 1), Y_train.reshape(-1, 1))

# %%
# Predicting the median curve
maxDegree = 5
alpha = 1e-3
alpha = max(alpha, 1 / (N_full + 1))
alpha_med = 0.5
sigma = 0.9

# %%
mean_line_style = "dashed"
lb_line_style = "solid"

l_train_med = []
for degree in range(maxDegree + 1):
    # Instantiate predictor
    rp = predictors.conformal_ridge_regression_1S(degree)
    rp.fit(data_train)
    med, _ = rp.predictive_region(data_train, N_train, data_evals, sigma**2, alpha_med)
    l_train_med.append(med)
    # Show the result
    plt.figure()
    plt.plot(
        data_evals.get_input(),
        med,
        color="darkorange",
        linestyle=mean_line_style,
        label="Proposed median",
    )

    plt.scatter(
        data_train.get_input(),
        data_train.get_output(),
        marker="d",
        label="Measurements",
        color="k",
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("degree : {}".format(degree))
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(0, intercept)
    
    plt.show()
# %%
# Choosing the optimal training degree
h_evals = (X_evals[1:] - X_evals[:-1]).reshape(-1, 1)
l2_dist = [
    np.sum(np.power(bsln_med.reshape(-1, 1) - med.reshape(-1, 1), 2)[:-1] / h_evals) for med in l_train_med
]
# Visualize
plt.figure()

plt.plot(np.arange(len(l2_dist)), np.array(l2_dist))
plt.ylabel("L2 distance")
plt.xlabel("degree")
plt.grid()
plt.show()

train_opt_deg = np.argmin(np.array(l2_dist))

# %%
################################
# Minimum curve construction
u_Xs_train = np.unique(data_train.get_input())
ids = np.unique(data_train.get_input(), return_inverse=True)[-1]
Ys_train = [data_train.get_output()[el == ids] for el in range(u_Xs_train.shape[0])]
# Visualize
plt.figure()
for u, y in zip(u_Xs_train, Ys_train):
    plt.scatter(np.array([u] * y.shape[0]), y, marker="+")
plt.grid()
plt.xlim(x_min, x_max)
plt.ylim(0, intercept)
plt.show()

# %%
# Predict the median function at each value of x
dt_u_Xs_train = data_prepare.dataLoader(u_Xs_train.reshape(-1, 1), u_Xs_train.reshape(-1, 1))
alphaMed = 0.5

# Predict on unique x points
rp = predictors.conformal_ridge_regression_1S(train_opt_deg)
rp.fit(data_train)
u_med, _ = rp.predictive_region(data_train, N_train, dt_u_Xs_train, sigma**2, alphaMed)

plt.figure()
for u, y, m in zip(u_Xs_train, Ys_train, u_med):
    plt.scatter(np.array([u] * y.shape[0]), y, marker="+")
    plt.scatter(u, m, marker="x", color="red")
plt.xlim(x_min, x_max)
plt.ylim(0, intercept)
plt.grid()
plt.show()
# %%
# Compute residuals
u_resid = [y - u for y, u in zip(Ys_train, u_med)]

# Visualize
plt.figure()
for x, resid in zip(u_Xs_train, u_resid):
    plt.scatter(np.array([x] * resid.shape[0]), resid, marker="+")
plt.grid()
plt.show()

# %%
resid = np.concatenate(tuple(u_resid))

# Test of normality
fig = plt.figure()
ax = fig.add_axes(111)
ts, pv = stats.shapiro(resid)
print("stat : {} - p-value : {}".format(ts, pv))
stats.probplot(resid.flatten(), dist="norm", plot=plt)
ax.grid("True")
plt.title("")
plt.show()


# %%
# Compute lower bound curve
std_resid = resid.std()
risk = 1e-3
lower_bound_resid = std_resid * stats.t.ppf(risk, df=57)
mediane_curve, _ = rp.predictive_region(data_train, N_train, data_evals, sigma**2, alphaMed)
lower_bound_curve = mediane_curve + lower_bound_resid

# Visualize
plt.figure()

plt.plot(
    data_evals.get_input(),
    mediane_curve,
    color="orange",
    linestyle=mean_line_style,
    label="Proposed median",
)

plt.plot(
    data_evals.get_input(),
    bsln_med,
    color="red",
    linestyle=mean_line_style,
    label="Baseline median",
)

plt.plot(
    data_evals.get_input(),
    lower_bound_curve,
    color="magenta",
    linestyle=lb_line_style,
    label="Proposed lwb",
)

plt.plot(
    data_evals.get_input(),
    bsln_lwb,
    color="cyan",
    linestyle=lb_line_style,
    label="Baseline lwb",
)

plt.scatter(
    data_full.get_input(),
    data_full.get_output(),
    marker="x",
    label="Measurements",
    color="green",
    alpha = 0.4,
)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.title("degree : {}".format(train_opt_deg))
plt.xlim(x_min, x_max)
plt.ylim(0, intercept)
plt.grid()
plt.show()


# %%
