"""
==============================================
Regularization Paths for Linear Models
==============================================

This example illustrates the regularization paths for different linear models:

1. **Ridge Regularization Path**: Shows how Ridge coefficients change with regularization
   strength and demonstrates the effect of regularization on ill-conditioned matrices.

2. **Lasso, Lasso-LARS, and Elastic Net Paths**: Compares the regularization paths of
   these models, showing how coefficients change with regularization strength.

Regularization is a technique used to prevent overfitting by adding a penalty term
to the loss function. The regularization parameter controls the trade-off between
fitting the data well and keeping the model simple.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import enet_path, lars_path, lasso_path, Ridge

# %%
# Part 1: Ridge Regularization Path
# ---------------------------------
#
# This section shows the effect of collinearity in the coefficients of a Ridge estimator.
# Each color represents a different feature of the coefficient vector, and this is
# displayed as a function of the regularization parameter.
#
# When alpha is very large, the regularization effect dominates the squared loss function
# and the coefficients tend to zero. At the end of the path, as alpha tends toward zero
# and the solution tends towards the ordinary least squares, coefficients exhibit big
# oscillations.

plt.figure(figsize=(10, 5))

# X is the 10x10 Hilbert matrix
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# Compute paths
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# Display results
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha (regularization strength)")
plt.ylabel("coefficients")
plt.title("Ridge Coefficients as a Function of Regularization")
plt.axis("tight")

# %%
# Part 2: Lasso, Lasso-LARS, and Elastic Net Paths
# ------------------------------------------------
#
# This section compares the regularization paths of Lasso, Lasso-LARS, and Elastic Net.
# It shows how the coefficients change as the regularization strength changes.

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True)
X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

# Compute paths
eps = 5e-3  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X, y, eps=eps, positive=True
)

print("Computing regularization path using the LARS...")
alphas_lars, _, coefs_lars = lars_path(X, y, method="lasso")

print("Computing regularization path using the positive LARS...")
alphas_positive_lars, _, coefs_positive_lars = lars_path(
    X, y, method="lasso", positive=True
)

print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=0.8)

print("Computing regularization path using the positive elastic net...")
alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, positive=True
)

# %%
# Lasso vs LARS
# -------------

plt.figure(figsize=(10, 5))
colors = cycle(["b", "r", "g", "c", "k"])
for coef_lasso, coef_lars, c in zip(coefs_lasso, coefs_lars, colors):
    l1 = plt.semilogx(alphas_lasso, coef_lasso, c=c)
    l2 = plt.semilogx(alphas_lars, coef_lars, linestyle="--", c=c)

plt.xlabel("alpha")
plt.ylabel("coefficients")
plt.title("Lasso and LARS Paths")
plt.legend((l1[-1], l2[-1]), ("Lasso", "LARS"), loc="lower right")
plt.axis("tight")

# %%
# Lasso vs Elastic-Net
# --------------------

plt.figure(figsize=(10, 5))
colors = cycle(["b", "r", "g", "c", "k"])
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.semilogx(alphas_lasso, coef_l, c=c)
    l2 = plt.semilogx(alphas_enet, coef_e, linestyle="--", c=c)

plt.xlabel("alpha")
plt.ylabel("coefficients")
plt.title("Lasso and Elastic-Net Paths")
plt.legend((l1[-1], l2[-1]), ("Lasso", "Elastic-Net"), loc="lower right")
plt.axis("tight")

# %%
# Lasso vs Positive Lasso
# -----------------------

plt.figure(figsize=(10, 5))
colors = cycle(["b", "r", "g", "c", "k"])
for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.semilogx(alphas_lasso, coef_l, c=c)
    l2 = plt.semilogx(alphas_positive_lasso, coef_pl, linestyle="--", c=c)

plt.xlabel("alpha")
plt.ylabel("coefficients")
plt.title("Lasso and Positive Lasso")
plt.legend((l1[-1], l2[-1]), ("Lasso", "Positive Lasso"), loc="lower right")
plt.axis("tight")

# %%
# LARS vs Positive LARS
# ---------------------

plt.figure(figsize=(10, 5))
colors = cycle(["b", "r", "g", "c", "k"])
for coef_lars, coef_positive_lars, c in zip(coefs_lars, coefs_positive_lars, colors):
    l1 = plt.semilogx(alphas_lars, coef_lars, c=c)
    l2 = plt.semilogx(alphas_positive_lars, coef_positive_lars, linestyle="--", c=c)

plt.xlabel("alpha")
plt.ylabel("coefficients")
plt.title("LARS and Positive LARS")
plt.legend((l1[-1], l2[-1]), ("LARS", "Positive LARS"), loc="lower right")
plt.axis("tight")

# %%
# Elastic-Net vs Positive Elastic-Net
# -----------------------------------

plt.figure(figsize=(10, 5))
colors = cycle(["b", "r", "g", "c", "k"])
for coef_e, coef_pe, c in zip(coefs_enet, coefs_positive_enet, colors):
    l1 = plt.semilogx(alphas_enet, coef_e, c=c)
    l2 = plt.semilogx(alphas_positive_enet, coef_pe, linestyle="--", c=c)

plt.xlabel("alpha")
plt.ylabel("coefficients")
plt.title("Elastic-Net and Positive Elastic-Net")
plt.legend((l1[-1], l2[-1]), ("Elastic-Net", "Positive Elastic-Net"), loc="lower right")
plt.axis("tight")

plt.show()
