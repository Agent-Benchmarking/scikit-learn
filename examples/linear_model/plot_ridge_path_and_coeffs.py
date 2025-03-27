"""
===========================================================
Ridge Regularization: Coefficients and Regularization Path
===========================================================

This example illustrates the effect of L2 regularization (Ridge) on model
coefficients. Ridge regression adds a penalty term to the ordinary least
squares objective function, effectively shrinking the coefficients.

We demonstrate:
1. How coefficients change with increasing alpha (regularization strength)
2. The benefit of Ridge regularization in handling ill-conditioned matrices
3. How Ridge regularization helps prevent overfitting

The example combines and expands on the material from the previous examples:
`plot_ridge_path.py` and `plot_ridge_coeffs.py`.

.. currentmodule:: sklearn.linear_model

:class:`Ridge` Regression is the estimator used in this example.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Part 1: Ridge coefficients on the Hilbert matrix
# ------------------------------------------------
#
# First, we illustrate the effect of regularization on an ill-conditioned matrix.
# The Hilbert matrix is a classic example of such a matrix, where a small change
# in the target variable can cause large changes in the coefficients.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# X is the 10x10 Hilbert matrix
X_hilbert = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y_hilbert = np.ones(10)

# Compute paths for Hilbert matrix
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs_hilbert = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_hilbert, y_hilbert)
    coefs_hilbert.append(ridge.coef_)

# %%
# We now plot how the coefficients change with increasing regularization

plt.figure(figsize=(10, 6))
ax = plt.gca()

ax.plot(alphas, coefs_hilbert)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha (regularization strength)")
plt.ylabel("Coefficient values")
plt.title("Ridge coefficients as a function of regularization")
plt.axis("tight")

# %%
# The plot shows how Ridge regularization reduces the variance of the coefficients.
# As alpha increases, the coefficients become more stable but also smaller in magnitude.
# With very high alpha values, coefficients approach zero as the regularization
# dominates the loss function.
#
# Part 2: Demonstrating the effect on a synthetic dataset
# ------------------------------------------------------
#
# Now we'll create a non-noisy dataset to show clearly how regularization affects
# the coefficients on a more realistic problem.

# Create a toy dataset with known coefficients
X, y, w = make_regression(
    n_samples=100, n_features=10, n_informative=8, coef=True, random_state=1
)

print(f"The true coefficients of this regression problem are:\n{w}")

# %%
# Training Ridge models with different regularization strengths
# ------------------------------------------------------------
#
# We'll train Ridge models with different alpha values and track the coefficients
# and their error compared to the true values.

# Generate values for alpha that are evenly distributed on a logarithmic scale
alphas_synth = np.logspace(-3, 4, 200)
coefs_synth = []
errors_coefs = []

# Train the model with different regularization strengths
for a in alphas_synth:
    clf = linear_model.Ridge(alpha=a)
    clf.fit(X, y)
    coefs_synth.append(clf.coef_)
    errors_coefs.append(mean_squared_error(clf.coef_, w))

# %%
# Plotting the results
# -------------------
#
# We'll create two side-by-side plots: one showing the coefficients as they change
# with alpha, and another showing the error between the estimated and true coefficients.

alphas_df = pd.Index(alphas_synth, name="alpha")
coefs_df = pd.DataFrame(
    coefs_synth, index=alphas_df, columns=[f"Feature {i}" for i in range(10)]
)
errors = pd.Series(errors_coefs, index=alphas_df, name="Mean squared error")

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

coefs_df.plot(
    ax=axs[0],
    logx=True,
    title="Ridge coefficients as a function of the regularization strength",
)
axs[0].set_ylabel("Ridge coefficient values")

errors.plot(
    ax=axs[1],
    logx=True,
    title="Coefficient error as a function of the regularization strength",
)
axs[1].set_ylabel("Mean squared error")

plt.tight_layout()
plt.show()

# %%
# Interpreting the results
# -----------------------
#
# These plots illustrate the regularization effect:
#
# - The left plot shows how coefficients shrink toward zero as alpha increases
# - The right plot shows the mean squared error between estimated and true coefficients
#
# With low regularization (small alpha), the coefficients closely match the true values.
# As alpha increases, the coefficients are increasingly shrunk toward zero.
#
# In real-world scenarios with noisy data, some regularization is usually beneficial
# to prevent overfitting. The optimal regularization strength can be found through
# cross-validation, typically using `RidgeCV`.
#
# Conclusion
# ----------
#
# Ridge regression provides an effective way to control model complexity and prevent
# overfitting by shrinking coefficients. It is particularly useful for:
#
# - Dealing with ill-conditioned problems where OLS would be unstable
# - Handling datasets with high multicollinearity
# - Preventing overfitting in high-dimensional spaces
#
# The regularization parameter alpha controls the trade-off between fitting the
# training data well and keeping the coefficients small and stable.
