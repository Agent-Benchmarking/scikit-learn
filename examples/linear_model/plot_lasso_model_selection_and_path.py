"""
==================================================
Lasso Model Selection: Cross-Validation and BIC/AIC
==================================================

This example demonstrates two approaches for model selection in Lasso regression:

1. Cross-validation: Using :class:`~sklearn.linear_model.LassoCV` to find
   the best regularization parameter (alpha) via cross-validation.

2. Information criteria: Using :class:`~sklearn.linear_model.LassoLarsIC`
   to select the model based on Akaike's Information Criterion (AIC) or
   Bayesian Information Criterion (BIC).

We compare these two approaches to understand their differences and when
to use each method.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print(__doc__)

# %%
# Part 1: Data Preparation
# ------------------------
#
# We'll use the diabetes dataset for this example.
X, y = load_diabetes(return_X_y=True, as_frame=True)
n_samples = X.shape[0]
n_features = X.shape[1]

X = X.values
y = y.values

# %%
# Part 2: Cross-Validation Approach
# --------------------------------
#
# First, we use :class:`~sklearn.linear_model.LassoCV` with cross-validation to find
# the optimal value of alpha.

# Define the range of alphas to test
n_alphas = 100
alphas = np.logspace(-6, -1, n_alphas)

# Train the model with cross-validation
cv_model = make_pipeline(
    StandardScaler(), LassoCV(alphas=alphas, cv=10, random_state=0, max_iter=10000)
)

t1 = time.time()
cv_model.fit(X, y)
t_lasso_cv = time.time() - t1

# Get the best alpha value
best_alpha = cv_model[-1].alpha_

# Train a normal Lasso with the best alpha from CV
best_model = make_pipeline(StandardScaler(), Lasso(alpha=best_alpha, max_iter=10000))
best_model.fit(X, y)

# %%
# Plot the cross-validation path to see how the model score changes with alpha
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

# Compute the log of alphas for better visualization
log_alphas = -np.log10(alphas)
mse_path = np.mean(cv_model[-1].mse_path_, axis=1)
std_error = np.std(cv_model[-1].mse_path_, axis=1) / np.sqrt(10)

plt.plot(log_alphas, mse_path)
plt.fill_between(log_alphas, mse_path + std_error, mse_path - std_error, alpha=0.2)
plt.axvline(
    -np.log10(best_alpha), linestyle="--", color="k", label=f"alpha: {best_alpha:.5f}"
)
plt.xlabel("-log(alpha)")
plt.ylabel("Mean squared error")
plt.title("Mean squared error on each fold")
plt.legend()

# Plot number of features selected with cross-validation
plt.subplot(1, 2, 2)
plt.semilogx(
    alphas,
    cv_model[-1].coef_.shape[0] - np.sum(cv_model[-1].coef_ == 0, axis=1),
    marker="o",
    label="Features",
)
plt.axvline(best_alpha, linestyle="--", color="k", label=f"alpha: {best_alpha:.5f}")
plt.xlabel("alpha")
plt.ylabel("Number of active features")
plt.title("Feature selection as a function of alpha")
plt.legend()
plt.tight_layout()

# %%
# Part 3: Information Criteria Approach
# -----------------------------------
#
# Now we'll use :class:`~sklearn.linear_model.LassoLarsIC` to select the best model
# based on AIC or BIC.

# Initialize models using AIC and BIC
aic_model = make_pipeline(
    StandardScaler(), LassoLarsIC(criterion="aic", normalize=False)
)
bic_model = make_pipeline(
    StandardScaler(), LassoLarsIC(criterion="bic", normalize=False)
)

# Fit the models
t1 = time.time()
aic_model.fit(X, y)
bic_model.fit(X, y)
t_ic = time.time() - t1

# Get the optimal alphas selected by each criterion
alpha_aic = aic_model[-1].alpha_
alpha_bic = bic_model[-1].alpha_


# %%
# Rescale the AIC and BIC to match the definition in Zou et al. (2007)
def zou_et_al_criterion_rescaling(criterion, n_samples, noise_variance):
    """Rescale the information criterion to follow the definition of Zou et al."""
    return criterion - n_samples * np.log(2 * np.pi * noise_variance) - n_samples


aic_criterion = zou_et_al_criterion_rescaling(
    aic_model[-1].criterion_,
    n_samples,
    aic_model[-1].noise_variance_,
)

bic_criterion = zou_et_al_criterion_rescaling(
    bic_model[-1].criterion_,
    n_samples,
    bic_model[-1].noise_variance_,
)

# Find the index of the selected alphas
index_alpha_aic = np.flatnonzero(aic_model[-1].alphas_ == alpha_aic)[0]
index_alpha_bic = np.flatnonzero(bic_model[-1].alphas_ == alpha_bic)[0]

# %%
# Plot the AIC and BIC paths
plt.figure(figsize=(10, 6))
plt.plot(aic_criterion, color="tab:blue", marker="o", label="AIC criterion")
plt.plot(bic_criterion, color="tab:orange", marker="o", label="BIC criterion")
plt.vlines(
    index_alpha_aic,
    aic_criterion.min(),
    aic_criterion.max(),
    color="tab:blue",
    linestyle="--",
    label=f"Selected alpha (AIC): {alpha_aic:.5f}",
)
plt.vlines(
    index_alpha_bic,
    aic_criterion.min(),
    aic_criterion.max(),
    color="tab:orange",
    linestyle="--",
    label=f"Selected alpha (BIC): {alpha_bic:.5f}",
)
plt.legend()
plt.ylabel("Information criterion")
plt.xlabel("Lasso model sequence")
plt.title("Lasso model selection via AIC and BIC")

# %%
# Part 4: Comparing the approaches
# ------------------------------
#
# Finally, let's compare the selected models:
# - How many features each method selects
# - How well they perform on the dataset
# - Time required for model selection

# Get active features for each model
active_cv = np.sum(best_model[-1].coef_ != 0)
active_aic = np.sum(aic_model[-1].coef_ != 0)
active_bic = np.sum(bic_model[-1].coef_ != 0)

# Calculate R-squared scores
r2_cv = r2_score(y, best_model.predict(X))
r2_aic = r2_score(y, aic_model.predict(X))
r2_bic = r2_score(y, bic_model.predict(X))

print("Model comparison results:")
print("-" * 50)
print(f"Cross-validation (alpha={best_alpha:.5f}):")
print(f"  Active features: {active_cv}")
print(f"  R^2 score: {r2_cv:.4f}")
print(f"  Selection time: {t_lasso_cv:.3f} seconds")
print("\nAIC criterion (alpha={:.5f}):".format(alpha_aic))
print(f"  Active features: {active_aic}")
print(f"  R^2 score: {r2_aic:.4f}")
print(f"  Selection time: {t_ic/2:.3f} seconds")
print("\nBIC criterion (alpha={:.5f}):".format(alpha_bic))
print(f"  Active features: {active_bic}")
print(f"  R^2 score: {r2_bic:.4f}")
print(f"  Selection time: {t_ic/2:.3f} seconds")

# %%
# Compare coefficients from different approaches
plt.figure(figsize=(10, 6))
coef_names = [f"Feature {i}" for i in range(n_features)]

# Create a dataframe of coefficients for visualization
import pandas as pd

coef_data = pd.DataFrame(
    {
        "CV": best_model[-1].coef_,
        "AIC": aic_model[-1].coef_,
        "BIC": bic_model[-1].coef_,
    },
    index=coef_names,
)

# Plot the coefficients as a bar chart
coef_data.plot(kind="bar", figsize=(12, 6))
plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
plt.title("Lasso coefficients from different selection methods")
plt.ylabel("Coefficient value")
plt.tight_layout()
plt.show()

# %%
# Conclusion
# ----------
#
# From this example, we can observe several key differences between these model
# selection approaches:
#
# **Cross-validation (CV):**
# * Evaluates model performance directly by testing on held-out data
# * Tends to be more computationally expensive than information criteria
# * Generally more robust, especially with small datasets
# * Often selects more complex models (more non-zero coefficients)
#
# **Information Criteria (AIC/BIC):**
# * Faster to compute as they don't require multiple model fits
# * BIC typically selects simpler models than AIC (fewer non-zero coefficients)
# * Use theoretical penalties for model complexity rather than empirical performance
# * Require assumptions about the underlying data distribution
#
# **When to use each approach:**
#
# * Use cross-validation when:
#   - You have sufficient data for validation
#   - Prediction accuracy is your primary goal
#   - You can afford the computational cost
#
# * Use information criteria when:
#   - You have limited data
#   - Model parsimony is important
#   - Computational efficiency is a concern
#   - Your data reasonably satisfies the underlying assumptions
#
# * BIC tends to work better when the true model is sparse and in the model space
# * AIC may work better when the true model is complex or not fully captured
#   by any model in your space
#
# These techniques can also be used together to gain multiple perspectives on
# model selection.
