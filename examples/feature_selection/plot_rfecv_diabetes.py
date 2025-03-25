"""
================================================================
RFECV with multiple metrics for feature selection on diabetes
================================================================

This example demonstrates the use of Recursive Feature Elimination with
Cross-Validation (RFECV) with multiple scoring metrics on the diabetes dataset.
Given that the cost of making a poor prediction in healthcare can be high,
it's important to evaluate feature selection using multiple metrics.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Loading the dataset
# -----------------
#
# The diabetes dataset consists of 10 features collected from 442 diabetes patients,
# with a quantitative measure of disease progression one year later as the target.

from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")

# %%
# Setting up RFECV with multiple metrics
# ------------------------------------
#
# For regression problems like this, we'll use several metrics:
# - R² score: measures the proportion of variance explained by the model
# - Negative Mean Squared Error: measures absolute prediction error (squared)
# - Negative Mean Absolute Error: more robust to outliers
# - D² Tweedie score: specialized for healthcare prediction tasks with skewed data
#
# We'll use the R² score for selecting the optimal number of features.

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import d2_tweedie_score, make_scorer
from sklearn.model_selection import KFold

# Create cross-validation folds
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Define the estimator
estimator = GradientBoostingRegressor(n_estimators=100, random_state=0)

# Define multiple scoring metrics
# D² Tweedie score with power=2 for healthcare predictions
d2_tweedie = make_scorer(d2_tweedie_score, power=2)

scoring = {
    "r2": "r2",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "d2_tweedie": d2_tweedie,
}

# Set up RFECV with multiple metrics
rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=cv,
    scoring=scoring,
    refit="r2",  # Select features based on R²
    min_features_to_select=3,
    n_jobs=2,
    verbose=1,
)

# Fit the RFECV model
rfecv.fit(X, y)

print(f"Optimal number of features (using R²): {rfecv.n_features_}")
print(f"Selected features: {[feature_names[i] for i in np.where(rfecv.support_)[0]]}")

# %%
# Visualizing the results
# --------------------
#
# Let's visualize how the performance changes with the number of selected features.

import matplotlib.pyplot as plt
import pandas as pd

cv_results = pd.DataFrame(rfecv.cv_results_)

# Create a plot with 2 rows and 2 columns of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics = list(scoring.keys())

# Plot each metric
for i, (metric, ax) in enumerate(zip(metrics, axes.flat)):
    mean_key = f"mean_test_{metric}"
    std_key = f"std_test_{metric}"

    ax.errorbar(
        x=cv_results["n_features"],
        y=cv_results[mean_key],
        yerr=cv_results[std_key],
        marker="o",
        linestyle="-",
    )
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel(f"Cross-validation {metric}")
    ax.set_title(f"{metric} vs number of features")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add a vertical line at the optimal number of features for this metric
    optimal_n = cv_results.loc[cv_results[mean_key].idxmax(), "n_features"]
    ax.axvline(
        x=optimal_n, color="r", linestyle="--", label=f"Optimal: {optimal_n} features"
    )
    ax.legend()

plt.tight_layout()
plt.suptitle(
    "Feature selection with multiple metrics on diabetes dataset", fontsize=16, y=1.02
)
plt.show()

# %%
# Comparing optimal features across metrics
# --------------------------------------
#
# Different metrics may suggest different optimal feature subsets. Let's see if
# there's a consensus or divergence.

print("Optimal number of features per metric:")
for metric in metrics:
    mean_key = f"mean_test_{metric}"
    optimal_n = cv_results.loc[cv_results[mean_key].idxmax(), "n_features"]
    print(f"  - {metric}: {optimal_n}")

# %%
# Feature importance and rankings
# ----------------------------
#
# Let's look at which features were deemed most important by RFECV.

# Display feature rankings (lower is better, 1 means selected)
ranking_df = pd.DataFrame(
    {"Feature": feature_names, "Ranking": rfecv.ranking_, "Selected": rfecv.support_}
)
ranking_df = ranking_df.sort_values("Ranking")
print("Feature rankings (1 = selected):")
print(ranking_df)

# %%
# Train a model with the selected features
# -------------------------------------
#
# Let's evaluate the performance of a model trained on just the selected features.

from sklearn.model_selection import cross_validate

X_selected = X[:, rfecv.support_]

# Perform cross-validation with multiple metrics
cv_results = cross_validate(
    estimator, X_selected, y, cv=cv, scoring=scoring, return_train_score=True
)

# Print the results
print("\nModel performance with selected features:")
for metric in metrics:
    test_key = f"test_{metric}"
    train_key = f"train_{metric}"
    print(f"  - {metric}:")
    print(
        f"      Test:  {cv_results[test_key].mean():.4f} ± "
        f"{cv_results[test_key].std():.4f}"
    )
    print(
        f"      Train: {cv_results[train_key].mean():.4f} ± "
        f"{cv_results[train_key].std():.4f}"
    )

# %%
# The results demonstrate how RFECV with multiple metrics allows us to make more
# informed decisions about feature selection for a real-world medical dataset.
# This is particularly important in healthcare applications where multiple performance
# criteria may be relevant for making clinical decisions.
