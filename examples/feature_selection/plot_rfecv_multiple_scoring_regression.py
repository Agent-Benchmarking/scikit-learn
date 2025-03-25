"""
=======================================================
Recursive feature elimination for regression with RFECV
=======================================================

This example illustrates the use of Recursive Feature Elimination with
Cross-Validation (RFECV) with multiple scoring metrics for a regression task.
This allows us to monitor different regression metrics during feature selection
and select the optimal number of features based on the most relevant metric.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Data generation
# ---------------
#
# We generate a regression dataset with 100 samples, 10 features where only
# 5 features are actually useful for predicting the target.

import numpy as np

from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=100, n_features=10, n_informative=5, noise=0.1, random_state=42
)

# %%
# Feature selection with multiple metrics
# --------------------------------------
#
# We set up the RFECV with three different regression metrics: R² score,
# negative mean squared error, and negative mean absolute error. By setting
# `refit='r2'`, we indicate that the optimal number of features should be
# selected based on the R² score.

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

# Create cross-validation folds and estimator
cv = KFold(n_splits=5, shuffle=True, random_state=42)
estimator = RandomForestRegressor(random_state=42)

# Define multiple scoring metrics for regression
scoring = {
    "r2": "r2",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
}

# Set up RFECV with multiple metrics
rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=cv,
    scoring=scoring,
    refit="r2",  # Select features based on R²
    min_features_to_select=1,
    n_jobs=2,
)

# Fit the model
rfecv.fit(X, y)

print(f"Optimal number of features (using R²): {rfecv.n_features_}")
print(f"Selected features: {np.where(rfecv.support_)[0]}")

# %%
# Visualizing the results for different metrics
# --------------------------------------------
#
# We can visualize how the performance of each metric changes as we decrease
# the number of features. This helps us understand if all metrics agree on the
# optimal number of features.

import matplotlib.pyplot as plt
import pandas as pd

cv_results = pd.DataFrame(rfecv.cv_results_)

plt.figure(figsize=(10, 6))

# Plot R² scores (higher is better)
plt.subplot(1, 2, 1)
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_r2"],
    yerr=cv_results["std_test_r2"],
    color="blue",
    label="R²",
)
plt.xlabel("Number of features selected")
plt.ylabel("R² score")
plt.title("R² score vs number of features")
plt.grid(True, linestyle="--", alpha=0.7)

# Plot error metrics (lower is better, but negated in scikit-learn)
plt.subplot(1, 2, 2)
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_neg_mean_squared_error"],
    yerr=cv_results["std_test_neg_mean_squared_error"],
    color="red",
    label="Neg. MSE",
)
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_neg_mean_absolute_error"],
    yerr=cv_results["std_test_neg_mean_absolute_error"],
    color="green",
    label="Neg. MAE",
)
plt.xlabel("Number of features selected")
plt.ylabel("Negative error (higher is better)")
plt.title("Error metrics vs number of features")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# Comparing optimal features across metrics
# ---------------------------------------
#
# Different metrics might suggest different optimal numbers of features.
# Let's find and compare the optimal number for each metric.


def find_optimal_features(cv_results, metric):
    mean_key = f"mean_test_{metric}"
    best_idx = np.argmax(cv_results[mean_key])
    return cv_results["n_features"][best_idx]


print("Optimal number of features per metric:")
for metric in scoring.keys():
    optimal_n = find_optimal_features(cv_results, metric)
    print(f"  - {metric}: {optimal_n}")

# %%
# Performance evaluation with selected features
# -------------------------------------------
#
# Now let's evaluate the model's performance on the selected features
# for each metric.

from sklearn.model_selection import cross_val_score

X_selected = X[:, rfecv.support_]

print("\nCross-validation scores with selected features:")
for metric in scoring.keys():
    scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring=metric)
    print(f"  - {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

# %%
# Feature importance of selected features
# -------------------------------------
#
# Finally, let's look at the importance of the selected features
# as determined by the random forest model.

estimator.fit(X_selected, y)

plt.figure(figsize=(10, 6))
plt.bar(
    range(X_selected.shape[1]),
    estimator.feature_importances_,
    tick_label=[f"Feature {i}" for i in np.where(rfecv.support_)[0]],
)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature importances of selected features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# The results show how using multiple metrics in RFECV helps us make better
# decisions about feature selection, particularly for regression problems where
# different error metrics might provide complementary insights.
