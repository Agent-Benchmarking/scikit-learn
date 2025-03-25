"""
==================================================
Recursive feature elimination with multiple metrics
==================================================

This example demonstrates Recursive Feature Elimination with Cross-Validation (RFECV)
using multiple scoring metrics simultaneously. This functionality allows us to select
features based on one metric while monitoring the performance on others.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Data generation
# ---------------
#
# We build a classification task using 3 informative features. The introduction
# of 2 additional redundant (i.e. correlated) features has the effect that the
# selected features vary depending on the cross-validation fold. The remaining
# features are non-informative as they are drawn at random.

import numpy as np

from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500,
    n_features=15,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=8,
    n_clusters_per_class=1,
    class_sep=0.8,
    random_state=0,
)

# %%
# Model training and selection with multiple metrics
# -------------------------------------------------
#
# We create the RFECV object and define multiple scoring metrics.
# We'll use 'accuracy', 'balanced_accuracy', and 'f1_macro' to evaluate
# feature selection. By setting `refit='accuracy'`, the optimal number of features
# will be determined based on the accuracy metric.

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 1  # Minimum number of features to consider
clf = LogisticRegression()
cv = StratifiedKFold(5)

# Define multiple scoring metrics
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
}

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring=scoring,
    refit="accuracy",  # Select features based on accuracy
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(X, y)

print(f"Optimal number of features (using accuracy): {rfecv.n_features_}")

# %%
# Plotting results for multiple metrics
# ------------------------------------
#
# Now we'll plot the cross-validation scores for each metric as a function
# of the number of features. This allows us to see how different metrics
# behave during feature selection.

import matplotlib.pyplot as plt
import pandas as pd

# Convert cv_results_ to a DataFrame
cv_results = pd.DataFrame(rfecv.cv_results_)

# Create a figure for plotting
plt.figure(figsize=(10, 6))
metrics = list(scoring.keys())
colors = ["blue", "green", "red"]

for i, metric in enumerate(metrics):
    mean_key = f"mean_test_{metric}"
    std_key = f"std_test_{metric}"

    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results[mean_key],
        yerr=cv_results[std_key],
        color=colors[i],
        marker="o",
        linestyle="-",
        label=metric,
    )

plt.xlabel("Number of features selected")
plt.ylabel("Mean cross-validation score")
plt.title("Recursive Feature Elimination with Multiple Metrics")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Comparing Feature Selection Results
# -----------------------------------
#
# Let's compare the optimal number of features for each metric, which might
# differ based on the metric's specific characteristics.


def find_optimal_features(cv_results, metric):
    mean_key = f"mean_test_{metric}"
    best_idx = np.argmax(cv_results[mean_key])
    return cv_results["n_features"][best_idx]


print("Optimal number of features per metric:")
for metric in metrics:
    optimal_n = find_optimal_features(cv_results, metric)
    print(f"  - {metric}: {optimal_n}")

# %%
# We can also check which specific features were selected by the model
# by examining the `support_` attribute:

feature_indices = np.arange(X.shape[1])
selected_features = feature_indices[rfecv.support_]
print(f"Selected features (indices): {selected_features}")

# %%
# Evaluating the model with the selected features
# ----------------------------------------------
#
# Finally, we can evaluate the model's performance using the selected features
# on all the metrics.

from sklearn.model_selection import cross_val_score

X_selected = X[:, rfecv.support_]

print("\nCross-validation scores with selected features:")
for metric in metrics:
    scores = cross_val_score(clf, X_selected, y, cv=cv, scoring=metric)
    print(f"  - {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

# %%
# The results demonstrate how RFECV with multiple metrics allows us to select features
# optimizing for one metric while monitoring performance on others. This is particularly
# useful when accuracy alone is insufficient.
