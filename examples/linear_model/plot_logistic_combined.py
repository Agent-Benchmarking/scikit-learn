"""
=========================================================
Logistic Regression: Function, Regularization, and Paths
=========================================================

This example demonstrates key aspects of logistic regression, a linear
classification method based on fitting a logistic function to data.

The example illustrates:

1. The logistic function and how it differs from linear regression
2. How L1 regularization affects the coefficients in logistic regression
3. The regularization path showing how coefficients evolve with different
   regularization strengths

Logistic Regression is commonly used for binary classification problems
and serves as a baseline for more complex classification methods.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Part 1: The Logistic Function
# -----------------------------
#
# First, we visualize how logistic regression uses the logistic function to
# classify values as either 0 or 1.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

# Generate a toy dataset for binary classification
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(float)
X[X > 0] *= 4
X += 0.3 * np.random.normal(size=n_samples)

X = X[:, np.newaxis]

# Fit the logistic regression classifier
clf = LogisticRegression(C=1e5)
clf.fit(X, y)

# Plot the data and models
plt.figure(figsize=(10, 6))
plt.scatter(X.ravel(), y, label="Training data", color="black", zorder=20)
X_test = np.linspace(-5, 10, 300)

# Plot logistic regression model
loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)

# Compare with linear regression
ols = LinearRegression()
ols.fit(X, y)
plt.plot(
    X_test,
    ols.coef_ * X_test + ols.intercept_,
    label="Linear Regression Model",
    linewidth=1,
)
plt.axhline(0.5, color=".5")

plt.ylabel("Probability")
plt.xlabel("X")
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-0.25, 1.25)
plt.xlim(-4, 10)
plt.legend(loc="lower right")
plt.title("Logistic Regression vs Linear Regression")

# %%
# The plot illustrates how:
#
# * Logistic regression produces a sigmoidal curve that maps any input value to a
#   value between 0 and 1 (interpreted as a probability)
# * The decision boundary is at 0.5 probability
# * Linear regression doesn't constrain outputs to [0,1], making it less suitable for
#   classification problems
#
# Part 2: L1 Regularization Path
# ------------------------------
#
# Now we'll demonstrate how regularization affects the coefficients in logistic
# regression using a different dataset. We train models with varying regularization
# strength and observe how the coefficients change.

# Load a different dataset - the iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Make it a binary classification problem by removing the third class
X = X[y != 2]
y = y[y != 2]

# Compute regularization path
cs = l1_min_c(X, y, loss="log") * np.logspace(0, 1, 16)

# Create a pipeline with StandardScaler and LogisticRegression
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty="l1",
        solver="liblinear",
        tol=1e-6,
        max_iter=int(1e6),
        warm_start=True,
        fit_intercept=False,
    ),
)

# Collect coefficients for different regularization strengths
coefs_ = []
for c in cs:
    clf.set_params(logisticregression__C=c)
    clf.fit(X, y)
    coefs_.append(clf["logisticregression"].coef_.ravel().copy())

coefs_ = np.array(coefs_)

# %%
# Plot the regularization path
plt.figure(figsize=(10, 6))

# Colorblind-friendly palette
colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]

for i in range(coefs_.shape[1]):
    plt.semilogx(cs, coefs_[:, i], marker="o", color=colors[i], label=feature_names[i])

plt.xlabel("C (inverse regularization strength)")
plt.ylabel("Coefficient value")
plt.title("Logistic Regression Path with L1 Regularization")
plt.legend()
plt.axis("tight")

# %%
# Understanding the regularization path:
#
# * The x-axis shows C, the inverse of regularization strength (larger C = less
#   regularization)
# * With strong regularization (small C), all coefficients are close to zero
# * As regularization decreases (C increases), coefficients can grow in magnitude
# * With L1 regularization, some coefficients remain at zero, effectively performing
#   feature selection
# * Different features become non-zero at different levels of regularization
#
# Conclusion
# ----------
#
# These visualizations demonstrate important properties of logistic regression:
#
# 1. The logistic function provides a smooth S-shaped curve that maps any real value
#    to a probability between 0 and 1.
#
# 2. The L1 regularization path shows how regularization can help with feature selection
#    by keeping some coefficients at zero.
#
# 3. The regularization parameter (C in scikit-learn) controls the trade-off between
#    fitting the training data well and creating a simpler model.
#
# Logistic regression with appropriate regularization is a powerful and interpretable
# classification method, particularly useful for problems where understanding feature
# importance is as important as prediction accuracy.
