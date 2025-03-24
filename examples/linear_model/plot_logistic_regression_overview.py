"""
==============================================
Logistic Regression - Overview
==============================================

This example illustrates various aspects of Logistic Regression in scikit-learn:

1. **Basic Logistic Function**: Visualization of the logistic function and comparison
   with linear regression on a synthetic dataset.
2. **Regularization Path**: Demonstration of L1-regularized logistic regression and
   how coefficients change with regularization strength.

Logistic regression is a linear model for classification that models the probabilities
of the outcomes using the logistic function. It's widely used for binary classification
problems, though it can be extended to multi-class classification as well.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

# %%
# Part 1: Logistic Function
# -------------------------
#
# This section shows how logistic regression classifies values as either 0 or 1
# using the logistic curve, and compares it with linear regression.

plt.figure(figsize=(10, 5))

# Generate a toy dataset, it's just a straight line with some Gaussian noise:
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(float)
X[X > 0] *= 4
X += 0.3 * np.random.normal(size=n_samples)

X = X[:, np.newaxis]

# Fit the classifier
clf = LogisticRegression(C=1e5)
clf.fit(X, y)

# Plot the result
plt.scatter(X.ravel(), y, label="example data", color="black", zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)

ols = LinearRegression()
ols.fit(X, y)
plt.plot(
    X_test,
    ols.coef_ * X_test + ols.intercept_,
    label="Linear Regression Model",
    linewidth=1,
)
plt.axhline(0.5, color=".5")

plt.ylabel("y")
plt.xlabel("X")
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-0.25, 1.25)
plt.xlim(-4, 10)
plt.legend(
    loc="lower right",
    fontsize="small",
)
plt.title("Logistic Function vs Linear Regression")

# %%
# Part 2: Regularization Path of L1-Logistic Regression
# ----------------------------------------------------
#
# This section demonstrates how the coefficients of L1-regularized logistic regression
# change as the regularization parameter changes. The models are ordered from strongest
# regularized to least regularized.

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Remove the third class to make the problem a binary classification
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
coefs_ = []
for c in cs:
    clf.set_params(logisticregression__C=c)
    clf.fit(X, y)
    coefs_.append(clf["logisticregression"].coef_.ravel().copy())

coefs_ = np.array(coefs_)

# Plot regularization path
plt.figure(figsize=(10, 5))

# Colorblind-friendly palette (IBM Color Blind Safe palette)
colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]

for i in range(coefs_.shape[1]):
    plt.semilogx(cs, coefs_[:, i], marker="o", color=colors[i], label=feature_names[i])

plt.xlabel("C (inverse of regularization strength)")
plt.ylabel("Coefficients")
plt.title("Regularization Path of L1-Logistic Regression")
plt.legend()
plt.axis("tight")

plt.tight_layout()
plt.show()
