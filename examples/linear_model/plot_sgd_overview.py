"""
==============================================
Stochastic Gradient Descent (SGD) - Overview
==============================================

This example illustrates various aspects of Stochastic Gradient Descent (SGD)
in scikit-learn:

1. **Loss Functions**: Visualization of different loss functions used in SGD
2. **Penalties**: Visualization of L1, L2, and Elastic-Net penalties
3. **Binary Classification**: Maximum margin separating hyperplane
4. **Multi-class Classification**: Decision boundaries on the Iris dataset
5. **Sample Weighting**: Effect of weighted samples on the decision boundary

SGD is a simple yet efficient approach to fit linear models. It is particularly
useful when the number of samples is very large or for online learning.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from sklearn import datasets, linear_model
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import SGDClassifier

# %%
# Part 1: SGD Loss Functions
# --------------------------
#
# This section visualizes the various convex loss functions supported by
# :class:`~sklearn.linear_model.SGDClassifier`.


def modified_huber_loss(y_true, y_pred):
    z = y_pred * y_true
    loss = -4 * z
    loss[z >= -1] = (1 - z[z >= -1]) ** 2
    loss[z >= 1.0] = 0
    return loss


plt.figure(figsize=(10, 5))
xmin, xmax = -4, 4
xx = np.linspace(xmin, xmax, 100)
lw = 2
plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], color="gold", lw=lw, label="Zero-one loss")
plt.plot(xx, np.where(xx < 1, 1 - xx, 0), color="teal", lw=lw, label="Hinge loss")
plt.plot(xx, -np.minimum(xx, 0), color="yellowgreen", lw=lw, label="Perceptron loss")
plt.plot(xx, np.log2(1 + np.exp(-xx)), color="cornflowerblue", lw=lw, label="Log loss")
plt.plot(
    xx,
    np.where(xx < 1, 1 - xx, 0) ** 2,
    color="orange",
    lw=lw,
    label="Squared hinge loss",
)
plt.plot(
    xx,
    modified_huber_loss(xx, 1),
    color="darkorchid",
    lw=lw,
    linestyle="--",
    label="Modified Huber loss",
)
plt.ylim((0, 8))
plt.legend(loc="upper right")
plt.xlabel(r"Decision function $f(x)$")
plt.ylabel("$L(y=1, f(x))$")
plt.title("SGD Loss Functions")

# %%
# Part 2: SGD Penalties
# ---------------------
#
# This section shows contours of where the penalty is equal to 1
# for the three penalties L1, L2 and elastic-net.

plt.figure(figsize=(10, 5))
l1_color = "navy"
l2_color = "c"
elastic_net_color = "darkorange"

line = np.linspace(-1.5, 1.5, 1001)
xx, yy = np.meshgrid(line, line)

l2 = xx**2 + yy**2
l1 = np.abs(xx) + np.abs(yy)
rho = 0.5
elastic_net = rho * l1 + (1 - rho) * l2

ax = plt.gca()

elastic_net_contour = plt.contour(
    xx, yy, elastic_net, levels=[1], colors=elastic_net_color
)
l2_contour = plt.contour(xx, yy, l2, levels=[1], colors=l2_color)
l1_contour = plt.contour(xx, yy, l1, levels=[1], colors=l1_color)
ax.set_aspect("equal")
ax.spines["left"].set_position("center")
ax.spines["right"].set_color("none")
ax.spines["bottom"].set_position("center")
ax.spines["top"].set_color("none")

plt.clabel(
    elastic_net_contour,
    inline=1,
    fontsize=12,
    fmt={1.0: "elastic-net"},
    manual=[(-1, -1)],
)
plt.clabel(l2_contour, inline=1, fontsize=12, fmt={1.0: "L2"}, manual=[(-1, -1)])
plt.clabel(l1_contour, inline=1, fontsize=12, fmt={1.0: "L1"}, manual=[(-1, -1)])
plt.title("SGD Penalties")

# %%
# Part 3: Maximum Margin Separating Hyperplane
# --------------------------------------------
#
# This section demonstrates a linear Support Vector Machines classifier
# trained using SGD to find the maximum margin separating hyperplane
# within a two-class separable dataset.

plt.figure(figsize=(10, 5))

# Create 50 separable points
X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# Fit the model
clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)
clf.fit(X, Y)

# Plot the line, the points, and the nearest vectors to the plane
xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)

X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = clf.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ["dashed", "solid", "dashed"]
colors = "k"
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolor="black", s=20)
plt.axis("tight")
plt.title("SGD: Maximum Margin Separating Hyperplane")

# %%
# Part 4: Multi-class SGD on the Iris Dataset
# -------------------------------------------
#
# This section demonstrates multi-class classification using SGD on the iris dataset.
# The hyperplanes corresponding to the three one-versus-all (OVA) classifiers
# are represented by the dashed lines.

plt.figure(figsize=(10, 5))

# Import iris data
iris = datasets.load_iris()

# We only take the first two features
X = iris.data[:, :2]
y = iris.target
colors = "bry"

# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# Standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    xlabel=iris.feature_names[0],
    ylabel=iris.feature_names[1],
)
plt.axis("tight")

# Plot also the training points
for i, color in zip(clf.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(
        X[idx, 0],
        X[idx, 1],
        c=color,
        label=iris.target_names[i],
        edgecolor="black",
        s=20,
    )
plt.title("Decision Surface of Multi-class SGD")

# Plot the three one-against-all classifiers
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = clf.coef_
intercept = clf.intercept_


def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)


for i, color in zip(clf.classes_, colors):
    plot_hyperplane(i, color)
plt.legend()

# %%
# Part 5: SGD with Weighted Samples
# ---------------------------------
#
# This section demonstrates the effect of sample weighting on the decision boundary.
# The size of points is proportional to its weight.

plt.figure(figsize=(10, 5))

# Create 20 points
np.random.seed(0)
X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
y = [1] * 10 + [-1] * 10
sample_weight = 100 * np.abs(np.random.randn(20))
# Assign a bigger weight to the last 10 samples
sample_weight[:10] *= 10

# Plot the weighted data points
xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
ax = plt.gca()
ax.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    s=sample_weight,
    alpha=0.9,
    cmap=plt.cm.bone,
    edgecolor="black",
)

# Fit the unweighted model
clf = linear_model.SGDClassifier(alpha=0.01, max_iter=100)
clf.fit(X, y)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
no_weights = ax.contour(xx, yy, Z, levels=[0], linestyles=["solid"])

# Fit the weighted model
clf = linear_model.SGDClassifier(alpha=0.01, max_iter=100)
clf.fit(X, y, sample_weight=sample_weight)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
samples_weights = ax.contour(xx, yy, Z, levels=[0], linestyles=["dashed"])

no_weights_handles, _ = no_weights.legend_elements()
weights_handles, _ = samples_weights.legend_elements()
ax.legend(
    [no_weights_handles[0], weights_handles[0]],
    ["no weights", "with weights"],
    loc="lower left",
)

ax.set(xticks=(), yticks=())
plt.title("SGD: Effect of Weighted Samples")

plt.tight_layout()
plt.show()
