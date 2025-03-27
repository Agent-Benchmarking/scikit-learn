"""
===============================================
SGD: Convex Loss Functions and Regularization
===============================================

This example illustrates the various loss functions and penalties supported
by :class:`~sklearn.linear_model.SGDClassifier` and
:class:`~sklearn.linear_model.SGDRegressor`.

The first plot shows the different convex loss functions for classification, while
the second plot displays the contours of where different penalties are equal to 1.

Understanding loss functions and penalties is crucial for effective use of SGD-based
models in scikit-learn.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

# %%
# Part 1: SGD Loss Functions
# -------------------------
#
# First, we visualize the different loss functions available for
# :class:`~sklearn.linear_model.SGDClassifier`.
#
# These loss functions determine how the model penalizes prediction errors.


def modified_huber_loss(y_true, y_pred):
    z = y_pred * y_true
    loss = -4 * z
    loss[z >= -1] = (1 - z[z >= -1]) ** 2
    loss[z >= 1.0] = 0
    return loss


plt.figure(figsize=(10, 6))

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
plt.title("SGD Loss Functions for Classification")

# %%
# The plot shows various loss functions used in SGD-based classifiers:
#
# - **Zero-one loss**: The ideal classification loss that we're approximating
#   with convex surrogates
# - **Hinge loss**: Used by Support Vector Machines (SVMs), convex approximation of
#   zero-one loss
# - **Perceptron loss**: Linear function, doesn't penalize correct classifications
# - **Log loss**: Used in logistic regression, smooth and probabilistic
# - **Squared hinge loss**: Squared version of hinge loss, more strongly penalizes
#   margin violations
# - **Modified Huber loss**: Smoothed hinge loss that is less sensitive to outliers
#
# Different loss functions have different properties in terms of margin sensitivity,
# probabilistic interpretation, and robustness to outliers.
#
# Part 2: SGD Penalties (Regularization)
# -------------------------------------
#
# Next, we visualize the different regularization penalties used in
# :class:`~sklearn.linear_model.SGDClassifier` and
# :class:`~sklearn.linear_model.SGDRegressor`.

plt.figure(figsize=(10, 8))

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
    fontsize=18,
    fmt={1.0: "elastic-net"},
    manual=[(-0.3, -1.2)],
)
plt.clabel(l2_contour, inline=1, fontsize=18, fmt={1.0: "L2"}, manual=[(1.1, 0.3)])
plt.clabel(l1_contour, inline=1, fontsize=18, fmt={1.0: "L1"}, manual=[(-1.1, 0.3)])

plt.title("Regularization Penalties: Contours Where Penalty = 1")
plt.axis([-1.7, 1.7, -1.7, 1.7])

plt.tight_layout()
plt.show()

# %%
# The plot shows the contours where different penalties equal 1:
#
# - **L1 penalty (Lasso)**: Creates a diamond shape, promotes sparsity by
#   driving some coefficients to exactly zero
# - **L2 penalty (Ridge)**: Creates a circle shape, shrinks all coefficients
#   proportionally but rarely to exactly zero
# - **Elastic Net**: A mixture of L1 and L2 penalties (in this case with ρ=0.5),
#   combines the benefits of both
#
# Choosing the right penalty:
#
# - **L1 (Lasso)**: Good for feature selection and sparse models
# - **L2 (Ridge)**: Good for dealing with multicollinearity and preventing overfitting
# - **Elastic Net**: Good when you have many correlated features
#
# These regularization techniques help prevent overfitting by constraining the
# magnitude of the coefficient values in the model.
#
# Conclusion
# ----------
#
# When using SGD-based models in scikit-learn, you need to select both an appropriate
# loss function and regularization method:
#
# - The loss function determines how errors are penalized during training
# - The regularization penalty controls model complexity and prevents overfitting
#
# The optimal combination depends on your specific problem, data characteristics,
# and whether you need sparsity, robustness to outliers, or probabilistic outputs.
