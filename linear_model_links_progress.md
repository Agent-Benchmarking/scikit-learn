# Linear Model Module Example Links Progress

## Available Example Files

### Linear Model Examples
- `plot_ard.py` - Demonstrates ARDRegression
- `plot_bayesian_ridge_curvefit.py` - Shows BayesianRidge for curve fitting
- `plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.py` - ElasticNet with precomputed gram matrix
- `plot_huber_vs_ridge.py` - Compares HuberRegressor vs Ridge on corrupted data
- `plot_lasso_and_elasticnet.py` - Lasso and ElasticNet for sparse signals
- `plot_lasso_dense_vs_sparse_data.py` - Lasso on dense vs. sparse data
- `plot_lasso_lars_ic.py` - Lasso model selection with Lars using BIC/AIC
- `plot_lasso_lasso_lars_elasticnet_path.py` - Regularization path of Lasso, LassoLars and ElasticNet
- `plot_lasso_model_selection.py` - Lasso model selection with cross-validation
- `plot_logistic_l1_l2_sparsity.py` - L1 and L2 LogisticRegression for sparsity
- `plot_logistic_multinomial.py` - L1 LogisticRegression with multinomial loss
- `plot_logistic_path.py` - Regularization path of LogisticRegression
- `plot_logistic.py` - Logistic Regression 3-class classifier
- `plot_multi_task_lasso_support.py` - Joint feature selection with multi-task Lasso
- `plot_nnls.py` - Non-negative least squares
- `plot_ols_ridge.py` - Ordinary Least Squares and Ridge regression
- `plot_omp.py` - Orthogonal Matching Pursuit (OMP)
- `plot_poisson_regression_non_normal_loss.py` - Poisson regression
- `plot_polynomial_interpolation.py` - Polynomial interpolation
- `plot_quantile_regression.py` - Quantile regression
- `plot_ransac.py` - RANSAC regression
- `plot_ridge_coeffs.py` - Ridge coefficients as a function of regularization
- `plot_ridge_path.py` - Regularization path of Ridge regression
- `plot_robust_fit.py` - Robust linear model estimation
- `plot_sgd_comparison.py` - SGD comparison of different algorithms
- `plot_sgd_early_stopping.py` - Early stopping in SGD
- `plot_sgd_iris.py` - SGD classification on Iris dataset
- `plot_sgd_loss_functions.py` - SGD loss functions
- `plot_sgdocsvm_vs_ocsvm.py` - SGD OneClassSVM vs traditional OneClassSVM
- `plot_sgd_penalties.py` - SGD various penalties
- `plot_sgd_separating_hyperplane.py` - SGD separating hyperplane
- `plot_sgd_weighted_samples.py` - SGD with weighted samples
- `plot_sparse_logistic_regression_20newsgroups.py` - Sparse logistic regression on 20newsgroups
- `plot_sparse_logistic_regression_mnist.py` - Sparse logistic regression on MNIST
- `plot_theilsen.py` - Theil-Sen regression
- `plot_tweedie_regression_insurance_claims.py` - Tweedie regression on insurance claims

## Classes and Their Example Links

### Base and Helper Classes
- [ ] LinearRegression - Check `plot_ols_ridge.py`
- [ ] Ridge - Check `plot_ols_ridge.py`, `plot_ridge_path.py`, `plot_ridge_coeffs.py`, `plot_huber_vs_ridge.py`
- [ ] RidgeClassifier - Check for relevant examples
- [ ] RidgeCV - Check for relevant examples
- [ ] RidgeClassifierCV - Check for relevant examples

### Bayesian Regression
- [ ] ARDRegression - Check `plot_ard.py`
- [ ] BayesianRidge - Check `plot_bayesian_ridge_curvefit.py`

### Coordinate Descent Models
- [ ] ElasticNet - Check `plot_lasso_and_elasticnet.py`, `plot_lasso_lasso_lars_elasticnet_path.py`
- [ ] ElasticNetCV - Check for relevant examples
- [ ] Lasso - Check `plot_lasso_and_elasticnet.py`, `plot_lasso_lasso_lars_elasticnet_path.py`, `plot_lasso_dense_vs_sparse_data.py`
- [ ] LassoCV - Check `plot_lasso_model_selection.py`
- [ ] LassoLars - Check `plot_lasso_lasso_lars_elasticnet_path.py`
- [ ] LassoLarsCV - Check for relevant examples
- [ ] LassoLarsIC - Check `plot_lasso_lars_ic.py`
- [ ] MultiTaskElasticNet - Check for relevant examples
- [ ] MultiTaskElasticNetCV - Check for relevant examples
- [ ] MultiTaskLasso - Check `plot_multi_task_lasso_support.py`
- [ ] MultiTaskLassoCV - Check for relevant examples

### Generalized Linear Models (GLM)
- [ ] PoissonRegressor - Check `plot_poisson_regression_non_normal_loss.py`
- [ ] GammaRegressor - Check for relevant examples
- [ ] TweedieRegressor - Check `plot_tweedie_regression_insurance_claims.py`

### Least Angle Regression
- [ ] Lars - Check for relevant examples
- [ ] LarsCV - Check for relevant examples

### Logistic Regression
- [ ] LogisticRegression - Check `plot_logistic.py`, `plot_logistic_path.py`, `plot_logistic_l1_l2_sparsity.py`, `plot_logistic_multinomial.py`
- [ ] LogisticRegressionCV - Check for relevant examples

### Orthogonal Matching Pursuit (OMP)
- [ ] OrthogonalMatchingPursuit - Check `plot_omp.py`
- [ ] OrthogonalMatchingPursuitCV - Check for relevant examples

### Stochastic Gradient Descent
- [ ] SGDClassifier - Check `plot_sgd_comparison.py`, `plot_sgd_iris.py`, `plot_sgd_loss_functions.py`, `plot_sgd_penalties.py`, `plot_sgd_separating_hyperplane.py`
- [ ] SGDRegressor - Check `plot_sgd_comparison.py`, `plot_sgd_loss_functions.py`
- [ ] SGDOneClassSVM - Check `plot_sgdocsvm_vs_ocsvm.py`

### Perceptron
- [ ] Perceptron - Check for relevant examples

### Passive Aggressive Algorithms
- [ ] PassiveAggressiveClassifier - Check for relevant examples
- [ ] PassiveAggressiveRegressor - Check for relevant examples

### Robust Regression
- [ ] HuberRegressor - Check `plot_huber_vs_ridge.py`
- [ ] QuantileRegressor - Check `plot_quantile_regression.py`
- [ ] RANSACRegressor - Check `plot_ransac.py`
- [ ] TheilSenRegressor - Check `plot_theilsen.py`

## Strategy
1. First check each example file to understand what classes it demonstrates
2. Update the mapping above based on the actual content of examples
3. Add links to docstrings one by one
4. Track progress in this file

## Notes
- Focus on making examples more discoverable in the API documentation
- Some classes may already have example links - check before adding
- Will add one example reference at a time to avoid edit conflicts
