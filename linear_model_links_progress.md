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
- `plot_out_of_core_classification.py` - Out of core classification
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
- [x] LinearRegression - Added link to `plot_ols_ridge.py`
- [x] Ridge - Added links to `plot_ols_ridge.py`, `plot_ridge_path.py`, `plot_ridge_coeffs.py`, and `plot_huber_vs_ridge.py`
- [x] RidgeClassifier - Added link to `plot_document_classification_20newsgroups.py`
- [x] RidgeCV - Added links to `plot_transformed_target.py`, `plot_cyclical_feature_engineering.py`, `plot_linear_model_coefficient_interpretation.py`, `plot_select_from_model_diabetes.py`, and `plot_stack_predictors.py`
- [ ] RidgeClassifierCV - No specific examples found

### Bayesian Regression
- [x] ARDRegression - Added links to `plot_ard.py` and `plot_lasso_and_elasticnet.py`
- [x] BayesianRidge - Added links to `plot_bayesian_ridge_curvefit.py`, `plot_ard.py`, `plot_feature_agglomeration_vs_univariate_selection.py`, and `plot_iterative_imputer_variants_comparison.py`

### Coordinate Descent Models
- [x] ElasticNet - Added links to `plot_lasso_and_elasticnet.py`, `plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.py`, and `plot_train_error_vs_test_error.py`
- [x] ElasticNetCV - Added link to `plot_lasso_model_selection.py`
- [x] Lasso - Added links to `plot_lasso_and_elasticnet.py`, `plot_lasso_model_selection.py`, `plot_lasso_lars_ic.py`, and `plot_tomography_l1_reconstruction.py`
- [x] LassoCV - Added links to `plot_lasso_model_selection.py` and `plot_linear_model_coefficient_interpretation.py`
- [x] LassoLars - Added link to `plot_lasso_lasso_lars_elasticnet_path.py`
- [x] LassoLarsCV - Added link to `plot_lasso_model_selection.py`
- [x] LassoLarsIC - Added links to `plot_lasso_lars_ic.py` and `plot_lasso_model_selection.py`
- [ ] MultiTaskElasticNet - No examples found specifically for this class
- [ ] MultiTaskElasticNetCV - No examples found specifically for this class
- [x] MultiTaskLasso - Added link to `plot_multi_task_lasso_support.py`
- [ ] MultiTaskLassoCV - No examples found specifically for this class

### Generalized Linear Models (GLM)
- [x] PoissonRegressor - Completed - Added links to `plot_poisson_regression_non_normal_loss.py` and `plot_tweedie_regression_insurance_claims.py`
- [x] GammaRegressor - Completed - Added link to `plot_tweedie_regression_insurance_claims.py`
- [x] TweedieRegressor - Completed - Added link to `plot_tweedie_regression_insurance_claims.py`

### Least Angle Regression
- [x] Lars - Added example links
- [x] LarsCV - Added example links

### Logistic Regression
- [x] LogisticRegression - Added example links: `plot_logistic.py`, `plot_logistic_path.py`, `plot_logistic_l1_l2_sparsity.py`, `plot_logistic_multinomial.py`
- [x] LogisticRegressionCV - Added example links: `plot_compare_calibration.py`, `plot_scaling_importance.py`

### Orthogonal Matching Pursuit (OMP)
- [x] OrthogonalMatchingPursuit - Added example link: `plot_omp.py`
- [x] OrthogonalMatchingPursuitCV - Added example link: `plot_omp.py`

### Stochastic Gradient Descent
- [x] SGDClassifier - Added example links: `plot_sgd_comparison.py`, `plot_sgd_iris.py`, `plot_sgd_loss_functions.py`, `plot_sgd_penalties.py`, `plot_sgd_separating_hyperplane.py`, `plot_sgd_early_stopping.py`
- [x] SGDRegressor - Added example links: `plot_sgd_comparison.py`, `plot_sgd_loss_functions.py`, `plot_sgd_early_stopping.py`
- [x] SGDOneClassSVM - Added example link: `plot_sgdocsvm_vs_ocsvm.py`

### Perceptron
- [x] Perceptron - Added example links: `plot_sgd_comparison.py`, `plot_sgd_loss_functions.py`, `plot_out_of_core_classification.py`

### Passive Aggressive Algorithms
- [x] PassiveAggressiveClassifier - Added example links: `plot_sgd_comparison.py`, `plot_out_of_core_classification.py`
- [ ] PassiveAggressiveRegressor - No suitable examples found in the codebase

### Robust Regression
- [x] HuberRegressor - Added example link: `plot_huber_vs_ridge.py`
- [x] QuantileRegressor - Added example link: `plot_quantile_regression.py`
- [x] RANSACRegressor - Already had example link to `plot_ransac.py`
- [x] TheilSenRegressor - Added example link: `plot_theilsen.py`

## Strategy
1. First check each example file to understand what classes it demonstrates
2. Update the mapping above based on the actual content of examples
3. Add links to docstrings one by one
4. Track progress in this file

## Notes
- The automated Python scripting approach worked well for adding example links to classes in large files that were causing timeout issues when edited directly.
- We've successfully added example links to coordinate descent models and least angle regression models.
- The same approach can be applied to any remaining large files requiring docstring updates.
