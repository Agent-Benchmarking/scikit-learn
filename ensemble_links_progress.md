# Ensemble Module Example Links Progress

## Available Example Files

### Bagging and Boosting Examples
- `plot_adaboost_twoclass.py` - Demonstrates AdaBoostClassifier on a binary classification task
- `plot_adaboost_multiclass.py` - Shows AdaBoostClassifier on a multi-class classification problem
- `plot_adaboost_regression.py` - Demonstrates AdaBoostRegressor for regression
- `plot_bias_variance.py` - Shows how bagging can reduce overfitting
- `plot_ensemble_oob.py` - Demonstrates the oob_score parameter in bagging methods

### Random Forest Examples
- `plot_forest_iris.py` - Demonstrates RandomForestClassifier on the Iris dataset
- `plot_forest_importances.py` - Shows feature importances in RandomForest
- `plot_forest_hist_grad_boosting_comparison.py` - Compares RandomForestClassifier with HistGradientBoostingClassifier
- `plot_random_forest_regression_multioutput.py` - Shows RandomForestRegressor with multiple outputs
- `plot_random_forest_embedding.py` - Using RandomForestClassifier to create embeddings

### Gradient Boosting Examples
- `plot_gradient_boosting_regression.py` - Shows GradientBoostingRegressor for regression
- `plot_gradient_boosting_oob.py` - Demonstrates out-of-bag estimates with GradientBoostingRegressor
- `plot_gradient_boosting_regularization.py` - Shows regularization in gradient boosting
- `plot_gradient_boosting_early_stopping.py` - Demonstrates early stopping in gradient boosting
- `plot_gradient_boosting_quantile.py` - Shows quantile regression with GradientBoostingRegressor
- `plot_gradient_boosting_categorical.py` - Demonstrates handling categorical features in gradient boosting
- `plot_hgbt_regression.py` - Shows HistGradientBoostingRegressor
- `plot_monotonic_constraints.py` - Demonstrates monotonic constraints in HistGradientBoosting

### Voting and Stacking Examples
- `plot_voting_probas.py` - Shows VotingClassifier with probability outputs
- `plot_voting_decision_regions.py` - Shows decision regions with VotingClassifier
- `plot_voting_regressor.py` - Demonstrates VotingRegressor
- `plot_stack_predictors.py` - Shows stacking ensemble methods

### Other Examples
- `plot_isolation_forest.py` - Demonstrates IsolationForest for anomaly detection
- `plot_anomaly_comparison.py` - Compares different anomaly detection methods
- `plot_outlier_detection_bench.py` - Benchmarks different outlier detection methods
- `plot_feature_transformation.py` - Shows feature transformation with ensemble models

## Classes and Their Example Links

### To Do
- [x] BaggingClassifier - Added links to `plot_bias_variance.py` and `plot_ensemble_oob.py`
- [x] BaggingRegressor - Added links to `plot_bias_variance.py` and `plot_ensemble_oob.py`
- [x] RandomForestClassifier - Added links to `plot_forest_iris.py`, `plot_forest_importances.py`, and `plot_random_forest_embedding.py`
- [x] RandomForestRegressor - Added links to `plot_forest_importances.py` and `plot_gradient_boosting_regression.py`
- [x] ExtraTreesClassifier - Added links to `plot_forest_importances.py`, `plot_forest_iris.py`, and `plot_feature_selection.py`
- [x] ExtraTreesRegressor - Added links to `plot_forest_importances.py` and `plot_gradient_boosting_regression.py`
- [x] GradientBoostingClassifier - Added links to `plot_gradient_boosting_oob.py`, `plot_gradient_boosting_regularization.py`, and `plot_feature_transformation.py`
- [x] GradientBoostingRegressor - Added links to `plot_gradient_boosting_regression.py`, `plot_gradient_boosting_quantile.py`, and `plot_gradient_boosting_oob.py`
- [x] HistGradientBoostingClassifier - Added links to `plot_forest_hist_grad_boosting_comparison.py`, `plot_feature_transformation.py`, and `plot_cost_sensitive_learning.py`
- [x] HistGradientBoostingRegressor - Added links to `plot_hgbt_regression.py`, `plot_forest_hist_grad_boosting_comparison.py`, `plot_time_series_lagged_features.py`, and `plot_partial_dependence.py`
- [x] IsolationForest - Added links to `plot_isolation_forest.py`, `plot_anomaly_comparison.py`, and `plot_outlier_detection_bench.py`
- [x] StackingClassifier - Added links to `plot_release_highlights_0_22_0.py` and `plot_stack_predictors.py`
- [x] StackingRegressor - Added links to `plot_stack_predictors.py` and `plot_release_highlights_0_22_0.py`
- [x] VotingClassifier - Added links to `plot_voting_decision_regions.py` and `plot_voting_probas.py`
- [x] VotingRegressor - Added link to `plot_voting_regressor.py`
- [x] AdaBoostClassifier - Already has links to `plot_adaboost_multiclass.py` and `plot_adaboost_twoclass.py`
- [x] AdaBoostRegressor - Already has link to `plot_adaboost_regression.py`

## Strategy
1. First check each class docstring to understand its current structure
2. Identify relevant examples for each class
3. Add links to docstrings one by one
4. Track progress in this file

## Notes
- Focus on making examples more discoverable in the API documentation
- Will add one example reference at a time to avoid edit conflicts
