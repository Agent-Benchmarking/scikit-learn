# Model Selection Module Example Links Progress

This document tracks progress on adding example links to the scikit-learn model_selection module documentation as part of issue #30621.

## Classes and Examples Mapping

### Cross-Validation and Splitters
- [x] KFold - plot_cv_indices.py
- [x] GroupKFold - plot_cv_indices.py
- [x] StratifiedKFold - plot_cv_indices.py, plot_grid_search_digits.py, plot_roc_crossval.py
- [x] TimeSeriesSplit - plot_cv_indices.py
- [x] LeaveOneOut - plot_cv_indices.py
- [x] LeavePOut - plot_cv_indices.py
- [x] ShuffleSplit - plot_cv_indices.py
- [x] GroupShuffleSplit - plot_cv_indices.py
- [x] StratifiedShuffleSplit - plot_cv_indices.py
- [x] PredefinedSplit - plot_cv_indices.py

### Model Selection
- [x] GridSearchCV - plot_grid_search_digits.py, plot_grid_search_text_feature_extraction.py, plot_grid_search_refit_callable.py, plot_grid_search_stats.py
- [x] RandomizedSearchCV - plot_randomized_search.py
- [x] HalvingGridSearchCV - plot_successive_halving_heatmap.py, plot_successive_halving_iterations.py
- [x] HalvingRandomSearchCV - plot_successive_halving_heatmap.py, plot_successive_halving_iterations.py

### Metrics and Validation
- [x] cross_validate - plot_cv_predict.py, plot_nested_cross_validation_iris.py
- [x] cross_val_score - plot_cv_predict.py, plot_learning_curve.py
- [x] cross_val_predict - plot_cv_predict.py
- [x] learning_curve - plot_learning_curve.py, plot_train_error_vs_test_error.py
- [x] validation_curve - plot_underfitting_overfitting.py
- [x] TunedThresholdClassifierCV - plot_tuned_decision_threshold.py

## Progress

- [x] KFold
- [x] GroupKFold
- [x] StratifiedKFold
- [x] TimeSeriesSplit
- [x] LeaveOneOut
- [x] LeavePOut
- [x] ShuffleSplit
- [x] GroupShuffleSplit
- [x] StratifiedShuffleSplit
- [x] PredefinedSplit
- [x] GridSearchCV
- [x] RandomizedSearchCV
- [x] HalvingGridSearchCV
- [x] HalvingRandomSearchCV
- [x] cross_validate
- [x] cross_val_score
- [x] cross_val_predict
- [x] learning_curve
- [x] validation_curve
- [x] TunedThresholdClassifierCV

## Notes
- The model_selection module has both classes and functions that need example links
- Many of the examples demonstrate multiple classes/functions in the same file
- All example links have been successfully added to the docstrings
- Fixed search patterns for KFold, StratifiedKFold, HalvingRandomSearchCV, and TunedThresholdClassifierCV to match actual docstrings
