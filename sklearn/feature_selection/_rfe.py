# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Recursive feature elimination for feature ranking"""

import warnings
from copy import deepcopy
from numbers import Integral

import numpy as np
from joblib import effective_n_jobs

from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..metrics import get_scorer
from ..model_selection import check_cv
from ..model_selection._validation import _score
from ..utils import Bunch, metadata_routing
from ..utils._metadata_requests import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from ..utils._param_validation import HasMethods, Interval, RealNotInt
from ..utils._tags import get_tags
from ..utils.metaestimators import _safe_split, available_if
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
    _check_method_params,
    _deprecate_positional_args,
    _estimator_has,
    check_is_fitted,
    validate_data,
)
from ._base import SelectorMixin, _get_feature_importances


def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer, routed_params):
    """
    Return the score and n_features per step for a fit across one fold.

    Parameters
    ----------
    rfe : RFE instance
        The RFE instance to fit.
    estimator : estimator instance
        The estimator to fit.
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    train : array-like of shape (n_train_samples,)
        The indices of the training samples.
    test : array-like of shape (n_test_samples,)
        The indices of the test samples.
    scorer : callable or dict of callables
        The scorer(s) to use. If multiple scorers are specified,
        a dict mapping scorer name to the scorer callable is returned.
    routed_params : dict
        Parameters to route to the estimator and scorer.

    Returns
    -------
    scores : array-like or dict of array-like
        The scores for each step. If multiple scorers are used, a dict
        mapping scorer name to scores is returned.
    n_features : array-like
        The number of features used at each step.
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    fit_params = _check_method_params(
        X, params=routed_params.estimator.fit, indices=train
    )
    score_params = _check_method_params(
        X=X, params=routed_params.scorer.score, indices=test
    )

    # Handle multiple scorers
    if isinstance(scorer, dict):
        # Create a scoring function that returns a dict of scores
        def _multimetric_score(estimator, features):
            return {
                name: _score(
                    estimator,
                    X_test[:, features],
                    y_test,
                    scorer_,
                    score_params=score_params,
                )
                for name, scorer_ in scorer.items()
            }

        # Fit the RFE with the multimetric scorer
        rfe._fit(X_train, y_train, _multimetric_score, **fit_params)

        # Return scores for each metric and n_features
        return rfe.step_scores_, rfe.step_n_features_
    else:
        # Single scorer case (original behavior)
        rfe._fit(
            X_train,
            y_train,
            lambda estimator, features: _score(
                estimator,
                X_test[:, features],
                y_test,
                scorer,
                score_params=score_params,
            ),
            **fit_params,
        )

        return rfe.step_scores_, rfe.step_n_features_


class RFE(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through
    any specific attribute or callable.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance
        (e.g. `coef_`, `feature_importances_`).

    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

        .. versionchanged:: 0.24
           Added float values for fractions.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, default=0
        Controls verbosity of output.

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a classifier.

    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.

    n_features_ : int
        The number of selected features.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    See Also
    --------
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features.
    SelectFromModel : Feature selection based on thresholds of importance
        weights.
    SequentialFeatureSelector : Sequential cross-validation based feature
        selection. Does not rely on importance weights.

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.

    Examples
    --------
    The following example shows how to retrieve the 5 most informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, n_features_to_select=5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "n_features_to_select": [
            None,
            Interval(RealNotInt, 0, 1, closed="right"),
            Interval(Integral, 0, None, closed="neither"),
        ],
        "step": [
            Interval(Integral, 0, None, closed="neither"),
            Interval(RealNotInt, 0, 1, closed="neither"),
        ],
        "verbose": ["verbose"],
        "importance_getter": [str, callable],
    }

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select=None,
        step=1,
        verbose=0,
        importance_getter="auto",
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.importance_getter = importance_getter
        self.verbose = verbose

    # TODO(1.8) remove this property
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        """Classes labels available when `estimator` is a classifier.

        Returns
        -------
        ndarray of shape (n_classes,)
        """
        return self.estimator_.classes_

    @_fit_context(
        # RFE.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        """Fit the RFE model and then the underlying estimator on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        **fit_params : dict
            - If `enable_metadata_routing=False` (default): Parameters directly passed
              to the ``fit`` method of the underlying estimator.

            - If `enable_metadata_routing=True`: Parameters safely routed to the ``fit``
              method of the underlying estimator.

            .. versionchanged:: 1.6
                See :ref:`Metadata Routing User Guide <metadata_routing>`
                for more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch(estimator=Bunch(fit=fit_params))

        return self._fit(X, y, **routed_params.estimator.fit)

    def _fit(self, X, y, step_score=None, **fit_params):
        # Parameter step_score controls the calculation of self.step_scores_
        # step_score is not exposed to users and is used when implementing RFECV
        # self.step_scores_ will not be calculated when calling _fit through fit

        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            ensure_all_finite=False,
            multi_output=True,
        )

        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):  # int
            n_features_to_select = self.n_features_to_select
            if n_features_to_select > n_features:
                warnings.warn(
                    (
                        f"Found {n_features_to_select=} > {n_features=}. There will be"
                        " no feature selection and all features will be kept."
                    ),
                    UserWarning,
                )
        else:  # float
            n_features_to_select = int(n_features * self.n_features_to_select)

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)

        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)

        if step_score:
            self.step_n_features_ = []
            self.step_scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y, **fit_params)

            # Get importance and rank them
            importances = _get_feature_importances(
                estimator,
                self.importance_getter,
                transform_func="square",
            )
            ranks = np.argsort(importances)

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.step_n_features_.append(len(features))
                self.step_scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y, **fit_params)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.step_n_features_.append(len(features))
            self.step_scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Reduce X to the selected features and predict using the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        **predict_params : dict
            Parameters to route to the ``predict`` method of the
            underlying estimator.

            .. versionadded:: 1.6
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>`
                for more details.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        _raise_for_params(predict_params, self, "predict")
        check_is_fitted(self)
        if _routing_enabled():
            routed_params = process_routing(self, "predict", **predict_params)
        else:
            routed_params = Bunch(estimator=Bunch(predict={}))

        return self.estimator_.predict(
            self.transform(X), **routed_params.estimator.predict
        )

    @available_if(_estimator_has("score"))
    def score(self, X, y, **score_params):
        """Reduce X to the selected features and return the score of the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.

        **score_params : dict
            - If `enable_metadata_routing=False` (default): Parameters directly passed
              to the ``score`` method of the underlying estimator.

            - If `enable_metadata_routing=True`: Parameters safely routed to the `score`
              method of the underlying estimator.

            .. versionadded:: 1.0

            .. versionchanged:: 1.6
                See :ref:`Metadata Routing User Guide <metadata_routing>`
                for more details.

        Returns
        -------
        score : float
            Score of the underlying base estimator computed with the selected
            features returned by `rfe.transform(X)` and `y`.
        """
        check_is_fitted(self)
        if _routing_enabled():
            routed_params = process_routing(self, "score", **score_params)
        else:
            routed_params = Bunch(estimator=Bunch(score=score_params))

        return self.estimator_.score(
            self.transform(X), y, **routed_params.estimator.score
        )

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape = [n_samples, n_classes] or [n_samples]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        check_is_fitted(self)
        return self.estimator_.decision_function(self.transform(X))

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.estimator_.predict_proba(self.transform(X))

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.estimator_.predict_log_proba(self.transform(X))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        sub_estimator_tags = get_tags(self.estimator)
        tags.estimator_type = sub_estimator_tags.estimator_type
        tags.classifier_tags = deepcopy(sub_estimator_tags.classifier_tags)
        tags.regressor_tags = deepcopy(sub_estimator_tags.regressor_tags)
        if tags.classifier_tags is not None:
            tags.classifier_tags.poor_score = True
        if tags.regressor_tags is not None:
            tags.regressor_tags.poor_score = True
        tags.target_tags.required = True
        tags.input_tags.sparse = sub_estimator_tags.input_tags.sparse
        tags.input_tags.allow_nan = sub_estimator_tags.input_tags.allow_nan
        return tags

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.6

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="score", callee="score"),
        )
        return router


class RFECV(RFE):
    """Recursive feature elimination with cross-validation to select features.

    The number of features selected is tuned automatically by fitting an :class:`RFE`
    selector on the different cross-validation splits (provided by the `cv` parameter).
    The performance of each :class:`RFE` selector is evaluated using `scoring` for
    different numbers of selected features and aggregated together. Finally, the scores
    are averaged across folds and the number of features selected is set to the number
    of features that maximize the cross-validation score.
    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.

    min_features_to_select : int, default=1
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and ``min_features_to_select`` isn't divisible by
        ``step``.

        .. versionadded:: 0.20

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is not a classifier or if ``y`` is neither binary nor multiclass,
        :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value of None changed from 3-fold to 5-fold.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_string_names`);
        - a callable (see :ref:`scoring_callable`) that returns a single value;
        - `None`, the `estimator`'s
          :ref:`default evaluation criterion <scoring_api_overview>` is used.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables as values.

        When multiple metrics are used, the `cv_results_` attribute will contain
        metrics for each scorer, but the best number of features will not be
        automatically selected. The user must decide which metric to use for
        feature selection.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int or None, default=None
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance.
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a classifier.

    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.

    cv_results_ : dict of ndarrays
        All arrays (values of the dictionary) are sorted in ascending order
        by the number of features used (i.e., the first element of the array
        represents the models that used the least number of features, while the
        last element represents the models that used all available features).

        .. versionadded:: 1.0

        This dictionary contains the following keys:

        split(k)_test_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.
            When multiple metrics are used, this will be replaced by
            split(k)_test_{scorer_name} for each scorer.

        mean_test_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.
            When multiple metrics are used, this will be replaced by
            mean_test_{scorer_name} for each scorer.

        std_test_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.
            When multiple metrics are used, this will be replaced by
            std_test_{scorer_name} for each scorer.

        n_features : ndarray of shape (n_subsets_of_features,)
            Number of features used at each step.

            .. versionadded:: 1.5

    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    multimetric_ : bool
        Whether multiple metrics were used for scoring.

        .. versionadded:: 1.6

    ranking_ : narray of shape (n_features,)
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    See Also
    --------
    RFE : Recursive feature elimination.

    Notes
    -----
    The size of all values in ``cv_results_`` is equal to
    ``ceil((n_features - min_features_to_select) / step) + 1``,
    where step is the number of features removed at each iteration.

    Allows NaN/Inf in the input if the underlying estimator does as well.

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFECV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    The following example shows how to use multiple scoring metrics to
    evaluate feature selection.

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(n_samples=100, n_features=10,
    ...                            n_informative=5, random_state=42)
    >>> estimator = SVC(kernel="linear")
    >>> scoring = ['accuracy', 'precision_macro', 'recall_macro']
    >>> selector = RFECV(estimator, step=1, cv=5, scoring=scoring)
    >>> selector = selector.fit(X, y)
    >>> # Access scores for different metrics
    >>> selector.cv_results_['mean_test_accuracy']
    array([...])
    >>> selector.cv_results_['mean_test_precision_macro']
    array([...])
    >>> # Score using a specific metric
    >>> selector.score(X, y, metric='accuracy')
    0.98
    """

    _parameter_constraints: dict = {
        **RFE._parameter_constraints,
        "min_features_to_select": [Interval(Integral, 0, None, closed="neither")],
        "cv": ["cv_object"],
        "scoring": [None, str, callable, list, tuple, dict],
        "n_jobs": [None, Integral],
    }
    _parameter_constraints.pop("n_features_to_select")
    __metadata_request__fit = {"groups": metadata_routing.UNUSED}

    def __init__(
        self,
        estimator,
        *,
        step=1,
        min_features_to_select=1,
        cv=None,
        scoring=None,
        verbose=0,
        n_jobs=None,
        importance_getter="auto",
    ):
        self.estimator = estimator
        self.step = step
        self.importance_getter = importance_getter
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_features_to_select = min_features_to_select

    # TODO(1.8): remove `groups` from the signature after deprecation cycle.
    @_deprecate_positional_args(version="1.8")
    @_fit_context(
        # RFECV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, *, groups=None, **params):
        """Fit the RFE model and automatically tune the number of selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.

        y : array-like of shape (n_samples,)
            Target values (integers for classification, real numbers for
            regression).

        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

            .. versionadded:: 0.20

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator,
            the scorer, and the CV splitter.

            .. versionadded:: 1.6
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>`
                for more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _raise_for_params(params, self, "fit")
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse="csr",
            ensure_min_features=2,
            ensure_all_finite=False,
            multi_output=True,
        )

        if _routing_enabled():
            if groups is not None:
                params.update({"groups": groups})
            routed_params = process_routing(self, "fit", **params)
        else:
            routed_params = Bunch(
                estimator=Bunch(fit={}),
                splitter=Bunch(split={"groups": groups}),
                scorer=Bunch(score={}),
            )

        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = self._get_scorer()

        # Build an RFE object, which will evaluate and score each possible
        # feature count, down to self.min_features_to_select
        n_features = X.shape[1]
        if self.min_features_to_select > n_features:
            warnings.warn(
                (
                    f"Found min_features_to_select={self.min_features_to_select} > "
                    f"{n_features=}. There will be no feature selection and all "
                    "features will be kept."
                ),
                UserWarning,
            )
        rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=min(self.min_features_to_select, n_features),
            importance_getter=self.importance_getter,
            step=self.step,
            verbose=self.verbose,
        )

        # Determine the number of subsets of features by fitting across
        # the train folds and choosing the "features_to_select" parameter
        # that gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods
        # even if n_jobs is set to 1 with the default multiprocessing
        # backend.
        # This branching is done so that to
        # make sure that user code that sets n_jobs to 1
        # and provides bound methods as scorers is not broken with the
        # addition of n_jobs parameter in version 0.18.

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)

        scores_features = parallel(
            func(clone(rfe), self.estimator, X, y, train, test, scorer, routed_params)
            for train, test in cv.split(X, y, **routed_params.splitter.split)
        )
        scores, step_n_features = zip(*scores_features)

        step_n_features_rev = np.array(step_n_features[0])[::-1]
        scores = np.array(scores)

        # Reverse order such that lowest number of features is selected in case of tie.
        scores_sum_rev = np.sum(scores, axis=0)[::-1]
        n_features_to_select = step_n_features_rev[np.argmax(scores_sum_rev)]

        # Re-execute an elimination with best_k over the whole set
        rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=n_features_to_select,
            step=self.step,
            importance_getter=self.importance_getter,
            verbose=self.verbose,
        )

        rfe.fit(X, y, **routed_params.estimator.fit)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._transform(X), y, **routed_params.estimator.fit)

        # Handle multiple scorers
        self.multimetric_ = isinstance(scores[0], dict)

        if self.multimetric_:
            # Initialize cv_results_ with n_features
            self.cv_results_ = {"n_features": step_n_features_rev}

            # Convert scores to a more convenient format
            # scores is a list of dicts, one dict per fold
            # We want to convert it to a dict of lists, one list per metric
            metrics = list(scores[0].keys())
            scores_dict = {metric: [] for metric in metrics}

            for fold_scores in scores:
                for metric in metrics:
                    scores_dict[metric].append(fold_scores[metric])

            # Now scores_dict is a dict of lists, one list per metric
            # Each list contains the scores for each fold
            for metric in metrics:
                # Reverse to stay consistent with original implementation
                metric_scores = np.array(scores_dict[metric])[:, ::-1]
                self.cv_results_[f"mean_test_{metric}"] = np.mean(metric_scores, axis=0)
                self.cv_results_[f"std_test_{metric}"] = np.std(metric_scores, axis=0)
                for i in range(metric_scores.shape[0]):
                    self.cv_results_[f"split{i}_test_{metric}"] = metric_scores[i]
        else:
            # Original behavior for single scorer
            scores_rev = scores[:, ::-1]
            self.cv_results_ = {
                "mean_test_score": np.mean(scores_rev, axis=0),
                "std_test_score": np.std(scores_rev, axis=0),
                **{
                    f"split{i}_test_score": scores_rev[i]
                    for i in range(scores.shape[0])
                },
                "n_features": step_n_features_rev,
            }
        return self

    def score(self, X, y, metric=None, **score_params):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        metric : str, default=None
            Metric to use for scoring when multiple metrics were used during
            fit. If None and only one metric was used during fit, that metric
            will be used. If None and multiple metrics were used during fit,
            an error will be raised.

        **score_params : dict
            Parameters to pass to the `score` method of the underlying scorer.

            .. versionadded:: 1.6
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>`
                for more details.

        Returns
        -------
        score : float
            Score of self.predict(X) w.r.t. y defined by `scoring`.

        Raises
        ------
        ValueError
            If multiple metrics were used during fit and `metric` is None.
        """
        _raise_for_params(score_params, self, "score")
        scorer = self._get_scorer()
        if _routing_enabled():
            routed_params = process_routing(self, "score", **score_params)
        else:
            routed_params = Bunch()
            routed_params.scorer = Bunch(score={})

        # Handle multiple scorers
        if hasattr(self, "multimetric_") and self.multimetric_:
            if metric is None:
                available_metrics = list(scorer.keys())
                raise ValueError(
                    "When multiple scoring metrics are used, the `metric` "
                    "parameter must be specified to select which metric to "
                    "use for scoring. Available metrics: %s" % available_metrics
                )
            if metric not in scorer:
                available_metrics = list(scorer.keys())
                raise ValueError(
                    "The metric '%s' was not among the scoring metrics used "
                    "during fit. Available metrics: %s" % (metric, available_metrics)
                )
            return scorer[metric](self, X, y, **routed_params.scorer.score)
        else:
            return scorer(self, X, y, **routed_params.scorer.score)

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.6

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)
        router.add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        router.add(
            splitter=check_cv(self.cv),
            method_mapping=MethodMapping().add(
                caller="fit",
                callee="split",
            ),
        )
        router.add(
            scorer=self._get_scorer(),
            method_mapping=MethodMapping()
            .add(caller="fit", callee="score")
            .add(caller="score", callee="score"),
        )

        return router

    def _get_scorer(self):
        """Get the scorer(s) to be used.

        Returns
        -------
        scorers : callable or dict of callables
            The scorer(s) to use. If multiple scorers are specified,
            a dict mapping scorer name to the scorer callable is returned.
        """
        if self.scoring is None:
            scoring = "accuracy" if is_classifier(self.estimator) else "r2"
            return get_scorer(scoring)
        elif isinstance(self.scoring, str):
            return get_scorer(self.scoring)
        elif callable(self.scoring):
            return self.scoring
        else:
            # Dictionary or list/tuple of strings/callables
            from ..metrics._scorer import _check_multimetric_scoring

            return _check_multimetric_scoring(self.estimator, self.scoring)
