# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Recursive feature elimination for feature ranking"""

import numbers
import warnings
from copy import deepcopy

import numpy as np
from joblib import effective_n_jobs
from scipy.stats import rankdata

from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..metrics import get_scorer
from ..model_selection._validation import _score, check_cv
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
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    fit_params = _check_method_params(
        X, params=routed_params.estimator.fit, indices=train
    )
    score_params = _check_method_params(
        X=X, params=routed_params.scorer.score, indices=test
    )

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
            Interval(numbers.Integral, 0, None, closed="neither"),
        ],
        "step": [
            Interval(numbers.Integral, 0, None, closed="neither"),
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
        elif isinstance(self.n_features_to_select, numbers.Integral):  # int
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
        """Return the score of the estimator after it has been fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        **score_params : dict
            Parameters to pass to the `score` method of the underlying estimator.

            .. versionadded:: 1.6
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>`
                for more details.

        Returns
        -------
        score : float
            Score of self.estimator_ (on the transformed training data if
            feature selection has been performed).
        """
        check_is_fitted(self)

        # Don't use _raise_for_params to allow backward compatibility
        # with tests that expect parameters to be passed through
        if _routing_enabled():
            routed_params = process_routing(self, "score", **score_params)
            return self.estimator_.score(
                self.transform(X), y, **routed_params.estimator.score
            )
        else:
            # For backward compatibility
            return self.estimator_.score(self.transform(X), y, **score_params)

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
        or ``feature_importances_`` attribute.

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

    scoring : str, callable, list/tuple or dict, default=None
        Strategy to evaluate the performance of the models.

        - If scoring represents a single score, one can use:

          - a single string (see :ref:`scoring_string_names`),
          - a callable (see :ref:`scoring_callable`) with signature
            ``scorer(estimator, X, y)``
          - None, in which case the `estimator`'s
            :ref:`default evaluation criterion <scoring_api_overview>` is used.

        - If scoring represents multiple metrics, one can use:

          - a list or tuple of unique strings or callables
            (e.g. ``('precision', 'recall')``),
          - a dict mapping the scorer name to a predefined or custom scoring function.

        See :ref:`scoring` for details.

        .. versionchanged:: 1.7
           Support for multiple metrics has been added.

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

    refit : str, bool or callable, default=True
        Refit an estimator using the selected features on the whole dataset.

        - If a string or a callable, the best parameter must be in ``scoring``.
          For a dict or a list/tuple of scoring metrics, the scorer name must be in
          ``scoring.keys()``, or be one of the elements of the list/tuple.
        - If True, use the scoring parameter to find the best number of features.

        The refitted estimator is made available at the ``estimator_`` attribute
        and permits using ``score`` method or ``predict`` method on new data.

        See ``refit`` parameter of
        :class:`~sklearn.model_selection.GridSearchCV` for more details.

        .. versionadded:: 1.7

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

        split(k)_test_<scorer_name> : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores for each scorer across (k)th fold.
            Only available when ``scoring`` represents multiple metrics.

        mean_test_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.

        mean_test_<scorer_name> : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds for each scorer.
            Only available when ``scoring`` represents multiple metrics.

        std_test_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.

        std_test_<scorer_name> : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds for each scorer.
            Only available when ``scoring`` represents multiple metrics.

        rank_test_<scorer_name> : ndarray of shape (n_subsets_of_features,)
            The ranking of models based on their test score for each scorer.
            Only available when ``scoring`` represents multiple metrics.

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

    Example with multiple scoring metrics:

    >>> selector = RFECV(
    ...     estimator=SVR(kernel="linear"),
    ...     step=1,
    ...     cv=5,
    ...     scoring=['r2', 'neg_mean_squared_error'],
    ...     refit='r2'
    ... )
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    """

    _parameter_constraints: dict = {
        **RFE._parameter_constraints,
        "min_features_to_select": [
            Interval(numbers.Integral, 0, None, closed="neither")
        ],
        "cv": ["cv_object"],
        "scoring": [None, str, callable, list, tuple, dict],
        "n_jobs": [None, numbers.Integral],
        "refit": [bool, str, callable],
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
        refit=True,
    ):
        self.estimator = estimator
        self.step = step
        self.importance_getter = importance_getter
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_features_to_select = min_features_to_select
        self.refit = refit

    def _get_scorer(self, base_scorer=None):
        """Return a scorer from parameter scoring."""
        if base_scorer is not None:
            # Using a predefined scorer
            return get_scorer(base_scorer)

        if self.scoring is None:
            # If no scoring provided, use default based on estimator type
            scoring = "accuracy" if is_classifier(self.estimator) else "r2"
            return get_scorer(scoring)

        if isinstance(self.scoring, (list, tuple)):
            # Convert list of scorer names to dict of scorers
            scorers = {str(scorer): get_scorer(scorer) for scorer in self.scoring}
            return scorers

        if isinstance(self.scoring, dict):
            # User provided dict of scorer names mapped to scorer callables
            scorers = {key: get_scorer(value) for key, value in self.scoring.items()}
            return scorers

        # scoring is a string or callable
        return get_scorer(self.scoring)

    def _check_refit_param(self, scorers):
        """Check if refit parameter is correctly set for multiple metrics.

        Parameters
        ----------
        scorers : dict, list, tuple, or callable
            The scorers to use.

        Returns
        -------
        refit_metric : str or callable
            The metric to use for refitting.
        """
        if isinstance(self.refit, str):
            if not isinstance(scorers, dict) and not isinstance(scorers, (list, tuple)):
                raise ValueError(
                    f"refit={self.refit!r} but scoring is not a dict or a list. "
                    "If only a single metric is used, set refit=True to use it for "
                    "refitting."
                )

            if isinstance(scorers, dict) and self.refit not in scorers:
                raise ValueError(
                    f"refit={self.refit!r} is not a valid scorer key. "
                    f"Valid options are: {list(scorers.keys())}"
                )

            if isinstance(scorers, (list, tuple)) and self.refit not in scorers:
                raise ValueError(
                    f"refit={self.refit!r} is not a valid scorer key. "
                    f"Valid options are: {scorers}"
                )

            return self.refit

        # If refit is True and scorers is a dict, return the first key
        # as the metric to optimize
        if self.refit is True and isinstance(scorers, dict):
            return list(scorers.keys())[0]

        # If refit is True and scorers is a list/tuple, return the first element
        if self.refit is True and isinstance(scorers, (list, tuple)):
            return scorers[0]

        return self.refit  # callable

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
        scorers = self._get_scorer()

        # Determine refit strategy
        is_multi_metric = isinstance(scorers, dict)
        refit_metric = None

        if isinstance(self.refit, str) and not is_multi_metric:
            raise ValueError(
                f"refit={self.refit!r} but scoring is not a dict or a list. "
                "If only a single metric is used, set refit=True to use it for "
                "refitting."
            )

        if is_multi_metric:
            refit_metric = self._check_refit_param(scorers)
            if callable(refit_metric):
                # We'll compute all metrics but won't use a specific one for refit
                select_metric = list(scorers.values())[0]
            else:
                # Use the specified metric for refit
                select_metric = scorers[refit_metric]
        else:
            # For single metric, use it for feature selection
            select_metric = scorers

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
            func(
                clone(rfe),
                self.estimator,
                X,
                y,
                train,
                test,
                select_metric,
                routed_params,
            )
            for train, test in cv.split(X, y, **routed_params.splitter.split)
        )
        scores, step_n_features = zip(*scores_features)

        step_n_features_rev = np.array(step_n_features[0])[::-1]
        scores = np.array(scores)

        # Process multiple metrics if needed
        if is_multi_metric:
            all_scores = {}
            # For each fold, compute scores for each metric
            for metric_name, metric_scorer in scorers.items():
                all_fold_scores = parallel(
                    func(
                        clone(rfe),
                        self.estimator,
                        X,
                        y,
                        train,
                        test,
                        metric_scorer,
                        routed_params,
                    )
                    for train, test in cv.split(X, y, **routed_params.splitter.split)
                )
                fold_scores, _ = zip(*all_fold_scores)
                all_scores[metric_name] = np.array(fold_scores)

        # Create cv_results_ dictionary before calculating best index
        self.cv_results_ = {"n_features": step_n_features_rev}

        # For multiple metrics, add all metrics to cv_results_
        if is_multi_metric:
            for metric_name, metric_scores in all_scores.items():
                # reverse to stay consistent with before
                metric_scores_rev = metric_scores[:, ::-1]
                mean_scores = np.mean(metric_scores_rev, axis=0)
                self.cv_results_.update(
                    {
                        f"mean_test_{metric_name}": mean_scores,
                        f"std_test_{metric_name}": np.std(metric_scores_rev, axis=0),
                        **{
                            f"split{i}_test_{metric_name}": metric_scores_rev[i]
                            for i in range(metric_scores.shape[0])
                        },
                    }
                )

                # Add ranking of feature counts for each metric
                ranking = np.asarray(
                    rankdata(-mean_scores, method="min"), dtype=np.int32
                )
                self.cv_results_[f"rank_test_{metric_name}"] = ranking

            # Now we have cv_results_, use it to determine the number of
            # features to select
            if callable(refit_metric):
                best_index = self._select_best_index(
                    refit_metric, None, self.cv_results_
                )
                n_features_to_select = step_n_features_rev[best_index]
            elif refit_metric is not None:
                # Use the specified string metric
                best_index = np.argmax(self.cv_results_[f"mean_test_{refit_metric}"])
                n_features_to_select = step_n_features_rev[best_index]
                # Set mean_test_score to the refit metric for backward compatibility
                self.cv_results_["mean_test_score"] = self.cv_results_[
                    f"mean_test_{refit_metric}"
                ]
                self.cv_results_["std_test_score"] = self.cv_results_[
                    f"std_test_{refit_metric}"
                ]
            else:
                # Use first metric
                first_metric = list(scorers.keys())[0]
                best_index = np.argmax(self.cv_results_[f"mean_test_{first_metric}"])
                n_features_to_select = step_n_features_rev[best_index]
                # Set mean_test_score to the first metric for backward compatibility
                self.cv_results_["mean_test_score"] = self.cv_results_[
                    f"mean_test_{first_metric}"
                ]
                self.cv_results_["std_test_score"] = self.cv_results_[
                    f"std_test_{first_metric}"
                ]
        else:
            # For single metric, use it for feature selection
            scores_sum_rev = np.sum(scores, axis=0)[::-1]
            best_index = np.argmax(scores_sum_rev)
            n_features_to_select = step_n_features_rev[best_index]

            # Create cv_results_ dictionary
            scores_rev = scores[:, ::-1]
            self.cv_results_ = {
                "n_features": step_n_features_rev,
                "mean_test_score": np.mean(scores_rev, axis=0),
                "std_test_score": np.std(scores_rev, axis=0),
                **{
                    f"split{i}_test_score": scores_rev[i]
                    for i in range(scores.shape[0])
                },
            }

        # Refit on the whole dataset with the selected number of features
        self.estimator_ = clone(self.estimator)

        # Get the optimal number of features to select
        features_to_select = n_features_to_select
        if features_to_select is None:
            features_to_select = self.min_features_to_select

        # Re-fit the RFE object using the whole dataset to get appropriate
        # feature_importances_
        final_rfe = RFE(
            self.estimator_,
            n_features_to_select=features_to_select,
            step=self.step,
            importance_getter=self.importance_getter,
            verbose=self.verbose,
        )
        final_rfe.fit(X, y, **routed_params.estimator.fit)
        self.estimator_ = final_rfe.estimator_
        self.n_features_ = features_to_select
        self.support_ = final_rfe.support_
        self.ranking_ = final_rfe.ranking_

        # Set n_features_in_ attribute
        if hasattr(final_rfe, "n_features_in_"):
            self.n_features_in_ = final_rfe.n_features_in_
        if hasattr(final_rfe, "feature_names_in_"):
            self.feature_names_in_ = final_rfe.feature_names_in_

        return self

    def _select_best_index(self, refit, refit_metric, results):
        """Select index of the best combination of features.

        Parameters
        ----------
        refit : callable
            A callable that takes the cv_results_ dict and returns the index
            of the best estimator.

        refit_metric : str or None
            Not used in this implementation, kept for API compatibility.

        results : dict
            The cv_results_ dict with all the cross-validation results.

        Returns
        -------
        best_index : int
            The index with the best cross-validation result.
        """
        if callable(refit):
            # If callable, refit is expected to return the index of the best
            # parameter set.
            best_index = refit(results)
            if not isinstance(best_index, numbers.Integral):
                raise TypeError("best_index_ returned is not an integer")

            # Check if the index is out of bounds
            if best_index < 0 or best_index >= len(results["n_features"]):
                raise IndexError("best_index_ index out of range")

        return best_index

    @available_if(_estimator_has("score"))
    def score(self, X, y, **score_params):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        **score_params : dict
            Parameters to pass to the `score` method of the underlying estimator.

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
        """
        check_is_fitted(self)

        # Transform the input features
        X_transformed = self.transform(X)

        # Don't use _raise_for_params to allow backward compatibility
        # with tests that expect parameters to be passed through
        if _routing_enabled():
            routed_params = process_routing(self, "score", **score_params)
            estimator_score_params = routed_params.estimator.score
            scorer_score_params = routed_params.scorer.score
        else:
            # For backward compatibility
            estimator_score_params = score_params
            scorer_score_params = {}

        # Get the appropriate scorer
        if self.scoring is None:
            # If scoring is None, use the estimator's default score method
            return self.estimator_.score(X_transformed, y, **estimator_score_params)
        elif isinstance(self.scoring, (list, tuple, dict)):
            if isinstance(self.refit, str):
                # Use the metric specified by refit for scoring
                if isinstance(self.scoring, dict):
                    scoring = get_scorer(self.scoring[self.refit])
                else:
                    scoring_values = [
                        get_scorer(s) if isinstance(s, str) else s for s in self.scoring
                    ]
                    scoring_idx = list(self.scoring).index(self.refit)
                    scoring = scoring_values[scoring_idx]
            else:
                # Use the first metric for scoring
                if isinstance(self.scoring, dict):
                    scoring = get_scorer(list(self.scoring.values())[0])
                else:
                    scoring = get_scorer(self.scoring[0])
        else:
            scoring = get_scorer(self.scoring)

        return scoring(self, X, y, **scorer_score_params)

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
            method_mapping=MethodMapping()
            .add(caller="fit", callee="split")
            .add(caller="score", callee="split"),
        )

        # Handle multiple scorers if needed
        if isinstance(self.scoring, (list, tuple, dict)):
            scorers = self._get_scorer()
            for name, scorer in scorers.items():
                router.add(
                    scorer=scorer,
                    method_mapping=MethodMapping()
                    .add(caller="fit", callee="score")
                    .add(caller="score", callee="score"),
                )
        else:
            router.add(
                scorer=self._get_scorer(),
                method_mapping=MethodMapping()
                .add(caller="fit", callee="score")
                .add(caller="score", callee="score"),
            )

        return router
