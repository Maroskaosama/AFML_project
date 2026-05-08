import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    log_loss,
)
from sklearn.base import clone


def weighted_score(estimator, X, y, sample_weight=None,
                   scoring="accuracy", labels=None):
    """
    Test-fold scoring that honours `sample_weight` for the metrics that
    accept it.

    Supported names
    ---------------
    'accuracy'      : accuracy_score(y, estimator.predict(X), sample_weight=...)
    'neg_log_loss'  : -log_loss(y, estimator.predict_proba(X),
                                sample_weight=..., labels=labels or estimator.classes_)
    'f1'            : f1_score(y, estimator.predict(X),
                               sample_weight=..., average='binary')

    Anything else falls back to `sklearn.metrics.get_scorer(scoring)` and
    the call IS NOT WEIGHTED — a UserWarning is emitted so the user knows.

    `sample_weight=None` reproduces the unweighted reference value, which
    is what we use for the "score_with_weights=False" arm of the
    four-way validation.
    """
    if scoring == "accuracy":
        y_pred = estimator.predict(X)
        return float(accuracy_score(y, y_pred, sample_weight=sample_weight))

    if scoring == "neg_log_loss":
        y_proba = estimator.predict_proba(X)
        cls = labels if labels is not None else getattr(estimator, "classes_", None)
        return -float(log_loss(y, y_proba, sample_weight=sample_weight, labels=cls))

    if scoring == "f1":
        y_pred = estimator.predict(X)
        return float(f1_score(y, y_pred, sample_weight=sample_weight,
                              average="binary"))

    warnings.warn(
        f"weighted_score: scoring={scoring!r} has no weighted path; "
        "falling back to sklearn get_scorer (sample_weight is ignored).",
        UserWarning,
        stacklevel=2,
    )
    return float(get_scorer(scoring)(estimator, X, y))

class PurgedKFold(BaseCrossValidator):
    """
    Purged and Embargoed K-Fold Cross Validation.
    Removes overlap between train and test periods (Purge)
    and removes a portion of data immediately following the test period (Embargo).
    """
    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        if not isinstance(t1, pd.Series):
            raise ValueError('t1 must be a pandas Series')
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
        
    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and t1 must have the same index')
            
        indices = np.arange(X.shape[0])
        embargo_step = int(X.shape[0] * self.pct_embargo)
        
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]
        
        for start_idx, end_idx in test_starts:
            # Test indices
            test_indices = indices[start_idx:end_idx]
            
            # Test bounds
            test_times = self.t1.index[test_indices]
            test_start_time = test_times.min()
            test_end_time = test_times.max()
            
            # 1. Purge: remove any train index that started before test but ends inside or after test starts
            # condition: t1[idx] > test_start and idx < test_start
            t1_before_test = self.t1.iloc[:start_idx]
            purged = t1_before_test[t1_before_test >= test_start_time].index
            
            # Get train indices before test
            train_indices_before = pd.Series(indices[:start_idx], index=self.t1.index[:start_idx])
            train_indices_before = train_indices_before.drop(purged).values
            
            # 2. Embargo: remove indices after test that fall within embargo period
            # Or simply remove step sizes
            embargo_end_idx = min(end_idx + embargo_step, X.shape[0])
            train_indices_after = indices[embargo_end_idx:]
            
            train_indices = np.concatenate([train_indices_before, train_indices_after])
            
            yield train_indices, test_indices

def cv_score(clf, X, y, sample_weight, scoring, cv, t1=None,
             fit_with_weights=True, score_with_weights=True):
    """
    For each fold: fit clf on train and score on test, both honouring
    `sample_weight` by default.

    Parameters
    ----------
    clf, X, y, sample_weight, scoring, cv, t1
        Same positional contract as before, so existing call sites
        (modelling.py, hyperparameter_tuning.py, feature_importance.py)
        keep working untouched.
    fit_with_weights : bool, default True
        If False, `sample_weight` is NOT passed to `.fit()`. Useful for
        validation to isolate the effect of weighted scoring vs weighted
        fitting.
    score_with_weights : bool, default True
        If False, the test fold is scored with `sample_weight=None`,
        which reproduces the legacy (unweighted) behaviour.

    Notes
    -----
    `sample_weight=None` makes both flags no-ops. With weights, the
    default behaviour now matches what AFML Ch 4 / 7 expect: the same
    sample weights that drove fitting also drive evaluation.
    """
    if t1 is None and hasattr(cv, "t1"):
        t1 = cv.t1

    scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if sample_weight is not None:
            sw_train = sample_weight.iloc[train_idx].values
            sw_test = sample_weight.iloc[test_idx].values
        else:
            sw_train = None
            sw_test = None

        clf_clone = clone(clf)
        if sw_train is not None and fit_with_weights:
            clf_clone.fit(X_train, y_train, sample_weight=sw_train)
        else:
            clf_clone.fit(X_train, y_train)

        sw_for_score = sw_test if (sw_test is not None and score_with_weights) else None
        score = weighted_score(
            clf_clone, X_test, y_test,
            sample_weight=sw_for_score,
            scoring=scoring,
        )
        scores.append(score)

    return pd.Series(scores)

class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged K-Fold Cross Validation (CPCV).
    Generates paths of splits. Not fully implemented here as the prompt
    primarily asked for PurgedKFold and cv_score, but scaffolded for extension.
    """
    def __init__(self, n_splits=6, n_test_splits=2, t1=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
    def get_n_splits(self, X=None, y=None, groups=None):
        import scipy.special
        return int(scipy.special.comb(self.n_splits, self.n_test_splits))
        
    def split(self, X, y=None, groups=None):
        # Full implementation would use itertools.combinations
        # to generate combinatorial test sets, and apply purge/embargo
        # identical to PurgedKFold.
        pass
