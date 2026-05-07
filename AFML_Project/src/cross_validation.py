import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import get_scorer
from sklearn.base import clone

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

def cv_score(clf, X, y, sample_weight, scoring, cv, t1=None):
    """
    For each fold: fit clf on train with sample_weight, score on test.
    """
    if t1 is None and hasattr(cv, 't1'):
        t1 = cv.t1
        
    scorer = get_scorer(scoring)
    scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        # Slice data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Slice weights if available
        if sample_weight is not None:
            sw_train = sample_weight.iloc[train_idx]
        else:
            sw_train = None
            
        # Clone classifier to avoid fitting the same instance
        clf_clone = clone(clf)
        
        # Fit
        if sw_train is not None:
            clf_clone.fit(X_train, y_train, sample_weight=sw_train.values)
        else:
            clf_clone.fit(X_train, y_train)
            
        # Score
        # For sklearn < 1.4, scorer takes (estimator, X, y_true)
        score = scorer(clf_clone, X_test, y_test)
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
