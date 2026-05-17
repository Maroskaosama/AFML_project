"""Unit tests for PurgedKFold and MultiAssetPurgedKFold in src/cross_validation.py."""
import numpy as np
import pandas as pd
import pytest

from src.cross_validation import MultiAssetPurgedKFold, PurgedKFold


def _make_events(n=30, freq='B', start='2020-01-02'):
    idx = pd.date_range(start, periods=n, freq=freq)
    return idx


def _make_t1(idx, horizon=5):
    """t1 = event_date + horizon business days (capped at last date)."""
    t1_dates = []
    bdays = idx.tolist()
    for i, d in enumerate(bdays):
        t1_dates.append(bdays[min(i + horizon, len(bdays) - 1)])
    return pd.Series(t1_dates, index=idx)


class TestPurgedKFold:
    def _make_data(self, n=50):
        idx = _make_events(n)
        t1 = _make_t1(idx, horizon=3)
        X = pd.DataFrame({'f': np.arange(n, dtype=float)}, index=idx)
        y = pd.Series(np.ones(n), index=idx)
        return X, y, t1

    def test_n_splits(self):
        X, y, t1 = self._make_data()
        cv = PurgedKFold(n_splits=5, t1=t1)
        assert cv.get_n_splits() == 5

    def test_yields_correct_split_count(self):
        X, y, t1 = self._make_data()
        cv = PurgedKFold(n_splits=5, t1=t1)
        splits = list(cv.split(X, y))
        assert len(splits) == 5

    def test_test_indices_cover_all(self):
        X, y, t1 = self._make_data()
        cv = PurgedKFold(n_splits=5, t1=t1)
        covered = set()
        for _, test in cv.split(X, y):
            covered.update(test.tolist())
        assert covered == set(range(len(X)))

    def test_train_test_disjoint(self):
        X, y, t1 = self._make_data()
        cv = PurgedKFold(n_splits=5, t1=t1)
        for train, test in cv.split(X, y):
            assert len(set(train) & set(test)) == 0

    def test_no_temporal_leakage(self):
        """No train sample starting after test-fold start should appear."""
        X, y, t1 = self._make_data(n=50)
        cv = PurgedKFold(n_splits=5, t1=t1)
        for train, test in cv.split(X, y):
            test_start = X.index[test.min()]
            train_times = X.index[train]
            # Training times must all be strictly before test_start OR after test_end
            test_end = X.index[test.max()]
            after_mask = train_times > test_end
            before_mask = train_times < test_start
            assert (before_mask | after_mask).all(), \
                "Train set contains times overlapping the test fold"

    def test_requires_series_t1(self):
        idx = _make_events(20)
        X = pd.DataFrame({'f': np.zeros(20)}, index=idx)
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=3, t1=np.zeros(20))


class TestMultiAssetPurgedKFold:
    def _make_pooled(self, n_dates=20, n_tickers=3, horizon=3):
        """Build a small pooled DataFrame with MultiIndex-style (date × ticker)."""
        dates = pd.date_range('2020-01-02', periods=n_dates, freq='B')
        tickers = [f'T{i}' for i in range(n_tickers)]
        idx = pd.DatetimeIndex(
            [d for d in dates for _ in tickers]
        )
        data = pd.DataFrame({'f': np.arange(len(idx), dtype=float)}, index=idx)
        data['ticker'] = [t for _ in dates for t in tickers]

        # t1: each event exits horizon calendar days later (or last date)
        t1_list = []
        for d in idx:
            future = d + pd.Timedelta(days=horizon * 1)
            t1_list.append(min(future, dates[-1]))
        t1 = pd.Series(t1_list, index=idx)

        y = pd.Series(np.ones(len(idx)), index=idx)
        return data, y, t1

    def test_n_splits(self):
        X, y, t1 = self._make_pooled()
        cv = MultiAssetPurgedKFold(n_splits=4, t1=t1)
        assert cv.get_n_splits() == 4

    def test_yields_correct_split_count(self):
        X, y, t1 = self._make_pooled()
        cv = MultiAssetPurgedKFold(n_splits=4, t1=t1)
        splits = list(cv.split(X, y))
        assert len(splits) == 4

    def test_cross_sectional_integrity(self):
        """All stocks at the same event date must go to the same fold."""
        X, y, t1 = self._make_pooled(n_dates=20, n_tickers=3)
        cv = MultiAssetPurgedKFold(n_splits=4, t1=t1)
        for _, test in cv.split(X, y):
            test_dates = set(X.index[test])
            # Every row at any of those dates must be in test
            for i, d in enumerate(X.index):
                if d in test_dates:
                    assert i in test, \
                        f"Date {d} is in test_dates but row {i} not in test set"

    def test_train_test_disjoint(self):
        X, y, t1 = self._make_pooled()
        cv = MultiAssetPurgedKFold(n_splits=4, t1=t1)
        for train, test in cv.split(X, y):
            assert len(set(train) & set(test)) == 0

    def test_purging_removes_overlap(self):
        """After purging, no train sample should have t1 >= test_start."""
        X, y, t1 = self._make_pooled(n_dates=30, n_tickers=2, horizon=5)
        cv = MultiAssetPurgedKFold(n_splits=5, t1=t1)
        for train, test in cv.split(X, y):
            if len(train) == 0 or len(test) == 0:
                continue
            test_start = X.index[test].min()
            train_before = [i for i in train if X.index[i] < test_start]
            for i in train_before:
                assert pd.isna(t1.iloc[i]) or t1.iloc[i] < test_start, \
                    f"Train sample {i} (t1={t1.iloc[i]}) leaks into test_start={test_start}"

    def test_embargo_removes_post_test_samples(self):
        """After embargo, no train sample immediately after test should remain."""
        X, y, t1 = self._make_pooled(n_dates=40, n_tickers=2, horizon=2)
        pct_embargo = 0.05
        cv = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=pct_embargo)
        unique_times = sorted(set(X.index))
        n_times = len(unique_times)
        embargo_n = max(1, int(n_times * pct_embargo))

        for train, test in cv.split(X, y):
            if len(test) == 0:
                continue
            test_end = X.index[test].max()
            te_idx = unique_times.index(test_end)
            cutoff_idx = min(te_idx + embargo_n, n_times - 1)
            embargo_cutoff = unique_times[cutoff_idx]

            for i in train:
                et = X.index[i]
                assert not (test_end < et <= embargo_cutoff), \
                    f"Train sample at {et} is within embargo window [{test_end},{embargo_cutoff}]"
