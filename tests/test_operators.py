"""Unit tests for src/alphas/operators.py."""
import numpy as np
import pandas as pd
import pytest

from src.alphas.operators import (
    adv, decay_linear, delay, delta, rank_cs, scale_cs,
    signed_power, ts_argmax, ts_argmin, ts_corr, ts_max,
    ts_mean, ts_min, ts_product, ts_rank, ts_std, ts_sum,
)


def _series(values):
    return pd.Series(values, dtype=float)


def _wide(data):
    return pd.DataFrame(data, dtype=float)


class TestDeltaDelay:
    def test_delta_basic(self):
        s = _series([10, 11, 13, 16])
        result = delta(s, 1).dropna().tolist()
        assert result == pytest.approx([1, 2, 3])

    def test_delta_period2(self):
        s = _series([10, 11, 13, 16])
        result = delta(s, 2).dropna().tolist()
        assert result == pytest.approx([3, 5])

    def test_delay_basic(self):
        s = _series([10, 11, 13, 16])
        result = delay(s, 2).tolist()
        assert np.isnan(result[0]) and np.isnan(result[1])
        assert result[2:] == pytest.approx([10, 11])


class TestRollingOps:
    def test_ts_sum(self):
        s = _series([1, 2, 3, 4, 5])
        result = ts_sum(s, 3).dropna().tolist()
        assert result == pytest.approx([6, 9, 12])

    def test_ts_mean(self):
        s = _series([1, 2, 3, 4, 5])
        result = ts_mean(s, 3).dropna().tolist()
        assert result == pytest.approx([2, 3, 4])

    def test_ts_min(self):
        s = _series([3, 1, 4, 1, 5])
        result = ts_min(s, 3).dropna().tolist()
        assert result == pytest.approx([1, 1, 1])

    def test_ts_max(self):
        s = _series([3, 1, 4, 1, 5])
        result = ts_max(s, 3).dropna().tolist()
        assert result == pytest.approx([4, 4, 5])

    def test_ts_rank_max(self):
        s = _series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_rank(s, 5).dropna()
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_ts_rank_min(self):
        s = _series([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ts_rank(s, 5).dropna()
        assert float(result.iloc[-1]) == pytest.approx(0.0, abs=0.25)

    def test_ts_argmax(self):
        s = _series([1, 5, 3, 2, 4])
        result = ts_argmax(s, 5).dropna()
        # argmax of [1,5,3,2,4] = index 1 (0-based)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_ts_argmin(self):
        s = _series([5, 1, 3, 2, 4])
        result = ts_argmin(s, 5).dropna()
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_ts_std_positive(self):
        s = _series([1, 2, 3, 4, 5, 6, 7])
        result = ts_std(s, 3).dropna()
        assert (result > 0).all()

    def test_ts_product_ones(self):
        s = _series([1.0, 1.0, 1.0, 1.0, 1.0])
        result = ts_product(s, 3).dropna()
        assert result.tolist() == pytest.approx([1.0] * len(result))

    def test_ts_corr_perfect(self):
        s = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = ts_corr(s, s, 5).dropna()
        assert result.tolist() == pytest.approx([1.0] * len(result))

    def test_min_periods_enforced(self):
        s = _series([1, 2, 3, 4, 5])
        result = ts_sum(s, 3)
        assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])


class TestCrossSectionalOps:
    def test_rank_cs_ordering(self):
        df = _wide({'A': [1, 2, 3], 'B': [3, 2, 1]})
        r = rank_cs(df)
        # Row 0: A=1 (min) < B=3 (max) -> A rank < B rank
        assert r.iloc[0, 0] < r.iloc[0, 1]
        # Row 2: A=3 > B=1 -> A rank > B rank
        assert r.iloc[2, 0] > r.iloc[2, 1]

    def test_rank_cs_range(self):
        df = _wide({'A': [1, 2], 'B': [2, 3], 'C': [3, 1]})
        r = rank_cs(df)
        assert (r >= 0).all().all() and (r <= 1).all().all()

    def test_scale_cs_unit_sum(self):
        df = _wide({'A': [1.0, -2.0], 'B': [3.0, 4.0]})
        s = scale_cs(df, a=1.0)
        row_abs_sums = s.abs().sum(axis=1)
        assert row_abs_sums.tolist() == pytest.approx([1.0, 1.0])


class TestAdvAndUtils:
    def test_adv_dollar_volume(self):
        close  = _wide({'A': [100.0, 100.0, 100.0], 'B': [200.0, 200.0, 200.0]})
        volume = _wide({'A': [1.0, 1.0, 1.0],       'B': [1.0, 1.0, 1.0]})
        result = adv(close, volume, 3)
        assert float(result.iloc[-1, 0]) == pytest.approx(100.0)
        assert float(result.iloc[-1, 1]) == pytest.approx(200.0)

    def test_decay_linear_constant(self):
        s = _series([5.0, 5.0, 5.0, 5.0, 5.0])
        result = decay_linear(s, 3).dropna()
        assert result.tolist() == pytest.approx([5.0] * len(result))

    def test_decay_linear_weights_increasing(self):
        # Most recent observation should have highest weight
        s = _series([0.0, 0.0, 1.0])
        result = decay_linear(s, 3).dropna()
        # weights = [1/6, 2/6, 3/6] -> result = 3/6 = 0.5
        assert float(result.iloc[-1]) == pytest.approx(0.5)

    def test_signed_power_positive(self):
        assert signed_power(pd.Series([8.0]), 1/3).iloc[0] == pytest.approx(2.0)

    def test_signed_power_negative(self):
        assert signed_power(pd.Series([-8.0]), 1/3).iloc[0] == pytest.approx(-2.0)
