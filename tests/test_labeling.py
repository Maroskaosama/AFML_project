"""Unit tests for src/labeling.py."""
import numpy as np
import pandas as pd
import pytest

from src.labeling import apply_triple_barrier, get_bins, get_daily_vol


def _make_close(prices, start='2020-01-01', freq='B'):
    idx = pd.date_range(start, periods=len(prices), freq=freq)
    return pd.Series(prices, index=idx, dtype=float)


class TestGetDailyVol:
    def test_returns_same_length(self):
        close = _make_close([100 + i for i in range(60)])
        vol = get_daily_vol(close, span=10)
        assert len(vol) == len(close)

    def test_all_positive_after_warmup(self):
        close = _make_close([100 * (1.01 ** i) for i in range(60)])
        vol = get_daily_vol(close, span=10)
        # EWMA std is 0 for the first observation (single-point window) then grows
        assert (vol.dropna() >= 0).all()
        assert (vol.dropna().iloc[5:] > 0).all()

    def test_flat_close_zero_vol(self):
        close = _make_close([100.0] * 60)
        vol = get_daily_vol(close, span=10)
        assert (vol.dropna().abs() < 1e-10).all()


class TestApplyTripleBarrier:
    def _events(self, close, t1_offset=5, trgt_val=0.05):
        """Build a minimal events DataFrame for triple-barrier."""
        idx = close.index[:3]
        t1_idx = [min(i + t1_offset, len(close) - 1) for i in range(3)]
        t1_dates = close.index[t1_idx]
        events = pd.DataFrame(
            {'t1': t1_dates, 'trgt': trgt_val},
            index=idx,
        )
        return events

    def test_output_columns(self):
        close = _make_close([100.0] * 20)
        events = self._events(close)
        out = apply_triple_barrier(close, events, pt_sl=[1.0, 1.0])
        assert set(['t1', 'pt', 'sl']).issubset(out.columns)

    def test_profit_take_triggered(self):
        """Create a scenario where price always rises — pt must fire."""
        prices = [100.0 * (1.10 ** i) for i in range(20)]  # +10% each day
        close = _make_close(prices)
        events = self._events(close, t1_offset=10, trgt_val=0.05)
        out = apply_triple_barrier(close, events, pt_sl=[1.0, 1.0])
        # With 10% daily gains and 5% target, pt should fire for all events
        assert out['pt'].notna().any()

    def test_stop_loss_triggered(self):
        """Price drops sharply — sl must fire."""
        prices = [100.0 * (0.90 ** i) for i in range(20)]
        close = _make_close(prices)
        events = self._events(close, t1_offset=10, trgt_val=0.05)
        out = apply_triple_barrier(close, events, pt_sl=[1.0, 1.0])
        assert out['sl'].notna().any()

    def test_vertical_barrier_as_fallback(self):
        """Flat price — no pt/sl fires; t1 remains the vertical barrier."""
        close = _make_close([100.0] * 30)
        idx = close.index[:3]
        t1_dates = close.index[[5, 10, 15]]
        events = pd.DataFrame({'t1': t1_dates, 'trgt': 0.05}, index=idx)
        out = apply_triple_barrier(close, events, pt_sl=[1.0, 1.0])
        # pt and sl should be NaT for flat prices
        assert out['pt'].isna().all()
        assert out['sl'].isna().all()


class TestGetBins:
    def _full_events(self, close, t1_offset=5, trgt_val=0.05):
        idx = close.index[:5]
        t1_idx = [min(i + t1_offset, len(close) - 1) for i in range(5)]
        t1_dates = close.index[t1_idx]
        events = pd.DataFrame({'t1': t1_dates, 'trgt': trgt_val}, index=idx)
        barrier_out = apply_triple_barrier(close, events, pt_sl=[1.0, 1.0])
        # Merge t1 from barrier output back into events
        events_merged = events.copy()
        events_merged['t1'] = barrier_out['t1']
        events_merged['pt'] = barrier_out['pt']
        events_merged['sl'] = barrier_out['sl']
        return events_merged

    def test_labels_in_valid_set(self):
        prices = [100.0 * (1.02 ** i) for i in range(30)]
        close = _make_close(prices)
        events = self._full_events(close)
        bins = get_bins(events, close)
        valid = {-1.0, 0.0, 1.0}
        assert set(bins['bin'].dropna().unique()).issubset(valid)

    def test_upward_trend_positive_labels(self):
        prices = [100.0 * (1.05 ** i) for i in range(30)]
        close = _make_close(prices)
        events = self._full_events(close, t1_offset=5, trgt_val=0.02)
        bins = get_bins(events, close)
        non_nan = bins['bin'].dropna()
        assert (non_nan == 1.0).all()

    def test_downward_trend_negative_labels(self):
        prices = [100.0 * (0.95 ** i) for i in range(30)]
        close = _make_close(prices)
        events = self._full_events(close, t1_offset=5, trgt_val=0.02)
        bins = get_bins(events, close)
        non_nan = bins['bin'].dropna()
        assert (non_nan == -1.0).all()

    def test_t1_after_t0(self):
        prices = [100.0 + float(i) for i in range(30)]
        close = _make_close(prices)
        events = self._full_events(close)
        bins = get_bins(events, close)
        # exit time in bins must be >= event time
        for t0, row in bins.iterrows():
            if pd.notna(row['t1']):
                assert row['t1'] >= t0
