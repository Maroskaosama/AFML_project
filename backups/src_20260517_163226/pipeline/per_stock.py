"""
AFML Stages 0-3 for a single stock.
Stage 0: Load + clean
Stage 1: CUSUM filter
Stage 2: Triple-barrier labels + sample weights
Stage 3: Time-series features + fracdiff (optimal d*)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

from src.data_structures import cusum_filter, calibrate_cusum_h
from src.labeling import (
    get_daily_vol, add_vertical_barrier,
    apply_triple_barrier, get_bins,
)
from src.sample_weights import (
    num_co_events, sample_tw, get_return_attribution,
    get_time_decay, get_sample_weight,
)
from src.features import (
    compute_momentum_features, compute_volatility_features,
    compute_volume_features, compute_microstructure_features,
    compute_entropy_features,
)
from src.fracdiff import find_min_d, frac_diff_ffd


def run_per_stock_pipeline(
    ticker: str,
    raw_path: str,
    output_dir: str = 'data/processed/per_stock',
    target_events: int = 300,
    pt_sl: list = None,
    num_days: int = 10,
    d_range=None,
    corr_threshold: float = 0.85,
    verbose: bool = True,
) -> dict:
    """
    Run AFML Stages 0-3 for a single stock and save artifacts.

    Returns dict of DataFrames: {labels, weights, ts_features, clean}.
    """
    if pt_sl is None:
        pt_sl = [1.0, 1.0]
    if d_range is None:
        d_range = np.arange(0.05, 0.55, 0.05)

    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 0: Load & clean ──────────────────────────────────────────────
    raw = pd.read_csv(raw_path, parse_dates=['Date'], index_col='Date')
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
    raw = raw.dropna()
    close = raw['Adj Close']

    assert (close > 0).all(), f"{ticker}: non-positive prices detected"
    assert close.index.is_monotonic_increasing, f"{ticker}: dates not sorted"

    # ── Stage 1: CUSUM filter ──────────────────────────────────────────────
    h = calibrate_cusum_h(close, target_events=target_events)
    cusum_events = cusum_filter(close, h)

    # Safety: widen if < 100 events; narrow if > 600
    while len(cusum_events) < 100 and h > 1e-5:
        h *= 0.8
        cusum_events = cusum_filter(close, h)
    while len(cusum_events) > 600:
        h *= 1.2
        cusum_events = cusum_filter(close, h)

    if verbose:
        print(f"  {ticker}: h={h:.5f}, {len(cusum_events)} CUSUM events")

    # ── Stage 2: Triple-barrier labeling ──────────────────────────────────
    daily_vol = get_daily_vol(close, span=50)

    t1_series = add_vertical_barrier(close, cusum_events, num_days=num_days)
    trgt = daily_vol.reindex(cusum_events).ffill()

    events_df = pd.DataFrame({'t1': t1_series, 'trgt': trgt}).dropna()

    touch_df = apply_triple_barrier(close, events_df, pt_sl=pt_sl)

    merged = pd.concat([events_df, touch_df[['pt', 'sl']]], axis=1)

    labels = get_bins(merged, close)
    labels = labels.dropna(subset=['t1', 'bin'])

    # Drop 0-label rows (exact zero return — very rare)
    labels = labels[labels['bin'] != 0]

    if verbose:
        n_pos = (labels['bin'] == 1).sum()
        n_neg = (labels['bin'] == -1).sum()
        print(f"  {ticker}: {len(labels)} labels, +1={n_pos}, -1={n_neg}")

    # ── Sample weights ────────────────────────────────────────────────────
    events_for_w = pd.DataFrame({
        't1': labels['t1'],
        'ret': labels['ret'],
    }, index=labels.index)

    weights_series = get_sample_weight(events_for_w, close)
    weights = weights_series.reindex(labels.index).fillna(weights_series.mean())
    weights = weights.rename('weight').to_frame()

    # ── Stage 3: Time-series features ────────────────────────────────────
    mom_feat  = compute_momentum_features(close)
    vol_feat  = compute_volatility_features(close)
    vol_feats = compute_volume_features(raw)
    mic_feat  = compute_microstructure_features(raw)

    ret_series = np.log(close / close.shift(1))
    ent_feat = compute_entropy_features(ret_series)

    raw_feats = pd.concat([mom_feat, vol_feat, vol_feats, mic_feat, ent_feat], axis=1)

    # ── Fracdiff: find optimal d* ─────────────────────────────────────────
    log_close = np.log(close)
    frac_result = find_min_d(
        log_close,
        d_range=d_range,
        corr_threshold=corr_threshold,
    )
    d_star = frac_result['d_star']
    frac_series = frac_diff_ffd(log_close, d_star)
    frac_series.name = 'fracdiff'

    raw_feats['fracdiff'] = frac_series

    if verbose:
        n_frac = frac_series.notna().sum()
        print(f"  {ticker}: d*={d_star:.2f}, fracdiff has {n_frac} valid rows")

    # ── Align features to event timestamps ───────────────────────────────
    ts_features = raw_feats.reindex(labels.index)
    ts_features = ts_features.dropna(how='any')

    # Keep only labels/weights that have full features
    valid_idx = ts_features.index
    labels = labels.reindex(valid_idx)
    weights = weights.reindex(valid_idx)

    if verbose:
        print(f"  {ticker}: {len(ts_features)} rows with complete features")

    # ── Save artifacts ───────────────────────────────────────────────────
    # clean OHLCV
    clean_path   = os.path.join(output_dir, f'{ticker}_clean.parquet')
    labels_path  = os.path.join(output_dir, f'{ticker}_labels.parquet')
    weights_path = os.path.join(output_dir, f'{ticker}_weights.parquet')
    feat_path    = os.path.join(output_dir, f'{ticker}_ts_features.parquet')

    raw.to_parquet(clean_path)
    labels.to_parquet(labels_path)
    weights.to_parquet(weights_path)
    ts_features.to_parquet(feat_path)

    if verbose:
        print(f"  {ticker}: saved artifacts to {output_dir}/")

    return {
        'ticker':      ticker,
        'h':           h,
        'n_events':    len(cusum_events),
        'n_labels':    len(labels),
        'd_star':      d_star,
        'labels':      labels,
        'weights':     weights,
        'ts_features': ts_features,
        'clean':       raw,
    }


def create_pooled_dataset(
    tickers: list,
    per_stock_dir: str = 'data/processed/per_stock',
    common_start: str = None,
    common_end: str = None,
) -> tuple:
    """
    Stack all per-stock labels, weights, ts_features into pooled DataFrames.
    Optionally filter to the common date range.
    """
    all_labels   = []
    all_weights  = []
    all_features = []

    for ticker in tickers:
        labels  = pd.read_parquet(os.path.join(per_stock_dir, f'{ticker}_labels.parquet'))
        weights = pd.read_parquet(os.path.join(per_stock_dir, f'{ticker}_weights.parquet'))
        feats   = pd.read_parquet(os.path.join(per_stock_dir, f'{ticker}_ts_features.parquet'))

        # Filter to common date range if provided
        if common_start:
            mask = labels.index >= pd.Timestamp(common_start)
            labels  = labels[mask]
            weights = weights[mask]
            feats   = feats[feats.index >= pd.Timestamp(common_start)]

        if common_end:
            mask = labels.index <= pd.Timestamp(common_end)
            labels  = labels[mask]
            weights = weights[mask]
            feats   = feats[feats.index <= pd.Timestamp(common_end)]

        # Align
        common = labels.index.intersection(feats.index)
        labels  = labels.loc[common]
        weights = weights.loc[common]
        feats   = feats.loc[common]

        labels['ticker']  = ticker
        weights['ticker'] = ticker
        feats['ticker']   = ticker

        all_labels.append(labels)
        all_weights.append(weights)
        all_features.append(feats)

    pooled_labels   = pd.concat(all_labels).sort_index()
    pooled_weights  = pd.concat(all_weights).sort_index()
    pooled_features = pd.concat(all_features).sort_index()

    return pooled_labels, pooled_weights, pooled_features
