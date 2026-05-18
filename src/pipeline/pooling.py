"""
Pooling: stack per-stock AFML artifacts into pooled multi-asset datasets.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def load_universe(cfg_path: str = 'configs/universe.json') -> dict:
    with open(cfg_path) as f:
        return json.load(f)


def build_pooled_modelling_dataset(
    tickers: List[str],
    per_stock_dir: str,
    alpha_panel_path: str,
    selected_alphas: List[str],
    common_start: str,
    common_end: str,
    output_path: Optional[str] = None,
    macro_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge per-stock {ticker}_ts_features + selected alpha features
    + {ticker}_labels + {ticker}_weights into one pooled DataFrame.

    Each row is one event. Index = event date (DatetimeIndex).
    Columns = [TS features] + [macro features (opt)] + [alpha features]
              + [label, t1, weight, ticker].

    Parameters
    ----------
    tickers : list of ticker strings
    per_stock_dir : directory containing {ticker}_labels/weights/ts_features parquets
    alpha_panel_path : path to panel_alpha_features_pruned.parquet
    selected_alphas : list of alpha column names to include
    common_start, common_end : date range filter strings
    output_path : if given, save result to this parquet path
    macro_features : optional date-indexed DataFrame of macro regime features
        (same value for all tickers on a given date); aligned by reindex + ffill

    Returns
    -------
    pd.DataFrame with DatetimeIndex and ticker column
    """
    alpha_panel = pd.read_parquet(alpha_panel_path)

    pooled_rows = []
    skipped     = []

    for ticker in tickers:
        labels_path  = os.path.join(per_stock_dir, f'{ticker}_labels.parquet')
        weights_path = os.path.join(per_stock_dir, f'{ticker}_weights.parquet')
        feat_path    = os.path.join(per_stock_dir, f'{ticker}_ts_features.parquet')

        if not all(os.path.exists(p) for p in [labels_path, weights_path, feat_path]):
            skipped.append((ticker, 'missing artifacts'))
            continue

        labels  = pd.read_parquet(labels_path)
        weights = pd.read_parquet(weights_path)
        ts_feat = pd.read_parquet(feat_path)

        # Filter to common date range
        mask    = (labels.index >= common_start) & (labels.index <= common_end)
        labels  = labels[mask]
        weights = weights.reindex(labels.index)
        ts_feat = ts_feat.reindex(labels.index)

        if len(labels) == 0:
            skipped.append((ticker, 'no labels in common range'))
            continue

        # Keep only valid labels
        valid_label_col = 'bin' if 'bin' in labels.columns else 'label'
        labels = labels[labels[valid_label_col].isin([-1.0, 1.0])]
        if len(labels) == 0:
            skipped.append((ticker, 'no valid bins'))
            continue

        # Alpha features for this ticker
        try:
            alpha_for_ticker = alpha_panel.xs(ticker, level='ticker')
        except KeyError:
            alpha_for_ticker = pd.DataFrame(
                index=pd.DatetimeIndex([]), columns=selected_alphas
            )

        alpha_aligned = alpha_for_ticker[selected_alphas].reindex(labels.index)

        # TS feature columns (exclude label/weight/meta cols)
        ts_cols = [
            c for c in ts_feat.columns
            if c not in {'label', 'weight', 't1', 'ret', 'bin'}
        ]

        # t1 column
        t1_col = labels['t1'] if 't1' in labels.columns else pd.Series(
            index=labels.index, dtype='datetime64[ns]'
        )

        label_series  = labels[valid_label_col].rename('label')
        weight_series = weights.iloc[:, 0].rename('weight') if weights.shape[1] > 0 \
                        else pd.Series(1.0, index=labels.index, name='weight')

        ts_part    = ts_feat[ts_cols]
        macro_part = macro_features.reindex(labels.index) if macro_features is not None else None

        # Build non-alpha part first; drop rows where TS/macro are NaN
        non_alpha_parts = [ts_part]
        if macro_part is not None:
            non_alpha_parts.append(macro_part)
        non_alpha_parts += [label_series, t1_col.rename('t1'), weight_series]
        row = pd.concat(non_alpha_parts, axis=1).dropna()

        if len(row) == 0:
            skipped.append((ticker, 'all rows NaN in TS/macro features'))
            continue

        # Alpha NaN → 0 (neutral cross-sectional rank; avoids dropping entire stocks)
        alpha_clean = alpha_aligned.reindex(row.index).fillna(0.0)
        row = pd.concat([row, alpha_clean], axis=1)

        row['ticker'] = ticker
        pooled_rows.append(row)

    if not pooled_rows:
        raise RuntimeError(
            f'No valid per-stock data found. Skipped: {skipped}'
        )

    pooled = pd.concat(pooled_rows).sort_index()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        pooled.to_parquet(output_path)

    if skipped:
        print(f'  WARNING: skipped {len(skipped)} tickers: {skipped}')

    return pooled


def pool_per_stock_artifacts(
    tickers: List[str],
    per_stock_dir: str,
    common_start: str,
    common_end: str,
    output_dir: str = 'data/processed/pooled',
) -> dict:
    """
    Stack per-stock labels, weights, and ts_features into pooled parquets
    (without alpha features — used as intermediate artifacts).

    Returns dict with keys: labels, weights, features.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_labels   = []
    all_weights  = []
    all_features = []

    for ticker in tickers:
        for name, store in [
            ('labels',   all_labels),
            ('weights',  all_weights),
            ('ts_features', all_features),
        ]:
            p = os.path.join(per_stock_dir, f'{ticker}_{name}.parquet')
            if not os.path.exists(p):
                continue
            df = pd.read_parquet(p)
            df = df[(df.index >= common_start) & (df.index <= common_end)]
            df['ticker'] = ticker
            store.append(df)

    result = {}
    for name, store, fname in [
        ('labels',   all_labels,   'pooled_labels.parquet'),
        ('weights',  all_weights,  'pooled_weights.parquet'),
        ('features', all_features, 'pooled_ts_features.parquet'),
    ]:
        if store:
            pooled = pd.concat(store).sort_index()
            pooled.to_parquet(os.path.join(output_dir, fname))
            result[name] = pooled

    return result
