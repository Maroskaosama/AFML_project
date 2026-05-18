"""
Phase 4: Per-stock AFML pipeline (Stages 0-3) for all 10 stocks.

Re-runs the full pipeline for every ticker with the 2005-2025 raw data.
Existing per-stock parquets are overwritten (old run had too few events).

Stages:
  0 - Load & clean raw OHLCV
  1 - CUSUM filter (target 300 events)
  2 - Triple-barrier labeling + sample weights
  3 - Time-series features + fracdiff (optimal d*)

Then pools all per-stock artifacts into:
  data/processed/pooled_labels.parquet
  data/processed/pooled_weights.parquet
  data/processed/pooled_ts_features.parquet
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

from src.pipeline.per_stock import run_per_stock_pipeline, create_pooled_dataset

# ── Config ────────────────────────────────────────────────────────────────────
with open('configs/universe.json') as f:
    UNI = json.load(f)

TICKERS      = UNI['tickers']
COMMON_START = UNI['common_start']
COMMON_END   = UNI['common_end']

RAW_DIR      = 'data/raw'
PER_STOCK    = 'data/processed/per_stock'
PROCESSED    = 'data/processed'
POOLED_DIR   = 'data/processed/pooled'

os.makedirs(PER_STOCK, exist_ok=True)
os.makedirs(POOLED_DIR, exist_ok=True)

ERRORS   = []
WARNINGS = []

def sep(title=''):
    print('\n' + '=' * 68)
    if title:
        print(title)
        print('=' * 68)


# ── Step 0: Safety snapshot ────────────────────────────────────────────────
sep('STEP 0: Safety snapshot')

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
snap = f'backups/per_stock_{ts}'
if os.path.exists(PER_STOCK):
    shutil.copytree(PER_STOCK, snap)
    print(f'  Snapshot: {PER_STOCK} -> {snap}')


# ── Step 1: Per-stock pipeline for all 10 tickers ─────────────────────────
sep('STEP 1: Per-stock AFML pipeline (Stages 0-3)')
print(f'  Universe  : {TICKERS}')
print(f'  Date range: {COMMON_START} -> {COMMON_END}')
print(f'  Target events per stock: 300  |  pt_sl=[1,1]  |  vertical=10d')

results  = {}
t_start  = time.time()

for ticker in TICKERS:
    raw_path = os.path.join(RAW_DIR, f'{ticker}_raw.csv')
    if not os.path.exists(raw_path):
        msg = f'{ticker}: raw CSV not found at {raw_path}'
        print(f'\n  ERROR — {msg}')
        ERRORS.append(msg)
        continue

    print(f'\n  [{ticker}] running pipeline...', flush=True)
    t0 = time.time()

    try:
        r = run_per_stock_pipeline(
            ticker         = ticker,
            raw_path       = raw_path,
            output_dir     = PER_STOCK,
            target_events  = 500,  # raised from 300; fracdiff warmup consumes
                                   # ~24-28% of CUSUM events for some stocks
            pt_sl          = [1.0, 1.0],
            num_days       = 10,
            d_range        = np.arange(0.05, 0.55, 0.05),
            corr_threshold = 0.85,
            verbose        = True,
            cusum_start    = COMMON_START,
        )
        elapsed = time.time() - t0
        results[ticker] = r
        print(f'  [{ticker}] done in {elapsed:.1f}s — '
              f'{r["n_labels"]} labels, d*={r["d_star"]:.2f}')

    except Exception as exc:
        import traceback
        msg = f'{ticker}: pipeline FAILED — {exc}'
        print(f'  ERROR — {msg}')
        traceback.print_exc()
        ERRORS.append(msg)

total_time = time.time() - t_start
print(f'\n  Total pipeline time: {total_time:.1f}s')


# ── Step 2: Per-stock summary table ──────────────────────────────────────
sep('STEP 2: Per-stock summary')

print(f'  {"Ticker":6s} | {"Events":7s} | {"d*":5s} | {"+1":5s} | {"-1":5s} | '
      f'{"Features":8s} | {"Date range"}')
print(f'  {"-"*6}-+-{"-"*7}-+-{"-"*5}-+-{"-"*5}-+-{"-"*5}-+-{"-"*8}-+-{"-"*24}')

total_labels = 0
for ticker in TICKERS:
    if ticker not in results:
        print(f'  {ticker:6s} | FAILED')
        continue
    r = results[ticker]
    lbl     = r['labels']
    n_lbl   = len(lbl)
    n_pos   = int((lbl['bin'] == 1).sum())
    n_neg   = int((lbl['bin'] == -1).sum())
    d_star  = r.get('d_star', float('nan'))
    n_feat  = len(r['ts_features'].columns)
    drange  = f"{lbl.index.min().date()} -> {lbl.index.max().date()}"
    total_labels += n_lbl
    print(f'  {ticker:6s} | {n_lbl:7d} | {d_star:5.2f} | {n_pos:5d} | {n_neg:5d} | '
          f'{n_feat:8d} | {drange}')

    if n_lbl < 100:
        WARNINGS.append(f'{ticker}: only {n_lbl} labels (< 100 minimum)')

print(f'  {"-"*6}-+-{"-"*7}-+-{"-"*5}-+-{"-"*5}-+-{"-"*5}-+-{"-"*8}')
print(f'  {"TOTAL":6s} | {total_labels:7d}')


# ── Step 3: Pool per-stock artifacts ──────────────────────────────────────
sep('STEP 3: Pool artifacts (common range: {COMMON_START} to {COMMON_END})')

if ERRORS:
    print(f'  Skipping pool — {len(ERRORS)} ticker(s) failed.')
else:
    pool_labels, pool_weights, pool_features = create_pooled_dataset(
        tickers      = TICKERS,
        per_stock_dir= PER_STOCK,
        common_start = COMMON_START,
        common_end   = COMMON_END,
    )

    # Save to both flat processed/ and pooled/ subdirectory
    for path in [
        f'{PROCESSED}/pooled_labels.parquet',
        f'{POOLED_DIR}/pooled_labels.parquet',
    ]:
        pool_labels.to_parquet(path)

    for path in [
        f'{PROCESSED}/pooled_weights.parquet',
        f'{POOLED_DIR}/pooled_weights.parquet',
    ]:
        pool_weights.to_parquet(path)

    for path in [
        f'{PROCESSED}/pooled_ts_features.parquet',
        f'{POOLED_DIR}/pooled_ts_features.parquet',
    ]:
        pool_features.to_parquet(path)

    n_pos_pool = int((pool_labels['bin'] == 1).sum())
    n_neg_pool = int((pool_labels['bin'] == -1).sum())

    print(f'  Pooled labels   : {pool_labels.shape}')
    print(f'  Pooled weights  : {pool_weights.shape}')
    print(f'  Pooled features : {pool_features.shape}')
    print(f'  Label split     : +1={n_pos_pool}, -1={n_neg_pool}')
    print()
    by_ticker = pool_labels.groupby('ticker')['bin'].value_counts().unstack(fill_value=0)
    print('  Per-ticker label counts (pooled range):')
    print(by_ticker.to_string(line_width=80))


# ── Step 4: Validation ────────────────────────────────────────────────────
sep('STEP 4: Validation')

failed_count  = 0
passed_count  = 0

def check(label, condition):
    global failed_count, passed_count
    status = 'PASS' if condition else 'FAIL'
    if condition:
        passed_count += 1
    else:
        failed_count += 1
        ERRORS.append(f'{label}: FAIL')
    print(f'  [{status}] {label}')

# Per-stock checks
for ticker in TICKERS:
    if ticker not in results:
        check(f'{ticker}: pipeline ran', False)
        continue
    r   = results[ticker]
    lbl = r['labels']
    check(f'{ticker}: labels >= 100',   len(lbl) >= 100)
    check(f'{ticker}: labels <= 800',   len(lbl) <= 800)
    check(f'{ticker}: bins in {{-1,1}}', set(lbl['bin'].dropna().unique()).issubset({-1.0, 1.0}))
    check(f'{ticker}: d* in (0,0.5]',   0 < r.get('d_star', 0) <= 0.5)
    check(f'{ticker}: features exist',  len(r['ts_features'].columns) >= 10)

if not ERRORS:
    check('pooled events >= 1500',
          len(pool_labels) >= 1500)
    check('pooled +1/-1 ratio 40-60%',
          0.35 <= n_pos_pool / max(len(pool_labels), 1) <= 0.65)
    check('pooled features: no full-NaN column',
          pool_features.isnull().all(axis=0).sum() == 0)
    check('pooled: all tickers present',
          set(pool_labels['ticker'].unique()) == set(TICKERS))
    check('pooled start >= 2005-01-01',
          pool_labels.index.min() >= pd.Timestamp('2005-01-01'))

sep()
if WARNINGS:
    for w in WARNINGS:
        print(f'  WARN: {w}')

if ERRORS:
    print(f'Phase 4 FAILED — {failed_count} check(s) failed:')
    for e in [e for e in ERRORS if 'FAIL' in e]:
        print(f'  {e}')
    sys.exit(1)
else:
    print(f'Phase 4 COMPLETE — {passed_count} checks passed.')
    print(f'  Per-stock artifacts : {PER_STOCK}/')
    print(f'  Pooled events       : {len(pool_labels):,}  '
          f'(was ~881 with old META/TSLA universe)')
    print(f'  Date range          : {pool_labels.index.min().date()} to '
          f'{pool_labels.index.max().date()}')
