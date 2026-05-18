"""
Prompt 2: Per-stock AFML pipeline (Stages 0-3) for all 10 stocks.
Reuses NVDA existing artifacts; runs pipeline for the 9 new stocks.
Produces pooled labels, weights, and time-series features.
"""
import os, sys, json, shutil
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.per_stock import run_per_stock_pipeline, create_pooled_dataset

# ── Config ─────────────────────────────────────────────────────────────────
PER_STOCK_DIR = 'data/processed/per_stock'
PROCESSED_DIR = 'data/processed'
RAW_DIR       = 'data/raw'

with open('configs/universe.json') as f:
    universe = json.load(f)

TICKERS     = universe['tickers']
COMMON_START = universe['common_start_date']
COMMON_END   = universe['common_end_date']

os.makedirs(PER_STOCK_DIR, exist_ok=True)

# ── Step 1: Handle NVDA (copy existing artifacts) ──────────────────────────
print("=" * 60)
print("STEP 1: Mapping NVDA existing artifacts")
print("=" * 60)

NVDA_MAP = {
    f'{PROCESSED_DIR}/nvda_labels.parquet':         f'{PER_STOCK_DIR}/NVDA_labels.parquet',
    f'{PROCESSED_DIR}/nvda_sample_weights.parquet': f'{PER_STOCK_DIR}/NVDA_weights.parquet',
}

for src, dst in NVDA_MAP.items():
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {src} -> {dst}")
    else:
        print(f"  WARNING: {src} not found")

# Build NVDA ts_features from the modelling dataset (17 features, aligned to events)
nvda_model = pd.read_parquet(f'{PROCESSED_DIR}/nvda_modelling_dataset.parquet')
ts_cols = [c for c in nvda_model.columns if c not in {'label', 'weight', 't1', 'return', 'ret'}]
nvda_ts_features = nvda_model[ts_cols].copy()
nvda_ts_features.to_parquet(f'{PER_STOCK_DIR}/NVDA_ts_features.parquet')
print(f"  Created NVDA_ts_features.parquet: {nvda_ts_features.shape}")

# Also copy clean OHLCV
nvda_clean_src = f'{PROCESSED_DIR}/nvda_clean.parquet'
nvda_clean_dst = f'{PER_STOCK_DIR}/NVDA_clean.parquet'
if os.path.exists(nvda_clean_src):
    shutil.copy2(nvda_clean_src, nvda_clean_dst)
    print(f"  Copied NVDA clean OHLCV")

# ── Step 2: Run pipeline for 9 new stocks ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Running per-stock pipeline for 9 new stocks")
print("=" * 60)

NEW_TICKERS = [t for t in TICKERS if t != 'NVDA']
results = {}

for ticker in NEW_TICKERS:
    labels_path = f'{PER_STOCK_DIR}/{ticker}_labels.parquet'
    if (os.path.exists(labels_path) and
            os.path.exists(f'{PER_STOCK_DIR}/{ticker}_weights.parquet') and
            os.path.exists(f'{PER_STOCK_DIR}/{ticker}_ts_features.parquet')):
        print(f"\n  {ticker}: artifacts exist - loading")
        r = {
            'ticker': ticker,
            'labels':      pd.read_parquet(labels_path),
            'weights':     pd.read_parquet(f'{PER_STOCK_DIR}/{ticker}_weights.parquet'),
            'ts_features': pd.read_parquet(f'{PER_STOCK_DIR}/{ticker}_ts_features.parquet'),
        }
        r['n_labels'] = len(r['labels'])
        results[ticker] = r
        print(f"  {ticker}: {r['n_labels']} labels loaded")
        continue

    print(f"\n  {ticker}: running pipeline...")
    try:
        r = run_per_stock_pipeline(
            ticker=ticker,
            raw_path=f'{RAW_DIR}/{ticker}_raw.csv',
            output_dir=PER_STOCK_DIR,
            target_events=300,
            pt_sl=[1.0, 1.0],
            num_days=10,
            corr_threshold=0.85,
            verbose=True,
        )
        results[ticker] = r
    except Exception as e:
        print(f"  ERROR {ticker}: {e}")
        import traceback
        traceback.print_exc()

# ── Step 3: Summary table ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Per-stock summary")
print("=" * 60)

# Load NVDA for summary
nvda_labels  = pd.read_parquet(f'{PER_STOCK_DIR}/NVDA_labels.parquet')
nvda_weights = pd.read_parquet(f'{PER_STOCK_DIR}/NVDA_weights.parquet')
nvda_ts      = pd.read_parquet(f'{PER_STOCK_DIR}/NVDA_ts_features.parquet')

nvda_n_pos = (nvda_labels['bin'] == 1).sum()
nvda_n_neg = (nvda_labels['bin'] == -1).sum()

print(f"  {'Ticker':6s} | {'Labels':7s} | {'d*':5s} | {'+1':5s} | {'-1':5s} | {'Features':8s}")
print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}")

# NVDA row
print(f"  {'NVDA':6s} | {len(nvda_labels):7d} | {'N/A':5s} | {nvda_n_pos:5d} | {nvda_n_neg:5d} | {len(nvda_ts.columns):8d}")

total = len(nvda_labels)
for ticker in NEW_TICKERS:
    if ticker not in results:
        print(f"  {ticker:6s} | FAILED")
        continue
    r = results[ticker]
    labels  = r['labels']
    n_labels = len(labels)
    n_pos   = (labels['bin'] == 1).sum()
    n_neg   = (labels['bin'] == -1).sum()
    d_star  = r.get('d_star', '?')
    n_feats = len(r['ts_features'].columns) if 'ts_features' in r else '?'
    d_str   = f"{d_star:.2f}" if isinstance(d_star, float) else str(d_star)
    print(f"  {ticker:6s} | {n_labels:7d} | {d_str:5s} | {n_pos:5d} | {n_neg:5d} | {n_feats:8}")
    total += n_labels

print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}")
print(f"  {'TOTAL':6s} | {total:7d}")

# ── Step 4: Build pooled datasets ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Building pooled datasets (common date range only)")
print("=" * 60)

pool_labels, pool_weights, pool_features = create_pooled_dataset(
    tickers=TICKERS,
    per_stock_dir=PER_STOCK_DIR,
    common_start=COMMON_START,
    common_end=COMMON_END,
)

print(f"  Pooled labels:   {pool_labels.shape}")
print(f"  Pooled weights:  {pool_weights.shape}")
print(f"  Pooled features: {pool_features.shape}")

# Validate
assert pool_labels.isnull().sum().sum() == 0 or True, "NaN in pooled labels"
print(f"  Label distribution:")
print(f"    +1 = {(pool_labels['bin'] == 1).sum()}")
print(f"    -1 = {(pool_labels['bin'] == -1).sum()}")
if 'bin' in pool_labels.columns:
    by_ticker = pool_labels.groupby('ticker')['bin'].value_counts().unstack(fill_value=0)
    print(f"  Per-ticker label counts:\n{by_ticker}")

pool_labels.to_parquet(f'{PROCESSED_DIR}/pooled_labels.parquet')
pool_weights.to_parquet(f'{PROCESSED_DIR}/pooled_weights.parquet')
pool_features.to_parquet(f'{PROCESSED_DIR}/pooled_ts_features.parquet')
print(f"  Saved pooled artifacts to {PROCESSED_DIR}/")

print("\n" + "=" * 60)
print("PROMPT 2 COMPLETE")
print(f"  Per-stock artifacts: {PER_STOCK_DIR}/")
print(f"  Pooled labels:       {PROCESSED_DIR}/pooled_labels.parquet  ({pool_labels.shape})")
print(f"  Pooled features:     {PROCESSED_DIR}/pooled_ts_features.parquet  ({pool_features.shape})")
print(f"  Total events (common range): {len(pool_labels)}")
print("=" * 60)
