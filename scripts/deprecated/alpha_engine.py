"""
Prompt 3: Alpha Engine — compute all 101 alphas on the 10-stock panel,
run diagnostics, and validate outputs.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alphas.engine import compute_all_alphas, compute_alpha_diagnostics
from src.alphas.registry import SECTOR_MAP

# ── Config ─────────────────────────────────────────────────────────────────
PANEL_PATH   = 'data/processed/panel_ohlcv.parquet'
OUTPUT_PATH  = 'data/processed/panel_alpha_features.parquet'
DIAG_PATH    = 'data/processed/alpha_diagnostics.parquet'

# ── Step 1: Load panel ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading panel_ohlcv.parquet")
print("=" * 60)

panel = pd.read_parquet(PANEL_PATH)
print(f"  Panel shape: {panel.shape}")
print(f"  Tickers: {panel.index.get_level_values('ticker').unique().tolist()}")
print(f"  Date range: {panel.index.get_level_values('Date').min().date()} to {panel.index.get_level_values('Date').max().date()}")
print(f"  NaN count: {panel.isnull().sum().sum()}")

# ── Step 2: Compute all alphas ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Computing all 101 alpha formulas")
print("=" * 60)

alpha_panel = compute_all_alphas(panel, sector_map=SECTOR_MAP, verbose=True)

print(f"\nAlpha panel shape: {alpha_panel.shape}")
print(f"Alpha columns: {len(alpha_panel.columns)}")

# ── Step 3: Save ───────────────────────────────────────────────────────────
alpha_panel.to_parquet(OUTPUT_PATH)
print(f"\nSaved to {OUTPUT_PATH}")

# ── Step 4: Quick diagnostic ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Alpha diagnostics (NaN rate per alpha)")
print("=" * 60)

nan_pcts = alpha_panel.isnull().mean() * 100
print(f"  Alphas with <20% NaN: {(nan_pcts < 20).sum()}")
print(f"  Alphas with 20-40% NaN: {((nan_pcts >= 20) & (nan_pcts < 40)).sum()}")
print(f"  Alphas with 40-60% NaN: {((nan_pcts >= 40) & (nan_pcts < 60)).sum()}")
print(f"  Alphas with >60% NaN:   {(nan_pcts >= 60).sum()}")

# Constant alphas (zero std)
stds = alpha_panel.std()
constant = stds[stds < 1e-8]
print(f"\n  Constant alphas (std < 1e-8): {len(constant)}")
if len(constant) > 0:
    print(f"    {constant.index.tolist()}")

# NaN-only alphas (alpha056)
all_nan = alpha_panel.isnull().all()
print(f"  All-NaN alphas: {all_nan.sum()} — {all_nan[all_nan].index.tolist()}")

# Top 10 by NaN rate
print("\n  Top 10 highest NaN rate:")
for name, pct in nan_pcts.nlargest(10).items():
    print(f"    {name}: {pct:.1f}%")

print("\n  Top 10 lowest NaN rate:")
for name, pct in nan_pcts.nsmallest(10).items():
    print(f"    {name}: {pct:.1f}%")

# ── Step 5: Full diagnostics ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Computing full diagnostics (ADF, autocorr, etc.)")
print("=" * 60)

# Only run diagnostics on non-all-NaN alphas
valid_alphas = alpha_panel.loc[:, ~all_nan]
diag = compute_alpha_diagnostics(valid_alphas)
diag.to_parquet(DIAG_PATH)
print(f"  Diagnostics saved to {DIAG_PATH}")
print(f"  Shape: {diag.shape}")
print(f"\n  Median ADF p-value across alphas: {diag['adf_pval_median'].median():.4f}")
print(f"  Alphas stationary (ADF p<0.05): {(diag['adf_pval_median'] < 0.05).sum()}")
print(f"  Alphas with inf: {diag['any_inf'].sum()}")

print("\n" + "=" * 60)
print("PROMPT 3 COMPLETE")
print(f"  Alpha features:  {OUTPUT_PATH}  {alpha_panel.shape}")
print(f"  Diagnostics:     {DIAG_PATH}")
print(f"  Total alphas OK: {(~all_nan & (stds >= 1e-8)).sum()} / 101")
print("=" * 60)
