"""
Phase 5: Alpha Engine — compute 101 alphas on the new 10-stock panel,
run full diagnostics, apply exclusion + pruning + budget, save artifacts.

Outputs:
  data/processed/panel_alpha_features.parquet       (all computed alphas)
  data/processed/panel_alpha_features_pruned.parquet (selected alphas only)
  data/processed/alpha_diagnostics.parquet          (per-alpha stats)
  configs/selected_alphas.json                      (final selection metadata)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

from src.alphas.engine       import compute_all_alphas, compute_alpha_diagnostics
from src.alphas.diagnostics  import run_full_diagnostics, save_selected_alphas
from src.alphas.registry     import SECTOR_MAP

# ── Config ────────────────────────────────────────────────────────────────────
with open('configs/universe.json') as f:
    UNI = json.load(f)

TICKERS       = UNI['tickers']
COMMON_START  = UNI['common_start']
COMMON_END    = UNI['common_end']
ALPHA_PARAMS  = UNI['alpha_params']   # max_nan_pct, min_std, max_corr, max_alphas

PANEL_PATH       = 'data/processed/panel_ohlcv.parquet'
ALPHA_PATH       = 'data/processed/panel_alpha_features.parquet'
PRUNED_PATH      = 'data/processed/panel_alpha_features_pruned.parquet'
DIAG_PATH        = 'data/processed/alpha_diagnostics.parquet'
SELECTED_CFG     = 'configs/selected_alphas.json'
FIGURES_DIR      = 'reports/figures'

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs('configs',   exist_ok=True)

ERRORS = []

def sep(title=''):
    print('\n' + '=' * 68)
    if title:
        print(title)
        print('=' * 68)


# ── Step 1: Load panel ────────────────────────────────────────────────────────
sep('STEP 1: Load panel_ohlcv.parquet')

panel = pd.read_parquet(PANEL_PATH)
tickers_in = sorted(panel.index.get_level_values('ticker').unique().tolist())
date_min   = panel.index.get_level_values('Date').min().date()
date_max   = panel.index.get_level_values('Date').max().date()

print(f'  Shape    : {panel.shape}')
print(f'  Tickers  : {tickers_in}')
print(f'  Dates    : {date_min} -> {date_max}')
print(f'  NaN total: {panel.isnull().sum().sum()}')

if set(tickers_in) != set(TICKERS):
    ERRORS.append(f'Panel tickers mismatch: {tickers_in} vs {TICKERS}')

if date_min.isoformat() != COMMON_START:
    ERRORS.append(f'Panel start mismatch: {date_min} vs {COMMON_START}')

if ERRORS:
    print(f'  FATAL: {ERRORS}')
    sys.exit(1)


# ── Step 2: Compute all 101 alphas ────────────────────────────────────────────
sep('STEP 2: Compute all 101 alpha formulas')

if os.path.exists(ALPHA_PATH):
    print(f'  Cached panel found — loading {ALPHA_PATH}')
    alpha_panel = pd.read_parquet(ALPHA_PATH)
    print(f'  Alpha panel shape : {alpha_panel.shape}')
else:
    print(f'  SECTOR_MAP: {SECTOR_MAP}')
    t0 = time.time()
    alpha_panel = compute_all_alphas(panel, sector_map=SECTOR_MAP, verbose=True)
    elapsed = time.time() - t0
    print(f'\n  Alpha panel shape : {alpha_panel.shape}')
    print(f'  Alpha count       : {len(alpha_panel.columns)}')
    print(f'  Computation time  : {elapsed:.1f}s')
    alpha_panel.to_parquet(ALPHA_PATH)
    print(f'  Saved: {ALPHA_PATH}')


# ── Step 3: Quick NaN / inf summary ───────────────────────────────────────────
sep('STEP 3: NaN / constant / inf summary')

nan_pcts = alpha_panel.isnull().mean() * 100
stds     = alpha_panel.std()
all_nan  = alpha_panel.isnull().all()

print(f'  Alphas <20% NaN  : {(nan_pcts < 20).sum()}')
print(f'  Alphas 20-40% NaN: {((nan_pcts >= 20) & (nan_pcts < 40)).sum()}')
print(f'  Alphas 40-60% NaN: {((nan_pcts >= 40) & (nan_pcts < 60)).sum()}')
print(f'  Alphas >60% NaN  : {(nan_pcts >= 60).sum()}')
print(f'  All-NaN alphas   : {all_nan.sum()} — {all_nan[all_nan].index.tolist()}')
print(f'  Constant alphas  : {(stds < 1e-8).sum()} — '
      f'{stds[stds < 1e-8].index.tolist()}')

print('\n  Top 10 NaN rate:')
for name, pct in nan_pcts.nlargest(10).items():
    print(f'    {name}: {pct:.1f}%')


# ── Step 4: Per-alpha diagnostics (ADF, autocorr, etc.) ──────────────────────
sep('STEP 4: Full per-alpha diagnostics (ADF, skew, autocorr)')

valid_for_diag = alpha_panel.loc[:, ~all_nan]
print(f'  Computing diagnostics for {len(valid_for_diag.columns)} non-all-NaN alphas...')
t0 = time.time()
diag = compute_alpha_diagnostics(valid_for_diag)
diag.to_parquet(DIAG_PATH)
print(f'  Done in {time.time()-t0:.1f}s  |  saved: {DIAG_PATH}')
print(f'  Stationary (ADF p<0.05): {(diag["adf_pval_median"] < 0.05).sum()} alphas')
print(f'  Median ADF p-value      : {diag["adf_pval_median"].median():.4f}')
print(f'  Alphas with any-inf     : {diag["any_inf"].sum()}')


# ── Step 5: Exclusion → pruning → budget ──────────────────────────────────────
sep('STEP 5: Exclusion rules -> redundancy pruning -> budget')

# Reference ticker for correlation matrix: use NVDA (most events)
ref_ticker = 'NVDA'
max_nan_pct = ALPHA_PARAMS['max_nan_pct'] * 100   # 0.40 -> 40.0
min_std     = ALPHA_PARAMS['min_std']
corr_thresh = ALPHA_PARAMS['max_corr']
budget      = ALPHA_PARAMS['max_alphas']

result = run_full_diagnostics(
    alpha_panel,
    reference_ticker = ref_ticker,
    max_nan_pct      = max_nan_pct,
    min_std          = min_std,
    corr_threshold   = corr_thresh,
    budget           = budget,
    diag_df          = diag,
)

print(f'  After exclusion rules : {result["n_surviving"]} alphas')
print(f'  After pruning (|r|>{corr_thresh}): {result["n_post_prune"]} alphas')
print(f'  After budget ({budget}): {result["n_selected"]} alphas')
print(f'  Max |corr| in selected: {result["max_corr_selected"]:.4f}')
print(f'\n  Excluded (NaN>40%): {len(result["exclusion_log"]["nan_gt_40pct"])} '
      f'— {result["exclusion_log"]["nan_gt_40pct"]}')
print(f'  Excluded (constant): {len(result["exclusion_log"]["constant"])} '
      f'— {result["exclusion_log"]["constant"]}')
print(f'  Excluded (inf)     : {len(result["exclusion_log"]["has_inf"])} '
      f'— {result["exclusion_log"]["has_inf"]}')
print(f'  Pruned redundant   : {len(result["pruned_pairs"])}')
print(f'\n  Selected alphas ({len(result["selected"])}): {result["selected"]}')


# ── Step 6: Save pruned panel & config ────────────────────────────────────────
sep('STEP 6: Save pruned panel & configs/selected_alphas.json')

pruned = alpha_panel[result['selected']]
pruned.to_parquet(PRUNED_PATH)
print(f'  Saved pruned panel  : {PRUNED_PATH}  {pruned.shape}')

save_selected_alphas(result, path=SELECTED_CFG)
print(f'  Saved selected cfg  : {SELECTED_CFG}')


# ── Step 7: Correlation heatmap ───────────────────────────────────────────────
sep('STEP 7: Correlation heatmap (selected alphas, NVDA slice)')

try:
    ref_data = alpha_panel[result['selected']].xs(ref_ticker, level='ticker')
    corr_sel = ref_data.corr()
    n_sel    = len(result['selected'])

    fig, ax = plt.subplots(figsize=(max(10, n_sel * 0.4), max(8, n_sel * 0.4)))
    im = ax.imshow(corr_sel.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n_sel))
    ax.set_yticks(range(n_sel))
    ax.set_xticklabels(corr_sel.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr_sel.columns, fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Selected Alpha Correlation Matrix ({ref_ticker}, n={n_sel})')
    plt.tight_layout()
    heatmap_path = os.path.join(FIGURES_DIR, 'phase5_alpha_correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=100)
    plt.close()
    print(f'  Heatmap saved: {heatmap_path}')
except Exception as e:
    print(f'  Heatmap skipped: {e}')


# ── Step 8: Validation ────────────────────────────────────────────────────────
sep('STEP 8: Validation')

passed = 0
failed = 0

def check(label, cond):
    global passed, failed
    s = 'PASS' if cond else 'FAIL'
    if cond:
        passed += 1
    else:
        failed += 1
        ERRORS.append(f'{label}: FAIL')
    print(f'  [{s}] {label}')

check('alpha panel shape: 101 columns',        len(alpha_panel.columns) == 101)
check('alpha panel: correct tickers',          set(alpha_panel.index.get_level_values('ticker').unique()) == set(TICKERS))
check('alpha panel: starts at 2005-01-03',     alpha_panel.index.get_level_values('Date').min().date().isoformat() == COMMON_START)
check('diagnostics saved',                     os.path.exists(DIAG_PATH))
check('surviving > 60',                        result['n_surviving'] > 60)
check('post_prune > 20',                       result['n_post_prune'] > 20)
check(f'selected == {budget}',                 len(result['selected']) == budget)
check(f'max corr <= {corr_thresh}',            result['max_corr_selected'] <= corr_thresh)
check('pruned panel has selected columns',     set(pruned.columns) == set(result['selected']))
check('no inf in pruned panel',                not np.isinf(pruned.values).any())
check('selected_alphas.json saved',            os.path.exists(SELECTED_CFG))
check('pruned panel parquet saved',            os.path.exists(PRUNED_PATH))

sep()
if ERRORS:
    print(f'Phase 5 FAILED — {failed} check(s) failed:')
    for e in ERRORS:
        print(f'  {e}')
    sys.exit(1)
else:
    print(f'Phase 5 COMPLETE — {passed} checks passed.')
    print(f'  Panel          : {alpha_panel.shape}')
    print(f'  Selected alphas: {len(result["selected"])} / 101')
    print(f'  Pruned panel   : {pruned.shape}')
    print(f'  Max |corr|     : {result["max_corr_selected"]:.4f}')
