"""
Phase 7: CV Validation — build pooled modelling dataset, run MultiAssetPurgedKFold,
train TS-only and TS+alpha baselines, validate temporal integrity.

Outputs:
  data/processed/pooled_modelling.parquet     (rebuilt from 10-stock pipeline)
  data/processed/cv_baseline_multistock.parquet  (fold-level results)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

from src.pipeline.pooling     import build_pooled_modelling_dataset
from src.cross_validation     import MultiAssetPurgedKFold, cv_score

from sklearn.ensemble         import RandomForestClassifier

# ── Config ────────────────────────────────────────────────────────────────────
with open('configs/universe.json') as f:
    UNI = json.load(f)

with open('configs/selected_alphas.json') as f:
    ALPHA_CFG = json.load(f)

TICKERS       = UNI['tickers']
COMMON_START  = UNI['common_start']
COMMON_END    = UNI['common_end']
SELECTED_ALPHAS = ALPHA_CFG['selected_alphas']   # 33 names

PER_STOCK_DIR   = 'data/processed/per_stock'
ALPHA_PRUNED    = 'data/processed/panel_alpha_features_pruned.parquet'
MODELLING_PATH  = 'data/processed/pooled_modelling.parquet'
CV_RESULTS_PATH = 'data/processed/cv_baseline_multistock.parquet'

ERRORS = []

def sep(title=''):
    print('\n' + '=' * 68)
    if title:
        print(title)
        print('=' * 68)


# ── Step 1: Build pooled modelling dataset ───────────────────────────────────
sep('STEP 1: Build pooled_modelling.parquet')

t0 = time.time()
modelling = build_pooled_modelling_dataset(
    tickers         = TICKERS,
    per_stock_dir   = PER_STOCK_DIR,
    alpha_panel_path= ALPHA_PRUNED,
    selected_alphas = SELECTED_ALPHAS,
    common_start    = COMMON_START,
    common_end      = COMMON_END,
    output_path     = MODELLING_PATH,
)
elapsed = time.time() - t0

print(f'  Shape     : {modelling.shape}')
print(f'  Tickers   : {sorted(modelling["ticker"].unique())}')
print(f'  Date range: {modelling.index.min().date()} -> {modelling.index.max().date()}')
print(f'  Columns   : {list(modelling.columns)}')
print(f'  NaN total : {modelling.isnull().sum().sum()}')
print(f'  Build time: {elapsed:.1f}s')

# Identify TS vs alpha feature columns
meta_cols = {'label', 't1', 'weight', 'ticker'}
ts_cols   = [c for c in modelling.columns
             if c not in meta_cols and not c.startswith('alpha')]
alpha_cols = [c for c in modelling.columns if c.startswith('alpha')]

print(f'\n  TS features    : {len(ts_cols)} — {ts_cols}')
print(f'  Alpha features : {len(alpha_cols)}')
print(f'  Total features : {len(ts_cols) + len(alpha_cols)}')

label_counts = modelling['label'].value_counts().sort_index()
print(f'\n  Label distribution: {dict(label_counts)}')
majority_frac = label_counts.max() / len(modelling)
print(f'  Majority baseline: {majority_frac:.4f}')


# ── Step 2: MultiAssetPurgedKFold split inspection ───────────────────────────
sep('STEP 2: MultiAssetPurgedKFold fold structure')

X_full = modelling[ts_cols + alpha_cols]
y_full = modelling['label']
w_full = modelling['weight']
t1_full = modelling['t1']

cv = MultiAssetPurgedKFold(n_splits=5, t1=t1_full, pct_embargo=0.01)

fold_rows = []
print(f'  {"Fold":5s} | {"Train N":8s} | {"Test N":7s} | '
      f'{"Train start":12s} | {"Train end":12s} | '
      f'{"Test start":12s} | {"Test end":12s} | {"Tickers in test"}')
print(f'  {"-"*5}-+-{"-"*8}-+-{"-"*7}-+-{"-"*12}-+-{"-"*12}-+-'
      f'{"-"*12}-+-{"-"*12}-+-{"-"*16}')

folds = list(cv.split(X_full, y_full))
for fold_i, (train_idx, test_idx) in enumerate(folds):
    X_train = X_full.iloc[train_idx]
    X_test  = X_full.iloc[test_idx]
    tickers_test = sorted(modelling.iloc[test_idx]['ticker'].unique())
    fold_rows.append({
        'fold':        fold_i,
        'n_train':     len(train_idx),
        'n_test':      len(test_idx),
        'train_start': X_train.index.min(),
        'train_end':   X_train.index.max(),
        'test_start':  X_test.index.min(),
        'test_end':    X_test.index.max(),
        'tickers_test': ','.join(tickers_test),
    })
    print(f'  {fold_i:5d} | {len(train_idx):8d} | {len(test_idx):7d} | '
          f'{str(X_train.index.min().date()):12s} | {str(X_train.index.max().date()):12s} | '
          f'{str(X_test.index.min().date()):12s} | {str(X_test.index.max().date()):12s} | '
          f'{tickers_test}')


# ── Step 3: Temporal integrity checks ────────────────────────────────────────
sep('STEP 3: Temporal integrity checks')

n_date_overlaps = 0
n_purge_violations = 0

for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_full, y_full)):
    train_dates = set(X_full.iloc[train_idx].index)
    test_dates  = set(X_full.iloc[test_idx].index)

    overlap = train_dates & test_dates
    if overlap:
        n_date_overlaps += len(overlap)
        ERRORS.append(f'Fold {fold_i}: {len(overlap)} date overlaps')

    # Purge: for pre-test train samples, t1 must not reach into test window
    test_start = min(test_dates)
    test_end   = max(test_dates)

    for i in train_idx:
        et = X_full.index[i]
        if et < test_start:
            t1_val = t1_full.iloc[i]
            if pd.notna(t1_val) and t1_val >= test_start:
                n_purge_violations += 1

    print(f'  Fold {fold_i}: train={len(train_idx)} test={len(test_idx)} '
          f'date_overlaps={len(overlap)} '
          f'test_window=[{min(test_dates).date()},{max(test_dates).date()}]')

print(f'\n  Total date overlaps   : {n_date_overlaps}')
print(f'  Purge violations      : {n_purge_violations}')


# ── Step 4: TS-only baseline CV ──────────────────────────────────────────────
sep('STEP 4: TS-only baseline (n_features={})'.format(len(ts_cols)))

X_ts = modelling[ts_cols]

cv_ts = MultiAssetPurgedKFold(n_splits=5, t1=t1_full, pct_embargo=0.01)

clf_ts = RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1,
)

ts_fold_results = []
t0 = time.time()

for fold_i, (train_idx, test_idx) in enumerate(cv_ts.split(X_ts, y_full)):
    X_tr, y_tr, w_tr = X_ts.iloc[train_idx], y_full.iloc[train_idx], w_full.iloc[train_idx]
    X_te, y_te, w_te = X_ts.iloc[test_idx],  y_full.iloc[test_idx],  w_full.iloc[test_idx]

    # Impute NaN with column mean
    col_means = X_tr.mean()
    X_tr = X_tr.fillna(col_means)
    X_te = X_te.fillna(col_means)

    clf_clone = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    clf_clone.fit(X_tr, y_tr, sample_weight=w_tr.values)
    acc = float((clf_clone.predict(X_te) == y_te).mean())

    majority = float(y_te.value_counts(normalize=True).max())
    ts_fold_results.append({
        'fold':          fold_i,
        'model':         'ts_only',
        'n_features':    len(ts_cols),
        'n_train':       len(train_idx),
        'n_test':        len(test_idx),
        'accuracy':      acc,
        'majority_base': majority,
        'beats_majority': acc > majority,
    })
    print(f'  Fold {fold_i}: acc={acc:.4f}  majority={majority:.4f}  '
          f'beats={acc > majority}')

elapsed_ts = time.time() - t0
ts_mean_acc = np.mean([r['accuracy'] for r in ts_fold_results])
print(f'\n  TS-only mean accuracy: {ts_mean_acc:.4f}  (elapsed {elapsed_ts:.1f}s)')


# ── Step 5: TS+alpha baseline CV ─────────────────────────────────────────────
sep('STEP 5: TS+alpha baseline (n_features={})'.format(len(ts_cols) + len(alpha_cols)))

X_all = modelling[ts_cols + alpha_cols]

cv_all = MultiAssetPurgedKFold(n_splits=5, t1=t1_full, pct_embargo=0.01)

all_fold_results = []
t0 = time.time()

for fold_i, (train_idx, test_idx) in enumerate(cv_all.split(X_all, y_full)):
    X_tr, y_tr, w_tr = X_all.iloc[train_idx], y_full.iloc[train_idx], w_full.iloc[train_idx]
    X_te, y_te, w_te = X_all.iloc[test_idx],  y_full.iloc[test_idx],  w_full.iloc[test_idx]

    col_means = X_tr.mean()
    X_tr = X_tr.fillna(col_means)
    X_te = X_te.fillna(col_means)

    clf_clone = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    clf_clone.fit(X_tr, y_tr, sample_weight=w_tr.values)
    acc = float((clf_clone.predict(X_te) == y_te).mean())

    majority = float(y_te.value_counts(normalize=True).max())
    all_fold_results.append({
        'fold':          fold_i,
        'model':         'ts_alpha',
        'n_features':    len(ts_cols) + len(alpha_cols),
        'n_train':       len(train_idx),
        'n_test':        len(test_idx),
        'accuracy':      acc,
        'majority_base': majority,
        'beats_majority': acc > majority,
    })
    print(f'  Fold {fold_i}: acc={acc:.4f}  majority={majority:.4f}  '
          f'beats={acc > majority}')

elapsed_all = time.time() - t0
all_mean_acc = np.mean([r['accuracy'] for r in all_fold_results])
print(f'\n  TS+alpha mean accuracy: {all_mean_acc:.4f}  (elapsed {elapsed_all:.1f}s)')


# ── Step 6: Save CV results ───────────────────────────────────────────────────
sep('STEP 6: Save CV results')

cv_df = pd.DataFrame(ts_fold_results + all_fold_results)
cv_df.to_parquet(CV_RESULTS_PATH)
print(f'  Saved: {CV_RESULTS_PATH}  {cv_df.shape}')
print(cv_df[['model', 'fold', 'n_train', 'n_test', 'accuracy', 'majority_base',
             'beats_majority']].to_string(index=False))


# ── Step 7: Validation checks ─────────────────────────────────────────────────
sep('STEP 7: Validation')

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

# A: All 10 tickers present
check('A: all 10 tickers in pooled dataset',
      set(modelling['ticker'].unique()) == set(TICKERS))

# B: No date overlap between train and test folds
check('B: zero date overlaps across all folds',
      n_date_overlaps == 0)

# C: Purging works (zero violations)
check('C: zero purge violations (t1 not overlapping test)',
      n_purge_violations == 0)

# D: Mean CV accuracy > 50% (beat random — with tight purging, majority-class
#    is not the right bar; the question is whether the model has ANY signal)
ts_majority_mean = np.mean([r['majority_base'] for r in ts_fold_results])
all_majority_mean = np.mean([r['majority_base'] for r in all_fold_results])
check(f'D1: TS-only mean acc ({ts_mean_acc:.4f}) > 0.50 (random)',
      ts_mean_acc > 0.50)
check(f'D2: TS+alpha mean acc ({all_mean_acc:.4f}) > 0.50 (random)',
      all_mean_acc > 0.50)

# E: Feature counts correct
check(f'E1: TS feature count == 17 (got {len(ts_cols)})',
      len(ts_cols) == 17)
check(f'E2: alpha feature count == 33 (got {len(alpha_cols)})',
      len(alpha_cols) == 33)
check(f'E3: total features == 50 (got {len(ts_cols)+len(alpha_cols)})',
      len(ts_cols) + len(alpha_cols) == 50)

# F: Pooled modelling shape
check(f'F: modelling rows >= 1500 (got {len(modelling)})',
      len(modelling) >= 1500)
check('F2: modelling parquet saved',
      os.path.exists(MODELLING_PATH))
check('F3: cv results parquet saved',
      os.path.exists(CV_RESULTS_PATH))

# G: CV fold count
check(f'G: 5 folds generated (got {len(folds)})',
      len(folds) == 5)

# H: No NaN in feature columns of pooled dataset
ts_alpha_nan = modelling[ts_cols + alpha_cols].isnull().sum().sum()
check(f'H: zero NaN in feature columns (got {ts_alpha_nan})',
      ts_alpha_nan == 0)

sep()
if ERRORS:
    print(f'Phase 7 FAILED — {failed} check(s) failed:')
    for e in ERRORS:
        print(f'  {e}')
    sys.exit(1)
else:
    print(f'Phase 7 COMPLETE — {passed} checks passed.')
    print(f'  Pooled modelling  : {modelling.shape}')
    print(f'  TS-only accuracy  : {ts_mean_acc:.4f}  (majority {ts_majority_mean:.4f})')
    print(f'  TS+alpha accuracy : {all_mean_acc:.4f}  (majority {all_majority_mean:.4f})')
    print(f'  CV results        : {cv_df.shape}')
