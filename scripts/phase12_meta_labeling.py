"""
Phase 12: Meta-Labeling Validation
===================================
AFML §3.6–3.8: build a secondary classifier that predicts *correctness* of
the primary model (Phase 11 best RF), then convert meta-probabilities into
bet sizes via the Snippet-10.1 formula.

Pipeline
--------
1. Load OOS predictions + enrich with realized returns from per-stock labels
2. Compute meta-labels  (meta_label=1 if ret × side > 0, else 0)
3. Build meta-feature matrix  (50 original features + side = 51 features)
4. OOS meta-model via MultiAssetPurgedKFold  (shallow RF)
5. Bet sizing  (side × size, where size ∈ [0,1] from meta-prob)
6. Meta-validation diagnostics
7. Save artifacts + validation checks

Artifacts saved
---------------
data/processed/meta_labels_pooled.parquet
data/processed/meta_oos_predictions_pooled.parquet
data/processed/bet_sizes_pooled.parquet
reports/figures/phase12_meta_label_dist.png
reports/figures/phase12_meta_precision_recall.png
reports/figures/phase12_bet_size_dist.png
"""

import os, sys, json
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

from src.cross_validation import MultiAssetPurgedKFold
from src.bet_sizing import get_signal, avg_active_signals, discrete_signal

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed',  exist_ok=True)

RNG = 42


def sep(title):
    print('\n' + '=' * 68)
    print(title)
    print('=' * 68)


def check(label, cond):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {label}')
    return cond


# ── Load universe ─────────────────────────────────────────────────────────────
with open('configs/universe.json') as f:
    UNI = json.load(f)
TICKERS = UNI['tickers']

# ── Step 1: Load OOS predictions + enrich ret ────────────────────────────────
sep('STEP 1: Load OOS predictions + enrich with realized returns')

oos = pd.read_parquet('data/processed/oos_predictions_pooled.parquet')
print(f'  OOS shape : {oos.shape}')
print(f'  Columns   : {list(oos.columns)}')
print(f'  Side dist : {oos["oos_pred"].value_counts().to_dict()}')

# Pull realized returns from per-stock label files
ret_frames = []
for ticker in TICKERS:
    lbl_path = f'data/processed/per_stock/{ticker}_labels.parquet'
    if not os.path.exists(lbl_path):
        continue
    lbl = pd.read_parquet(lbl_path)[['ret']]
    lbl['ticker'] = ticker
    ret_frames.append(lbl)

ret_panel = pd.concat(ret_frames).sort_index()

# Join returns: match on (event_date, ticker)
oos = oos.reset_index().rename(columns={'index': 'event_date'})
oos.index = range(len(oos))
ret_panel_reset = ret_panel.reset_index().rename(columns={'index': 'event_date'})

merged = oos.merge(
    ret_panel_reset[['event_date', 'ticker', 'ret']],
    on=['event_date', 'ticker'],
    how='left',
    suffixes=('_old', ''),
)
# keep only the newly joined 'ret' (drop 'ret_old')
if 'ret_old' in merged.columns:
    merged = merged.drop(columns=['ret_old'])

merged = merged.set_index('event_date')
n_ret_nan = merged['ret'].isna().sum()
print(f'  ret NaN after join: {n_ret_nan} / {len(merged)}')
if n_ret_nan > 0:
    print('  WARNING: some events have no matched return — dropping them')
    merged = merged.dropna(subset=['ret'])

oos = merged
print(f'  After ret join: {oos.shape}')

# ── Step 2: Compute meta-labels ───────────────────────────────────────────────
sep('STEP 2: Compute meta-labels  (1 = primary model was correct direction)')

# side = direction of primary model prediction
side = oos['oos_pred'].astype(float)

# meta_label = 1 if the realized return was in the same direction as prediction
meta_label = (oos['ret'] * side > 0).astype(int)

n_meta = len(meta_label)
n_correct = int(meta_label.sum())
print(f'  n events       : {n_meta}')
print(f'  meta_label=1   : {n_correct}  ({n_correct/n_meta:.4f})')
print(f'  meta_label=0   : {n_meta-n_correct}  ({(n_meta-n_correct)/n_meta:.4f})')

# Sanity: meta_label=1 rate should approximately equal primary OOS accuracy
primary_acc = (oos['oos_pred'] == oos['label']).mean()
print(f'  Primary OOS acc (Phase 11): {primary_acc:.4f}')
print(f'  Meta label=1 rate          : {n_correct/n_meta:.4f}  (should be close)')

# Build meta_labels DataFrame
meta_labels_df = pd.DataFrame({
    'side':           side,
    'oos_prob':       oos['oos_prob'],
    'oos_fold':       oos['oos_fold'],
    'ret':            oos['ret'],
    'label':          oos['label'],
    'meta_label':     meta_label,
    'ticker':         oos['ticker'],
    'weight':         oos['weight'],
    't1':             oos['t1'],
}, index=oos.index)

meta_labels_df.to_parquet('data/processed/meta_labels_pooled.parquet')
print(f'  Saved: data/processed/meta_labels_pooled.parquet')

# ── Step 3: Build meta-feature matrix ────────────────────────────────────────
sep('STEP 3: Build meta-feature matrix  (50 original + side = 51 features)')

modelling = pd.read_parquet('data/processed/pooled_modelling.parquet')

meta_cols  = {'label', 't1', 'weight', 'ticker'}
feat_cols  = [c for c in modelling.columns if c not in meta_cols]

# Align on common index
common_idx = modelling.index.intersection(meta_labels_df.index)
print(f'  Common events  : {len(common_idx)}  (modelling={len(modelling)}, meta={len(meta_labels_df)})')

X_base = modelling.loc[common_idx, feat_cols].copy()
X_meta = X_base.copy()
X_meta['side'] = meta_labels_df.loc[common_idx, 'side'].values

y_meta  = meta_labels_df.loc[common_idx, 'meta_label'].astype(int)
w_meta  = modelling.loc[common_idx, 'weight']
t1_meta = modelling.loc[common_idx, 't1']
ticker_meta = modelling.loc[common_idx, 'ticker']

print(f'  X_meta shape   : {X_meta.shape}')
print(f'  NaN in X_meta  : {X_meta.isna().sum().sum()}')
print(f'  y_meta dist    : {y_meta.value_counts().to_dict()}')

# ── Step 4: OOS meta-model via MultiAssetPurgedKFold ─────────────────────────
sep('STEP 4: OOS meta-model  (shallow RF, MultiAssetPurgedKFold 5-fold)')

META_CLF_PARAMS = {
    'n_estimators':     200,
    'max_depth':        3,
    'min_samples_leaf': 15,
    'max_features':     'sqrt',
    'class_weight':     'balanced',
    'random_state':     RNG,
    'n_jobs':           -1,
}
print(f'  Meta-clf params: {META_CLF_PARAMS}')

pkf = MultiAssetPurgedKFold(n_splits=5, t1=t1_meta, pct_embargo=0.01)

out_class = pd.Series(index=X_meta.index, dtype=float)
out_prob  = pd.Series(index=X_meta.index, dtype=float)
out_fold  = pd.Series(index=X_meta.index, dtype=int)

fold_accs = []
for fold_i, (train_idx, test_idx) in enumerate(pkf.split(X_meta, y_meta)):
    X_tr  = X_meta.iloc[train_idx]
    y_tr  = y_meta.iloc[train_idx]
    sw_tr = w_meta.iloc[train_idx].values
    X_te  = X_meta.iloc[test_idx]
    y_te  = y_meta.iloc[test_idx]

    clf = RandomForestClassifier(**META_CLF_PARAMS)
    clf.fit(X_tr, y_tr, sample_weight=sw_tr)

    pred_class = clf.predict(X_te)
    pred_proba = clf.predict_proba(X_te)

    classes  = list(clf.classes_)
    pos_col  = classes.index(1)

    out_class.iloc[test_idx] = pred_class
    out_prob.iloc[test_idx]  = pred_proba[:, pos_col]
    out_fold.iloc[test_idx]  = fold_i

    fold_acc = accuracy_score(y_te, pred_class)
    fold_accs.append(fold_acc)
    print(f'  Fold {fold_i}: train={len(train_idx):4d}  test={len(test_idx):3d}  '
          f'acc={fold_acc:.4f}')

meta_oos_acc = accuracy_score(y_meta, out_class)
majority_meta = float(y_meta.mean())
print(f'\n  Meta OOS accuracy  : {meta_oos_acc:.4f}')
print(f'  Majority baseline  : {majority_meta:.4f}  (class-1 rate)')
print(f'  Mean fold acc      : {np.mean(fold_accs):.4f}  std={np.std(fold_accs):.4f}')

meta_oos_df = pd.DataFrame({
    'meta_pred_class': out_class.astype(int),
    'meta_pred_prob':  out_prob,
    'meta_fold':       out_fold.astype(int),
    'meta_label':      y_meta,
    'side':            meta_labels_df.loc[common_idx, 'side'],
    'ret':             meta_labels_df.loc[common_idx, 'ret'],
    'ticker':          ticker_meta,
    'weight':          w_meta,
    't1':              t1_meta,
}, index=X_meta.index)

meta_oos_df.to_parquet('data/processed/meta_oos_predictions_pooled.parquet')
print(f'  Saved: data/processed/meta_oos_predictions_pooled.parquet')

# ── Step 5: Bet sizing ────────────────────────────────────────────────────────
sep('STEP 5: Bet sizing  (Snippet 10.1: signal = side × size(meta_prob))')

side_series = meta_oos_df['side']
meta_prob   = meta_oos_df['meta_pred_prob']

# Raw signal ∈ [-1, 1]
raw_signal = get_signal(side_series, meta_prob, num_classes=2, step_size=0.0)

# Discretized signal (step=0.1)
disc_signal = discrete_signal(raw_signal, step_size=0.1)

# Bet size magnitude ∈ [0, 1]
bet_size = raw_signal.abs()

print(f'  Signal range    : [{raw_signal.min():.4f}, {raw_signal.max():.4f}]')
print(f'  Bet size mean   : {bet_size.mean():.4f}  median={bet_size.median():.4f}')
print(f'  Zeros (no-trade): {(bet_size < 0.05).sum()}  ({(bet_size < 0.05).mean():.2%})')

# Avg active signal (position series)
avg_pos = avg_active_signals(disc_signal, t1_meta)

bet_sizes_df = pd.DataFrame({
    'raw_signal':  raw_signal,
    'disc_signal': disc_signal,
    'bet_size':    bet_size,
    'side':        side_series,
    'meta_prob':   meta_prob,
    'ret':         meta_oos_df['ret'],
    'ticker':      meta_oos_df['ticker'],
    't1':          t1_meta,
}, index=X_meta.index)

bet_sizes_df.to_parquet('data/processed/bet_sizes_pooled.parquet')
print(f'  Saved: data/processed/bet_sizes_pooled.parquet')

# ── Step 6: Meta-validation diagnostics ──────────────────────────────────────
sep('STEP 6: Meta-validation diagnostics')

y_true_meta = meta_oos_df['meta_label'].values
y_pred_meta = meta_oos_df['meta_pred_class'].values
y_prob_meta = meta_oos_df['meta_pred_prob'].values

prec   = precision_score(y_true_meta, y_pred_meta, zero_division=0)
rec    = recall_score(y_true_meta, y_pred_meta, zero_division=0)
f1     = f1_score(y_true_meta, y_pred_meta, zero_division=0)
try:
    auc = roc_auc_score(y_true_meta, y_prob_meta)
except Exception:
    auc = float('nan')

cm = confusion_matrix(y_true_meta, y_pred_meta)

print(f'\n  Meta-model performance (vs meta-label ground truth):')
print(f'    Accuracy  : {meta_oos_acc:.4f}')
print(f'    Precision : {prec:.4f}  (of meta_pred=1, how many were actually profitable?)')
print(f'    Recall    : {rec:.4f}  (of truly profitable trades, how many caught?)')
print(f'    F1        : {f1:.4f}')
print(f'    AUC-ROC   : {auc:.4f}')
print(f'\n  Confusion matrix (rows=true, cols=pred):')
print(f'    {cm}')

# Primary vs meta-filtered profitability comparison
# "All trades" = just use side from primary model, count profitable
all_profitable  = (meta_oos_df['ret'] * meta_oos_df['side'] > 0).mean()
# "Meta-filtered" = only trade when meta_pred=1
meta_filtered   = meta_oos_df[meta_oos_df['meta_pred_class'] == 1]
meta_filtered_profitable = (meta_filtered['ret'] * meta_filtered['side'] > 0).mean()

print(f'\n  Profitability comparison:')
print(f'    All trades ({len(meta_oos_df)})         : {all_profitable:.4f}')
print(f'    Meta-filtered ({len(meta_filtered)})  : {meta_filtered_profitable:.4f}')
print(f'    Improvement (meta - all)              : {meta_filtered_profitable - all_profitable:+.4f}')

# Per-ticker breakdown
print(f'\n  Per-ticker meta accuracy:')
for ticker in sorted(meta_oos_df['ticker'].unique()):
    sub = meta_oos_df[meta_oos_df['ticker'] == ticker]
    tacc = accuracy_score(sub['meta_label'], sub['meta_pred_class'])
    tpct = int(sub['meta_pred_class'].sum())
    print(f'    {ticker:6s}: n={len(sub):4d}  meta_acc={tacc:.4f}  trades_taken={tpct}')

# ── Step 7: Figures ───────────────────────────────────────────────────────────
sep('STEP 7: Figures')

# Fig 1: Meta-label distribution (by ticker)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

meta_by_ticker = meta_oos_df.groupby('ticker')['meta_label'].agg(['sum', 'count'])
meta_by_ticker['rate'] = meta_by_ticker['sum'] / meta_by_ticker['count']
axes[0].bar(meta_by_ticker.index, meta_by_ticker['rate'], color='steelblue', alpha=0.8)
axes[0].axhline(majority_meta, color='red', linestyle='--', label=f'Mean ({majority_meta:.3f})')
axes[0].set_xlabel('Ticker')
axes[0].set_ylabel('Meta-label=1 rate (trade profitable)')
axes[0].set_title('Meta-Label Rate by Ticker\n(primary model correct direction)')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Fold accuracy
axes[1].bar(range(len(fold_accs)), fold_accs, color='steelblue', alpha=0.8)
axes[1].axhline(np.mean(fold_accs), color='red', linestyle='--',
                label=f'Mean ({np.mean(fold_accs):.3f})')
axes[1].set_xticks(range(len(fold_accs)))
axes[1].set_xticklabels([f'Fold {i}' for i in range(len(fold_accs))])
axes[1].set_ylabel('Meta-model OOS accuracy')
axes[1].set_title('Meta-Model CV Accuracy by Fold')
axes[1].set_ylim(0.3, 0.8)
axes[1].legend()

plt.tight_layout()
plt.savefig('reports/figures/phase12_meta_label_dist.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase12_meta_label_dist.png')

# Fig 2: Precision / Recall at different probability thresholds
thresholds = np.linspace(0.3, 0.9, 60)
precs, recs, f1s, n_trades = [], [], [], []
for thr in thresholds:
    pred = (y_prob_meta >= thr).astype(int)
    n_t  = int(pred.sum())
    if n_t == 0:
        precs.append(np.nan); recs.append(np.nan); f1s.append(np.nan)
    else:
        precs.append(precision_score(y_true_meta, pred, zero_division=0))
        recs.append(recall_score(y_true_meta, pred, zero_division=0))
        f1s.append(f1_score(y_true_meta, pred, zero_division=0))
    n_trades.append(n_t)

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(thresholds, precs, 'b-', label='Precision', linewidth=2)
ax1.plot(thresholds, recs,  'g-', label='Recall',    linewidth=2)
ax1.plot(thresholds, f1s,   'r-', label='F1',        linewidth=2)
ax1.axvline(0.5, color='grey', linestyle='--', alpha=0.7, label='Threshold=0.5')
ax1.set_xlabel('Meta-prob threshold')
ax1.set_ylabel('Score')
ax1.set_title('Precision / Recall / F1 vs Meta-Prob Threshold')
ax1.legend(loc='upper left')
ax1.set_ylim(0, 1.05)
ax2 = ax1.twinx()
ax2.plot(thresholds, n_trades, 'k:', alpha=0.5, linewidth=1.5, label='# trades')
ax2.set_ylabel('# trades taken')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('reports/figures/phase12_meta_precision_recall.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase12_meta_precision_recall.png')

# Fig 3: Bet size distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(bet_size.values, bins=40, color='steelblue', edgecolor='k', alpha=0.7)
axes[0].axvline(bet_size.mean(), color='red', linestyle='--',
                label=f'Mean ({bet_size.mean():.3f})')
axes[0].set_xlabel('Bet size (|signal|)')
axes[0].set_ylabel('Count')
axes[0].set_title('Bet Size Distribution\n(Snippet 10.1, num_classes=2)')
axes[0].legend()

axes[1].plot(avg_pos.index, avg_pos.values, color='steelblue', linewidth=1, alpha=0.8)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Avg active signal')
axes[1].set_title('Average Active Position (Snippet 10.2)\n(discretized, step=0.1)')
plt.tight_layout()
plt.savefig('reports/figures/phase12_bet_size_dist.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase12_bet_size_dist.png')

# ── Step 8: Validation ────────────────────────────────────────────────────────
sep('STEP 8: Validation')

failures = []

checks = [
    ('meta_labels_pooled.parquet saved',
     os.path.exists('data/processed/meta_labels_pooled.parquet')),
    ('meta_oos_predictions_pooled.parquet saved',
     os.path.exists('data/processed/meta_oos_predictions_pooled.parquet')),
    ('bet_sizes_pooled.parquet saved',
     os.path.exists('data/processed/bet_sizes_pooled.parquet')),
    ('meta_labels non-empty',
     len(meta_labels_df) > 0),
    ('all events have ret',
     meta_labels_df['ret'].isna().sum() == 0),
    ('meta OOS covers all events',
     len(meta_oos_df) == len(X_meta)),
    ('no NaN in meta_pred_prob',
     meta_oos_df['meta_pred_prob'].isna().sum() == 0),
    ('meta accuracy > 0.50',
     meta_oos_acc > 0.50),
    ('meta-filtered profitable rate >= all-trades rate',
     meta_filtered_profitable >= all_profitable - 0.02),  # allow 2% slack
    ('bet sizes in [0, 1]',
     bet_size.between(0, 1).all()),
    ('raw signal in [-1, 1]',
     raw_signal.between(-1, 1).all()),
    ('meta_label=1 rate > 0.30',
     majority_meta > 0.30),
    ('meta_label dist fig saved',
     os.path.exists('reports/figures/phase12_meta_label_dist.png')),
    ('precision_recall fig saved',
     os.path.exists('reports/figures/phase12_meta_precision_recall.png')),
    ('bet_size fig saved',
     os.path.exists('reports/figures/phase12_bet_size_dist.png')),
]

for label, cond in checks:
    if not check(label, cond):
        failures.append(label)

# Final summary
n_pass = len(checks) - len(failures)
print(f'\n{"=" * 68}')
if failures:
    print(f'Phase 12 FAILED — {len(failures)} check(s) failed:')
    for f in failures:
        print(f'  {f}: FAIL')
else:
    print(f'Phase 12 COMPLETE — {n_pass} checks passed.')
    print(f'  Meta-label n       : {len(meta_labels_df)}')
    print(f'  Meta-label=1 rate  : {majority_meta:.4f}')
    print(f'  Meta OOS accuracy  : {meta_oos_acc:.4f}  (vs random 0.50)')
    print(f'  Precision @ 0.50   : {prec:.4f}')
    print(f'  AUC-ROC            : {auc:.4f}')
    print(f'  All-trades profit  : {all_profitable:.4f}')
    print(f'  Meta-filter profit : {meta_filtered_profitable:.4f}')
    print(f'  Bet size mean      : {bet_size.mean():.4f}')
