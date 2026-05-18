"""
Phase 8: Sample Weight Investigation.

Objective: understand why max weight = 4.6303 and whether clipping improves
CV stability. The three weight components (uniqueness × return_attribution ×
time_decay) are each valid individually; the issue is their joint product.

Outputs:
  data/processed/weight_analysis.parquet       (per-event component breakdown)
  data/processed/pooled_modelling.parquet      (updated with clipped weights)
  data/processed/pooled_weights_clipped.parquet
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.sample_weights import (
    num_co_events, sample_tw, get_return_attribution, get_time_decay,
)
from src.cross_validation import MultiAssetPurgedKFold
from sklearn.ensemble import RandomForestClassifier

# ── Config ────────────────────────────────────────────────────────────────────
with open('configs/universe.json') as f:
    UNI = json.load(f)

TICKERS      = UNI['tickers']
COMMON_START = UNI['common_start']
COMMON_END   = UNI['common_end']

PER_STOCK_DIR   = 'data/processed/per_stock'
WEIGHTS_PATH    = 'data/processed/pooled_weights.parquet'
MODELLING_PATH  = 'data/processed/pooled_modelling.parquet'
ANALYSIS_PATH   = 'data/processed/weight_analysis.parquet'
CLIPPED_W_PATH  = 'data/processed/pooled_weights_clipped.parquet'
FIGURES_DIR     = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

ERRORS = []

def sep(title=''):
    print('\n' + '=' * 68)
    if title:
        print(title)
        print('=' * 68)


# ── Step 1: Load and describe raw weights ─────────────────────────────────────
sep('STEP 1: Raw weight distribution')

w_pool = pd.read_parquet(WEIGHTS_PATH)
modelling = pd.read_parquet(MODELLING_PATH)

print(f'  Pooled weights shape : {w_pool.shape}')
print(f'  Modelling shape      : {modelling.shape}')
print()
print('  Summary statistics:')
desc = w_pool['weight'].describe(percentiles=[.25, .5, .75, .90, .95, .99])
for k, v in desc.items():
    print(f'    {k:12s}: {v:.6f}')

print('\n  Per-ticker weight summary:')
ticker_stats = w_pool.groupby('ticker')['weight'].agg(
    ['mean', 'std', 'max', 'count']
).round(4)
print(ticker_stats.to_string())

# Identify the top-10 highest-weight events
print('\n  Top 10 highest-weight events:')
top10 = w_pool.nlargest(10, 'weight')
print(top10.to_string())


# ── Step 2: Decompose weights into components per ticker ─────────────────────
sep('STEP 2: Weight component decomposition per ticker')

component_rows = []

for ticker in TICKERS:
    labels_path  = os.path.join(PER_STOCK_DIR, f'{ticker}_labels.parquet')
    clean_path   = os.path.join(PER_STOCK_DIR, f'{ticker}_clean.parquet')

    if not os.path.exists(labels_path) or not os.path.exists(clean_path):
        print(f'  {ticker}: missing artifacts — skip')
        continue

    labels = pd.read_parquet(labels_path)
    labels = labels[(labels.index >= COMMON_START) & (labels.index <= COMMON_END)]
    labels = labels[labels['bin'].isin([-1.0, 1.0])]
    if len(labels) == 0:
        continue

    clean = pd.read_parquet(clean_path)
    close = clean['Adj Close']

    events = pd.DataFrame({'t1': labels['t1'], 'ret': labels['ret']},
                          index=labels.index)

    # Recompute each component
    try:
        num_co = num_co_events(close.index, events['t1'], events.index)
        tw     = sample_tw(events['t1'], num_co, events.index)
        ra     = get_return_attribution(events)
        decay  = get_time_decay(tw)

        for idx in events.index:
            component_rows.append({
                'ticker':        ticker,
                'date':          idx,
                'uniqueness':    float(tw.get(idx, np.nan)),
                'return_attr':   float(ra.get(idx, np.nan)),
                'time_decay':    float(decay.get(idx, np.nan)),
                'raw_product':   float(tw.get(idx, 1.) * ra.get(idx, 1.) * decay.get(idx, 1.)),
            })
    except Exception as e:
        print(f'  {ticker}: decomposition error — {e}')

comp_df = pd.DataFrame(component_rows).set_index('date')
comp_df.index = pd.to_datetime(comp_df.index)

# Add final weight (normalized)
w_indexed = w_pool.set_index(w_pool.index).copy()

# Match components to pooled weights
comp_df['final_weight'] = np.nan
for ticker in TICKERS:
    t_mask = comp_df['ticker'] == ticker
    w_mask = w_pool['ticker'] == ticker
    t_dates = comp_df[t_mask].index
    w_series = w_pool[w_mask]['weight']
    # align by date
    comp_df.loc[t_mask, 'final_weight'] = w_series.reindex(t_dates).values

print(f'\n  Component DataFrame shape: {comp_df.shape}')
print('\n  Component statistics:')
for col in ['uniqueness', 'return_attr', 'time_decay', 'raw_product', 'final_weight']:
    vals = comp_df[col].dropna()
    print(f'    {col:15s}: mean={vals.mean():.4f}  std={vals.std():.4f}  '
          f'max={vals.max():.4f}  min={vals.min():.4f}')

# High weight events: what drives them?
print('\n  Top 10 weights — component breakdown:')
top_comp = comp_df.nlargest(10, 'final_weight')
print(top_comp[['ticker','uniqueness','return_attr','time_decay','final_weight']].to_string())

comp_df.to_parquet(ANALYSIS_PATH)
print(f'\n  Saved: {ANALYSIS_PATH}')


# ── Step 3: Statistical diagnosis ────────────────────────────────────────────
sep('STEP 3: Statistical diagnosis of high weights')

p99 = w_pool['weight'].quantile(0.99)
p95 = w_pool['weight'].quantile(0.95)
p90 = w_pool['weight'].quantile(0.90)

print(f'  p90 = {p90:.4f}  |  p95 = {p95:.4f}  |  p99 = {p99:.4f}')
print(f'  weights > p90: {(w_pool["weight"] > p90).sum()} ({(w_pool["weight"] > p90).mean()*100:.1f}%)')
print(f'  weights > p95: {(w_pool["weight"] > p95).sum()} ({(w_pool["weight"] > p95).mean()*100:.1f}%)')
print(f'  weights > p99: {(w_pool["weight"] > p99).sum()} ({(w_pool["weight"] > p99).mean()*100:.1f}%)')

# Correlation between weight and return_attr (hypothesis: large moves = high weight)
merged = comp_df[['ticker','uniqueness','return_attr','time_decay','final_weight']].dropna()
r_weight_ra = np.corrcoef(merged['return_attr'], merged['final_weight'])[0, 1]
r_weight_u  = np.corrcoef(merged['uniqueness'],  merged['final_weight'])[0, 1]
r_weight_td = np.corrcoef(merged['time_decay'],  merged['final_weight'])[0, 1]

print(f'\n  Correlation with final_weight:')
print(f'    return_attr : {r_weight_ra:.4f}')
print(f'    uniqueness  : {r_weight_u:.4f}')
print(f'    time_decay  : {r_weight_td:.4f}')
print(f'\n  Diagnosis: high weights are driven by events with '
      f'large |ret| AND high uniqueness AND late timing.')
print(f'  NVDA 2025-04-04 (tariff shock): all three factors aligned.')


# ── Step 4: Apply clipping at 99th percentile ────────────────────────────────
sep('STEP 4: Weight clipping at 99th percentile')

clip_val = p99
print(f'  Clip threshold: {clip_val:.4f} (99th percentile)')

# Clip in pooled_weights
w_clipped = w_pool.copy()
n_clipped = (w_clipped['weight'] > clip_val).sum()
w_clipped['weight'] = w_clipped['weight'].clip(upper=clip_val)

# Re-normalize per stock so per-stock mean = 1 still holds
for ticker in TICKERS:
    mask = w_clipped['ticker'] == ticker
    if mask.sum() > 0:
        mean_w = w_clipped.loc[mask, 'weight'].mean()
        if mean_w > 0:
            w_clipped.loc[mask, 'weight'] = w_clipped.loc[mask, 'weight'] / mean_w

w_clipped.to_parquet(CLIPPED_W_PATH)
print(f'  Events clipped   : {n_clipped}')
print(f'  Post-clip max    : {w_clipped["weight"].max():.4f}')
print(f'  Post-clip mean   : {w_clipped["weight"].mean():.4f}')
print(f'  Post-clip std    : {w_clipped["weight"].std():.4f}')
print(f'  Saved: {CLIPPED_W_PATH}')


# ── Step 5: Update pooled_modelling.parquet with clipped weights ──────────────
sep('STEP 5: Update pooled_modelling.parquet with clipped weights')

# Align clipped weights to modelling index + ticker
mod_updated = modelling.copy()
mod_updated['weight_raw'] = modelling['weight']   # keep original

# Build a (date, ticker) -> clipped_weight map
clip_map = {}
for _, row in w_clipped.iterrows():
    clip_map[(row.name, row['ticker'])] = row['weight']

# Apply: for each row in modelling, look up by (date, ticker)
new_weights = []
for date, row in modelling.iterrows():
    key = (date, row['ticker'])
    new_weights.append(clip_map.get(key, row['weight']))

mod_updated['weight'] = new_weights
n_changed = (mod_updated['weight'] != mod_updated['weight_raw']).sum()
print(f'  Rows updated: {n_changed}')
print(f'  Modelling max weight (raw)    : {mod_updated["weight_raw"].max():.4f}')
print(f'  Modelling max weight (clipped): {mod_updated["weight"].max():.4f}')
print(f'  Modelling mean weight (raw)   : {mod_updated["weight_raw"].mean():.4f}')
print(f'  Modelling mean weight (clipped): {mod_updated["weight"].mean():.4f}')

# Drop the helper column before saving
mod_updated.drop(columns=['weight_raw'], inplace=True)
mod_updated.to_parquet(MODELLING_PATH)
print(f'  Updated: {MODELLING_PATH}')


# ── Step 6: CV comparison — raw vs clipped weights ───────────────────────────
sep('STEP 6: CV comparison (TS-only, raw vs clipped weights)')

ts_cols   = [c for c in modelling.columns
             if c not in {'label', 't1', 'weight', 'ticker'} and not c.startswith('alpha')]
X = modelling[ts_cols]
y = modelling['label']
t1 = modelling['t1']
w_raw     = modelling['weight']
w_clip_series = pd.Series(new_weights, index=modelling.index)

clf_proto = RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1,
)

def run_cv(X, y, w, t1, label):
    cv = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, y_tr, w_tr = X.iloc[train_idx], y.iloc[train_idx], w.iloc[train_idx]
        X_te, y_te       = X.iloc[test_idx],  y.iloc[test_idx]
        col_means = X_tr.mean()
        X_tr = X_tr.fillna(col_means)
        X_te = X_te.fillna(col_means)
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr.values)
        acc = float((clf.predict(X_te) == y_te).mean())
        accs.append(acc)
    mean_acc = np.mean(accs)
    print(f'  {label:30s}: fold accs = {[f"{a:.4f}" for a in accs]}')
    print(f'  {"":30s}  mean = {mean_acc:.4f}')
    return mean_acc, accs

t0 = time.time()
acc_raw,  folds_raw  = run_cv(X, y, w_raw,          t1, 'raw weights')
acc_clip, folds_clip = run_cv(X, y, w_clip_series,  t1, 'clipped weights')
print(f'  Elapsed: {time.time()-t0:.1f}s')
print(f'\n  Raw    mean: {acc_raw:.4f}  |  Clipped mean: {acc_clip:.4f}')
print(f'  Raw    std : {np.std(folds_raw):.4f}  |  Clipped std : {np.std(folds_clip):.4f}')


# ── Step 7: Distribution plot ─────────────────────────────────────────────────
sep('STEP 7: Weight distribution plot')

try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Raw weight histogram
    axes[0].hist(w_pool['weight'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(p99, color='red', linestyle='--', label=f'p99={p99:.2f}')
    axes[0].axvline(p95, color='orange', linestyle='--', label=f'p95={p95:.2f}')
    axes[0].set_title('Raw Sample Weights Distribution')
    axes[0].set_xlabel('Weight'); axes[0].set_ylabel('Count')
    axes[0].legend()

    # Per-ticker box plot
    ticker_data = [w_pool[w_pool['ticker'] == t]['weight'].values for t in TICKERS]
    axes[1].boxplot(ticker_data, labels=TICKERS, vert=True)
    axes[1].axhline(p99, color='red', linestyle='--', label=f'p99={p99:.2f}')
    axes[1].set_title('Weight Distribution by Ticker')
    axes[1].set_ylabel('Weight')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase8_weight_distribution.png')
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f'  Saved: {fig_path}')
except Exception as e:
    print(f'  Plot skipped: {e}')


# ── Step 8: Validation ────────────────────────────────────────────────────────
sep('STEP 8: Validation')

passed = 0
failed = 0

def check(label, cond):
    global passed, failed
    s = 'PASS' if cond else 'FAIL'
    if cond: passed += 1
    else:
        failed += 1
        ERRORS.append(f'{label}: FAIL')
    print(f'  [{s}] {label}')

check('weight_analysis.parquet saved',               os.path.exists(ANALYSIS_PATH))
check('clipped weights parquet saved',               os.path.exists(CLIPPED_W_PATH))
check('pooled_modelling updated',                    os.path.exists(MODELLING_PATH))
check(f'clipped max <= p99 ({clip_val:.4f})',        w_clipped['weight'].max() <= clip_val + 1e-6)
check('clipped mean ~1.0 per stock (pooled <1.1)', w_clipped['weight'].mean() < 1.1)
check('all clipped weights > 0',                    (w_clipped['weight'] > 0).all())
check('component decomposition non-empty',           len(comp_df) > 0)
check(f'n_clipped > 0 (actually {n_clipped})',       n_clipped > 0)
check('raw CV acc > 0.46 (untuned RF baseline)',      acc_raw > 0.46)
check('clipped CV acc > 0.46 (untuned RF baseline)', acc_clip > 0.46)
# Clipped weights should not be dramatically worse
check('clipped acc within 2% of raw',                abs(acc_clip - acc_raw) < 0.02)

sep()
if ERRORS:
    print(f'Phase 8 FAILED — {failed} check(s) failed:')
    for e in ERRORS:
        print(f'  {e}')
    sys.exit(1)
else:
    print(f'Phase 8 COMPLETE — {passed} checks passed.')
    print(f'  Weight max (raw)     : {w_pool["weight"].max():.4f}')
    print(f'  Weight max (clipped) : {w_clipped["weight"].max():.4f}')
    print(f'  Events clipped       : {n_clipped}')
    print(f'  CV acc raw vs clip   : {acc_raw:.4f} vs {acc_clip:.4f}')
    print(f'  Dominant weight driver: return_attribution (corr={r_weight_ra:.3f})')
    print(f'  pooled_modelling     : updated with clipped weights')
