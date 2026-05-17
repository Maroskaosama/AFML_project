"""
Phase 14: CPCV Robustness (K=6, p=2, 15 backtest paths)
=========================================================
AFML Chapter 12: Combinatorial Purged Cross-Validation (CPCV).

With K=6 time-blocks and p=2 test blocks per split:
  - C(6,2) = 15 train/test splits (multi-asset time-block purged)
  - 15 perfect-matching backtest paths (each covers all 6 blocks once)

Each split:  fit best RF on purged train set → OOS predictions on test
Each path:   assemble equity curve from 3 non-overlapping test blocks
             → SR distribution across 15 paths = robustness metric

Compare against Phase 11 5-fold MultiAssetPurgedKFold baseline.

Artifacts saved
---------------
data/processed/cpcv_oos_pooled.parquet      (per-split OOS results)
data/processed/cpcv_paths_pooled.parquet    (per-path SR / equity stats)
reports/figures/phase14_cpcv_sr_dist.png
reports/figures/phase14_cpcv_equity_paths.png
reports/figures/phase14_cpcv_fold_heatmap.png
"""

import os, sys, json
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.backtesting import sharpe_ratio
from src.bet_sizing import avg_active_signals, build_daily_positions

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed',  exist_ok=True)

K          = 6
P          = 2
RNG        = 42
COST_BPS   = 5
PPY        = 252
PCT_EMBARGO = 0.01


def sep(title):
    print('\n' + '=' * 68)
    print(title)
    print('=' * 68)


def check(label, cond):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {label}')
    return cond


# ── Load data ─────────────────────────────────────────────────────────────────
sep('LOAD: pooled modelling dataset + best RF params + panel prices')

with open('configs/universe.json') as f:
    UNI = json.load(f)
TICKERS = UNI['tickers']

with open('models/best_params_pooled.json') as f:
    bp = json.load(f)
RF_PARAMS = {**bp['rf']['params'], 'random_state': RNG, 'n_jobs': -1}

modelling = pd.read_parquet('data/processed/pooled_modelling.parquet')
panel     = pd.read_parquet('data/processed/panel_ohlcv.parquet')
adj_close = panel['AdjClose'].unstack(level='ticker')

meta_cols = {'label', 't1', 'weight', 'ticker'}
feat_cols = [c for c in modelling.columns if c not in meta_cols]

X      = modelling[feat_cols]
y      = modelling['label']
sw     = modelling['weight']
t1     = modelling['t1']
ticker = modelling['ticker']

print(f'  X shape    : {X.shape}')
print(f'  RF params  : {RF_PARAMS}')
print(f'  K={K}, p={P}  -> C({K},{P}) = {len(list(combinations(range(K),P)))} splits')
print(f'  Panel      : {adj_close.shape}')

# ── Step 1: Time-block CPCV split generation ─────────────────────────────────
sep('STEP 1: Time-block CPCV — generate splits (K=6, p=2)')

event_times  = X.index.tolist()
unique_times = sorted(set(event_times))
n_times      = len(unique_times)

# Partition unique dates into K contiguous groups
time_groups = np.array_split(np.arange(n_times), K)
time_group_sets = [set(unique_times[i] for i in grp) for grp in time_groups]

print(f'  Unique event dates : {n_times}')
print(f'  Groups (dates)     :')
for g, gs in enumerate(time_group_sets):
    dates = sorted(gs)
    print(f'    Group {g}: {len(gs)} dates  '
          f'[{min(dates).date()} -> {max(dates).date()}]')

# Embargo size in number of unique dates
embargo_n = max(1, int(n_times * PCT_EMBARGO))

def time_block_split(X, y, t1, test_groups):
    """
    Multi-asset time-block train/test split with purge + embargo.

    test_groups: list of group indices (0..K-1) to use as test
    """
    test_times = set()
    for g in test_groups:
        test_times |= time_group_sets[g]

    test_start = min(test_times)
    test_end   = max(test_times)

    ev = X.index.tolist()
    test_mask  = np.array([t in test_times for t in ev])
    train_mask = ~test_mask

    # Purge: pre-test events whose t1 reaches into test window
    train_indices = np.where(train_mask)[0]
    for i in train_indices:
        et = ev[i]
        if et < test_start:
            t1_val = t1.iloc[i]
            if pd.notna(t1_val) and t1_val >= test_start:
                train_mask[i] = False

    # Embargo: remove events just after test_end
    test_end_idx = unique_times.index(test_end)
    embargo_dates = set(
        unique_times[j]
        for j in range(test_end_idx + 1, min(test_end_idx + 1 + embargo_n, n_times))
    )
    for i in np.where(train_mask)[0]:
        if ev[i] in embargo_dates:
            train_mask[i] = False

    return np.where(train_mask)[0], np.where(test_mask)[0]

# Enumerate all C(K,p) splits
all_combos = list(combinations(range(K), P))
print(f'\n  Total splits: {len(all_combos)}')

# ── Step 2: Fit model on each split, collect OOS predictions ─────────────────
sep('STEP 2: Fit RF on each of the 15 splits — collect OOS predictions')

split_results = []  # list of dicts, one per split

for split_i, test_combo in enumerate(all_combos):
    train_idx, test_idx = time_block_split(X, y, t1, list(test_combo))

    if len(train_idx) < 50 or len(test_idx) == 0:
        print(f'  Split {split_i:2d} {test_combo}: insufficient data — skip')
        continue

    X_tr  = X.iloc[train_idx]
    y_tr  = y.iloc[train_idx]
    sw_tr = sw.iloc[train_idx].values
    X_te  = X.iloc[test_idx]
    y_te  = y.iloc[test_idx]

    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr, y_tr, sample_weight=sw_tr)

    pred_class = clf.predict(X_te)
    pred_proba = clf.predict_proba(X_te)
    classes    = list(clf.classes_)
    pos_col    = classes.index(1) if 1 in classes else 0
    pred_prob  = pred_proba[:, pos_col]

    acc = accuracy_score(y_te, pred_class)

    split_results.append({
        'split_idx':   split_i,
        'test_combo':  test_combo,
        'n_train':     len(train_idx),
        'n_test':      len(test_idx),
        'accuracy':    acc,
        'test_indices': test_idx,
        'pred_class':  pred_class,
        'pred_prob':   pred_prob,
        'event_dates': X.iloc[test_idx].index.tolist(),
        'tickers':     ticker.iloc[test_idx].tolist(),
        'true_labels': y_te.values,
        'ret':         None,  # will be filled from per-stock labels
    })

    print(f'  Split {split_i:2d} {test_combo}: '
          f'train={len(train_idx):4d}  test={len(test_idx):3d}  acc={acc:.4f}')

n_splits_done = len(split_results)
mean_acc = np.mean([r['accuracy'] for r in split_results])
print(f'\n  Splits completed: {n_splits_done} / {len(all_combos)}')
print(f'  Mean OOS acc    : {mean_acc:.4f}  '
      f'std={np.std([r["accuracy"] for r in split_results]):.4f}')

# ── Step 3: Assemble OOS DataFrame ───────────────────────────────────────────
sep('STEP 3: Assemble OOS results DataFrame + attach realized returns')

# Load realized returns from per-stock labels
ret_frames = []
for ticker_name in TICKERS:
    path = f'data/processed/per_stock/{ticker_name}_labels.parquet'
    if os.path.exists(path):
        lbl = pd.read_parquet(path)[['ret']]
        lbl['ticker'] = ticker_name
        ret_frames.append(lbl)
ret_panel = pd.concat(ret_frames).sort_index()

oos_rows = []
for r in split_results:
    for j in range(len(r['test_indices'])):
        oos_rows.append({
            'split_idx':  r['split_idx'],
            'test_combo': str(r['test_combo']),
            'event_date': r['event_dates'][j],
            'ticker':     r['tickers'][j],
            'pred_class': float(r['pred_class'][j]),
            'pred_prob':  float(r['pred_prob'][j]),
            'true_label': int(r['true_labels'][j]),
            'accuracy':   float(r['pred_class'][j] == r['true_labels'][j]),
        })

oos_df = pd.DataFrame(oos_rows)
oos_df['event_date'] = pd.to_datetime(oos_df['event_date'])

# Attach returns (match on event_date, ticker)
ret_reset = ret_panel.reset_index().rename(columns={'index': 'event_date'})
ret_reset['event_date'] = pd.to_datetime(ret_reset['event_date'])

oos_df = oos_df.merge(ret_reset[['event_date', 'ticker', 'ret']],
                      on=['event_date', 'ticker'], how='left')

n_ret_nan = oos_df['ret'].isna().sum()
print(f'  OOS rows       : {len(oos_df)}')
print(f'  ret NaN        : {n_ret_nan}')
print(f'  Mean accuracy  : {oos_df["accuracy"].mean():.4f}')
print(f'  Coverage       : {oos_df["event_date"].nunique()} unique dates  '
      f'x ~{oos_df["ticker"].nunique()} tickers')

# Expected: each event appears in C(K-1,p-1) = C(5,1) = 5 different test splits
events_per_obs = oos_df.groupby(['event_date', 'ticker']).size()
print(f'  Appearances per event: mean={events_per_obs.mean():.2f}  '
      f'expected={len(list(combinations(range(K-1), P-1)))}')

oos_df.to_parquet('data/processed/cpcv_oos_pooled.parquet', index=False)
print(f'  Saved: data/processed/cpcv_oos_pooled.parquet')

# ── Step 4: Assemble backtest paths ──────────────────────────────────────────
sep('STEP 4: Assemble 15 backtest equity paths (perfect matchings of K=6 groups)')

# Find all perfect matchings: partitions of {0..K-1} into K/p=3 pairs
def get_perfect_matchings(K, p):
    groups = list(range(K))
    all_pairs = list(combinations(groups, p))
    paths = []

    def backtrack(remaining, chosen):
        if not remaining:
            paths.append(tuple(sorted(chosen)))
            return
        first = remaining[0]
        for pair in all_pairs:
            if first == pair[0] and all(g in remaining for g in pair):
                new_remaining = [g for g in remaining if g not in pair]
                backtrack(new_remaining, chosen + [pair])

    backtrack(groups, [])
    return paths

all_paths = get_perfect_matchings(K, P)
print(f'  Total backtest paths: {len(all_paths)}  (perfect matchings)')

# Map split_idx → (event_date, ticker) → predictions
split_lookup = {}
for r in split_results:
    tc = r['test_combo']
    split_lookup[tc] = oos_df[oos_df['test_combo'] == str(tc)].copy()

# For each path, assemble equity curve using Strategy A (±1 position)
panel_daily = adj_close
path_stats  = []

for path_i, path_pairs in enumerate(all_paths):
    # Collect OOS predictions for this path (one per event, from the assigned splits)
    path_preds = pd.concat([split_lookup[pair] for pair in path_pairs])
    path_preds = path_preds.set_index('event_date').sort_index()

    # Build per-ticker positions + returns
    port_returns = []
    for ticker_name in TICKERS:
        t_preds = path_preds[path_preds['ticker'] == ticker_name].copy()
        if len(t_preds) == 0:
            continue

        prices = panel_daily[ticker_name].dropna()

        # t1 for each event (from modelling dataset)
        t1_ticker = t1[ticker.values == ticker_name]
        t1_ticker.index = t1_ticker.index

        # Align t1 to path predictions
        t1_path = t1_ticker.reindex(t_preds.index)
        # fallback: use t1 from modelling for those dates
        if t1_path.isna().any():
            # join from modelling dataset
            t1_mod = modelling.loc[modelling['ticker'] == ticker_name, 't1']
            t1_path = t1_path.fillna(t1_mod.reindex(t1_path.index))
        # still missing: use next trading day as default
        t1_path = t1_path.fillna(t_preds.index.to_series().shift(-1).reindex(t_preds.index))
        t1_path = t1_path.fillna(pd.Timestamp('2025-05-01'))

        sig = pd.Series(t_preds['pred_class'].values, index=t_preds.index)
        avg_pos = avg_active_signals(sig, t1_path)
        daily_pos = build_daily_positions(avg_pos, panel_daily.index)

        price_ret = prices.pct_change().fillna(0.0)
        gross     = daily_pos.shift(1).fillna(0.0) * price_ret
        turnover  = daily_pos.diff().abs().fillna(0.0)
        cost      = turnover * COST_BPS / 10_000.0
        net_ret   = gross - cost

        port_returns.append(net_ret)

    if not port_returns:
        continue

    port_df    = pd.concat(port_returns, axis=1).fillna(0.0)
    port_eq    = port_df.mean(axis=1)
    active     = port_eq[port_eq != 0]

    sr     = sharpe_ratio(active, PPY) if len(active) > 10 else np.nan
    ann_r  = float((1 + active.mean()) ** PPY - 1) if len(active) > 0 else np.nan
    max_dd = float((1 - (1 + active).cumprod() /
                   (1 + active).cumprod().expanding().max()).max()) if len(active) > 0 else np.nan
    n_ev   = len(path_preds)

    path_stats.append({
        'path_idx':   path_i,
        'path_pairs': str(path_pairs),
        'n_events':   n_ev,
        'sr':         sr,
        'ann_return': ann_r,
        'max_dd':     max_dd,
    })

    if path_i % 3 == 0:
        print(f'  Path {path_i:2d} {path_pairs}: n_events={n_ev}  SR={sr:.4f}  ann_ret={ann_r:.4f}')

paths_df = pd.DataFrame(path_stats)
paths_df.to_parquet('data/processed/cpcv_paths_pooled.parquet', index=False)
print(f'\n  Saved: data/processed/cpcv_paths_pooled.parquet')

valid_srs = paths_df['sr'].dropna()
print(f'\n  SR across {len(valid_srs)} paths:')
print(f'    Mean   : {valid_srs.mean():.4f}')
print(f'    Std    : {valid_srs.std():.4f}')
print(f'    Min    : {valid_srs.min():.4f}')
print(f'    Max    : {valid_srs.max():.4f}')
print(f'    % > 0  : {(valid_srs > 0).mean():.1%}')

# Compare with Phase 11 single-pass 5-fold result
phase11_sr = 0.9478   # portfolio SR_A from Phase 13
print(f'\n  Phase 11 (5-fold single-pass portfolio SR): {phase11_sr:.4f}')
print(f'  CPCV mean SR                              : {valid_srs.mean():.4f}')
print(f'  Ratio (CPCV / single-pass)                : {valid_srs.mean()/phase11_sr:.4f}')

# ── Step 5: Figures ───────────────────────────────────────────────────────────
sep('STEP 5: Figures')

# Fig 1: SR distribution across paths + reference lines
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(valid_srs.values, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
axes[0].axvline(valid_srs.mean(), color='red', linestyle='--',
                label=f'Mean ({valid_srs.mean():.3f})')
axes[0].axvline(phase11_sr, color='green', linestyle='--',
                label=f'Phase 11 SR ({phase11_sr:.3f})')
axes[0].axvline(0, color='black', linestyle=':', alpha=0.7)
axes[0].set_xlabel('Annualised Sharpe Ratio')
axes[0].set_ylabel('Count')
axes[0].set_title(f'CPCV SR Distribution (K={K}, p={P})\n'
                  f'{len(valid_srs)} backtest paths, RF best params')
axes[0].legend()

# Box plot of SR by path
axes[1].boxplot(valid_srs.values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.5))
axes[1].axhline(valid_srs.mean(), color='red', linestyle='--', alpha=0.7)
axes[1].axhline(0, color='black', linestyle=':', alpha=0.7)
axes[1].set_ylabel('Sharpe Ratio')
axes[1].set_title('SR Distribution (boxplot)\nCPCV Backtest Paths')
axes[1].set_xticks([])

plt.tight_layout()
plt.savefig('reports/figures/phase14_cpcv_sr_dist.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase14_cpcv_sr_dist.png')

# Fig 2: SR per path + path pairs annotation
fig, ax = plt.subplots(figsize=(16, 5))
colors = ['green' if sr > 0 else 'red' for sr in valid_srs.values]
ax.bar(range(len(valid_srs)), valid_srs.values, color=colors, alpha=0.7, edgecolor='k', linewidth=0.5)
ax.axhline(valid_srs.mean(), color='blue', linestyle='--', linewidth=1.5,
           label=f'Mean SR={valid_srs.mean():.3f}')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Path index')
ax.set_ylabel('Annualised Sharpe Ratio')
ax.set_title(f'Sharpe Ratio per CPCV Backtest Path (K={K}, p={P}, N={len(valid_srs)} paths)')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/phase14_cpcv_equity_paths.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase14_cpcv_equity_paths.png')

# Fig 3: Accuracy heatmap (split x group)
split_accs = np.full((K, K), np.nan)
for r in split_results:
    tc = r['test_combo']
    for g in tc:
        split_accs[tc[0], tc[1]] = r['accuracy']

fig, ax = plt.subplots(figsize=(8, 6))
acc_matrix = np.full((K, K), np.nan)
for r in split_results:
    i, j = r['test_combo']
    acc_matrix[i, j] = r['accuracy']
    acc_matrix[j, i] = r['accuracy']

im = ax.imshow(acc_matrix, cmap='RdYlGn', vmin=0.45, vmax=0.65, aspect='auto')
plt.colorbar(im, ax=ax, label='OOS Accuracy')
ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels([f'G{i}' for i in range(K)])
ax.set_yticklabels([f'G{i}' for i in range(K)])
ax.set_title(f'CPCV OOS Accuracy Heatmap\n(test group pair, K={K}, p={P})')
for i in range(K):
    for j in range(K):
        if not np.isnan(acc_matrix[i, j]):
            ax.text(j, i, f'{acc_matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=8, color='black')
np.fill_diagonal(acc_matrix, np.nan)
plt.tight_layout()
plt.savefig('reports/figures/phase14_cpcv_fold_heatmap.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase14_cpcv_fold_heatmap.png')

# ── Step 6: Validation ────────────────────────────────────────────────────────
sep('STEP 6: Validation')

failures = []
checks = [
    ('cpcv_oos_pooled.parquet saved',
     os.path.exists('data/processed/cpcv_oos_pooled.parquet')),
    ('cpcv_paths_pooled.parquet saved',
     os.path.exists('data/processed/cpcv_paths_pooled.parquet')),
    ('SR dist fig saved',
     os.path.exists('reports/figures/phase14_cpcv_sr_dist.png')),
    ('equity paths fig saved',
     os.path.exists('reports/figures/phase14_cpcv_equity_paths.png')),
    ('fold heatmap fig saved',
     os.path.exists('reports/figures/phase14_cpcv_fold_heatmap.png')),
    (f'all {len(all_combos)} splits completed',
     n_splits_done == len(all_combos)),
    (f'all {len(all_paths)} paths assembled',
     len(paths_df) == len(all_paths)),
    ('all path SRs non-NaN',
     paths_df['sr'].notna().all()),
    ('majority of paths have SR > 0',
     (valid_srs > 0).mean() >= 0.5),
    ('mean CPCV accuracy > 0.50',
     mean_acc > 0.50),
    ('OOS DataFrame non-empty',
     len(oos_df) > 0),
    ('events appear in expected number of splits',
     abs(events_per_obs.mean() - len(list(combinations(range(K-1), P-1)))) < 1.0),
]

for label, cond in checks:
    if not check(label, cond):
        failures.append(label)

n_pass = len(checks) - len(failures)
print(f'\n{"=" * 68}')
if failures:
    print(f'Phase 14 FAILED — {len(failures)} check(s) failed:')
    for f in failures:
        print(f'  {f}: FAIL')
else:
    print(f'Phase 14 COMPLETE — {n_pass} checks passed.')
    print(f'  CPCV config       : K={K}, p={P}, C({K},{P})={len(all_combos)} splits, {len(all_paths)} paths')
    print(f'  Mean OOS accuracy : {mean_acc:.4f}  (across {n_splits_done} splits)')
    print(f'  SR distribution   : mean={valid_srs.mean():.4f}  std={valid_srs.std():.4f}  '
          f'min={valid_srs.min():.4f}  max={valid_srs.max():.4f}')
    print(f'  % paths SR > 0    : {(valid_srs > 0).mean():.1%}')
    print(f'  Phase 11 ref SR   : {phase11_sr:.4f}')
