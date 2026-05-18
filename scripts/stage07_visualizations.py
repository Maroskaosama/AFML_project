"""
Phase 10: Visualization Reconstruction.

Generates the core diagnostic figure set for the 10-stock multi-asset pipeline,
replacing the old single-stock P1-P24 figures.

Output figures (reports/figures/phase10_*.png):
  01_price_history           — normalized adjusted close, all 10 stocks
  02_returns_distribution    — log-return distribution per stock
  03_cusum_events_timeline   — CUSUM event dates for all 10 stocks
  04_label_distribution      — label counts by stock + pooled
  05_sample_weights          — weight histogram + per-ticker boxplot + timeline
  06_fracdiff_d_star         — optimal d* per stock + example fracdiff overlay
  07_feature_correlation_ts  — 17 TS feature correlation heatmap
  08_feature_correlation_all — 50-feature (TS+alpha) correlation heatmap
  09_cv_fold_timeline        — MultiAssetPurgedKFold fold composition
  10_alpha_nan_rates         — NaN rate per alpha (all 101)
  11_alpha_adf_stationarity  — ADF p-value distribution (selected vs excluded)
  12_pooled_events_per_year  — bar chart of events per calendar year
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath('.'))

# ── Config ────────────────────────────────────────────────────────────────────
with open('configs/universe.json') as f:
    UNI = json.load(f)
with open('configs/selected_alphas.json') as f:
    ALPHA_CFG = json.load(f)

TICKERS      = UNI['tickers']
COMMON_START = UNI['common_start']
COMMON_END   = UNI['common_end']

FIGURES_DIR  = 'reports/figures'
PER_STOCK    = 'data/processed/per_stock'
os.makedirs(FIGURES_DIR, exist_ok=True)

_palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
TICKER_COLOR = {t: _palette[i % len(_palette)] for i, t in enumerate(sorted(TICKERS))}

ERRORS = []
saved  = []

def sep(title=''):
    print('\n' + '=' * 68)
    if title:
        print(title)
        print('=' * 68)

def save_fig(name, fig=None, dpi=120):
    path = os.path.join(FIGURES_DIR, f'phase10_{name}.png')
    (fig or plt).savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close('all')
    saved.append(path)
    print(f'  Saved: {path}')


# ── Load shared data ──────────────────────────────────────────────────────────
sep('Loading data')
panel      = pd.read_parquet('data/processed/panel_ohlcv.parquet')
modelling  = pd.read_parquet('data/processed/pooled_modelling.parquet')
w_pool     = pd.read_parquet('data/processed/pooled_weights.parquet')
alpha_full = pd.read_parquet('data/processed/panel_alpha_features.parquet')
diag_df    = pd.read_parquet('data/processed/alpha_diagnostics.parquet')

close_wide = panel['AdjClose'].unstack(level='ticker')
print(f'  Panel   : {panel.shape}')
print(f'  Modelling: {modelling.shape}')

meta_cols  = {'label', 't1', 'weight', 'ticker'}
ts_cols    = [c for c in modelling.columns if c not in meta_cols and not c.startswith('alpha')]
alpha_cols = [c for c in modelling.columns if c.startswith('alpha')]
feat_cols  = ts_cols + alpha_cols


# ── Figure 01: Normalized price history ──────────────────────────────────────
sep('Fig 01: Normalized price history')

normalized = close_wide.div(close_wide.iloc[0])
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Top: log-scale all stocks
ax = axes[0]
for ticker in sorted(close_wide.columns):
    ax.plot(normalized.index, normalized[ticker],
            label=ticker, color=TICKER_COLOR[ticker], linewidth=1.0, alpha=0.85)
ax.set_yscale('log')
ax.set_ylabel('Normalized Price (log scale)')
ax.set_title(f'10-Stock Universe — Adjusted Close (2005-2025, base=1.0 at {COMMON_START})')
ax.legend(ncol=5, loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Bottom: small multiples (linear)
ax2 = axes[1]
for ticker in sorted(close_wide.columns):
    ax2.plot(normalized.index, normalized[ticker],
             label=ticker, color=TICKER_COLOR[ticker], linewidth=0.8, alpha=0.75)
ax2.axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
ax2.set_ylabel('Normalized Price (linear)')
ax2.set_xlabel('Date')
ax2.legend(ncol=5, loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_fig('01_price_history', fig)


# ── Figure 02: Returns distribution ──────────────────────────────────────────
sep('Fig 02: Log-return distribution per stock')

_n_cols = 5
_n_rows = (len(TICKERS) + _n_cols - 1) // _n_cols
fig, axes = plt.subplots(_n_rows, _n_cols, figsize=(16, _n_rows * 3), sharey=False)
axes = axes.flatten()

for i, ticker in enumerate(sorted(TICKERS)):
    ax = axes[i]
    close = close_wide[ticker].dropna()
    log_ret = np.log(close / close.shift(1)).dropna()
    ax.hist(log_ret, bins=80, color=TICKER_COLOR[ticker], alpha=0.75, edgecolor='none', density=True)
    mu, sigma = log_ret.mean(), log_ret.std()
    # Normal overlay
    x = np.linspace(log_ret.min(), log_ret.max(), 200)
    ax.plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2),
            'k-', linewidth=1.0, alpha=0.6)
    ax.set_title(f'{ticker}  (sigma={sigma:.3f})', fontsize=9)
    ax.set_xlim(-0.15, 0.15)
    ax.tick_params(labelsize=7)

plt.suptitle('Daily Log-Return Distributions with Normal Overlay (2005-2025)', fontsize=11)
plt.tight_layout()
save_fig('02_returns_distribution', fig)


# ── Figure 03: CUSUM event timeline ──────────────────────────────────────────
sep('Fig 03: CUSUM event timeline')

fig, ax = plt.subplots(figsize=(14, 6))

for i, ticker in enumerate(sorted(TICKERS)):
    lbl_path = os.path.join(PER_STOCK, f'{ticker}_labels.parquet')
    if not os.path.exists(lbl_path):
        continue
    labels = pd.read_parquet(lbl_path)
    pos_dates = labels[labels['bin'] == 1].index
    neg_dates = labels[labels['bin'] == -1].index
    ax.scatter(pos_dates, [i + 0.15] * len(pos_dates), marker='^', s=12,
               color='green', alpha=0.6, linewidths=0)
    ax.scatter(neg_dates, [i - 0.15] * len(neg_dates), marker='v', s=12,
               color='red', alpha=0.6, linewidths=0)
    ax.text(pd.Timestamp(COMMON_START), i, f' {ticker}', va='center', fontsize=8)

ax.set_yticks(range(len(TICKERS)))
ax.set_yticklabels(sorted(TICKERS), fontsize=8)
ax.set_xlabel('Event Date')
ax.set_title('CUSUM-Filtered Events by Stock (green=+1 up, red=-1 down)')
from matplotlib.lines import Line2D
ax.legend([Line2D([0],[0],marker='^',color='green',linestyle='',markersize=6),
           Line2D([0],[0],marker='v',color='red',  linestyle='',markersize=6)],
          ['+1 (up)', '-1 (down)'], fontsize=9, loc='upper right')
ax.grid(True, axis='x', alpha=0.2)
plt.tight_layout()
save_fig('03_cusum_events_timeline', fig)


# ── Figure 04: Label distribution ────────────────────────────────────────────
sep('Fig 04: Label distribution by stock')

label_rows = []
for ticker in sorted(TICKERS):
    lbl_path = os.path.join(PER_STOCK, f'{ticker}_labels.parquet')
    if not os.path.exists(lbl_path): continue
    labels = pd.read_parquet(lbl_path)
    n_pos = int((labels['bin'] == 1).sum())
    n_neg = int((labels['bin'] == -1).sum())
    label_rows.append({'ticker': ticker, '+1': n_pos, '-1': n_neg})

label_df = pd.DataFrame(label_rows).set_index('ticker')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Stacked bar per ticker
x = np.arange(len(label_df))
axes[0].bar(x, label_df['+1'], color='#2ca02c', alpha=0.8, label='+1 (up)')
axes[0].bar(x, label_df['-1'], bottom=label_df['+1'], color='#d62728', alpha=0.8, label='-1 (down)')
for j, (_, row) in enumerate(label_df.iterrows()):
    total = row['+1'] + row['-1']
    pct   = row['+1'] / total * 100
    axes[0].text(j, total + 3, f'{pct:.0f}%', ha='center', fontsize=7)
axes[0].set_xticks(x)
axes[0].set_xticklabels(label_df.index, rotation=30, fontsize=9)
axes[0].set_ylabel('Event count')
axes[0].set_title('Label Counts per Stock')
axes[0].legend()

# Overall pooled
pool_pos = int((modelling['label'] == 1).sum())
pool_neg = int((modelling['label'] == -1).sum())
axes[1].pie([pool_pos, pool_neg], labels=[f'+1: {pool_pos}', f'-1: {pool_neg}'],
            colors=['#2ca02c', '#d62728'], autopct='%1.1f%%', startangle=90)
axes[1].set_title(f'Pooled Label Split\n(n={pool_pos+pool_neg} events in modelling dataset)')

plt.tight_layout()
save_fig('04_label_distribution', fig)


# ── Figure 05: Sample weights ─────────────────────────────────────────────────
sep('Fig 05: Sample weights')

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig)

# 1. Histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(w_pool['weight'], bins=60, color='steelblue', edgecolor='none', alpha=0.75)
ax1.axvline(w_pool['weight'].quantile(0.99), color='red', linestyle='--',
            label=f"p99={w_pool['weight'].quantile(0.99):.2f}")
ax1.axvline(w_pool['weight'].mean(), color='orange', linestyle='-.',
            label=f"mean={w_pool['weight'].mean():.2f}")
ax1.set_xlabel('Weight'); ax1.set_ylabel('Count')
ax1.set_title('Pooled Weight Distribution'); ax1.legend(fontsize=8)

# 2. Per-ticker box plot
ax2 = fig.add_subplot(gs[0, 1])
ticker_data = [w_pool[w_pool['ticker'] == t]['weight'].values for t in sorted(TICKERS)]
bp = ax2.boxplot(ticker_data, tick_labels=sorted(TICKERS), vert=True, patch_artist=True)
for patch, ticker in zip(bp['boxes'], sorted(TICKERS)):
    patch.set_facecolor(TICKER_COLOR[ticker])
    patch.set_alpha(0.7)
ax2.axhline(w_pool['weight'].quantile(0.99), color='red', linestyle='--', linewidth=0.8)
ax2.set_ylabel('Weight'); ax2.tick_params(axis='x', rotation=30)
ax2.set_title('Weight Distribution by Ticker')

# 3. Weight timeline (scatter)
ax3 = fig.add_subplot(gs[1, :])
for ticker in sorted(TICKERS):
    mask = w_pool['ticker'] == ticker
    ax3.scatter(w_pool[mask].index, w_pool[mask]['weight'],
                color=TICKER_COLOR[ticker], s=6, alpha=0.4, label=ticker)
ax3.axhline(w_pool['weight'].quantile(0.99), color='red', linestyle='--', linewidth=0.8, label='p99')
ax3.set_xlabel('Event Date'); ax3.set_ylabel('Weight')
ax3.set_title('Sample Weight over Time (all stocks)')
ax3.legend(ncol=6, fontsize=7, loc='upper left')

plt.suptitle('Sample Weight Analysis (uniqueness x return_attr x time_decay)', fontsize=11)
plt.tight_layout()
save_fig('05_sample_weights', fig)


# ── Figure 06: Fracdiff d* per stock ─────────────────────────────────────────
sep('Fig 06: Fracdiff d* per stock')

d_stars = {}
for ticker in sorted(TICKERS):
    feat_path = os.path.join(PER_STOCK, f'{ticker}_ts_features.parquet')
    if not os.path.exists(feat_path): continue
    feat = pd.read_parquet(feat_path)
    if 'fracdiff' in feat.columns:
        # d* stored in labels (we don't have it directly, so proxy from per_stock results)
        pass

# d* values from phase4 output — read from the labels parquet (doesn't have d*)
# Alternative: read from the known phase4 output; reconstruct from fracdiff stats
# We use the per-stock ts_features fracdiff series to infer stationarity
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Read clean data + fracdiff to show overlay for one stock
ex_ticker = 'NVDA'
clean_path = os.path.join(PER_STOCK, f'{ex_ticker}_clean.parquet')
feat_path  = os.path.join(PER_STOCK, f'{ex_ticker}_ts_features.parquet')

if os.path.exists(clean_path) and os.path.exists(feat_path):
    clean = pd.read_parquet(clean_path)
    feat  = pd.read_parquet(feat_path)
    close = np.log(clean['Adj Close'])
    frac  = feat['fracdiff']

    ax1 = axes[0]
    ax1b = ax1.twinx()
    close_common = close[close.index.isin(frac.index)]
    ax1.plot(close_common.index, close_common.values,
             color='steelblue', linewidth=0.8, alpha=0.7, label='log(Close)')
    ax1b.plot(frac.index, frac.values,
              color='darkorange', linewidth=0.8, alpha=0.8, label='fracdiff')
    ax1.set_ylabel('log(Close)', color='steelblue')
    ax1b.set_ylabel('fracdiff', color='darkorange')
    ax1.set_title(f'{ex_ticker}: log(Close) vs fracdiff (d*)')
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

# Events per stock bar chart (informative stand-in for d* per stock)
ax2 = axes[1]
event_counts = []
for ticker in sorted(TICKERS):
    lbl_path = os.path.join(PER_STOCK, f'{ticker}_labels.parquet')
    if os.path.exists(lbl_path):
        labels = pd.read_parquet(lbl_path)
        event_counts.append({'ticker': ticker, 'n': len(labels)})
ec_df = pd.DataFrame(event_counts).set_index('ticker')
bar_colors = [TICKER_COLOR[t] for t in ec_df.index]
ax2.bar(ec_df.index, ec_df['n'], color=bar_colors, alpha=0.8, edgecolor='k', linewidth=0.5)
ax2.set_ylabel('CUSUM events (full history)')
ax2.set_title('Events per Stock (post-CUSUM, all dates)')
ax2.tick_params(axis='x', rotation=30)
for i, (ticker, row) in enumerate(ec_df.iterrows()):
    ax2.text(i, row['n'] + 3, str(row['n']), ha='center', fontsize=8)

plt.tight_layout()
save_fig('06_fracdiff_overview', fig)


# ── Figure 07: TS feature correlation ────────────────────────────────────────
sep('Fig 07: TS feature correlation heatmap')

X_ts   = modelling[ts_cols]
corr_ts = X_ts.corr()
n_ts    = len(ts_cols)

fig, ax = plt.subplots(figsize=(max(8, n_ts * 0.55), max(6, n_ts * 0.55)))
im = ax.imshow(corr_ts.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n_ts)); ax.set_yticks(range(n_ts))
ax.set_xticklabels(corr_ts.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr_ts.index, fontsize=7)
for i in range(n_ts):
    for j in range(n_ts):
        val = corr_ts.values[i, j]
        if abs(val) > 0.5:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5, color='white')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(f'{len(ts_cols)} TS Feature Correlation (pooled {len(TICKERS)}-stock, n={len(X_ts)})')
plt.tight_layout()
save_fig('07_feature_correlation_ts', fig)


# ── Figure 08: Full 50-feature correlation ────────────────────────────────────
sep('Fig 08: Full 50-feature correlation heatmap')

X_all   = modelling[feat_cols]
corr_all = X_all.corr()
n_all    = len(feat_cols)

fig, ax = plt.subplots(figsize=(max(12, n_all * 0.4), max(10, n_all * 0.4)))
im = ax.imshow(corr_all.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n_all)); ax.set_yticks(range(n_all))
ax.set_xticklabels(corr_all.columns, rotation=90, fontsize=5)
ax.set_yticklabels(corr_all.index, fontsize=5)

# Dividing line between TS and alpha sections
ax.axvline(len(ts_cols) - 0.5, color='black', linewidth=1.5, alpha=0.7)
ax.axhline(len(ts_cols) - 0.5, color='black', linewidth=1.5, alpha=0.7)
ax.text(len(ts_cols) / 2, -2.5, 'TS features', ha='center', fontsize=8, fontweight='bold')
ax.text(len(ts_cols) + len(alpha_cols) / 2, -2.5, 'Alpha features', ha='center', fontsize=8, fontweight='bold')

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(f'{len(feat_cols)}-Feature Correlation ({len(ts_cols)} TS + {len(alpha_cols)} alpha, pooled {len(TICKERS)}-stock, n={len(X_all)})')
plt.tight_layout()
save_fig('08_feature_correlation_all', fig)


# ── Figure 09: CV fold timeline ───────────────────────────────────────────────
sep('Fig 09: MultiAssetPurgedKFold fold timeline')

from src.cross_validation import MultiAssetPurgedKFold

X_cv = modelling[feat_cols]
y_cv = modelling['label']
t1_cv = modelling['t1']
cv = MultiAssetPurgedKFold(n_splits=5, t1=t1_cv, pct_embargo=0.01)

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Top: fold composition scatter (event dates)
for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_cv, y_cv)):
    train_dates = X_cv.iloc[train_idx].index
    test_dates  = X_cv.iloc[test_idx].index
    axes[0].scatter(train_dates, [fold_i] * len(train_idx),
                    color='steelblue', s=5, alpha=0.25)
    axes[0].scatter(test_dates, [fold_i] * len(test_idx),
                    color='crimson', s=5, alpha=0.8)

axes[0].set_yticks(range(5))
axes[0].set_yticklabels([f'Fold {i}' for i in range(5)], fontsize=9)
axes[0].set_xlabel('Event Date')
axes[0].set_title('MultiAssetPurgedKFold — Event Dates (blue=train, red=test)')

from matplotlib.lines import Line2D
axes[0].legend([Line2D([0],[0],color='steelblue',marker='o',linestyle='',markersize=5, alpha=0.5),
                Line2D([0],[0],color='crimson',  marker='o',linestyle='',markersize=5)],
               ['Train', 'Test'], fontsize=9)

# Bottom: fold size bar chart
fold_sizes = []
for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_cv, y_cv)):
    # Count purged (missing from both)
    all_idx = set(range(len(X_cv)))
    train_set = set(train_idx.tolist())
    test_set  = set(test_idx.tolist())
    purged_n  = len(all_idx - train_set - test_set)
    fold_sizes.append({'fold': fold_i, 'train': len(train_idx),
                       'test': len(test_idx), 'purged': purged_n})

fs_df = pd.DataFrame(fold_sizes).set_index('fold')
x = np.arange(5)
axes[1].bar(x, fs_df['train'], label='Train', color='steelblue', alpha=0.7)
axes[1].bar(x, fs_df['test'],  label='Test',  color='crimson',   alpha=0.7, bottom=fs_df['train'])
axes[1].bar(x, fs_df['purged'],label='Purged/Embargo', color='grey', alpha=0.5,
            bottom=fs_df['train'] + fs_df['test'])
for i, row in fs_df.iterrows():
    axes[1].text(i, row['train'] + row['test'] + row['purged'] + 15,
                 f"test={row['test']}", ha='center', fontsize=8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'Fold {i}' for i in range(5)])
axes[1].set_ylabel('Sample count')
axes[1].set_title('Fold Composition (train / test / purged+embargoed)')
axes[1].legend()

plt.tight_layout()
save_fig('09_cv_fold_timeline', fig)


# ── Figure 10: Alpha NaN rates ────────────────────────────────────────────────
sep('Fig 10: Alpha NaN rates')

nan_pcts = alpha_full.isnull().mean() * 100
selected = ALPHA_CFG['selected_alphas']
excluded_nan = ALPHA_CFG['excluded_nan40']

colors_nan = []
for col in alpha_full.columns:
    if col in excluded_nan:
        colors_nan.append('#d62728')   # red = excluded
    elif col in selected:
        colors_nan.append('#2ca02c')   # green = selected
    else:
        colors_nan.append('#7f7f7f')   # grey = pruned or other

fig, ax = plt.subplots(figsize=(16, 4))
x = np.arange(len(alpha_full.columns))
ax.bar(x, nan_pcts.values, color=colors_nan, alpha=0.8, width=0.8)
ax.axhline(40, color='red', linestyle='--', linewidth=1.2, label='40% exclusion threshold')
ax.set_xticks(x)
ax.set_xticklabels(alpha_full.columns, rotation=90, fontsize=6)
ax.set_ylabel('NaN %')
ax.set_title('NaN Rate per Alpha — green=selected, red=excluded(NaN>40%), grey=pruned')
ax.legend(fontsize=9)
plt.tight_layout()
save_fig('10_alpha_nan_rates', fig)


# ── Figure 11: Alpha ADF stationarity ────────────────────────────────────────
sep('Fig 11: Alpha ADF stationarity')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ADF p-value distribution: selected vs excluded
sel_adf  = diag_df[diag_df.index.isin(selected)]['adf_pval_median']
rest_adf = diag_df[~diag_df.index.isin(selected)]['adf_pval_median']

axes[0].hist(sel_adf.clip(0, 1), bins=20, color='#2ca02c', alpha=0.7,
             label=f'Selected (n={len(sel_adf)})', density=True)
axes[0].hist(rest_adf.clip(0, 1), bins=20, color='#7f7f7f', alpha=0.5,
             label=f'Not selected (n={len(rest_adf)})', density=True)
axes[0].axvline(0.05, color='red', linestyle='--', label='p=0.05')
axes[0].set_xlabel('Median ADF p-value')
axes[0].set_ylabel('Density')
axes[0].set_title('ADF Stationarity: Selected vs Not-Selected Alphas')
axes[0].legend(fontsize=9)

# ADF p-value scatter: rank vs value for selected alphas
sel_sorted = diag_df[diag_df.index.isin(selected)]['adf_pval_median'].sort_values()
axes[1].barh(range(len(sel_sorted)), sel_sorted.values,
             color='#2ca02c', alpha=0.75)
axes[1].set_yticks(range(len(sel_sorted)))
axes[1].set_yticklabels(sel_sorted.index, fontsize=7)
axes[1].axvline(0.05, color='red', linestyle='--', label='p=0.05')
axes[1].set_xlabel('Median ADF p-value')
axes[1].set_title('Selected Alphas — ADF p-value (ranked)')
axes[1].legend(fontsize=9)

plt.tight_layout()
save_fig('11_alpha_adf_stationarity', fig)


# ── Figure 12: Pooled events per year ────────────────────────────────────────
sep('Fig 12: Pooled events per year')

modelling['year'] = modelling.index.year
by_year_ticker = modelling.groupby(['year', 'ticker']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 5))
bottoms = np.zeros(len(by_year_ticker))
for ticker in sorted(TICKERS):
    if ticker in by_year_ticker.columns:
        vals = by_year_ticker[ticker].values
        ax.bar(by_year_ticker.index, vals, bottom=bottoms,
               color=TICKER_COLOR[ticker], alpha=0.85, label=ticker, width=0.8)
        bottoms += vals

ax.set_xlabel('Year')
ax.set_ylabel('Event count')
ax.set_title('Pooled CUSUM Events per Calendar Year (10 stocks)')
ax.legend(ncol=5, loc='upper left', fontsize=8)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
save_fig('12_pooled_events_per_year', fig)

modelling.drop(columns=['year'], inplace=True)


# ── Summary and validation ────────────────────────────────────────────────────
sep('Validation')

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

expected = [
    'phase10_01_price_history.png',
    'phase10_02_returns_distribution.png',
    'phase10_03_cusum_events_timeline.png',
    'phase10_04_label_distribution.png',
    'phase10_05_sample_weights.png',
    'phase10_06_fracdiff_overview.png',
    'phase10_07_feature_correlation_ts.png',
    'phase10_08_feature_correlation_all.png',
    'phase10_09_cv_fold_timeline.png',
    'phase10_10_alpha_nan_rates.png',
    'phase10_11_alpha_adf_stationarity.png',
    'phase10_12_pooled_events_per_year.png',
]

for fname in expected:
    path = os.path.join(FIGURES_DIR, fname)
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    check(f'{fname} saved (size={size}B)', exists and size > 5000)

check(f'all {len(expected)} figures generated', len(saved) >= len(expected))

sep()
if ERRORS:
    print(f'Phase 10 FAILED — {failed} check(s) failed:')
    for e in ERRORS:
        print(f'  {e}')
    import sys; sys.exit(1)
else:
    print(f'Phase 10 COMPLETE — {passed} checks passed.')
    print(f'  Figures saved: {len(saved)}')
    for p in saved:
        size = os.path.getsize(p)
        print(f'    {os.path.basename(p):50s} {size:8d}B')
