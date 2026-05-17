"""
Prompt 6: Feature importance (MDI/MDA/SFI), hyperparameter tuning,
meta-labeling, bet sizing, and backtesting with full statistics table.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cross_validation import MultiAssetPurgedKFold

os.makedirs('reports/figures', exist_ok=True)

# ── Step 1: Load data ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading pooled modelling dataset")
print("=" * 60)

with open('configs/universe.json') as f:
    universe = json.load(f)
TICKERS      = universe['tickers']
COMMON_START = universe['common_start_date']
COMMON_END   = universe['common_end_date']

pooled = pd.read_parquet('data/processed/pooled_modelling.parquet')
print(f"  Pooled: {pooled.shape}")

# Add 'actual_ret' from per-stock labels for backtesting
PER_STOCK_DIR = 'data/processed/per_stock'
ret_rows = []
for ticker in TICKERS:
    labels_path = f'{PER_STOCK_DIR}/{ticker}_labels.parquet'
    if not os.path.exists(labels_path):
        continue
    lbl = pd.read_parquet(labels_path)
    lbl = lbl[(lbl.index >= COMMON_START) & (lbl.index <= COMMON_END)]
    lbl = lbl[['ret']].copy()
    lbl['ticker'] = ticker
    ret_rows.append(lbl)

ret_df = pd.concat(ret_rows).sort_index()
# Join on (date, ticker) multi-index to handle multiple tickers per date
ret_mi  = ret_df.set_index('ticker', append=True)
pool_mi = pooled.set_index('ticker', append=True)
pool_mi = pool_mi.join(ret_mi[['ret']], how='left')
pool_mi = pool_mi.reset_index(level='ticker')
pooled['actual_ret'] = pool_mi['ret'].values

print(f"  Missing actual_ret: {pooled['actual_ret'].isnull().sum()}")
print(f"  Label distribution: +1={int((pooled['label']==1).sum())}, -1={int((pooled['label']==-1).sum())}")

# Feature / label setup
feat_cols  = [c for c in pooled.columns if c not in {'label', 'weight', 't1', 'ticker', 'actual_ret'}]
ts_cols    = [c for c in feat_cols if not c.startswith('alpha')]
alpha_cols = [c for c in feat_cols if c.startswith('alpha')]
print(f"  Features: {len(feat_cols)} total ({len(ts_cols)} TS + {len(alpha_cols)} alpha)")

X  = pooled[feat_cols]
y  = pooled['label']
w  = pooled['weight']
t1 = pooled['t1']

cv = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

rf_base = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=30,
    max_features='sqrt', random_state=42, n_jobs=-1
)

# ── Step 2: MDI Feature Importance ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: MDI Feature Importance")
print("=" * 60)

rf_mdi = clone(rf_base)
rf_mdi.fit(X, y, sample_weight=w.values)

tree_imps    = np.array([t.feature_importances_ for t in rf_mdi.estimators_])
mdi_mean     = pd.Series(tree_imps.mean(axis=0), index=feat_cols)
mdi_std      = pd.Series(tree_imps.std(axis=0),  index=feat_cols)
mdi_df       = pd.DataFrame({'mean': mdi_mean, 'std': mdi_std}).sort_values('mean', ascending=False)

print("  Top 10 MDI features:")
for feat, row in mdi_df.head(10).iterrows():
    print(f"    {feat}: {row['mean']:.4f} +/- {row['std']:.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
top20 = mdi_df.head(20)
ax.barh(range(20), top20['mean'].values[::-1], xerr=top20['std'].values[::-1],
        align='center', color='steelblue', ecolor='black', capsize=3)
ax.set_yticks(range(20))
ax.set_yticklabels(top20.index.tolist()[::-1], fontsize=9)
ax.set_xlabel('MDI Score')
ax.set_title('Top 20 Features by MDI (Mean Decrease Impurity)')
plt.tight_layout()
plt.savefig('reports/figures/P6_mdi_importance.png', dpi=100)
plt.close()
print("  Saved reports/figures/P6_mdi_importance.png")

# ── Step 3: MDA Feature Importance ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: MDA Feature Importance (Permutation)")
print("=" * 60)

mda_records = []
for fold_i, (tr, te) in enumerate(cv.split(X)):
    clf = clone(rf_base)
    clf.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr].values)
    perm = permutation_importance(
        clf, X.iloc[te], y.iloc[te],
        sample_weight=w.iloc[te].values,
        n_repeats=10, random_state=42, n_jobs=-1
    )
    for j, feat in enumerate(feat_cols):
        mda_records.append({'feature': feat, 'mean': perm.importances_mean[j], 'fold': fold_i})
    print(f"  Fold {fold_i} done")

mda_all = pd.DataFrame(mda_records)
mda_df  = mda_all.groupby('feature')['mean'].mean().sort_values(ascending=False).to_frame()

print("  Top 10 MDA features:")
for feat, row in mda_df.head(10).iterrows():
    print(f"    {feat}: {row['mean']:.6f}")

fig, ax = plt.subplots(figsize=(10, 8))
top20_mda = mda_df.head(20)
ax.barh(range(20), top20_mda['mean'].values[::-1], align='center', color='coral')
ax.set_yticks(range(20))
ax.set_yticklabels(top20_mda.index.tolist()[::-1], fontsize=9)
ax.set_xlabel('MDA Score (Permutation Importance)')
ax.set_title('Top 20 Features by MDA (Mean Decrease Accuracy)')
plt.tight_layout()
plt.savefig('reports/figures/P6_mda_importance.png', dpi=100)
plt.close()
print("  Saved reports/figures/P6_mda_importance.png")

# ── Step 4: SFI Feature Importance ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: SFI Feature Importance (Single Feature)")
print("=" * 60)

sfi_rf = RandomForestClassifier(
    n_estimators=100, max_depth=3, min_samples_leaf=20,
    max_features=1.0, random_state=42
)

sfi_scores = {}
for feat in feat_cols:
    X_single  = X[[feat]]
    fold_accs = []
    for tr, te in cv.split(X_single):
        clf = clone(sfi_rf)
        clf.fit(X_single.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr].values)
        pred = clf.predict(X_single.iloc[te])
        fold_accs.append(accuracy_score(y.iloc[te], pred, sample_weight=w.iloc[te].values))
    sfi_scores[feat] = np.mean(fold_accs)

sfi_df = pd.Series(sfi_scores).sort_values(ascending=False)
print("  Top 10 SFI features:")
for feat, score in sfi_df.head(10).items():
    print(f"    {feat}: {score:.4f}")

# ── Step 5: Hyperparameter Tuning ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Hyperparameter Tuning (PurgedKFold grid search)")
print("=" * 60)

grid = [
    (md, ml, mf)
    for md in [3, 5, 7]
    for ml in [20, 30, 50]
    for mf in ['sqrt', 0.5]
]

best_score  = -np.inf
best_params = {}
grid_rows   = []

for max_depth, min_leaf, max_feat in grid:
    fold_scores = []
    for tr, te in cv.split(X):
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=max_depth,
            min_samples_leaf=min_leaf, max_features=max_feat,
            random_state=42, n_jobs=-1
        )
        clf.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr].values)
        pred = clf.predict(X.iloc[te])
        fold_scores.append(accuracy_score(y.iloc[te], pred, sample_weight=w.iloc[te].values))
    mean_score = np.mean(fold_scores)
    grid_rows.append({
        'max_depth': max_depth, 'min_samples_leaf': min_leaf,
        'max_features': str(max_feat), 'cv_accuracy': mean_score
    })
    if mean_score > best_score:
        best_score  = mean_score
        best_params = {'max_depth': max_depth, 'min_samples_leaf': min_leaf, 'max_features': max_feat}
    print(f"  depth={max_depth}, min_leaf={min_leaf}, max_feat={max_feat}: {mean_score:.4f}")

grid_df = pd.DataFrame(grid_rows)
grid_df.to_parquet('data/processed/hp_grid_results.parquet')
print(f"\n  Best params: {best_params}  ->  CV accuracy: {best_score:.4f}")

# ── Step 6: Meta-Labeling ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Meta-Labeling (OOB primary -> secondary model)")
print("=" * 60)

primary_rf = RandomForestClassifier(
    n_estimators=200, oob_score=True,
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42, n_jobs=-1
)
secondary_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42, n_jobs=-1
)

meta_results   = []
cv_meta_scores = []

for fold_i, (tr, te) in enumerate(cv.split(X)):
    X_tr_ts  = X.iloc[tr][ts_cols]
    X_te_ts  = X.iloc[te][ts_cols]
    y_tr     = y.iloc[tr]
    y_te     = y.iloc[te]
    w_tr     = w.iloc[tr]
    w_te     = w.iloc[te]

    # ── Primary model (TS only) with OOB ──────────────────────────────────
    primary = clone(primary_rf)
    primary.fit(X_tr_ts, y_tr, sample_weight=w_tr.values)

    oob_proba   = primary.oob_decision_function_
    oob_pred    = primary.classes_[np.argmax(oob_proba, axis=1)]
    meta_lbl_tr = (oob_pred == y_tr.values).astype(int)

    # ── Secondary model (all 50 features) ─────────────────────────────────
    secondary = clone(secondary_rf)
    secondary.fit(X.iloc[tr], meta_lbl_tr, sample_weight=w_tr.values)

    # ── Test set predictions ───────────────────────────────────────────────
    primary_dir  = primary.predict(X_te_ts)
    meta_prob    = secondary.predict_proba(X.iloc[te])[:, 1]
    bet_size     = primary_dir * meta_prob  # signed position [-1, +1]

    primary_acc  = accuracy_score(y_te, primary_dir, sample_weight=w_te.values)
    meta_lbl_te  = (primary_dir == y_te.values).astype(int)
    meta_acc     = accuracy_score(meta_lbl_te, (meta_prob >= 0.5).astype(int),
                                  sample_weight=w_te.values)

    cv_meta_scores.append({'fold': fold_i, 'primary_acc': primary_acc, 'meta_acc': meta_acc})

    fold_df = pd.DataFrame({
        'date':        pooled.index[te],
        'ticker':      pooled['ticker'].iloc[te].values,
        'true_label':  y_te.values,
        'primary_dir': primary_dir,
        'meta_prob':   meta_prob,
        'bet_size':    bet_size,
        'actual_ret':  pooled['actual_ret'].iloc[te].values,
        'weight':      w_te.values,
        'fold':        fold_i,
    })
    meta_results.append(fold_df)
    print(f"  Fold {fold_i}: primary_acc={primary_acc:.4f}, meta_acc={meta_acc:.4f}, "
          f"avg|bet|={np.abs(bet_size).mean():.4f}")

meta_df        = pd.concat(meta_results).sort_values('date').reset_index(drop=True)
meta_scores_df = pd.DataFrame(cv_meta_scores)

print(f"\n  Primary CV acc: {meta_scores_df['primary_acc'].mean():.4f} +/- {meta_scores_df['primary_acc'].std():.4f}")
print(f"  Meta CV acc:   {meta_scores_df['meta_acc'].mean():.4f} +/- {meta_scores_df['meta_acc'].std():.4f}")
print(f"  Avg |bet|:     {meta_df['bet_size'].abs().mean():.4f}")

# ── Step 7: Backtesting ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Backtesting (per-trade P&L)")
print("=" * 60)

meta_df['pnl']       = meta_df['bet_size']  * meta_df['actual_ret']
meta_df['pnl_naive'] = meta_df['true_label'] * meta_df['actual_ret']


def backtest_stats(pnl_series):
    n = len(pnl_series)
    if n < 5:
        return {}
    mean_r  = pnl_series.mean()
    std_r   = pnl_series.std()
    sharpe  = (mean_r / std_r * np.sqrt(n)) if std_r > 0 else 0.0
    hit     = (pnl_series > 0).mean()
    cum     = pnl_series.cumsum()
    max_dd  = (cum - cum.cummax()).min()
    n_years = max(n / 25.0, 0.1)        # ~25 events per year per stock
    calmar  = (pnl_series.sum() / n_years) / abs(max_dd) if max_dd < 0 else np.inf
    return {
        'n_trades':       int(n),
        'total_ret':      float(pnl_series.sum()),
        'mean_pnl':       float(mean_r),
        'std_pnl':        float(std_r),
        'sharpe':         float(sharpe),
        'hit_rate':       float(hit),
        'max_drawdown':   float(max_dd),
        'calmar':         float(calmar),
    }


print(f"\n  {'Ticker':<8} {'N':>5} {'TotalRet':>9} {'Sharpe':>8} {'HitRate':>8} {'MaxDD':>9} {'Calmar':>8}")
print("  " + "-" * 65)

ticker_stats = {}
for ticker in TICKERS:
    mask = meta_df['ticker'] == ticker
    if mask.sum() < 5:
        continue
    s = backtest_stats(meta_df.loc[mask, 'pnl'])
    ticker_stats[ticker] = s
    print(f"  {ticker:<8} {s['n_trades']:>5d} {s['total_ret']:>9.4f} "
          f"{s['sharpe']:>8.4f} {s['hit_rate']:>8.3f} "
          f"{s['max_drawdown']:>9.4f} {s['calmar']:>8.3f}")

print("\n  Portfolio-level results:")
port_meta  = backtest_stats(meta_df['pnl'])
port_naive = backtest_stats(meta_df['pnl_naive'])

print(f"  Meta-labeled : total={port_meta['total_ret']:.4f}, sharpe={port_meta['sharpe']:.4f}, "
      f"hit={port_meta['hit_rate']:.3f}, max_dd={port_meta['max_drawdown']:.4f}")
print(f"  Naive labels : total={port_naive['total_ret']:.4f}, sharpe={port_naive['sharpe']:.4f}, "
      f"hit={port_naive['hit_rate']:.3f}, max_dd={port_naive['max_drawdown']:.4f}")

# ── Step 8: Save outputs and plots ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Saving outputs")
print("=" * 60)

# Cumulative P&L + bet size distribution
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

meta_by_date  = meta_df.groupby('date')['pnl'].sum()
naive_by_date = meta_df.groupby('date')['pnl_naive'].sum()

axes[0].plot(meta_by_date.index,  meta_by_date.cumsum(),  label='Meta-labeled', linewidth=1.5)
axes[0].plot(naive_by_date.index, naive_by_date.cumsum(), label='Naive (label dir)', linewidth=1.5, alpha=0.7)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_title('Portfolio Cumulative P&L (OOF test folds)')
axes[0].set_ylabel('Cumulative Return')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(meta_df['bet_size'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(0, color='red', linewidth=1.5)
axes[1].set_title('Bet Size Distribution (primary_dir x meta_prob)')
axes[1].set_xlabel('Bet Size')
axes[1].set_ylabel('Count')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/P6_backtest_results.png', dpi=100)
plt.close()
print("  Saved reports/figures/P6_backtest_results.png")

# Importance comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax, (df, title) in zip(axes, [
    (mdi_df['mean'].head(15), 'MDI'),
    (mda_df['mean'].head(15), 'MDA'),
    (sfi_df.head(15), 'SFI'),
]):
    ax.barh(range(len(df)), df.values[::-1], align='center')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist()[::-1], fontsize=8)
    ax.set_title(f'Top 15 by {title}')
    ax.set_xlabel('Score')
    ax.grid(True, alpha=0.3)
plt.suptitle('Feature Importance Comparison: MDI vs MDA vs SFI', fontsize=13)
plt.tight_layout()
plt.savefig('reports/figures/P6_importance_comparison.png', dpi=100)
plt.close()
print("  Saved reports/figures/P6_importance_comparison.png")

# Parquet outputs
meta_df.to_parquet('data/processed/meta_labeled_predictions.parquet')
mdi_df.to_parquet('data/processed/mdi_importance.parquet')
mda_df.to_parquet('data/processed/mda_importance.parquet')
sfi_df.to_frame('sfi_score').to_parquet('data/processed/sfi_importance.parquet')

stats_table = pd.DataFrame(ticker_stats).T
stats_table.loc['PORTFOLIO_META']  = port_meta
stats_table.loc['PORTFOLIO_NAIVE'] = port_naive
stats_table.to_parquet('data/processed/backtest_stats.parquet')

print("  Saved data/processed/meta_labeled_predictions.parquet")
print("  Saved data/processed/mdi/mda/sfi_importance.parquet")
print("  Saved data/processed/backtest_stats.parquet")

print("\n" + "=" * 60)
print("PROMPT 6 COMPLETE")
print(f"  Best RF params:      {best_params}")
print(f"  Primary CV acc:      {meta_scores_df['primary_acc'].mean():.4f}")
print(f"  Meta CV acc:         {meta_scores_df['meta_acc'].mean():.4f}")
print(f"  Portfolio meta P&L:  {port_meta['total_ret']:.4f}")
print(f"  Portfolio naive P&L: {port_naive['total_ret']:.4f}")
print("=" * 60)
