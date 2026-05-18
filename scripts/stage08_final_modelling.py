"""
Phase 11: Final Modelling — Feature Importance, HP Tuning, OOS Predictions.

Steps:
  1  MDI / MDA / SFI feature importance on pooled 50-feature dataset
  2  Tri-method consensus pruning (bottom-N in ALL three rankings)
  3  HP tuning: 30-trial randomised search for RF and XGB (reduced features)
  4  OOS predictions for ALL 2071 samples via MultiAssetPurgedKFold loop

Outputs:
  data/processed/feature_importance_pooled.parquet
  data/processed/tuning_log_pooled.parquet
  data/processed/oos_predictions_pooled.parquet
  models/best_params_pooled.json
  reports/figures/phase11_mdi_importance.png
  reports/figures/phase11_mda_importance.png
  reports/figures/phase11_sfi_importance.png
  reports/figures/phase11_hp_tuning.png
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.base    import clone
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, classification_report,
)
from xgboost         import XGBClassifier

from src.cross_validation    import MultiAssetPurgedKFold, cv_score
from src.feature_importance  import (
    feat_imp_MDI, feat_imp_MDA, feat_imp_SFI, plot_feature_importance,
)
from src.hyperparameter_tuning import (
    purged_random_search,
    deflated_sharpe_ratio_for_trials,
    log_trials,
)

# ── Config ────────────────────────────────────────────────────────────────────
FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

RNG    = 42
N_ITER = 30    # tuning trials per model (master plan spec)
ERRORS = []

def sep(title=''):
    print('\n' + '=' * 68)
    if title: print(title); print('=' * 68)

def check(label, cond):
    s = 'PASS' if cond else 'FAIL'
    if not cond: ERRORS.append(f'{label}: FAIL')
    print(f'  [{s}] {label}')


# ── Load data ─────────────────────────────────────────────────────────────────
sep('Load pooled modelling dataset')

modelling = pd.read_parquet('data/processed/pooled_modelling.parquet')

meta_cols  = {'label', 't1', 'weight', 'ticker'}
ts_cols    = [c for c in modelling.columns
              if c not in meta_cols and not c.startswith('alpha')]
alpha_cols = [c for c in modelling.columns if c.startswith('alpha')]
feat_cols  = ts_cols + alpha_cols

X  = modelling[feat_cols]
y  = modelling['label'].astype(int)
w  = modelling['weight']
t1 = modelling['t1']

majority_baseline = float((y == y.mode()[0]).mean())

print(f'  Shape       : {X.shape}')
print(f'  TS features : {len(ts_cols)}')
print(f'  Alpha feat  : {len(alpha_cols)}')
print(f'  Label dist  : {dict(y.value_counts().sort_index())}')
print(f'  Majority    : {majority_baseline:.4f}')

cv = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

# Baseline RF for importance (shallow for speed, 200 trees for MDI quality)
rf_base = RandomForestClassifier(
    n_estimators=200, max_depth=6, min_samples_leaf=5,
    class_weight='balanced', max_features='sqrt',
    random_state=RNG, n_jobs=-1,
)


# ── Step 1: MDI ───────────────────────────────────────────────────────────────
sep('STEP 1: MDI — fit RF on full data, extract impurity importances')

t0 = time.time()
rf_base.fit(X.fillna(X.mean()), y, sample_weight=w.values)
mdi = feat_imp_MDI(rf_base, feat_cols)
print(f'  Done in {time.time()-t0:.1f}s')
print(f'  Top 10 by MDI:')
print(mdi.head(10).round(4).to_string())


# ── Step 2: MDA ───────────────────────────────────────────────────────────────
sep('STEP 2: MDA — permutation importance (neg_log_loss, weighted CV)')

rf_mda = RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_leaf=5,
    class_weight='balanced', max_features='sqrt',
    random_state=RNG, n_jobs=-1,
)

t0 = time.time()
mda = feat_imp_MDA(
    clf=rf_mda, X=X.fillna(X.mean()), y=y, cv=cv,
    sample_weight=w, scoring='neg_log_loss', random_state=RNG,
)
print(f'  Done in {time.time()-t0:.1f}s')
print(f'  Top 10 by MDA:')
print(mda.head(10).round(4).to_string())


# ── Step 3: SFI ───────────────────────────────────────────────────────────────
sep('STEP 3: SFI — single-feature purged CV (neg_log_loss, weighted)')

rf_sfi = RandomForestClassifier(
    n_estimators=50, max_depth=3, min_samples_leaf=5,
    class_weight='balanced', max_features=1,
    random_state=RNG, n_jobs=-1,
)

t0 = time.time()
sfi = feat_imp_SFI(
    clf_template=rf_sfi, X=X.fillna(X.mean()), y=y, cv=cv,
    sample_weight=w, scoring='neg_log_loss',
)
print(f'  Done in {time.time()-t0:.1f}s')
print(f'  Top 10 by SFI:')
print(sfi.head(10).round(4).to_string())


# ── Step 4: Rank table + consensus pruning ────────────────────────────────────
sep('STEP 4: Rank table + tri-method consensus pruning')

n_feat = len(feat_cols)

rank_df = pd.DataFrame({
    'MDI_rank': mdi['mean'].rank(ascending=False).astype(int),
    'MDA_rank': mda['mean'].rank(ascending=False).astype(int),
    'SFI_rank': sfi['mean'].rank(ascending=False).astype(int),
}, index=feat_cols)
rank_df['avg_rank'] = rank_df.mean(axis=1)
rank_df = rank_df.sort_values('avg_rank')

print(f'\n  Full rank table (top 20 by avg_rank):')
print(rank_df.head(20).round(2).to_string())

# Tri-method consensus: drop only if bottom-N in ALL THREE
bottom_n   = max(1, int(np.ceil(n_feat * 0.10)))  # bottom 10%
threshold  = n_feat - bottom_n + 1
is_bottom  = (rank_df[['MDI_rank', 'MDA_rank', 'SFI_rank']] >= threshold)
drop_feats = list(rank_df.index[is_bottom.all(axis=1)])
keep_feats = [c for c in feat_cols if c not in drop_feats]

print(f'\n  Bottom-10% threshold rank : {threshold} (bottom {bottom_n} of {n_feat})')
print(f'  Dropped (bottom in all 3) : {len(drop_feats)} — {drop_feats}')
print(f'  Kept                      : {len(keep_feats)}')

# Alpha vs TS in top-10
top10_avg = rank_df.head(10).index.tolist()
alpha_in_top10 = [f for f in top10_avg if f.startswith('alpha')]
print(f'\n  Alpha features in top-10 (avg rank): {alpha_in_top10}')
print(f'  TS features in top-10    (avg rank): {[f for f in top10_avg if not f.startswith("alpha")]}')


# ── Step 5: Save importance artefact ─────────────────────────────────────────
sep('STEP 5: Save feature importance parquet')

importance_df = pd.concat({'MDI': mdi, 'MDA': mda, 'SFI': sfi}, axis=1)
importance_df.columns = ['_'.join(c) for c in importance_df.columns]
importance_df = importance_df.join(rank_df)
importance_df['kept']    = ~importance_df.index.isin(drop_feats)
importance_df['is_alpha'] = importance_df.index.str.startswith('alpha')
importance_df.to_parquet('data/processed/feature_importance_pooled.parquet')
print(f'  Saved: data/processed/feature_importance_pooled.parquet  {importance_df.shape}')


# ── Step 6: Importance plots ──────────────────────────────────────────────────
sep('STEP 6: Importance plots')

def save_importance_fig(imp, title, color, fname):
    fig = plot_feature_importance(imp, title, color=color)
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

save_importance_fig(mdi, 'MDI — 50 features (10-stock pooled)', '#2c7fb8',
                    'phase11_mdi_importance.png')
save_importance_fig(mda, 'MDA — neg_log_loss permutation (weighted)', '#e34a33',
                    'phase11_mda_importance.png')
save_importance_fig(sfi, 'SFI — single-feature neg_log_loss (weighted)', '#31a354',
                    'phase11_sfi_importance.png')


# ── Step 7: HP tuning — RF ────────────────────────────────────────────────────
sep(f'STEP 7: HP tuning — RF ({N_ITER} trials, reduced {len(keep_feats)} features)')

X_red = X[keep_feats].fillna(X[keep_feats].mean())

rf_search_space = {
    'n_estimators':    [100, 200, 300],
    'max_depth':       [3, 4, 5, 6, None],
    'min_samples_leaf':[3, 5, 10, 20],
    'max_features':    ['sqrt', 0.5, 0.3],
    'class_weight':    ['balanced', 'balanced_subsample'],
}

t0 = time.time()
rf_result = purged_random_search(
    clf         = RandomForestClassifier(random_state=RNG, n_jobs=-1),
    X           = X_red,
    y           = y,
    param_dist  = rf_search_space,
    cv          = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01),
    n_iter      = N_ITER,
    sample_weight = w,
    scoring     = 'balanced_accuracy',
    random_state = RNG,
)
rf_time = time.time() - t0

print(f'  RF tuning done in {rf_time:.1f}s')
print(f'  Best mean acc : {rf_result["best_mean_score"]:.4f}  '
      f'std={rf_result["best_std_score"]:.4f}')
print(f'  Best params   : {rf_result["best_params"]}')


# ── Step 8: HP tuning — XGB ──────────────────────────────────────────────────
sep(f'STEP 8: HP tuning — XGB ({N_ITER} trials, reduced {len(keep_feats)} features)')

xgb_search_space = {
    'n_estimators':   [100, 200, 300],
    'max_depth':      [3, 4, 5],
    'learning_rate':  [0.01, 0.05, 0.1, 0.2],
    'subsample':      [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'reg_lambda':     [1.0, 5.0, 10.0],
    'gamma':          [0.0, 0.1, 0.5],
}

t0 = time.time()
xgb_result = purged_random_search(
    clf          = XGBClassifier(
                       random_state=RNG, eval_metric='logloss',
                       n_jobs=-1,
                   ),
    X            = X_red,
    y            = (y == 1).astype(int),   # XGB expects 0/1
    param_dist   = xgb_search_space,
    cv           = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01),
    n_iter       = N_ITER,
    sample_weight = w,
    scoring      = 'balanced_accuracy',
    random_state  = RNG,
)
xgb_time = time.time() - t0

print(f'  XGB tuning done in {xgb_time:.1f}s')
print(f'  Best mean acc : {xgb_result["best_mean_score"]:.4f}  '
      f'std={xgb_result["best_std_score"]:.4f}')
print(f'  Best params   : {xgb_result["best_params"]}')


# ── Step 9: DSR over tuning trials ────────────────────────────────────────────
sep('STEP 9: Deflated Sharpe Ratio over tuning trials')

rf_fold_scores  = [t['fold_scores'] for t in rf_result['trials']]
xgb_fold_scores = [t['fold_scores'] for t in xgb_result['trials']]

rf_dsr_info  = deflated_sharpe_ratio_for_trials(rf_fold_scores,  best_idx=rf_result['best_idx'])
xgb_dsr_info = deflated_sharpe_ratio_for_trials(xgb_fold_scores, best_idx=xgb_result['best_idx'])

print(f'\n  RF  DSR={rf_dsr_info["dsr"]:.4f}  best_SR={rf_dsr_info["best_sr"]:.4f}  '
      f'expected_max_SR={rf_dsr_info["expected_max_sr"]:.4f}  n_trials={rf_dsr_info["n_trials"]}')
print(f'  XGB DSR={xgb_dsr_info["dsr"]:.4f}  best_SR={xgb_dsr_info["best_sr"]:.4f}  '
      f'expected_max_SR={xgb_dsr_info["expected_max_sr"]:.4f}  n_trials={xgb_dsr_info["n_trials"]}')


# ── Step 10: Save tuning log + best params ────────────────────────────────────
sep('STEP 10: Save tuning log + best_params_pooled.json')

rf_log  = log_trials(rf_result['trials'])
rf_log['model'] = 'rf'
xgb_log = log_trials(xgb_result['trials'])
xgb_log['model'] = 'xgb'
tuning_log = pd.concat([rf_log, xgb_log], ignore_index=True)
# cast object columns with mixed types to string so parquet can serialize them
for col in tuning_log.select_dtypes('object').columns:
    tuning_log[col] = tuning_log[col].astype(str)
tuning_log.to_parquet('data/processed/tuning_log_pooled.parquet')
print(f'  Saved: data/processed/tuning_log_pooled.parquet  {tuning_log.shape}')

best_params = {
    'rf': {
        'params':          rf_result['best_params'],
        'mean_score':      rf_result['best_mean_score'],
        'std_score':       rf_result['best_std_score'],
        'n_features':      len(keep_feats),
        'dsr':             rf_dsr_info['dsr'],
        'best_sr':         rf_dsr_info['best_sr'],
        'n_trials':        N_ITER,
        'oos_metrics':     _rf_oos_metrics,
    },
    'xgb': {
        'params':          xgb_result['best_params'],
        'mean_score':      xgb_result['best_mean_score'],
        'std_score':       xgb_result['best_std_score'],
        'n_features':      len(keep_feats),
        'dsr':             xgb_dsr_info['dsr'],
        'best_sr':         xgb_dsr_info['best_sr'],
        'n_trials':        N_ITER,
    },
    'feature_selection': {
        'n_total':   len(feat_cols),
        'n_kept':    len(keep_feats),
        'n_dropped': len(drop_feats),
        'dropped':   drop_feats,
        'kept':      keep_feats,
        'alpha_in_top10_avg': alpha_in_top10,
    },
    'majority_baseline': majority_baseline,
}

with open('models/best_params_pooled.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print(f'  Saved: models/best_params_pooled.json')

print(f'\n  Summary:')
print(f'  {"Model":6s}  {"Mean acc":9s}  {"Std":6s}  {"vs majority":11s}  {"DSR":6s}')
print(f'  {"-"*6}  {"-"*9}  {"-"*6}  {"-"*11}  {"-"*6}')
for nm, res, dsr in [
    ('RF',  rf_result,  rf_dsr_info),
    ('XGB', xgb_result, xgb_dsr_info),
]:
    beat = res['best_mean_score'] - majority_baseline
    print(f'  {nm:6s}  {res["best_mean_score"]:.4f}     '
          f'{res["best_std_score"]:.4f}  {beat:+.4f}       '
          f'{dsr["dsr"]:.4f}')


# ── Step 11: HP tuning plot ───────────────────────────────────────────────────
sep('STEP 11: HP tuning scatter plot')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, log_df, title, color in [
    (axes[0], rf_log,  'RF — 30 trials',  '#2c7fb8'),
    (axes[1], xgb_log, 'XGB — 30 trials', '#e34a33'),
]:
    ax.scatter(range(len(log_df)), log_df['mean_score'],
               c=color, s=40, alpha=0.7, zorder=3)
    ax.errorbar(range(len(log_df)), log_df['mean_score'],
                yerr=log_df['std_score'], fmt='none', alpha=0.3, color=color)
    best_i = int(log_df['mean_score'].idxmax())
    ax.scatter([best_i], [log_df.loc[best_i, 'mean_score']],
               color='gold', s=120, zorder=5, marker='*', label=f'Best ({log_df.loc[best_i,"mean_score"]:.4f})')
    ax.axhline(majority_baseline, color='k', linestyle='--', alpha=0.6,
               label=f'Majority ({majority_baseline:.4f})')
    ax.set_xlabel('Trial'); ax.set_ylabel('Mean CV Accuracy')
    ax.set_title(title); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'phase11_hp_tuning.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f'  Saved: {path}')


# ── Step 12: OOS predictions ──────────────────────────────────────────────────
sep('STEP 12: OOS predictions for all 2071 samples (best RF)')

best_rf_params = rf_result['best_params'].copy()
tuned_rf = RandomForestClassifier(**best_rf_params, random_state=RNG, n_jobs=-1)

oos_pred  = pd.Series(np.nan, index=X_red.index, dtype=float)
oos_prob  = pd.Series(np.nan, index=X_red.index, dtype=float)
oos_fold  = pd.Series(-1,     index=X_red.index, dtype=int)

t0 = time.time()
for fold_i, (train_idx, test_idx) in enumerate(
        MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01).split(X_red, y)):

    X_tr = X_red.iloc[train_idx].fillna(X_red.iloc[train_idx].mean())
    X_te = X_red.iloc[test_idx].fillna(X_red.iloc[train_idx].mean())
    y_tr = y.iloc[train_idx]
    w_tr = w.iloc[train_idx]

    clf = clone(tuned_rf)
    clf.fit(X_tr, y_tr, sample_weight=w_tr.values)

    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]

    idx = X_red.iloc[test_idx].index
    oos_pred.iloc[test_idx]  = preds.astype(float)
    oos_prob.iloc[test_idx]  = probs
    oos_fold.iloc[test_idx]  = fold_i
    print(f'  Fold {fold_i}: test={len(test_idx):4d}  '
          f'acc={float((preds==y.iloc[test_idx].values).mean()):.4f}')

elapsed_oos = time.time() - t0

# Verify full coverage
n_covered = oos_pred.notna().sum()
oos_acc   = float((oos_pred == y).mean())
print(f'\n  OOS elapsed   : {elapsed_oos:.1f}s')
print(f'  Covered       : {n_covered} / {len(X_red)} samples')
print(f'  OOS accuracy  : {oos_acc:.4f}  (majority={majority_baseline:.4f})')
print(f'  Side dist (+1): {(oos_pred==1).mean():.4f}')

oos_df = pd.DataFrame({
    'label':     y,
    'oos_pred':  oos_pred,
    'oos_prob':  oos_prob,
    'oos_fold':  oos_fold,
    'weight':    w,
    'ticker':    modelling['ticker'],
    't1':        t1,
})
oos_df['ret'] = modelling.get('ret', pd.Series(np.nan, index=modelling.index))
oos_df.to_parquet('data/processed/oos_predictions_pooled.parquet')
print(f'  Saved: data/processed/oos_predictions_pooled.parquet  {oos_df.shape}')


# ── Step 12b: OOS classification metrics (RF) ────────────────────────────────
sep('STEP 12b: OOS classification metrics — RF (pooled, all folds)')

y_true_arr = y.values
y_pred_arr = oos_pred.values.astype(int)
y_prob_arr = oos_prob.values

oos_prec   = precision_score(y_true_arr, y_pred_arr, zero_division=0)
oos_rec    = recall_score(y_true_arr, y_pred_arr, zero_division=0)
oos_f1     = f1_score(y_true_arr, y_pred_arr, zero_division=0)
oos_bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
oos_mcc    = matthews_corrcoef(y_true_arr, y_pred_arr)
oos_macro_f1 = f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
try:
    oos_auc = roc_auc_score(y_true_arr, y_prob_arr)
except Exception:
    oos_auc = float('nan')
try:
    oos_pr_auc = average_precision_score(
        (y_true_arr == 1).astype(int), y_prob_arr)
except Exception:
    oos_pr_auc = float('nan')

print(f'\n  RF OOS metrics (all {n_covered} samples):')
print(f'    Accuracy          : {oos_acc:.4f}  (majority baseline={majority_baseline:.4f})')
print(f'    Balanced Accuracy : {oos_bal_acc:.4f}  (random=0.5000)')
print(f'    MCC               : {oos_mcc:.4f}  (random=0.0000)')
print(f'    Macro F1          : {oos_macro_f1:.4f}')
print(f'    Precision (+1)    : {oos_prec:.4f}')
print(f'    Recall (+1)       : {oos_rec:.4f}')
print(f'    F1 (+1)           : {oos_f1:.4f}')
print(f'    AUC-ROC           : {oos_auc:.4f}')
print(f'    PR-AUC (+1)       : {oos_pr_auc:.4f}')
print(f'\n  Full classification report:')
print(classification_report(y_true_arr, y_pred_arr,
                             target_names=['label=-1', 'label=+1']))

# Per-fold precision / recall / F1
print(f'\n  Per-fold breakdown:')
print(f'  {"Fold":>4}  {"n":>5}  {"Acc":>6}  {"Prec":>6}  {"Rec":>6}  {"F1":>6}')
for fi in sorted(oos_df['oos_fold'].unique()):
    mask   = oos_df['oos_fold'] == fi
    yt     = y.values[mask]
    yp     = oos_pred.values[mask].astype(int)
    facc   = float((yp == yt).mean())
    fprec  = precision_score(yt, yp, zero_division=0)
    frec   = recall_score(yt, yp, zero_division=0)
    ff1    = f1_score(yt, yp, zero_division=0)
    print(f'  {fi:>4}  {mask.sum():>5}  {facc:.4f}  {fprec:.4f}  {frec:.4f}  {ff1:.4f}')

print(f'\n  Note: XGB precision/recall/F1 are not computed here because Phase 11')
print(f'  only runs an OOS prediction loop for the best RF. XGB CV accuracy')
print(f'  from 30 HP trials: best={xgb_result["best_mean_score"]:.4f} ± {xgb_result["best_std_score"]:.4f}.')
print(f'  Full XGB OOS metrics are available via Phase 12 meta-model diagnostics.')

# Store in best_params later (update the dict before json dump)
_rf_oos_metrics = {
    'accuracy':          oos_acc,
    'balanced_accuracy': oos_bal_acc,
    'mcc':               oos_mcc,
    'macro_f1':          oos_macro_f1,
    'precision':         oos_prec,
    'recall':            oos_rec,
    'f1':                oos_f1,
    'auc_roc':           oos_auc,
    'pr_auc':            oos_pr_auc,
}


# ── Step 13: Validation ───────────────────────────────────────────────────────
sep('STEP 13: Validation')

passed = 0
failed = 0

def check(label, cond):
    global passed, failed
    s = 'PASS' if cond else 'FAIL'
    if cond: passed += 1
    else: failed += 1; ERRORS.append(f'{label}: FAIL')
    print(f'  [{s}] {label}')

check('feature_importance_pooled.parquet saved',
      os.path.exists('data/processed/feature_importance_pooled.parquet'))
check('tuning_log_pooled.parquet saved',
      os.path.exists('data/processed/tuning_log_pooled.parquet'))
check('best_params_pooled.json saved',
      os.path.exists('models/best_params_pooled.json'))
check('oos_predictions_pooled.parquet saved',
      os.path.exists('data/processed/oos_predictions_pooled.parquet'))
check('importance df non-empty',
      len(importance_df) == len(feat_cols))
check(f'OOS covers all {len(X_red)} samples',
      n_covered == len(X_red))
check('OOS accuracy > 0.50',
      oos_acc > 0.50)
check('RF OOS precision > 0.0 (model predicts +1 at least sometimes)',
      oos_prec > 0.0)
check('RF OOS F1 > 0.0',
      oos_f1 > 0.0)
check('RF OOS AUC-ROC computed (not nan)',
      not np.isnan(oos_auc))
check('RF best balanced_acc > 0.50 (random baseline)',
      rf_result['best_mean_score'] > 0.50)
check('tuning log has 60 rows (30 RF + 30 XGB)',
      len(tuning_log) == 2 * N_ITER)
check('MDI fig saved',
      os.path.exists(os.path.join(FIGURES_DIR, 'phase11_mdi_importance.png')))
check('MDA fig saved',
      os.path.exists(os.path.join(FIGURES_DIR, 'phase11_mda_importance.png')))
check('SFI fig saved',
      os.path.exists(os.path.join(FIGURES_DIR, 'phase11_sfi_importance.png')))
check('HP tuning fig saved',
      os.path.exists(os.path.join(FIGURES_DIR, 'phase11_hp_tuning.png')))

sep()
if ERRORS:
    print(f'Phase 11 FAILED — {failed} check(s) failed:')
    for e in ERRORS: print(f'  {e}')
    sys.exit(1)
else:
    print(f'Phase 11 COMPLETE — {passed} checks passed.')
    print(f'  Features used  : {len(keep_feats)} / {len(feat_cols)} (dropped {len(drop_feats)})')
    print(f'  RF  best acc   : {rf_result["best_mean_score"]:.4f}  '
          f'(vs majority {majority_baseline:.4f})')
    print(f'  XGB best acc   : {xgb_result["best_mean_score"]:.4f}')
    print(f'  OOS accuracy   : {oos_acc:.4f}  ({n_covered}/{len(X_red)} covered)')
    print(f'  RF OOS precision: {oos_prec:.4f}')
    print(f'  RF OOS recall   : {oos_rec:.4f}')
    print(f'  RF OOS F1       : {oos_f1:.4f}')
    print(f'  RF OOS AUC-ROC  : {oos_auc:.4f}')
    print(f'  Alpha top-10   : {alpha_in_top10}')
    print(f'  RF  DSR        : {rf_dsr_info["dsr"]:.4f}')
    print(f'  XGB DSR        : {xgb_dsr_info["dsr"]:.4f}')
