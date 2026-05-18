"""
Phase 11: Final Modelling — Feature Importance, HP Tuning, OOS Predictions.

Wave-2 enhancements (on top of the original implementation):
  - XGB full OOS prediction loop (was HP-tuning only)
  - Isotonic probability calibration (CalibratedClassifierCV, inside each fold)
  - PCA on alpha block inside each OOS fold (95% variance, no leakage)
  - Ensemble: 0.5 * RF_calibrated + 0.5 * XGB_calibrated
  - Per-fold threshold calibration (maximise balanced accuracy on training probs)

Bug fixed: best_params_pooled.json is now saved AFTER OOS metrics are computed
           (the previous code referenced _rf_oos_metrics before it existed).

Steps:
  1  MDI / MDA / SFI feature importance on pooled dataset
  2  Tri-method consensus pruning (bottom-N in ALL three rankings)
  3  HP tuning: 30-trial randomised search for RF and XGB
  4  DSR over tuning trials
  5  Save tuning log
  6  HP tuning plot
  7  OOS predictions — calibrated RF+XGB ensemble, PCA alpha block, fold threshold
  8  OOS metrics comparison (RF raw vs ensemble)
  9  Save best_params_pooled.json (includes OOS metrics)
  10 Validation

Outputs:
  data/processed/feature_importance_pooled.parquet
  data/processed/tuning_log_pooled.parquet
  data/processed/oos_predictions_pooled.parquet
  models/best_params_pooled.json
  reports/figures/phase11_{mdi,mda,sfi,hp_tuning}.png
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

from sklearn.calibration   import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble      import RandomForestClassifier
from sklearn.base          import clone
from sklearn.metrics       import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, classification_report,
)
from xgboost               import XGBClassifier

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
N_ITER = 30
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

bottom_n  = max(1, int(np.ceil(n_feat * 0.10)))
threshold = n_feat - bottom_n + 1
is_bottom = (rank_df[['MDI_rank', 'MDA_rank', 'SFI_rank']] >= threshold)
drop_feats = list(rank_df.index[is_bottom.all(axis=1)])
keep_feats = [c for c in feat_cols if c not in drop_feats]

print(f'\n  Bottom-10% threshold rank : {threshold} (bottom {bottom_n} of {n_feat})')
print(f'  Dropped (bottom in all 3) : {len(drop_feats)} — {drop_feats}')
print(f'  Kept                      : {len(keep_feats)}')

top10_avg      = rank_df.head(10).index.tolist()
alpha_in_top10 = [f for f in top10_avg if f.startswith('alpha')]
print(f'\n  Alpha features in top-10 (avg rank): {alpha_in_top10}')
print(f'  TS features in top-10    (avg rank): {[f for f in top10_avg if not f.startswith("alpha")]}')


# ── Step 5: Save importance artefact ─────────────────────────────────────────
sep('STEP 5: Save feature importance parquet')

importance_df = pd.concat({'MDI': mdi, 'MDA': mda, 'SFI': sfi}, axis=1)
importance_df.columns = ['_'.join(c) for c in importance_df.columns]
importance_df = importance_df.join(rank_df)
importance_df['kept']     = ~importance_df.index.isin(drop_feats)
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

save_importance_fig(mdi, f'MDI — {len(feat_cols)} features (10-stock pooled)', '#2c7fb8',
                    'phase11_mdi_importance.png')
save_importance_fig(mda, 'MDA — neg_log_loss permutation (weighted)', '#e34a33',
                    'phase11_mda_importance.png')
save_importance_fig(sfi, 'SFI — single-feature neg_log_loss (weighted)', '#31a354',
                    'phase11_sfi_importance.png')


# ── Step 7: HP tuning — RF ────────────────────────────────────────────────────
sep(f'STEP 7: HP tuning — RF ({N_ITER} trials, {len(keep_feats)} features)')

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
    clf          = RandomForestClassifier(random_state=RNG, n_jobs=-1),
    X            = X_red,
    y            = y,
    param_dist   = rf_search_space,
    cv           = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01),
    n_iter       = N_ITER,
    sample_weight= w,
    scoring      = 'balanced_accuracy',
    random_state = RNG,
)
rf_time = time.time() - t0

print(f'  RF tuning done in {rf_time:.1f}s')
print(f'  Best mean acc : {rf_result["best_mean_score"]:.4f}  '
      f'std={rf_result["best_std_score"]:.4f}')
print(f'  Best params   : {rf_result["best_params"]}')


# ── Step 8: HP tuning — XGB ──────────────────────────────────────────────────
sep(f'STEP 8: HP tuning — XGB ({N_ITER} trials, {len(keep_feats)} features)')

xgb_search_space = {
    'n_estimators':    [100, 200, 300],
    'max_depth':       [3, 4, 5],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.6, 0.8, 1.0],
    'colsample_bytree':[0.5, 0.7, 1.0],
    'reg_lambda':      [1.0, 5.0, 10.0],
    'gamma':           [0.0, 0.1, 0.5],
}

t0 = time.time()
xgb_result = purged_random_search(
    clf          = XGBClassifier(
                       random_state=RNG, eval_metric='logloss', n_jobs=-1,
                   ),
    X            = X_red,
    y            = (y == 1).astype(int),   # XGB expects 0/1
    param_dist   = xgb_search_space,
    cv           = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01),
    n_iter       = N_ITER,
    sample_weight= w,
    scoring      = 'balanced_accuracy',
    random_state = RNG,
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

print(f'  RF  DSR={rf_dsr_info["dsr"]:.4f}  best_SR={rf_dsr_info["best_sr"]:.4f}  '
      f'n_trials={rf_dsr_info["n_trials"]}')
print(f'  XGB DSR={xgb_dsr_info["dsr"]:.4f}  best_SR={xgb_dsr_info["best_sr"]:.4f}  '
      f'n_trials={xgb_dsr_info["n_trials"]}')


# ── Step 10: Save tuning log ──────────────────────────────────────────────────
sep('STEP 10: Save tuning log')

rf_log  = log_trials(rf_result['trials'])
rf_log['model'] = 'rf'
xgb_log = log_trials(xgb_result['trials'])
xgb_log['model'] = 'xgb'
tuning_log = pd.concat([rf_log, xgb_log], ignore_index=True)
for col in tuning_log.select_dtypes('object').columns:
    tuning_log[col] = tuning_log[col].astype(str)
tuning_log.to_parquet('data/processed/tuning_log_pooled.parquet')
print(f'  Saved: data/processed/tuning_log_pooled.parquet  {tuning_log.shape}')


# ── Step 11: HP tuning plot ───────────────────────────────────────────────────
sep('STEP 11: HP tuning scatter plot')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, log_df, title, color in [
    (axes[0], rf_log,  f'RF — {N_ITER} trials',  '#2c7fb8'),
    (axes[1], xgb_log, f'XGB — {N_ITER} trials', '#e34a33'),
]:
    ax.scatter(range(len(log_df)), log_df['mean_score'],
               c=color, s=40, alpha=0.7, zorder=3)
    ax.errorbar(range(len(log_df)), log_df['mean_score'],
                yerr=log_df['std_score'], fmt='none', alpha=0.3, color=color)
    best_i = int(log_df['mean_score'].idxmax())
    ax.scatter([best_i], [log_df.loc[best_i, 'mean_score']],
               color='gold', s=120, zorder=5, marker='*',
               label=f'Best ({log_df.loc[best_i,"mean_score"]:.4f})')
    ax.axhline(majority_baseline, color='k', linestyle='--', alpha=0.6,
               label=f'Majority ({majority_baseline:.4f})')
    ax.set_xlabel('Trial'); ax.set_ylabel('Mean CV Balanced Accuracy')
    ax.set_title(title); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'phase11_hp_tuning.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f'  Saved: {path}')


# ── Step 12: OOS predictions — calibrated RF+XGB ensemble ────────────────────
sep('STEP 12: OOS — calibrated RF+XGB ensemble (PCA alpha block, fold threshold)')

best_rf_params  = rf_result['best_params'].copy()
best_xgb_params = xgb_result['best_params'].copy()

# Split keep_feats into per-stock TS and alpha subsets
ts_keep    = [c for c in keep_feats if not c.startswith('alpha')]
alpha_keep = [c for c in keep_feats if c.startswith('alpha')]

X_raw = X[keep_feats]   # unfilled — imputation happens per fold

oos_pred_ens  = pd.Series(np.nan, index=X_raw.index, dtype=float)
oos_prob_ens  = pd.Series(np.nan, index=X_raw.index, dtype=float)
oos_pred_rf   = pd.Series(np.nan, index=X_raw.index, dtype=float)
oos_prob_rf_s = pd.Series(np.nan, index=X_raw.index, dtype=float)
oos_pred_xgb  = pd.Series(np.nan, index=X_raw.index, dtype=float)
oos_prob_xgb_s= pd.Series(np.nan, index=X_raw.index, dtype=float)
oos_fold_s    = pd.Series(-1,     index=X_raw.index, dtype=int)
oos_thresh_s  = pd.Series(np.nan, index=X_raw.index, dtype=float)

fold_pca_n    = []   # track PCA component counts per fold

t0 = time.time()
for fold_i, (train_idx, test_idx) in enumerate(
        MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01).split(X_raw, y)):

    # Per-fold imputation: use training-fold column means only
    tr_means = X_raw.iloc[train_idx].mean()
    X_tr_raw = X_raw.iloc[train_idx].fillna(tr_means)
    X_te_raw = X_raw.iloc[test_idx].fillna(tr_means)
    y_tr     = y.iloc[train_idx]
    y_te     = y.iloc[test_idx]
    w_tr     = w.iloc[train_idx]

    # ── PCA on alpha block (fit on training fold only) ──────────────────────
    if len(alpha_keep) > 0:
        pca = PCA(n_components=0.95, svd_solver='full', random_state=RNG)
        alpha_pcs_tr = pca.fit_transform(X_tr_raw[alpha_keep].values)
        alpha_pcs_te = pca.transform(X_te_raw[alpha_keep].values)
        n_pca = alpha_pcs_tr.shape[1]
    else:
        alpha_pcs_tr = np.empty((len(X_tr_raw), 0))
        alpha_pcs_te = np.empty((len(X_te_raw), 0))
        n_pca = 0
    fold_pca_n.append(n_pca)

    X_tr = np.hstack([X_tr_raw[ts_keep].values, alpha_pcs_tr])
    X_te = np.hstack([X_te_raw[ts_keep].values, alpha_pcs_te])
    y_tr_01 = (y_tr.values == 1).astype(int)   # 0/1 for XGB and metrics

    # ── Calibrated RF ───────────────────────────────────────────────────────
    rf_calib = CalibratedClassifierCV(
        RandomForestClassifier(**best_rf_params, random_state=RNG, n_jobs=-1),
        method='isotonic', cv=3,
    )
    rf_calib.fit(X_tr, y_tr.values, sample_weight=w_tr.values)

    rf_prob_te = rf_calib.predict_proba(X_te)
    rf_pos_idx = list(rf_calib.classes_).index(1) if 1 in rf_calib.classes_ else 1
    rf_prob_te = rf_prob_te[:, rf_pos_idx]

    # ── Calibrated XGB ──────────────────────────────────────────────────────
    xgb_calib = CalibratedClassifierCV(
        XGBClassifier(**best_xgb_params, random_state=RNG,
                      eval_metric='logloss', n_jobs=-1),
        method='isotonic', cv=3,
    )
    xgb_calib.fit(X_tr, y_tr_01, sample_weight=w_tr.values)
    xgb_prob_te = xgb_calib.predict_proba(X_te)[:, 1]

    # ── Per-fold threshold calibration ─────────────────────────────────────
    # Evaluate ensemble probs on the *training* fold (calibrated, so OOF for
    # the isotonic step — no raw leakage from the test fold).
    rf_prob_tr  = rf_calib.predict_proba(X_tr)[:, rf_pos_idx]
    xgb_prob_tr = xgb_calib.predict_proba(X_tr)[:, 1]
    ens_prob_tr = 0.5 * rf_prob_tr + 0.5 * xgb_prob_tr

    best_thresh, best_ba = 0.5, 0.0
    for thresh in np.linspace(0.35, 0.65, 31):
        ba = balanced_accuracy_score(y_tr_01, (ens_prob_tr >= thresh).astype(int))
        if ba > best_ba:
            best_ba, best_thresh = ba, thresh

    # ── Ensemble and final predictions ─────────────────────────────────────
    ens_prob_te  = 0.5 * rf_prob_te + 0.5 * xgb_prob_te
    ens_pred_te  = np.where(ens_prob_te >= best_thresh, 1, -1)
    rf_pred_te   = np.where(rf_prob_te  >= 0.5, 1, -1)
    xgb_pred_te  = np.where(xgb_prob_te >= 0.5, 1, -1)

    acc_rf  = float((rf_pred_te  == y_te.values).mean())
    acc_xgb = float((xgb_pred_te == y_te.values).mean())
    acc_ens = float((ens_pred_te  == y_te.values).mean())
    print(f'  Fold {fold_i}: n={len(test_idx):4d}  pca={n_pca:2d}  '
          f'thresh={best_thresh:.2f}  '
          f'RF={acc_rf:.4f}  XGB={acc_xgb:.4f}  Ens={acc_ens:.4f}')

    # ── Store ───────────────────────────────────────────────────────────────
    oos_pred_ens.iloc[test_idx]   = ens_pred_te.astype(float)
    oos_prob_ens.iloc[test_idx]   = ens_prob_te
    oos_pred_rf.iloc[test_idx]    = rf_pred_te.astype(float)
    oos_prob_rf_s.iloc[test_idx]  = rf_prob_te
    oos_pred_xgb.iloc[test_idx]   = xgb_pred_te.astype(float)
    oos_prob_xgb_s.iloc[test_idx] = xgb_prob_te
    oos_fold_s.iloc[test_idx]     = fold_i
    oos_thresh_s.iloc[test_idx]   = best_thresh

elapsed_oos = time.time() - t0
n_covered   = oos_pred_ens.notna().sum()

print(f'\n  OOS elapsed      : {elapsed_oos:.1f}s')
print(f'  Covered          : {n_covered} / {len(X_raw)}')
print(f'  PCA components   : {fold_pca_n}  (per fold)')

oos_df = pd.DataFrame({
    'label':         y,
    'oos_pred':      oos_pred_ens,      # ensemble (primary, used by downstream)
    'oos_prob':      oos_prob_ens,      # ensemble probability
    'oos_prob_rf':   oos_prob_rf_s,
    'oos_prob_xgb':  oos_prob_xgb_s,
    'oos_pred_rf':   oos_pred_rf,
    'oos_pred_xgb':  oos_pred_xgb,
    'oos_threshold': oos_thresh_s,
    'oos_fold':      oos_fold_s,
    'weight':        w,
    'ticker':        modelling['ticker'],
    't1':            t1,
})
oos_df['ret'] = modelling.get('ret', pd.Series(np.nan, index=modelling.index))
oos_df.to_parquet('data/processed/oos_predictions_pooled.parquet')
print(f'  Saved: data/processed/oos_predictions_pooled.parquet  {oos_df.shape}')


# ── Step 12b: OOS metrics comparison ─────────────────────────────────────────
sep('STEP 12b: OOS metrics — RF vs XGB vs Ensemble')

def _metrics(y_true, y_pred, y_prob, name):
    acc  = float((y_pred == y_true).mean())
    ba   = balanced_accuracy_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mf1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:   auc = roc_auc_score(y_true, y_prob)
    except: auc = float('nan')
    try:   prauc = average_precision_score((y_true==1).astype(int), y_prob)
    except: prauc = float('nan')
    print(f'\n  [{name}]')
    print(f'    Accuracy          : {acc:.4f}  (majority={majority_baseline:.4f})')
    print(f'    Balanced Accuracy : {ba:.4f}')
    print(f'    MCC               : {mcc:.4f}')
    print(f'    Macro F1          : {mf1:.4f}')
    print(f'    Precision(+1)     : {prec:.4f}  Recall(+1): {rec:.4f}  F1: {f1:.4f}')
    print(f'    AUC-ROC           : {auc:.4f}  PR-AUC: {prauc:.4f}')
    return dict(accuracy=acc, balanced_accuracy=ba, mcc=mcc, macro_f1=mf1,
                precision=prec, recall=rec, f1=f1, auc_roc=auc, pr_auc=prauc)

y_arr = y.values

rf_oos_metrics  = _metrics(y_arr, oos_pred_rf.values.astype(int),
                            oos_prob_rf_s.values,  'RF calibrated')
xgb_oos_metrics = _metrics(y_arr, oos_pred_xgb.values.astype(int),
                            oos_prob_xgb_s.values, 'XGB calibrated')
ens_oos_metrics = _metrics(y_arr, oos_pred_ens.values.astype(int),
                            oos_prob_ens.values,   'Ensemble (RF+XGB, thresh)')

print(f'\n  Full classification report — Ensemble:')
print(classification_report(y_arr, oos_pred_ens.values.astype(int),
                             target_names=['label=-1', 'label=+1']))

print(f'\n  Per-fold ensemble breakdown:')
print(f'  {"Fold":>4}  {"n":>5}  {"Thresh":>6}  {"Acc":>6}  '
      f'{"Prec":>6}  {"Rec":>6}  {"F1":>6}')
for fi in sorted(oos_df['oos_fold'].unique()):
    mask  = oos_df['oos_fold'] == fi
    yt    = y_arr[mask.values]
    yp    = oos_pred_ens.values[mask.values].astype(int)
    thr   = float(oos_thresh_s.values[mask.values][0])
    facc  = float((yp == yt).mean())
    fprec = precision_score(yt, yp, zero_division=0)
    frec  = recall_score(yt, yp, zero_division=0)
    ff1   = f1_score(yt, yp, zero_division=0)
    print(f'  {fi:>4}  {mask.sum():>5}  {thr:.2f}    '
          f'{facc:.4f}  {fprec:.4f}  {frec:.4f}  {ff1:.4f}')

oos_acc = ens_oos_metrics['accuracy']


# ── Step 13: Save best_params_pooled.json ─────────────────────────────────────
sep('STEP 13: Save best_params_pooled.json')

best_params = {
    'rf': {
        'params':       best_rf_params,
        'mean_score':   rf_result['best_mean_score'],
        'std_score':    rf_result['best_std_score'],
        'n_features':   len(keep_feats),
        'dsr':          rf_dsr_info['dsr'],
        'best_sr':      rf_dsr_info['best_sr'],
        'n_trials':     N_ITER,
        'oos_metrics':  rf_oos_metrics,
    },
    'xgb': {
        'params':       best_xgb_params,
        'mean_score':   xgb_result['best_mean_score'],
        'std_score':    xgb_result['best_std_score'],
        'n_features':   len(keep_feats),
        'dsr':          xgb_dsr_info['dsr'],
        'best_sr':      xgb_dsr_info['best_sr'],
        'n_trials':     N_ITER,
        'oos_metrics':  xgb_oos_metrics,
    },
    'ensemble': {
        'oos_metrics':  ens_oos_metrics,
        'pca_components_per_fold': fold_pca_n,
    },
    'feature_selection': {
        'n_total':           len(feat_cols),
        'n_kept':            len(keep_feats),
        'n_dropped':         len(drop_feats),
        'dropped':           drop_feats,
        'kept':              keep_feats,
        'ts_keep':           ts_keep,
        'alpha_keep':        alpha_keep,
        'alpha_in_top10_avg': alpha_in_top10,
    },
    'majority_baseline': majority_baseline,
}

with open('models/best_params_pooled.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print(f'  Saved: models/best_params_pooled.json')

print(f'\n  Summary:')
print(f'  {"Model":10s}  {"Acc":7s}  {"BA":7s}  {"MCC":7s}  {"DSR":7s}')
print(f'  {"-"*10}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*7}')
for nm, met, dsr in [
    ('RF',       rf_oos_metrics,  rf_dsr_info),
    ('XGB',      xgb_oos_metrics, xgb_dsr_info),
    ('Ensemble', ens_oos_metrics, {'dsr': float('nan')}),
]:
    print(f'  {nm:10s}  {met["accuracy"]:.4f}   {met["balanced_accuracy"]:.4f}   '
          f'{met["mcc"]:.4f}   {dsr["dsr"]:.4f}')


# ── Step 14: Validation ───────────────────────────────────────────────────────
sep('STEP 14: Validation')

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
check(f'OOS covers all {len(X_raw)} samples',
      n_covered == len(X_raw))
check('Ensemble OOS accuracy > 0.50',
      oos_acc > 0.50)
check('Ensemble balanced accuracy > 0.50',
      ens_oos_metrics['balanced_accuracy'] > 0.50)
check('Ensemble precision > 0.0 (predicts +1 at least sometimes)',
      ens_oos_metrics['precision'] > 0.0)
check('RF OOS AUC-ROC computed',
      not np.isnan(rf_oos_metrics['auc_roc']))
check('Ensemble OOS AUC-ROC computed',
      not np.isnan(ens_oos_metrics['auc_roc']))
check('RF HP best balanced_acc > 0.50',
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
check('oos_predictions has ensemble + RF + XGB columns',
      all(c in oos_df.columns for c in
          ['oos_pred', 'oos_prob', 'oos_pred_rf', 'oos_pred_xgb']))
check('PCA reduced alpha block (at least 1 fold < n_alpha_keep)',
      any(n < len(alpha_keep) for n in fold_pca_n) if alpha_keep else True)

sep()
if ERRORS:
    print(f'Phase 11 FAILED — {failed} check(s) failed:')
    for e in ERRORS: print(f'  {e}')
    sys.exit(1)
else:
    print(f'Phase 11 COMPLETE — {passed} checks passed.')
    print(f'  Features used  : {len(keep_feats)} / {len(feat_cols)} (dropped {len(drop_feats)})')
    print(f'  Alpha keep     : {len(alpha_keep)}  →  PCA {fold_pca_n} components (per fold)')
    print(f'  RF  HP acc     : {rf_result["best_mean_score"]:.4f}')
    print(f'  XGB HP acc     : {xgb_result["best_mean_score"]:.4f}')
    print(f'  OOS accuracy   : RF={rf_oos_metrics["accuracy"]:.4f}  '
          f'XGB={xgb_oos_metrics["accuracy"]:.4f}  '
          f'Ens={ens_oos_metrics["accuracy"]:.4f}')
    print(f'  OOS bal. acc   : RF={rf_oos_metrics["balanced_accuracy"]:.4f}  '
          f'XGB={xgb_oos_metrics["balanced_accuracy"]:.4f}  '
          f'Ens={ens_oos_metrics["balanced_accuracy"]:.4f}')
    print(f'  OOS MCC        : RF={rf_oos_metrics["mcc"]:.4f}  '
          f'XGB={xgb_oos_metrics["mcc"]:.4f}  '
          f'Ens={ens_oos_metrics["mcc"]:.4f}')
    print(f'  RF  DSR        : {rf_dsr_info["dsr"]:.4f}')
    print(f'  XGB DSR        : {xgb_dsr_info["dsr"]:.4f}')
    print(f'  Alpha top-10   : {alpha_in_top10}')
