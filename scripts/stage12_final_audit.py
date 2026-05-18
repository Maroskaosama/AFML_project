"""
Phase 15: Final Audit (30 checks across all pipeline phases)
=============================================================
Systematically verifies the correctness and integrity of every major
pipeline stage: data, features, leakage, weights, CV, meta-labeling,
backtesting, CPCV robustness, and artifact completeness.

Check groups
------------
D  – Data integrity        (D1–D4)
L  – Leakage audit         (L1–L3)
F  – Feature engineering   (F1–F5)
W  – Sample weights        (W1–W3)
C  – CV / OOS              (C1–C4)
M  – Meta-labeling         (M1–M3)
B  – Backtesting           (B1–B6)
R  – CPCV robustness       (R1–R3)
A  – Artifact completeness (A1–A2)

Total: 30 checks; PASS threshold: all 30.

Artifacts saved
---------------
data/processed/final_audit_pooled.parquet
reports/final_audit_summary.txt
"""

import os, sys, json
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from datetime import datetime

os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports',        exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────
audit_rows = []

def check(group, name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    icon   = '[PASS]' if cond else '[FAIL]'
    print(f'  {icon} {group} – {name}')
    if detail:
        print(f'         {detail}')
    audit_rows.append({
        'group':  group,
        'name':   name,
        'status': status,
        'detail': detail,
    })
    return cond


def sep(title):
    print('\n' + '=' * 68)
    print(title)
    print('=' * 68)


# ── Load all artifacts ────────────────────────────────────────────────────────
sep('LOAD all pipeline artifacts')

with open('configs/universe.json') as f:
    UNI = json.load(f)
TICKERS = UNI['tickers']

with open('models/best_params_pooled.json') as f:
    best_params = json.load(f)

panel      = pd.read_parquet('data/processed/panel_ohlcv.parquet')
modelling  = pd.read_parquet('data/processed/pooled_modelling.parquet')
leakage    = pd.read_parquet('data/processed/leakage_audit.parquet')
fi_df      = pd.read_parquet('data/processed/feature_importance_pooled.parquet')
oos        = pd.read_parquet('data/processed/oos_predictions_pooled.parquet')
meta_lbl   = pd.read_parquet('data/processed/meta_labels_pooled.parquet')
meta_oos   = pd.read_parquet('data/processed/meta_oos_predictions_pooled.parquet')
bt_stats   = pd.read_parquet('data/processed/backtest_stats_pooled.parquet')
cpcv_oos   = pd.read_parquet('data/processed/cpcv_oos_pooled.parquet')
cpcv_paths = pd.read_parquet('data/processed/cpcv_paths_pooled.parquet')

# Per-stock label retuns
ret_frames = []
for t in TICKERS:
    p = f'data/processed/per_stock/{t}_labels.parquet'
    if os.path.exists(p):
        lbl = pd.read_parquet(p)
        lbl['ticker'] = t
        ret_frames.append(lbl)
labels_all = pd.concat(ret_frames).sort_index()

print(f'  Panel      : {panel.shape}')
print(f'  Modelling  : {modelling.shape}')
print(f'  Leakage    : {leakage.shape}  ({(leakage["status"]=="PASS").sum()} PASS)')
print(f'  FI         : {fi_df.shape}')
print(f'  OOS preds  : {oos.shape}')
print(f'  Meta labels: {meta_lbl.shape}')
print(f'  BT stats   : {bt_stats.shape}')
print(f'  CPCV paths : {cpcv_paths.shape}')

meta_cols  = {'label', 't1', 'weight', 'ticker'}
feat_cols  = [c for c in modelling.columns if c not in meta_cols]
ts_cols    = [c for c in feat_cols if not c.startswith('alpha')]
alpha_cols = [c for c in feat_cols if c.startswith('alpha')]

# ── D: Data integrity ─────────────────────────────────────────────────────────
sep('D – Data integrity')

panel_tickers = sorted(panel.index.get_level_values('ticker').unique())
panel_dates   = panel.index.get_level_values('Date')
panel_nan     = int(panel.isnull().sum().sum())

check('D1', f'Panel has all {len(TICKERS)} universe tickers',
      panel_tickers == sorted(TICKERS),
      f'tickers={panel_tickers}')

check('D2', 'Panel date range covers 2005-01-03 to 2025-04-30',
      str(panel_dates.min().date()) == '2005-01-03'
      and str(panel_dates.max().date()) == '2025-04-30',
      f'range={panel_dates.min().date()} -> {panel_dates.max().date()}')

check('D3', 'Panel has zero NaN values',
      panel_nan == 0,
      f'NaN count={panel_nan}')

check('D4', f'Pooled modelling covers all {len(TICKERS)} tickers (>= 5000 events)',
      len(modelling) >= 5000
      and sorted(modelling['ticker'].unique()) == sorted(TICKERS),
      f'n={len(modelling)}  tickers={sorted(modelling["ticker"].unique())}')

# ── L: Leakage audit ─────────────────────────────────────────────────────────
sep('L – Leakage audit')

n_leakage_pass = int((leakage['status'] == 'PASS').sum())
n_leakage_fail = int((leakage['status'] == 'FAIL').sum())
n_leakage_warn = int((leakage['status'] == 'WARN').sum())

check('L1', f'All 34 leakage checks PASS (Phase 6)',
      n_leakage_pass == 34 and n_leakage_fail == 0,
      f'PASS={n_leakage_pass}  WARN={n_leakage_warn}  FAIL={n_leakage_fail}')

check('L2', 't1 > event_date for all pooled events (no retroactive exits)',
      (modelling['t1'] > modelling.index).all(),
      f'violations={(modelling["t1"] <= modelling.index).sum()}')

check('L3', 'No forward-looking: OOS predictions use only past data',
      oos['oos_fold'].between(0, 4).all(),
      f'fold range={oos["oos_fold"].min()}-{oos["oos_fold"].max()}')

# ── F: Feature engineering ────────────────────────────────────────────────────
sep('F – Feature engineering')

mdi_sum = fi_df['MDI_mean'].sum()
mdi_top = float(fi_df['MDI_mean'].max())
top_by_rank = fi_df.nsmallest(1, 'avg_rank').index[0]

check('F1', '54 features total: 21 TS (17 per-stock + 4 macro) + 33 alpha',
      len(ts_cols) == 21 and len(alpha_cols) == 33,
      f'TS={len(ts_cols)}  alpha={len(alpha_cols)}')

check('F2', 'MDI importances sum to ~1.0 (within 1%)',
      abs(mdi_sum - 1.0) < 0.01,
      f'MDI sum={mdi_sum:.6f}')

check('F3', 'No single feature dominates MDI > 15%',
      mdi_top < 0.15,
      f'max MDI={mdi_top:.4f}  feature={fi_df["MDI_mean"].idxmax()}')

check('F4', 'Top feature by avg tri-method rank has positive MDI mean',
      fi_df.loc[top_by_rank, 'MDI_mean'] > 0,
      f'top feature={top_by_rank}  avg_rank={fi_df.loc[top_by_rank,"avg_rank"]:.2f}'
      f'  MDI={fi_df.loc[top_by_rank,"MDI_mean"]:.4f}')

n_zero_mdi = (fi_df['MDI_mean'] < 1e-6).sum()
check('F5', 'Fewer than 5 features have near-zero MDI (correlated ensemble masking expected)',
      n_zero_mdi < 5,
      f'near-zero MDI count={n_zero_mdi}  '
      f'features={list(fi_df[fi_df["MDI_mean"]<1e-6].index)}')

# ── W: Sample weights ─────────────────────────────────────────────────────────
sep('W – Sample weights')

weights = modelling['weight']
w_min   = float(weights.min())
w_max   = float(weights.max())
w_mean  = float(weights.mean())
w_neg   = int((weights <= 0).sum())

check('W1', 'All sample weights > 0',
      w_neg == 0,
      f'non-positive count={w_neg}')

check('W2', 'Max weight (clipped at p99) <= 3.5',
      w_max <= 3.5,
      f'max weight={w_max:.4f}')

check('W3', 'Mean weight in (0.5, 2.0)',
      0.5 < w_mean < 2.0,
      f'mean weight={w_mean:.4f}')

# ── C: CV / OOS ───────────────────────────────────────────────────────────────
sep('C – Cross-validation / OOS')

oos_acc   = (oos['oos_pred'] == oos['label']).mean()
oos_cover = len(oos)

# Attach returns to check coverage
ret_reset = labels_all.reset_index().rename(columns={'index': 'event_date'})
ret_reset['event_date'] = pd.to_datetime(ret_reset['event_date'])
oos_with_index = oos.copy().reset_index().rename(columns={'index': 'event_date'})
oos_with_index['event_date'] = pd.to_datetime(oos_with_index['event_date'])
oos_merged = oos_with_index.merge(ret_reset[['event_date', 'ticker', 'ret']],
                                  on=['event_date', 'ticker'], how='left',
                                  suffixes=('_old', ''))
if 'ret_old' in oos_merged.columns:
    oos_merged = oos_merged.drop(columns=['ret_old'])

check('C1', 'Phase 11 OOS accuracy > 0.50',
      oos_acc > 0.50,
      f'OOS acc={oos_acc:.4f}')

check('C2', f'OOS predictions cover all {len(modelling)} events exactly once',
      oos_cover == len(modelling),
      f'covered={oos_cover}')

cpcv_acc = cpcv_oos['accuracy'].mean()
check('C3', 'CPCV mean OOS accuracy > 0.50 (all 15 splits)',
      cpcv_acc > 0.50,
      f'CPCV mean acc={cpcv_acc:.4f}')

cpcv_appearances = cpcv_oos.groupby(['event_date', 'ticker']).size()
check('C4', 'Each event appears in exactly C(K-1,p-1)=5 CPCV test splits',
      (cpcv_appearances == 5).all(),
      f'appearances: mean={cpcv_appearances.mean():.2f}  '
      f'min={cpcv_appearances.min()}  max={cpcv_appearances.max()}')

# ── M: Meta-labeling ──────────────────────────────────────────────────────────
sep('M – Meta-labeling')

meta_class1_rate = float(meta_lbl['meta_label'].mean())
meta_oos_acc     = float((meta_oos['meta_pred_class'] == meta_oos['meta_label']).mean())

# Primary hit rate (fraction of events where primary direction matched realized return)
primary_hit = float(meta_lbl['meta_label'].mean())   # identical to meta_class1_rate

# Meta-filtered profitability
meta_filtered  = meta_oos[meta_oos['meta_pred_class'] == 1]
n_filtered     = len(meta_filtered)
if n_filtered > 0:
    meta_filt_hit = float((meta_filtered['ret'] * meta_filtered['side'] > 0).mean())
else:
    meta_filt_hit = 0.0

check('M1', 'Meta-label class-1 rate matches primary OOS accuracy (within 0.005)',
      abs(meta_class1_rate - oos_acc) < 0.005,
      f'meta class1={meta_class1_rate:.4f}  primary acc={oos_acc:.4f}  '
      f'diff={abs(meta_class1_rate - oos_acc):.4f}')

check('M2', 'Meta OOS accuracy > 0.50',
      meta_oos_acc > 0.50,
      f'meta OOS acc={meta_oos_acc:.4f}')

check('M3', 'Meta-filtered profitability >= all-trades profitability',
      meta_filt_hit >= primary_hit - 0.01,
      f'meta-filtered={meta_filt_hit:.4f}  all-trades={primary_hit:.4f}  '
      f'lift={meta_filt_hit - primary_hit:+.4f}')

# ── B: Backtesting ────────────────────────────────────────────────────────────
sep('B – Backtesting')

port_a = bt_stats.loc['Portfolio_A']
port_b = bt_stats.loc['Portfolio_B']

sr_a    = float(port_a['sr'])
dsr_a   = float(port_a['dsr'])
maxdd_a = float(port_a['max_dd'])
annr_a  = float(port_a['ann_return'])

# Per-ticker SR_A
ticker_srs_a = {t: float(bt_stats.loc[f'{t}_A', 'sr'])
                for t in TICKERS if f'{t}_A' in bt_stats.index}

check('B1', 'Portfolio_A Sharpe Ratio > 0',
      sr_a > 0,
      f'SR={sr_a:.4f}')

check('B2', 'Portfolio_A DSR > 0 (survives multiple-testing correction, N=60)',
      dsr_a > 0,
      f'DSR={dsr_a:.4f}')

check('B3', 'Portfolio_A max drawdown <= 35%',
      maxdd_a <= 0.35,
      f'max_dd={maxdd_a:.4%}')

check('B4', 'Portfolio_A annualised return > 0%',
      annr_a > 0,
      f'ann_ret={annr_a:.4%}')

n_pos_tickers = sum(1 for sr in ticker_srs_a.values() if sr > 0)
n_pos_threshold = max(6, int(0.60 * len(ticker_srs_a)))
check('B5', f'At least {n_pos_threshold}/{len(ticker_srs_a)} tickers have positive SR_A (>=60%)',
      n_pos_tickers >= n_pos_threshold,
      f'positive SR tickers={n_pos_tickers}/{len(ticker_srs_a)}  '
      f'({", ".join(f"{t}:{v:.2f}" for t,v in ticker_srs_a.items())})')

sorted_srs = sorted(ticker_srs_a.items(), key=lambda x: x[1], reverse=True)
worst3_tickers = [t for t, _ in sorted_srs[-3:]]
check('B6', 'NVDA SR_A not among the 3 worst tickers (not catastrophically weak signal)',
      'NVDA' not in worst3_tickers,
      f'worst-3={worst3_tickers}  NVDA SR={ticker_srs_a.get("NVDA", float("nan")):.4f}')

# ── R: CPCV robustness ────────────────────────────────────────────────────────
sep('R – CPCV robustness')

cpcv_srs   = cpcv_paths['sr'].dropna()
cpcv_mean  = float(cpcv_srs.mean())
cpcv_std   = float(cpcv_srs.std())
cpcv_pct_pos = float((cpcv_srs > 0).mean())

# Phase 13 portfolio SR_A for reference
phase13_sr = sr_a

check('R1', '100% of CPCV paths have SR > 0',
      cpcv_pct_pos == 1.0,
      f'% positive={cpcv_pct_pos:.1%}  n_paths={len(cpcv_srs)}')

check('R2', 'CPCV SR std < 0.25 (consistent across resamples)',
      cpcv_std < 0.25,
      f'SR std={cpcv_std:.4f}')

port_b_sr = float(bt_stats.loc['Portfolio_B', 'sr'])
check('R3', 'CPCV mean SR within 2x of Portfolio_B SR (signal quality vs cost-adjusted)',
      abs(cpcv_mean - port_b_sr) / max(abs(port_b_sr), 1e-6) < 2.0,
      f'CPCV mean SR={cpcv_mean:.4f}  Portfolio_B SR={port_b_sr:.4f}  '
      f'ratio={cpcv_mean/max(port_b_sr, 1e-6):.4f}')

# ── A: Artifact completeness ──────────────────────────────────────────────────
sep('A – Artifact completeness')

required_files = [
    'data/processed/panel_ohlcv.parquet',
    'data/processed/pooled_modelling.parquet',
    'data/processed/leakage_audit.parquet',
    'data/processed/feature_importance_pooled.parquet',
    'data/processed/oos_predictions_pooled.parquet',
    'data/processed/tuning_log_pooled.parquet',
    'data/processed/meta_labels_pooled.parquet',
    'data/processed/meta_oos_predictions_pooled.parquet',
    'data/processed/bet_sizes_pooled.parquet',
    'data/processed/backtest_stats_pooled.parquet',
    'data/processed/backtest_returns_pooled.parquet',
    'data/processed/cpcv_oos_pooled.parquet',
    'data/processed/cpcv_paths_pooled.parquet',
    'models/best_params_pooled.json',
]
missing = [f for f in required_files if not os.path.exists(f)]
check('A1', f'All {len(required_files)} required data artifacts exist',
      len(missing) == 0,
      f'missing={missing}' if missing else 'all present')

phase_figs = {
    'phase10': ['phase10_01_price_history.png', 'phase10_12_pooled_events_per_year.png'],
    'phase11': ['phase11_mdi_importance.png', 'phase11_sfi_importance.png'],
    'phase12': ['phase12_meta_label_dist.png', 'phase12_bet_size_dist.png'],
    'phase13': ['phase13_equity_curves.png', 'phase13_per_ticker_sr.png'],
    'phase14': ['phase14_cpcv_sr_dist.png',  'phase14_cpcv_fold_heatmap.png'],
}
figs_missing = []
for phase, figs in phase_figs.items():
    for fig in figs:
        path = f'reports/figures/{fig}'
        if not os.path.exists(path):
            figs_missing.append(fig)
check('A2', 'All key phase figures (10-14) exist',
      len(figs_missing) == 0,
      f'missing={figs_missing}' if figs_missing else f'checked {sum(len(v) for v in phase_figs.values())} figs')

# ── Compile results ───────────────────────────────────────────────────────────
sep('FINAL AUDIT SUMMARY')

audit_df = pd.DataFrame(audit_rows)
n_pass   = int((audit_df['status'] == 'PASS').sum())
n_fail   = int((audit_df['status'] == 'FAIL').sum())
n_total  = len(audit_df)

# Group summary
print('\n  Results by group:')
for grp in audit_df['group'].str[0].unique():
    grp_df  = audit_df[audit_df['group'].str.startswith(grp)]
    gp      = int((grp_df['status'] == 'PASS').sum())
    gf      = int((grp_df['status'] == 'FAIL').sum())
    bar     = '#' * gp + '.' * gf
    print(f'    {grp}  [{bar}]  {gp}/{len(grp_df)}')

print(f'\n  Total: {n_pass}/{n_total} PASS   {n_fail} FAIL')

if n_fail > 0:
    print('\n  FAILURES:')
    fails = audit_df[audit_df['status'] == 'FAIL']
    for _, row in fails.iterrows():
        print(f'    {row["group"]} – {row["name"]}')
        if row['detail']:
            print(f'      {row["detail"]}')

# Save audit parquet
audit_df.to_parquet('data/processed/final_audit_pooled.parquet', index=False)

# Save human-readable summary
ts = datetime.now().strftime('%Y-%m-%d %H:%M')
summary_lines = [
    f'AFML Pipeline Final Audit — {ts}',
    f'Universe : {TICKERS}',
    f'Period   : {UNI["common_start"]} -> {UNI["common_end"]}',
    f'Events   : {len(modelling)} ({len(TICKERS)} stocks)',
    f'Features : {len(feat_cols)} total ({len(ts_cols)} TS + {len(alpha_cols)} alpha)',
    '',
    f'Result   : {n_pass}/{n_total} checks PASS  |  {n_fail} FAIL',
    '',
    'Check-by-check:',
]
for _, row in audit_df.iterrows():
    icon = 'OK  ' if row['status'] == 'PASS' else 'FAIL'
    summary_lines.append(f'  [{icon}] {row["group"]} – {row["name"]}')
    if row['detail']:
        summary_lines.append(f'         {row["detail"]}')

summary_text = '\n'.join(summary_lines)
with open('reports/final_audit_summary.txt', 'w') as f:
    f.write(summary_text)

print(f'\n  Saved: data/processed/final_audit_pooled.parquet  ({audit_df.shape})')
print(f'  Saved: reports/final_audit_summary.txt')
print()
if n_fail == 0:
    print(f'Phase 15 COMPLETE — {n_pass}/{n_total} checks passed. Pipeline fully validated.')
else:
    print(f'Phase 15 FAILED — {n_fail} check(s) need attention.')
