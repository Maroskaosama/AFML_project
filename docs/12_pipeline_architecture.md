# AFML Pipeline Architecture

**Universe:** AAPL, AMZN, BAC, GOOGL, JNJ, JPM, MSFT, NVDA, UNH, XOM · **Period:** 2005-01-03 → 2025-04-30 · **Events:** ~2,071 · **Features:** 50 (17 TS + 33 α)

---

## High-Level Data Flow

```
Raw OHLCV (yfinance)
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 01  Data Acquisition                                          │
│  • Download / validate 10 raw CSVs                                  │
│  • Build panel_ohlcv.parquet  (MultiIndex Date×ticker, ~51K rows)   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
┌──────────────────────┐                 ┌──────────────────────────┐
│ STAGE 02  Per-Stock  │                 │ STAGE 03  Alpha Engine   │
│ (×10 tickers)        │                 │  • Compute all 101 WQ    │
│  • CUSUM filter      │                 │    alphas on full panel   │
│  • Triple-barrier    │                 │  • Prune: NaN>40%, const, │
│    labels            │                 │    |corr|>0.85            │
│  • Sample weights    │                 │  • Select top 33 alphas  │
│  • Fracdiff (d*) +   │                 │  Output:                 │
│    17 TS features    │                 │   panel_alpha_features   │
│  • Pool across tickers│                │   _pruned.parquet        │
│  Outputs (per ticker):│                └────────────┬─────────────┘
│   _labels / _weights │                             │
│   _ts_features       │                             │
│   _clean             │                             │
└──────────┬───────────┘                             │
           │  pooled_labels / pooled_weights /        │
           │  pooled_ts_features                      │
           └──────────────────┬──────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 04  Leakage Validation                   34 checks (8 groups) │
│  L1 Alpha panel alignment   L2 TS feature causality                 │
│  L3 Event-label integrity   L4 Event-feature alignment              │
│  L5 Sample weight sanity    L6 Alpha temporal ordering              │
│  L7 CV temporal integrity   L8 Cross-sectional alpha universe       │
│  Output: leakage_audit.parquet    ← MUST ALL PASS                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 05  CV Validation  →  Build pooled_modelling.parquet          │
│  • Merge labels + weights + 17 TS + 33 α = 2,071 × 50 features     │
│  • MultiAssetPurgedKFold (5 folds, 1% embargo)                      │
│  • Baseline RF: TS-only and TS+alpha accuracy / log-loss            │
│  Output: pooled_modelling.parquet, cv_baseline_multistock.parquet   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 06  Sample Weight Investigation                               │
│  • Decompose: uniqueness × return_attribution × time_decay          │
│  • Identify p99 outliers, clip if needed                            │
│  Output: weight_analysis.parquet, pooled_weights_clipped.parquet    │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 07  Visualizations   (12 diagnostic figures → reports/)       │
│  price_history · returns_dist · cusum_events · label_dist           │
│  sample_weights · fracdiff_d_star · feature_corr (TS + all)         │
│  cv_fold_timeline · alpha_nan_rates · alpha_adf · pooled_events     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 08  Final Modelling                                           │
│  1. Feature importance:                                             │
│     • MDI  — RF (200 trees, depth 6) split impurity                 │
│     • MDA  — RF (100 trees, depth 4) permutation, purged CV         │
│     • SFI  — per-feature RF, purged CV                              │
│     • Tri-method consensus pruning → feature_importance.parquet     │
│  2. Hyperparameter tuning:                                          │
│     • RandomizedSearch 30 trials each: RF + XGB, purged CV          │
│     • Metric: neg_log_loss  → best_params_pooled.json               │
│  3. OOS predictions (5-fold purged):                                │
│     • Best RF → oos_predictions_pooled.parquet                      │
│     • OOS accuracy ~56.6% (XGB), ~56.0% (RF)                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 09  Meta-Labeling                                             │
│  • meta_label = 1 if ret × oos_pred > 0  (correct direction)        │
│  • Secondary RF (depth 3) on 50 features + side                     │
│  • Bet sizing: size = 2×meta_prob−1,  disc_signal = side × size     │
│  Output: meta_labels.parquet, meta_oos_predictions.parquet,         │
│          bet_sizes_pooled.parquet                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 10  Backtesting                                               │
│  Strategy A: signal = oos_pred (±1 binary)                          │
│  Strategy B: signal = disc_signal (continuous, meta-sized)          │
│  • Expand events → daily positions (forward-fill through t1)        │
│  • Apply 5 bps transaction costs                                    │
│  • Portfolio = equal-weight average of active tickers               │
│  Metrics: SR · PSR · DSR · Max DD · Calmar · Hit Ratio · PF         │
│  Result: Portfolio SR = 1.06,  DSR = 0.97                           │
│  Output: backtest_returns.parquet, backtest_stats.parquet           │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 11  CPCV Robustness  (K=6, p=2 → 15 paths)                   │
│  • Partition event dates into 6 time blocks                         │
│  • C(6,2)=15 train/test splits; reassemble into 15 full paths       │
│  • Purge + embargo per split  (same rules as stage 05)              │
│  • Equity curve + SR per path                                       │
│  Output: cpcv_oos.parquet, cpcv_paths.parquet (15-row SR table)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 12  Final Audit                       33 checks (9 groups)    │
│  D Data integrity (4)    L Leakage (3)    F Features (5)            │
│  W Weights (3)           C CV/OOS (4)     M Meta-labeling (3)       │
│  B Backtest (6)          R CPCV (3)       A Artifacts (2)           │
│  Output: final_audit_pooled.parquet    ← ALL 33 MUST PASS           │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 13  Final Report                                              │
│  AFML_multistock_report.md  +  5 CSV summary tables                 │
│  T_pipeline_summary · T_backtest_summary · T_feature_top10          │
│  T_cpcv_paths · T_final_audit                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage Reference Table

| Stage | Script | Key Inputs | Key Outputs | Checks |
|---|---|---|---|---|
| 01 | `stage01_data_acquisition.py` | yfinance / raw CSVs | `panel_ohlcv.parquet` | 6×10 ticker checks |
| 02 | `stage02_per_stock_pipeline.py` | raw CSVs, `universe.json` | `{TICKER}_{labels,weights,ts_features,clean}.parquet`, pooled variants | n_events ≥ 200 |
| 03 | `stage03_alpha_engine.py` | `panel_ohlcv.parquet` | `panel_alpha_features_pruned.parquet`, `alpha_diagnostics.parquet`, `selected_alphas.json` | 101→33 alpha pruning |
| 04 | `stage04_leakage_validation.py` | all stage 1–3 artifacts | `leakage_audit.parquet` | **34 checks** |
| 05 | `stage05_cv_validation.py` | pooled + pruned alphas | `pooled_modelling.parquet`, `cv_baseline_multistock.parquet` | fold integrity |
| 06 | `stage06_weight_investigation.py` | pooled weights + modelling | `weight_analysis.parquet`, `pooled_weights_clipped.parquet` | p99 outlier clip |
| 07 | `stage07_visualizations.py` | panel, modelling, alphas | `reports/figures/phase10_*.png` (12 figs) | — |
| 08 | `stage08_final_modelling.py` | `pooled_modelling.parquet` | `feature_importance_pooled.parquet`, `best_params_pooled.json`, `oos_predictions_pooled.parquet` | 30+30 HP trials |
| 09 | `stage09_meta_labeling.py` | OOS preds + per-stock returns | `meta_labels.parquet`, `bet_sizes_pooled.parquet` | meta acc, lift |
| 10 | `stage10_backtesting.py` | OOS preds, bet sizes, prices | `backtest_{returns,stats}_pooled.parquet` | SR, DSR, max DD |
| 11 | `stage11_cpcv.py` | modelling + best params + prices | `cpcv_{oos,paths}_pooled.parquet` | 15 paths profitable |
| 12 | `stage12_final_audit.py` | all artifacts | `final_audit_pooled.parquet` | **33 checks** |
| 13 | `stage13_final_report.py` | all artifacts | `.md` report + 5 CSV tables | — |

---

## Key Parameters

| Domain | Parameter | Value |
|---|---|---|
| Universe | Tickers | 10 (AAPL AMZN BAC GOOGL JNJ JPM MSFT NVDA UNH XOM) |
| Labeling | `pt_sl` | `[1.0, 1.0]` symmetric, `vertical_days=10`, `vol_span=50` |
| CUSUM | `target_events` | 500/stock (range 200–600) |
| Fracdiff | `d_range` | `[0.05, 0.10, …, 0.50]`, ADF p-value ≤ 0.05, `min_corr=0.85` |
| CV | `n_splits` | 5, `pct_embargo=0.01` |
| Alpha pruning | thresholds | NaN < 40%, std > 1e-8, \|corr\| < 0.85 → 33 survive |
| HP tuning | trials | 30 RF + 30 XGB = 60 total (DSR basis) |
| Backtesting | `cost_bps` | 5 |
| CPCV | `K=6, p=2` | C(6,2) = 15 splits → 15 equity-curve paths |

---

## Module Map (`src/`)

```
src/
├── pipeline/
│   ├── per_stock.py          run_per_stock_pipeline()
│   └── pooling.py            build_pooled_modelling_dataset()
├── alphas/
│   ├── formulas.py           alpha001–alpha101 definitions
│   ├── engine.py             compute_all_alphas()
│   ├── operators.py          ts_rank, ts_corr, rank_cs, scale_cs, …
│   ├── diagnostics.py        alpha selection / pruning logic
│   └── registry.py           SECTOR_MAP
├── labeling.py               get_daily_vol, triple_barrier, get_bins
├── features.py               17 TS features (momentum/vol/micro/entropy)
├── fracdiff.py               frac_diff_ffd, find_min_d
├── sample_weights.py         num_co_events, sample_tw, return_attribution, time_decay
├── cross_validation.py       MultiAssetPurgedKFold, cv_score
├── feature_importance.py     feat_imp_MDI, feat_imp_MDA, feat_imp_SFI
├── hyperparameter_tuning.py  purged_random_search, deflated_sharpe_ratio
├── backtesting.py            backtest_strategy, SR, PSR, DSR, calmar, hit_ratio
├── meta_labeling.py          secondary model, profitability lift
├── bet_sizing.py             get_signal, avg_active_signals, discrete_signal
└── structural_breaks.py      CUSUM filter
```

---

## Validation Gates

### Leakage Audit — Stage 04 (34 checks)

| Group | Checks | Description |
|---|---|---|
| L1 | 4 | Alpha panel alignment: date index, phantom rows, tickers, start/end dates |
| L2 | 1 | TS feature causality: fracdiff value at t0 uses only data ≤ t0 |
| L3 | 3 | Event-label integrity: t1 > t0, bins ∈ {−1,0,1}, no same-day exits |
| L4 | 3 | Event-feature alignment: dates match, no off-by-one, all events have features |
| L5 | 4 | Sample weight sanity: weights > 0, non-NaN, co-event counts valid |
| L6 | 2 | Alpha temporal ordering: alpha at t uses only data up to t |
| L7 | 6 | CV temporal integrity: no train-test overlap, purge and embargo enforced |
| L8 | 3 | Cross-sectional alpha universe: BAC/UNH signals reasonable, no extremes |

### Final Audit — Stage 12 (33 checks)

| Group | Checks | Coverage |
|---|---|---|
| D – Data | 4 | Panel tickers (10), date range, NaN count (0), event count (2,071) |
| L – Leakage | 3 | All 34 leakage checks PASS, t1 > event_date, OOS uses only past data |
| F – Features | 5 | No constant/all-NaN features, TS corr sensible, alpha NaN documented, fracdiff causal |
| W – Weights | 3 | Weights in (0, ∞), per-ticker distribution reasonable, clipping applied if needed |
| C – CV/OOS | 4 | Fold split integrity, no train-test overlap, purge/embargo enforced, OOS acc consistent |
| M – Meta | 3 | Meta-labels ∈ {0,1}, meta-model OOS acc, bet sizes in [0,1] |
| B – Backtest | 6 | Equity curves present (A & B), SR computed, max DD ≥ 0, cost_bps=5 applied |
| R – CPCV | 3 | 15 paths generated, all paths profitable (SR > 0), SR distribution reasonable |
| A – Artifacts | 2 | All expected output files exist, no missing data in key columns |

---

## Achieved Results

| Metric | Value |
|---|---|
| OOS Accuracy (XGB) | 56.6% |
| OOS Accuracy (RF) | 56.0% |
| Portfolio Sharpe (Strategy A) | 1.06 |
| Deflated Sharpe Ratio | 0.97 |
| Leakage checks | 34/34 PASS |
| Final audit | 33/33 PASS |
| CPCV paths profitable | 15/15 |
| Top alpha features (MDI) | alpha041, alpha028, alpha012, alpha009 |

---

## Data Directory Structure

```
data/
├── raw/
│   ├── {TICKER}_raw.csv               (10 files, OHLCV 2005–2025)
│   └── archive/                       (retired tickers: META, TSLA)
└── processed/
    ├── panel_ohlcv.parquet            (MultiIndex Date×ticker, ~51K rows)
    ├── panel_alpha_features.parquet   (all 101 alphas)
    ├── panel_alpha_features_pruned.parquet  (33 selected alphas)
    ├── alpha_diagnostics.parquet      (ADF, autocorr per alpha)
    ├── pooled_labels.parquet
    ├── pooled_weights.parquet
    ├── pooled_ts_features.parquet
    ├── pooled_modelling.parquet       (2,071 × 54: metadata + 50 features)
    ├── leakage_audit.parquet
    ├── cv_baseline_multistock.parquet
    ├── weight_analysis.parquet
    ├── feature_importance_pooled.parquet
    ├── tuning_log_pooled.parquet
    ├── oos_predictions_pooled.parquet
    ├── meta_labels_pooled.parquet
    ├── meta_oos_predictions_pooled.parquet
    ├── bet_sizes_pooled.parquet
    ├── backtest_returns_pooled.parquet
    ├── backtest_stats_pooled.parquet
    ├── cpcv_oos_pooled.parquet
    ├── cpcv_paths_pooled.parquet
    ├── final_audit_pooled.parquet
    └── per_stock/
        ├── {TICKER}_clean.parquet     (10 files)
        ├── {TICKER}_labels.parquet    (10 files)
        ├── {TICKER}_weights.parquet   (10 files)
        └── {TICKER}_ts_features.parquet  (10 files)
```
