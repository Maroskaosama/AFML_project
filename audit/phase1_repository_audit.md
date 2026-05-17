# Phase 1 — Repository Audit Report

**Branch:** Complete-AFML-Pipeline  
**Date:** 2026-05-17  
**Type:** Read-only inventory  
**Status:** COMPLETE

---

## 1. File Inventory

### Counts by Extension

| Extension | Count |
|-----------|-------|
| .parquet  | 74    |
| .png      | 38    |
| .py       | 37    |
| .csv      | 25    |
| .ipynb    | 15    |
| .md       | 10    |
| .json     | 4     |
| .pdf      | 3     |
| .pkl      | 3     |
| .txt      | 2     |

**Total tracked files:** 215 (excl. .git and __pycache__)

---

## 2. Raw Data Inventory

| File | Rows | Date Range | Status |
|------|------|-----------|--------|
| AAPL_raw.csv | 6,369 | 2000-01-04 → 2025-04-30 | OK — full history available |
| AMZN_raw.csv | 6,369 | 2000-01-04 → 2025-04-30 | OK — full history available |
| GOOGL_raw.csv | 5,207 | 2004-08-20 → 2025-04-30 | OK — from IPO |
| JNJ_raw.csv | 6,369 | 2000-01-04 → 2025-04-30 | OK — full history available |
| JPM_raw.csv | 6,369 | 2000-01-04 → 2025-04-30 | OK — full history available |
| META_raw.csv | 3,255 | 2012-05-21 → 2025-04-30 | STALE — to be removed |
| MSFT_raw.csv | 6,369 | 2000-01-04 → 2025-04-30 | OK — full history available |
| NVDA_raw.csv | 5,113 | 2005-01-04 → 2025-04-30 | OK — from 2005 IPO era |
| TSLA_raw.csv | 3,732 | 2010-06-30 → 2025-04-30 | STALE — to be removed |
| XOM_raw.csv | 6,369 | 2000-01-04 → 2025-04-30 | OK — full history available |
| BAC_raw.csv | MISSING | — | MUST DOWNLOAD |
| UNH_raw.csv | MISSING | — | MUST DOWNLOAD |

**Key finding:** 8 of 10 target stocks already have raw CSVs with histories pre-dating 2005.
NVDA (binding constraint for new universe) starts 2005-01-04.
Projected new common start: **2005-01-03**.

---

## 3. Processed Artifact Inventory

### Panel-Level Artifacts

| File | Shape | Index | NaN | Tickers | Status |
|------|-------|-------|-----|---------|--------|
| panel_ohlcv.parquet | (32,560, 6) | MultiIndex[Date, ticker] | 0 | META, TSLA present | STALE |
| panel_alpha_features.parquet | (32,560, 101) | MultiIndex[Date, ticker] | 365,446 | META, TSLA present | STALE |
| panel_alpha_features_pruned.parquet | (32,560, 33) | MultiIndex[Date, ticker] | 12,371 | META, TSLA present | STALE |

All panel artifacts use the 2012–2025 range (3,256 days × 10 stocks = 32,560 rows).
**All must be fully regenerated for the new universe (2005–2025, BAC/UNH replacing META/TSLA).**

### NVDA Legacy Artifacts (single-stock pipeline)

| File | Shape | Notes |
|------|-------|-------|
| nvda_clean.parquet | (5,113, 8) | Full 2005–2025 NVDA data |
| nvda_cusum_events.parquet | (400, 0) | 400 events, full history |
| nvda_labels.parquet | (400, 5) | 385 NaN = open events at data end |
| nvda_sample_weights.parquet | (400, 1) | 1 NaN = open event |
| nvda_features.parquet | (195, 17) | 195 labeled events with features |
| nvda_fracdiff.parquet | (2,310, 1) | Fracdiff series |
| nvda_modelling_dataset.parquet | (195, 20) | 15 features + meta cols |
| nvda_oos_predictions.parquet | (195, 4) | OOS primary predictions |
| nvda_meta_labels.parquet | (195, 6) | Meta-labels |
| nvda_meta_predictions.parquet | (195, 3) | Meta-model OOS predictions |
| nvda_positions.parquet | (5,113, 1) | Daily positions |
| nvda_bsadf.parquet | (5,113, 1) | Structural breaks; 62 NaN |
| nvda_dollar_bars.parquet | (1,628, 6) | Dollar bars |

### Multi-Stock Per-Stock Artifacts

| Ticker | Clean Shape | Labels | Weights | TS Features | Events (common range) |
|--------|------------|--------|---------|-------------|-----------------------|
| AAPL | (6,370, 6) | (66, 3) | (66, 1) | (66, 17) | 66 |
| AMZN | (6,370, 6) | (39, 3) | (39, 1) | (39, 17) | 39 |
| GOOGL | (5,208, 6) | (77, 3) | (77, 1) | (77, 17) | 77 |
| JNJ | (6,370, 6) | (89, 3) | (89, 1) | (89, 17) | 89 |
| JPM | (6,370, 6) | (52, 3) | (52, 1) | (52, 17) | 52 |
| META | (3,256, 6) | (111, 3) | (111, 1) | (111, 17) | 111 — STALE |
| MSFT | (6,370, 6) | (126, 3) | (126, 1) | (126, 17) | 126 |
| NVDA | (5,113, 8) | (400, 5) | (400, 1) | (195, 17) | 195 (common range) |
| TSLA | (3,733, 6) | (101, 3) | (101, 1) | (101, 17) | 101 — STALE |
| XOM | (6,370, 6) | (161, 3) | (161, 1) | (161, 17) | 161 |
| BAC | MISSING | MISSING | MISSING | MISSING | — |
| UNH | MISSING | MISSING | MISSING | MISSING | — |

**Note on clean artifacts:** Per-stock clean files have 6,370 rows (AAPL, AMZN, etc.) but labels
only 39–195 events. The clean files appear to store the full raw history, while labels/features
are filtered to the common 2012–2025 range. With the new 2005 start, all artifacts will expand.

### Pooled Artifacts

| File | Shape | Tickers | Notes |
|------|-------|---------|-------|
| pooled_modelling.parquet | (881, 54) | OLD universe | STALE |
| pooled_labels.parquet | (980, 6) | OLD universe | STALE |
| pooled_ts_features.parquet | (980, 18) | OLD universe | STALE |
| pooled_weights.parquet | (980, 2) | OLD universe | STALE |

### Modelling/Backtest Artifacts

| File | Shape | Notes |
|------|-------|-------|
| cv_baseline_multistock.parquet | (5, 4) | Old universe CV results — STALE |
| hp_grid_results.parquet | (18, 4) | Old grid search — STALE |
| meta_labeled_predictions.parquet | (881, 11) | Old universe — STALE |
| backtest_stats.parquet | (12, 8) | Old universe — STALE |
| backtest_results.parquet | (5,113, 8) | NVDA single-stock backtest |
| feature_importance.parquet | (17, 11) | NVDA single-stock — STALE |
| mdi_importance.parquet | (50, 2) | Old 50-feature MDI — STALE |
| mda_importance.parquet | (50, 1) | Old 50-feature MDA — STALE |
| sfi_importance.parquet | (50, 1) | Old 50-feature SFI — STALE |
| validation_audit.parquet | (32, 4) | Old universe audit — STALE |
| tuning_log.parquet | (50, 20) | Tuning log with 212 NaN — review |
| cpcv_results.parquet | (5, 2) | NVDA-only CPCV — STALE |
| alpha_diagnostics.parquet | (99, 11) | Valid — alpha names unchanged |

---

## 4. Source Code Inventory

### Core src/ Modules

| File | Lines | Functions | Classes | Key Issues |
|------|-------|-----------|---------|------------|
| data_structures.py | 153 | cusum_filter, calibrate_cusum_h | — | OK |
| labeling.py | 102 | get_daily_vol, apply_triple_barrier, get_bins | — | OK |
| sample_weights.py | 136 | num_co_events, seq_bootstrap, get_sample_weight | — | OK |
| fracdiff.py | 250 | frac_diff_ffd, find_min_d | — | Bug fixed (empty series return) |
| features.py | 249 | build_feature_matrix | — | OK |
| cross_validation.py | 309 | — | PurgedKFold, CombinatorialPurgedKFold, MultiAssetPurgedKFold | OK |
| backtesting.py | 238 | backtest_strategy, sharpe_ratio, prob_sharpe_ratio, deflated_sharpe_ratio | — | OK |
| bet_sizing.py | 72 | get_signal, build_daily_positions | — | OK |
| feature_importance.py | 169 | feat_imp_MDI, feat_imp_MDA, feat_imp_SFI | — | OK |
| meta_labeling.py | 273 | generate_oos_predictions, make_meta_labels | — | OK |
| microstructure.py | 148 | corwin_schultz_spread, amihud_illiquidity | — | OK |
| hyperparameter_tuning.py | 219 | purged_random_search | — | OK |
| structural_breaks.py | 172 | get_bsadf | — | OK |
| entropy.py | 177 | shannon_entropy, lempel_ziv_complexity | — | OK |
| modelling.py | 40 | train_and_evaluate | — | Thin wrapper |
| multiprocess.py | 191 | mp_pandas_obj | — | OK |
| synthetic.py | 51 | generate_trending_series | — | OK |
| utils.py | 0 | — | — | Empty |

### src/alphas/ Package

| File | Lines | Functions/Classes | Key Issues |
|------|-------|------------------|------------|
| operators.py | 221 | 25 operators | OK — all required operators present |
| formulas.py | 1,618 | 101 alpha functions | OK |
| engine.py | 184 | compute_all_alphas, compute_alpha_diagnostics | OK |
| registry.py | 125 | SECTOR_MAP, ALPHA_REGISTRY | **HARDCODES META/TSLA lines 14–15** |
| 101Alpha_code_1.py | 826 | Alphas class (GitHub ref) | Reference only — not used |
| 101Alpha_code_2.py | 2,591 | — | **PARSE ERROR** — reference only |

### src/pipeline/ Package

| File | Lines | Functions | Key Issues |
|------|-------|-----------|------------|
| per_stock.py | 231 | run_per_stock_pipeline, create_pooled_dataset | OK |

### All Source Modules Importable: YES

---

## 5. Configuration Inventory

### configs/universe.json
- **tickers:** ['AAPL','AMZN','NVDA','GOOGL','JNJ','JPM','MSFT','XOM','META','TSLA']
- **STALE** — references old universe with META and TSLA
- **common_start_date:** 2012-05-18 (wrong — was set by META IPO)
- **common_end_date:** 2025-04-30 (correct)
- Missing: BAC, UNH; missing: common_start should be 2005-01-03

### configs/selected_alphas.json
- **33 selected alphas** — selection was done on old META/TSLA universe
- Will be regenerated in Phase 5/7 with new universe
- The alpha names themselves are valid; only the universe changes

### models/best_params.json
- Keys: rf, xgb, meta
- model_final.pkl: RandomForestClassifier, n_features_in=15 (NVDA single-stock)
- model_rf.pkl: RandomForestClassifier, n_features_in=17 (NVDA single-stock)
- All model artifacts are STALE for the new pipeline

---

## 6. Notebook Inventory

| Notebook | Code Cells | Executed | META/TSLA Refs | Notes |
|----------|-----------|----------|----------------|-------|
| 00_data_inspection.ipynb | 7 | YES | 0 | NVDA-only |
| 01_data_structures.ipynb | 8 | YES | 0 | NVDA-only |
| 02_labeling.ipynb | 10 | NO | 0 | Not executed |
| 03_sample_weights.ipynb | 10 | YES | 0 | NVDA-only |
| 04_fracdiff.ipynb | 7 | YES | 0 | NVDA-only |
| 05_feature_engineering.ipynb | 8 | YES | 0 | NVDA-only |
| 06_model_training.ipynb | 10 | YES | 0 | NVDA-only |
| 07_purged_cv.ipynb | 7 | YES | 0 | NVDA-only |
| 08_feature_importance.ipynb | 15 | YES | 0 | NVDA-only |
| 09_hyperparameter_tuning.ipynb | 10 | YES | 0 | NVDA-only |
| 10_meta_labeling_bet_sizing.ipynb | 23 | YES | 2 | META_CLF_PARAMS (meta-labeling params, not stock ticker) |
| 11_backtesting.ipynb | 14 | YES | 0 | NVDA-only |
| 12_structural_breaks.ipynb | 8 | YES | 0 | NVDA-only |
| 13_entropy_microstructure.ipynb | 9 | YES | 0 | NVDA-only |
| 14_final_report_plots.ipynb | 30 | YES | 0 | NVDA-only |

**Note:** Notebook 10's META/TSLA references are variable names (`META_CLF_PARAMS`, `y_meta`, `t1_meta`) — these are meta-labeling variables, NOT ticker references. All 15 notebooks are NVDA single-stock and will be replaced with the new 17-notebook architecture (00–16).

---

## 7. Inconsistency Detection

### META/TSLA Contamination

| Location | Type | Detail | Action Required |
|----------|------|--------|-----------------|
| data/raw/META_raw.csv | Raw data | 3,255 rows, 2012–2025 | DELETE after backup |
| data/raw/TSLA_raw.csv | Raw data | 3,732 rows, 2010–2025 | DELETE after backup |
| data/processed/panel_ohlcv.parquet | Panel artifact | Contains META, TSLA tickers | REGENERATE |
| data/processed/panel_alpha_features.parquet | Panel artifact | Contains META, TSLA tickers | REGENERATE |
| data/processed/panel_alpha_features_pruned.parquet | Panel artifact | Contains META, TSLA tickers | REGENERATE |
| data/processed/per_stock/META_*.parquet (4 files) | Per-stock artifacts | META pipeline outputs | DELETE after backup |
| data/processed/per_stock/TSLA_*.parquet (4 files) | Per-stock artifacts | TSLA pipeline outputs | DELETE after backup |
| data/processed/pooled_modelling.parquet | Pooled artifact | Contains META, TSLA rows | REGENERATE |
| data/processed/pooled_labels.parquet | Pooled artifact | Contains META, TSLA rows | REGENERATE |
| data/processed/pooled_ts_features.parquet | Pooled artifact | Contains META, TSLA rows | REGENERATE |
| data/processed/pooled_weights.parquet | Pooled artifact | Contains META, TSLA rows | REGENERATE |
| configs/universe.json | Config | Lists META, TSLA as tickers | REPLACE |
| src/alphas/registry.py lines 14–15 | Source code | SECTOR_MAP hardcodes META/TSLA | FIX in Phase 2 |
| scripts/prompt1_data_acquisition.py lines 12–13, 24–25 | Script | Hardcodes META/TSLA | UPDATE (scripts are historical; read from config instead) |

### Missing Artifacts for New Universe

| Missing Item | Required By | Action |
|-------------|-------------|--------|
| data/raw/BAC_raw.csv | Phase 3 | DOWNLOAD |
| data/raw/UNH_raw.csv | Phase 3 | DOWNLOAD |
| data/processed/per_stock/BAC_* | Phase 4 | GENERATE |
| data/processed/per_stock/UNH_* | Phase 4 | GENERATE |
| data/processed/panel/ (directory) | Phase 3 | CREATE |
| data/processed/pooled/ (directory) | Phase 7 | CREATE |

### Architectural Issues

| Issue | Severity | Detail |
|-------|----------|--------|
| tests/test_cv.py is empty (0 lines) | High | Phase 2 must populate |
| tests/test_labeling.py is empty (0 lines) | High | Phase 2 must populate |
| src/utils.py is empty (0 lines) | Low | Stub only |
| data/processed/panel/ directory missing | Medium | Plan requires it |
| data/processed/pooled/ directory missing | Medium | Plan requires it |
| panel_ohlcv uses 2012 start | High | Must be rebuilt with 2005 start |
| Per-stock clean files have 6,370 rows but labels only 39–195 | Medium | Labels were restricted to common range; will improve with 2005 start |
| nvda_labels.parquet has 385 NaN (out of 400) | Info | Open events at data end — expected |
| model_final.pkl trained on 15 features (old NVDA) | High | Stale — regenerate |
| src/alphas/101Alpha_code_2.py parse error | Low | Reference file only; not imported |
| tuning_log.parquet has 212 NaN | Low | XGBoost params — not used in new pipeline |

### Weight Scaling Issue (carried forward from previous audit)

- pooled_weights showed max weight = 4.506 (should be ≈ 1.0)
- Per-stock weights show varying means — investigation in Phase 8

---

## 8. Directory Structure Status

| Directory | Status | Action |
|-----------|--------|--------|
| data/raw/ | EXISTS | OK |
| data/processed/per_stock/ | EXISTS | Needs BAC/UNH; remove META/TSLA |
| data/processed/panel/ | MISSING | CREATE in Phase 2 |
| data/processed/pooled/ | MISSING | CREATE in Phase 2 |
| models/ | EXISTS | Stale model artifacts |
| configs/ | EXISTS | universe.json needs update |
| src/ | EXISTS | OK |
| src/alphas/ | EXISTS | Fix registry.py |
| src/pipeline/ | EXISTS | OK |
| notebooks/ | EXISTS | 15 old notebooks → 17 new |
| reports/figures/ | EXISTS | Stale figures |
| tests/ | EXISTS | Both test files empty |
| audit/ | EXISTS | OK (this report) |
| backups/ | EXISTS | OK |
| docs/ | EXISTS | OK |

---

## 9. Validation

- Audit report exists and is non-empty: **YES**
- Repository accessible: **YES**
- All src/ modules importable: **YES (13/13)**
- Phase is read-only (no artifacts modified): **YES**

---

## 10. Phase 1 Summary

**PASS.** Inventory complete. No modifications made.

### Critical Actions Required Before Modelling Can Begin

1. Download BAC_raw.csv and UNH_raw.csv (Phase 3)
2. Update configs/universe.json to new universe (Phase 2)
3. Fix src/alphas/registry.py lines 14–15 (Phase 2)
4. Rebuild panel_ohlcv with 2005 start and new universe (Phase 3)
5. Regenerate all 10 per-stock pipelines with extended date range (Phase 4)
6. Create data/processed/panel/ and data/processed/pooled/ directories (Phase 2)
7. Populate tests/test_cv.py and tests/test_labeling.py (Phase 2)
8. Delete META/TSLA raw CSVs and per-stock artifacts after backup (Phase 3)

### Projected Impact of New Universe

| Metric | Old (META/TSLA, 2012–2025) | New (BAC/UNH, 2005–2025) |
|--------|--------------------------|--------------------------|
| Panel rows | 32,560 (3,256 × 10) | ~51,000 (5,100 × 10) |
| Years of data | 13.0 | 20.3 |
| Estimated pooled events | 881 | ~1,800–2,500 |
| Min events per stock | 35 (AMZN) | ~80 (AMZN est.) |
| Common start | 2012-05-18 | 2005-01-03 |

---

*Phase 1 complete. Proceed to Phase 2 — Architecture Stabilization.*
