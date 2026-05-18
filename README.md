# Leakage-Resistant Multi-Asset AFML Research Framework

A 13-stage pipeline implementing the full AFML (Advances in Financial Machine Learning)
methodology on a 10-stock universe (AAPL AMZN BAC GOOGL JNJ JPM MSFT NVDA UNH XOM)
with 101 WorldQuant formulaic alpha factors.

---

## Quick Start

```bash
# Run the full pipeline end-to-end
python scripts/stage01_data_acquisition.py
python scripts/stage02_per_stock_pipeline.py
python scripts/stage03_alpha_engine.py
python scripts/stage04_leakage_validation.py   # 34 checks — must be 34/34 PASS
python scripts/stage05_cv_validation.py        # 13 checks — must be 13/13 PASS
python scripts/stage06_weight_investigation.py
python scripts/stage07_visualizations.py
python scripts/stage08_final_modelling.py
python scripts/stage09_meta_labeling.py
python scripts/stage10_backtesting.py
python scripts/stage11_cpcv.py
python scripts/stage12_final_audit.py         # 33 checks — must be 33/33 PASS
python scripts/stage13_final_report.py

# Run all notebooks for all stocks
python notebooks/run_all_stocks.py
```

---

## Project Structure

```
AFML_Project/
├── configs/
│   ├── universe.json             10 stocks, date range 2005-2025
│   └── selected_alphas.json      33 selected alpha factors
│
├── data/
│   ├── raw/                      {TICKER}_raw.csv (one per stock)
│   └── processed/
│       ├── per_stock/            {TICKER}_clean/labels/ts_features/weights.parquet
│       ├── pooled/               pooled_labels/ts_features/weights
│       ├── pooled_modelling.parquet     2071 events × 50 features
│       ├── *_pooled.parquet             pipeline stage outputs
│       └── legacy/               old single-stock NVDA artifacts (archived)
│
├── models/
│   ├── best_params_pooled.json   RF & XGB best hyperparameters + OOS metrics
│   ├── model_rf.pkl
│   └── model_xgb.pkl
│
├── notebooks/
│   ├── per_stock/                Parametrized by TICKER — run for any of 10 stocks
│   │   ├── NB01_data_inspection.ipynb
│   │   ├── NB02_data_structures.ipynb    CUSUM events + dollar bars
│   │   ├── NB03_labeling.ipynb           Triple-barrier labels
│   │   ├── NB04_sample_weights.ipynb     Uniqueness × return attribution × decay
│   │   ├── NB05_fracdiff.ipynb           Fractional differentiation sweep
│   │   ├── NB06_feature_engineering.ipynb
│   │   ├── NB07_structural_breaks.ipynb  SADF/GSADF bubble detection
│   │   └── NB08_entropy_microstructure.ipynb
│   ├── pooled/                   Full 10-stock pooled dataset analysis
│   │   ├── NB09_model_training.ipynb
│   │   ├── NB10_purged_cv.ipynb          MultiAssetPurgedKFold
│   │   ├── NB11_feature_importance.ipynb MDI / MDA / SFI
│   │   ├── NB12_hyperparameter_tuning.ipynb
│   │   ├── NB13_meta_labeling_bet_sizing.ipynb
│   │   ├── NB14_backtesting.ipynb        Strategy A/B, SR/DSR/CPCV
│   │   ├── NB15_final_report_plots.ipynb
│   │   ├── NB16_alpha_diagnostics.ipynb  101 WorldQuant alphas
│   │   └── NB17_pipeline_overview.ipynb
│   ├── outputs/                  Executed notebooks (gitignored)
│   │   ├── AAPL/  AMZN/  BAC/  GOOGL/  JNJ/  JPM/  MSFT/  NVDA/  UNH/  XOM/
│   │   └── pooled/
│   └── run_all_stocks.py         Executes all notebooks for all tickers
│
├── reports/
│   ├── AFML_multistock_report.md
│   ├── figures/
│   │   ├── per_stock/{TICKER}/  Figures generated per ticker
│   │   └── pooled/              Pooled pipeline figures
│   └── tables/                  T01-T18 CSV result tables
│
├── scripts/                     Pipeline stages (run in order)
│   ├── stage01_data_acquisition.py
│   ├── stage02_per_stock_pipeline.py
│   ├── stage03_alpha_engine.py
│   ├── stage04_leakage_validation.py
│   ├── stage05_cv_validation.py
│   ├── stage06_weight_investigation.py
│   ├── stage07_visualizations.py
│   ├── stage08_final_modelling.py
│   ├── stage09_meta_labeling.py
│   ├── stage10_backtesting.py
│   ├── stage11_cpcv.py
│   ├── stage12_final_audit.py
│   └── stage13_final_report.py
│
├── src/                         AFML library modules
│   ├── alphas/                  101 formulaic alpha engine
│   ├── pipeline/                Per-stock and pooling helpers
│   ├── labeling.py              Triple-barrier, CUSUM
│   ├── sample_weights.py        Uniqueness, sequential bootstrap
│   ├── fracdiff.py              FFD stationarity
│   ├── cross_validation.py      MultiAssetPurgedKFold
│   ├── feature_importance.py    MDI / MDA / SFI
│   ├── hyperparameter_tuning.py Purged random search + DSR
│   ├── meta_labeling.py         Secondary classifier
│   ├── bet_sizing.py            Snippet 10.1
│   ├── backtesting.py           SR / PSR / DSR / CPCV
│   ├── structural_breaks.py     SADF / GSADF
│   ├── entropy.py               Shannon entropy / Lempel-Ziv
│   └── microstructure.py        Corwin-Schultz, Amihud, Roll
│
└── tests/                       Unit tests (pytest)
```

---

## Key Results

| Metric | Value |
|---|---|
| Universe | 10 stocks, 2005-2025 |
| Alpha factors | 33 of 101 (after NaN + redundancy pruning) |
| Events (pooled) | 2,071 × 50 features |
| RF OOS accuracy | 56.0% (vs 50.0% random baseline) |
| XGB OOS accuracy | 56.6% |
| RF DSR | 0.9589 (60 HP trials) |
| Portfolio Sharpe | 1.06 (Strategy A, ±1 position) |
| Portfolio DSR | 0.97 |
| CPCV paths (15) | 100% SR > 0, mean SR = 1.04 |
| Final audit | 33/33 PASS |

---

## Pipeline Validation

Run in order to confirm full integrity:
```bash
python scripts/stage04_leakage_validation.py   # 34/34 PASS
python scripts/stage05_cv_validation.py        # 13/13 PASS
python scripts/stage12_final_audit.py          # 33/33 PASS
```

---

## Notebooks

Change `TICKER = 'NVDA'` in the first cell of any per-stock notebook to run it for
any of the 10 stocks. Or run `python notebooks/run_all_stocks.py` to execute all
notebooks for all tickers automatically.
