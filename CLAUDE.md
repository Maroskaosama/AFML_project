# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 13-stage ML research pipeline implementing López de Prado's *Advances in Financial Machine Learning* (AFML) methodology on a **50-stock universe** across 10 GICS sectors (2005–2025). It computes 101 WorldQuant formulaic alphas, prunes to 33 non-redundant features, and achieves **52.8% OOS ensemble accuracy** (Portfolio B Sharpe 0.43, DSR 0.19, CPCV 100% paths SR > 0) on binary direction prediction.

Three enhancement waves have been applied on top of the original AFML skeleton:
- **Wave 1**: Minimum return filter (`|ret| ≥ 0.5%`) + 4 macro regime features (VIX level, VIX 5d change, SPY 20d return, yield curve slope), all shifted +1 day to prevent look-ahead.
- **Wave 2**: Calibrated RF+XGB ensemble (isotonic calibration), PCA on alpha block per fold (prevents leakage), per-fold threshold calibration, buy-and-hold EW baseline in backtesting.
- **Wave 3**: Universe expanded from 10 to 50 tickers across all 10 GICS sectors; all hardcoded "10-stock" checks made dynamic.

## Commands

### Run the full pipeline (sequential)
```powershell
# Run all 13 stages in order
foreach ($i in 1..13) {
    $stage = $i.ToString("D2")
    $script = Get-ChildItem "scripts/stage${stage}_*.py" | Select-Object -First 1
    python $script.FullName
}
```

### Run a specific stage
```powershell
python scripts/stage01_data_acquisition.py
python scripts/stage08_final_modelling.py
# etc.
```

### Validation checkpoints (all must PASS)
```powershell
python scripts/stage04_leakage_validation.py    # 34/34 checks
python scripts/stage05_cv_validation.py         # 15/15 checks
python scripts/stage12_final_audit.py           # 33/33 checks
```

### Tests
```powershell
pytest tests/
pytest tests/test_cv.py           # Purged K-Fold temporal leakage logic
pytest tests/test_labeling.py     # Triple-barrier + CUSUM
pytest tests/test_operators.py    # Alpha operator correctness
```

## Architecture & Data Flow

```
Raw OHLCV (yfinance, 50 stocks)
    → stage01: Download missing tickers + rebuild panel_ohlcv (255,658 rows)
    → stage02: Per-stock pipeline — CUSUM → triple-barrier labels (min_ret≥0.5%)
                → fracdiff → 17 TS features; 11,890 pooled events
    → stage03: Alpha engine — 101 WorldQuant alphas → 33 non-redundant selected
    → stage04/05: Leakage (34 checks) + CV validation (15 checks) checkpoints
                  stage05 also builds pooled_modelling.parquet with 4 macro features
    → stage06: Sample weight clipping (p99)
    → stage07: 12 diagnostic visualizations
    → stage08: MDI/MDA/SFI importance + RF+XGB HP tuning (30 trials each)
                → calibrated ensemble OOS loop (PCA alpha block, fold threshold)
    → stage09: Meta-labeling — secondary RF for bet sizing
    → stage10: Backtesting — SR, PSR, DSR, CPCV + buy-and-hold EW baseline
    → stage11: CPCV robustness (K=6, p=2, 15 paths)
    → stage12: Final audit (33 integrity checks, 33/33 PASS)
    → stage13: Final report + 6 CSV summary tables
```

**Feature set per event (54 total):**
- 17 per-stock TS features: momentum, volatility, microstructure (Amihud, Roll, Corwin-Schultz), entropy, fracdiff
- 4 macro regime features: `vix_level`, `vix_5d_chg`, `spy_20d_ret`, `yield_curve_slope`
- 33 WorldQuant alpha features (cross-sectional rank-based, meaningful with 50 stocks)

**Data storage**: All intermediate artifacts are Parquet files in `data/processed/`.
- `data/processed/per_stock/` — per-ticker CUSUM events, labels, features (50 × 5 files)
- `data/processed/panel_*.parquet` — OHLCV panel + alpha features (MultiIndex `(Date, ticker)`)
- `data/processed/pooled_modelling.parquet` — 11,890 events × 58 columns (features + meta)

**Models** saved to `models/` as pickle + JSON. Reports/figures to `reports/`.

## Key Source Modules (`src/`)

| Module | Purpose |
|---|---|
| `alphas/` | 101 WorldQuant alpha implementations (vectorized, wide OHLCV matrix) |
| `labeling.py` | CUSUM event detection + triple-barrier labeling |
| `fracdiff.py` | Fractional differentiation (stationarity with memory preservation) |
| `cross_validation.py` | `MultiAssetPurgedKFold` — temporal leak-free CV across multiple tickers |
| `sample_weights.py` | Uniqueness, sequential bootstrap, decay, return attribution |
| `features.py` | 54 features: 17 TS + 4 macro (`compute_macro_features`) + microstructure + entropy |
| `feature_importance.py` | MDI, MDA, SFI importance methods |
| `backtesting.py` | Sharpe, PSR, DSR, CPCV metrics |
| `meta_labeling.py` | Secondary model for bet-sizing confidence |
| `pipeline/per_stock.py` | Per-ticker AFML pipeline; `min_ret` filter drops noise labels |
| `pipeline/pooling.py` | Stacks per-stock artifacts; alpha NaN → 0 (neutral) to avoid dropping tickers |

## Configuration

`configs/universe.json` — single source of truth for all parameters:
- **50 tickers** across 10 GICS sectors (IT×9, FN×8, HC×7, CST×6, CD×5, CM×4, EN×3, IN×4, MT×2, UT×2)
- CUSUM: adaptive volatility, targeting 200–600 events/stock
- Labeling: `pt_sl=[1,1]`, 10-day vertical barrier, `min_ret=0.005` (drops trades < 0.5% move)
- Fracdiff: d* minimises ADF p-value while keeping correlation ≥ 0.85 with original
- CV: 5 folds, 1% embargo period
- Alpha pruning: NaN < 40%, non-constant, |correlation| < 0.85 → 33 survive

`configs/selected_alphas.json` — 33 selected alpha IDs + 4 pruned (redundant).

## Stage 08 Modelling Architecture

The OOS loop (Step 12 in `stage08_final_modelling.py`) uses:
1. **PCA on alpha block**: `PCA(n_components=0.95)` fitted on training-fold alpha columns only — prevents alpha leakage and compresses 33 alphas to ~5–7 components.
2. **Isotonic calibration**: both RF and XGB wrapped in `CalibratedClassifierCV(method='isotonic', cv=3)`.
3. **Per-fold threshold**: grid search [0.35, 0.65] on training-fold ensemble probabilities maximising balanced accuracy.
4. **Ensemble**: `ens_prob = 0.5 * rf_prob + 0.5 * xgb_prob`, then threshold applied.
5. `oos_predictions_pooled.parquet` stores: `oos_pred`, `oos_prob`, `oos_prob_rf`, `oos_prob_xgb`, `oos_pred_rf`, `oos_pred_xgb`, `oos_threshold`, `oos_fold`.

## Known Quirks

- **NVDA alpha015 NaN**: `alpha015` is all-NaN for NVDA after 2015 (specific to its price history). `pooling.py` fills alpha NaN with 0 (neutral cross-sectional rank) rather than dropping entire stocks.
- **Unicode on Windows**: All `→` arrows in `print()` statements must be plain ASCII `->` — Windows cp1252 console cannot encode U+2192.
- **Stage03 cache**: `data/processed/panel_alpha_features.parquet` is cached. Delete it if the universe changes before re-running stage03.

## Naming Conventions

- Time-series features: `ret_{n}d`, `vol_{n}d`, `rsi_14`
- Macro features: `vix_level`, `vix_5d_chg`, `spy_20d_ret`, `yield_curve_slope`
- Microstructure: `amihud_illiquidity`, `roll_spread`, `corwin_schultz_spread`
- Alphas: `alpha001`–`alpha101`
- Label columns: `label` (binary ±1), `bin` (ternary), `t1` (exit time), `weight`, `ticker`
- MultiIndex on panel data: `(Date, ticker)`

## Diagnostic Conventions in Scripts

```python
sep("Section Title")              # prints header separator
check("Description", condition)   # logs PASS/FAIL and accumulates to ERRORS list
# Scripts exit non-zero if any check fails
```

## Extending the Pipeline

- **Add a stock**: update `configs/universe.json` (ticker + sector_code) → delete `data/processed/panel_alpha_features.parquet` → re-run stages 01–05
- **Add a feature**: add `compute_*_features()` in `src/features.py`, call from `stage02`; update feature count checks in `stage05` (E1/E3) and `stage12` (F1)
- **Add an alpha**: implement in `src/alphas/`, register in stage03 alpha list; pruning happens automatically
- **Change CV folds**: edit `n_splits` in `configs/universe.json` and `src/cross_validation.py`
