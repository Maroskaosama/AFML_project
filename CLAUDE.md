# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 13-stage ML research pipeline implementing López de Prado's *Advances in Financial Machine Learning* (AFML) methodology on a 10-stock universe (AAPL, AMZN, BAC, GOOGL, JNJ, JPM, MSFT, NVDA, UNH, XOM) over 2005–2025. It computes 101 WorldQuant formulaic alphas, prunes to 33 non-redundant features, and achieves 56.6% OOS accuracy (Sharpe 1.06) on binary direction prediction.

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
python scripts/stage05_cv_validation.py         # 13/13 checks
python scripts/stage12_final_audit.py           # 33/33 checks
```

### Tests
```powershell
pytest tests/
pytest tests/test_cv.py           # Purged K-Fold temporal leakage logic
pytest tests/test_labeling.py     # Triple-barrier + CUSUM
pytest tests/test_operators.py    # Alpha operator correctness
```

### Notebooks
```powershell
python notebooks/run_all_stocks.py   # Execute all per-stock notebooks
jupyter notebook                      # Interactive exploration
```

## Architecture & Data Flow

```
Raw OHLCV (yfinance)
    → stage01: Download + validate 10 stocks
    → stage02: Per-stock pipeline (CUSUM filtering → triple-barrier labeling → features)
    → stage03: Alpha engine (101 WorldQuant alphas → prune to 33 non-redundant)
    → stage04/05: Leakage + CV validation checkpoints
    → stage06: Sample weight analysis
    → stage07: Diagnostic visualizations
    → stage08: Feature importance (MDI/MDA/SFI) + hyperparameter tuning + OOS predictions
    → stage09: Meta-labeling (secondary classifier for bet sizing)
    → stage10: Backtesting (SR, PSR, DSR, CPCV)
    → stage11: CPCV robustness (15 combinatorial purged CV paths)
    → stage12: Final audit (33 integrity checks)
    → stage13: Final report + summary tables
```

**Data storage**: All intermediate artifacts are Parquet files in `data/processed/`.
- `data/processed/per_stock/` — per-ticker CUSUM events, labels, features
- `data/processed/pooled/` — cross-sectional merged data
- `data/processed/panel_*.parquet` — final modelling dataset (MultiIndex `(Date, ticker)`)

**Models** saved to `models/` as pickle + JSON hyperparameters. Reports/figures to `reports/`.

## Key Source Modules (`src/`)

| Module | Purpose |
|---|---|
| `alphas/` | 101 WorldQuant alpha implementations (vectorized, wide OHLCV matrix) |
| `labeling.py` | CUSUM event detection + triple-barrier labeling |
| `fracdiff.py` | Fractional differentiation (stationarity with memory preservation) |
| `cross_validation.py` | `MultiAssetPurgedKFold` — temporal leak-free CV across multiple tickers |
| `sample_weights.py` | Uniqueness, sequential bootstrap, decay, return attribution |
| `features.py` | 50+ features: momentum, volatility, microstructure, entropy |
| `feature_importance.py` | MDI, MDA, SFI importance methods |
| `backtesting.py` | Sharpe, PSR, DSR, CPCV metrics |
| `meta_labeling.py` | Secondary model for bet-sizing confidence |
| `pipeline/` | Per-stock and pooling orchestration |

## Configuration

`configs/universe.json` — single source of truth for all parameters:
- Stock list, CUSUM thresholds (adaptive volatility targeting 200–600 events/stock)
- Labeling: `pt_sl=[1,1]` symmetric barriers, 10-day vertical barrier
- Fracdiff: d* minimizes ADF p-value while keeping correlation ≥ 0.85 with original
- CV: 5 folds, 1% embargo period
- Alpha pruning: NaN < 40%, non-constant, |correlation| < 0.85 → 33 survive

`configs/selected_alphas.json` — 33 selected alpha IDs + 4 pruned (redundant).

## Naming Conventions

- Time-series features: `ret_{n}d`, `vol_{n}d`, `rsi_14`
- Microstructure: `amihud_illiquidity`, `roll_spread`, `corwin_schultz_spread`
- Alphas: `alpha001`–`alpha101`
- Label columns: `label` (binary), `bin` (ternary), `t1` (exit time), `weight`, `ticker`
- MultiIndex on panel data: `(Date, ticker)`

## Diagnostic Conventions in Scripts

```python
sep("Section Title")              # prints header separator
check("Description", condition)   # logs PASS/FAIL and accumulates to ERRORS list
# Scripts exit non-zero if any check fails
```

## Extending the Pipeline

- **Add a stock**: update `configs/universe.json` → re-run stages 01–02 → re-pool
- **Add a feature**: add `compute_*_features()` in `src/features.py`, call from stage 02
- **Add an alpha**: implement in `src/alphas/`, register in stage 03 alpha list; pruning happens automatically
- **Change CV folds**: edit `n_splits` in `configs/universe.json` and `src/cross_validation.py`
