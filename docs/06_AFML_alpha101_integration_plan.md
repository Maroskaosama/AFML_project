# 101 Formulaic Alphas × AFML Pipeline — Integration Specification

**Document Type:** Quantitative Research Implementation Blueprint
**Date:** May 15, 2026
**Scope:** Integration of WorldQuant 101 Formulaic Alphas into a validated AFML NVDA pipeline
**Authoritative References:**
- Kakushadze, Z. "101 Formulaic Alphas" (2015)
- López de Prado, M. "Advances in Financial Machine Learning" (2018)
- AFML-Validation-Pipeline repository (Complete-AFML-Pipeline branch)
- yli188/WorldQuant_alpha101_code GitHub implementation

---

## 1. Strategic Analysis — Why This Integration Matters

### 1.1 What the 101 Formulaic Alphas Are

The 101 Formulaic Alphas are explicit, production-origin quantitative trading signals published by Kakushadze with WorldQuant LLC's permission. They are mathematical formulas — simultaneously human-readable and machine-executable — that transform daily OHLCV + VWAP data into predicted return signals. Their average holding period ranges from 0.6 to 6.4 days. Their average pair-wise correlation is only 15.9%, meaning they capture substantially different aspects of price and volume dynamics.

These alphas are NOT labels. They are NOT strategies. They are feature generators — each one computes a time-series signal that may carry weak predictive information about future returns. The paper's Table 1 shows annualized Sharpe ratios between 1.24 and 4.16, but these are for cross-sectional dollar-neutral portfolios of ~2,000 US stocks — not for a single stock.

### 1.2 Relationship to AFML

The existing AFML pipeline uses 15 hand-engineered features (momentum, volatility, volume, microstructure, entropy, fracdiff). These features were selected based on financial intuition and AFML methodology. The 101 alphas represent a radically different approach: systematic, formulaic signal generation that explores a broader feature space.

The integration transforms the project from "AFML infrastructure validation" into "quantitative alpha research using AFML validation." The alphas become candidate features evaluated through the same purged CV, sample-weighted, leakage-free framework that validates the existing pipeline. This is the correct AFML paradigm: generate many candidate features, then use MDI/MDA/SFI to identify which carry genuine predictive information.

### 1.3 Why AFML Validation Is Critical Here

Without AFML validation, alpha integration degenerates into a Kaggle-style feature factory where 50+ correlated features are thrown at a model and overfit is inevitable. The 101 alphas include rolling correlations, rolling covariances, lagged ranks, and cumulative sums — every one of which creates temporal dependencies that standard train-test splitting ignores. PurgedKFold with embargo is the only safe evaluation methodology for these features.

Furthermore, adding ~50+ alpha features to a 195-sample dataset would be catastrophic without proper feature selection (MDI/MDA/SFI). The samples-per-feature ratio would collapse below 4, virtually guaranteeing overfitting. The integration plan must include aggressive, principled feature pruning.

---

## 2. Current AFML Repository Architecture

### 2.1 Repository Structure (from tree.txt and repo inspection)

```
AFML-Validation-Pipeline/
├── data/
│   ├── raw/NVDA_raw.csv                    # 5114 rows, 2005-01-03 → 2025-04-30
│   └── processed/
│       ├── nvda_clean.parquet              # Cleaned daily OHLCV
│       ├── nvda_dollar_bars.parquet        # Dollar bars (generated, possibly unused)
│       ├── nvda_cusum_events.parquet       # CUSUM event timestamps
│       ├── nvda_labels.parquet             # Triple-barrier labels + t1
│       ├── nvda_sample_weights.parquet     # Uniqueness × return-attr × time-decay
│       ├── nvda_fracdiff.parquet           # FFD(log close, d=0.25)
│       ├── nvda_features.parquet           # Full feature matrix (17 cols)
│       ├── nvda_modelling_dataset.parquet  # 195 × 20 final dataset
│       ├── cv_results.parquet              # Stage 4 CV scores
│       ├── tuning_log.parquet             # Stage 5 tuning trials
│       ├── feature_importance.parquet      # Stage 6 MDI/MDA/SFI
│       ├── nvda_positions.parquet          # Stage 7 daily positions
│       └── backtest_results.parquet        # Stage 8 backtest returns
├── models/
│   ├── model_rf.pkl, model_xgb.pkl, model_final.pkl
│   └── best_params.json
├── src/
│   ├── data_structures.py    # CUSUM filter, dollar bars
│   ├── labeling.py           # Triple barrier, daily vol, getBins
│   ├── sample_weights.py     # Concurrency, uniqueness, seq bootstrap
│   ├── fracdiff.py           # FFD fractional differentiation
│   ├── features.py           # 17 features: momentum/vol/volume/micro/entropy
│   ├── cross_validation.py   # PurgedKFold, cv_score (weighted)
│   ├── modelling.py          # train_and_evaluate
│   ├── feature_importance.py # MDI, MDA, SFI
│   ├── hyperparameter_tuning.py
│   ├── bet_sizing.py         # Snippets 10.1-10.3
│   ├── backtesting.py        # Backtest engine + SR/PSR/DSR
│   ├── entropy.py, microstructure.py, structural_breaks.py
│   ├── synthetic.py, multiprocess.py, utils.py
│   └── __init__.py
├── notebooks/
│   ├── 00-09: Stages 0-5
│   ├── 10_meta_labeling_bet_sizing.ipynb   # Stage 7
│   ├── 11_backtesting.ipynb                # Stage 8
│   └── 12-14: Structural breaks, entropy, final report
└── tests/test_cv.py, test_labeling.py
```

### 2.2 Feature Pipeline Architecture

The current feature pipeline in `src/features.py` follows this flow:

```
nvda_clean.parquet
    ├── compute_momentum_features()    → 6 features
    ├── compute_volatility_features()  → 2 features
    ├── compute_volume_features()      → 2 features
    ├── compute_microstructure_features() → 4 features
    ├── compute_entropy_features()     → 2 features
    └── fracdiff(log_close, d=0.25)    → 1 feature
                                         ─────────
                                         17 features → align to events → drop NaN → 195 rows
```

### 2.3 Where Alpha Integration Should Occur

The alpha engine must slot into the feature pipeline between raw data loading and event alignment. The correct insertion point is:

```
nvda_clean.parquet
    ├── [EXISTING] compute_momentum_features()
    ├── [EXISTING] compute_volatility_features()
    ├── [EXISTING] ...
    ├── [NEW] alpha_engine.compute_single_asset_alphas()  ← NEW MODULE
    └── build_feature_matrix() → align to events → drop NaN → modelling dataset
```

The alpha engine MUST be a separate module (`src/alpha_engine.py` or `src/alphas/`), NOT embedded in notebooks.

---

## 3. WorldQuant GitHub Implementation Audit

### 3.1 Architecture Assessment

The yli188 implementation consists of two files (`101Alpha_code_1.py` and `101Alpha_code_2.py`) containing:

1. **Helper functions** at module level: `rank()`, `delta()`, `delay()`, `correlation()`, `covariance()`, `scale()`, `ts_rank()`, `ts_argmax()`, `ts_argmin()`, `ts_sum()`, `sma()`, `stddev()`, `product()`, `adv()`, `decay_linear()`, `log()`, `sign()`, `SignedPower()`.

2. **Alphas class** accepting a DataFrame with columns `S_DQ_OPEN`, `S_DQ_HIGH`, `S_DQ_LOW`, `S_DQ_CLOSE`, `S_DQ_VOLUME`, `S_DQ_PCTCHANGE`, `S_DQ_AMOUNT`.

3. **101 alpha methods** (`alpha001()` through `alpha101()`) implementing each formula.

### 3.2 Critical Problems Identified

**Problem 1 — Cross-Sectional Assumption Throughout**

The `rank()` function is defined as: `rank(x) = x.rank(axis=1, pct=True)`. This performs a **cross-sectional rank** (across columns = across assets). With a single stock (NVDA), every `rank()` call returns a constant (0.5 or 1.0 depending on implementation), collapsing the alpha to a trivial signal.

34 of the 101 alphas use `rank()` in a way that requires multiple assets. An additional ~15 use `indneutralize()` (cross-sectional demeaning within industry groups), which requires multiple stocks in the same industry.

**Problem 2 — `rank()` Operating on Panel vs Series**

The implementation assumes DataFrames where rows = dates, columns = assets. With one asset, the DataFrame has one column, and cross-sectional rank is undefined.

**Problem 3 — VWAP Not Available in Standard OHLCV**

Many alphas use `vwap`. The implementation computes it as `(dollar_amount) / (volume)`, which requires intraday dollar-amount data. For daily OHLCV, VWAP must be approximated.

**Problem 4 — `adv{d}` Computation**

`adv20`, `adv40`, `adv60`, etc. are defined as average daily dollar volume over d days. The implementation uses `sma(volume, d)` which is average share volume, not dollar volume. This is a definitional inconsistency with the paper.

**Problem 5 — `indneutralize()` Not Implementable**

15 alphas (48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100) use `indneutralize()` which requires industry classification data and multiple stocks per industry. These are completely non-functional for single-stock NVDA.

**Problem 6 — `cap` (Market Cap) Not in OHLCV**

Alpha#56 uses `cap` (market capitalization), which is not available in standard OHLCV data.

**Problem 7 — Potential Look-Ahead in Helper Functions**

Some rolling operations may use `min_periods=1` which produces estimates from fewer observations than specified, potentially creating unstable signals early in the time series.

### 3.3 Verdict on GitHub Implementation

The yli188 code is a reasonable cross-sectional implementation but is **NOT safe for direct use** in a single-stock AFML pipeline. It must be substantially rewritten with:
- All `rank()` calls converted to time-series rank or removed
- All `indneutralize()` calls removed or replaced
- VWAP approximated from daily data
- `adv{d}` computed as dollar volume, not share volume
- Each alpha individually verified for single-asset validity

---

## 4. The Cross-Sectional vs Single-Asset Problem

### 4.1 Why This Is the Central Technical Challenge

The 101 Formulaic Alphas were designed for cross-sectional trading: at each point in time, rank all ~2,000 stocks by some measure, go long the top-ranked and short the bottom-ranked. The `rank()` operator in the paper is explicitly defined as "cross-sectional rank" — a ranking across assets at a fixed point in time.

With a single stock (NVDA), cross-sectional operations collapse:
- `rank(x)` for a single value is always 1.0 (or 0.5 if using pct=True)
- `indneutralize(x, group)` subtracts the group mean, which IS x for a single stock, yielding zero
- `scale(x)` divides by `sum(abs(x))`, which for a single stock returns `sign(x)`

This means approximately 40-50 of the 101 alphas are mathematically degenerate for single-asset use.

### 4.2 Two Adaptation Approaches

**Approach A — Single-Asset Time-Series Adaptation**

Replace every `rank()` with `ts_rank()` (time-series rank over a rolling window), converting cross-sectional alphas into time-series alphas. Replace `indneutralize()` with time-series demeaning.

Advantages: No additional data needed. All computation stays on NVDA.
Disadvantages: Fundamentally changes what the alphas measure. A cross-sectional alpha asks "is NVDA cheap relative to AAPL today?" A time-series alpha asks "is NVDA cheap relative to its own history?" These are different signals. Research validity is reduced because the alphas are no longer faithful to the original paper.

**Approach B — Multi-Asset Correct Implementation**

Download daily OHLCV for a universe of ~50-100 stocks (e.g., NASDAQ-100 constituents). Compute all 101 alphas cross-sectionally across the full universe. Extract only the NVDA column as the feature for the AFML pipeline.

Advantages: Faithful to the paper. Cross-sectional rank and industry neutralization work correctly. Research conclusions are valid.
Disadvantages: Requires additional data. Introduces survivorship bias if the universe is not handled carefully. Increases computational cost by ~100x.

### 4.3 Recommendation

**Use a hybrid approach:**

1. **Tier 1 (50+ alphas):** Alphas that do NOT use `rank()` or `indneutralize()` in a cross-sectional way. These are pure time-series alphas that work directly on single-stock data. Examples: Alpha#9, #10, #12, #21, #23, #24, #33, #41, #42, #46, #49, #51, #53, #54, #101.

2. **Tier 2 (20-30 alphas):** Alphas where `rank()` can be meaningfully replaced by `ts_rank()` without destroying the signal's economic intuition. Examples: Alpha#1, #4, #8, #17, #34, #35.

3. **Tier 3 — Excluded (~20 alphas):** Alphas requiring `indneutralize()` or `cap`. These MUST be excluded unless multi-asset data is obtained. Examples: Alpha#48, #56, #58, #59, #63, #67.

4. **Tier 4 — Optional Multi-Asset (~20 alphas):** If the user later expands to multi-asset data, these become available. Implement the multi-asset infrastructure now (as a future-proofing measure) but only populate it if data is obtained.

---

## 5. Data Requirements and Universe Design

### 5.1 Required Data Fields for NVDA

| Field | Source | Derivation |
|-------|--------|-----------|
| open | NVDA_raw.csv | Direct |
| high | NVDA_raw.csv | Direct |
| low | NVDA_raw.csv | Direct |
| close | NVDA_raw.csv | Adj Close |
| volume | NVDA_raw.csv | Direct |
| returns | Derived | `close.pct_change()` |
| vwap | **Approximated** | `(high + low + close) / 3` (typical price proxy) |
| adv{d} | Derived | `(close * volume).rolling(d).mean()` |

### 5.2 VWAP Approximation

True VWAP requires intraday trade data. For daily OHLCV, three approximations exist:

1. **Typical price:** `(H + L + C) / 3` — most common daily proxy
2. **Weighted close:** `(H + L + 2*C) / 4` — emphasizes close
3. **Midpoint:** `(H + L) / 2` — simplest

Recommendation: Use **typical price** `(H + L + C) / 3`. This is standard in the academic literature for daily VWAP approximation and is used in the Amihud illiquidity ratio literature.

### 5.3 adv{d} Computation

Per the paper (Appendix A.3): "adv{d} = average daily dollar volume for the past d days."

```python
def adv(close, volume, d):
    """Average daily dollar volume over past d days."""
    dollar_volume = close * volume
    return dollar_volume.rolling(d).mean()
```

The GitHub implementation uses `sma(volume, d)` (share volume only), which is incorrect. This must be fixed.

### 5.4 Universe for Multi-Asset Expansion (Optional)

If multi-asset data is acquired:
- **Universe:** NASDAQ-100 constituents as of a fixed date (e.g., 2020-01-01)
- **Survivorship bias:** Use point-in-time constituents if available; otherwise, acknowledge the bias
- **Liquidity filter:** Exclude stocks with ADV < $1M
- **Data source:** Yahoo Finance (via `yfinance`) provides free daily OHLCV for US equities
- **Industry classification:** GICS sectors from Wikipedia/public sources

---

## 6. Alpha Engine Architecture

### 6.1 Module Organization

```
src/
├── alphas/
│   ├── __init__.py
│   ├── operators.py         # Reusable operators: ts_rank, decay_linear, etc.
│   ├── single_asset.py      # Tier 1+2: alphas that work on single-stock data
│   ├── cross_sectional.py   # Tier 4: alphas requiring multi-asset data (future)
│   ├── registry.py          # Alpha registry and metadata
│   └── diagnostics.py       # Per-alpha diagnostics: missingness, stationarity, correlation
```

### 6.2 Operator Functions

Every operator must be a pure function with explicit window sizes, no hidden state, and strict NaN handling.

```python
# src/alphas/operators.py

import numpy as np
import pandas as pd

def ts_rank(x: pd.Series, d: int) -> pd.Series:
    """Time-series percentile rank over past d days."""
    return x.rolling(d, min_periods=d).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )

def ts_argmax(x: pd.Series, d: int) -> pd.Series:
    """Index (0-based from d days ago) of max value in past d days."""
    return x.rolling(d, min_periods=d).apply(np.argmax, raw=True)

def ts_argmin(x: pd.Series, d: int) -> pd.Series:
    """Index (0-based from d days ago) of min value in past d days."""
    return x.rolling(d, min_periods=d).apply(np.argmin, raw=True)

def decay_linear(x: pd.Series, d: int) -> pd.Series:
    """Linearly decaying weighted moving average: weights d, d-1, ..., 1."""
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    return x.rolling(d, min_periods=d).apply(lambda arr: np.dot(arr, weights), raw=True)

def delta(x: pd.Series, d: int) -> pd.Series:
    """x_t - x_{t-d}."""
    return x.diff(d)

def delay(x: pd.Series, d: int) -> pd.Series:
    """Value of x d days ago."""
    return x.shift(d)

def correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling correlation over past d days."""
    return x.rolling(d, min_periods=d).corr(y)

def covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling covariance over past d days."""
    return x.rolling(d, min_periods=d).cov(y)

def scale(x: pd.Series, a: float = 1.0) -> pd.Series:
    """Rescale x such that sum(abs(x)) = a. Single-asset: returns sign(x) * a."""
    denom = x.abs().sum()
    return x * a / denom if denom != 0 else x * 0

def signed_power(x: pd.Series, a: float) -> pd.Series:
    """x^a preserving sign."""
    return x.abs().pow(a) * np.sign(x)
```

**Critical design decisions:**
- `min_periods=d` (not `min_periods=1`): Prevents look-ahead by requiring a full window before producing a value. The first `d-1` observations are NaN.
- No cross-sectional operations in single-asset operators.
- All functions accept and return pd.Series with proper datetime index.

### 6.3 Alpha Registry

```python
# src/alphas/registry.py

ALPHA_REGISTRY = {
    # Tier 1: Pure time-series, no rank/indneutralize
    'alpha009': {'tier': 1, 'max_lookback': 5,  'uses_vwap': False, 'uses_adv': False},
    'alpha010': {'tier': 1, 'max_lookback': 4,  'uses_vwap': False, 'uses_adv': False},
    'alpha012': {'tier': 1, 'max_lookback': 1,  'uses_vwap': False, 'uses_adv': False},
    'alpha019': {'tier': 1, 'max_lookback': 250, 'uses_vwap': False, 'uses_adv': False},
    'alpha021': {'tier': 1, 'max_lookback': 20, 'uses_vwap': False, 'uses_adv': True},
    'alpha023': {'tier': 1, 'max_lookback': 20, 'uses_vwap': False, 'uses_adv': False},
    'alpha024': {'tier': 1, 'max_lookback': 100, 'uses_vwap': False, 'uses_adv': False},
    'alpha041': {'tier': 1, 'max_lookback': 0,  'uses_vwap': True,  'uses_adv': False},
    'alpha042': {'tier': 1, 'max_lookback': 0,  'uses_vwap': True,  'uses_adv': False},
    'alpha046': {'tier': 1, 'max_lookback': 20, 'uses_vwap': False, 'uses_adv': False},
    'alpha049': {'tier': 1, 'max_lookback': 20, 'uses_vwap': False, 'uses_adv': False},
    'alpha051': {'tier': 1, 'max_lookback': 20, 'uses_vwap': False, 'uses_adv': False},
    'alpha053': {'tier': 1, 'max_lookback': 9,  'uses_vwap': False, 'uses_adv': False},
    'alpha054': {'tier': 1, 'max_lookback': 0,  'uses_vwap': False, 'uses_adv': False},
    'alpha101': {'tier': 1, 'max_lookback': 0,  'uses_vwap': False, 'uses_adv': False},

    # Tier 2: rank() replaced with ts_rank()
    'alpha001': {'tier': 2, 'max_lookback': 20, 'uses_vwap': False, 'uses_adv': False},
    'alpha002': {'tier': 2, 'max_lookback': 6,  'uses_vwap': False, 'uses_adv': False},
    'alpha004': {'tier': 2, 'max_lookback': 9,  'uses_vwap': False, 'uses_adv': False},
    'alpha006': {'tier': 2, 'max_lookback': 10, 'uses_vwap': False, 'uses_adv': False},
    'alpha012': {'tier': 2, 'max_lookback': 1,  'uses_vwap': False, 'uses_adv': False},
    # ... (complete for all implementable alphas)

    # Tier 3: Excluded — requires indneutralize or cap
    'alpha048': {'tier': 3, 'excluded': True, 'reason': 'indneutralize(subindustry)'},
    'alpha056': {'tier': 3, 'excluded': True, 'reason': 'uses cap (market cap)'},
    'alpha058': {'tier': 3, 'excluded': True, 'reason': 'indneutralize(sector)'},
    'alpha059': {'tier': 3, 'excluded': True, 'reason': 'indneutralize(industry)'},
    # ... (complete for all excluded alphas)
}

def get_max_lookback():
    """Maximum lookback across all implementable alphas."""
    return max(
        a['max_lookback'] for a in ALPHA_REGISTRY.values()
        if not a.get('excluded', False)
    )
```

### 6.4 Alpha Computation Example

```python
# src/alphas/single_asset.py

from src.alphas.operators import *

def alpha009(close):
    """
    Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
              ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) :
               (-1 * delta(close, 1))))
    
    Logic: If close has been rising for 5 days, momentum. If falling for 5 days,
    momentum. Otherwise, mean-reversion.
    Tier 1: No rank() or indneutralize(). Pure time-series.
    """
    d1 = delta(close, 1)
    cond1 = ts_min(d1, 5) > 0
    cond2 = ts_max(d1, 5) < 0
    result = pd.Series(index=close.index, dtype=float)
    result[cond1] = d1[cond1]
    result[cond2] = d1[cond2]
    result[~cond1 & ~cond2] = -1 * d1[~cond1 & ~cond2]
    return result

def alpha012(close, volume):
    """
    Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    
    Logic: Mean-reversion weighted by volume change direction.
    If volume increased and price went up, short. If volume decreased
    and price went down, long.
    Tier 1: No rank() or indneutralize(). Pure time-series.
    """
    return np.sign(delta(volume, 1)) * (-1 * delta(close, 1))

def alpha042(close, vwap):
    """
    Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    
    Single-asset adaptation: replace rank() with ts_rank() over 20-day window.
    Original: cross-sectional rank. Adapted: time-series percentile rank.
    Tier 2: rank() → ts_rank(20).
    """
    return ts_rank(vwap - close, 20) / (ts_rank(vwap + close, 20) + 1e-10)

def alpha101(close, open_, high, low):
    """
    Alpha#101: ((close - open) / ((high - low) + .001))
    
    Logic: Intraday return normalized by range. Positive = close near high.
    Tier 1: No rank() or indneutralize(). Pure time-series.
    """
    return (close - open_) / ((high - low) + 0.001)
```

### 6.5 Master Compute Function

```python
def compute_all_alphas(ohlcv_df, tiers=(1, 2)):
    """
    Compute all implementable alphas for the given OHLCV DataFrame.
    
    Parameters
    ----------
    ohlcv_df : pd.DataFrame with columns [Open, High, Low, Close/Adj Close, Volume]
    tiers : tuple of int, which tiers to compute
    
    Returns
    -------
    pd.DataFrame : columns = alpha names, index = dates
    """
    close = ohlcv_df['Adj Close']
    open_ = ohlcv_df['Open']
    high = ohlcv_df['High']
    low = ohlcv_df['Low']
    volume = ohlcv_df['Volume']
    returns = close.pct_change()
    vwap = (high + low + close) / 3  # daily VWAP approximation
    
    alphas = {}
    for alpha_name, meta in ALPHA_REGISTRY.items():
        if meta.get('excluded', False) or meta['tier'] not in tiers:
            continue
        try:
            func = globals()[alpha_name]  # or use a proper dispatch
            # Each alpha function has its own signature
            result = func(close=close, open_=open_, high=high, low=low,
                         volume=volume, returns=returns, vwap=vwap)
            alphas[alpha_name] = result
        except Exception as e:
            print(f"  ⚠ {alpha_name} failed: {e}")
    
    return pd.DataFrame(alphas)
```

---

## 7. Implementation Roadmap — Phase by Phase

### Phase 1 — Repository Preparation (Est. 2 hours)

**Objectives:** Create the alpha engine module structure. No computation yet.

**Deliverables:**
- `src/alphas/__init__.py`
- `src/alphas/operators.py` — all operator functions with unit tests
- `src/alphas/registry.py` — alpha metadata and tier classification
- `tests/test_operators.py` — unit tests for every operator

**Validation:**
```python
# Every operator must pass these:
assert ts_rank(pd.Series([1,2,3,4,5]), 5).iloc[-1] == 1.0  # 5 is max → rank 1.0
assert delta(pd.Series([10,11,13,16]), 1).iloc[-1] == 3    # 16-13
assert delay(pd.Series([10,11,13,16]), 2).iloc[-1] == 11   # value 2 days ago
assert abs(decay_linear(pd.Series([1,1,1,1,1]), 5).iloc[-1] - 1.0) < 1e-10  # constant
```

### Phase 2 — Tier 1 Alpha Implementation (Est. 4 hours)

**Objectives:** Implement all ~15-20 Tier 1 alphas (pure time-series, no rank adaptation needed).

**Deliverables:**
- `src/alphas/single_asset.py` — Tier 1 alpha functions
- Verification against paper formulas (manual spot-check of 5 alphas)
- NaN analysis: how many observations lost per alpha due to lookback

**Engineering Risks:**
- Alpha#19 uses `sum(returns, 250)` — 250-day lookback loses the first year of data
- Alpha#24 uses `sum(close, 100)` and `delay(close, 100)` — 100-day lookback
- Alpha#52 uses `sum(returns, 240)` — 240-day lookback

**Validation:** For each alpha, verify:
1. No NaN in the valid data range (after lookback)
2. Variance > 0 (not constant)
3. Finite values only (no inf)
4. Stationarity (ADF test at 10% significance)

### Phase 3 — Tier 2 Alpha Implementation (Est. 4 hours)

**Objectives:** Implement ~20-30 Tier 2 alphas with `rank()` → `ts_rank()` adaptation.

**Key Decision: ts_rank window size**

The window for `ts_rank()` controls how much history defines the "rank." Options:
- 20 days (1 month) — responsive but noisy
- 60 days (3 months) — balanced
- 252 days (1 year) — stable but slow

Recommendation: Use **60 days** as the default ts_rank window. This balances responsiveness with stability and is consistent with the paper's average holding period of ~2 days (the rank is a normalization, not a signal).

### Phase 4 — Feature Integration (Est. 3 hours)

**Objectives:** Merge alpha features with existing AFML features and align to event timestamps.

**Deliverables:**
- Updated `src/features.py` with `compute_alpha_features()` function
- Updated `nvda_features.parquet` and `nvda_modelling_dataset.parquet`
- Sample-count analysis: how many samples survive after alpha lookback NaN drop

**CRITICAL: Sample Attrition Risk**

The current 195 samples come from ~5100 trading days after dropping NaN from 60-day feature lookbacks. If Tier 1 alphas introduce 250-day lookbacks (Alpha#19, #52), the first ~250 trading days (~1 year) are lost. This reduces the clean data from ~5100 to ~4850, but since the CUSUM events are spread across the full range, the impact on the modelling dataset depends on how many events fall in the first year.

**Mitigation:** Compute the number of surviving samples before and after adding each alpha. If any alpha reduces the sample count below 150, exclude it regardless of tier.

### Phase 5 — Feature Diagnostics (Est. 3 hours)

**Objectives:** Comprehensive analysis of all alpha features.

**Deliverables:**
- `notebooks/15_alpha_diagnostics.ipynb`
- Feature correlation heatmap (existing 15 features + new alphas)
- Redundancy analysis: identify alpha clusters with correlation > 0.8
- Stationarity report: ADF test for each alpha
- Missingness report: NaN counts per alpha
- Variance report: identify near-constant alphas

**Feature Pruning Rules:**
1. Remove any alpha with variance < 1e-8 (effectively constant)
2. Remove any alpha with > 10% NaN in the valid range
3. For each cluster of alphas with mutual correlation > 0.8, keep only the one with lowest ADF p-value (most stationary)
4. Cap total features at 30 (given 195 samples, ~6.5 samples per feature — still low but manageable with tree-based models)

### Phase 6 — Model Retraining with Alpha Features (Est. 2 hours)

**Objectives:** Retrain RF/XGB with augmented feature set using the same AFML methodology.

**Deliverables:**
- Updated `nvda_modelling_dataset.parquet` (195 × N where N ≤ 30+5 meta columns)
- Purged CV accuracy with alphas vs without (paired comparison)
- Feature importance (MDI/MDA/SFI) including alpha features

**Evaluation Protocol:**
1. Run PurgedKFold CV with original 15 features → baseline accuracy
2. Run PurgedKFold CV with 15 original + K selected alphas → augmented accuracy
3. Compare using paired t-test on fold scores (same folds, different feature sets)
4. Report whether any alpha features appear in top-10 MDA importance

### Phase 7 — End-to-End Validation (Est. 2 hours)

**Objectives:** Verify the full pipeline (features → meta-labeling → bet sizing → backtest) with alpha features.

**Deliverables:**
- Updated backtest results with alpha-augmented model
- Sharpe ratio comparison: baseline vs augmented
- DSR comparison
- Synthetic data validation on augmented model

---

## 8. Leakage and Validation Audit

### 8.1 Leakage Risk Matrix for Alpha Features

| Risk | Description | Severity | Prevention |
|------|------------|----------|------------|
| R1 | Rolling operations with `min_periods=1` use partial windows | HIGH | Enforce `min_periods=d` in all operators |
| R2 | `ts_rank()` uses future data if window includes test period | HIGH | `ts_rank` is backward-looking by construction — safe |
| R3 | Alpha computed on full series then aligned to events | MEDIUM | Verify alignment is point-in-time (no future features) |
| R4 | Feature standardization leaks test statistics | MEDIUM | Do not standardize (tree models don't need it) |
| R5 | Alpha selection based on IS performance, then evaluated IS | HIGH | Use SFI/MDA (OOS via purged CV) for feature selection |
| R6 | Correlated alphas create redundant signal, inflating importance | MEDIUM | Correlation pruning before model training |
| R7 | Alpha lookback creates temporal dependence between samples | MEDIUM | PurgedKFold handles this via t1 and embargo |
| R8 | VWAP approximation introduces bias | LOW | Acknowledge as limitation — not leakage |

### 8.2 Validation Guarantees That Must Remain Intact

1. **PurgedKFold unchanged.** The same PurgedKFold (n_splits=5, t1=t1, pct_embargo=0.01) must be used for all CV evaluations. No new validation methodology.

2. **Sample weights unchanged.** The same sample weights from Stage 2 must propagate into all fits and scores. Alpha features do not change the weighting scheme.

3. **OOS predictions unchanged in methodology.** If the augmented model is used for meta-labeling, OOS predictions must still be generated via the PurgedKFold loop.

4. **DSR trial count updated.** If K additional model configurations are tried with alpha features, `num_trials` in the DSR computation must increase by K.

---

## 9. Feature Diagnostics Framework

### 9.1 Per-Alpha Diagnostic Checks

For each computed alpha, run:

```python
def diagnose_alpha(alpha_series, name):
    """Comprehensive single-alpha diagnostics."""
    report = {
        'name': name,
        'n_total': len(alpha_series),
        'n_nan': alpha_series.isna().sum(),
        'pct_nan': alpha_series.isna().mean(),
        'mean': alpha_series.mean(),
        'std': alpha_series.std(),
        'min': alpha_series.min(),
        'max': alpha_series.max(),
        'n_unique': alpha_series.nunique(),
        'n_inf': np.isinf(alpha_series).sum(),
        'is_constant': alpha_series.std() < 1e-8,
    }
    # Stationarity
    clean = alpha_series.dropna()
    if len(clean) > 100:
        from statsmodels.tsa.stattools import adfuller
        adf = adfuller(clean, maxlag=1, regression='c', autolag=None)
        report['adf_pvalue'] = adf[1]
        report['is_stationary'] = adf[1] < 0.05
    # Autocorrelation (lag 1)
    report['autocorr_1'] = clean.autocorr(1) if len(clean) > 10 else None
    return report
```

### 9.2 Cross-Feature Redundancy Analysis

```python
def find_redundant_features(feature_df, threshold=0.8):
    """Identify pairs of features with |correlation| > threshold."""
    corr = feature_df.corr()
    redundant = []
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if abs(corr.iloc[i, j]) > threshold:
                redundant.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    return sorted(redundant, key=lambda x: -abs(x[2]))
```

---

## 10. Computational Considerations

### 10.1 Complexity Analysis

For N=5114 daily observations and K alphas:
- Simple alphas (delta, delay): O(N) per alpha
- Rolling statistics (stddev, correlation): O(N × d) per alpha, where d is window size
- `ts_rank()`: O(N × d × log(d)) per alpha (ranking within each window)
- `decay_linear()`: O(N × d) per alpha

Total for ~50 alphas: O(50 × N × d_max) ≈ O(50 × 5114 × 252) ≈ 64M operations. This runs in seconds on a modern CPU. No parallelization needed.

### 10.2 Memory

50 alphas × 5114 rows × 8 bytes = ~2 MB. Negligible.

### 10.3 Optimization Recommendations

- Compute all alphas once, cache as `nvda_alpha_features.parquet`
- Use vectorized pandas operations exclusively (no Python for-loops over rows)
- For `ts_rank()`, consider using `scipy.stats.rankdata` inside the rolling apply for speed

---

## 11. Model Retraining and Evaluation

### 11.1 Comparison Framework

| Metric | Baseline (15 features) | Augmented (15 + K alphas) |
|--------|----------------------|-------------------------|
| Purged CV Accuracy | from Stage 4 | recomputed |
| Purged CV F1 (meta) | from Stage 7 | recomputed |
| Sharpe Ratio | from Stage 8 | recomputed |
| PSR | from Stage 8 | recomputed |
| DSR | from Stage 8 | recomputed (higher N) |
| Max Drawdown | from Stage 8 | recomputed |
| Top-5 MDA features | from Stage 6 | recomputed — do any alphas appear? |

### 11.2 Success Criteria

The alpha integration is considered successful if ANY of:
1. At least one alpha feature appears in the top-10 MDA importance
2. Augmented model CV accuracy > baseline accuracy by > 1 percentage point
3. Augmented model Sharpe ratio > baseline Sharpe ratio

The integration is NOT considered a failure if none of these hold — a negative result ("formulaic alphas add no value to single-stock daily NVDA prediction") is itself a valid research finding, consistent with the alpha-decay hypothesis and the single-asset limitation.

---

## 12. Expected Research Outcomes

### 12.1 Realistic Expectations

Most alphas will fail to add predictive value for single-stock NVDA. This is expected because:

1. **Alpha decay:** The paper's alphas were in production in 2013-2015. Ten years of market microstructure evolution, crowding, and adaptation have likely eroded their edge.

2. **Single-asset limitation:** Cross-sectional alphas exploit relative mispricing between stocks. Single-stock prediction is fundamentally harder.

3. **Daily frequency:** The paper's alphas have average holding periods of 0.6-6.4 days. At daily frequency with 10-day barriers, the signals may be too fast or too slow for the event-based labeling scheme.

4. **Sample size:** With 195 samples, even a genuinely predictive alpha may fail to demonstrate statistical significance.

### 12.2 What Would Be Interesting

- If momentum-style alphas (e.g., #19, #39, #52 using 250-day returns) correlate with the existing `ret_60d` feature, confirming feature redundancy
- If microstructure-style alphas (e.g., #12, #41, #42, #101) provide information beyond the existing Amihud/Roll/CS features
- If the feature importance landscape shifts (e.g., alpha features displace `amihud_illiquidity` as most important)

---

## 13. Final Professional Recommendations

### 13.1 Implementation Priorities

1. **Start with Tier 1 alphas only.** These are mathematically sound for single-asset use. Get 15-20 working, validated, and integrated before attempting Tier 2.

2. **Respect the sample-size constraint.** With 195 samples, the total feature count (original + alphas) must not exceed ~25-30. Aggressive pruning is essential.

3. **Document every adaptation.** Every `rank()` → `ts_rank()` conversion, every excluded alpha, every VWAP approximation must be documented in the notebook with an explanation of why the adaptation was made and what was lost.

4. **Do not touch the validation infrastructure.** PurgedKFold, sample weights, embargo, OOS prediction generation, meta-labeling, and backtesting must remain exactly as implemented. The only change is the feature matrix.

5. **Treat negative results as results.** If the alphas add no value, report this honestly. A well-validated null result is more valuable than a weakly validated positive result.

### 13.2 What an AFML Professor Would Criticize

1. **"You added 50 features to 195 samples without pruning."** — This guarantees overfitting. Always prune before training.

2. **"You used cross-sectional rank on a single stock."** — This produces a constant, which is computationally wasteful and methodologically wrong.

3. **"You selected features based on in-sample performance."** — Feature selection must use OOS metrics (MDA/SFI through purged CV).

4. **"You didn't adjust DSR for the additional trials."** — Every model configuration tried with different feature subsets is a trial. DSR must reflect the total.

5. **"You changed the VWAP definition without acknowledging it."** — Approximations must be documented, not hidden.

### 13.3 Architecture Recommendation

```
CORRECT FLOW:
  Raw OHLCV → [Alpha Engine] → Alpha Features (5114 rows × K cols)
                                      ↓
  Existing Features (5114 rows × 17 cols)
                                      ↓
  [Merge + Align to Events] → Combined Features (195 rows × (17+K') cols, K' ≤ K after pruning)
                                      ↓
  [Feature Diagnostics] → Pruned Features (195 rows × ≤30 cols)
                                      ↓
  [STANDARD AFML PIPELINE — UNCHANGED]
  PurgedKFold → MDI/MDA/SFI → Tuning → Meta-labeling → Backtest → DSR

INCORRECT FLOW (DO NOT DO):
  Raw OHLCV → [All 101 alphas + all features] → 195 × 120 → [Model] → "great accuracy!"
  (This is guaranteed overfitting on 195 samples with 120 features.)
```

---

*End of Integration Specification*
