# 101 Formulaic Alphas × 10-Stock AFML Pipeline
# Master Implementation Plan with Claude Code Prompts

**Date:** May 15, 2026
**Universe:** AAPL, AMZN, NVDA, GOOGL, JNJ, JPM, MSFT, XOM, META, TSLA
**Scope:** Full integration — data acquisition through backtesting

---

## TABLE OF CONTENTS

1. Universe Analysis & Data Strategy
2. Alpha Classification for 10-Stock Universe
3. Architecture Blueprint
4. PHASE 1 — Data Acquisition & Per-Stock AFML Pipeline
5. PHASE 2 — Alpha Engine Build (Operators + Registry)
6. PHASE 3 — Alpha Computation on Panel
7. PHASE 4 — Pooled Modelling Dataset & Multi-Asset PurgedKFold
8. PHASE 5 — Model Training, Feature Importance, Tuning
9. PHASE 6 — Meta-Labeling, Bet Sizing, Backtesting
10. PHASE 7 — Full Validation & Final Report
11. Claude Code Prompt Sequence (Ready to Execute)

---

## 1. Universe Analysis & Data Strategy

### 1.1 Stock Universe

```
Ticker  Sector                  Sub-Industry              IPO         Data From
──────  ─────────────────────── ────────────────────────── ──────────  ─────────
AAPL    Information Technology  Consumer Electronics       1980-12     2005-01
AMZN    Consumer Discretionary  Internet Retail            1997-05     2005-01
NVDA    Information Technology  Semiconductors             1999-01     2005-01
GOOGL   Communication Services Interactive Media           2004-08     2005-01
JNJ     Health Care             Pharmaceuticals            1944-01     2005-01
JPM     Financials              Diversified Banks          1978-01     2005-01
MSFT    Information Technology  Systems Software           1986-03     2005-01
XOM     Energy                  Integrated Oil & Gas       1970-01     2005-01
META    Communication Services  Interactive Media          2012-05     2012-06
TSLA    Consumer Discretionary  Automobile Manufacturers   2010-06     2010-07
```

### 1.2 Date Intersection

META's IPO (May 2012) is the binding constraint. The common period where all 10 stocks have data is approximately **June 2012 – April 2025** (~3,230 trading days). However, several alpha formulas require 250-day lookbacks, so the first usable alpha values begin around June 2013, giving ~3,000 effective trading days.

**Decision:** Download each stock's full available history from 2005 (or IPO if later). Compute per-stock AFML stages (Stages 0–3) on each stock's full history independently. Compute cross-sectional alpha features only from June 2012 onward (when all 10 stocks have data). The per-stock time-series features use each stock's full history.

### 1.3 GICS Sector Groups for indneutralize

```
Sector                   Stocks         Count   indneutralize viable?
──────────────────────── ────────────── ─────   ──────────────────────
Information Technology   AAPL,MSFT,NVDA   3     YES (3 stocks to demean)
Communication Services   GOOGL,META       2     YES (2 stocks)
Consumer Discretionary   AMZN,TSLA        2     YES (2 stocks)
Health Care              JNJ              1     NO  (singleton → zero)
Financials               JPM              1     NO  (singleton → zero)
Energy                   XOM              1     NO  (singleton → zero)
```

For the ~20 alphas using `indneutralize`, we apply sector-level demeaning. Singletons get full-universe demeaning (subtract mean across all 10 stocks) as a fallback. This is a documented approximation.

### 1.4 Data Schema

Every stock must have these columns after cleaning:

```
Column       Type       Description
──────────── ────────── ─────────────────────────────────
Date         datetime   Trading day (index)
Open         float64    Split-adjusted open
High         float64    Split-adjusted high
Low          float64    Split-adjusted low
Close        float64    Split-adjusted close (= Adj Close)
Volume       int64      Trading volume in shares
```

Derived fields computed during alpha engine:
```
returns      float64    close.pct_change()
vwap         float64    (high + low + close) / 3
adv{d}       float64    (close * volume).rolling(d).mean()
```

---

## 2. Alpha Classification for 10-Stock Universe

After auditing every alpha formula against the paper, here is the complete classification:

### 2.1 Tier 1 — Pure Time-Series (no rank, no indneutralize, no cap)
These work identically whether you have 1 stock or 1000.

```
Alpha#6, #7, #9, #12, #21, #23, #24, #26, #35, #41, #43, #46, #49, #51, #53, #54, #101
Total: 17 alphas
```

### 2.2 Tier 2 — Cross-Sectional rank() (now functional with 10 stocks)
These use `rank()` which operates across 10 stocks at each date.

```
Alpha#1, #2, #3, #4, #5, #8, #10, #11, #13, #14, #15, #16, #17, #18, #19, #20, #22,
#25, #27, #28, #29, #30, #31, #32, #33, #34, #36, #37, #38, #39, #40, #42, #44, #45,
#47, #50, #52, #55, #57, #60, #62, #64, #65, #66, #68, #71, #72, #73, #75, #77, #78,
#81, #83, #84, #85, #86, #88, #92, #94, #95, #96, #98, #99
Total: ~63 alphas
```

### 2.3 Tier 3 — indneutralize (partially functional with sector groups)

```
Alpha#48, #58, #59, #63, #67, #69, #70, #76, #79, #80, #82, #87, #89, #90, #91, #93,
#97, #100
Total: 18 alphas (implementable with sector-level approximation)
```

### 2.4 Tier 4 — Excluded (requires cap / market capitalization)

```
Alpha#56
Total: 1 alpha (excluded unless market cap data is sourced)
```

### 2.5 Missing from paper (Alpha#8 listed but no #44 in some versions — verify during implementation)

**Total implementable: ~98 of 101 alphas** (excluding Alpha#56, Alpha#61 if indneutralize fails, Alpha#74 if adv30 issues arise). Conservative estimate: **85–90 alphas will produce valid signals.**

---

## 3. Architecture Blueprint

### 3.1 Final Repository Structure

```
project_root/
├── data/
│   ├── raw/
│   │   ├── AAPL_raw.csv
│   │   ├── AMZN_raw.csv
│   │   ├── NVDA_raw.csv   (existing)
│   │   ├── GOOGL_raw.csv
│   │   ├── JNJ_raw.csv
│   │   ├── JPM_raw.csv
│   │   ├── MSFT_raw.csv
│   │   ├── XOM_raw.csv
│   │   ├── META_raw.csv
│   │   └── TSLA_raw.csv
│   └── processed/
│       ├── per_stock/
│       │   ├── NVDA_clean.parquet        (existing, kept)
│       │   ├── NVDA_labels.parquet       (existing, kept)
│       │   ├── NVDA_weights.parquet      (existing, kept)
│       │   ├── NVDA_ts_features.parquet
│       │   ├── AAPL_clean.parquet
│       │   ├── AAPL_labels.parquet
│       │   └── ... (for all 10 stocks)
│       ├── panel_ohlcv.parquet           # 10-stock panel (dates × tickers)
│       ├── panel_alpha_features.parquet  # Cross-sectional alpha values
│       ├── pooled_modelling.parquet      # Final: all stocks' events merged
│       ├── alpha_diagnostics.parquet     # Per-alpha diagnostic statistics
│       └── ... (existing NVDA artifacts kept for reference)
├── src/
│   ├── alphas/
│   │   ├── __init__.py
│   │   ├── operators.py          # All reusable operators
│   │   ├── formulas.py           # All 101 alpha formulas
│   │   ├── engine.py             # Master compute: panel → alpha features
│   │   ├── registry.py           # Alpha metadata + tier classification
│   │   └── diagnostics.py        # Per-alpha QA: NaN, variance, stationarity
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── per_stock.py          # Run AFML Stages 0-3 for any stock
│   │   └── pooling.py            # Stack per-stock results + merge alphas
│   ├── data_structures.py        (existing)
│   ├── labeling.py               (existing)
│   ├── sample_weights.py         (existing)
│   ├── fracdiff.py               (existing)
│   ├── features.py               (existing, extended)
│   ├── cross_validation.py       (existing, extended for multi-asset)
│   ├── modelling.py              (existing)
│   ├── feature_importance.py     (existing)
│   ├── hyperparameter_tuning.py  (existing)
│   ├── bet_sizing.py             (existing)
│   ├── backtesting.py            (existing, extended for portfolio)
│   └── ...
├── notebooks/
│   ├── 15_data_acquisition.ipynb
│   ├── 16_per_stock_pipeline.ipynb
│   ├── 17_alpha_engine.ipynb
│   ├── 18_alpha_diagnostics.ipynb
│   ├── 19_pooled_training.ipynb
│   ├── 20_meta_labeling_portfolio.ipynb
│   ├── 21_portfolio_backtest.ipynb
│   └── 22_final_comparison.ipynb
├── tests/
│   ├── test_operators.py
│   ├── test_alphas.py
│   └── test_multi_asset_cv.py
└── configs/
    └── universe.json             # Stock list, sectors, parameters
```

---

## 4–10. DETAILED PHASE DESCRIPTIONS

Each phase below is designed as a self-contained Claude Code prompt. They must be executed in order. Each prompt includes everything Claude Code needs: context, objectives, exact steps, validation criteria, and error handling.

---

## 11. Claude Code Prompt Sequence

---

### ═══════════════════════════════════════════════
### PROMPT 1 OF 7 — DATA ACQUISITION & CLEANING
### ═══════════════════════════════════════════════

```
You are a senior quantitative researcher. You have an existing AFML pipeline for NVDA in this repository. Your task is to download and clean daily OHLCV data for 9 additional stocks and prepare all 10 for the alpha integration pipeline.

EXISTING STATE:
- data/raw/NVDA_raw.csv exists (5114 rows, 2005-01-03 to 2025-04-30)
- The NVDA AFML pipeline (Stages 0-8) is fully implemented and validated
- All src/ modules exist and work

YOUR TASK:

STEP 1: Download data for 9 stocks
Download daily OHLCV data for: AAPL, AMZN, GOOGL, JNJ, JPM, MSFT, XOM, META, TSLA
Use yfinance: pip install yfinance
For each stock, download the maximum available history.
Save as data/raw/{TICKER}_raw.csv with columns: Date, Adj Close, Close, High, Low, Open, Volume
Match the exact column format of NVDA_raw.csv.

```python
import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'AMZN', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'XOM', 'META', 'TSLA']
for ticker in tickers:
    df = yf.download(ticker, start='2000-01-01', end='2025-05-01', auto_adjust=False)
    df = df[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
    df.index.name = 'Date'
    df.to_csv(f'data/raw/{ticker}_raw.csv')
    print(f"{ticker}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
```

STEP 2: Validate every downloaded file
For each of the 10 stocks (including NVDA):
- Assert rows >= 3000
- Assert no null values in any column
- Assert all prices > 0
- Assert all volumes > 0
- Assert dates are monotonically increasing
- Print: ticker, row count, date range, adj close range, volume range

STEP 3: Create configs/universe.json

```json
{
  "tickers": ["AAPL", "AMZN", "NVDA", "GOOGL", "JNJ", "JPM", "MSFT", "XOM", "META", "TSLA"],
  "sectors": {
    "AAPL": "Information Technology",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Information Technology",
    "GOOGL": "Communication Services",
    "JNJ": "Health Care",
    "JPM": "Financials",
    "MSFT": "Information Technology",
    "XOM": "Energy",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary"
  },
  "common_start_date": null,
  "common_end_date": null
}
```

After downloading, compute the intersection date range (the first date where ALL 10 stocks have data) and update common_start_date and common_end_date.

STEP 4: Build the panel OHLCV dataset
Load all 10 stocks, align to the common date range, and create a panel DataFrame:

```python
panel = pd.DataFrame()
for ticker in tickers_all:
    df = pd.read_csv(f'data/raw/{ticker}_raw.csv', parse_dates=['Date'], index_col='Date')
    df = df.loc[common_start:common_end]
    df['ticker'] = ticker
    panel = pd.concat([panel, df])

panel = panel.reset_index().set_index(['Date', 'ticker']).sort_index()
panel.to_parquet('data/processed/panel_ohlcv.parquet')
```

Print the final panel shape. It should be approximately (common_days × 10) rows × 6 columns.

STEP 5: Verify panel integrity
- Assert every ticker has the same dates (no missing trading days for any stock)
- Assert no NaN in any column
- For any date, assert exactly 10 rows (one per ticker)
- Print the 5 dates with lowest total volume across all stocks (sanity check)

VALIDATION CRITERIA:
- 10 CSV files in data/raw/
- configs/universe.json exists with correct sectors
- panel_ohlcv.parquet exists with shape ~(32000, 6) give or take
- Zero NaN values
- All 10 stocks present for every trading day in the common range

If yfinance fails or is blocked, create a helper script that explains what data format is needed and where to source it (Alpha Vantage, Polygon, etc.) so the user can manually provide the CSVs.
```

---

### ═══════════════════════════════════════════════
### PROMPT 2 OF 7 — PER-STOCK AFML PIPELINE
### ═══════════════════════════════════════════════

```
You are a senior quantitative researcher continuing the AFML pipeline expansion. The panel_ohlcv.parquet and all 10 stock CSVs are ready.

EXISTING STATE:
- NVDA already has Stages 0-8 completed (labels, weights, features, models)
- The existing src/ modules (labeling.py, sample_weights.py, fracdiff.py, features.py, data_structures.py) work correctly for NVDA
- 9 new stocks need Stages 0-3 run on them

YOUR TASK: Run AFML Stages 0-3 for each of the 9 new stocks, producing per-stock labels, weights, and time-series features. Then stack all 10 stocks into pooled datasets.

Create src/pipeline/per_stock.py:

```python
"""
Run AFML Stages 0-3 for a single stock.
Reuses existing src/ modules without modification.
"""
import pandas as pd
import numpy as np
from src.data_structures import cusum_filter
from src.labeling import get_daily_vol, apply_triple_barrier, get_bins
# ... import all needed functions

def run_stages_0_to_3(ticker, raw_path, output_dir='data/processed/per_stock'):
    """
    Run the full per-stock AFML pipeline:
    Stage 0: Load and clean
    Stage 1: CUSUM filter
    Stage 2: Triple-barrier labels + sample weights
    Stage 3: Time-series features + fracdiff
    
    Returns dict of DataFrames.
    """
    # Stage 0
    raw = pd.read_csv(raw_path, parse_dates=['Date'], index_col='Date')
    close = raw['Adj Close']
    assert (close > 0).all() and close.isnull().sum() == 0
    
    # Stage 1: CUSUM with adaptive h
    daily_vol = get_daily_vol(close, span=50)
    h = daily_vol.dropna().mean()  # adaptive threshold
    cusum_events = cusum_filter(close, h)
    
    # If too few/many events, adjust h
    while len(cusum_events) < 150 and h > daily_vol.mean() * 0.3:
        h *= 0.8
        cusum_events = cusum_filter(close, h)
    while len(cusum_events) > 800:
        h *= 1.2
        cusum_events = cusum_filter(close, h)
    
    print(f"  {ticker}: h={h:.4f}, {len(cusum_events)} CUSUM events")
    
    # Stage 2: Triple-barrier labels
    # [Use the SAME labeling functions as NVDA pipeline]
    # pt_sl = [1, 1], vertical barrier = 10 days
    # Compute labels, then sample weights
    
    # Stage 3: Time-series features
    # [Use the SAME feature functions as NVDA pipeline]
    # ret_5d, ret_10d, ret_20d, ret_60d, momentum_12_1, rsi_14
    # vol_20d, vol_50d, log_dollar_volume, volume_ratio
    # corwin_schultz_spread (clipped >= 0), bekker_parkinson_vol
    # amihud_illiquidity, roll_spread
    # shannon_entropy, lempel_ziv_complexity
    # fracdiff(log(close), d=optimal d* for this stock)
    
    # Fracdiff: find optimal d* per stock
    # Sweep d from 0.05 to 0.50 in steps of 0.05
    # Select minimum d where ADF p < 0.05 AND corr with log price > 0.85
    
    # Save per-stock artifacts
    # {ticker}_clean.parquet, {ticker}_labels.parquet, 
    # {ticker}_weights.parquet, {ticker}_ts_features.parquet
    
    return results
```

EXECUTION:
For each of the 9 NEW stocks (NVDA already done):
1. Call run_stages_0_to_3()
2. Print: ticker, n_cusum_events, n_labeled_samples, optimal_d*, label_distribution
3. Save all artifacts to data/processed/per_stock/

For NVDA, copy/symlink existing artifacts:
- nvda_labels.parquet → per_stock/NVDA_labels.parquet
- nvda_sample_weights.parquet → per_stock/NVDA_weights.parquet
- etc.

THEN create src/pipeline/pooling.py:

```python
def create_pooled_dataset(tickers, per_stock_dir='data/processed/per_stock'):
    """Stack all per-stock labels, weights, and time-series features."""
    all_labels = []
    all_weights = []
    all_features = []
    
    for ticker in tickers:
        labels = pd.read_parquet(f'{per_stock_dir}/{ticker}_labels.parquet')
        labels['ticker'] = ticker
        all_labels.append(labels)
        
        weights = pd.read_parquet(f'{per_stock_dir}/{ticker}_weights.parquet')
        weights['ticker'] = ticker
        all_weights.append(weights)
        
        features = pd.read_parquet(f'{per_stock_dir}/{ticker}_ts_features.parquet')
        features['ticker'] = ticker
        all_features.append(features)
    
    pooled_labels = pd.concat(all_labels).sort_index()
    pooled_weights = pd.concat(all_weights).sort_index()
    pooled_features = pd.concat(all_features).sort_index()
    
    return pooled_labels, pooled_weights, pooled_features
```

VALIDATION CRITERIA:
- Every stock produces >= 100 labeled samples (if fewer, h calibration is wrong)
- Total pooled samples across 10 stocks: expected 1,500-2,500
- All label values in {-1, +1}
- All weights > 0
- No NaN in any feature column (after the per-stock lookback window is satisfied)
- Print a summary table:

```
Ticker  Events  Labels  d*    +1    -1    Avg_Weight
NVDA     195     195   0.25  114    81    1.00
AAPL     ???     ???   ???   ???   ???    ???
...
TOTAL           ????
```

CRITICAL: The existing NVDA pipeline results (models, backtest) are NOT invalidated. They remain as the single-stock baseline. This prompt creates the multi-stock expansion alongside them.

If any src/ function fails on a new stock (e.g., different column names, edge cases), fix the function to be generic, then revalidate that NVDA still produces the same results.
```

---

### ═══════════════════════════════════════════════
### PROMPT 3 OF 7 — ALPHA ENGINE BUILD
### ═══════════════════════════════════════════════

```
You are a senior quantitative researcher. Your task is to build a production-grade alpha engine that computes the 101 Formulaic Alphas from the WorldQuant paper on a 10-stock panel dataset.

REFERENCES:
- Paper: "101 Formulaic Alphas" by Kakushadze (2015)
- GitHub reference (DO NOT COPY — it has bugs): https://github.com/yli188/WorldQuant_alpha101_code
- Paper Appendix A.1 has all formulas; A.2 has function definitions; A.3 has input data definitions

KNOWN BUGS IN THE GITHUB IMPLEMENTATION (from audit):
1. rank() uses axis=1 (cross-sectional) which is CORRECT for multi-asset, but helper functions (rank, correlation, etc.) have inconsistent axis handling
2. adv{d} uses sma(volume, d) — WRONG. Paper says "average daily dollar volume": adv{d} = sma(close * volume, d)
3. vwap computed from S_DQ_AMOUNT — unavailable. Use (high + low + close) / 3
4. min_periods not enforced — produces values from partial windows
5. alpha001 modifies self.close in-place: inner = self.close; inner[self.returns < 0] = ... — THIS CORRUPTS THE CLOSE SERIES FOR ALL SUBSEQUENT ALPHAS. Must use .copy()
6. Many alphas silently return None or incomplete Series
7. No NaN handling — inf values propagate

YOUR TASK: Build the alpha engine FROM SCRATCH, guided by the paper's formulas (not the GitHub code). Use the GitHub code only as a cross-reference to verify your implementation, never as a source to copy.

STEP 1: Create src/alphas/operators.py

Implement every operator from paper Appendix A.2. Each function must:
- Accept pandas Series (time-series ops) or DataFrame (cross-sectional ops)
- Use min_periods equal to the window size (NO partial windows)
- Handle NaN gracefully (propagate, don't crash)
- Be pure functions (no side effects, no in-place modification)

Required operators:
```python
# Time-series operators
def ts_sum(x, d): ...           # sum over past d days
def ts_mean(x, d): ...          # mean over past d days (= ts_sum/d)
def ts_std(x, d): ...           # std over past d days
def ts_rank(x, d): ...          # percentile rank within past d days
def ts_min(x, d): ...           # min over past d days
def ts_max(x, d): ...           # max over past d days
def ts_argmax(x, d): ...        # which day (0-indexed from d ago) was the max
def ts_argmin(x, d): ...        # which day (0-indexed from d ago) was the min
def ts_corr(x, y, d): ...       # rolling correlation over d days
def ts_cov(x, y, d): ...        # rolling covariance over d days
def ts_product(x, d): ...       # product over past d days
def delta(x, d): ...            # x_t - x_{t-d}
def delay(x, d): ...            # x_{t-d}
def decay_linear(x, d): ...     # linearly decaying WMA: weights d, d-1, ..., 1
def signed_power(x, a): ...     # sign(x) * |x|^a

# Cross-sectional operators (operate across stocks at each date)
def rank_cs(df): ...             # cross-sectional percentile rank (axis=1)
def scale_cs(df, a=1): ...       # rescale so sum(|x|) = a at each date
def indneutralize_cs(df, groups): ...  # demean within industry groups

# Utility
def adv(close, volume, d): ...  # average daily DOLLAR volume = sma(close*volume, d)
```

STEP 2: Create src/alphas/registry.py

```python
SECTOR_MAP = {
    'AAPL': 'IT', 'AMZN': 'CD', 'NVDA': 'IT', 'GOOGL': 'CS',
    'JNJ': 'HC', 'JPM': 'FN', 'MSFT': 'IT', 'XOM': 'EN',
    'META': 'CS', 'TSLA': 'CD'
}

ALPHA_REGISTRY = {
    'alpha001': {'tier': 2, 'max_lookback': 20, 'uses_vwap': False, 'uses_adv': False, 'uses_indneutralize': False},
    # ... for all 101 alphas
}
```

STEP 3: Create src/alphas/formulas.py

Implement ALL 101 alpha formulas. Each alpha is a function that takes a dict of wide-format DataFrames (dates × tickers) and returns a wide-format DataFrame of alpha values.

```python
def alpha001(data):
    """
    Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    """
    close = data['close'].copy()  # MUST copy to avoid corruption
    returns = data['returns']
    
    inner = close.copy()
    cond = returns < 0
    inner[cond] = ts_std(returns, 20)[cond]
    
    powered = signed_power(inner, 2.0)
    argmaxed = ts_argmax(powered, 5)
    ranked = rank_cs(argmaxed)
    return ranked - 0.5

def alpha002(data):
    """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    x = rank_cs(delta(np.log(data['volume']), 2))
    y = rank_cs((data['close'] - data['open']) / data['open'])
    return -1 * ts_corr(x, y, 6)

# ... continue for ALL 101 alphas ...

def alpha101(data):
    """Alpha#101: ((close - open) / ((high - low) + .001))"""
    return (data['close'] - data['open']) / ((data['high'] - data['low']) + 0.001)
```

CRITICAL IMPLEMENTATION RULES:
1. NEVER modify data dict values in-place. Always .copy() before mutation.
2. For rank(): use rank_cs() which ranks across the 10 stocks (axis=1).
3. For indneutralize(): use indneutralize_cs() with SECTOR_MAP.
4. For adv{d}: compute as sma(close * volume, d), NOT sma(volume, d).
5. For vwap: use (high + low + close) / 3.
6. For non-integer window parameters (e.g., 3.92795): use floor(d).
7. For ts_argmax/ts_argmin: return 0-indexed position within the window.
8. Replace any inf values with NaN after computation.
9. Clip extreme values: if |alpha| > 1e6, set to NaN.

STEP 4: Create src/alphas/engine.py

```python
def compute_all_alphas(panel_ohlcv, sector_map):
    """
    Master compute function.
    
    panel_ohlcv: DataFrame with MultiIndex (Date, ticker), columns [Open,High,Low,Close,Volume]
    Returns: DataFrame with MultiIndex (Date, ticker), columns = alpha names
    """
    # Pivot to wide format
    close = panel_ohlcv['Close'].unstack('ticker')
    open_ = panel_ohlcv['Open'].unstack('ticker')
    high = panel_ohlcv['High'].unstack('ticker')
    low = panel_ohlcv['Low'].unstack('ticker')
    volume = panel_ohlcv['Volume'].unstack('ticker')
    returns = close.pct_change()
    vwap = (high + low + close) / 3
    
    data = {
        'close': close, 'open': open_, 'high': high, 'low': low,
        'volume': volume, 'returns': returns, 'vwap': vwap
    }
    
    results = {}
    for alpha_name, alpha_func in get_all_alpha_functions():
        try:
            result = alpha_func(data)
            # Replace inf with NaN
            result = result.replace([np.inf, -np.inf], np.nan)
            results[alpha_name] = result
            n_nan = result.isnull().sum().sum()
            n_total = result.size
            print(f"  ✓ {alpha_name}: {n_nan}/{n_total} NaN ({n_nan/n_total*100:.1f}%)")
        except Exception as e:
            print(f"  ✗ {alpha_name}: FAILED — {e}")
    
    # Stack back to MultiIndex
    alpha_panel = {}
    for name, df in results.items():
        alpha_panel[name] = df.stack()
    
    return pd.DataFrame(alpha_panel)
```

STEP 5: Create tests/test_operators.py

```python
def test_rank_cs():
    df = pd.DataFrame({'A': [1,2,3], 'B': [3,2,1], 'C': [2,3,2]})
    r = rank_cs(df)
    # Row 0: A=1(rank 1/3=0.33), B=3(rank 3/3=1.0), C=2(rank 2/3=0.67)
    assert abs(r.iloc[0, 0] - 1/3) < 0.01
    assert abs(r.iloc[0, 1] - 1.0) < 0.01

def test_adv_uses_dollar_volume():
    close = pd.Series([100, 101, 102])
    volume = pd.Series([1000, 2000, 3000])
    result = adv(close, volume, 2)
    # adv = sma(close*volume, 2)
    # Day 2: mean([100*1000, 101*2000]) = mean([100000, 202000]) = 151000
    expected = (100*1000 + 101*2000) / 2
    assert abs(result.iloc[1] - expected) < 0.01

def test_alpha001_does_not_corrupt_close():
    # This was the major bug in the GitHub implementation
    data = make_test_data()
    close_before = data['close'].copy()
    alpha001(data)
    pd.testing.assert_frame_equal(data['close'], close_before)
```

STEP 6: Run and validate
- Compute all alphas on panel_ohlcv.parquet
- Print the number of successfully computed alphas
- Print the number of failed alphas with error messages
- Save panel_alpha_features.parquet
- For each alpha, print: name, NaN%, mean, std, n_unique_values_per_stock

TARGET: >= 85 alphas compute successfully with < 30% NaN rate.
Any alpha with > 50% NaN should be flagged and investigated.
Any alpha that is constant (std < 1e-8) should be flagged and excluded.
```

---

### ═══════════════════════════════════════════════
### PROMPT 4 OF 7 — ALPHA DIAGNOSTICS & PRUNING
### ═══════════════════════════════════════════════

```
You are a senior quantitative researcher. The alpha engine has computed ~85-90 alpha features on the 10-stock panel. Your task is to run comprehensive diagnostics and prune the feature set to a manageable size for the AFML modelling pipeline.

STEP 1: Load computed alphas
Load panel_alpha_features.parquet. Verify shape: ~(32000, 85-90).

STEP 2: Per-alpha diagnostics
For each alpha, compute:
- NaN percentage (across all stocks and dates)
- Mean, std, min, max, skewness, kurtosis
- Number of unique values per stock (detect constants)
- ADF test for stationarity (on each stock separately, report median p-value)
- Autocorrelation at lag 1 (averaged across stocks)

Create src/alphas/diagnostics.py with a diagnose_all_alphas() function.
Save results as data/processed/alpha_diagnostics.parquet.

STEP 3: Exclusion rules (apply in order)
1. EXCLUDE if NaN% > 40% (too much missing data after lookback)
2. EXCLUDE if std < 1e-8 for any stock (constant — carries no information)
3. EXCLUDE if contains any inf values (numerical instability)

Print: "Excluded X alphas in step 1, Y in step 2, Z in step 3. Remaining: N alphas."

STEP 4: Cross-alpha correlation matrix
Compute the correlation matrix across all surviving alphas.
For this computation, use only the NVDA column to avoid cross-stock averaging artifacts.
Save a correlation heatmap as reports/figures/P_alpha_correlation_heatmap.png.

STEP 5: Redundancy pruning
For each pair of alphas with |correlation| > 0.85:
- Keep the one with lower median ADF p-value (more stationary)
- Drop the other
Print: "Pruned X redundant alphas. Remaining: N alphas."

STEP 6: Feature-count budget
The pooled dataset will have approximately 1,500-2,000 samples.
Maximum safe feature count = 50-60 (targeting 30+ samples per feature).
Current time-series features: 17 (existing AFML features).
Alpha budget: 50 - 17 = 33 alphas maximum.

If more than 33 alphas survive after pruning:
- Rank by median ADF p-value (most stationary first)
- Keep the top 33
Print: "Final alpha feature set: N alphas (within budget of 33)."

STEP 7: Save the curated alpha set
Save the list of surviving alpha names to configs/selected_alphas.json.
Save the pruned alpha features to data/processed/panel_alpha_features_pruned.parquet.

STEP 8: Generate diagnostic report
Create notebook 18_alpha_diagnostics.ipynb with:
- Table of all computed alphas with diagnostic statistics
- Correlation heatmap
- Distribution of NaN rates across alphas
- Distribution of stationarity across alphas
- List of excluded alphas with reasons
- List of pruned redundant pairs
- Final selected alpha list

VALIDATION:
- Selected alphas: between 15 and 33
- No selected alpha has NaN% > 40%
- No selected alpha is constant
- No pair of selected alphas has |corr| > 0.85
- configs/selected_alphas.json exists and is valid JSON
```

---

### ═══════════════════════════════════════════════
### PROMPT 5 OF 7 — POOLED DATASET & MULTI-ASSET CV
### ═══════════════════════════════════════════════

```
You are a senior quantitative researcher. Your task is to merge per-stock time-series features with cross-sectional alpha features into a pooled modelling dataset, and implement multi-asset PurgedKFold cross-validation.

STEP 1: Build the pooled modelling dataset
For each of the 10 stocks:
1. Load {ticker}_labels.parquet (event timestamps, t1, labels)
2. Load {ticker}_ts_features.parquet (17 time-series features)
3. Load alpha features for this ticker from panel_alpha_features_pruned.parquet
4. Align alpha features to this ticker's event timestamps (point-in-time lookup)
5. Merge: [17 ts_features] + [N alpha_features] + [label] + [weight] + [t1] + [ticker]

```python
pooled_rows = []
for ticker in tickers:
    labels = load(f'{ticker}_labels.parquet')
    ts_feat = load(f'{ticker}_ts_features.parquet')
    
    # Alpha features: for each event timestamp, look up the alpha value
    # for this ticker at this date
    alpha_feats = alpha_panel.xs(ticker, level='ticker')
    alpha_aligned = alpha_feats.reindex(labels.index)  # point-in-time alignment
    
    # Merge
    row = pd.concat([ts_feat, alpha_aligned, labels[['label','t1','weight']]], axis=1)
    row['ticker'] = ticker
    row = row.dropna()  # drop rows where any feature or alpha is NaN
    pooled_rows.append(row)

pooled = pd.concat(pooled_rows).sort_index()
pooled.to_parquet('data/processed/pooled_modelling.parquet')
```

Print:
- Total rows (target: 1,200-2,000)
- Rows per stock
- Total feature columns
- NaN count (should be 0 after dropna)
- Label distribution across entire pooled dataset
- Label distribution per stock

STEP 2: Implement multi-asset PurgedKFold
Create or extend src/cross_validation.py with:

```python
class MultiAssetPurgedKFold:
    """
    Time-block PurgedKFold for multi-asset datasets.
    
    Splits the TIME AXIS into contiguous blocks. All stocks' events
    in the same time block go to the same fold. This prevents
    cross-sectional leakage from alpha features.
    
    Purging: removes train samples whose label period [t0, t1]
    overlaps the test time block.
    
    Embargo: removes train samples within pct_embargo of the
    dataset length after the test block end.
    """
    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
    
    def split(self, X, y=None, groups=None):
        # Extract event times (ignoring ticker)
        event_times = X.index  # these are the event timestamps
        unique_times = sorted(set(event_times))
        n_times = len(unique_times)
        
        # Create time-based fold boundaries
        fold_boundaries = np.array_split(range(n_times), self.n_splits)
        
        for fold_i in range(self.n_splits):
            test_time_indices = fold_boundaries[fold_i]
            test_times = set([unique_times[j] for j in test_time_indices])
            test_start = min(test_times)
            test_end = max(test_times)
            
            # Test: all samples with event time in test_times
            test_mask = np.array([t in test_times for t in event_times])
            test_idx = np.where(test_mask)[0]
            
            # Train: all samples NOT in test
            train_mask = ~test_mask
            
            # Purge: remove train samples whose t1 > test_start
            # (their label period overlaps the test block)
            if self.t1 is not None:
                for i in np.where(train_mask)[0]:
                    if event_times[i] < test_start and self.t1.iloc[i] >= test_start:
                        train_mask[i] = False
            
            # Embargo: remove train samples just after test_end
            embargo_n = max(1, int(n_times * self.pct_embargo))
            embargo_cutoff_idx = min(
                unique_times.index(test_end) + embargo_n if test_end in unique_times else len(unique_times),
                n_times - 1
            )
            embargo_cutoff = unique_times[embargo_cutoff_idx]
            for i in np.where(train_mask)[0]:
                if test_end < event_times[i] <= embargo_cutoff:
                    train_mask[i] = False
            
            train_idx = np.where(train_mask)[0]
            yield train_idx, test_idx
    
    def get_n_splits(self):
        return self.n_splits
```

STEP 3: Validate the multi-asset CV
Run the following checks on the pooled dataset:

```python
pooled = pd.read_parquet('data/processed/pooled_modelling.parquet')
feature_cols = [c for c in pooled.columns if c not in 
                {'label','weight','t1','ticker','return','ret'}]
X = pooled[feature_cols]
y = pooled['label']
t1 = pooled['t1']

cv = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
    train_times = X.index[train_idx]
    test_times = X.index[test_idx]
    
    # CHECK 1: No temporal overlap
    assert max(train_times[train_times < min(test_times)]) < min(test_times) or \
           min(train_times[train_times > max(test_times)]) > max(test_times), \
           f"Fold {fold_i}: temporal overlap between train and test"
    
    # CHECK 2: All stocks in test block are test (no cross-sectional leakage)
    test_dates = set(test_times)
    for t in test_dates:
        train_at_t = [train_times[j] for j in range(len(train_times)) if train_times[j] == t]
        assert len(train_at_t) == 0, \
            f"Fold {fold_i}: train sample at test date {t} — CROSS-SECTIONAL LEAKAGE"
    
    # CHECK 3: Purging works
    if t1 is not None:
        train_t1 = t1.iloc[train_idx]
        test_start = min(test_times)
        leaking = train_t1[(train_t1.index < test_start) & (train_t1 > test_start)]
        assert len(leaking) == 0, f"Fold {fold_i}: {len(leaking)} purging failures"
    
    print(f"  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}, "
          f"test_range=[{min(test_times).date()}→{max(test_times).date()}] ✓")
```

STEP 4: Baseline CV on pooled dataset
Run PurgedKFold CV with RF on the pooled dataset using ALL features (17 ts + N alphas):

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=30,
                            max_features='sqrt', random_state=42)
cv = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
# Use cv_score with weighted scoring
```

Print:
- CV accuracy: mean ± std
- Majority baseline
- Whether the model beats baseline

STEP 5: Compare against NVDA-only baseline
Load the NVDA-only CV results from data/processed/cv_results.parquet.
Print side-by-side:

```
Configuration         Accuracy    Std     vs Baseline
────────────────────  ────────    ─────   ──────────
NVDA only (15 feat)   0.628       0.072   +4.3%
10-stock (17 feat)    ???         ???     ???
10-stock (17+N feat)  ???         ???     ???
```

VALIDATION:
- pooled_modelling.parquet has 1,200-2,000 rows
- Zero NaN in feature columns
- MultiAssetPurgedKFold passes all 3 leakage checks for all 5 folds
- CV accuracy is computable without errors
```

---

### ═══════════════════════════════════════════════
### PROMPT 6 OF 7 — FEATURE IMPORTANCE, TUNING, META-LABELING, BACKTEST
### ═══════════════════════════════════════════════

```
You are a senior quantitative researcher. The pooled modelling dataset is ready with 10 stocks' events and 17+N features (time-series + alphas). Your task is to run the full AFML pipeline: feature importance, hyperparameter tuning, meta-labeling, bet sizing, and backtesting.

STEP 1: Feature Importance (AFML Ch 8)
Run MDI, MDA, and SFI on the pooled dataset using MultiAssetPurgedKFold.

```python
# MDI: from fitted RF
# MDA: permutation importance with purged CV (weighted scoring)
# SFI: single-feature purged CV accuracy

# For each method, rank all features
# Print top-15 features by each method
# Identify: do ANY alpha features appear in the top-10 of ANY method?
```

Save: data/processed/feature_importance_pooled.parquet
Generate: reports/figures/P_pooled_mdi.png, P_pooled_mda.png, P_pooled_sfi.png

STEP 2: Feature pruning by tri-method consensus
Remove features that are bottom-5 in ALL THREE methods (MDI, MDA, SFI).
Report which features were removed.
Rebuild the feature matrix with the reduced set.

STEP 3: Hyperparameter tuning (AFML Ch 9)
Run 30 trials of randomized search for RF and XGB on the pruned pooled dataset.
Use MultiAssetPurgedKFold as the inner CV.
Use weighted accuracy as scoring metric.

Save: models/best_params_pooled.json, data/processed/tuning_log_pooled.parquet

Print:
```
Model    Best CV Acc    Std      Best Params
RF       ???            ???      {n_est, max_depth, min_leaf, max_feat}
XGB      ???            ???      {n_est, max_depth, lr, reg_lambda, subsample}
```

STEP 4: OOS Predictions (for meta-labeling)
Generate OOS predictions for ALL pooled samples using the tuned RF and MultiAssetPurgedKFold.
CRITICAL: Use the PurgedKFold loop — do NOT use model_final.pkl.

```python
oos_pred = pd.Series(dtype=float, index=X.index)
oos_prob = pd.Series(dtype=float, index=X.index)

for fold_i, (tr, te) in enumerate(cv.split(X)):
    clf = clone(tuned_rf)
    clf.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr].values)
    pred = clf.predict(X.iloc[te])
    proba = clf.predict_proba(X.iloc[te])
    # Store predictions for test indices
```

Assert: every sample has exactly one OOS prediction.
Print: OOS accuracy, side distribution.

STEP 5: Meta-labeling (AFML Ch 3.6-3.7)
For each sample:
  meta_label = 1 if (realized_return × predicted_side) > 0 else 0

Train meta-model (shallow RF, max_depth=3, class_weight='balanced') with PurgedKFold.
Use F1 as scoring metric (per AFML Snippet 9.1 for binary 0/1 labels).
Generate OOS meta-probabilities.

STEP 6: Bet sizing (AFML Ch 10)
For each sample:
  z = (meta_prob - 0.5) / sqrt(meta_prob * (1 - meta_prob))
  size = 2 * Φ(z) - 1
  signal = side × size
Discretize with step_size=0.1 (finer granularity now that we have 10x more samples).
Average active signals using t1 for overlapping events.

Build per-stock daily position series.

STEP 7: Backtesting (AFML Ch 14)
For EACH stock:
  - Compute daily strategy returns using positions.shift(1) × price_returns
  - Deduct 5 bps transaction costs on turnover
  - Compute: SR, PSR, Max DD, Calmar

For the PORTFOLIO (equal-weight average of per-stock returns):
  - Compute all the same metrics
  - Also compute: DSR with num_trials = tuning_trials + meta_trials

Print the full statistics table:

```
           SR      PSR    DSR    MaxDD    Calmar   HitRate
AAPL       ???     ???    ???    ???      ???      ???
AMZN       ???     ???    ???    ???      ???      ???
NVDA       ???     ???    ???    ???      ???      ???
GOOGL      ???     ???    ???    ???      ???      ???
JNJ        ???     ???    ???    ???      ???      ???
JPM        ???     ???    ???    ???      ???      ???
MSFT       ???     ???    ???    ???      ???      ???
XOM        ???     ???    ???    ???      ???      ???
META       ???     ???    ???    ???      ???      ???
TSLA       ???     ???    ???    ???      ???      ???
────────────────────────────────────────────────────
PORTFOLIO  ???     ???    ???    ???      ???      ???
```

Save: data/processed/backtest_results_pooled.parquet

STEP 8: Comparison table
Print the final comparison:

```
Pipeline Configuration                  CV Acc   SR      PSR     DSR
─────────────────────────────────────── ──────── ─────── ─────── ───────
1. NVDA only, 15 features (baseline)    0.628    ???     ???     ???
2. 10-stock, 17 TS features only        ???      ???     ???     ???
3. 10-stock, 17 TS + N alpha features   ???      ???     ???     ???
```

CRITICAL RULES:
- All OOS predictions via PurgedKFold loop — NEVER use a full-data-fit model
- All scoring uses sample weights
- Backtest positions use .shift(1) — NEVER same-day
- Transaction costs of 5 bps ALWAYS included
- DSR num_trials includes ALL tuning trials + meta-model trials
```

---

### ═══════════════════════════════════════════════
### PROMPT 7 OF 7 — FULL VALIDATION & FINAL REPORT
### ═══════════════════════════════════════════════

```
You are a senior quantitative researcher performing the final validation of the 10-stock AFML + 101 Alphas pipeline. Run every check, fix every error, and produce the final report.

STEP 1: Source code audit
Read every file in src/ and src/alphas/. For each function, verify:
- No in-place mutation of input data
- No look-ahead bias (all rolling operations use past data only)
- min_periods = window size in all rolling calls
- rank_cs() uses axis=1 (across stocks, not across time)
- adv() uses close * volume (dollar volume), not volume alone
- No inf values escape into the modelling dataset

STEP 2: Data integrity audit
Load every parquet file. Verify:
- panel_ohlcv: ~32000 rows, 10 tickers, no NaN
- panel_alpha_features_pruned: correct shape, no inf
- pooled_modelling: 1200-2000 rows, no NaN, labels in {-1,+1}
- All per-stock label files: correct format, t1 within dataset bounds

STEP 3: Leakage audit
For the MultiAssetPurgedKFold:
- Verify: at every test date, ALL stocks at that date are in the test set (no cross-sectional leakage)
- Verify: no train sample's t1 extends into the test period (purging works)
- Verify: embargo removes samples after test period (embargo works)
- Verify: OOS predictions have no train-test overlap

For the alpha features:
- Verify: every alpha uses only past data (rolling windows look backward)
- Verify: no alpha uses future returns, future prices, or future volumes
- Verify: rank_cs() at date t only uses date-t data (not future dates)

For the backtest:
- Verify: positions.shift(1) is used (not same-day positions)
- Verify: transaction costs are subtracted

STEP 4: AFML fidelity audit
Verify each AFML concept is correctly implemented:
- CUSUM filter (Snippet 2.4): symmetric, resets after trigger
- Triple barrier (Snippet 3.2): path-dependent, walks forward bar by bar
- Meta-labeling (Snippet 3.7): ret *= side, bin = 0 if ret <= 0
- Concurrency (Snippet 4.1): count >= 1 everywhere
- Uniqueness (Snippet 4.2): values in (0, 1]
- FFD fracdiff (Snippet 5.3): w_0=1 multiplies newest observation
- PurgedKFold (Snippet 7.3): contiguous test blocks, purging, embargo
- Weighted scoring (Snippet 7.4): weights in both fit() and score()
- MDI/MDA/SFI (Snippets 8.2-8.4): correct methodology
- Bet sizing (Snippet 10.1): z = (p-0.5)/sqrt(p*(1-p)), signal = side × (2Φ(z)-1)
- PSR (Ch 14.7.2): correct formula with skew and kurtosis
- DSR (Ch 14.7.3): SR* from expected max under null, correct num_trials

STEP 5: Alpha engine audit
For 5 randomly selected alphas, manually compute the value for NVDA on a specific date using the raw OHLCV data, and compare against the engine's output. They must match within floating-point tolerance (1e-6).

STEP 6: Sequential bootstrap verification
Run the Snippet 4.8 Monte Carlo test:
- 500 trials comparing sequential vs standard bootstrap uniqueness
- Sequential MUST be > standard
- If not, the sequential bootstrap implementation is buggy — FIX IT

STEP 7: Generate final report

Print the complete validation report:

```
══════════════════════════════════════════════════════════
         AFML + 101 ALPHAS PIPELINE VALIDATION REPORT
══════════════════════════════════════════════════════════

  Universe:           10 stocks (AAPL,AMZN,NVDA,...,TSLA)
  Date range:         YYYY-MM-DD → YYYY-MM-DD
  Total trading days: N
  Pooled samples:     N
  Feature count:      N (17 TS + N alphas)
  
  ── ALPHA ENGINE ──────────────────────────────────────
  Alphas computed:    N / 101
  Alphas excluded:    N (NaN/constant/redundant)
  Alphas selected:    N
  Top alpha (MDA):    alphaXXX
  
  ── CROSS-VALIDATION ──────────────────────────────────
  CV accuracy:        X.XXXX ± X.XXXX
  Majority baseline:  X.XXXX
  Beats baseline:     YES/NO
  
  ── META-LABELING ─────────────────────────────────────
  OOS primary acc:    X.XXXX
  Meta-model F1:      X.XXXX ± X.XXXX
  
  ── BACKTEST (PORTFOLIO) ──────────────────────────────
  Sharpe Ratio:       X.XXXX
  PSR (SR*=0):        X.XXXX
  DSR (N=XX):         X.XXXX
  Max Drawdown:       XX.XX%
  Calmar Ratio:       X.XXXX
  Hit Ratio:          XX.XX%
  
  ── LEAKAGE CHECKS ────────────────────────────────────
  Cross-sectional:    ✓ / ✗
  Temporal (purging): ✓ / ✗
  Embargo:            ✓ / ✗
  Position lag:       ✓ / ✗
  Cost deduction:     ✓ / ✗
  
  ── AFML FIDELITY ─────────────────────────────────────
  Score: NN/NN checks passed
  
  ── COMPARISON ────────────────────────────────────────
  Config              CV Acc   Portfolio SR   DSR
  NVDA baseline       X.XXX   X.XXX         X.XXX
  10-stock baseline   X.XXX   X.XXX         X.XXX
  10-stock + alphas   X.XXX   X.XXX         X.XXX
  
  ── VERDICT ───────────────────────────────────────────
  [VALID/INVALID] — [summary statement]
══════════════════════════════════════════════════════════
```

STEP 8: Save final artifacts
- Git commit all code, data, and results
- Ensure all notebooks are runnable and have executed outputs
- Ensure all figures are saved to reports/figures/

If ANY check fails: fix the issue, rerun affected stages, and rerun this validation prompt until zero errors remain.
```

---

## END OF IMPLEMENTATION PLAN

### Quick Reference: Prompt Execution Order

```
Prompt 1 → Download data, build panel           (depends on: nothing)
Prompt 2 → Per-stock AFML pipeline               (depends on: Prompt 1)
Prompt 3 → Alpha engine build                    (depends on: Prompt 1)
Prompt 4 → Alpha diagnostics & pruning           (depends on: Prompt 3)
Prompt 5 → Pooled dataset & multi-asset CV       (depends on: Prompts 2 + 4)
Prompt 6 → Training, meta-labeling, backtest     (depends on: Prompt 5)
Prompt 7 → Full validation                       (depends on: Prompt 6)
```

Prompts 2 and 3 can run in parallel.
All other prompts are sequential.
Total estimated time: 8-12 hours of Claude Code execution.
