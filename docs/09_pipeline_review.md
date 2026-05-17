# AFML Pipeline Review: Results, Concerns, Fixes & Improvements

**Project**: 10-Stock Universe + 101 Formulaic Alphas  
**Branch**: `Complete-AFML-Pipeline`  
**Date**: 2026-05-16  
**Stocks**: AAPL, AMZN, NVDA, GOOGL, JNJ, JPM, MSFT, XOM, META, TSLA  
**Date Range**: 2012-05-18 to 2025-04-30

---

## 1. Current Results

### 1.1 Dataset


| Metric                    | Value                                   |
| ------------------------- | --------------------------------------- |
| Panel OHLCV rows          | 32,560 (10 stocks × 3,256 trading days) |
| Panel NaN count           | 0                                       |
| Total pooled events       | 881                                     |
| Label distribution        | +1 = 478 (54.3%), −1 = 403 (45.7%)      |
| Total features            | 50 (17 time-series + 33 alpha)          |
| Alpha features (raw)      | 101                                     |
| Alpha features (selected) | 33                                      |


### 1.2 Cross-Validation (5-fold MultiAssetPurgedKFold)


| Model                                 | CV Accuracy | Std Dev |
| ------------------------------------- | ----------- | ------- |
| Majority baseline                     | 0.5426      | —       |
| TS-only (17 features)                 | 0.5504      | ±0.049  |
| Full 50 features (17 TS + 33 alpha)   | 0.5438      | ±0.023  |
| Tuned RF (depth=7, min_leaf=30, sqrt) | 0.5551      | —       |


### 1.3 Meta-Labeling


| Model                                  | CV Accuracy |
| -------------------------------------- | ----------- |
| Primary model (TS-only → direction)    | 0.5473      |
| Secondary model (all 50 → correctness) | 0.5357      |
| Average absolute bet size              | 0.5677      |


### 1.4 Feature Importance (Top 10 by MDI)


| Rank | Feature              | MDI Score | Type  |
| ---- | -------------------- | --------- | ----- |
| 1    | alpha041             | 0.0646    | Alpha |
| 2    | alpha012             | 0.0567    | Alpha |
| 3    | alpha028             | 0.0565    | Alpha |
| 4    | alpha009             | 0.0414    | Alpha |
| 5    | ret_10d              | 0.0353    | TS    |
| 6    | ret_20d              | 0.0342    | TS    |
| 7    | amihud_illiquidity   | 0.0337    | TS    |
| 8    | bekker_parkinson_vol | 0.0326    | TS    |
| 9    | alpha022             | 0.0300    | Alpha |
| 10   | shannon_entropy      | 0.0295    | TS    |


Alpha features occupy 6 of the top 10 positions. The highest-ranked pure TS feature (ret_10d) appears at rank 5.

### 1.5 Per-Stock Backtest Results (OOF Meta-Labeled Signals)


| Ticker        | Trades  | Total Return | Sharpe    | Hit Rate  | Max Drawdown | Calmar    |
| ------------- | ------- | ------------ | --------- | --------- | ------------ | --------- |
| AAPL          | 50      | +0.3686      | 1.664     | 62.0%     | −0.143       | 1.288     |
| AMZN          | 35      | −0.2236      | −1.128    | 45.7%     | −0.457       | −0.350    |
| NVDA          | 166     | +1.2262      | 2.544     | 59.6%     | −0.393       | 0.470     |
| GOOGL         | 74      | +0.2623      | 1.393     | 55.4%     | −0.154       | 0.575     |
| JNJ           | 81      | +0.0969      | 0.550     | 59.3%     | −0.193       | 0.155     |
| JPM           | 50      | +0.4031      | 1.461     | 64.0%     | −0.165       | 1.223     |
| MSFT          | 90      | +0.1867      | 0.709     | 51.1%     | −0.250       | 0.208     |
| XOM           | 137     | +0.3248      | 1.040     | 55.5%     | −0.265       | 0.224     |
| META          | 107     | −0.3871      | −1.036    | 44.9%     | −0.646       | −0.140    |
| TSLA          | 91      | +0.0374      | 0.094     | 46.2%     | −0.534       | 0.019     |
| **Portfolio** | **881** | **+2.2953**  | **2.377** | **54.4%** | **−0.813**   | **0.080** |


### 1.6 Validation Audit Summary


| Status | Count |
| ------ | ----- |
| PASS   | 31    |
| WARN   | 1     |
| FAIL   | 0     |


---

## 2. Concerns

### 2.1 Sample Weight Scaling (WARN)

**Severity**: Medium  
Sample weights produced by the pipeline range from **0.022 to 4.506**. AFML uniqueness-scaled weights are conceptually bounded in (0, 1] — they represent the average overlap-adjusted label uniqueness multiplied by a time-decay factor. Values exceeding 1.0 suggest a potential scaling error in the weight computation, likely in how uniqueness and time-decay are combined or normalised. This could cause the model to over-emphasise certain events during training and distort the weighted accuracy metric used in cross-validation.

### 2.2 Weak Predictive Signal

**Severity**: High  
The full 50-feature model achieves a cross-validated accuracy of **0.5438** against a majority baseline of **0.5426** — a margin of just **0.12 percentage points**. This is statistically indistinguishable from noise for a dataset of 881 events. Even the TS-only model's margin of 0.78 percentage points is very small. The 101 Formulaic Alphas do not appear to be providing meaningful directional lift in this configuration, though they do reduce CV variance (std drops from ±0.049 to ±0.023).

### 2.3 Small Per-Stock Event Counts

**Severity**: High  
The CUSUM filter, combined with the common date range restriction, produced very few events for several stocks:


| Ticker | Events | 5-Fold Test Size |
| ------ | ------ | ---------------- |
| AMZN   | 35     | ~7 per fold      |
| AAPL   | 50     | ~10 per fold     |
| JPM    | 50     | ~10 per fold     |
| GOOGL  | 74     | ~15 per fold     |


With 7–10 test samples per fold, cross-validation estimates are unreliable. No per-stock individual CV, feature importance, or hyperparameter tuning was performed — all modelling was done on the pooled dataset only. The backtest statistics for low-event stocks cannot be trusted.

### 2.4 Meta-Model Adds Marginal Value

**Severity**: Medium  
The secondary (meta-labeling) model, which predicts whether the primary direction model is correct, achieves only **0.5357 accuracy** — 3.57 percentage points above random guessing. The resulting bet-sizing signal is weak, with an average absolute bet size of 0.568, meaning positions are rarely sized confidently. The meta-labeling architecture is correct in design but is limited by the underlying signal quality.

### 2.5 Two Stocks Show Persistent Negative Performance

**Severity**: Medium  
META (Sharpe −1.036, total return −0.387) and AMZN (Sharpe −1.128, total return −0.224) show consistent losses across OOF folds. This may reflect:

- Insufficient events (AMZN: 35) making the model unreliable
- Structural differences in these stocks (META high-volatility post-2022 regime change; AMZN low event count)
- The pooled model not capturing ticker-specific dynamics

### 2.6 No Per-Stock Modelling

**Severity**: Medium  
All cross-validation, feature importance, hyperparameter tuning, and meta-labeling were performed on the **pooled dataset only**. No individual stock received its own model, importance ranking, or tuned parameters. The per-stock backtest results are derived by filtering the single pooled model's OOF predictions — they do not reflect whether the model generalises per stock.

### 2.7 TSLA and META Fracdiff Edge Case (Resolved)

**Severity**: Low (fixed)  
META (IPO May 2012, ~3,256 rows) and TSLA (IPO June 2010, ~3,733 rows) have shorter price histories than the other eight stocks. For low values of d (0.10, 0.20), the FFD convolution window length exceeds the available series length. A bug in `frac_diff_ffd` caused it to return a full-length NaN series in this case rather than an empty series, which bypassed the downstream length guard and crashed the ADF stationarity test. This was fixed and both stocks now proceed through the pipeline normally, with their d* selected from the subset of d values where the window fits.

---

## 3. Fixes Applied

### Fix 1: Fractional Differentiation Empty-Series Bug

**File**: `src/fracdiff.py`, line 86  
**Affected stocks**: META, TSLA  
**Root cause**: When the FFD window width exceeded the series length, the function returned a NaN-filled Series with the original index (length ~3,256). The downstream guard `if len(s) < 10: continue` saw length 3,256 and passed the all-NaN array to ADF, causing a crash.

```python
# Before (bug):
if len(s) < width:
    return pd.Series(index=series.index, dtype=float)

# After (fix):
if len(s) < width:
    return pd.Series(dtype=float)  # empty — length guard fires correctly
```

**Result**: The d sweep skips values where the window does not fit, and find_min_d selects a higher d* for shorter-history stocks. Both META and TSLA process correctly with no special-casing.

---

### Fix 2: Validation Check for rank_cs

**File**: `scripts/prompt7_validation_report.py`  
**Issue**: The test assertion expected pandas `pct=True` ranks to return 0.0 for the minimum value, but pandas percentile ranks range from 1/n to 1.0 (not 0.0 to 1.0). The function was correct; the test expectation was wrong.  
**Fix**: Changed the check to verify ordinal correctness (lower-valued ticker receives a lower rank than higher-valued ticker) rather than asserting an exact value.

---

### Fix 3: Validation Check for Invalid Bins

**File**: `scripts/prompt7_validation_report.py`  
**Issue**: NVDA had one row on 2025-04-25 with a NaN bin — an open event at the data end date that never hit a barrier before the price history ended. The check flagged this as an invalid bin.  
**Fix**: Updated the check to exclude NaN bins (which represent legitimate open events filtered out during pooled dataset construction by the `label.isin([-1.0, 1.0])` filter).

---

## 4. Recommended Improvements

### Improvement 1: Fix Sample Weight Scaling

**Priority**: High  
Investigate and correct the weight computation in `src/pipeline/per_stock.py` so that all sample weights fall within (0, 1]. Verify the uniqueness calculation and time-decay combination against AFML Chapter 4 equations. Re-run CV after the fix as weighted accuracy scores will change.

### Improvement 2: Increase Events Per Stock

**Priority**: High  
The CUSUM filter threshold `h` is currently calibrated to target ~300 events over the full history. After restricting to the common date range (2012–2025), many stocks drop to 35–100 events. Options:

- Reduce the CUSUM `h` multiplier to generate more frequent events
- Widen the triple-barrier vertical barrier from 10 days to 15 or 20 days
- Remove the common date range restriction and handle per-stock date ranges in the CV scheme

### Improvement 3: Per-Stock Individual Modelling

**Priority**: Medium  
Run separate PurgedKFold CV, feature importance, and hyperparameter tuning for each stock with sufficient events (suggested threshold: ≥ 100 events). This would reveal which features matter per stock and whether the pooled model's poor performance on META and AMZN is a feature or a data issue.

### Improvement 4: Investigate META and AMZN Underperformance

**Priority**: Medium  
META's negative Sharpe (−1.036) may be driven by the post-2022 regime change (Meta rebranding, major drawdown). AMZN's poor results (35 events) are likely a small-sample artefact. Consider:

- Running META-specific feature importance to identify what drives misprediction
- Extending AMZN's history by relaxing the common start date for TS-only features
- Adding regime indicators (e.g. a rolling volatility state) as features

### Improvement 5: Expand the Stock Universe

**Priority**: Medium  
With 881 events across 10 stocks (average 88 per stock), the pooled dataset is small for a 50-feature Random Forest. Expanding to 20–30 stocks would increase event count, improve CV stability, and make the alpha cross-sectional operators (rank, indneutralize) more statistically meaningful with more tickers per sector.

### Improvement 6: Alpha Signal Quality Assessment

**Priority**: Medium  
The 33 selected alphas improved CV variance but not mean accuracy. A more targeted selection approach would:

- Compute per-alpha information coefficient (IC = rank correlation with forward return) independently of the RF model
- Select alphas by IC > 0.02 rather than purely by ADF stationarity
- Test alpha decay (IC over time) to filter out alphas that are no longer live

### Improvement 7: Walk-Forward Validation

**Priority**: Medium  
The current OOF backtest uses all 5 folds including early folds where little data is available. A strict walk-forward validation (train on first N years, test on year N+1, expand window) would give a more realistic picture of out-of-sample performance and avoid the appearance of fold 0 (only 137 test events) inflating or deflating aggregate statistics.

### Improvement 8: Transaction Cost Modelling

**Priority**: Low  
The backtest computes raw P&L with no transaction costs. Adding a simple cost model (e.g. 5–10 bps per trade) would significantly impact the results given the short holding periods (10-day vertical barrier) and provide a more realistic performance estimate.

---

## 5. Validation Audit Detail


| Section          | Check                                              | Status                 |
| ---------------- | -------------------------------------------------- | ---------------------- |
| Source Code      | fracdiff w_0 == 1.0                                | PASS                   |
| Source Code      | fracdiff weights monotone                          | PASS                   |
| Source Code      | fracdiff returns empty series when window > length | PASS                   |
| Source Code      | MultiAssetPurgedKFold importable                   | PASS                   |
| Source Code      | rank_cs is cross-sectional                         | PASS                   |
| Source Code      | adv(d) = sma(close × volume, d)                    | PASS                   |
| Data             | Panel OHLCV has all 10 tickers                     | PASS                   |
| Data             | Panel OHLCV zero NaN                               | PASS                   |
| Data             | Panel OHLCV has 6 columns                          | PASS                   |
| Data             | Pruned alpha panel has 33 features                 | PASS                   |
| Data             | No pruned alpha has > 40% NaN                      | PASS                   |
| Data             | All labels have t1 > t0                            | PASS                   |
| Data             | All non-NaN bins are −1 or +1                      | PASS                   |
| Data             | Pooled dataset zero NaN                            | PASS                   |
| Data             | Pooled label balance < 25% imbalance               | PASS                   |
| Data             | No t1 < t0 in pooled dataset                       | PASS                   |
| Data             | cv_baseline_multistock.parquet exists              | PASS                   |
| Data             | meta_labeled_predictions.parquet exists            | PASS                   |
| AFML Fidelity    | Sample weights sum > 0                             | PASS                   |
| AFML Fidelity    | Sample weights in (0, 1]                           | **WARN** (max = 4.506) |
| AFML Fidelity    | fracdiff feature present for all tickers           | PASS                   |
| AFML Fidelity    | No train/test date overlap in any CV fold          | PASS                   |
| AFML Fidelity    | Purging removes all overlapping train samples      | PASS                   |
| AFML Fidelity    | Baseline CV accuracy > majority class              | PASS                   |
| AFML Fidelity    | Meta predictions have required columns             | PASS                   |
| AFML Fidelity    | Meta probabilities in [0, 1]                       | PASS                   |
| Alpha Spot-Check | alpha001 in [−0.5, 0.5]                            | PASS                   |
| Alpha Spot-Check | alpha012 vs manual formula (corr = 1.000)          | PASS                   |
| Alpha Spot-Check | alpha028 has values                                | PASS                   |
| Alpha Spot-Check | alpha041 vs manual formula (corr = 0.998)          | PASS                   |
| Alpha Spot-Check | alpha002 bounded in [−1, 1]                        | PASS                   |
| Alpha Spot-Check | alpha056 is all-NaN (expected)                     | PASS                   |


---

*Generated from pipeline run on 2026-05-16. Branch: `Complete-AFML-Pipeline`, commit: b763544.*