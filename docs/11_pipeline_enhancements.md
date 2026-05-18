# Pipeline Enhancement Roadmap

Compiled from the zero-tolerance audit (2026-05-18) that diagnosed classifier collapse
toward +1 predictions (82.5% pred=+1, Balanced Accuracy 0.483, MCC –0.045, AUC 0.485).

---

## Part 1 — Ten Enhancements (Priority Order)

### 1. Expand the Stock Universe

**What**: Grow from 10 stocks to 50–200 stocks across multiple sectors (e.g., S&P 500
constituents grouped by GICS sector).

**Why it helps**:
- Cross-sectional rank signals (`rank_cs`, `scale_cs`, `indneutralize_cs`) require broad
  dispersion to be meaningful. With 10 nearly-correlated large-caps, ranking produces
  near-constant outputs — the root cause of `alpha007` outputting –1 for 99.76% of
  samples and `alpha027` being 96.6% binary.
- More stocks = more labeled events per fold = better model generalization.
- `MultiAssetPurgedKFold` already handles pooled multi-stock data; no CV redesign needed.

**Effort**: Medium — extend the ticker list in `stage01_data_pipeline.py` and rerun
from stage01 onward. All downstream stages inherit automatically.

**Expected gain**: Revives most degenerate alphas; cross-ticker std of alpha signals
(currently 0.005–0.051) should widen substantially.

---

### 2. Minimum Return Filter on Labels

**What**: Before training, drop labeled events where `|ret| < threshold` (suggested: 0.5%).

**Why it helps**:
- The current pooled dataset has 98 events (4.7%) with `|ret| < 1%` and 44 (2.1%) with
  `|ret| < 0.5%`. These near-zero-return events are economically meaningless (transaction
  costs erase any edge) and statistically noisy — their labels are essentially coin flips
  that degrade the feature-label relationship.
- Removing them sharpens the signal without reducing event count significantly.

**Implementation**:
```python
# In stage02 / feature pipeline, after labeling:
MIN_RET = 0.005
labels = labels[labels['ret'].abs() >= MIN_RET]
```

**Effort**: Low — one filter line; all downstream datasets are smaller but cleaner.

**Expected gain**: Improves max feature-label correlation from 0.057 toward 0.07–0.09.

---

### 3. Macro / Regime Features

**What**: Add market-environment features that are currently absent from the 50-feature
matrix.

**Why it helps**: The model has zero visibility into whether the market is in a risk-on
or risk-off regime, despite this being one of the strongest predictors of whether a
triple-barrier trade will be profitable. Every current feature is stock-specific.

**Suggested features**:

| Feature | Proxy ticker / calculation | Signal |
|---|---|---|
| VIX level | `^VIX` daily close | Volatility regime |
| VIX 5-day change | `VIX.pct_change(5)` | Regime transition |
| Yield curve slope | 10Y – 2Y treasury spread | Risk-on / risk-off |
| SPY 5d / 20d return | `SPY` log returns | Market momentum context |
| HYG / LQD spread | High-yield minus IG | Credit risk appetite |
| Dollar index (DXY) | `DX-Y.NYB` | FX / sector rotation |

All six are freely available from yfinance and merge to the feature matrix by date.

**Effort**: Medium — add a `compute_macro_features(date_index)` function in
`src/features.py` and merge in `stage02`.

---

### 4. Calibrated Classification Threshold (Per Fold)

**What**: Instead of a fixed threshold of 0.50, find the optimal threshold on each
fold's validation set and apply it to that fold's test predictions.

**Why it helps**: With compressed probability distributions (all OOS probabilities in
[0.32, 0.84], mean 0.559), a flat 0.50 threshold produces 82.5% positive predictions.
Calibrating per fold ensures the decision boundary matches the actual probability
distribution without leaking test-set information.

**Implementation** (inside the OOS loop in `stage08_final_modelling.py`):
```python
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def find_threshold(y_val, prob_val, grid=np.arange(0.30, 0.70, 0.01)):
    best_t, best_ba = 0.5, 0.0
    for t in grid:
        yp = np.where(prob_val >= t, 1, -1)
        ba = balanced_accuracy_score(y_val, yp)
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t
```

Apply to the held-out test fold using the threshold found on the train fold's OOS
probabilities (not the test fold — no leakage).

**Effort**: Low — a dozen lines of code in the OOS loop.

---

### 5. Gradient Boosting + Ensemble

**What**: Run a full OOS prediction loop for XGBoost (currently only RF gets the OOS
loop in stage08; XGB is only HP-tuned). Ensemble final probabilities:
`final_prob = 0.5 * rf_prob + 0.5 * xgb_prob`.

**Why it helps**:
- XGBoost/LightGBM sequentially corrects residuals rather than averaging independent
  trees, making it better at extracting weak signals from high-dimensional noisy data.
- Ensembling reduces variance without sacrificing the individual model's contribution.

**Effort**: Low-medium — duplicate the RF OOS loop in stage08 for XGB, then average
probabilities before thresholding.

---

### 6. Longer Data History

**What**: Extend the historical window from the current range (approx. 2014–2024) to
the full available history (2000–2024 for most large-caps).

**Why it helps**:
- Current pooled dataset: 2071 events across 10 stocks (~207 per stock). For a 50-feature
  RF with 5-fold purged CV, each training fold has ~160 events — borderline underfitting.
- Doubling the history roughly doubles labeled events and gives the model exposure to
  multiple market regimes (dot-com recovery, GFC, COVID shock, rate hike cycles).

**Effort**: Low — change `start_date` in stage01; everything downstream reruns.

**Caveat**: Some alpha operators (e.g., `adv180`) require 180 days of warm-up, so the
first ~9 months of data will produce NaN alphas regardless of start date.

---

### 7. Probability Calibration (Platt / Isotonic)

**What**: Wrap the RF/XGB model in `sklearn.calibration.CalibratedClassifierCV` with
`method='isotonic'` inside each CV fold.

**Why it helps**:
- Random Forest probabilities are systematically biased toward 0.5 (a well-documented
  RF artifact from majority-vote averaging). Isotonic regression re-maps the raw
  probabilities to match the empirical class frequencies.
- This makes probabilities meaningful for bet sizing (the Kelly criterion and AFML
  Chapter 10 fractional Kelly both require calibrated probabilities to be valid).

**Effort**: Low — one wrapper class, fitted on the training fold only.

---

### 8. PCA on Alpha Block

**What**: Replace the 33 raw alpha columns with the top K principal components
explaining 95% of variance. Fit PCA inside each CV fold to prevent leakage.

**Why it helps**:
- Several alphas are near-constant (`alpha007`, `alpha021`, `alpha027`) or highly
  correlated with each other (cross-sectional rank signals from similar inputs tend to
  co-move). PCA eliminates redundant variance and removes the degenerate components
  automatically.
- Reduces the feature matrix from 50 to roughly 25–30 columns, lowering the risk of
  the model learning alpha-specific noise.

**Implementation**:
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Inside each fold:
pca = PCA(n_components=0.95, random_state=42)
alpha_pcs_train = pca.fit_transform(X_tr[alpha_cols].fillna(0))
alpha_pcs_test  = pca.transform(X_te[alpha_cols].fillna(0))
```

**Effort**: Medium — requires refactoring the feature matrix construction inside the
OOS loop.

---

### 9. Sector Neutralization of Alpha Signals

**What**: After computing each alpha's cross-sectional value, subtract the sector mean
at each date (true `indneutralize_cs`).

**Why it helps**:
- The current `indneutralize_cs` in `src/alphas/operators.py` requires a populated
  `sector_map` dict. In the current pipeline, `sector_map` is sparse or empty, so
  neutralization is effectively skipped.
- Sector neutralization removes the market-wide and sector-wide factor from each alpha,
  leaving only idiosyncratic stock signal. This is standard practice in quant equity
  (long-short books) and materially improves alpha IC (information coefficient).

**Effort**: Medium — populate `sector_map` from a GICS sector file (one-off) and
ensure `indneutralize_cs` is called correctly in each alpha that references it.

---

### 10. Asymmetric Barriers

**What**: Instead of symmetric profit-take / stop-loss multipliers, use PT:SL = 2:1
(or fit the ratio from historical data per stock).

**Why it helps**:
- The triple-barrier label `bin = sign(ret at first barrier touch)` conflates two
  different outcomes: a large +1 from a 2× PT touch and a +1 from a tiny 0.01 move
  before the time barrier. With asymmetric barriers, +1 labels are associated with
  larger return magnitudes, making the signal easier to detect.
- AFML Section 3.4 discusses this: the ratio of PT to SL determines the class balance
  and the economic significance of each label.

**Effort**: Low — change the `pt_sl` argument in the triple-barrier labeler call in
`stage02`.

---

## Part 2 — 5-Minute OHLCV Data: Trade-offs

### Where 5-min Data Improves the Pipeline

#### Barrier Detection Precision (High Impact)
With daily bars, barriers are checked once per day at the close. A stock can touch the
profit-take and stop-loss intrabar; the daily close obscures which fired first and what
the actual exit price was. With 5-min bars, `t1` and `ret` reflect the true barrier touch
time. This is the single largest quality improvement from intraday data.

#### Microstructure Features (High Impact)
Corwin-Schultz, Amihud illiquidity, Roll spread, and Bekker-Parkinson were all derived
for high-frequency data. On daily bars they are approximations that violate several of
the estimators' underlying assumptions. On 5-min bars they compute what the papers
actually specify.

| Feature | Daily accuracy | 5-min accuracy |
|---|---|---|
| Corwin-Schultz spread | Approximate (violates no-arb ~30% of days) | Good |
| Amihud illiquidity | Raw values ~1e-11 (scale issue) | Proper scale |
| Roll spread | Noisy (20-day window = 20 obs) | Rich (20-day = ~1560 obs) |
| VWAP | Proxy: (H+L+C)/3 | True volume-weighted price |

#### Entropy / Complexity Features (Medium Impact)
Shannon entropy and Lempel-Ziv complexity computed over 50 daily returns = 50 points.
Over 5-min bars, 50 bars = ~4 trading hours. Intraday sequences carry richer
microstructure regime information.

---

### Where 5-min Data Does NOT Improve the Pipeline

#### The 101 WorldQuant Alphas (No Improvement, Risk of Degradation)
The 101 alphas are daily-resolution signals by design. `delta(close, 7)` means seven
trading days; `adv20` means 20-day average dollar volume. Feeding 5-min bars changes
their semantics entirely (`delta(close, 7)` becomes "change over 35 minutes"). These
alphas must continue to be computed on daily bars regardless of the intraday upgrade.

#### Multi-Day Momentum and Volatility Features (No Improvement)
`ret_5d`, `ret_20d`, `RSI-14`, `vol_20d`, `vol_50d` — these measure dynamics at the
holding-period scale (days to weeks). Recomputing them on 5-min bars adds no signal
for a model whose labels span 5–20 days.

---

### The Binding Constraint: Data Availability

Free sources provide very limited intraday history:

| Source | 5-min history depth | Free? |
|---|---|---|
| yfinance | Last 60 days only | Yes |
| Alpaca Markets | 5+ years (US equities) | Free tier |
| Polygon.io | 2+ years | Paid ($29+/mo) |
| IEX Cloud | Limited depth | Paid |

Without 5+ years of intraday history the pipeline cannot be trained — the CUSUM
filter plus rolling windows consume the first several months of each stock's data,
and the model needs multiple years of events to train a meaningful classifier.

---

### AFML's Actual Recommendation

Prado argues that **time bars at any frequency are the wrong abstraction**, because
equal-time intervals do not correspond to equal-information intervals. During the
pre-market hours a 5-min bar contains near-zero information; at 9:35am it may contain
the most information of the day. The book advocates:

- **Tick bars**: one bar per N transactions
- **Volume bars**: one bar per N shares traded
- **Dollar bars**: one bar per $N of dollar volume

These normalize information content per bar, produce return distributions closer to
Gaussian, and reduce intrabar autocorrelation. A 5-min bar is better than a daily bar
but still a time bar — the underlying issue is not fixed.

---

### Recommended Approach: Multi-Resolution Pipeline

Rather than fully replacing daily bars with 5-min bars, use both layers:

| Pipeline layer | Resolution | Rationale |
|---|---|---|
| Event detection (CUSUM filter) | Daily | Adequate; avoids noise |
| Barrier touching (find `t1`, `ret`) | 5-min | Highest-ROI intraday upgrade |
| 101 alpha features | Daily | Designed for daily resolution |
| Momentum / volatility features | Daily | Holding period is multi-day |
| Microstructure features | 5-min (if available) | Designed for HF data |
| True VWAP, realized vol | 5-min (if available) | Accurate computation |
| Model training / CV | Daily event index | No change to CV structure |

This hybrid approach captures the precision benefit of intraday barrier detection and
the accuracy of intraday microstructure estimation, while preserving the daily alpha
signals and avoiding the data-availability constraint for features that don't benefit
from higher frequency.

---

## Summary Table

| Enhancement | Impact | Effort | Dependencies |
|---|---|---|---|
| 1. Expand universe (50–200 stocks) | High | Medium | stage01 rerun |
| 2. Minimum return filter (|ret| ≥ 0.5%) | High | Low | stage02 rerun |
| 3. Macro / regime features | High | Medium | New data fetch |
| 4. Per-fold threshold calibration | Medium | Low | stage08 only |
| 5. XGB OOS loop + ensemble | Medium | Low-Medium | stage08 only |
| 6. Longer history (2000–present) | Medium | Low | stage01 rerun |
| 7. Probability calibration (isotonic) | Medium | Low | stage08 only |
| 8. PCA on alpha block | Medium | Medium | stage08 refactor |
| 9. Sector neutralization | Medium | Medium | sector_map file |
| 10. Asymmetric barriers (PT:SL = 2:1) | Low-Medium | Low | stage02 rerun |
| 5-min barrier detection | High | High | Intraday data source |
| 5-min microstructure features | Medium | High | Intraday data source |
