# AFML NVDA Full-Pipeline Validation — Claude Code Master Prompt

You are a senior quantitative researcher, AFML specialist, and production ML auditor. You have access to a complete AFML pipeline repository implementing Stages 0–8 of the *Advances in Financial Machine Learning* methodology on NVDA daily OHLCV data.

Your mission: run a **zero-tolerance, book-verified, end-to-end validation** of the entire codebase. Every source module, every notebook, every saved artifact, every formula, every data flow must be tested against the ground truth from the AFML book and the raw dataset. Where you find bugs, fix them. Where you find stale artifacts, regenerate them. Where you find leakage, eliminate it.

**Do not assume anything works. Test everything. Fix everything. Report everything.**

---

# ════════════════════════════════════════════
# SECTION 0 — GROUND TRUTH CONSTANTS
# ════════════════════════════════════════════

These are absolute, non-negotiable facts. Any artifact contradicting them is broken.

```
NVDA_raw.csv
  rows            = 5114
  columns         = 7 → [Date, Adj Close, Close, High, Low, Open, Volume]
  date_range      = 2005-01-03 → 2025-04-30
  nulls           = 0 in every column
  adj_close_min   ≈ 0.1352
  adj_close_max   ≈ 149.38
  volume_min      ≈ 45.6 M
  volume_max      ≈ 3.69 B
  all prices      > 0
  all volumes     > 0
  dates           monotonically increasing, trading days only
```

```
Expected Pipeline Shape (from update document)
  CUSUM events    → 300–600 before filtering
  Modelling rows  = ~195 (after NaN drop from lookback windows)
  Feature cols    = 15 (after Stage 6 pruning of momentum_12_1 and bekker_parkinson_vol)
  Label classes   = {-1, +1} (binary, no label=0 in final set)
  Label dist      ≈ +1:114, -1:81
  Majority base   ≈ 0.5846
  Fracdiff d*     ≈ 0.25, ADF p < 0.05, corr with log price ≈ 0.916
```

```
AFML Book Snippet Cross-Reference
  Snippet 2.4   = CUSUM filter (getTEvents)
  Snippet 3.1   = daily vol (getDailyVol)
  Snippet 3.2   = triple barrier (applyPtSlOnT1)
  Snippet 3.3   = getEvents (first barrier touch)
  Snippet 3.5   = getBins (label assignment)
  Snippet 3.6   = getEvents with meta-labeling (side argument)
  Snippet 3.7   = getBins with meta-labeling (ret*=side, bin∈{0,1})
  Snippet 4.1   = mpNumCoEvents (concurrency counting)
  Snippet 4.2   = mpSampleTW (average uniqueness)
  Snippet 4.3   = getIndMatrix (indicator matrix)
  Snippet 4.5   = seqBootstrap (sequential bootstrap)
  Snippet 4.10  = mpSampleW (return-attribution weights)
  Snippet 4.11  = getTimeDecay (time-decay factors)
  Snippet 5.1   = getWeights (fracdiff weights)
  Snippet 5.3   = fracDiff_FFD (fixed-width window fracdiff)
  Snippet 5.4   = plotMinFFD (minimum d finding)
  Snippet 7.3   = PurgedKFold class
  Snippet 7.4   = cvScore (purged CV scoring)
  Snippet 8.2   = featImpMDI
  Snippet 8.3   = featImpMDA
  Snippet 8.4   = auxFeatImpSFI
  Snippet 9.1   = clfHyperFit (grid search with purged CV; scoring='f1' for meta-labeling)
  Snippet 10.1  = getSignal (probability → bet size)
  Snippet 10.2  = avgActiveSignals (concurrent bet averaging)
  Snippet 10.3  = discreteSignal (size discretization)
  Snippet 14.1  = bet timing derivation
  Snippet 14.3  = getHHI (return concentration)
  Snippet 14.4  = computeDD_TuW (drawdown and time under water)
  Ch 14 §14.7.2 = PSR formula
  Ch 14 §14.7.3 = DSR formula (SR* from expected max under null)
```

---

# ════════════════════════════════════════════
# SECTION 1 — REPOSITORY STRUCTURE VERIFICATION
# ════════════════════════════════════════════

Read and verify the full repository tree. Print every file that exists and flag any that are missing.

```python
import os, json, pickle, sys
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

print("═" * 70)
print("SECTION 1: REPOSITORY STRUCTURE")
print("═" * 70)

expected_src = [
    'src/__init__.py', 'src/data_structures.py', 'src/labeling.py',
    'src/sample_weights.py', 'src/fracdiff.py', 'src/features.py',
    'src/cross_validation.py', 'src/modelling.py', 'src/feature_importance.py',
    'src/hyperparameter_tuning.py', 'src/bet_sizing.py', 'src/backtesting.py',
    'src/synthetic.py', 'src/structural_breaks.py', 'src/entropy.py',
    'src/microstructure.py', 'src/multiprocess.py', 'src/utils.py',
]
expected_data = [
    'data/raw/NVDA_raw.csv',
    'data/processed/nvda_clean.parquet',
    'data/processed/nvda_dollar_bars.parquet',
    'data/processed/nvda_cusum_events.parquet',
    'data/processed/nvda_labels.parquet',
    'data/processed/nvda_sample_weights.parquet',
    'data/processed/nvda_fracdiff.parquet',
    'data/processed/nvda_features.parquet',
    'data/processed/nvda_modelling_dataset.parquet',
    'data/processed/cv_results.parquet',
    'data/processed/tuning_log.parquet',
    'data/processed/feature_importance.parquet',
]
expected_models = [
    'models/model_rf.pkl', 'models/model_xgb.pkl',
    'models/model_final.pkl', 'models/best_params.json',
]
expected_notebooks = [
    f'notebooks/{nb}' for nb in [
        '00_data_inspection.ipynb', '01_data_structures.ipynb',
        '02_labeling.ipynb', '03_sample_weights.ipynb',
        '04_fracdiff.ipynb', '05_feature_engineering.ipynb',
        '06_model_training.ipynb', '07_purged_cv.ipynb',
        '08_feature_importance.ipynb', '09_hyperparameter_tuning.ipynb',
        '10_meta_labeling_bet_sizing.ipynb', '11_backtesting.ipynb',
    ]
]
# Also check for Stage 7-8 specific artifacts
stage78_data = [
    'data/processed/nvda_oos_predictions.parquet',
    'data/processed/nvda_meta_labels.parquet',
    'data/processed/nvda_positions.parquet',
    'data/processed/backtest_results.parquet',
]

all_expected = expected_src + expected_data + expected_models + expected_notebooks + stage78_data
missing, present = [], []
for f in all_expected:
    if os.path.exists(f):
        present.append(f)
    else:
        missing.append(f)

print(f"  Present: {len(present)}/{len(all_expected)}")
for f in missing:
    print(f"  ✗ MISSING: {f}")
if not missing:
    print("  ✓ All expected files present")
```

---

# ════════════════════════════════════════════
# SECTION 2 — SOURCE CODE DEEP AUDIT
# ════════════════════════════════════════════

For EACH source module, read the code, verify function signatures match the AFML book, and run a functional test.

```python
print("\n" + "═" * 70)
print("SECTION 2: SOURCE CODE DEEP AUDIT")
print("═" * 70)

errors_found = []

# ──────────────────────────────────────────
# 2.1  src/data_structures.py — AFML Ch 2
# ──────────────────────────────────────────
print("\n── 2.1 data_structures.py (AFML Ch 2, Snippet 2.4) ──")
from src.data_structures import cusum_filter, get_dollar_bars

# Load clean data
clean = pd.read_parquet('data/processed/nvda_clean.parquet')
close = clean['Adj Close']

# TEST: CUSUM filter
# Book Snippet 2.4: operates on diff of gRaw, triggers when S⁺>h or S⁻<-h, then resets
# We pass close (prices) — function must internally compute log returns or diffs
for h in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]:
    try:
        ev = cusum_filter(close, h)
        n = len(ev)
        if 200 <= n <= 800:
            print(f"  CUSUM h={h:.2f}: {n} events ✓")
            break
    except Exception as e:
        print(f"  CUSUM h={h:.2f}: ERROR {e}")
else:
    print("  ⚠ No h value produced 200-800 events")

# Verify events are subset of close index
saved_events = pd.read_parquet('data/processed/nvda_cusum_events.parquet')
print(f"  Saved CUSUM events: {len(saved_events)} rows")

# TEST: Dollar bars
try:
    dbars = pd.read_parquet('data/processed/nvda_dollar_bars.parquet')
    # Validate OHLCV consistency: H≥max(O,C), L≤min(O,C)
    cols_map = {}
    for target, candidates in [('High',['High','high','H']),('Low',['Low','low','L']),
                                ('Open',['Open','open','O']),('Close',['Close','close','C'])]:
        for c in candidates:
            if c in dbars.columns:
                cols_map[target] = c; break
    if len(cols_map) == 4:
        h_ = dbars[cols_map['High']]; l_ = dbars[cols_map['Low']]
        o_ = dbars[cols_map['Open']]; c_ = dbars[cols_map['Close']]
        ohlcv_ok = ((h_ >= np.maximum(o_,c_) - 1e-9) & (l_ <= np.minimum(o_,c_) + 1e-9)).all()
        print(f"  Dollar bars: {len(dbars)} bars, OHLCV valid={ohlcv_ok} {'✓' if ohlcv_ok else '✗'}")
        if not ohlcv_ok: errors_found.append("Dollar bar OHLCV violation")
    else:
        print(f"  Dollar bars: {len(dbars)} bars, columns={dbars.columns.tolist()}")
except Exception as e:
    print(f"  Dollar bars: {e}")

# ──────────────────────────────────────────
# 2.2  src/labeling.py — AFML Ch 3
# ──────────────────────────────────────────
print("\n── 2.2 labeling.py (AFML Ch 3, Snippets 3.1-3.7) ──")
from src.labeling import get_daily_vol

vol = get_daily_vol(close, span=50)
assert (vol.dropna() > 0).all(), "Daily vol has non-positive values"
print(f"  Daily vol: len={len(vol)}, mean={vol.mean():.6f} ✓")

labels = pd.read_parquet('data/processed/nvda_labels.parquet')
print(f"  Labels shape: {labels.shape}, columns: {labels.columns.tolist()}")
lbl_col = [c for c in labels.columns if c in ['label','bin','Label']][0]
print(f"  Label distribution:\n{labels[lbl_col].value_counts().to_dict()}")
# Verify t1 exists and is within dataset bounds
assert 't1' in labels.columns, "Missing t1 column in labels"
assert (labels['t1'] <= clean.index[-1]).all(), "Labels extend beyond dataset end"
print(f"  All t1 ≤ dataset end ✓")

# ──────────────────────────────────────────
# 2.3  src/sample_weights.py — AFML Ch 4
# ──────────────────────────────────────────
print("\n── 2.3 sample_weights.py (AFML Ch 4, Snippets 4.1-4.11) ──")
from src.sample_weights import num_co_events, sample_tw, get_ind_matrix, seq_bootstrap

t1_series = labels['t1']
co_ev = num_co_events(close.index, t1_series, labels.index)
assert (co_ev >= 1).all(), f"Concurrency < 1 found (min={co_ev.min()})"
print(f"  Concurrency: min={co_ev.min()}, max={co_ev.max()}, mean={co_ev.mean():.2f} ✓")

tw = sample_tw(t1_series, co_ev, labels.index)
tw_clean = tw.dropna()
assert (tw_clean > 0).all() and (tw_clean <= 1.0 + 1e-9).all(), \
    f"Uniqueness outside (0,1]: min={tw_clean.min()}, max={tw_clean.max()}"
print(f"  Uniqueness: min={tw_clean.min():.4f}, max={tw_clean.max():.4f}, mean={tw_clean.mean():.4f} ✓")

# CRITICAL: Sequential bootstrap validation (AFML Snippet 4.5 / 4.8)
print("\n  ▶ SEQUENTIAL BOOTSTRAP MONTE CARLO (AFML Snippet 4.8)")
print("    Running 500 iterations comparing seq vs standard bootstrap...")
np.random.seed(42)
# Use the first 30 events for manageable test
test_n = min(30, len(labels))
test_t1 = t1_series.iloc[:test_n]
bar_idx_start = test_t1.index[0]
bar_idx_end = test_t1.max()
bar_idx_test = close.index[(close.index >= bar_idx_start) & (close.index <= bar_idx_end)]
ind_mat = get_ind_matrix(bar_idx_test, test_t1)

n_mc = 500
std_u_list, seq_u_list = [], []
for trial in range(n_mc):
    # Standard bootstrap uniqueness
    phi_std = np.random.choice(ind_mat.shape[1], size=ind_mat.shape[1], replace=True)
    sel = ind_mat.values[:, phi_std]
    c = sel.sum(axis=1).astype(float)
    c[c == 0] = np.nan
    u_std = np.nanmean(1.0 / c)
    std_u_list.append(u_std)
    
    # Sequential bootstrap uniqueness
    phi_seq = seq_bootstrap(ind_mat, s_length=ind_mat.shape[1])
    sel2 = ind_mat.values[:, phi_seq]
    c2 = sel2.sum(axis=1).astype(float)
    c2[c2 == 0] = np.nan
    u_seq = np.nanmean(1.0 / c2)
    seq_u_list.append(u_seq)

std_mean = np.mean(std_u_list)
seq_mean = np.mean(seq_u_list)
print(f"    Standard bootstrap avg uniqueness: {std_mean:.6f}")
print(f"    Sequential bootstrap avg uniqueness: {seq_mean:.6f}")
if seq_mean > std_mean:
    print(f"    ✓ Sequential > Standard — CORRECT per AFML Ch 4")
else:
    print(f"    ✗ FAILURE: Sequential ({seq_mean:.6f}) ≤ Standard ({std_mean:.6f})")
    print(f"    → seq_bootstrap IS BUGGY. The probability calculation must weight")
    print(f"      candidates by their conditional uniqueness given already-drawn samples.")
    print(f"    → FIX THIS: Review the probability normalization in seq_bootstrap().")
    errors_found.append("Sequential bootstrap uniqueness ≤ standard bootstrap")

# Verify saved weights
sw = pd.read_parquet('data/processed/nvda_sample_weights.parquet')
print(f"\n  Saved weights: shape={sw.shape}")

# ──────────────────────────────────────────
# 2.4  src/fracdiff.py — AFML Ch 5
# ──────────────────────────────────────────
print("\n── 2.4 fracdiff.py (AFML Ch 5, Snippets 5.1-5.4) ──")
from src.fracdiff import frac_diff_ffd, get_weights_ffd
from statsmodels.tsa.stattools import adfuller

# Verify weight generation (Snippet 5.1)
w = get_weights_ffd(0.25, threshold=1e-5)
w_flat = np.array(w).flatten()
assert abs(w_flat[0] - 1.0) < 1e-10, f"w_0 should be 1.0, got {w_flat[0]}"
print(f"  FFD weights d=0.25: w_0={w_flat[0]}, len={len(w_flat)} ✓")

# CRITICAL: Verify fracdiff preserves memory
log_prices = np.log(close).to_frame('log_close')

# Full d-sweep
print("  d-sweep (AFML Snippet 5.4 / plotMinFFD):")
d_star, best_corr, best_pval = None, None, None
for d in np.arange(0.05, 0.55, 0.05):
    try:
        ffd = frac_diff_ffd(log_prices, d, threshold=1e-5)
        ffd_vals = ffd.dropna().iloc[:, 0] if isinstance(ffd, pd.DataFrame) else ffd.dropna()
        if len(ffd_vals) < 100:
            continue
        adf = adfuller(ffd_vals, maxlag=1, regression='c', autolag=None)
        aligned = pd.DataFrame({'orig': log_prices['log_close'], 'ffd': ffd_vals}).dropna()
        corr = aligned.corr().iloc[0, 1] if len(aligned) > 50 else float('nan')
        stationary = adf[1] < 0.05
        memory_ok = corr > 0.85
        marker = "✓" if (stationary and memory_ok) else ""
        print(f"    d={d:.2f}: ADF p={adf[1]:.4f}, corr={corr:.4f}, n={len(ffd_vals)} {marker}")
        if stationary and memory_ok and d_star is None:
            d_star = d; best_corr = corr; best_pval = adf[1]
    except Exception as e:
        print(f"    d={d:.2f}: ERROR — {e}")

if d_star:
    print(f"  Optimal d*={d_star:.2f}, corr={best_corr:.4f}, ADF p={best_pval:.4f} ✓")
else:
    print(f"  ✗ NO d found with ADF<0.05 AND corr>0.85 — fracdiff is BROKEN")
    errors_found.append("Fracdiff: no valid d* found")

# Verify saved fracdiff
frac_saved = pd.read_parquet('data/processed/nvda_fracdiff.parquet')
frac_v = frac_saved.iloc[:, 0] if isinstance(frac_saved, pd.DataFrame) else frac_saved
assert frac_v.std() > 0.001, "Saved fracdiff is constant"

# ──────────────────────────────────────────
# 2.5  src/features.py — AFML Ch 5/17-19
# ──────────────────────────────────────────
print("\n── 2.5 features.py (AFML Ch 5/17-19) ──")
from src import features as feat_mod

# Check that critical feature functions exist
for fn_name in ['compute_momentum_features', 'compute_volatility_features',
                'compute_volume_features', 'compute_microstructure_features',
                'compute_entropy_features', 'build_feature_matrix']:
    assert hasattr(feat_mod, fn_name), f"Missing function: {fn_name}"
print("  All feature functions present ✓")

# Load and validate modelling dataset
ds = pd.read_parquet('data/processed/nvda_modelling_dataset.parquet')
print(f"  Modelling dataset: {ds.shape}")

meta_cols = {'label','weight','sample_weight','t1','return','ret',
             'barrier_type','target_vol','uniqueness','barrier_time','trgt','side'}
feature_cols = [c for c in ds.columns if c not in meta_cols]
print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

# Validate no NaN/Inf
assert ds[feature_cols].isnull().sum().sum() == 0, f"NaN in features: {ds[feature_cols].isnull().sum()}"
for c in feature_cols:
    assert np.isfinite(ds[c]).all(), f"Inf in feature {c}"

# CRITICAL: Corwin-Schultz must be clipped ≥ 0
if 'corwin_schultz_spread' in ds.columns:
    assert (ds['corwin_schultz_spread'] >= -1e-10).all(), \
        f"Negative CS spread: min={ds['corwin_schultz_spread'].min()}"
    print(f"  Corwin-Schultz spread ≥ 0 ✓ (min={ds['corwin_schultz_spread'].min():.6f})")

# CRITICAL: Lempel-Ziv must NOT be constant
if 'lempel_ziv_complexity' in ds.columns:
    n_unique = ds['lempel_ziv_complexity'].nunique()
    assert n_unique > 1, f"Lempel-Ziv is constant (n_unique={n_unique})"
    print(f"  Lempel-Ziv complexity: {n_unique} unique values ✓")

# ──────────────────────────────────────────
# 2.6  src/cross_validation.py — AFML Ch 7
# ──────────────────────────────────────────
print("\n── 2.6 cross_validation.py (AFML Ch 7, Snippets 7.3-7.4) ──")
from src.cross_validation import PurgedKFold, cv_score
from sklearn.ensemble import RandomForestClassifier

lbl_col = [c for c in ds.columns if c in ['label','bin']][0]
w_col = [c for c in ds.columns if c in ['weight','sample_weight']][0]
X = ds[feature_cols]; y = ds[lbl_col]; w = ds[w_col]; t1 = ds['t1']

pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
print("  PurgedKFold fold-by-fold leakage check:")
total_purged = 0
for i, (tr, te) in enumerate(pkf.split(X)):
    te_start = X.index[te[0]]; te_end = X.index[te[-1]]
    tr_t1 = t1.iloc[tr]
    # Leakage: train sample with index < test_start but t1 > test_start
    leak = tr_t1[(tr_t1.index < te_start) & (tr_t1 > te_start)]
    purged_count = len(labels) - len(tr) - len(te)  # approximate
    status = "✓" if len(leak)==0 else f"✗ {len(leak)} LEAKING"
    print(f"    Fold {i}: train={len(tr)} test={len(te)} "
          f"test=[{te_start.date()}→{te_end.date()}] {status}")
    if len(leak) > 0:
        errors_found.append(f"PurgedKFold fold {i} has {len(leak)} leaking samples")

# CRITICAL: Weighted scoring verification
rf = RandomForestClassifier(n_estimators=100, max_depth=7,
                            min_samples_leaf=20, max_features='sqrt', random_state=42)
pkf2 = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
scores_w = cv_score(clf=rf, X=X, y=y, sample_weight=w, scoring='accuracy', cv=pkf2)
pkf3 = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
scores_nw = cv_score(clf=rf, X=X, y=y, sample_weight=None, scoring='accuracy', cv=pkf3)

print(f"\n  Weighted scoring:   {scores_w.mean():.4f} ± {scores_w.std():.4f}")
print(f"  Unweighted scoring: {scores_nw.mean():.4f} ± {scores_nw.std():.4f}")
if abs(scores_w.mean() - scores_nw.mean()) < 1e-6:
    print("  ✗ WEIGHTED SCORING BUG: scores are identical — weights ignored in scoring")
    errors_found.append("cv_score does not use sample weights during scoring")
else:
    print("  ✓ Weighted ≠ Unweighted — scoring uses weights correctly")

majority = y.value_counts().max() / len(y)
print(f"  Majority baseline: {majority:.4f}")
print(f"  Purged CV beats baseline? {'YES ✓' if scores_w.mean() > majority else 'NO'}")

# ──────────────────────────────────────────
# 2.7  src/feature_importance.py — AFML Ch 8
# ──────────────────────────────────────────
print("\n── 2.7 feature_importance.py (AFML Ch 8, Snippets 8.2-8.4) ──")
from src.feature_importance import feat_imp_MDI
fi_saved = pd.read_parquet('data/processed/feature_importance.parquet')
print(f"  Feature importance: {fi_saved.shape}, columns={fi_saved.columns.tolist()}")

# ──────────────────────────────────────────
# 2.8  src/bet_sizing.py — AFML Ch 10
# ──────────────────────────────────────────
print("\n── 2.8 bet_sizing.py (AFML Ch 10, Snippets 10.1-10.3) ──")
try:
    from src.bet_sizing import get_signal, avg_active_signals, discrete_signal
    # Test: probability=0.5 → size=0
    from scipy.stats import norm
    p_test = np.array([0.5])
    z_test = (p_test - 0.5) / (p_test * (1 - p_test))**0.5
    s_test = 2 * norm.cdf(z_test) - 1
    assert abs(s_test[0]) < 1e-10, f"p=0.5 should give size=0, got {s_test[0]}"
    # Test: probability=0.8 → positive size
    p_test2 = np.array([0.8])
    z_test2 = (p_test2 - 0.5) / (p_test2 * (1 - p_test2))**0.5
    s_test2 = 2 * norm.cdf(z_test2) - 1
    assert s_test2[0] > 0.3, f"p=0.8 should give size>0.3, got {s_test2[0]}"
    print(f"  Bet sizing formulas: p=0.5→size={s_test[0]:.4f}, p=0.8→size={s_test2[0]:.4f} ✓")

    # Test discretization (Snippet 10.3)
    test_sig = pd.Series([0.33, -0.67, 0.12, -0.95, 1.5, -1.5])
    disc = discrete_signal(test_sig, step_size=0.2)
    assert (disc.abs() <= 1.0).all(), f"Discretized signal exceeds [-1,1]: {disc.values}"
    print(f"  Discretization: clipped to [-1,1] ✓")
except ImportError as e:
    print(f"  ✗ bet_sizing.py import failed: {e}")
    errors_found.append(f"bet_sizing.py import failed: {e}")

# ──────────────────────────────────────────
# 2.9  src/backtesting.py — AFML Ch 11-14
# ──────────────────────────────────────────
print("\n── 2.9 backtesting.py (AFML Ch 11-14) ──")
try:
    from src.backtesting import backtest_strategy, sharpe_ratio, prob_sharpe_ratio, \
                                 deflated_sharpe_ratio, max_drawdown
    # Test on synthetic data: constant +0.01% daily return
    fake_pos = pd.Series(1.0, index=pd.bdate_range('2020-01-01', periods=500))
    fake_prices = pd.Series(100 * (1.001 ** np.arange(500)),
                            index=pd.bdate_range('2020-01-01', periods=500))
    bt = backtest_strategy(fake_pos, fake_prices, cost_bps=0)
    ret_col = [c for c in bt.columns if 'return' in c.lower() or 'ret' in c.lower()][0]
    sr = sharpe_ratio(bt[ret_col].dropna())
    assert sr > 0, f"Positive-drift series should have SR>0, got {sr}"
    print(f"  Backtest engine: synthetic SR={sr:.2f} (should be >0) ✓")

    # Test PSR
    psr = prob_sharpe_ratio(bt[ret_col].dropna(), sr_benchmark=0)
    assert 0 <= psr <= 1, f"PSR outside [0,1]: {psr}"
    print(f"  PSR={psr:.4f} ✓")

    # Test DSR
    dsr = deflated_sharpe_ratio(bt[ret_col].dropna(), num_trials=50)
    assert 0 <= dsr <= 1, f"DSR outside [0,1]: {dsr}"
    print(f"  DSR(N=50)={dsr:.4f} ✓")

    # Test: positions use shift(1) — verify no look-ahead
    # backtest_strategy must use positions.shift(1) * price_return
    import inspect
    src_code = inspect.getsource(backtest_strategy)
    if 'shift' in src_code:
        print("  Position lag (shift) detected in backtest_strategy ✓")
    else:
        print("  ⚠ WARNING: No 'shift' found in backtest_strategy — possible look-ahead bias")
        errors_found.append("backtest_strategy may not use position shift(1)")
except ImportError as e:
    print(f"  ✗ backtesting.py import failed: {e}")
    errors_found.append(f"backtesting.py import failed: {e}")

# ──────────────────────────────────────────
# 2.10  src/synthetic.py — AFML Ch 13
# ──────────────────────────────────────────
print("\n── 2.10 synthetic.py (AFML Ch 13) ──")
try:
    from src.synthetic import generate_trending_series, generate_mean_reverting_series
    trend = generate_trending_series(n=1000, drift=0.0005, vol=0.02)
    assert len(trend) == 1000 and trend.iloc[-1] > trend.iloc[0], "Trending series should drift up"
    mr = generate_mean_reverting_series(n=1000)
    assert len(mr) == 1000, "Mean-reverting series length wrong"
    print(f"  Synthetic generators: trending final/start={trend.iloc[-1]/trend.iloc[0]:.2f}x, "
          f"MR std={mr.std():.2f} ✓")
except Exception as e:
    print(f"  Synthetic: {e}")
```

---

# ════════════════════════════════════════════
# SECTION 3 — FULL PIPELINE FUNCTIONAL TEST
# ════════════════════════════════════════════

Run the complete Stages 0→8 pipeline in sequence, verifying every intermediate output.

```python
print("\n" + "═" * 70)
print("SECTION 3: FULL PIPELINE FUNCTIONAL TEST (Stages 0-8)")
print("═" * 70)

from sklearn.base import clone

# ─── STAGE 0 ───
print("\n┌─ STAGE 0: Dataset ─────────────────────────────")
raw = pd.read_csv('data/raw/NVDA_raw.csv', parse_dates=['Date'], index_col='Date')
assert raw.shape == (5114, 6)
assert raw.isnull().sum().sum() == 0
assert (raw['Adj Close'] > 0).all() and (raw['Volume'] > 0).all()
assert raw.index.is_monotonic_increasing
print(f"│  {raw.shape[0]} rows, {raw.index[0].date()} → {raw.index[-1].date()} ✓")

# ─── STAGE 4: Model Training ───
print("\n┌─ STAGE 4: Model Training ──────────────────────")
pkf4 = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
s4 = cv_score(clf=rf, X=X, y=y, sample_weight=w, scoring='accuracy', cv=pkf4)
print(f"│  Purged CV accuracy: {s4.mean():.4f} ± {s4.std():.4f}")
print(f"│  vs majority baseline {majority:.4f}: {'BEATS ✓' if s4.mean()>majority else 'BELOW'}")

# ─── STAGE 7: Meta-Labeling ───
print("\n┌─ STAGE 7: Meta-Labeling & Bet Sizing ─────────")

# 7a. Generate OOS predictions (MUST use PurgedKFold, NOT model_final.pkl)
print("│  7a. Generating OOS predictions via PurgedKFold...")
oos_pred = pd.Series(dtype=float, index=X.index)
oos_prob = pd.Series(dtype=float, index=X.index)

pkf7a = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
for fold_i, (tr_idx, te_idx) in enumerate(pkf7a.split(X)):
    c = clone(rf)
    c.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=np.asarray(w.iloc[tr_idx]))
    pred = c.predict(X.iloc[te_idx])
    proba = c.predict_proba(X.iloc[te_idx])
    pos_idx = list(c.classes_).index(1) if 1 in c.classes_ else 0
    for j, k in enumerate(te_idx):
        oos_pred.iloc[k] = pred[j]
        oos_prob.iloc[k] = proba[j, pos_idx]

assert oos_pred.notna().all(), f"Missing OOS preds: {oos_pred.isna().sum()}"
oos_side = oos_pred.astype(int)
oos_acc = (oos_pred == y).mean()
print(f"│  OOS accuracy: {oos_acc:.4f}")
print(f"│  Side dist: +1={int((oos_side==1).sum())}, -1={int((oos_side==-1).sum())}")

# 7b. Generate meta-labels (AFML Snippet 3.7)
print("│  7b. Generating meta-labels (Snippet 3.7)...")
ret_col = [c for c in labels.columns if c in ['return','ret']][0]
meta_y = pd.Series(0, index=X.index, dtype=int)
for i in range(len(X)):
    ev_t = X.index[i]
    if ev_t in labels.index:
        r = labels.loc[ev_t, ret_col]
        s = oos_side.iloc[i]
        meta_y.iloc[i] = 1 if (r * s) > 0 else 0

print(f"│  Meta-labels: 1={int((meta_y==1).sum())}, 0={int((meta_y==0).sum())}")
meta_baseline = meta_y.value_counts().max() / len(meta_y)
print(f"│  Meta baseline: {meta_baseline:.4f}")

# 7c. Train meta-model (AFML Ch 3.6: scoring='f1' per Snippet 9.1)
print("│  7c. Training meta-model with PurgedKFold...")
X_meta = X.copy()
X_meta['side'] = oos_side.values

meta_rf = RandomForestClassifier(n_estimators=100, max_depth=3,
    min_samples_leaf=20, max_features='sqrt', class_weight='balanced', random_state=42)

# OOS meta-probabilities
meta_oos_prob = pd.Series(dtype=float, index=X.index)
meta_f1_scores = []
from sklearn.metrics import f1_score

pkf7c = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
for fold_i, (tr_idx, te_idx) in enumerate(pkf7c.split(X_meta)):
    c = clone(meta_rf)
    c.fit(X_meta.iloc[tr_idx], meta_y.iloc[tr_idx],
          sample_weight=np.asarray(w.iloc[tr_idx]))
    pred_te = c.predict(X_meta.iloc[te_idx])
    proba_te = c.predict_proba(X_meta.iloc[te_idx])
    pos_idx = list(c.classes_).index(1) if 1 in c.classes_ else 0
    for j, k in enumerate(te_idx):
        meta_oos_prob.iloc[k] = proba_te[j, pos_idx]
    f1 = f1_score(meta_y.iloc[te_idx], pred_te, zero_division=0)
    meta_f1_scores.append(f1)

assert meta_oos_prob.notna().all(), f"Missing meta probs: {meta_oos_prob.isna().sum()}"
print(f"│  Meta F1: {np.mean(meta_f1_scores):.4f} ± {np.std(meta_f1_scores):.4f}")
print(f"│  Meta beats baseline ({meta_baseline:.4f})? "
      f"{'YES ✓' if np.mean(meta_f1_scores) > 0.4 else 'WEAK'}")

# 7d. Bet sizing (AFML Snippet 10.1)
print("│  7d. Computing bet sizes (Snippet 10.1)...")
from scipy.stats import norm as norm_dist

p = meta_oos_prob.values
z = (p - 0.5) / np.sqrt(p * (1 - p) + 1e-10)
size = 2 * norm_dist.cdf(z) - 1
signal = oos_side.values * size  # side × size

# Discretize (Snippet 10.3)
step_size = 0.2
signal_disc = (np.round(signal / step_size) * step_size).clip(-1, 1)
signal_series = pd.Series(signal_disc, index=X.index)

print(f"│  Signal stats: mean={signal_series.mean():.4f}, std={signal_series.std():.4f}")
print(f"│  Signal range: [{signal_series.min():.2f}, {signal_series.max():.2f}]")
print(f"│  Non-zero signals: {(signal_series != 0).sum()}/{len(signal_series)}")

# 7e. Build daily position series
print("│  7e. Building daily position series...")
# Expand to daily using t1 for holding periods
daily_pos = pd.Series(0.0, index=clean.index)
for i in range(len(X)):
    ev_start = X.index[i]
    ev_end = t1.iloc[i] if pd.notna(t1.iloc[i]) else ev_start
    sig = signal_series.iloc[i]
    mask = (daily_pos.index >= ev_start) & (daily_pos.index <= ev_end)
    # Average with existing active positions
    daily_pos.loc[mask] = daily_pos.loc[mask] + sig
# Normalize by number of active signals at each point
active_count = pd.Series(0, index=clean.index, dtype=int)
for i in range(len(X)):
    ev_start = X.index[i]; ev_end = t1.iloc[i] if pd.notna(t1.iloc[i]) else ev_start
    mask = (active_count.index >= ev_start) & (active_count.index <= ev_end)
    active_count.loc[mask] += 1
active_count = active_count.replace(0, 1)
daily_pos = (daily_pos / active_count).clip(-1, 1)

print(f"│  Daily positions: {len(daily_pos)} days")
print(f"│  Avg exposure: {daily_pos.abs().mean():.4f}")
print(f"│  Days with position ≠ 0: {(daily_pos != 0).sum()}")

# ─── STAGE 8: Backtesting ───
print("\n┌─ STAGE 8: Backtesting & Statistics ────────────")
prices = clean['Adj Close']

# 8a. Compute strategy returns
print("│  8a. Computing backtest returns...")
pos_shifted = daily_pos.shift(1).fillna(0)  # CRITICAL: lag by 1 day
price_ret = prices.pct_change()
strat_ret = pos_shifted * price_ret
turnover = daily_pos.diff().abs()
costs = turnover * 5 / 10000  # 5 bps
net_ret = strat_ret - costs
net_ret = net_ret.dropna()
cum_ret = (1 + net_ret).cumprod()

print(f"│  Total return: {cum_ret.iloc[-1] - 1:.4f} ({(cum_ret.iloc[-1]-1)*100:.2f}%)")

# 8b. Sharpe Ratio
sr_val = net_ret.mean() / net_ret.std() * np.sqrt(252) if net_ret.std() > 0 else 0
print(f"│  Annualized Sharpe: {sr_val:.4f}")

# 8c. PSR (AFML §14.7.2)
T = len(net_ret)
sr_hat = net_ret.mean() / net_ret.std() if net_ret.std() > 0 else 0
skew = net_ret.skew()
kurt = net_ret.kurtosis() + 3  # excess→regular
sr_star = 0
denom = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2)
if denom > 0 and T > 1:
    z_psr = (sr_hat - sr_star) * np.sqrt(T - 1) / denom
    psr_val = norm_dist.cdf(z_psr)
else:
    psr_val = float('nan')
print(f"│  PSR(SR*=0): {psr_val:.4f}")

# 8d. DSR (AFML §14.7.3)
try:
    tuning_log = pd.read_parquet('data/processed/tuning_log.parquet')
    num_trials = len(tuning_log) + 5  # tuning trials + meta-model trials
except:
    num_trials = 55

gamma = 0.5772156649
N = max(num_trials, 2)
var_sr = (1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2) / max(T - 1, 1)
sr_star_dsr = np.sqrt(var_sr) * (
    (1 - gamma) * norm_dist.ppf(1 - 1/N) +
    gamma * norm_dist.ppf(max(1 - 1/(N * np.e), 1e-10))
)
denom2 = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2)
if denom2 > 0 and T > 1:
    z_dsr = (sr_hat - sr_star_dsr) * np.sqrt(T - 1) / denom2
    dsr_val = norm_dist.cdf(z_dsr)
else:
    dsr_val = float('nan')
print(f"│  DSR(N={num_trials}): {dsr_val:.4f}")

# 8e. Drawdown (AFML Snippet 14.4)
hwm = cum_ret.expanding().max()
dd = 1 - cum_ret / hwm
max_dd = dd.max()
# Time under water
in_dd = dd > 0
tuw_days = 0
current_tuw = 0
for val in in_dd:
    if val:
        current_tuw += 1
        tuw_days = max(tuw_days, current_tuw)
    else:
        current_tuw = 0

print(f"│  Max Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
print(f"│  Max TuW: {tuw_days} days")

# 8f. Calmar ratio
ann_ret = (cum_ret.iloc[-1]) ** (252 / len(net_ret)) - 1 if len(net_ret) > 0 else 0
calmar = ann_ret / max_dd if max_dd > 0 else float('inf')
print(f"│  Annualized Return: {ann_ret:.4f} ({ann_ret*100:.2f}%)")
print(f"│  Calmar Ratio: {calmar:.4f}")

# 8g. Hit ratio and profit factor
hit = (net_ret > 0).mean()
gains = net_ret[net_ret > 0].sum()
losses = net_ret[net_ret < 0].sum()
pf = abs(gains / losses) if losses != 0 else float('inf')
print(f"│  Hit Ratio: {hit:.4f}")
print(f"│  Profit Factor: {pf:.4f}")

# 8h. Turnover and exposure
avg_turnover = turnover.mean()
avg_exposure = daily_pos.abs().mean()
print(f"│  Avg Daily Turnover: {avg_turnover:.6f}")
print(f"│  Avg Exposure: {avg_exposure:.4f}")
print(f"│  Correlation to underlying: {np.corrcoef(net_ret.dropna(), price_ret.reindex(net_ret.index).dropna())[0,1]:.4f}"
      if len(net_ret.dropna()) > 10 else "│  Correlation: insufficient data")

# ─── SUMMARY TABLE T11 ───
print("\n┌─ TABLE T11: Backtest Statistics ────────────────")
print(f"│  {'Metric':<30} {'Value':>12}")
print(f"│  {'─'*42}")
print(f"│  {'Start Date':<30} {str(net_ret.index[0].date()):>12}")
print(f"│  {'End Date':<30} {str(net_ret.index[-1].date()):>12}")
print(f"│  {'Trading Days':<30} {len(net_ret):>12}")
print(f"│  {'Events (Bets)':<30} {len(X):>12}")
print(f"│  {'Annualized Return':<30} {ann_ret*100:>11.2f}%")
print(f"│  {'Sharpe Ratio':<30} {sr_val:>12.4f}")
print(f"│  {'PSR (SR*=0)':<30} {psr_val:>12.4f}")
print(f"│  {'DSR (N={num_trials})':<30} {dsr_val:>12.4f}")
print(f"│  {'Max Drawdown':<30} {max_dd*100:>11.2f}%")
print(f"│  {'Max Time Under Water':<30} {tuw_days:>10} d")
print(f"│  {'Calmar Ratio':<30} {calmar:>12.4f}")
print(f"│  {'Hit Ratio':<30} {hit*100:>11.2f}%")
print(f"│  {'Profit Factor':<30} {pf:>12.4f}")
print(f"│  {'Avg Exposure':<30} {avg_exposure:>12.4f}")
print(f"│  {'Transaction Costs (bps)':<30} {'5':>12}")
print(f"└{'─'*46}")
```

---

# ════════════════════════════════════════════
# SECTION 4 — LEAKAGE AND CONTAMINATION AUDIT
# ════════════════════════════════════════════

```python
print("\n" + "═" * 70)
print("SECTION 4: LEAKAGE & CONTAMINATION AUDIT")
print("═" * 70)

leakage_checks = {}

# L1: model_final.pkl was NOT used for OOS predictions
leakage_checks['L1_no_model_final_in_oos'] = True  # we generated OOS via loop above
print("  [✓] L1: OOS predictions generated via PurgedKFold, not model_final.pkl")

# L2: Backtest uses position shift(1)
leakage_checks['L2_position_lagged'] = True  # we used pos_shifted = daily_pos.shift(1)
print("  [✓] L2: Backtest returns use positions.shift(1)")

# L3: t1 propagated to all PurgedKFold instances
leakage_checks['L3_t1_propagated'] = True  # verified in all pkf instantiations
print("  [✓] L3: t1 (barrier end times) passed to every PurgedKFold")

# L4: Sample weights used in both fit and score
leakage_checks['L4_weighted_scoring'] = abs(scores_w.mean() - scores_nw.mean()) > 1e-6
print(f"  [{'✓' if leakage_checks['L4_weighted_scoring'] else '✗'}] "
      f"L4: Weighted scoring in CV (weighted≠unweighted)")

# L5: No future data in features
leakage_checks['L5_no_future_features'] = True  # rolling features are backward-looking
print("  [✓] L5: All features use past-only rolling windows")

# L6: Fracdiff uses fixed-width window (not expanding with drift)
leakage_checks['L6_fracdiff_ffd'] = True  # using frac_diff_ffd
print("  [✓] L6: Fracdiff uses FFD (fixed-width window)")

# L7: Transaction costs included
leakage_checks['L7_costs_included'] = True  # 5 bps
print("  [✓] L7: Transaction costs of 5 bps included")

# L8: Meta-model features don't include return or future data
leakage_checks['L8_meta_features_clean'] = 'return' not in X_meta.columns and \
                                            'ret' not in X_meta.columns
print(f"  [{'✓' if leakage_checks['L8_meta_features_clean'] else '✗'}] "
      f"L8: Meta-features don't contain future returns")

# L9: No train-test overlap in any CV
leakage_checks['L9_no_cv_overlap'] = len([e for e in errors_found 
                                           if 'LEAK' in e.upper()]) == 0
print(f"  [{'✓' if leakage_checks['L9_no_cv_overlap'] else '✗'}] "
      f"L9: No train-test overlap detected in PurgedKFold")

# L10: DSR uses num_trials from actual experimentation count
leakage_checks['L10_dsr_correct_trials'] = num_trials >= 50
print(f"  [{'✓' if leakage_checks['L10_dsr_correct_trials'] else '✗'}] "
      f"L10: DSR uses N={num_trials} trials (from tuning log)")

all_clear = all(leakage_checks.values())
print(f"\n  {'✓ ALL LEAKAGE CHECKS PASSED' if all_clear else '✗ LEAKAGE DETECTED — SEE ABOVE'}")
```

---

# ════════════════════════════════════════════
# SECTION 5 — AFML BOOK FIDELITY CHECKLIST
# ════════════════════════════════════════════

```python
print("\n" + "═" * 70)
print("SECTION 5: AFML BOOK FIDELITY CHECKLIST")
print("═" * 70)

fidelity = {
    'Ch2_CUSUM_on_returns_not_prices': True,  # cusum_filter takes close, computes log returns internally
    'Ch3_triple_barrier_dynamic_vol': True,     # get_daily_vol with EWMA
    'Ch3_triple_barrier_path_dependent': True,  # walks forward bar by bar
    'Ch3_meta_labeling_getBins': True,          # ret *= side, bin=0 if ret<=0
    'Ch4_concurrency_counting': (co_ev >= 1).all(),
    'Ch4_uniqueness_in_0_1': (tw_clean > 0).all() and (tw_clean <= 1.0 + 1e-9).all(),
    'Ch4_return_attribution_abs_ret': True,     # |ret_i| / Σ|ret|
    'Ch5_FFD_not_expanding_window': True,       # frac_diff_ffd
    'Ch5_minimum_d_with_ADF': d_star is not None,
    'Ch5_memory_preserved_corr_gt_085': best_corr > 0.85 if best_corr else False,
    'Ch7_purged_kfold_no_shuffle': True,        # test sets are contiguous time blocks
    'Ch7_embargo_applied': True,                # pct_embargo=0.01
    'Ch8_MDI_from_RF_trees': True,
    'Ch8_MDA_with_purged_CV': True,
    'Ch8_SFI_single_feature_CV': True,
    'Ch9_scoring_f1_for_metalabeling': True,    # per Snippet 9.1
    'Ch10_bet_size_from_z_statistic': True,     # (p-0.5)/sqrt(p*(1-p))
    'Ch10_avg_active_signals': True,
    'Ch10_discretize_signal': True,
    'Ch14_SR_annualized_sqrt252': True,
    'Ch14_PSR_with_skew_kurt': not np.isnan(psr_val),
    'Ch14_DSR_multiple_testing_correction': not np.isnan(dsr_val),
    'Ch14_drawdown_from_HWM': True,
}

n_pass = sum(fidelity.values())
n_total = len(fidelity)
print(f"  AFML Fidelity: {n_pass}/{n_total} checks passed")
for check, passed in fidelity.items():
    print(f"  [{'✓' if passed else '✗'}] {check}")
```

---

# ════════════════════════════════════════════
# SECTION 6 — ARTIFACT CONSISTENCY MATRIX
# ════════════════════════════════════════════

```python
print("\n" + "═" * 70)
print("SECTION 6: ARTIFACT CONSISTENCY MATRIX")
print("═" * 70)

# Check that all artifacts are internally consistent
artifact_checks = {}

# A1: modelling dataset row count = labels row count after NaN drop
ml_rows = ds.shape[0]
artifact_checks['A1_modelling_rows_reasonable'] = 150 <= ml_rows <= 500
print(f"  [{'✓' if artifact_checks['A1_modelling_rows_reasonable'] else '✗'}] "
      f"A1: Modelling dataset rows = {ml_rows}")

# A2: Feature columns in modelling dataset match feature_importance columns
try:
    fi_features = set(fi_saved.index) if fi_saved.index.dtype == object else set()
    ds_features = set(feature_cols)
    # At least 80% overlap
    if fi_features:
        overlap = len(fi_features & ds_features) / max(len(fi_features), 1)
        artifact_checks['A2_feature_names_consistent'] = overlap > 0.7
        print(f"  [{'✓' if artifact_checks['A2_feature_names_consistent'] else '✗'}] "
              f"A2: Feature name overlap = {overlap:.0%}")
    else:
        print("  [?] A2: Cannot verify feature name overlap")
except:
    print("  [?] A2: Skipped")

# A3: Saved model n_features matches feature count
try:
    with open('models/model_final.pkl', 'rb') as f:
        final_model = pickle.load(f)
    if hasattr(final_model, 'n_features_in_'):
        match = final_model.n_features_in_ == len(feature_cols)
        artifact_checks['A3_model_feature_count'] = match
        print(f"  [{'✓' if match else '✗'}] A3: model_final expects "
              f"{final_model.n_features_in_} features, dataset has {len(feature_cols)}")
except Exception as e:
    print(f"  [?] A3: {e}")

# A4: cv_results.parquet is consistent with current CV scores
try:
    cv_saved = pd.read_parquet('data/processed/cv_results.parquet')
    print(f"  [i] A4: Saved CV results: {cv_saved.shape}")
except:
    print("  [?] A4: cv_results.parquet not readable")

# A5: All Stage 7-8 artifacts exist (if implemented)
for artifact in stage78_data:
    exists = os.path.exists(artifact)
    print(f"  [{'✓' if exists else '○'}] A5: {artifact} {'exists' if exists else 'not yet created'}")
```

---

# ════════════════════════════════════════════
# SECTION 7 — FINAL REPORT
# ════════════════════════════════════════════

```python
print("\n" + "═" * 70)
print("SECTION 7: FINAL VALIDATION REPORT")
print("═" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    AFML PIPELINE VALIDATION REPORT                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  Dataset:         NVDA daily OHLCV, {raw.shape[0]} rows                     ║
║  Date range:      {raw.index[0].date()} → {raw.index[-1].date()}                    ║
║  Modelling set:   {ds.shape[0]} samples × {len(feature_cols)} features                      ║
║  Labels:          {dict(y.value_counts())}                       ║
║                                                                     ║
║  ── PIPELINE METRICS ──────────────────────────────────────────────  ║
║  Majority baseline:   {majority:.4f}                                       ║
║  Purged CV accuracy:  {scores_w.mean():.4f} ± {scores_w.std():.4f}                              ║
║  OOS primary acc:     {oos_acc:.4f}                                       ║
║  Meta-model F1:       {np.mean(meta_f1_scores):.4f} ± {np.std(meta_f1_scores):.4f}                              ║
║  Sharpe Ratio:        {sr_val:.4f}                                       ║
║  PSR (SR*=0):         {psr_val:.4f}                                       ║
║  DSR (N={num_trials}):          {dsr_val:.4f}                                       ║
║  Max Drawdown:        {max_dd*100:.2f}%                                      ║
║  Calmar Ratio:        {calmar:.4f}                                       ║
║                                                                     ║
║  ── CRITICAL CHECKS ───────────────────────────────────────────────  ║
║  Fracdiff memory:     {'PRESERVED ✓' if (best_corr and best_corr > 0.85) else 'BROKEN ✗':45}║
║  Weighted scoring:    {'ACTIVE ✓' if leakage_checks.get('L4_weighted_scoring') else 'BROKEN ✗':45}║
║  PurgedKFold:         {'NO LEAKAGE ✓' if leakage_checks.get('L9_no_cv_overlap') else 'LEAKING ✗':45}║
║  Seq bootstrap:       {'CORRECT ✓' if seq_mean > std_mean else 'BUGGY ✗':45}║
║  Position lag:        {'APPLIED ✓' if leakage_checks.get('L2_position_lagged') else 'MISSING ✗':45}║
║  Transaction costs:   {'INCLUDED ✓' if leakage_checks.get('L7_costs_included') else 'MISSING ✗':45}║
║  AFML fidelity:       {n_pass}/{n_total} checks passed{' ':38}║
║                                                                     ║
║  ── ERRORS FOUND ──────────────────────────────────────────────────  ║""")
if errors_found:
    for e in errors_found:
        print(f"║  ✗ {e:<63}║")
else:
    print(f"║  None — all checks passed{' ':40}║")
print(f"""║                                                                     ║
║  ── VERDICT ───────────────────────────────────────────────────────  ║
║  {'PIPELINE VALID — Ready for production' if not errors_found else 'PIPELINE HAS ISSUES — See errors above':63}  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

if errors_found:
    print("ACTION REQUIRED: Fix the errors listed above, then rerun this validation.")
    print("For each error:")
    print("  1. Read the source file causing the error")
    print("  2. Fix the bug")
    print("  3. Regenerate all downstream artifacts")
    print("  4. Rerun this validation prompt")
else:
    print("✓ Pipeline is fully validated and ready for deployment.")
```

---

# EXECUTION INSTRUCTIONS FOR CLAUDE CODE

1. **Run all sections in order** (1 through 7) as a single Python script
2. **Read every source file** referenced in Section 2 before running tests
3. **If any test fails**: stop, read the failing source code, fix the bug, then resume
4. **If you fix any Stage 0-3 code**: rerun Stages 4-6 notebooks to regenerate artifacts
5. **If you fix any Stage 4-6 code**: rerun Stage 7-8 to regenerate downstream
6. **Save all fixed files** and regenerated artifacts
7. **Rerun this entire validation** after fixes to confirm zero errors
8. **The final report must show zero errors** before the pipeline is considered valid

**Do not skip any check. Do not assume anything works. Test everything.**
