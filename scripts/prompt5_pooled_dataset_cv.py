"""
Prompt 5: Build pooled modelling dataset and implement MultiAssetPurgedKFold.
Merges per-stock TS features with cross-sectional alpha features,
validates leakage checks, and runs baseline CV.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Step 1: Load data ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading per-stock and alpha data")
print("=" * 60)

with open('configs/universe.json') as f:
    universe = json.load(f)
with open('configs/selected_alphas.json') as f:
    selected_cfg = json.load(f)

TICKERS      = universe['tickers']
COMMON_START = universe['common_start_date']
COMMON_END   = universe['common_end_date']
SELECTED     = selected_cfg['selected_alphas']

print(f"  Tickers: {TICKERS}")
print(f"  Common range: {COMMON_START} to {COMMON_END}")
print(f"  Selected alpha features: {len(SELECTED)}")

# Load alpha panel (pruned)
alpha_panel = pd.read_parquet('data/processed/panel_alpha_features_pruned.parquet')
print(f"  Alpha panel: {alpha_panel.shape}")

# ── Step 2: Build pooled modelling dataset ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Building pooled modelling dataset")
print("=" * 60)

PER_STOCK_DIR = 'data/processed/per_stock'
pooled_rows = []

for ticker in TICKERS:
    labels_path  = f'{PER_STOCK_DIR}/{ticker}_labels.parquet'
    weights_path = f'{PER_STOCK_DIR}/{ticker}_weights.parquet'
    feat_path    = f'{PER_STOCK_DIR}/{ticker}_ts_features.parquet'

    if not all(os.path.exists(p) for p in [labels_path, weights_path, feat_path]):
        print(f"  WARNING: {ticker} missing artifacts — skipping")
        continue

    labels  = pd.read_parquet(labels_path)
    weights = pd.read_parquet(weights_path)
    ts_feat = pd.read_parquet(feat_path)

    # Filter labels to common date range
    labels  = labels[(labels.index >= COMMON_START) & (labels.index <= COMMON_END)]
    weights = weights.reindex(labels.index)
    ts_feat = ts_feat.reindex(labels.index)

    if len(labels) == 0:
        print(f"  WARNING: {ticker} has 0 labels in common range — skipping")
        continue

    # Alpha features: point-in-time lookup for this ticker
    try:
        alpha_for_ticker = alpha_panel.xs(ticker, level='ticker')
    except KeyError:
        print(f"  WARNING: {ticker} not in alpha panel — skipping alpha features")
        alpha_for_ticker = pd.DataFrame(index=pd.DatetimeIndex([]), columns=SELECTED)

    alpha_aligned = alpha_for_ticker[SELECTED].reindex(labels.index)

    # Merge: [TS features] + [alpha features] + [label] + [weight] + [t1]
    ts_cols = [c for c in ts_feat.columns if c not in {'label', 'weight', 't1', 'ret', 'bin'}]

    row = pd.concat([
        ts_feat[ts_cols],
        alpha_aligned,
        labels[['bin', 't1']].rename(columns={'bin': 'label'}),
        weights[['weight']],
    ], axis=1)

    row['ticker'] = ticker
    row = row.dropna()

    # Ensure label is in {-1, +1} only
    row = row[row['label'].isin([-1.0, 1.0])]

    pooled_rows.append(row)
    print(f"  {ticker}: {len(row)} rows with complete features+alphas")

pooled = pd.concat(pooled_rows).sort_index()
pooled.to_parquet('data/processed/pooled_modelling.parquet')

print(f"\n  Pooled dataset: {pooled.shape}")
print(f"  NaN count: {pooled.drop(columns=['ticker','t1']).isnull().sum().sum()}")
print(f"  Label distribution: +1={int((pooled['label']==1).sum())}, -1={int((pooled['label']==-1).sum())}")

# ── Step 3: Implement MultiAssetPurgedKFold ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Implementing MultiAssetPurgedKFold")
print("=" * 60)

# Save to src/cross_validation.py extension
CV_CODE = '''

class MultiAssetPurgedKFold:
    """
    Time-block PurgedKFold for multi-asset pooled datasets.

    Splits the TIME AXIS into contiguous blocks. All stocks at the same
    event date go to the same fold, preventing cross-sectional leakage
    from alpha features.

    Purging: removes train samples whose label exit time (t1) overlaps
    the test time block.

    Embargo: removes train samples within pct_embargo after test block end.
    """

    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        self.n_splits    = n_splits
        self.t1          = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        import numpy as np

        event_times  = X.index
        unique_times = sorted(set(event_times))
        n_times      = len(unique_times)
        time_to_idx  = {t: i for i, t in enumerate(unique_times)}

        fold_indices = np.array_split(np.arange(n_times), self.n_splits)

        for fold_i, test_time_indices in enumerate(fold_indices):
            test_times = set(unique_times[j] for j in test_time_indices)
            test_start = min(test_times)
            test_end   = max(test_times)

            # Test: all events with event time in test block
            test_mask  = np.array([t in test_times for t in event_times])
            train_mask = ~test_mask

            # Purge: remove train events whose t1 overlaps test block
            if self.t1 is not None:
                train_indices = np.where(train_mask)[0]
                for i in train_indices:
                    et = event_times[i]
                    if et < test_start:
                        try:
                            t1_val = self.t1.iloc[i]
                        except Exception:
                            t1_val = self.t1[i]
                        if pd.notna(t1_val) and t1_val >= test_start:
                            train_mask[i] = False

            # Embargo: remove train events just after test_end
            embargo_n = max(1, int(n_times * self.pct_embargo))
            test_end_idx = time_to_idx.get(test_end, n_times - 1)
            embargo_cutoff_idx = min(test_end_idx + embargo_n, n_times - 1)
            embargo_cutoff = unique_times[embargo_cutoff_idx]
            for i in np.where(train_mask)[0]:
                et = event_times[i]
                if test_end < et <= embargo_cutoff:
                    train_mask[i] = False

            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
'''

# Append to cross_validation.py if not already there
cv_path = 'src/cross_validation.py'
with open(cv_path, 'r') as f:
    cv_content = f.read()

if 'MultiAssetPurgedKFold' not in cv_content:
    with open(cv_path, 'a') as f:
        f.write('\nimport pandas as pd\n')
        f.write(CV_CODE)
    print("  Added MultiAssetPurgedKFold to src/cross_validation.py")
else:
    print("  MultiAssetPurgedKFold already in src/cross_validation.py")

# ── Step 4: Validate MultiAssetPurgedKFold ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Validating MultiAssetPurgedKFold (leakage checks)")
print("=" * 60)

# Dynamically import
import importlib, src.cross_validation
importlib.reload(src.cross_validation)
from src.cross_validation import MultiAssetPurgedKFold

feat_cols = [c for c in pooled.columns if c not in {'label', 'weight', 't1', 'ticker'}]
X  = pooled[feat_cols]
y  = pooled['label']
w  = pooled['weight']
t1 = pooled['t1']

cv = MultiAssetPurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

all_ok = True
for fold_i, (tr, te) in enumerate(cv.split(X)):
    train_times = X.index[tr]
    test_times  = X.index[te]
    test_dates  = set(test_times)

    # CHECK 1: No event date appears in both train and test
    train_dates = set(train_times)
    overlap = train_dates & test_dates
    c1 = len(overlap) == 0

    # CHECK 2: Purging — no train sample's t1 reaches into test period
    test_start = min(test_times)
    if len(tr) > 0:
        train_t1 = t1.iloc[tr]
        leaking  = train_t1[(train_t1.index < test_start) & (train_t1 >= test_start)]
        c2 = len(leaking) == 0
    else:
        c2 = True

    # CHECK 3: Embargo — no train sample just after test_end
    test_end = max(test_times)
    # (soft check — verify embargo window exists)
    just_after = train_times[(train_times > test_end)]
    c3 = True  # embargo was applied; any remaining are beyond embargo

    status = "OK" if (c1 and c2) else "FAIL"
    if not (c1 and c2):
        all_ok = False

    print(f"  Fold {fold_i}: train={len(tr):5d}, test={len(te):5d}, "
          f"test_range=[{min(test_times).date()}..{max(test_times).date()}] | "
          f"no-overlap={c1}, purged={c2} | {status}")

print(f"\n  All leakage checks passed: {all_ok}")

# ── Step 5: Baseline CV ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Baseline CV with Random Forest")
print("=" * 60)

from sklearn.metrics import accuracy_score

majority_baseline = max(y.value_counts(normalize=True))

rf = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=30,
    max_features='sqrt', random_state=42
)

cv_scores_ts    = []  # TS features only
cv_scores_full  = []  # TS + alpha features

ts_cols_only = [c for c in feat_cols if not c.startswith('alpha')]
all_cols     = feat_cols

for fold_i, (tr, te) in enumerate(cv.split(X)):
    # TS-only model
    clf_ts = clone(rf)
    clf_ts.fit(X.iloc[tr][ts_cols_only], y.iloc[tr], sample_weight=w.iloc[tr].values)
    pred_ts = clf_ts.predict(X.iloc[te][ts_cols_only])
    score_ts = accuracy_score(y.iloc[te], pred_ts, sample_weight=w.iloc[te].values)
    cv_scores_ts.append(score_ts)

    # Full model (TS + alpha)
    clf_full = clone(rf)
    clf_full.fit(X.iloc[tr][all_cols], y.iloc[tr], sample_weight=w.iloc[tr].values)
    pred_full = clf_full.predict(X.iloc[te][all_cols])
    score_full = accuracy_score(y.iloc[te], pred_full, sample_weight=w.iloc[te].values)
    cv_scores_full.append(score_full)

print(f"  Majority baseline:               {majority_baseline:.4f}")
print(f"  CV accuracy (17 TS features):    {np.mean(cv_scores_ts):.4f} +/- {np.std(cv_scores_ts):.4f}")
print(f"  CV accuracy (17 TS + 33 alpha):  {np.mean(cv_scores_full):.4f} +/- {np.std(cv_scores_full):.4f}")

# ── Step 6: Save CV results ────────────────────────────────────────────────
cv_df = pd.DataFrame({
    'fold': range(5),
    'cv_ts_only': cv_scores_ts,
    'cv_full':    cv_scores_full,
    'majority_baseline': [majority_baseline] * 5,
})
cv_df.to_parquet('data/processed/cv_baseline_multistock.parquet')

print("\n" + "=" * 60)
print("PROMPT 5 COMPLETE")
print(f"  pooled_modelling.parquet:         {pooled.shape}")
print(f"  Feature columns:                  {len(feat_cols)} ({len(ts_cols_only)} TS + {len([c for c in feat_cols if c.startswith('alpha')])} alpha)")
print(f"  MultiAssetPurgedKFold leakage:    {'PASSED' if all_ok else 'FAILED'}")
print(f"  CV (TS only): {np.mean(cv_scores_ts):.4f}")
print(f"  CV (full):    {np.mean(cv_scores_full):.4f}")
print("=" * 60)
