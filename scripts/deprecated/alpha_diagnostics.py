"""
Prompt 4: Alpha Diagnostics & Pruning.
Apply exclusion rules, redundancy pruning, and feature budget to produce
the curated alpha feature set for modelling.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('configs', exist_ok=True)

# ── Step 1: Load ───────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading alpha features & diagnostics")
print("=" * 60)

ap   = pd.read_parquet('data/processed/panel_alpha_features.parquet')
diag = pd.read_parquet('data/processed/alpha_diagnostics.parquet')

print(f"  Alpha panel: {ap.shape}")
print(f"  Diagnostics: {diag.shape}")

# ── Step 2: Exclusion rules ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Applying exclusion rules")
print("=" * 60)

nan_pcts = ap.isnull().mean() * 100
stds     = ap.std()

ex_nan  = nan_pcts[nan_pcts > 40].index.tolist()
ex_const = stds[stds < 1e-8].index.tolist()
ex_inf   = [c for c in ap.columns if np.isinf(ap[c]).any()]

print(f"  Rule 1 - NaN% > 40%:         {len(ex_nan):3d} excluded: {ex_nan}")
print(f"  Rule 2 - Constant (std<1e-8): {len(ex_const):3d} excluded: {ex_const}")
print(f"  Rule 3 - Any inf:             {len(ex_inf):3d} excluded: {ex_inf}")

excluded = set(ex_nan) | set(ex_const) | set(ex_inf)
surviving = [c for c in ap.columns if c not in excluded]
print(f"  After exclusion: {len(surviving)} alphas surviving")

ap_surv = ap[surviving].copy()

# ── Step 3: Cross-alpha correlation (on NVDA column only) ─────────────────
print("\n" + "=" * 60)
print("STEP 3: Cross-alpha correlation matrix (NVDA column)")
print("=" * 60)

nvda_alpha = ap_surv.xs('NVDA', level='ticker')
corr_matrix = nvda_alpha.corr()

# Save heatmap
fig, ax = plt.subplots(figsize=(20, 16))
n = len(surviving)
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=6)
ax.set_yticklabels(corr_matrix.columns, fontsize=6)
plt.colorbar(im, ax=ax)
ax.set_title('Cross-Alpha Correlation Matrix (NVDA column)')
plt.tight_layout()
plt.savefig('reports/figures/P_alpha_correlation_heatmap.png', dpi=100)
plt.close()
print(f"  Saved heatmap to reports/figures/P_alpha_correlation_heatmap.png")

# ── Step 4: Redundancy pruning (|corr| > 0.85) ────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Redundancy pruning (|corr| > 0.85)")
print("=" * 60)

# Stationarity scores from diagnostics (lower ADF p = more stationary)
adf_scores = diag['adf_pval_median'].reindex(surviving).fillna(1.0)

pruned_out = set()
pruned_pairs = []

alpha_list = list(surviving)
for i in range(len(alpha_list)):
    ai = alpha_list[i]
    if ai in pruned_out:
        continue
    for j in range(i + 1, len(alpha_list)):
        aj = alpha_list[j]
        if aj in pruned_out:
            continue
        if abs(corr_matrix.loc[ai, aj]) > 0.85:
            # Keep the one with lower ADF p-value (more stationary)
            if adf_scores.get(ai, 1.0) <= adf_scores.get(aj, 1.0):
                pruned_out.add(aj)
                pruned_pairs.append((ai, aj, corr_matrix.loc[ai, aj]))
            else:
                pruned_out.add(ai)
                pruned_pairs.append((aj, ai, corr_matrix.loc[ai, aj]))

post_prune = [a for a in alpha_list if a not in pruned_out]
print(f"  Pruned {len(pruned_out)} redundant alphas. Remaining: {len(post_prune)}")
if pruned_pairs[:5]:
    print(f"  Example pruned pairs (keeper, dropped, corr):")
    for k, d, c in pruned_pairs[:5]:
        print(f"    keep={k}, drop={d}, corr={c:.3f}")

# ── Step 5: Feature budget (max 33 alphas) ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Applying feature budget (max 33 alphas)")
print("=" * 60)

BUDGET = 33
if len(post_prune) > BUDGET:
    # Rank by median ADF p-value (most stationary first)
    scores = adf_scores.reindex(post_prune).fillna(1.0)
    selected = scores.nsmallest(BUDGET).index.tolist()
    print(f"  {len(post_prune)} survived → selecting top {BUDGET} by stationarity")
else:
    selected = post_prune
    print(f"  {len(post_prune)} survived — within budget of {BUDGET}")

print(f"  Final alpha feature set: {len(selected)} alphas")
print(f"  Selected: {selected}")

# ── Step 6: Save ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving pruned alpha features")
print("=" * 60)

ap_pruned = ap[selected].copy()
ap_pruned.to_parquet('data/processed/panel_alpha_features_pruned.parquet')

config = {
    'selected_alphas': selected,
    'n_alphas': len(selected),
    'excluded_nan40': ex_nan,
    'excluded_constant': ex_const,
    'excluded_inf': ex_inf,
    'pruned_redundant': list(pruned_out),
    'budget': BUDGET,
}
with open('configs/selected_alphas.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"  Saved panel_alpha_features_pruned.parquet: {ap_pruned.shape}")
print(f"  Saved configs/selected_alphas.json")

# ── Step 7: Validate ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Validation")
print("=" * 60)

nan_check  = (ap_pruned.isnull().mean() * 100 > 40).any()
const_check = (ap_pruned.std() < 1e-8).any()
corr_final  = nvda_alpha[selected].corr()
corr_arr = corr_final.to_numpy().copy()
np.fill_diagonal(corr_arr, 0)
max_corr = float(np.abs(corr_arr).max())

print(f"  Any selected alpha NaN% > 40%: {nan_check}")
print(f"  Any selected alpha constant:   {const_check}")
print(f"  Max pairwise |corr| in selected set: {max_corr:.3f} (threshold 0.85)")
print(f"  configs/selected_alphas.json valid: {os.path.exists('configs/selected_alphas.json')}")
print(f"  Selected count in range [15,33]: {15 <= len(selected) <= 33}")

print("\n" + "=" * 60)
print("PROMPT 4 COMPLETE")
print(f"  Surviving after exclusion:   {len(surviving)}")
print(f"  Surviving after redundancy:  {len(post_prune)}")
print(f"  Final selected (budget cap): {len(selected)}")
print(f"  Total pipeline features:     17 TS + {len(selected)} alpha = {17 + len(selected)}")
print("=" * 60)
