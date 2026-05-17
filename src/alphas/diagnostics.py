"""
Alpha diagnostics: exclusion rules, redundancy pruning, feature budget,
and selection of the final alpha feature set.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def compute_nan_rates(alpha_panel: pd.DataFrame) -> pd.Series:
    """Return NaN percentage (0–100) per alpha column."""
    return alpha_panel.isnull().mean() * 100


def compute_stds(alpha_panel: pd.DataFrame) -> pd.Series:
    return alpha_panel.std()


def apply_exclusion_rules(
    alpha_panel: pd.DataFrame,
    max_nan_pct: float = 40.0,
    min_std: float = 1e-8,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Return (surviving_alphas, exclusion_log) where exclusion_log maps
    rule name to list of excluded alpha names.
    """
    nan_pcts = compute_nan_rates(alpha_panel)
    stds     = compute_stds(alpha_panel)

    ex_nan   = nan_pcts[nan_pcts > max_nan_pct].index.tolist()
    ex_const = stds[stds < min_std].index.tolist()
    ex_inf   = [c for c in alpha_panel.columns if np.isinf(alpha_panel[c]).any()]

    excluded = set(ex_nan) | set(ex_const) | set(ex_inf)
    surviving = [c for c in alpha_panel.columns if c not in excluded]

    return surviving, {
        'nan_gt_40pct': ex_nan,
        'constant':     ex_const,
        'has_inf':      ex_inf,
    }


def redundancy_pruning(
    alpha_panel: pd.DataFrame,
    surviving: List[str],
    corr_threshold: float = 0.85,
    reference_ticker: str | None = None,
    adf_scores: pd.Series | None = None,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Remove redundant alphas (|corr| > corr_threshold).
    Between a correlated pair, keep the one with lower ADF p-value (more stationary).

    Returns (post_prune_list, pruned_pairs).
    """
    # Compute correlation on one ticker (representative cross-section)
    if reference_ticker and isinstance(alpha_panel.index, pd.MultiIndex):
        try:
            ref_data = alpha_panel.xs(reference_ticker, level='ticker')[surviving]
        except KeyError:
            ref_data = alpha_panel[surviving]
    else:
        ref_data = alpha_panel[surviving]

    corr_matrix = ref_data.corr()

    if adf_scores is None:
        adf_scores = pd.Series(1.0, index=surviving)

    pruned_out   = set()
    pruned_pairs = []

    for i, ai in enumerate(surviving):
        if ai in pruned_out:
            continue
        for aj in surviving[i + 1:]:
            if aj in pruned_out:
                continue
            if abs(corr_matrix.loc[ai, aj]) > corr_threshold:
                score_i = adf_scores.get(ai, 1.0)
                score_j = adf_scores.get(aj, 1.0)
                if score_i <= score_j:
                    pruned_out.add(aj)
                    pruned_pairs.append((ai, aj, float(corr_matrix.loc[ai, aj])))
                else:
                    pruned_out.add(ai)
                    pruned_pairs.append((aj, ai, float(corr_matrix.loc[ai, aj])))

    post_prune = [a for a in surviving if a not in pruned_out]
    return post_prune, pruned_pairs


def apply_budget(
    post_prune: List[str],
    adf_scores: pd.Series,
    budget: int = 33,
) -> List[str]:
    """Cap the selected set to `budget` by stationarity (lowest ADF p-value first)."""
    if len(post_prune) <= budget:
        return post_prune
    scores = adf_scores.reindex(post_prune).fillna(1.0)
    return scores.nsmallest(budget).index.tolist()


def run_full_diagnostics(
    alpha_panel: pd.DataFrame,
    reference_ticker: str = 'NVDA',
    max_nan_pct: float = 40.0,
    min_std: float = 1e-8,
    corr_threshold: float = 0.85,
    budget: int = 33,
    diag_df: pd.DataFrame | None = None,
) -> Dict:
    """
    Full pipeline: exclusion → pruning → budget → selected set.

    Parameters
    ----------
    alpha_panel : MultiIndex (Date, ticker) DataFrame of alpha values
    reference_ticker : ticker used for the correlation matrix
    diag_df : optional pre-computed diagnostics DataFrame (with 'adf_pval_median' column)

    Returns dict with keys:
        surviving, post_prune, selected, exclusion_log, pruned_pairs,
        max_corr_selected, adf_scores
    """
    surviving, excl_log = apply_exclusion_rules(alpha_panel, max_nan_pct, min_std)

    # ADF scores
    if diag_df is not None and 'adf_pval_median' in diag_df.columns:
        adf_scores = diag_df['adf_pval_median'].reindex(surviving).fillna(1.0)
    else:
        adf_scores = pd.Series(1.0, index=surviving)

    post_prune, pruned_pairs = redundancy_pruning(
        alpha_panel, surviving,
        corr_threshold=corr_threshold,
        reference_ticker=reference_ticker,
        adf_scores=adf_scores,
    )

    selected = apply_budget(post_prune, adf_scores, budget)

    # Verify max corr in selected set
    if reference_ticker and isinstance(alpha_panel.index, pd.MultiIndex):
        try:
            ref_data = alpha_panel.xs(reference_ticker, level='ticker')[selected]
        except KeyError:
            ref_data = alpha_panel[selected]
    else:
        ref_data = alpha_panel[selected]

    corr_arr = ref_data.corr().to_numpy().copy()
    np.fill_diagonal(corr_arr, 0)
    max_corr = float(np.nanmax(np.abs(corr_arr)))

    return {
        'surviving':          surviving,
        'post_prune':         post_prune,
        'selected':           selected,
        'exclusion_log':      excl_log,
        'pruned_pairs':       pruned_pairs,
        'max_corr_selected':  max_corr,
        'adf_scores':         adf_scores,
        'n_surviving':        len(surviving),
        'n_post_prune':       len(post_prune),
        'n_selected':         len(selected),
    }


def save_selected_alphas(result: Dict, path: str = 'configs/selected_alphas.json') -> None:
    cfg = {
        'selected_alphas':    result['selected'],
        'n_alphas':           len(result['selected']),
        'excluded_nan40':     result['exclusion_log']['nan_gt_40pct'],
        'excluded_constant':  result['exclusion_log']['constant'],
        'excluded_inf':       result['exclusion_log']['has_inf'],
        'pruned_redundant':   [p[1] for p in result['pruned_pairs']],
        'budget':             33,
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
