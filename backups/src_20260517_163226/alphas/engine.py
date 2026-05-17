"""
Master alpha computation engine.
Builds the data dict from panel_ohlcv.parquet, calls all alpha functions,
stacks results into a MultiIndex (Date, ticker) DataFrame.
"""
import numpy as np
import pandas as pd

from src.alphas.operators import adv as _adv
from src.alphas.registry import SECTOR_MAP


def build_data_dict(panel_ohlcv: pd.DataFrame) -> dict:
    """
    Pivot panel (MultiIndex Date×ticker) to wide dicts ready for alpha functions.

    All DataFrames have shape (dates × tickers).
    Pre-computes adv variants used across multiple alphas.
    """
    close  = panel_ohlcv['Close'].unstack('ticker')
    open_  = panel_ohlcv['Open'].unstack('ticker')
    high   = panel_ohlcv['High'].unstack('ticker')
    low    = panel_ohlcv['Low'].unstack('ticker')
    volume = panel_ohlcv['Volume'].unstack('ticker')

    # Use AdjClose if present, else Close (panel stores Close = AdjClose after our cleaning)
    if 'AdjClose' in panel_ohlcv.columns:
        adj = panel_ohlcv['AdjClose'].unstack('ticker')
    else:
        adj = close

    returns = adj.pct_change()
    vwap    = (high + low + close) / 3

    adv_periods = [5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180]

    data = {
        'close':   close,
        'open':    open_,
        'high':    high,
        'low':     low,
        'volume':  volume,
        'returns': returns,
        'vwap':    vwap,
    }

    for d in adv_periods:
        data[f'adv{d}'] = _adv(close, volume, d)

    return data


def compute_all_alphas(
    panel_ohlcv: pd.DataFrame,
    sector_map: dict = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute all 101 alpha formulas on a 10-stock panel.

    Parameters
    ----------
    panel_ohlcv : DataFrame with MultiIndex (Date, ticker),
                  columns [Open, High, Low, Close, Volume, AdjClose?]
    sector_map  : {ticker: sector} — defaults to SECTOR_MAP in registry.py

    Returns
    -------
    DataFrame with MultiIndex (Date, ticker), one column per alpha.
    """
    from src.alphas.formulas import get_all_alpha_functions

    if sector_map is None:
        sector_map = SECTOR_MAP

    data = build_data_dict(panel_ohlcv)
    alpha_funcs = get_all_alpha_functions()

    results = {}
    n_ok = 0
    n_fail = 0

    for name, func in alpha_funcs:
        try:
            df = func(data, sector_map)

            # Sanitise
            if isinstance(df, pd.DataFrame):
                df = df.replace([np.inf, -np.inf], np.nan)
                # Clip extreme outliers
                df = df.where(df.abs() <= 1e6, np.nan)
            else:
                # Some alphas may return a boolean DataFrame — cast to float
                df = df.astype(float).replace([np.inf, -np.inf], np.nan)
                df = df.where(df.abs() <= 1e6, np.nan)

            nan_pct = df.isnull().sum().sum() / df.size * 100
            if verbose:
                print(f"  OK  {name}: {nan_pct:.1f}% NaN")
            results[name] = df
            n_ok += 1

        except Exception as e:
            if verbose:
                print(f"  ERR {name}: {e}")
            n_fail += 1

    if verbose:
        print(f"\nAlpha engine: {n_ok} OK, {n_fail} failed out of {len(alpha_funcs)} total.")

    # Stack each wide DataFrame to Series, combine
    stacked = {}
    for name, df in results.items():
        stacked[name] = df.stack()

    alpha_panel = pd.DataFrame(stacked)
    alpha_panel.index.names = ['Date', 'ticker']

    return alpha_panel


def compute_alpha_diagnostics(alpha_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-alpha diagnostic statistics for QA and feature selection.

    Returns DataFrame indexed by alpha name with columns:
    nan_pct, mean, std, min, max, skew, kurt, autocorr_lag1,
    any_inf, n_unique_min (minimum unique values across tickers).
    """
    from scipy import stats as spstats
    from statsmodels.tsa.stattools import adfuller

    rows = []
    tickers = alpha_panel.index.get_level_values('ticker').unique()

    for col in alpha_panel.columns:
        s = alpha_panel[col]
        n_total = len(s)
        n_nan = s.isnull().sum()

        finite = s.dropna()
        if len(finite) == 0:
            rows.append({'alpha': col, 'nan_pct': 100.0,
                         'mean': np.nan, 'std': np.nan, 'min': np.nan,
                         'max': np.nan, 'skew': np.nan, 'kurt': np.nan,
                         'autocorr_lag1': np.nan, 'any_inf': False,
                         'n_unique_min': 0, 'adf_pval_median': np.nan})
            continue

        nan_pct = n_nan / n_total * 100
        any_inf = np.isinf(s).any()

        # ADF per ticker — use median p-value
        adf_pvals = []
        autocorrs = []
        n_unique_vals = []

        for tk in tickers:
            ts = s.xs(tk, level='ticker').dropna()
            if len(ts) < 20:
                continue
            try:
                adf_pvals.append(adfuller(ts, maxlag=1, autolag=None)[1])
            except Exception:
                pass
            autocorrs.append(ts.autocorr(lag=1))
            n_unique_vals.append(ts.nunique())

        rows.append({
            'alpha':          col,
            'nan_pct':        nan_pct,
            'mean':           float(finite.mean()),
            'std':            float(finite.std()),
            'min':            float(finite.min()),
            'max':            float(finite.max()),
            'skew':           float(spstats.skew(finite)),
            'kurt':           float(spstats.kurtosis(finite)),
            'autocorr_lag1':  float(np.nanmean(autocorrs)) if autocorrs else np.nan,
            'any_inf':        bool(any_inf),
            'n_unique_min':   int(min(n_unique_vals)) if n_unique_vals else 0,
            'adf_pval_median': float(np.median(adf_pvals)) if adf_pvals else np.nan,
        })

    return pd.DataFrame(rows).set_index('alpha')
