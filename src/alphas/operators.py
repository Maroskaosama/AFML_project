"""
Alpha operators from WorldQuant 101 Formulaic Alphas paper (Appendix A.2).

All functions are pure (no side effects, no in-place mutation of inputs).
Rolling windows use min_periods = window size — no partial-window values.
Cross-sectional operators work across the 10 stocks (axis=1 on wide DataFrames).
"""
import numpy as np
import pandas as pd


# ── Time-series operators ────────────────────────────────────────────────────

def ts_sum(x, d):
    d = int(d)
    return x.rolling(d, min_periods=d).sum()


def ts_mean(x, d):
    d = int(d)
    return x.rolling(d, min_periods=d).mean()


def ts_std(x, d):
    d = int(d)
    return x.rolling(d, min_periods=d).std()


def ts_min(x, d):
    d = int(d)
    return x.rolling(d, min_periods=d).min()


def ts_max(x, d):
    d = int(d)
    return x.rolling(d, min_periods=d).max()


def ts_rank(x, d):
    """Percentile rank of x[t] within the past d values (in [1/d, 1])."""
    d = int(d)

    def _rank_last(w):
        # w is chronological (oldest first); rank the last element
        s = pd.Series(w)
        return s.rank(pct=True).iloc[-1]

    if isinstance(x, pd.DataFrame):
        return x.rolling(d, min_periods=d).apply(_rank_last, raw=False)
    return x.rolling(d, min_periods=d).apply(_rank_last, raw=False)


def ts_argmax(x, d):
    """0-based index of the maximum value within the past d days (0 = oldest)."""
    d = int(d)
    return x.rolling(d, min_periods=d).apply(np.argmax, raw=True)


def ts_argmin(x, d):
    """0-based index of the minimum value within the past d days (0 = oldest)."""
    d = int(d)
    return x.rolling(d, min_periods=d).apply(np.argmin, raw=True)


def ts_corr(x, y, d):
    d = int(d)
    if isinstance(x, pd.DataFrame):
        # Column-wise rolling correlation
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            result[col] = x[col].rolling(d, min_periods=d).corr(y[col])
        return result
    return x.rolling(d, min_periods=d).corr(y)


def ts_cov(x, y, d):
    d = int(d)
    if isinstance(x, pd.DataFrame):
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            result[col] = x[col].rolling(d, min_periods=d).cov(y[col])
        return result
    return x.rolling(d, min_periods=d).cov(y)


def ts_product(x, d):
    d = int(d)
    return x.rolling(d, min_periods=d).apply(np.prod, raw=True)


def delta(x, d):
    """x[t] - x[t-d]"""
    d = int(d)
    return x - x.shift(d)


def delay(x, d):
    """x[t-d]"""
    d = int(d)
    return x.shift(d)


def decay_linear(x, d):
    """
    Linearly decaying WMA: x[t] has weight d, x[t-1] has weight d-1, ..., x[t-d+1] has weight 1.
    Normalized by d*(d+1)/2.
    """
    d = int(d)
    # Weights in chronological order (oldest to newest): [1, 2, ..., d]
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()

    def _wma(w):
        return np.dot(w, weights)

    return x.rolling(d, min_periods=d).apply(_wma, raw=True)


def signed_power(x, a):
    """sign(x) * |x|^a"""
    return np.sign(x) * (np.abs(x) ** a)


def log(x):
    return np.log(x)


def abs_val(x):
    return np.abs(x)


def sign(x):
    return np.sign(x)


# ── Cross-sectional operators ────────────────────────────────────────────────

def rank_cs(df):
    """
    Cross-sectional percentile rank across all tickers at each date.
    Returns DataFrame in [1/n, 1] where n = number of tickers.
    """
    if isinstance(df, pd.Series):
        # Single row, nothing to rank cross-sectionally
        return df.rank(pct=True)
    return df.rank(axis=1, pct=True)


def scale_cs(df, a=1):
    """
    Rescale each row so sum(|x|) = a.
    """
    row_sum = df.abs().sum(axis=1)
    # Avoid division by zero
    row_sum = row_sum.replace(0, np.nan)
    return df.div(row_sum, axis=0) * a


def indneutralize_cs(df, sector_map):
    """
    Demean within sector groups at each date.
    sector_map: {ticker: sector_label}

    For sectors with only 1 stock, use full-universe mean as fallback.
    For stocks not in sector_map, treat as their own sector (no demeaning).
    """
    result = df.copy()
    cols = [c for c in df.columns if c in sector_map]

    # Group tickers by sector
    sector_groups = {}
    for ticker, sector in sector_map.items():
        if ticker in df.columns:
            sector_groups.setdefault(sector, []).append(ticker)

    for sector, members in sector_groups.items():
        if len(members) > 1:
            group_mean = df[members].mean(axis=1)
            for m in members:
                result[m] = df[m] - group_mean
        else:
            # Singleton: subtract full-universe mean
            universe_mean = df[cols].mean(axis=1)
            result[members[0]] = df[members[0]] - universe_mean

    return result


# ── Dollar-volume helper ─────────────────────────────────────────────────────

def adv(close, volume, d):
    """
    Average daily DOLLAR volume over past d days.
    adv{d} = sma(close * volume, d)  [NOT sma(volume, d) — common bug]
    """
    d = int(d)
    return (close * volume).rolling(d, min_periods=d).mean()


# ── Utility wrappers ─────────────────────────────────────────────────────────

def where(condition, x, y):
    """Element-wise: x where condition is True, else y."""
    if isinstance(condition, pd.DataFrame):
        return x.where(condition, y)
    # Series
    return pd.Series(np.where(condition, x, y), index=condition.index)


def minimum(x, y):
    """Element-wise minimum of two DataFrames/Series."""
    if isinstance(x, pd.DataFrame):
        return x.combine(y, np.minimum)
    return np.minimum(x, y)


def maximum(x, y):
    """Element-wise maximum of two DataFrames/Series."""
    if isinstance(x, pd.DataFrame):
        return x.combine(y, np.maximum)
    return np.maximum(x, y)
