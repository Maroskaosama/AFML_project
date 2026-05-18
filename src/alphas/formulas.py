"""
101 Formulaic Alphas — Kakushadze (2015) / WorldQuant.

Each alpha function:
  - Takes data: dict and sector_map: dict
  - Returns a wide-format DataFrame (dates × tickers)
  - Clips |values| > 1e6 to NaN and replaces ±inf with NaN

data keys: close, open, high, low, volume, returns, vwap,
           adv5, adv10, adv15, adv20, adv30, adv40, adv50,
           adv60, adv81, adv120, adv150, adv180

Operator naming conventions (paper → this file):
  correlation()  → ts_corr()
  covariance()   → ts_cov()
  stddev()       → ts_std()
  sum()          → ts_sum()
  product()      → ts_product()
  min()/max()    → minimum()/maximum()  (element-wise)
  rank()         → rank_cs()            (cross-sectional)
  scale()        → scale_cs()
  indneutralize()→ indneutralize_cs()
  Ts_Rank()      → ts_rank()
  log()          → np.log()
  abs()          → abs_val()
  sign()         → sign() / np.sign()
"""

import numpy as np
import pandas as pd
from src.alphas.operators import (
    ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_rank, ts_argmax, ts_argmin,
    ts_corr, ts_cov, ts_product, delta, delay, decay_linear, signed_power,
    rank_cs, scale_cs, indneutralize_cs, adv, where, minimum, maximum,
    log, abs_val, sign,
)


def _clean(result):
    """Replace ±inf with NaN and clip |values| > 1e6 to NaN."""
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.where(result.abs() <= 1e6, np.nan)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#1
# ─────────────────────────────────────────────────────────────────────────────

def alpha001(data, sector_map):
    """Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    close = data['close'].copy()
    returns = data['returns']
    inner = close.copy()
    cond = returns < 0
    inner[cond] = ts_std(returns, 20)[cond]
    result = rank_cs(ts_argmax(signed_power(inner, 2.0), 5)) - 0.5
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#2
# ─────────────────────────────────────────────────────────────────────────────

def alpha002(data, sector_map):
    """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    close = data['close']
    open_ = data['open']
    volume = data['volume']
    x = rank_cs(delta(np.log(volume), 2))
    y = rank_cs((close - open_) / (open_ + 1e-8))
    result = -1 * ts_corr(x, y, 6)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#3
# ─────────────────────────────────────────────────────────────────────────────

def alpha003(data, sector_map):
    """Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))"""
    open_ = data['open']
    volume = data['volume']
    result = -1 * ts_corr(rank_cs(open_), rank_cs(volume), 10)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#4
# ─────────────────────────────────────────────────────────────────────────────

def alpha004(data, sector_map):
    """Alpha#4: (-1 * Ts_Rank(rank(low), 9))"""
    low = data['low']
    result = -1 * ts_rank(rank_cs(low), 9)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#5
# ─────────────────────────────────────────────────────────────────────────────

def alpha005(data, sector_map):
    """Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    close = data['close']
    open_ = data['open']
    vwap = data['vwap']
    result = rank_cs(open_ - ts_sum(vwap, 10) / 10) * (-1 * abs_val(rank_cs(close - vwap)))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#6
# ─────────────────────────────────────────────────────────────────────────────

def alpha006(data, sector_map):
    """Alpha#6: (-1 * correlation(open, volume, 10))"""
    open_ = data['open']
    volume = data['volume']
    result = -1 * ts_corr(open_, volume, 10)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#7
# ─────────────────────────────────────────────────────────────────────────────

def alpha007(data, sector_map):
    """Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))

    Note: 'volume' in the original formula refers to dollar volume (close*vol),
    not share volume.  adv20 is also dollar volume, so compare like to like.
    """
    close = data['close']
    volume = data['volume']
    adv20 = data['adv20']
    # Today's dollar volume vs 20-day average dollar volume
    dollar_vol_today = close * volume
    cond = adv20 < dollar_vol_today
    branch_true = (-1 * ts_rank(abs_val(delta(close, 7)), 60)) * sign(delta(close, 7))
    branch_false = pd.DataFrame(-1.0, index=close.index, columns=close.columns)
    result = branch_true.where(cond, branch_false)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#8
# ─────────────────────────────────────────────────────────────────────────────

def alpha008(data, sector_map):
    """Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    open_ = data['open']
    returns = data['returns']
    inner = ts_sum(open_, 5) * ts_sum(returns, 5)
    result = -1 * rank_cs(inner - delay(inner, 10))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#9
# ─────────────────────────────────────────────────────────────────────────────

def alpha009(data, sector_map):
    """Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
    close = data['close']
    d = delta(close, 1)
    cond1 = ts_min(d, 5) > 0
    cond2 = ts_max(d, 5) < 0
    # innermost branch
    inner = d.where(cond2, -1 * d)
    result = d.where(cond1, inner)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#10
# ─────────────────────────────────────────────────────────────────────────────

def alpha010(data, sector_map):
    """Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
    close = data['close']
    d = delta(close, 1)
    cond1 = ts_min(d, 4) > 0
    cond2 = ts_max(d, 4) < 0
    inner = d.where(cond2, -1 * d)
    result = rank_cs(d.where(cond1, inner))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#11
# ─────────────────────────────────────────────────────────────────────────────

def alpha011(data, sector_map):
    """Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    close = data['close']
    vwap = data['vwap']
    volume = data['volume']
    spread = vwap - close
    result = (rank_cs(ts_max(spread, 3)) + rank_cs(ts_min(spread, 3))) * rank_cs(delta(volume, 3))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#12
# ─────────────────────────────────────────────────────────────────────────────

def alpha012(data, sector_map):
    """Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    close = data['close']
    volume = data['volume']
    result = sign(delta(volume, 1)) * (-1 * delta(close, 1))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#13
# ─────────────────────────────────────────────────────────────────────────────

def alpha013(data, sector_map):
    """Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    close = data['close']
    volume = data['volume']
    result = -1 * rank_cs(ts_cov(rank_cs(close), rank_cs(volume), 5))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#14
# ─────────────────────────────────────────────────────────────────────────────

def alpha014(data, sector_map):
    """Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    open_ = data['open']
    volume = data['volume']
    returns = data['returns']
    result = (-1 * rank_cs(delta(returns, 3))) * ts_corr(open_, volume, 10)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#15
# ─────────────────────────────────────────────────────────────────────────────

def alpha015(data, sector_map):
    """Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    high = data['high']
    volume = data['volume']
    result = -1 * ts_sum(rank_cs(ts_corr(rank_cs(high), rank_cs(volume), 3)), 3)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#16
# ─────────────────────────────────────────────────────────────────────────────

def alpha016(data, sector_map):
    """Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    high = data['high']
    volume = data['volume']
    result = -1 * rank_cs(ts_cov(rank_cs(high), rank_cs(volume), 5))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#17
# ─────────────────────────────────────────────────────────────────────────────

def alpha017(data, sector_map):
    """Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
    close = data['close']
    volume = data['volume']
    adv20 = data['adv20']
    result = (
        (-1 * rank_cs(ts_rank(close, 10)))
        * rank_cs(delta(delta(close, 1), 1))
        * rank_cs(ts_rank(volume / (adv20 + 1e-8), 5))
    )
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#18
# ─────────────────────────────────────────────────────────────────────────────

def alpha018(data, sector_map):
    """Alpha#18: (-1 * rank((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))"""
    close = data['close']
    open_ = data['open']
    result = -1 * rank_cs(
        ts_std(abs_val(close - open_), 5) + (close - open_) + ts_corr(close, open_, 10)
    )
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#19
# ─────────────────────────────────────────────────────────────────────────────

def alpha019(data, sector_map):
    """Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    close = data['close']
    returns = data['returns']
    # close - delay(close,7) == delta(close,7), so inner = 2*delta(close,7)
    inner = (close - delay(close, 7)) + delta(close, 7)
    result = (-1 * sign(inner)) * (1 + rank_cs(1 + ts_sum(returns, 250)))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#20
# ─────────────────────────────────────────────────────────────────────────────

def alpha020(data, sector_map):
    """Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
    open_ = data['open']
    high = data['high']
    close = data['close']
    low = data['low']
    result = (
        (-1 * rank_cs(open_ - delay(high, 1)))
        * rank_cs(open_ - delay(close, 1))
        * rank_cs(open_ - delay(low, 1))
    )
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#21
# ─────────────────────────────────────────────────────────────────────────────

def alpha021(data, sector_map):
    """Alpha#21: (((sum(close,8)/8 + stddev(close,8)) < sum(close,2)/2) ? -1 : (((sum(close,2)/2) < (sum(close,8)/8 - stddev(close,8))) ? 1 : (((1 < vol/adv20) | (vol/adv20==1)) ? 1 : -1)))"""
    close = data['close']
    volume = data['volume']
    adv20 = data['adv20']
    sma8 = ts_sum(close, 8) / 8
    std8 = ts_std(close, 8)
    sma2 = ts_sum(close, 2) / 2
    vol_ratio = volume / (adv20 + 1e-8)
    cond1 = (sma8 + std8) < sma2
    cond2 = sma2 < (sma8 - std8)
    cond3 = (vol_ratio >= 1)
    ones = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    neg_ones = pd.DataFrame(-1.0, index=close.index, columns=close.columns)
    # innermost: cond3 → 1, else -1
    branch3 = ones.where(cond3, neg_ones)
    # middle: cond2 → 1, else branch3
    branch2 = ones.where(cond2, branch3)
    # outer: cond1 → -1, else branch2
    result = neg_ones.where(cond1, branch2)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#22
# ─────────────────────────────────────────────────────────────────────────────

def alpha022(data, sector_map):
    """Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
    high = data['high']
    volume = data['volume']
    close = data['close']
    result = -1 * delta(ts_corr(high, volume, 5), 5) * rank_cs(ts_std(close, 20))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#23
# ─────────────────────────────────────────────────────────────────────────────

def alpha023(data, sector_map):
    """Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
    high = data['high']
    cond = ts_sum(high, 20) / 20 < high
    branch_true = -1 * delta(high, 2)
    branch_false = pd.DataFrame(0.0, index=high.index, columns=high.columns)
    result = branch_true.where(cond, branch_false)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#24
# ─────────────────────────────────────────────────────────────────────────────

def alpha024(data, sector_map):
    """Alpha#24: (((delta(sum(close,100)/100, 100) / delay(close,100)) <= 0.05) ? (-1*(close - ts_min(close,100))) : (-1*delta(close,3)))"""
    close = data['close']
    ratio = delta(ts_sum(close, 100) / 100, 100) / (delay(close, 100) + 1e-8)
    cond = ratio <= 0.05
    branch_true = -1 * (close - ts_min(close, 100))
    branch_false = -1 * delta(close, 3)
    result = branch_true.where(cond, branch_false)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#25
# ─────────────────────────────────────────────────────────────────────────────

def alpha025(data, sector_map):
    """Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
    returns = data['returns']
    adv20 = data['adv20']
    vwap = data['vwap']
    high = data['high']
    close = data['close']
    result = rank_cs((-1 * returns) * adv20 * vwap * (high - close))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#26
# ─────────────────────────────────────────────────────────────────────────────

def alpha026(data, sector_map):
    """Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    high = data['high']
    volume = data['volume']
    result = -1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#27
# ─────────────────────────────────────────────────────────────────────────────

def alpha027(data, sector_map):
    """Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1) : 1)"""
    volume = data['volume']
    vwap = data['vwap']
    inner = ts_sum(ts_corr(rank_cs(volume), rank_cs(vwap), 6), 2) / 2.0
    cond = rank_cs(inner) > 0.5
    ones = pd.DataFrame(1.0, index=vwap.index, columns=vwap.columns)
    result = (-1 * ones).where(cond, ones)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#28
# ─────────────────────────────────────────────────────────────────────────────

def alpha028(data, sector_map):
    """Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    close = data['close']
    high = data['high']
    low = data['low']
    adv20 = data['adv20']
    result = scale_cs(ts_corr(adv20, low, 5) + (high + low) / 2 - close)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#29
# ─────────────────────────────────────────────────────────────────────────────

def alpha029(data, sector_map):
    """Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1*rank(delta((close-1),5))))),2),1))))),1),5) + ts_rank(delay((-1*returns),6),5))"""
    close = data['close']
    returns = data['returns']
    inner1 = -1 * rank_cs(delta(close - 1, 5))
    inner2 = rank_cs(rank_cs(inner1))
    inner3 = ts_min(inner2, 2)
    inner4 = ts_sum(inner3, 1)
    inner5 = scale_cs(np.log(inner4 + 1e-8))
    inner6 = rank_cs(rank_cs(inner5))
    inner7 = ts_product(inner6, 1)
    lhs = ts_min(inner7, 5)
    rhs = ts_rank(delay(-1 * returns, 6), 5)
    result = lhs + rhs
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#30
# ─────────────────────────────────────────────────────────────────────────────

def alpha030(data, sector_map):
    """Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))"""
    close = data['close']
    volume = data['volume']
    s = (
        sign(close - delay(close, 1))
        + sign(delay(close, 1) - delay(close, 2))
        + sign(delay(close, 2) - delay(close, 3))
    )
    result = ((1.0 - rank_cs(s)) * ts_sum(volume, 5)) / (ts_sum(volume, 20) + 1e-8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#31
# ─────────────────────────────────────────────────────────────────────────────

def alpha031(data, sector_map):
    """Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))"""
    close = data['close']
    low = data['low']
    adv20 = data['adv20']
    part1 = rank_cs(rank_cs(rank_cs(decay_linear(-1 * rank_cs(rank_cs(delta(close, 10))), 10))))
    part2 = rank_cs(-1 * delta(close, 3))
    part3 = sign(scale_cs(ts_corr(adv20, low, 12)))
    result = part1 + part2 + part3
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#32
# ─────────────────────────────────────────────────────────────────────────────

def alpha032(data, sector_map):
    """Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
    close = data['close']
    vwap = data['vwap']
    part1 = scale_cs(ts_sum(close, 7) / 7 - close)
    part2 = 20 * scale_cs(ts_corr(vwap, delay(close, 5), 230))
    result = part1 + part2
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#33
# ─────────────────────────────────────────────────────────────────────────────

def alpha033(data, sector_map):
    """Alpha#33: rank((-1 * ((1 - (open / close))**1)))"""
    close = data['close']
    open_ = data['open']
    result = rank_cs(-1 * (1 - open_ / (close + 1e-8)))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#34
# ─────────────────────────────────────────────────────────────────────────────

def alpha034(data, sector_map):
    """Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    close = data['close']
    returns = data['returns']
    ratio = ts_std(returns, 2) / (ts_std(returns, 5) + 1e-8)
    result = rank_cs((1 - rank_cs(ratio)) + (1 - rank_cs(delta(close, 1))))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#35
# ─────────────────────────────────────────────────────────────────────────────

def alpha035(data, sector_map):
    """Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    returns = data['returns']
    result = (
        ts_rank(volume, 32)
        * (1 - ts_rank((close + high) - low, 16))
        * (1 - ts_rank(returns, 32))
    )
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#36
# ─────────────────────────────────────────────────────────────────────────────

def alpha036(data, sector_map):
    """Alpha#36: (((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open))))"""
    close = data['close']
    open_ = data['open']
    volume = data['volume']
    vwap = data['vwap']
    returns = data['returns']
    adv20 = data['adv20']
    part1 = 2.21 * rank_cs(ts_corr(close - open_, delay(volume, 1), 15))
    part2 = 0.7 * rank_cs(open_ - close)
    part3 = 0.73 * rank_cs(ts_rank(delay(-1 * returns, 6), 5))
    part4 = rank_cs(abs_val(ts_corr(vwap, adv20, 6)))
    part5 = 0.6 * rank_cs((ts_sum(close, 200) / 200 - open_) * (close - open_))
    result = part1 + part2 + part3 + part4 + part5
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#37
# ─────────────────────────────────────────────────────────────────────────────

def alpha037(data, sector_map):
    """Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
    close = data['close']
    open_ = data['open']
    result = (
        rank_cs(ts_corr(delay(open_ - close, 1), close, 200))
        + rank_cs(open_ - close)
    )
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#38
# ─────────────────────────────────────────────────────────────────────────────

def alpha038(data, sector_map):
    """Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"""
    close = data['close']
    open_ = data['open']
    result = (-1 * rank_cs(ts_rank(close, 10))) * rank_cs(close / (open_ + 1e-8))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#39
# ─────────────────────────────────────────────────────────────────────────────

def alpha039(data, sector_map):
    """Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
    close = data['close']
    volume = data['volume']
    returns = data['returns']
    adv20 = data['adv20']
    part1 = -1 * rank_cs(delta(close, 7) * (1 - rank_cs(decay_linear(volume / (adv20 + 1e-8), 9))))
    part2 = 1 + rank_cs(ts_sum(returns, 250))
    result = part1 * part2
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#40
# ─────────────────────────────────────────────────────────────────────────────

def alpha040(data, sector_map):
    """Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))"""
    high = data['high']
    volume = data['volume']
    result = (-1 * rank_cs(ts_std(high, 10))) * ts_corr(high, volume, 10)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#41
# ─────────────────────────────────────────────────────────────────────────────

def alpha041(data, sector_map):
    """Alpha#41: (((high * low)**0.5) - vwap)"""
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    result = ((high * low) ** 0.5) - vwap
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#42
# ─────────────────────────────────────────────────────────────────────────────

def alpha042(data, sector_map):
    """Alpha#42: (rank((vwap - close)) / rank((vwap + close)))"""
    close = data['close']
    vwap = data['vwap']
    result = rank_cs(vwap - close) / (rank_cs(vwap + close) + 1e-8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#43
# ─────────────────────────────────────────────────────────────────────────────

def alpha043(data, sector_map):
    """Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
    close = data['close']
    volume = data['volume']
    adv20 = data['adv20']
    result = ts_rank(volume / (adv20 + 1e-8), 20) * ts_rank(-1 * delta(close, 7), 8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#44
# ─────────────────────────────────────────────────────────────────────────────

def alpha044(data, sector_map):
    """Alpha#44: (-1 * correlation(high, rank(volume), 5))"""
    high = data['high']
    volume = data['volume']
    result = -1 * ts_corr(high, rank_cs(volume), 5)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#45
# ─────────────────────────────────────────────────────────────────────────────

def alpha045(data, sector_map):
    """Alpha#45: (-1 * ((rank((sum(delay(returns, 5), 20) / 20)) * correlation(returns, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
    close = data['close']
    returns = data['returns']
    volume = data['volume']
    part1 = rank_cs(ts_sum(delay(returns, 5), 20) / 20)
    part2 = ts_corr(returns, volume, 2)
    part3 = rank_cs(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))
    result = -1 * (part1 * part2) * part3
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#46
# ─────────────────────────────────────────────────────────────────────────────

def alpha046(data, sector_map):
    """Alpha#46: ((0.25 < diff) ? -1 : ((diff < 0) ? 1 : (-1*(close - delay(close,1))))) where diff = (delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10"""
    close = data['close']
    diff = (delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10
    cond1 = diff > 0.25
    cond2 = diff < 0
    branch3 = -1 * (close - delay(close, 1))
    branch2 = pd.DataFrame(1.0, index=close.index, columns=close.columns).where(cond2, branch3)
    result = pd.DataFrame(-1.0, index=close.index, columns=close.columns).where(cond1, branch2)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#47
# ─────────────────────────────────────────────────────────────────────────────

def alpha047(data, sector_map):
    """Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))"""
    close = data['close']
    high = data['high']
    volume = data['volume']
    vwap = data['vwap']
    adv20 = data['adv20']
    part1 = (rank_cs(1.0 / (close + 1e-8)) * volume) / (adv20 + 1e-8)
    part2 = (high * rank_cs(high - close)) / (ts_sum(high, 5) / 5 + 1e-8)
    result = part1 * part2 - rank_cs(vwap - delay(vwap, 5))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#48
# ─────────────────────────────────────────────────────────────────────────────

def alpha048(data, sector_map):
    """Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), sector_map) - ((delta(close, 1) / close) * indneutralize(correlation(delta(close, 1), delta(delay(close, 1), 1), 250), sector_map)))"""
    close = data['close']
    dc1 = delta(close, 1)
    dc1_lag = delta(delay(close, 1), 1)
    corr250 = ts_corr(dc1, dc1_lag, 250)
    lhs = indneutralize_cs(corr250 * dc1 / (close + 1e-8), sector_map)
    rhs = (dc1 / (close + 1e-8)) * indneutralize_cs(corr250, sector_map)
    result = lhs - rhs
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#49
# ─────────────────────────────────────────────────────────────────────────────

def alpha049(data, sector_map):
    """Alpha#49: (((diff < -0.1) ? 1 : (-1*(close - delay(close,1)))) where diff = (delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10"""
    close = data['close']
    diff = (delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10
    cond = diff < -0.1
    branch_false = -1 * (close - delay(close, 1))
    result = pd.DataFrame(1.0, index=close.index, columns=close.columns).where(cond, branch_false)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#50
# ─────────────────────────────────────────────────────────────────────────────

def alpha050(data, sector_map):
    """Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    volume = data['volume']
    vwap = data['vwap']
    result = -1 * ts_max(rank_cs(ts_corr(rank_cs(volume), rank_cs(vwap), 5)), 5)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#51
# ─────────────────────────────────────────────────────────────────────────────

def alpha051(data, sector_map):
    """Alpha#51: (((diff < -0.05) ? 1 : (-1*(close - delay(close,1)))) where diff = (delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10"""
    close = data['close']
    diff = (delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10
    cond = diff < -0.05
    branch_false = -1 * (close - delay(close, 1))
    result = pd.DataFrame(1.0, index=close.index, columns=close.columns).where(cond, branch_false)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#52
# ─────────────────────────────────────────────────────────────────────────────

def alpha052(data, sector_map):
    """Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
    low = data['low']
    returns = data['returns']
    volume = data['volume']
    part1 = (-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)
    part2 = rank_cs((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220)
    part3 = ts_rank(volume, 5)
    result = part1 * part2 * part3
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#53
# ─────────────────────────────────────────────────────────────────────────────

def alpha053(data, sector_map):
    """Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low + 1e-8)), 9))"""
    close = data['close']
    high = data['high']
    low = data['low']
    inner = ((close - low) - (high - close)) / (close - low + 1e-8)
    result = -1 * delta(inner, 9)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#54
# ─────────────────────────────────────────────────────────────────────────────

def alpha054(data, sector_map):
    """Alpha#54: ((-1 * ((low - close) * (open**5))) / ((low - high) * (close**5) + 1e-8))"""
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']
    result = (-1 * (low - close) * (open_ ** 5)) / ((low - high) * (close ** 5) + 1e-8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#55
# ─────────────────────────────────────────────────────────────────────────────

def alpha055(data, sector_map):
    """Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 1e-8))), rank(volume), 6))"""
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    num = close - ts_min(low, 12)
    denom = ts_max(high, 12) - ts_min(low, 12) + 1e-8
    x = rank_cs(num / denom)
    result = -1 * ts_corr(x, rank_cs(volume), 6)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#56  — requires market cap, not implemented
# ─────────────────────────────────────────────────────────────────────────────

def alpha056(data, sector_map):
    """Alpha#56: NaN — requires market cap data, not implemented."""
    close = data['close']
    return pd.DataFrame(np.nan, index=close.index, columns=close.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#57
# ─────────────────────────────────────────────────────────────────────────────

def alpha057(data, sector_map):
    """Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
    close = data['close']
    vwap = data['vwap']
    denom = decay_linear(rank_cs(ts_argmax(close, 30)), 2)
    result = 0 - (close - vwap) / (denom + 1e-8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#58
# ─────────────────────────────────────────────────────────────────────────────

def alpha058(data, sector_map):
    """Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(indneutralize(vwap, sector_map), volume, 4), 8), 6))"""
    vwap = data['vwap']
    volume = data['volume']
    result = -1 * ts_rank(decay_linear(ts_corr(indneutralize_cs(vwap, sector_map), volume, 4), 8), 6)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#59
# ─────────────────────────────────────────────────────────────────────────────

def alpha059(data, sector_map):
    """Alpha#59: (-1 * Ts_Rank(decay_linear(correlation(indneutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), sector_map), volume, 4), 16), 8))"""
    vwap = data['vwap']
    volume = data['volume']
    # 0.728317 * vwap + (1-0.728317) * vwap = vwap
    blend = vwap * 0.728317 + vwap * (1 - 0.728317)
    result = -1 * ts_rank(decay_linear(ts_corr(indneutralize_cs(blend, sector_map), volume, 4), 16), 8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#60
# ─────────────────────────────────────────────────────────────────────────────

def alpha060(data, sector_map):
    """Alpha#60: (0 - (1 * ((2 * scale_cs(rank_cs(((((close - low) - (high - close)) / (high - low + 1e-8)) * volume)))) - scale_cs(rank_cs(ts_argmax(close, 10))))))"""
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    inner = ((close - low) - (high - close)) / (high - low + 1e-8) * volume
    result = 0 - (2 * scale_cs(rank_cs(inner)) - scale_cs(rank_cs(ts_argmax(close, 10))))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#61
# ─────────────────────────────────────────────────────────────────────────────

def alpha061(data, sector_map):
    """Alpha#61: (rank((vwap - ts_min(vwap, 16))) < rank(correlation(vwap, adv180, 18))).astype(float) * -1"""
    vwap = data['vwap']
    adv180 = data['adv180']
    cond = rank_cs(vwap - ts_min(vwap, 16)) < rank_cs(ts_corr(vwap, adv180, 18))
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#62
# ─────────────────────────────────────────────────────────────────────────────

def alpha062(data, sector_map):
    """Alpha#62: ((rank(correlation(vwap, sum(adv20, 22), 10)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)"""
    open_ = data['open']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    adv20 = data['adv20']
    lhs = rank_cs(ts_corr(vwap, ts_sum(adv20, 22), 10))
    rhs_inner = (rank_cs(open_) + rank_cs(open_)) < (rank_cs((high + low) / 2) + rank_cs(high))
    rhs = rank_cs(rhs_inner.astype(float))
    cond = lhs < rhs
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#63
# ─────────────────────────────────────────────────────────────────────────────

def alpha063(data, sector_map):
    """Alpha#63: (rank(decay_linear(delta(indneutralize(close, sector_map), 2), 8)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37), 14), 12)))"""
    close = data['close']
    open_ = data['open']
    vwap = data['vwap']
    adv180 = data['adv180']
    part1 = rank_cs(decay_linear(delta(indneutralize_cs(close, sector_map), 2), 8))
    blend = vwap * 0.318108 + open_ * (1 - 0.318108)
    part2 = rank_cs(decay_linear(ts_corr(blend, ts_sum(adv180, 37), 14), 12))
    result = part1 - part2
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#64
# ─────────────────────────────────────────────────────────────────────────────

def alpha064(data, sector_map):
    """Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 13), sum(adv120, 13), 17)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 4))) * -1)"""
    open_ = data['open']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    adv120 = data['adv120']
    blend_ol = open_ * 0.178404 + low * (1 - 0.178404)
    lhs = rank_cs(ts_corr(ts_sum(blend_ol, 13), ts_sum(adv120, 13), 17))
    blend_hlv = ((high + low) / 2) * 0.178404 + vwap * (1 - 0.178404)
    rhs = rank_cs(delta(blend_hlv, 4))
    cond = lhs < rhs
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#65
# ─────────────────────────────────────────────────────────────────────────────

def alpha065(data, sector_map):
    """Alpha#65: ((rank(correlation(((open * 0.00817522) + (vwap * (1 - 0.00817522))), sum(adv60, 9), 6)) < rank((open - ts_min(open, 14)))) * -1)"""
    open_ = data['open']
    vwap = data['vwap']
    adv60 = data['adv60']
    blend = open_ * 0.00817522 + vwap * (1 - 0.00817522)
    lhs = rank_cs(ts_corr(blend, ts_sum(adv60, 9), 6))
    rhs = rank_cs(open_ - ts_min(open_, 14))
    cond = lhs < rhs
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#66
# ─────────────────────────────────────────────────────────────────────────────

def alpha066(data, sector_map):
    """Alpha#66: ((rank(decay_linear(delta(vwap, 4), 7)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2) + 1e-8)), 11), 7)) * -1)"""
    open_ = data['open']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    part1 = rank_cs(decay_linear(delta(vwap, 4), 7))
    # low * 0.96633 + low * (1 - 0.96633) = low
    blend_low = low * 0.96633 + low * (1 - 0.96633)
    inner2 = (blend_low - vwap) / (open_ - (high + low) / 2 + 1e-8)
    part2 = ts_rank(decay_linear(inner2, 11), 7)
    result = (part1 + part2) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#67
# ─────────────────────────────────────────────────────────────────────────────

def alpha067(data, sector_map):
    """Alpha#67: ((rank((high - ts_min(high, 2)))**rank(correlation(indneutralize(vwap, sector_map), indneutralize(adv20, sector_map), 6))) * -1)"""
    high = data['high']
    vwap = data['vwap']
    adv20 = data['adv20']
    base = rank_cs(high - ts_min(high, 2))
    exp_ = rank_cs(ts_corr(indneutralize_cs(vwap, sector_map), indneutralize_cs(adv20, sector_map), 6))
    result = (base ** exp_) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#68
# ─────────────────────────────────────────────────────────────────────────────

def alpha068(data, sector_map):
    """Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 9), 14) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1))) * -1)"""
    high = data['high']
    close = data['close']
    low = data['low']
    adv15 = data['adv15']
    lhs = ts_rank(ts_corr(rank_cs(high), rank_cs(adv15), 9), 14)
    blend = close * 0.518371 + low * (1 - 0.518371)
    rhs = rank_cs(delta(blend, 1))
    cond = lhs < rhs
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#69
# ─────────────────────────────────────────────────────────────────────────────

def alpha069(data, sector_map):
    """Alpha#69: ((rank(ts_max(delta(indneutralize(vwap, sector_map), 3), 5))**Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 5), 9)) * -1)"""
    close = data['close']
    vwap = data['vwap']
    adv20 = data['adv20']
    base = rank_cs(ts_max(delta(indneutralize_cs(vwap, sector_map), 3), 5))
    blend = close * 0.490655 + vwap * (1 - 0.490655)
    exp_ = ts_rank(ts_corr(blend, adv20, 5), 9)
    result = (base ** exp_) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#70
# ─────────────────────────────────────────────────────────────────────────────

def alpha070(data, sector_map):
    """Alpha#70: ((rank(delta(vwap, 1))**Ts_Rank(correlation(indneutralize(close, sector_map), adv50, 18), 18)) * -1)"""
    close = data['close']
    vwap = data['vwap']
    adv50 = data['adv50']
    base = rank_cs(delta(vwap, 1))
    exp_ = ts_rank(ts_corr(indneutralize_cs(close, sector_map), adv50, 18), 18)
    result = (base ** exp_) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#71
# ─────────────────────────────────────────────────────────────────────────────

def alpha071(data, sector_map):
    """Alpha#71: maximum(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3), Ts_Rank(adv180, 12), 18), 4), 16), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))**2), 16), 4))"""
    close = data['close']
    open_ = data['open']
    low = data['low']
    vwap = data['vwap']
    adv180 = data['adv180']
    lhs = ts_rank(decay_linear(ts_corr(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16)
    rhs = ts_rank(decay_linear(rank_cs((low + open_ - 2 * vwap)) ** 2, 16), 4)
    result = maximum(lhs, rhs)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#72
# ─────────────────────────────────────────────────────────────────────────────

def alpha072(data, sector_map):
    """Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 9), 10)) / rank(decay_linear(correlation(Ts_Rank(vwap, 4), Ts_Rank(volume, 19), 7), 3)))"""
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']
    adv40 = data['adv40']
    num = rank_cs(decay_linear(ts_corr((high + low) / 2, adv40, 9), 10))
    denom = rank_cs(decay_linear(ts_corr(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))
    result = num / (denom + 1e-8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#73
# ─────────────────────────────────────────────────────────────────────────────

def alpha073(data, sector_map):
    """Alpha#73: (maximum(rank(decay_linear(delta(vwap, 5), 3)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3), 17)) * -1)"""
    open_ = data['open']
    low = data['low']
    vwap = data['vwap']
    lhs = rank_cs(decay_linear(delta(vwap, 5), 3))
    blend = open_ * 0.147155 + low * (1 - 0.147155)
    inner = (delta(blend, 2) / (blend + 1e-8)) * -1
    rhs = ts_rank(decay_linear(inner, 3), 17)
    result = maximum(lhs, rhs) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#74
# ─────────────────────────────────────────────────────────────────────────────

def alpha074(data, sector_map):
    """Alpha#74: ((rank(correlation(close, sum(adv30, 37), 15)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11))) * -1)"""
    close = data['close']
    high = data['high']
    vwap = data['vwap']
    volume = data['volume']
    adv30 = data['adv30']
    lhs = rank_cs(ts_corr(close, ts_sum(adv30, 37), 15))
    blend = high * 0.0261661 + vwap * (1 - 0.0261661)
    rhs = rank_cs(ts_corr(rank_cs(blend), rank_cs(volume), 11))
    cond = lhs < rhs
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#75
# ─────────────────────────────────────────────────────────────────────────────

def alpha075(data, sector_map):
    """Alpha#75: (rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12)))"""
    vwap = data['vwap']
    volume = data['volume']
    low = data['low']
    adv50 = data['adv50']
    lhs = rank_cs(ts_corr(vwap, volume, 4))
    rhs = rank_cs(ts_corr(rank_cs(low), rank_cs(adv50), 12))
    result = (lhs < rhs).astype(float)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#76
# ─────────────────────────────────────────────────────────────────────────────

def alpha076(data, sector_map):
    """Alpha#76: (maximum(rank(decay_linear(delta(vwap, 1), 12)), Ts_Rank(decay_linear(Ts_Rank(correlation(indneutralize(low, sector_map), adv81, 8), 20), 17), 19)) * -1)"""
    low = data['low']
    vwap = data['vwap']
    adv81 = data['adv81']
    lhs = rank_cs(decay_linear(delta(vwap, 1), 12))
    inner = ts_rank(ts_corr(indneutralize_cs(low, sector_map), adv81, 8), 20)
    rhs = ts_rank(decay_linear(inner, 17), 19)
    result = maximum(lhs, rhs) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#77
# ─────────────────────────────────────────────────────────────────────────────

def alpha077(data, sector_map):
    """Alpha#77: minimum(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3), 6)))"""
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    adv40 = data['adv40']
    inner1 = ((high + low) / 2 + high) - (vwap + high)
    lhs = rank_cs(decay_linear(inner1, 20))
    rhs = rank_cs(decay_linear(ts_corr((high + low) / 2, adv40, 3), 6))
    result = minimum(lhs, rhs)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#78
# ─────────────────────────────────────────────────────────────────────────────

def alpha078(data, sector_map):
    """Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 20), sum(adv40, 20), 7))**rank(correlation(rank(vwap), rank(volume), 6)))"""
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']
    adv40 = data['adv40']
    blend = low * 0.352233 + vwap * (1 - 0.352233)
    base = rank_cs(ts_corr(ts_sum(blend, 20), ts_sum(adv40, 20), 7))
    exp_ = rank_cs(ts_corr(rank_cs(vwap), rank_cs(volume), 6))
    result = base ** exp_
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#79
# ─────────────────────────────────────────────────────────────────────────────

def alpha079(data, sector_map):
    """Alpha#79: (rank(delta(indneutralize(((close * 0.60733) + (open * (1 - 0.60733))), sector_map), 1)) < rank(correlation(Ts_Rank(vwap, 4), Ts_Rank(adv150, 9), 15)))"""
    close = data['close']
    open_ = data['open']
    vwap = data['vwap']
    adv150 = data['adv150']
    blend = close * 0.60733 + open_ * (1 - 0.60733)
    lhs = rank_cs(delta(indneutralize_cs(blend, sector_map), 1))
    rhs = rank_cs(ts_corr(ts_rank(vwap, 4), ts_rank(adv150, 9), 15))
    result = (lhs < rhs).astype(float)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#80
# ─────────────────────────────────────────────────────────────────────────────

def alpha080(data, sector_map):
    """Alpha#80: ((rank(sign(delta(indneutralize(((open * 0.868128) + (high * (1 - 0.868128))), sector_map), 4)))**Ts_Rank(correlation(high, adv10, 5), 6)) * -1)"""
    open_ = data['open']
    high = data['high']
    adv10 = data['adv10']
    blend = open_ * 0.868128 + high * (1 - 0.868128)
    base = rank_cs(sign(delta(indneutralize_cs(blend, sector_map), 4)))
    exp_ = ts_rank(ts_corr(high, adv10, 5), 6)
    result = (base ** exp_) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#81
# ─────────────────────────────────────────────────────────────────────────────

def alpha081(data, sector_map):
    """Alpha#81: ((rank(log(ts_product(rank((rank(correlation(vwap, sum(adv10, 50), 8))**4)), 15))) < rank(correlation(rank(vwap), rank(volume), 5))) * -1)"""
    vwap = data['vwap']
    volume = data['volume']
    adv10 = data['adv10']
    inner = rank_cs(ts_corr(vwap, ts_sum(adv10, 50), 8)) ** 4
    inner2 = rank_cs(inner)
    prod15 = ts_product(inner2, 15)
    lhs = rank_cs(np.log(prod15 + 1e-8))
    rhs = rank_cs(ts_corr(rank_cs(vwap), rank_cs(volume), 5))
    cond = lhs < rhs
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#82
# ─────────────────────────────────────────────────────────────────────────────

def alpha082(data, sector_map):
    """Alpha#82: (minimum(rank(decay_linear(delta(open, 1), 15)), Ts_Rank(decay_linear(correlation(indneutralize(volume, sector_map), ((open * 0.634196) + (open * (1 - 0.634196))), 17), 7), 13)) * -1)"""
    open_ = data['open']
    volume = data['volume']
    lhs = rank_cs(decay_linear(delta(open_, 1), 15))
    # open * 0.634196 + open * (1 - 0.634196) = open
    blend = open_ * 0.634196 + open_ * (1 - 0.634196)
    inner = ts_corr(indneutralize_cs(volume, sector_map), blend, 17)
    rhs = ts_rank(decay_linear(inner, 7), 13)
    result = minimum(lhs, rhs) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#83
# ─────────────────────────────────────────────────────────────────────────────

def alpha083(data, sector_map):
    """Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5 + 1e-8)) / (vwap - close + 1e-8)))"""
    close = data['close']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']
    hl_ratio = (high - low) / (ts_sum(close, 5) / 5 + 1e-8)
    num = rank_cs(delay(hl_ratio, 2)) * rank_cs(rank_cs(volume))
    denom = hl_ratio / (vwap - close + 1e-8)
    result = num / (denom + 1e-8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#84
# ─────────────────────────────────────────────────────────────────────────────

def alpha084(data, sector_map):
    """Alpha#84: signed_power(Ts_Rank((vwap - ts_max(vwap, 15)), 21), delta(close, 5))"""
    close = data['close']
    vwap = data['vwap']
    base = ts_rank(vwap - ts_max(vwap, 15), 21)
    exp_ = delta(close, 5)
    result = signed_power(base, exp_)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#85
# ─────────────────────────────────────────────────────────────────────────────

def alpha085(data, sector_map):
    """Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 10))**rank(correlation(Ts_Rank(((high + low) / 2), 4), Ts_Rank(volume, 10), 7)))"""
    high = data['high']
    close = data['close']
    low = data['low']
    volume = data['volume']
    adv30 = data['adv30']
    blend = high * 0.876703 + close * (1 - 0.876703)
    base = rank_cs(ts_corr(blend, adv30, 10))
    exp_ = rank_cs(ts_corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7))
    result = base ** exp_
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#86
# ─────────────────────────────────────────────────────────────────────────────

def alpha086(data, sector_map):
    """Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 15), 6), 20) < rank(((open + close) - (vwap + open)))) * -1)"""
    close = data['close']
    open_ = data['open']
    vwap = data['vwap']
    adv20 = data['adv20']
    lhs = ts_rank(ts_corr(close, ts_sum(adv20, 15), 6), 20)
    rhs = rank_cs((open_ + close) - (vwap + open_))
    cond = lhs < rhs
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#87
# ─────────────────────────────────────────────────────────────────────────────

def alpha087(data, sector_map):
    """Alpha#87: (maximum(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 2), 3)), Ts_Rank(decay_linear(abs(correlation(indneutralize(adv81, sector_map), close, 13)), 5), 14)) * -1)"""
    close = data['close']
    vwap = data['vwap']
    adv81 = data['adv81']
    blend = close * 0.369701 + vwap * (1 - 0.369701)
    lhs = rank_cs(decay_linear(delta(blend, 2), 3))
    inner = abs_val(ts_corr(indneutralize_cs(adv81, sector_map), close, 13))
    rhs = ts_rank(decay_linear(inner, 5), 14)
    result = maximum(lhs, rhs) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#88
# ─────────────────────────────────────────────────────────────────────────────

def alpha088(data, sector_map):
    """Alpha#88: minimum(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8), Ts_Rank(adv60, 21), 8), 7), 3))"""
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']
    adv60 = data['adv60']
    inner1 = (rank_cs(open_) + rank_cs(low)) - (rank_cs(high) + rank_cs(close))
    lhs = rank_cs(decay_linear(inner1, 8))
    rhs = ts_rank(decay_linear(ts_corr(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3)
    result = minimum(lhs, rhs)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#89
# ─────────────────────────────────────────────────────────────────────────────

def alpha089(data, sector_map):
    """Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 7), 6), 4) - Ts_Rank(decay_linear(delta(indneutralize(vwap, sector_map), 3), 10), 15))"""
    low = data['low']
    vwap = data['vwap']
    adv10 = data['adv10']
    # low * 0.967285 + low * (1 - 0.967285) = low
    blend = low * 0.967285 + low * (1 - 0.967285)
    lhs = ts_rank(decay_linear(ts_corr(blend, adv10, 7), 6), 4)
    rhs = ts_rank(decay_linear(delta(indneutralize_cs(vwap, sector_map), 3), 10), 15)
    result = lhs - rhs
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#90
# ─────────────────────────────────────────────────────────────────────────────

def alpha090(data, sector_map):
    """Alpha#90: ((rank((close - ts_max(close, 5)))**Ts_Rank(correlation(indneutralize(adv10, sector_map), low, 6), 5)) * -1)"""
    close = data['close']
    low = data['low']
    adv10 = data['adv10']
    base = rank_cs(close - ts_max(close, 5))
    exp_ = ts_rank(ts_corr(indneutralize_cs(adv10, sector_map), low, 6), 5)
    result = (base ** exp_) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#91
# ─────────────────────────────────────────────────────────────────────────────

def alpha091(data, sector_map):
    """Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(indneutralize(close, sector_map), volume, 10), 16), 4), 5) - rank(decay_linear(correlation(vwap, adv30, 4), 3))) * -1)"""
    close = data['close']
    vwap = data['vwap']
    volume = data['volume']
    adv30 = data['adv30']
    inner = decay_linear(ts_corr(indneutralize_cs(close, sector_map), volume, 10), 16)
    lhs = ts_rank(decay_linear(inner, 4), 5)
    rhs = rank_cs(decay_linear(ts_corr(vwap, adv30, 4), 3))
    result = (lhs - rhs) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#92
# ─────────────────────────────────────────────────────────────────────────────

def alpha092(data, sector_map):
    """Alpha#92: minimum(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 15), 19), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7))"""
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']
    adv30 = data['adv30']
    cond = ((high + low) / 2 + close) < (low + open_)
    lhs = ts_rank(decay_linear(cond.astype(float), 15), 19)
    rhs = ts_rank(decay_linear(ts_corr(rank_cs(low), rank_cs(adv30), 8), 7), 7)
    result = minimum(lhs, rhs)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#93
# ─────────────────────────────────────────────────────────────────────────────

def alpha093(data, sector_map):
    """Alpha#93: (Ts_Rank(decay_linear(correlation(indneutralize(vwap, sector_map), adv81, 17), 20), 8) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 3), 16)))"""
    close = data['close']
    vwap = data['vwap']
    adv81 = data['adv81']
    lhs = ts_rank(decay_linear(ts_corr(indneutralize_cs(vwap, sector_map), adv81, 17), 20), 8)
    blend = close * 0.524434 + vwap * (1 - 0.524434)
    rhs = rank_cs(decay_linear(delta(blend, 3), 16))
    result = lhs / (rhs + 1e-8)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#94
# ─────────────────────────────────────────────────────────────────────────────

def alpha094(data, sector_map):
    """Alpha#94: ((rank((vwap - ts_min(vwap, 12)))**Ts_Rank(correlation(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3)) * -1)"""
    vwap = data['vwap']
    adv60 = data['adv60']
    base = rank_cs(vwap - ts_min(vwap, 12))
    exp_ = ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)
    result = (base ** exp_) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#95
# ─────────────────────────────────────────────────────────────────────────────

def alpha095(data, sector_map):
    """Alpha#95: (rank((open - ts_min(open, 12))) < Ts_Rank(rank(correlation(sum(((high + low) / 2), 19), sum(adv40, 19), 13)), 12))"""
    open_ = data['open']
    high = data['high']
    low = data['low']
    adv40 = data['adv40']
    lhs = rank_cs(open_ - ts_min(open_, 12))
    rhs = ts_rank(rank_cs(ts_corr(ts_sum((high + low) / 2, 19), ts_sum(adv40, 19), 13)), 12)
    result = (lhs < rhs).astype(float)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#96
# ─────────────────────────────────────────────────────────────────────────────

def alpha096(data, sector_map):
    """Alpha#96: (maximum(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8), Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(close, 7), Ts_Rank(adv60, 4), 4), 5), 2), 4)) * -1)"""
    close = data['close']
    vwap = data['vwap']
    volume = data['volume']
    adv60 = data['adv60']
    lhs = ts_rank(decay_linear(ts_corr(rank_cs(vwap), rank_cs(volume), 4), 4), 8)
    inner = ts_rank(ts_corr(ts_rank(close, 7), ts_rank(adv60, 4), 4), 5)
    rhs = ts_rank(decay_linear(inner, 2), 4)
    result = maximum(lhs, rhs) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#97
# ─────────────────────────────────────────────────────────────────────────────

def alpha097(data, sector_map):
    """Alpha#97: ((rank(decay_linear(delta(indneutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), sector_map), 3), 20)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 8), Ts_Rank(adv60, 17), 5), 19), 16), 7)) * -1)"""
    low = data['low']
    vwap = data['vwap']
    adv60 = data['adv60']
    blend = low * 0.721001 + vwap * (1 - 0.721001)
    lhs = rank_cs(decay_linear(delta(indneutralize_cs(blend, sector_map), 3), 20))
    inner = ts_rank(ts_corr(ts_rank(low, 8), ts_rank(adv60, 17), 5), 19)
    rhs = ts_rank(decay_linear(inner, 16), 7)
    result = (lhs - rhs) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#98
# ─────────────────────────────────────────────────────────────────────────────

def alpha098(data, sector_map):
    """Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26), 5), 7)) - rank(decay_linear(Ts_Rank(Ts_Rank(correlation(rank(open), rank(adv15), 21), 9), 7), 8)))"""
    open_ = data['open']
    vwap = data['vwap']
    adv5 = data['adv5']
    adv15 = data['adv15']
    lhs = rank_cs(decay_linear(ts_corr(vwap, ts_sum(adv5, 26), 5), 7))
    inner = ts_rank(ts_rank(ts_corr(rank_cs(open_), rank_cs(adv15), 21), 9), 7)
    rhs = rank_cs(decay_linear(inner, 8))
    result = lhs - rhs
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#99
# ─────────────────────────────────────────────────────────────────────────────

def alpha099(data, sector_map):
    """Alpha#99: ((rank(correlation(sum(((high + low) / 2), 20), sum(adv60, 20), 9)) < rank(correlation(low, volume, 6))) * -1)"""
    high = data['high']
    low = data['low']
    volume = data['volume']
    adv60 = data['adv60']
    lhs = rank_cs(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(adv60, 20), 9))
    rhs = rank_cs(ts_corr(low, volume, 6))
    cond = lhs < rhs
    result = cond.astype(float) * -1
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#100
# ─────────────────────────────────────────────────────────────────────────────

def alpha100(data, sector_map):
    """Alpha#100: (0 - (1 * (((1.5 * scale_cs(indneutralize_cs(indneutralize_cs(rank_cs(((((close - low) - (high - close)) / (high - low + 1e-8)) * volume)), sector_map), sector_map))) - scale_cs(indneutralize_cs((correlation(close, rank(adv20), 5) - rank(delta(close, 3))), sector_map))) * (volume / adv20))))"""
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    adv20 = data['adv20']
    raw = ((close - low) - (high - close)) / (high - low + 1e-8) * volume
    part1 = 1.5 * scale_cs(indneutralize_cs(indneutralize_cs(rank_cs(raw), sector_map), sector_map))
    inner2 = ts_corr(close, rank_cs(adv20), 5) - rank_cs(delta(close, 3))
    part2 = scale_cs(indneutralize_cs(inner2, sector_map))
    result = 0 - ((part1 - part2) * (volume / (adv20 + 1e-8)))
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha#101
# ─────────────────────────────────────────────────────────────────────────────

def alpha101(data, sector_map):
    """Alpha#101: ((close - open) / ((high - low) + 0.001))"""
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']
    result = (close - open_) / (high - low + 0.001)
    return _clean(result)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def get_all_alpha_functions():
    """Return sorted list of (name, function) for all implemented alphas."""
    import inspect
    import sys
    module = sys.modules[__name__]
    funcs = []
    for name in sorted(vars(module)):
        if name.startswith('alpha') and callable(getattr(module, name)):
            funcs.append((name, getattr(module, name)))
    return funcs
