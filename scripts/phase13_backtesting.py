"""
Phase 13: Backtesting with Full Statistics + DSR
=================================================
Two strategies evaluated per ticker and at portfolio level:

  Strategy A – Primary model only:
      position = side (±1), held from event date through t1 barrier

  Strategy B – Meta-labeled (bet-sized):
      position = disc_signal (side × size), held from event date through t1

For each strategy:
  - Expand event signals to daily positions (avg active signals → ffill)
  - Compute daily PnL via backtest_strategy (cost_bps = 5)
  - Compute SR, PSR, DSR (N = 60 tuning trials from Phase 11), max_dd,
    max_tuw, Calmar, hit ratio, profit factor

Portfolio = equal-weight average of per-ticker daily positions.

Artifacts saved
---------------
data/processed/backtest_returns_pooled.parquet   (daily returns, per-ticker + portfolio)
data/processed/backtest_stats_pooled.parquet     (summary stats table)
reports/figures/phase13_equity_curves.png
reports/figures/phase13_per_ticker_sr.png
reports/figures/phase13_drawdown.png
"""

import os, sys, json
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.backtesting import (
    backtest_strategy, sharpe_ratio, prob_sharpe_ratio,
    deflated_sharpe_ratio, compute_dd_tuw, calmar_ratio,
    hit_ratio, profit_factor, summary_table,
)
from src.bet_sizing import avg_active_signals, build_daily_positions

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed',  exist_ok=True)

COST_BPS   = 5
N_TRIALS   = 60     # 30 RF + 30 XGB tuning trials from Phase 11
PPY        = 252    # periods per year


def sep(title):
    print('\n' + '=' * 68)
    print(title)
    print('=' * 68)


def check(label, cond):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {label}')
    return cond


# ── Load artifacts ────────────────────────────────────────────────────────────
sep('LOAD: OOS predictions + bet sizes + panel prices')

with open('configs/universe.json') as f:
    UNI = json.load(f)
TICKERS = UNI['tickers']

oos_preds  = pd.read_parquet('data/processed/oos_predictions_pooled.parquet')
bet_sizes  = pd.read_parquet('data/processed/bet_sizes_pooled.parquet')
panel      = pd.read_parquet('data/processed/panel_ohlcv.parquet')

print(f'  OOS preds  : {oos_preds.shape}  cols={list(oos_preds.columns)}')
print(f'  Bet sizes  : {bet_sizes.shape}  cols={list(bet_sizes.columns)}')
print(f'  Panel      : {panel.shape}  (Date x ticker)')

# Adj Close wide (Date × ticker)
adj_close = panel['AdjClose'].unstack(level='ticker')
print(f'  AdjClose   : {adj_close.shape}  range={adj_close.index.min().date()} -> {adj_close.index.max().date()}')

# ── Step 1: Build per-ticker signal series ────────────────────────────────────
sep('STEP 1: Build per-ticker event-level signal series')

daily_index = adj_close.index

returns_A  = {}   # strategy A (primary ±1)
returns_B  = {}   # strategy B (meta disc_signal)
stats_rows = []

for ticker in TICKERS:
    prices = adj_close[ticker].dropna()
    if len(prices) == 0:
        print(f'  {ticker}: no prices — skip')
        continue

    # ── Strategy A: primary model side ────────────────────────────────────────
    mask_a  = oos_preds['ticker'] == ticker
    oos_t   = oos_preds[mask_a].copy()

    if len(oos_t) == 0:
        print(f'  {ticker}: no events — skip')
        continue

    # Build avg-active position for strategy A
    # Each event: signal = side, active from event_date through t1
    sig_a = pd.Series(
        oos_t['oos_pred'].values.astype(float),
        index=oos_t.index,
    )
    t1_a = oos_t['t1']
    pos_avg_a = avg_active_signals(sig_a, t1_a)
    pos_daily_a = build_daily_positions(pos_avg_a, daily_index)

    bt_a = backtest_strategy(pos_daily_a, prices, cost_bps=COST_BPS)
    returns_A[ticker] = bt_a['net_return']

    # ── Strategy B: meta disc_signal ─────────────────────────────────────────
    mask_b  = bet_sizes['ticker'] == ticker
    bet_t   = bet_sizes[mask_b].copy()

    sig_b = pd.Series(
        bet_t['disc_signal'].values.astype(float),
        index=bet_t.index,
    )
    t1_b = bet_t['t1']
    pos_avg_b = avg_active_signals(sig_b, t1_b)
    pos_daily_b = build_daily_positions(pos_avg_b, daily_index)

    bt_b = backtest_strategy(pos_daily_b, prices, cost_bps=COST_BPS)
    returns_B[ticker] = bt_b['net_return']

    # Stats for this ticker
    sr_a = sharpe_ratio(bt_a['net_return'], PPY)
    sr_b = sharpe_ratio(bt_b['net_return'], PPY)
    n_events = int(mask_a.sum())

    print(f'  {ticker:6s}: n_events={n_events:4d}  '
          f'SR_A={sr_a:+.3f}  SR_B={sr_b:+.3f}  '
          f'days_A={int((pos_daily_a != 0).sum()):5d}  days_B={int((pos_daily_b != 0).sum()):5d}')

# ── Step 2: Portfolio returns ─────────────────────────────────────────────────
sep('STEP 2: Equal-weight portfolio returns')

ret_df_A = pd.DataFrame(returns_A).reindex(daily_index).fillna(0.0)
ret_df_B = pd.DataFrame(returns_B).reindex(daily_index).fillna(0.0)

# Equal weight: average across non-zero-position tickers each day
port_A = ret_df_A.mean(axis=1)
port_B = ret_df_B.mean(axis=1)

print(f'  Portfolio A SR : {sharpe_ratio(port_A, PPY):+.4f}')
print(f'  Portfolio B SR : {sharpe_ratio(port_B, PPY):+.4f}')

# ── Step 3: Full statistics table ─────────────────────────────────────────────
sep('STEP 3: Full statistics table (SR, PSR, DSR, DD, Calmar, Hit, PF)')

def stats_row(name, returns):
    active = returns[returns != 0]
    if len(active) < 10:
        return {'strategy': name, 'sr': np.nan, 'psr': np.nan, 'dsr': np.nan,
                'max_dd': np.nan, 'max_tuw': np.nan, 'calmar': np.nan,
                'hit_ratio': np.nan, 'profit_factor': np.nan,
                'ann_return': np.nan, 'ann_vol': np.nan, 'n_days': len(active)}

    sr   = sharpe_ratio(active, PPY)
    psr  = prob_sharpe_ratio(active, sr_benchmark=0.0)
    dsr  = deflated_sharpe_ratio(active, num_trials=N_TRIALS)
    _, max_dd, max_tuw, _, _ = compute_dd_tuw(active)
    calmar = calmar_ratio(active, PPY)
    hit    = hit_ratio(active)
    pf     = profit_factor(active)
    ann_ret = float((1 + active.mean()) ** PPY - 1)
    ann_vol = float(active.std() * np.sqrt(PPY))

    return {
        'strategy': name, 'sr': sr, 'psr': psr, 'dsr': dsr,
        'max_dd': max_dd, 'max_tuw': max_tuw, 'calmar': calmar,
        'hit_ratio': hit, 'profit_factor': pf,
        'ann_return': ann_ret, 'ann_vol': ann_vol, 'n_days': len(active),
    }

rows = []

# Per-ticker stats
for ticker in TICKERS:
    if ticker not in returns_A:
        continue
    rows.append(stats_row(f'{ticker}_A', returns_A[ticker]))
    rows.append(stats_row(f'{ticker}_B', returns_B[ticker]))

# Portfolio stats
rows.append(stats_row('Portfolio_A', port_A))
rows.append(stats_row('Portfolio_B', port_B))

stats_df = pd.DataFrame(rows).set_index('strategy')

print('\n  Per-ticker + portfolio stats:')
print(stats_df[['sr', 'psr', 'dsr', 'max_dd', 'calmar', 'hit_ratio', 'profit_factor']].round(4).to_string())

print('\n  Portfolio summary:')
for col in ['sr', 'psr', 'dsr', 'max_dd', 'max_tuw', 'calmar', 'hit_ratio', 'profit_factor', 'ann_return', 'ann_vol']:
    va = stats_df.loc['Portfolio_A', col]
    vb = stats_df.loc['Portfolio_B', col]
    print(f'    {col:20s}: A={va:.4f}  B={vb:.4f}')

# Save
all_returns = pd.concat(
    [ret_df_A.add_suffix('_A'), ret_df_B.add_suffix('_B'),
     port_A.rename('Portfolio_A'), port_B.rename('Portfolio_B')],
    axis=1,
)
all_returns.to_parquet('data/processed/backtest_returns_pooled.parquet')
stats_df.to_parquet('data/processed/backtest_stats_pooled.parquet')
print(f'\n  Saved: data/processed/backtest_returns_pooled.parquet  {all_returns.shape}')
print(f'  Saved: data/processed/backtest_stats_pooled.parquet    {stats_df.shape}')

# ── Step 4: Figures ───────────────────────────────────────────────────────────
sep('STEP 4: Figures')

# Fig 1: Portfolio equity curves (A vs B) + drawdown
cum_A = (1 + port_A).cumprod()
cum_B = (1 + port_B).cumprod()
_, dd_A, *_ = compute_dd_tuw(port_A)
_, dd_B, *_ = compute_dd_tuw(port_B)
dd_series_A = 1 - (1 + port_A).cumprod() / (1 + port_A).cumprod().expanding().max()
dd_series_B = 1 - (1 + port_B).cumprod() / (1 + port_B).cumprod().expanding().max()

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(cum_A.index, cum_A.values, color='steelblue', linewidth=1.2,
             label=f'A: Primary (SR={stats_df.loc["Portfolio_A","sr"]:.3f})')
axes[0].plot(cum_B.index, cum_B.values, color='darkorange', linewidth=1.2,
             label=f'B: Meta-sized (SR={stats_df.loc["Portfolio_B","sr"]:.3f})')
axes[0].set_ylabel('Cumulative return (equity curve)')
axes[0].set_title('10-Stock Equal-Weight Portfolio — Equity Curves\n'
                  f'Strategy A (primary ±1) vs B (meta disc_signal), cost={COST_BPS}bps')
axes[0].legend(loc='upper left')
axes[0].axhline(1.0, color='grey', linestyle=':', alpha=0.5)

axes[1].fill_between(dd_series_A.index, -dd_series_A.values, 0,
                     alpha=0.4, color='steelblue', label='A drawdown')
axes[1].fill_between(dd_series_B.index, -dd_series_B.values, 0,
                     alpha=0.4, color='darkorange', label='B drawdown')
axes[1].set_ylabel('Drawdown')
axes[1].set_xlabel('Date')
axes[1].set_title('Drawdown Profile')
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.savefig('reports/figures/phase13_equity_curves.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase13_equity_curves.png')

# Fig 2: Per-ticker SR bars (A vs B side-by-side)
tick_stats = stats_df[stats_df.index.str.contains('_A|_B') &
                      ~stats_df.index.str.startswith('Portfolio')]
tickers_in_stats = [t for t in TICKERS if t in returns_A]

x = np.arange(len(tickers_in_stats))
width = 0.35
sr_a_vals = [stats_df.loc[f'{t}_A', 'sr'] for t in tickers_in_stats]
sr_b_vals = [stats_df.loc[f'{t}_B', 'sr'] for t in tickers_in_stats]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, sr_a_vals, width, label='A: Primary', color='steelblue', alpha=0.8)
ax.bar(x + width/2, sr_b_vals, width, label='B: Meta-sized', color='darkorange', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(tickers_in_stats)
ax.set_ylabel('Annualised Sharpe Ratio')
ax.set_title('Per-Ticker Sharpe Ratio: Primary vs Meta-Sized Strategy')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/phase13_per_ticker_sr.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase13_per_ticker_sr.png')

# Fig 3: Rolling SR (252-day) for portfolio
rolling_sr_A = (port_A.rolling(252).mean() / port_A.rolling(252).std() * np.sqrt(252))
rolling_sr_B = (port_B.rolling(252).mean() / port_B.rolling(252).std() * np.sqrt(252))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(rolling_sr_A.index, rolling_sr_A.values, color='steelblue', linewidth=1,
        label='A: Primary', alpha=0.85)
ax.plot(rolling_sr_B.index, rolling_sr_B.values, color='darkorange', linewidth=1,
        label='B: Meta-sized', alpha=0.85)
ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
ax.set_xlabel('Date')
ax.set_ylabel('Rolling 252-day Sharpe Ratio')
ax.set_title('Rolling Sharpe Ratio — 10-Stock Portfolio')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/phase13_drawdown.png', dpi=100)
plt.close()
print('  Saved: reports/figures/phase13_drawdown.png')

# ── Step 5: Validation ────────────────────────────────────────────────────────
sep('STEP 5: Validation')

failures = []

sr_port_A = float(stats_df.loc['Portfolio_A', 'sr'])
sr_port_B = float(stats_df.loc['Portfolio_B', 'sr'])
dsr_port_A = float(stats_df.loc['Portfolio_A', 'dsr'])
dsr_port_B = float(stats_df.loc['Portfolio_B', 'dsr'])
n_tickers_with_stats = len([t for t in TICKERS if f'{t}_A' in stats_df.index])

checks = [
    ('backtest_returns_pooled.parquet saved',
     os.path.exists('data/processed/backtest_returns_pooled.parquet')),
    ('backtest_stats_pooled.parquet saved',
     os.path.exists('data/processed/backtest_stats_pooled.parquet')),
    ('equity_curves fig saved',
     os.path.exists('reports/figures/phase13_equity_curves.png')),
    ('per_ticker_sr fig saved',
     os.path.exists('reports/figures/phase13_per_ticker_sr.png')),
    ('drawdown fig saved',
     os.path.exists('reports/figures/phase13_drawdown.png')),
    (f'all {len(TICKERS)} tickers have stats',
     n_tickers_with_stats == len(TICKERS)),
    ('Portfolio_A and Portfolio_B in stats',
     'Portfolio_A' in stats_df.index and 'Portfolio_B' in stats_df.index),
    ('no NaN SR for portfolio A',
     not np.isnan(sr_port_A)),
    ('no NaN SR for portfolio B',
     not np.isnan(sr_port_B)),
    ('DSR computed for portfolio A',
     not np.isnan(dsr_port_A)),
    ('DSR computed for portfolio B',
     not np.isnan(dsr_port_B)),
    ('returns parquet has expected columns',
     all(f'{t}_A' in all_returns.columns for t in tickers_in_stats)),
    ('max_dd <= 1.0 for portfolio A',
     float(stats_df.loc['Portfolio_A', 'max_dd']) <= 1.0),
    ('max_dd <= 1.0 for portfolio B',
     float(stats_df.loc['Portfolio_B', 'max_dd']) <= 1.0),
]

for label, cond in checks:
    if not check(label, cond):
        failures.append(label)

n_pass = len(checks) - len(failures)
print(f'\n{"=" * 68}')
if failures:
    print(f'Phase 13 FAILED — {len(failures)} check(s) failed:')
    for f in failures:
        print(f'  {f}: FAIL')
else:
    print(f'Phase 13 COMPLETE — {n_pass} checks passed.')
    print(f'  Strategies backtested : A (primary ±1) + B (meta disc_signal)')
    print(f'  Cost assumption       : {COST_BPS} bps per unit turnover')
    print(f'  N trials for DSR      : {N_TRIALS}')
    print(f'  Portfolio A SR / DSR  : {sr_port_A:.4f} / {dsr_port_A:.4f}')
    print(f'  Portfolio B SR / DSR  : {sr_port_B:.4f} / {dsr_port_B:.4f}')
    print(f'  Portfolio A max_dd    : {stats_df.loc["Portfolio_A","max_dd"]:.4%}')
    print(f'  Portfolio B max_dd    : {stats_df.loc["Portfolio_B","max_dd"]:.4%}')
