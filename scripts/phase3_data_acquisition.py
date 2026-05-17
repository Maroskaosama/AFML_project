"""
Phase 3: Data Acquisition — BAC/UNH universe, 2005-2025 panel rebuild.

Steps:
  1. Safety snapshot of data/processed -> backups/
  2. Download BAC and UNH raw CSVs (skip if already present and valid)
  3. Validate all 10 stocks cover 2005-01-03
  4. Rebuild panel_ohlcv.parquet with new 10-stock universe
  5. Archive META/TSLA artifacts (raw CSVs + per_stock parquets)
  6. Validation summary
"""
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
with open('configs/universe.json') as f:
    UNI = json.load(f)

TICKERS      = UNI['tickers']           # 10-stock list (BAC/UNH universe)
COMMON_START = UNI['common_start']      # '2005-01-03'
COMMON_END   = UNI['common_end']        # '2025-04-30'
REQUIRED_COLS = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

RAW_DIR      = 'data/raw'
PER_STOCK    = 'data/processed/per_stock'
PANEL_DIR    = 'data/processed/panel'
PANEL_OUT    = 'data/processed/panel_ohlcv.parquet'
PANEL_OUT2   = os.path.join(PANEL_DIR, 'panel_ohlcv.parquet')

ARCHIVE_RAW  = 'data/raw/archive'
ARCHIVE_PROC = 'data/processed/per_stock/archive'

ERRORS = []

def sep(title=''):
    print('\n' + '=' * 64)
    if title:
        print(title)
        print('=' * 64)


# ── Step 0: Safety snapshot ────────────────────────────────────────────────
sep('STEP 0: Safety snapshot')

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
for src, label in [
    ('data/processed', f'backups/processed_{ts}'),
    ('configs',        f'backups/configs_{ts}'),
]:
    if os.path.exists(src):
        shutil.copytree(src, label)
        print(f'  Snapshot: {src} -> {label}')

os.makedirs(ARCHIVE_RAW,  exist_ok=True)
os.makedirs(ARCHIVE_PROC, exist_ok=True)
os.makedirs(PANEL_DIR,    exist_ok=True)
print('  Archive directories ready.')


# ── Step 1: Download BAC and UNH ──────────────────────────────────────────
sep('STEP 1: Download BAC and UNH raw CSVs')

NEW_TICKERS = ['BAC', 'UNH']

for ticker in NEW_TICKERS:
    path = os.path.join(RAW_DIR, f'{ticker}_raw.csv')

    # Skip if already downloaded and valid
    if os.path.exists(path):
        df_check = pd.read_csv(path, index_col='Date', parse_dates=True)
        if len(df_check) >= 2000 and df_check.index.min() <= pd.Timestamp('2005-01-03'):
            print(f'  {ticker}: already downloaded ({len(df_check)} rows, '
                  f'{df_check.index.min().date()} to {df_check.index.max().date()}) - skipping')
            continue

    print(f'  {ticker}: downloading...', end=' ', flush=True)
    try:
        raw = yf.download(
            ticker,
            start='2000-01-01',
            end=COMMON_END,
            auto_adjust=False,
            progress=False,
        )
        if raw is None or len(raw) == 0:
            raise ValueError('empty download')

        # Flatten MultiIndex columns (yfinance >= 0.2)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw[REQUIRED_COLS]
        raw.index.name = 'Date'
        raw.to_csv(path)
        print(f'OK - {len(raw)} rows, {raw.index[0].date()} to {raw.index[-1].date()}')
    except Exception as e:
        msg = f'{ticker}: download FAILED — {e}'
        print(f'FAILED: {e}')
        ERRORS.append(msg)


# ── Step 2: Validate all 10 stocks ───────────────────────────────────────
sep('STEP 2: Validate all 10 raw CSVs')

stock_ok = {}
for ticker in TICKERS:
    path = os.path.join(RAW_DIR, f'{ticker}_raw.csv')
    if not os.path.exists(path):
        print(f'  {ticker:6s}: MISSING raw CSV')
        ERRORS.append(f'{ticker}: missing raw CSV')
        continue

    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    df.columns = [c.strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        ERRORS.append(f'{ticker}: missing columns {missing_cols}')
        print(f'  {ticker:6s}: missing columns {missing_cols}')
        continue

    df = df[REQUIRED_COLS]

    checks = {
        'rows >= 2000':       len(df) >= 2000,
        'starts <= 2005':     df.index.min() <= pd.Timestamp('2005-01-03'),
        'ends >= 2025-04':    df.index.max() >= pd.Timestamp('2025-04-01'),
        'prices > 0':         (df[['Adj Close', 'Close', 'High', 'Low', 'Open']] > 0).all().all(),
        'volume > 0':         (df['Volume'] > 0).all(),
        'monotone dates':     df.index.is_monotonic_increasing,
    }

    failed = [k for k, v in checks.items() if not v]
    status = 'OK' if not failed else f'FAIL: {failed}'
    print(f'  {ticker:6s}: {len(df):5d} rows | '
          f'{df.index.min().date()} -> {df.index.max().date()} | {status}')

    if not failed:
        stock_ok[ticker] = df
    else:
        ERRORS.append(f'{ticker}: {failed}')

if ERRORS:
    print(f'\n  Stopping — validation errors before panel build:')
    for e in ERRORS:
        print(f'    {e}')
    sys.exit(1)


# ── Step 3: Rebuild panel_ohlcv.parquet ──────────────────────────────────
sep('STEP 3: Rebuild panel_ohlcv.parquet (10 stocks, 2005-2025)')

frames = []
for ticker in TICKERS:
    path = os.path.join(RAW_DIR, f'{ticker}_raw.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[REQUIRED_COLS].rename(columns={'Adj Close': 'AdjClose'})
    df = df.loc[COMMON_START:COMMON_END]
    df['ticker'] = ticker
    frames.append(df)

panel = pd.concat(frames)
panel = panel.reset_index().set_index(['Date', 'ticker']).sort_index()

# Save to both locations (flat + panel sub-directory)
panel.to_parquet(PANEL_OUT)
panel.to_parquet(PANEL_OUT2)

tickers_in_panel = sorted(panel.index.get_level_values('ticker').unique().tolist())
date_min = panel.index.get_level_values('Date').min()
date_max = panel.index.get_level_values('Date').max()
n_rows   = len(panel)

print(f'  Tickers : {tickers_in_panel}')
print(f'  Dates   : {date_min.date()} -> {date_max.date()}')
print(f'  Rows    : {n_rows:,}')
print(f'  Saved   : {PANEL_OUT}')
print(f'  Saved   : {PANEL_OUT2}')


# ── Step 4: Archive META/TSLA artifacts ──────────────────────────────────
sep('STEP 4: Archive META/TSLA raw CSVs and per_stock parquets')

for ticker in ['META', 'TSLA']:
    # Raw CSV
    raw_src = os.path.join(RAW_DIR, f'{ticker}_raw.csv')
    if os.path.exists(raw_src):
        dst = os.path.join(ARCHIVE_RAW, f'{ticker}_raw.csv')
        shutil.move(raw_src, dst)
        print(f'  Archived: {raw_src} -> {dst}')

    # Per-stock parquets
    for suffix in ['_clean', '_labels', '_ts_features', '_weights']:
        src = os.path.join(PER_STOCK, f'{ticker}{suffix}.parquet')
        if os.path.exists(src):
            dst = os.path.join(ARCHIVE_PROC, f'{ticker}{suffix}.parquet')
            shutil.move(src, dst)
            print(f'  Archived: {src} -> {dst}')

# Also archive old flat-dir NVDA and per-stock artefacts that use old date range.
# These will be regenerated in Phase 4 from the new pipeline run.
# We keep them for now to avoid losing work — Phase 4 will overwrite.
print('  Per-stock parquets for AAPL/AMZN/etc. kept in place;')
print('  Phase 4 will regenerate them with the 2005-2025 date range.')


# ── Step 5: Validation summary ────────────────────────────────────────────
sep('STEP 5: Validation')

checks = {
    'panel has 10 tickers':            len(tickers_in_panel) == 10,
    'panel tickers match config':      set(tickers_in_panel) == set(TICKERS),
    'panel starts at 2005-01-03':      date_min.date().isoformat() == COMMON_START,
    'panel ends at 2025-04-30':        date_max.date().isoformat() == COMMON_END,
    'panel has > 45000 rows':          n_rows > 45000,
    'META not in panel':               'META' not in tickers_in_panel,
    'TSLA not in panel':               'TSLA' not in tickers_in_panel,
    'BAC in panel':                    'BAC' in tickers_in_panel,
    'UNH in panel':                    'UNH' in tickers_in_panel,
    'panel file exists':               os.path.exists(PANEL_OUT),
    'panel (panel/) file exists':      os.path.exists(PANEL_OUT2),
    'no NaN in AdjClose':              panel['AdjClose'].isnull().sum() == 0,
}

all_pass = True
for label, passed in checks.items():
    status = 'PASS' if passed else 'FAIL'
    if not passed:
        all_pass = False
        ERRORS.append(f'{label}: FAIL')
    print(f'  [{status}] {label}')

sep()
if ERRORS:
    print(f'Phase 3 FAILED — {len(ERRORS)} error(s):')
    for e in ERRORS:
        print(f'  {e}')
    sys.exit(1)
else:
    print('Phase 3 COMPLETE — panel_ohlcv rebuilt with BAC/UNH universe.')
    print(f'  {n_rows:,} rows | {len(tickers_in_panel)} tickers | {COMMON_START} to {COMMON_END}')
