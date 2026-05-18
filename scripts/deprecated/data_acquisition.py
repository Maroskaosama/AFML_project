"""
DEPRECATED — original acquisition script (META/TSLA universe, 2012-2025).
Replaced by Phase 3 (scripts/phase3_data_acquisition.py) which uses the
BAC/UNH universe and 2005-2025 date range from configs/universe.json.
Do NOT re-run this script.
"""
import os, json
import pandas as pd
import numpy as np
import yfinance as yf

# ── Setup ──────────────────────────────────────────────────────────────────
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('configs', exist_ok=True)

TICKERS_NEW = ['AAPL', 'AMZN', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'XOM', 'META', 'TSLA']
TICKERS_ALL = ['AAPL', 'AMZN', 'NVDA', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'XOM', 'META', 'TSLA']

SECTOR_MAP = {
    'AAPL': 'Information Technology',
    'AMZN': 'Consumer Discretionary',
    'NVDA': 'Information Technology',
    'GOOGL': 'Communication Services',
    'JNJ': 'Health Care',
    'JPM': 'Financials',
    'MSFT': 'Information Technology',
    'XOM': 'Energy',
    'META': 'Communication Services',
    'TSLA': 'Consumer Discretionary',
}

# ── Step 1: Download 9 new stocks ──────────────────────────────────────────
print("=" * 60)
print("STEP 1: Downloading 9 stocks")
print("=" * 60)

for ticker in TICKERS_NEW:
    path = f'data/raw/{ticker}_raw.csv'
    if os.path.exists(path):
        existing = pd.read_csv(path)
        print(f"  {ticker}: already exists ({len(existing)} rows) - skipping download")
        continue

    print(f"  Downloading {ticker}...", end=' ', flush=True)
    try:
        df = yf.download(ticker, start='2000-01-01', end='2025-05-01',
                         auto_adjust=False, progress=False)
        if len(df) == 0:
            print(f"FAILED (0 rows returned)")
            continue

        # Flatten MultiIndex columns if present (yfinance >= 0.2 sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
        df.index.name = 'Date'
        df.to_csv(path)
        print(f"OK - {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    except Exception as e:
        print(f"ERROR: {e}")

# ── Step 2: Validate all 10 stocks ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Validating all 10 stocks")
print("=" * 60)

stock_ranges = {}
errors = []

for ticker in TICKERS_ALL:
    path = f'data/raw/{ticker}_raw.csv'
    if not os.path.exists(path):
        errors.append(f"{ticker}: file not found")
        continue

    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')

    # Column normalisation: yfinance version differences
    df.columns = [c.strip() for c in df.columns]
    required = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"{ticker}: missing columns {missing}")
        continue

    checks = {
        'rows >= 3000': len(df) >= 3000,
        'no nulls': df[required].isnull().sum().sum() == 0,
        'all prices > 0': (df[['Adj Close', 'Close', 'High', 'Low', 'Open']] > 0).all().all(),
        'all volumes > 0': (df['Volume'] > 0).all(),
        'dates monotone': df.index.is_monotonic_increasing,
    }

    failed = [k for k, v in checks.items() if not v]
    status = "OK" if not failed else f"FAILED: {failed}"

    adj_min = df['Adj Close'].min()
    adj_max = df['Adj Close'].max()
    vol_min = df['Volume'].min()
    vol_max = df['Volume'].max()
    print(f"  {ticker:6s}: {len(df):5d} rows | {df.index[0].date()} to {df.index[-1].date()} | "
          f"AdjClose=[{adj_min:.2f},{adj_max:.2f}] | Vol=[{vol_min:.0f},{vol_max:.0f}] | {status}")

    if not failed:
        stock_ranges[ticker] = (df.index[0], df.index[-1])
    else:
        errors.append(f"{ticker}: {failed}")

if errors:
    print(f"\n  ERRORS:\n" + "\n".join(f"    {e}" for e in errors))
else:
    print("\n  All 10 stocks passed validation.")

# ── Step 3: Compute intersection date range ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing intersection date range & configs/universe.json")
print("=" * 60)

# Load all close series to find true intersection
closes = {}
for ticker in TICKERS_ALL:
    path = f'data/raw/{ticker}_raw.csv'
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    closes[ticker] = df['Adj Close']

# Align to intersection
panel_close = pd.DataFrame(closes)
panel_close = panel_close.dropna(how='any')  # only dates where ALL 10 have data

common_start = panel_close.index[0].strftime('%Y-%m-%d')
common_end   = panel_close.index[-1].strftime('%Y-%m-%d')
n_common_days = len(panel_close)

print(f"  Intersection: {common_start} to {common_end} ({n_common_days} trading days)")

universe_config = {
    "tickers": TICKERS_ALL,
    "sectors": SECTOR_MAP,
    "common_start_date": common_start,
    "common_end_date":   common_end,
}

with open('configs/universe.json', 'w') as f:
    json.dump(universe_config, f, indent=2)
print("  Saved configs/universe.json")

# ── Step 4: Build panel OHLCV dataset ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Building panel_ohlcv.parquet")
print("=" * 60)

frames = []
for ticker in TICKERS_ALL:
    path = f'data/raw/{ticker}_raw.csv'
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.columns = [c.strip() for c in df.columns]
    df = df[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
    df = df.rename(columns={'Adj Close': 'AdjClose'})
    df = df.loc[common_start:common_end]
    df['ticker'] = ticker
    frames.append(df)

panel = pd.concat(frames)
panel = panel.reset_index().set_index(['Date', 'ticker']).sort_index()

print(f"  Panel shape: {panel.shape}")
panel.to_parquet('data/processed/panel_ohlcv.parquet')
print("  Saved data/processed/panel_ohlcv.parquet")

# ── Step 5: Verify panel integrity ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Verifying panel integrity")
print("=" * 60)

panel = pd.read_parquet('data/processed/panel_ohlcv.parquet')

# Check 1: every ticker has same dates
dates_per_ticker = panel.groupby('ticker').apply(lambda g: g.index.get_level_values('Date').unique())
n_dates_per_ticker = dates_per_ticker.apply(len)
all_same = n_dates_per_ticker.nunique() == 1
print(f"  All tickers have same date count: {'OK' if all_same else 'FAIL'}")
print(f"  Dates per ticker: {n_dates_per_ticker.to_dict()}")

# Check 2: no NaN
nan_count = panel.isnull().sum().sum()
print(f"  NaN count: {nan_count} {'OK' if nan_count == 0 else 'FAIL'}")

# Check 3: exactly 10 rows per date
rows_per_date = panel.groupby('Date').size()
expected_10 = (rows_per_date == 10).all()
print(f"  Exactly 10 rows per date: {'OK' if expected_10 else 'FAIL'} "
      f"(min={rows_per_date.min()}, max={rows_per_date.max()})")

# Check 4: 5 lowest-volume dates (sanity)
total_vol_by_date = panel.groupby('Date')['Volume'].sum()
low_vol_dates = total_vol_by_date.nsmallest(5)
print(f"\n  5 lowest-volume dates (sanity check):")
for d, v in low_vol_dates.items():
    print(f"    {d.date()}: {v:,.0f}")

print("\n" + "=" * 60)
print("PROMPT 1 COMPLETE")
print(f"  Raw files:         {len(TICKERS_ALL)} CSVs in data/raw/")
print(f"  universe.json:     configs/universe.json")
print(f"  Panel shape:       {panel.shape}  (target ~32000×6)")
print(f"  Common date range: {common_start} → {common_end}")
print(f"  NaN in panel:      {nan_count}")
print("=" * 60)
