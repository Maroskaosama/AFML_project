"""
Run all per-stock notebooks for every ticker, then run pooled notebooks.

Usage:
    cd AFML_Project
    python notebooks/run_all_stocks.py

Per-stock outputs -> notebooks/outputs/{TICKER}/NB01_*.ipynb ... NB08_*.ipynb
Pooled outputs   -> notebooks/outputs/pooled/NB09_*.ipynb ... NB17_*.ipynb

Requirements: pip install nbconvert nbformat
"""
import json
import os
import sys
import time

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    print("ERROR: Install nbconvert: pip install nbconvert nbformat")
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_ROOT = os.path.join(ROOT, 'notebooks')
OUTPUTS  = os.path.join(NB_ROOT, 'outputs')

with open(os.path.join(ROOT, 'configs', 'universe.json')) as f:
    UNI = json.load(f)
TICKERS = UNI['tickers']

PER_STOCK_NBS = sorted([
    f for f in os.listdir(os.path.join(NB_ROOT, 'per_stock'))
    if f.endswith('.ipynb')
])
POOLED_NBS = sorted([
    f for f in os.listdir(os.path.join(NB_ROOT, 'pooled'))
    if f.endswith('.ipynb')
])

TIMEOUT = 600  # seconds per notebook

def run_notebook(nb_path, out_path, ticker=None):
    """Execute a notebook and save output to out_path."""
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Inject TICKER into the parameters cell if needed
    if ticker:
        for cell in nb.cells:
            if cell.cell_type == 'code' and 'parameters' in cell.get('metadata', {}).get('tags', []):
                cell.source = cell.source.replace("TICKER = 'NVDA'", f"TICKER = '{ticker}'")
                break

    ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name='python3')
    start = time.time()
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
        status = 'OK'
    except Exception as e:
        status = f'ERROR: {e}'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    elapsed = time.time() - start
    print(f"  [{status}] {os.path.basename(nb_path)} ({elapsed:.0f}s)")
    return status == 'OK'

# ── Per-stock notebooks ────────────────────────────────────────────────────────
print("=" * 60)
print("Running per-stock notebooks for all tickers")
print("=" * 60)
per_stock_dir = os.path.join(NB_ROOT, 'per_stock')
for ticker in TICKERS:
    print(f"
--- {ticker} ---")
    out_dir = os.path.join(OUTPUTS, ticker)
    os.makedirs(out_dir, exist_ok=True)
    for nb_name in PER_STOCK_NBS:
        nb_path  = os.path.join(per_stock_dir, nb_name)
        out_path = os.path.join(out_dir, nb_name)
        run_notebook(nb_path, out_path, ticker=ticker)

# ── Pooled notebooks ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Running pooled notebooks")
print("=" * 60)
pooled_dir = os.path.join(NB_ROOT, 'pooled')
out_dir = os.path.join(OUTPUTS, 'pooled')
os.makedirs(out_dir, exist_ok=True)
for nb_name in POOLED_NBS:
    nb_path  = os.path.join(pooled_dir, nb_name)
    out_path = os.path.join(out_dir, nb_name)
    run_notebook(nb_path, out_path)

print("\nDone. Executed notebooks saved to notebooks/outputs/")
