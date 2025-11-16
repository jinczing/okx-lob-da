# Trade Lab OKX Orderbook Workflow

Utilities, Rust tooling, and notebooks to turn historical OKX Level 2 dumps into analysis-ready CSV files and fixed-interval snapshots.

## Prerequisites

- Rust toolchain (https://rustup.rs) for the `okx-orderbook-csv` binary.
- Python environment with the notebook dependencies from `requirements.txt` (used in the notebooks listed below).
- Downloaded `.data` archives from OKX Historical Data: https://www.okx.com/zh-hant/historical-data.

## 1. Download & Uncompress

1. Request the Level 2 order-book dataset (e.g. `BTC-USDT-L2orderbook-400lv-2025-11-08.zip`) from the OKX portal.
2. Unzip the archive so the raw NDJSON file is available locally: `BTC-USDT-L2orderbook-400lv-2025-11-08.data`.
3. Place the `.data` file anywhere in your workspace (examples below assume `data/raw/`).

## 2. Convert `.data` to flat CSV (Rust)

The `rust/okx-orderbook-csv` crate includes a `convert` subcommand that mirrors `parse_okx_orderbook_file` but runs in Rust for speed.

```powershell
cd rust/okx-orderbook-csv
cargo run --release -- convert `
  --input ..\..\data\raw\BTC-USDT-L2orderbook-400lv-2025-11-08.data `
  --output ..\..\data\processed\btcusdt.csv
```

- Streams NDJSON rows, emits one CSV row per bid/ask level, preserves the raw millisecond `ts`, and adds an RFC3339 `timestamp` column.
- Optional flags: `--nrows 100000` to stop early, `--no-timestamp` to skip the human-readable column.

## 3. Rebuild fixed-interval snapshots (Rust)

Use the `rebuild` subcommand to replay the CSV and emit Level 2 snapshots at a consistent cadence (default 100 ms).

```powershell
cd rust/okx-orderbook-csv
cargo run --release -- rebuild `
  --input ..\..\data\processed\btcusdt.csv `
  --output ..\..\data\snapshots\btc.snapshots.csv `
  --freq-ms 100 `
  --depth 40 `
  --max-duration-minutes 10
```

- Maintains an in-memory book per instrument while streaming the CSV.
- Emits two rows per price level (`side=bid/ask`) with `action=snapshot` and the UTC timestamp.
- `--depth` trims each side to the top *n* levels; `--max-duration-*` options limit how far past the first timestamp to simulate.

## 4. Notebook workflows

- `snapshot-eda.ipynb` – ingests the fixed-interval snapshots and runs basic EDA: depth sanity checks, spread distributions, imbalance/liquidity heatmaps, and quick matplotlib/holoviews charts to validate the reconstructed books.
- `hftbacktest-demo.ipynb` – showcases the `hftbacktest` bindings: converts OKX CSVs into `event_dtype`, wires up `BacktestAsset`/`HashMapMarketDepthBacktest`, and runs the demo strategies from `strategies.py` to highlight latency, queue, and maker/taker behavior.

All notebooks assume processed CSVs live under `data/processed/` and snapshots under `data/snapshots/`; update the first cell if your paths differ.

## Python helpers (`data_utils.py`)

Use the pure-Python utilities when you need quick conversions without rebuilding the Rust binary:

- `parse_okx_orderbook_file(path, as_dataframe=True, convert_timestamp=True, nrows=None)` streams the raw `.data` NDJSON into a pandas DataFrame (or list of dicts) with `instrument`, `action`, `side`, `price`, `size`, `count`, `ts`, and optional UTC timestamps.
- `rebuild_snapshots_from_updates(df, depth=None)` replays parsed updates and emits Level 2 snapshots, optionally trimming to the top *n* levels per side to match the Rust `rebuild` output.
- `rebuild_snapshots_every_100ms(df, freq_ms=100, depth=None)` produces synthetic fixed-cadence snapshots entirely in pandas—ideal for short EDA sessions before running the Rust binary on the full day.
- `demo_convert_okx_csv_for_hftbacktest(csv_path, output_npz=None, limit=1000)` converts the flat CSV into the structured `HFT_EVENT_DTYPE` array expected by `hftbacktest`, optionally persisting it as a compressed `.npz` for lazy loading via `BacktestAsset().data([...])`.

Example:

```python
from data_utils import (
    parse_okx_orderbook_file,
    rebuild_snapshots_from_updates,
    demo_convert_okx_csv_for_hftbacktest,
)

updates = parse_okx_orderbook_file("data/raw/BTC-USDT.data", nrows=500_000)
snapshots = rebuild_snapshots_from_updates(updates, depth=40)
events = demo_convert_okx_csv_for_hftbacktest(
    "data/processed/btcusdt.csv",
    output_npz="btcusdt.npz",
    limit=2_000_000,
)
```

## Tips & Notes

- Keep raw `.data` dumps compressed once converted; the CSV + snapshot files are sufficient for most downstream analyses.
- When batching multiple instruments or days, script the two Rust commands (PowerShell, Makefile, etc.) before launching the notebooks.
- For long sessions, prefer `cargo run --release` (or build once with `cargo build --release`) to keep processing times predictable.
