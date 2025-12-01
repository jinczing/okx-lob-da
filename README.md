# Trade Lab OKX Feature Workflow

This repository focuses on transforming OKX BTC-USDT Level 2 book dumps and trade archives into aligned, fixed-interval feature datasets ready for research or downstream model training. The heavy lifting is handled by the Rust binary `okx-features`, while the accompanying notebook (`feature_lab.ipynb`) is used for inspection and lightweight analysis.

## Prerequisites

- **Rust toolchain:** Install via [rustup.rs](https://rustup.rs). Once installed, run `rustup update` so `cargo run` is available in your shell.
- **OKX historical data:** Download daily L2 archives (`BTC-USDT-L2orderbook-400lv-YYYY-MM-DD.tar.gz`) and trade archives (`BTC-USDT-trades-YYYY-MM-DD.zip`) from the OKX Historical Data portal. Place them under `btcusdt_l2/` and `btcusdt_trade/` respectively, one file per day (already the default layout in this repo).
- **Python environment (optional):** Only required if you plan to open `feature_lab.ipynb`. Install the packages listed in `requirements.txt`.

## Directory Expectations

```
btcusdt_l2/
    BTC-USDT-L2orderbook-400lv-2025-10-01.tar.gz
    ...
btcusdt_trade/
    BTC-USDT-trades-2025-10-01.zip
    ...
rust/
    okx-features/   # Rust crate with the CLI described below
features/          # Suggested output directory (created automatically)
```

The order book archives contain NDJSON snapshots/updates, while the trade ZIP files contain CSV rows where each file spans 16:00 UTC (prev day) to 15:59:59 UTC (current day). The `okx-features` CLI understands this layout and stitches the correct windows automatically.

## Generating Features with `okx-features`

Compile and run directly with Cargo:

```powershell
cargo run -p okx-features -- \
  --start 2025-10-01 --end 2025-10-03 `
  --freq 1s --depth 20 --include-price `
  --l2-dir btcusdt_l2 --trade-dir btcusdt_trade `
  --format parquet --days-per-file 1 `
  --output-dir features
```

Key options:

- `--start/--end` (UTC days) define the inclusive date range.
- `--freq` controls the sampling interval (e.g., `1s`, `500ms`).
- `--depth` sets how many book levels per side are exported.
- `--include-price` also writes `ask_price_n`/`bid_price_n` columns for each level (sizes are always exported).
- `--format` chooses between `csv` and `parquet`. Both formats are written into the `--output-dir` directory.
- `--days-per-file` groups multiple days into a single chunk. `1` means one file per day; `3` would produce rolling 3-day files.
- The tool automatically handles trade warmups/carry-over to ensure VWAP/buy/sell volumes are continuous at day boundaries and emits the final snapshot at exactly `t+1 00:00:00`.

Every output row contains:

- `timestamp` (ms) and ISO8601 string
- Rolling VWAP, buy volume, sell volume over the last sampling interval (VWAP is `-1` if no trades occurred in the window)
- Bid/ask sizes for the requested depth (`ask_size_1 ... ask_size_n`, `bid_size_1 ... bid_size_n`)
- When `--include-price` is set, bid/ask prices for each level (`ask_price_1 ... ask_price_n`, `bid_price_1 ... bid_price_n`)

CSV outputs create `features-YYYY-MM-DD-YYYY-MM-DD.csv` files inside the chosen directory. Parquet outputs follow the same naming convention but are Arrow-native `.parquet` files, so the directory can be treated as a partitioned dataset.

## Building hftbacktest feeds with `okx-backtest`

Use the new Rust binary to stream raw L2 + trade archives into an `.npz` that matches `hftbacktest`'s event dtype (`data[ev, exch_ts, local_ts, px, qty, order_id, ival, fval]`, ns timestamps):

```powershell
cargo run -p okx-backtest -- \
  --start 2025-10-01 --end 2025-10-02 `
  --depth 50 --l2-dir btcusdt_l2 --trade-dir btcusdt_trade `
  --output-dir backtests --instrument BTC-USDT `
  --local-latency-ns 0ns --days-per-file 1
```

Notes:

- Trades are stitched the same way as `okx-features`: the `end+1` trade archive is read so the 16:00–23:59 slice of the last day is included; data before `start 00:00` is dropped.
- `--depth` truncates each message to the first N ask/bid levels (as published by OKX) to keep event volume manageable.
- `--local-latency-ns` shifts `local_ts` relative to `exch_ts` for latency/queue modelling; `--skip-trades` emits depth-only feeds if you only want book events.
- Output files are compressed NPZ by default; add `--uncompressed` if you prefer plain ZIP storage.

## Notebook

- **feature_lab.ipynb** – Explore, visualize, or post-process the generated feature files. Update the filepaths in the first cell if you saved the features somewhere other than `features/`.

## Tips

- Running with `cargo run --release` (or building once with `cargo build --release`) significantly speeds up multi-day processing.
- If you only need CSV output, use `--format csv` and point `--output-dir` to a clean directory; the CLI will create it on demand.
- Keep the raw archives compressed after extraction—the `okx-features` binary streams from the compressed `.tar.gz`/`.zip` files directly, so no intermediate unpacking is required.
