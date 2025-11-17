# okx-orderbook-csv

Rust utilities that:

1. Convert OKX Level 2 `.data` (newline-delimited JSON) dumps to CSV rows.
2. Rebuild synthetic snapshots every _n_ milliseconds directly from the CSV (re-implementation of `rebuild_snapshots_every_100ms` in `data_utils.py`).
3. Emit feature-ready time series by replaying daily `.tar.gz` L2 archives alongside trade `.zip` files.

## Build

```powershell
cd rust/okx-orderbook-csv
cargo build --release   # binaries under target/release/
```

## Commands

### `convert`

```
cargo run -- convert -i ../../BTC-USDT-L2orderbook-400lv-2025-11-08.data -o ../../btcusdt.csv
```

Options:

- `-i, --input`: `.data` file.
- `-o, --output`: CSV path (defaults to `<input>.csv`).
- `--nrows`: limit NDJSON rows to read.
- `--no-timestamp`: omit RFC3339 timestamp column.

### `rebuild`

Reads the CSV produced by `convert`, replaying snapshots + updates while emitting a snapshot every fixed interval (default 100 ms) for each instrument.

```
cargo run -- rebuild -i ../../btcusdt.csv -o ../../btc.snapshots.csv --freq-ms 100 --depth 20 --max-duration-minutes 10
```

Key parameters:

- `-i, --input`: CSV path from the `convert` command.
- `-o, --output`: Target CSV (defaults to `<input>.snapshots.csv`).
- `--freq-ms`: Interval between emitted snapshots. Default 100.
- `--depth`: Number of price levels kept per side (omit for full depth).
- `--max-duration-minutes`: Process only the first N minutes from the earliest timestamp (use `--max-duration-ms` for millisecond precision). Example above keeps the first 10 minutes.

Output columns match the Python implementation: `instrument,action,side,price,size,count,ts,timestamp` with `action='snapshot'`.

### `features`

Streams daily OKX archives directly, maintaining the live order book (from `/btcusdt_l2/*.tar.gz`) while merging in the executed trades (from `/btcusdt_trade/*.zip`). The output keeps the top *d* bid/ask sizes from L2 snapshots and calculates VWAP plus buy/sell volumes from actual trades inside each interval.

```
cargo run -- features ^
  --l2-dir ../../btcusdt_l2 ^
  --trade-dir ../../btcusdt_trade ^
  --start-date 2025-10-01 ^
  --end-date 2025-10-07 ^
  --freq-ms 1000 ^
  --depth 20 ^
  --instrument BTC-USDT ^
  -o ../../features.csv
```

Key parameters:

- `--l2-dir`: Folder containing `.tar.gz` files (one per day) with Level 2 snapshots/updates.
- `--trade-dir`: Folder containing `.zip` files (one per day) with trade CSVs.
- `--start-date`, `--end-date`: Inclusive UTC range (`YYYY-MM-DD`). The tool auto-loads the previous day’s L2 archive (when available) so the first interval inherits the earlier closing book.
- `--freq-ms`: Feature cadence in milliseconds (default 1000).
- `--depth`: Number of best levels per side to export (default 5).
- `--instrument`: Optional instrument filter (all instruments are processed when omitted).
- `-o, --output`: Destination CSV (defaults to `<l2-dir>/features.csv`).

Each row is `instrument, ts, timestamp, bid_size_1..d, ask_size_1..d, vwap, buy_volume, sell_volume, total_bid_volume, total_ask_volume`. VWAP/buy/sell volumes come strictly from trade prints; depth/total volumes come from reconstructed L2 books.

## Implementation notes

- CSV input is streamed: each instrument keeps only the current timestamp slice plus the order book state (bid/ask maps), so memory usage stays low.
- Price levels are stored in `BTreeMap<OrderedFloat<f64>, f64>` to avoid repeated sorting; emitting a snapshot simply iterates bids descending and asks ascending.
- Data is assumed to be ordered by timestamp within each instrument (as produced by the `convert` command).
- For the `features` command, L2 archives are unpacked on the fly (per day) while trade CSVs are streamed from their `.zip` containers, so no intermediate huge CSVs are required.

