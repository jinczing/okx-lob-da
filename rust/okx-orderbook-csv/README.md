# okx-orderbook-csv

Rust utilities that:

1. Convert OKX Level 2 `.data` (newline-delimited JSON) dumps to CSV rows.
2. Rebuild synthetic snapshots every _n_ milliseconds directly from the CSV (re-implementation of `rebuild_snapshots_every_100ms` in `data_utils.py`).

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

## Implementation notes

- CSV input is streamed: each instrument keeps only the current timestamp slice plus the order book state (bid/ask maps), so memory usage stays low.
- Price levels are stored in `BTreeMap<OrderedFloat<f64>, f64>` to avoid repeated sorting; emitting a snapshot simply iterates bids descending and asks ascending.
- Data is assumed to be ordered by timestamp within each instrument (as produced by the `convert` command).

