
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binance Order Book Recorder (Spot)

Features:
- Connects to Binance Spot WebSocket depth stream (100ms) for a symbol (e.g., BTCUSDT).
- Records raw incremental depth updates to gzipped NDJSON (one JSON per line).
- (Optional) Reconstructs the in-memory L2 order book using the official snapshot + diff procedure,
  and emits periodic CSV snapshots of top-N levels.
- Auto-reconnect with exponential backoff.
- Graceful shutdown (Ctrl+C).

Usage examples:
  # Record raw deltas for BTCUSDT
  python binance_orderbook_recorder.py --symbol BTCUSDT --mode raw --out ./data

  # Reconstruct in-memory book and write top 50 levels every second
  python binance_orderbook_recorder.py --symbol BTCUSDT --mode book --top-n 50 --snapshot-interval 1.0 --out ./data

  # Poll REST snapshots (full depth) every 5 seconds
  python binance_orderbook_recorder.py --symbol BTCUSDT --mode snapshot --snapshot-interval 5 --limit-snapshot 1000 --out ./data

Requirements (see requirements.txt):
  websockets
  aiohttp

Notes:
- This script targets Binance **Spot**. For USDⓈ-M Futures, the endpoints differ:
  * REST snapshot:  https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=1000
  * WS stream:      wss://fstream.binance.com/ws/btcusdt@depth@100ms  (or depth20@100ms)
  You can adapt the endpoints below if needed.
"""

import argparse
import asyncio
import gzip
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import aiohttp
import websockets

# ---------- Config ----------

SPOT_REST_DEPTH_URL = "https://api.binance.com/api/v3/depth"  # ?symbol=BTCUSDT&limit=1000
SPOT_WS_URL_TMPL = "wss://stream.binance.com:9443/ws/{stream}"  # stream: e.g., btcusdt@depth@100ms

# Increase decimal precision to safely handle many price steps
getcontext().prec = 28

# ---------- Utilities ----------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def rotate_file_prefix(out_dir: Path, symbol: str, suffix: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"{symbol.upper()}_{suffix}_{ts}"
    return out_dir / fname

@dataclass
class OrderBookSide:
    """Maintains one side of the order book as a mapping price->qty (Decimal).
    For bids: highest price first; for asks: lowest price first.
    """
    is_bid: bool
    levels: Dict[Decimal, Decimal] = field(default_factory=dict)

    def apply(self, updates: List[Tuple[Decimal, Decimal]]):
        """Apply price level updates:
        - If qty == 0 -> remove level if exists
        - Else set/replace level to qty
        """
        for px, qty in updates:
            if qty == 0:
                self.levels.pop(px, None)
            else:
                self.levels[px] = qty

    def top_n(self, n: int) -> List[Tuple[Decimal, Decimal]]:
        # Sort based on price; bids descending, asks ascending
        reverse = self.is_bid
        items = sorted(self.levels.items(), key=lambda kv: kv[0], reverse=reverse)
        return items[:n]

@dataclass
class OrderBook:
    bids: OrderBookSide = field(default_factory=lambda: OrderBookSide(is_bid=True))
    asks: OrderBookSide = field(default_factory=lambda: OrderBookSide(is_bid=False))
    last_update_id: Optional[int] = None

    def apply_diff(self, event: dict):
        """Apply a Binance depthUpdate event to our book (Decimal-safe)."""
        bid_updates = [(Decimal(px), Decimal(qty)) for px, qty in event.get("b", [])]
        ask_updates = [(Decimal(px), Decimal(qty)) for px, qty in event.get("a", [])]
        self.bids.apply(bid_updates)
        self.asks.apply(ask_updates)
        self.last_update_id = event["u"]

# ---------- Recorder ----------

class Recorder:
    def __init__(
        self,
        symbol: str,
        out_dir: Path,
        mode: str = "raw",
        top_n: int = 50,
        snapshot_interval: float = 1.0,
        log_level: str = "INFO",
        limit_snapshot: int = 1000,
        ws_speed: str = "100ms",
    ):
        self.symbol = symbol.upper()
        self.stream = f"{self.symbol.lower()}@depth@{ws_speed}"  # depth diffs @100ms
        self.out_dir = out_dir
        self.mode = mode
        self.top_n = top_n
        self.snapshot_interval = snapshot_interval
        self.limit_snapshot = limit_snapshot
        self.ws_url = SPOT_WS_URL_TMPL.format(stream=self.stream)
        self.stop = asyncio.Event()
        self._raw_fp = None  # type: Optional[gzip.GzipFile]
        self._book_fp = None  # type: Optional[gzip.GzipFile]
        self._snapshot_fp = None  # type: Optional[gzip.GzipFile]
        self.book = OrderBook()
        self.buffer = deque()  # buffer events until snapshot is loaded
        self.session: Optional[aiohttp.ClientSession] = None

        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # ----- File management -----

    def _open_raw_file(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        prefix = rotate_file_prefix(self.out_dir, self.symbol, "depth_raw")
        path = Path(str(prefix) + ".ndjson.gz")
        logging.info(f"Writing raw deltas to: {path}")
        self._raw_fp = gzip.open(path, mode="ab")

    def _open_book_file(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        prefix = rotate_file_prefix(self.out_dir, self.symbol, f"book_top{self.top_n}")
        path = Path(str(prefix) + ".csv.gz")
        logging.info(f"Writing book snapshots to: {path}")
        # CSV header
        self._book_fp = gzip.open(path, mode="ab")
        header_cols = ["ts_utc", "event_time"]
        for i in range(1, self.top_n + 1):
            header_cols += [f"bid{i}_px", f"bid{i}_qty"]
        for i in range(1, self.top_n + 1):
            header_cols += [f"ask{i}_px", f"ask{i}_qty"]
        header_line = ",".join(header_cols) + "\n"
        self._book_fp.write(header_line.encode("utf-8"))
        self._book_fp.flush()

    def _open_snapshot_file(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        prefix = rotate_file_prefix(self.out_dir, self.symbol, f"depth_snapshot_{self.limit_snapshot}")
        path = Path(str(prefix) + ".ndjson.gz")
        logging.info(f"Writing order book snapshots to: {path}")
        self._snapshot_fp = gzip.open(path, mode="ab")

    def _close_files(self):
        if self._raw_fp is not None:
            self._raw_fp.close()
            self._raw_fp = None
        if self._book_fp is not None:
            self._book_fp.close()
            self._book_fp = None
        if self._snapshot_fp is not None:
            self._snapshot_fp.close()
            self._snapshot_fp = None

    # ----- Snapshot loading (REST) -----

    async def _fetch_snapshot(self) -> dict:
        assert self.session is not None
        url = SPOT_REST_DEPTH_URL
        params = {"symbol": self.symbol, "limit": str(self.limit_snapshot)}
        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            snap = await resp.json()
            return snap

    def _load_snapshot_into_book(self, snapshot: dict):
        self.book = OrderBook()  # reset
        bids = [(Decimal(px), Decimal(qty)) for px, qty in snapshot.get("bids", [])]
        asks = [(Decimal(px), Decimal(qty)) for px, qty in snapshot.get("asks", [])]
        self.book.bids.apply(bids)
        self.book.asks.apply(asks)
        self.book.last_update_id = snapshot["lastUpdateId"]

    # ----- WS consume -----

    async def _consume_ws(self):
        """Connect to WS and consume messages. Raw mode writes deltas directly.
        Book mode buffers until snapshot sync is established, then applies diffs to the in-memory book.
        """
        backoff = 1.0
        max_backoff = 30.0

        while not self.stop.is_set():
            try:
                logging.info(f"Connecting WS: {self.ws_url}")
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    logging.info("WS connected.")
                    backoff = 1.0  # reset backoff on success
                    while not self.stop.is_set():
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)
                        if msg is None:
                            continue
                        data = json.loads(msg)

                        # Expect a depthUpdate event
                        if data.get("e") != "depthUpdate":
                            continue

                        if self.mode == "raw":
                            await self._write_raw_event(data)
                        else:
                            # Book mode: buffer until we have loaded snapshot+synced
                            self.buffer.append(data)
                            await self._maybe_sync_and_apply()
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidStatus) as e:
                logging.warning(f"WS error: {e}. Reconnecting soon...")
            except Exception as e:
                logging.exception(f"Unexpected error in WS consume: {e}")
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)

    async def _write_raw_event(self, data: dict):
        """Write raw depthUpdate event to NDJSON (gz)."""
        if self._raw_fp is None:
            self._open_raw_file()
        rec = {
            "ts_utc": utc_now_iso(),
            "event_time": data.get("E"),
            "symbol": data.get("s"),
            "U": data.get("U"),
            "u": data.get("u"),
            "b": data.get("b"),
            "a": data.get("a"),
        }
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        self._raw_fp.write(line.encode("utf-8"))

    async def _write_snapshot_event(self, snapshot: dict):
        if self._snapshot_fp is None:
            self._open_snapshot_file()

        record = {
            "ts_utc": utc_now_iso(),
            "symbol": self.symbol,
            "lastUpdateId": snapshot.get("lastUpdateId"),
            "bids": snapshot.get("bids", []),
            "asks": snapshot.get("asks", []),
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._snapshot_fp.write(line.encode("utf-8"))
        self._snapshot_fp.flush()

    async def _maybe_sync_and_apply(self):
        """Implements Binance snapshot+diff synchronization for book reconstruction.
        Steps:
          1) When buffer receives first messages, fetch REST snapshot.
          2) Drop any event with u <= lastUpdateId.
          3) The first event to apply must satisfy U <= lastUpdateId + 1 <= u.
          4) After applying, ensure no gaps: for each subsequent event, event.U must be last_u + 1 (or overlap).
             If a gap is detected, resync by clearing buffer and reloading snapshot.
        """
        # If we don't yet have a snapshot, fetch it
        if self.book.last_update_id is None:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            try:
                snap = await self._fetch_snapshot()
                self._load_snapshot_into_book(snap)
                logging.info(f"Loaded snapshot lastUpdateId={self.book.last_update_id}")
            except Exception as e:
                logging.warning(f"Snapshot fetch failed: {e}")
                return  # try again next call

        # Drop events that are too old
        while self.buffer and self.buffer[0]["u"] <= self.book.last_update_id:
            self.buffer.popleft()

        # Now apply first event that bridges the snapshot
        applied_any = False
        while self.buffer:
            evt = self.buffer[0]
            U, u = evt["U"], evt["u"]
            target = self.book.last_update_id + 1 if self.book.last_update_id is not None else None

            # The first event to apply must satisfy: U <= lastUpdateId + 1 <= u
            if target is not None and U <= target <= u:
                # Apply and pop
                self.book.apply_diff(evt)
                self.buffer.popleft()
                applied_any = True
                break
            else:
                # If event is behind snapshot target, drop; if ahead, wait for more
                if u < target:
                    self.buffer.popleft()
                else:
                    return

        if not applied_any:
            return

        # Apply the remaining events, ensuring no gaps
        last_u = self.book.last_update_id
        while self.buffer:
            evt = self.buffer[0]
            U, u = evt["U"], evt["u"]

            # Accept either exact continuity or overlapping ranges
            if U <= (last_u + 1) <= u:
                self.book.apply_diff(evt)
                last_u = self.book.last_update_id
                self.buffer.popleft()
            elif u <= last_u:
                # Old event; drop
                self.buffer.popleft()
            else:
                # Gap detected; resync
                logging.warning("Gap detected in depth stream. Resynchronizing...")
                self.book.last_update_id = None
                self.buffer.clear()
                break

    # ----- Snapshot writer task -----

    async def _book_snapshot_task(self):
        """Periodically write top-N snapshot to CSV (gz)."""
        if self._book_fp is None:
            self._open_book_file()

        while not self.stop.is_set():
            try:
                if self.book.last_update_id is not None:
                    bids = self.book.bids.top_n(self.top_n)
                    asks = self.book.asks.top_n(self.top_n)
                    # Pad if fewer levels available
                    if len(bids) < self.top_n:
                        bids = bids + [(Decimal("0"), Decimal("0"))] * (self.top_n - len(bids))
                    if len(asks) < self.top_n:
                        asks = asks + [(Decimal("0"), Decimal("0"))] * (self.top_n - len(asks))

                    row = [utc_now_iso(), str(self.book.last_update_id)]
                    for px, qty in bids:
                        row += [format(px, 'f'), format(qty, 'f')]
                    for px, qty in asks:
                        row += [format(px, 'f'), format(qty, 'f')]
                    line = ",".join(row) + "\n"
                    self._book_fp.write(line.encode("utf-8"))
                await asyncio.sleep(self.snapshot_interval)
            except Exception as e:
                logging.warning(f"Snapshot task error: {e}")
                await asyncio.sleep(self.snapshot_interval)

    async def _periodic_snapshot_task(self):
        """Fetch full depth snapshots via REST at a fixed interval and persist them."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

        while not self.stop.is_set():
            try:
                snapshot = await self._fetch_snapshot()
                await self._write_snapshot_event(snapshot)
            except Exception as e:
                logging.warning(f"Snapshot polling error: {e}")
            await asyncio.sleep(self.snapshot_interval)

    # ----- Orchestration -----

    async def run(self):
        loop = asyncio.get_event_loop()

        def _graceful_shutdown():
            logging.info("Shutdown requested. Closing...")
            self.stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _graceful_shutdown)
            except NotImplementedError:
                pass  # Windows

        ws_task = asyncio.create_task(self._consume_ws())
        snapshot_task = None
        if self.mode == "book":
            snapshot_task = asyncio.create_task(self._book_snapshot_task())
        elif self.mode == "snapshot":
            ws_task.cancel()
            snapshot_task = asyncio.create_task(self._periodic_snapshot_task())
            ws_task = None

        await self.stop.wait()
        if ws_task:
            ws_task.cancel()
        if snapshot_task:
            snapshot_task.cancel()
        await asyncio.gather(*(t for t in [ws_task, snapshot_task] if t), return_exceptions=True)

        self._close_files()
        if self.session:
            await self.session.close()
        logging.info("Stopped.")

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Binance Spot Order Book Recorder")
    p.add_argument("--symbol", required=True, help="Trading pair symbol, e.g., BTCUSDT")
    p.add_argument("--mode", choices=["raw", "book", "snapshot"], default="raw", help="raw: record deltas; book: in-memory reconstruction + top-N CSV; snapshot: periodic REST depth snapshots")
    p.add_argument("--top-n", type=int, default=50, help="Top N levels to write in book mode")
    p.add_argument("--snapshot-interval", type=float, default=1.0, help="Seconds between book snapshots (book mode)")
    p.add_argument("--limit-snapshot", type=int, default=1000, help="REST snapshot depth limit (max 5000; 1000 recommended)")
    p.add_argument("--ws-speed", choices=["100ms", "1000ms"], default="100ms", help="WebSocket update speed")
    p.add_argument("--out", default="./data", help="Output directory")
    p.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING...")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    rec = Recorder(
        symbol=args.symbol,
        out_dir=out_dir,
        mode=args.mode,
        top_n=args.top_n,
        snapshot_interval=args.snapshot_interval,
        limit_snapshot=args.limit_snapshot,
        ws_speed=args.ws_speed,
        log_level=args.log_level,
    )
    asyncio.run(rec.run())

if __name__ == "__main__":
    main()
