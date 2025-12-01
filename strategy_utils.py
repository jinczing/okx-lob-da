"""Strategy helpers to marry alpha signals with hftbacktest loops.

This module keeps the numerically heavy backtest logic inside numba-compatible
functions while allowing alphas (regression/classification) to be computed in
regular Python. Signals are passed in as timestamp/value arrays, aligned to the
backtest feed's nanosecond clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numba import njit

from hftbacktest import (
    ALL_ASSETS,
    BacktestAsset,
    GTC,
    LIMIT,
    MARKET,
    HashMapMarketDepthBacktest,
    Recorder,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AlphaMakerConfig:
    """Inventory-aware maker that skews quotes using a signed signal."""

    step_ns: int = 50_000_000
    base_quote_offset_ticks: float = 0.5
    min_spread_ticks: float = 1.0
    skew_scale_ticks: float = 2.0
    max_skew_ticks: float = 3.0
    quote_ttl_ns: int = 200_000_000
    order_size_lots: float = 1.0
    max_inventory_lots: float = 3.0
    record_every_n_steps: int = 2

    def validate(self) -> None:
        if self.step_ns <= 0:
            raise ValueError("step_ns must be > 0")
        if self.quote_ttl_ns <= 0:
            raise ValueError("quote_ttl_ns must be > 0")
        if self.order_size_lots <= 0:
            raise ValueError("order_size_lots must be > 0")
        if self.max_inventory_lots < self.order_size_lots:
            raise ValueError("max_inventory_lots must be >= order_size_lots")


@dataclass
class AlphaTakerConfig:
    """Signal-threshold taker that sends market orders."""

    step_ns: int = 20_000_000
    buy_threshold: float = 0.0005
    sell_threshold: float = -0.0005
    cooldown_ns: int = 150_000_000
    order_size_lots: float = 1.0
    max_inventory_lots: float = 4.0
    slippage_ticks: float = 1.0
    record_every_n_steps: int = 2

    def validate(self) -> None:
        if self.step_ns <= 0:
            raise ValueError("step_ns must be > 0")
        if self.cooldown_ns <= 0:
            raise ValueError("cooldown_ns must be > 0")
        if self.order_size_lots <= 0:
            raise ValueError("order_size_lots must be > 0")
        if self.max_inventory_lots < self.order_size_lots:
            raise ValueError("max_inventory_lots must be >= order_size_lots")


@dataclass
class AlphaRunResult:
    records: np.ndarray
    frame: pd.DataFrame
    summary: Dict[str, float]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def align_signal_to_feed(signal_df: pd.DataFrame, feed_start_ns: int, ts_col: str = "timestamp_ns") -> pd.DataFrame:
    """Shift a timestamped signal so it starts at ``feed_start_ns``.

    Assumes the signal timestamps are monotone increasing. Useful when the
    alpha dataset and backtest feed come from different calendar days.
    """

    aligned = signal_df.copy()
    ts = pd.to_numeric(aligned[ts_col], errors="coerce").astype("int64")
    offset = ts.iloc[0]
    aligned[ts_col] = ts - offset + int(feed_start_ns)
    return aligned


def build_signal_arrays(signal_df: pd.DataFrame, value_col: str, ts_col: str = "timestamp_ns") -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_ns, values) sorted for numba loops."""

    arr = signal_df[[ts_col, value_col]].dropna().sort_values(ts_col)
    return arr[ts_col].to_numpy(np.int64), arr[value_col].to_numpy(np.float64)


def run_alpha_maker_backtest(
    asset_config,
    signal_ts_ns: np.ndarray,
    signal_values: np.ndarray,
    config: AlphaMakerConfig | None = None,
    record_capacity: int = 250_000,
) -> AlphaRunResult:
    cfg = config or AlphaMakerConfig()
    cfg.validate()
    if record_capacity <= 0:
        raise ValueError("record_capacity must be >= 1")

    asset = asset_config.to_asset()
    hbt = HashMapMarketDepthBacktest([asset])
    recorder = Recorder(num_assets=1, record_size=record_capacity)

    try:
        _alpha_maker_loop(
            hbt,
            recorder.recorder,
            signal_ts_ns,
            signal_values,
            cfg.step_ns,
            cfg.base_quote_offset_ticks,
            cfg.min_spread_ticks,
            cfg.skew_scale_ticks,
            cfg.max_skew_ticks,
            cfg.quote_ttl_ns,
            cfg.order_size_lots,
            cfg.max_inventory_lots,
            cfg.record_every_n_steps,
        )
    finally:
        hbt.clear_inactive_orders(ALL_ASSETS)
        hbt.close()

    raw_records = recorder.get(asset_no=0)
    frame = records_to_frame(raw_records)
    summary = build_summary(frame)
    return AlphaRunResult(records=raw_records, frame=frame, summary=summary)


def run_alpha_taker_backtest(
    asset_config,
    signal_ts_ns: np.ndarray,
    signal_values: np.ndarray,
    config: AlphaTakerConfig | None = None,
    record_capacity: int = 150_000,
) -> AlphaRunResult:
    cfg = config or AlphaTakerConfig()
    cfg.validate()
    if record_capacity <= 0:
        raise ValueError("record_capacity must be >= 1")

    asset = asset_config.to_asset()
    hbt = HashMapMarketDepthBacktest([asset])
    recorder = Recorder(num_assets=1, record_size=record_capacity)

    try:
        _alpha_taker_loop(
            hbt,
            recorder.recorder,
            signal_ts_ns,
            signal_values,
            cfg.step_ns,
            cfg.buy_threshold,
            cfg.sell_threshold,
            cfg.cooldown_ns,
            cfg.order_size_lots,
            cfg.max_inventory_lots,
            cfg.slippage_ticks,
            cfg.record_every_n_steps,
        )
    finally:
        hbt.clear_inactive_orders(ALL_ASSETS)
        hbt.close()

    raw_records = recorder.get(asset_no=0)
    frame = records_to_frame(raw_records)
    summary = build_summary(frame)
    return AlphaRunResult(records=raw_records, frame=frame, summary=summary)


# ---------------------------------------------------------------------------
# Utility transforms
# ---------------------------------------------------------------------------


def records_to_frame(records: np.ndarray) -> pd.DataFrame:
    if records.size == 0:
        cols = ["timestamp", "price", "position", "balance", "fee", "num_trades", "trading_volume", "trading_value"]
        return pd.DataFrame(columns=cols + ["equity"])

    frame = pd.DataFrame.from_records(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ns", utc=True)
    frame["equity"] = frame["balance"] + frame["position"] * frame["price"]
    return frame


def build_summary(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {"total_pnl": 0.0, "max_drawdown": 0.0, "num_trades": 0.0, "turnover": 0.0, "final_inventory": 0.0}

    # Some recorders start with a placeholder row (price/balance NaN). Drop NaNs before PnL math.
    equity = frame["equity"].dropna()
    if equity.empty:
        return {"total_pnl": float("nan"), "max_drawdown": float("nan"), "num_trades": 0.0, "turnover": 0.0, "final_inventory": 0.0}

    pnl = float(equity.iloc[-1] - equity.iloc[0])
    drawdown = float((equity.cummax() - equity).max())
    num_trades = float(frame["num_trades"].iloc[-1])
    turnover = float(frame["trading_value"].iloc[-1])
    final_inventory = float(frame["position"].iloc[-1])
    return {
        "total_pnl": pnl,
        "max_drawdown": drawdown,
        "num_trades": num_trades,
        "turnover": turnover,
        "final_inventory": final_inventory,
    }


# ---------------------------------------------------------------------------
# Numba strategy kernels
# ---------------------------------------------------------------------------


@njit
def _advance_signal(now: int, signal_ts: np.ndarray, signal_vals: np.ndarray, idx: int, current: float) -> tuple[int, float]:
    while idx + 1 < len(signal_ts) and signal_ts[idx + 1] <= now:
        idx += 1
        current = signal_vals[idx]
    return idx, current


@njit
def _snap(price: float, tick_size: float) -> float:
    return np.round(price / tick_size) * tick_size


@njit
def _alpha_maker_loop(
    hbt,
    recorder,
    signal_ts: np.ndarray,
    signal_vals: np.ndarray,
    step_ns: int,
    base_quote_offset_ticks: float,
    min_spread_ticks: float,
    skew_scale_ticks: float,
    max_skew_ticks: float,
    quote_ttl_ns: int,
    order_size_lots: float,
    max_inventory_lots: float,
    record_every_n_steps: int,
) -> None:
    asset_no = 0
    buy_id = 0
    sell_id = 0
    buy_px = 0.0
    sell_px = 0.0
    buy_ts = 0
    sell_ts = 0
    next_order_id = 1
    step_counter = 0
    sig_idx = 0
    sig_val = signal_vals[0] if len(signal_vals) > 0 else 0.0

    recorder.record(hbt)

    while True:
        status = hbt.elapse(step_ns)
        if status != 0:
            break

        step_counter += 1
        if step_counter % record_every_n_steps == 0:
            recorder.record(hbt)

        now = hbt.current_timestamp
        sig_idx, sig_val = _advance_signal(now, signal_ts, signal_vals, sig_idx, sig_val)

        depth = hbt.depth(asset_no)
        tick = depth.tick_size
        lot = depth.lot_size
        spread_ticks = (depth.best_ask - depth.best_bid) / tick
        qty = max(order_size_lots * lot, lot)

        # Cancel missing/expired orders
        orders = hbt.orders(asset_no)
        if buy_id != 0 and not orders.__contains__(buy_id):
            buy_id = 0
        if sell_id != 0 and not orders.__contains__(sell_id):
            sell_id = 0

        if buy_id != 0 and now - buy_ts >= quote_ttl_ns:
            hbt.cancel(asset_no, buy_id, True)
            buy_id = 0
        if sell_id != 0 and now - sell_ts >= quote_ttl_ns:
            hbt.cancel(asset_no, sell_id, True)
            sell_id = 0

        # np.clip on scalars is flaky under numba on some platforms; do a manual clamp
        raw_skew = sig_val * skew_scale_ticks
        if raw_skew > max_skew_ticks:
            skew = max_skew_ticks
        elif raw_skew < -max_skew_ticks:
            skew = -max_skew_ticks
        else:
            skew = raw_skew
        bid_offset = max(base_quote_offset_ticks - skew, 0.0) * tick
        ask_offset = max(base_quote_offset_ticks + skew, 0.0) * tick

        target_bid = _snap(depth.best_bid - bid_offset, tick)
        target_ask = _snap(depth.best_ask + ask_offset, tick)

        if buy_id != 0 and np.abs(target_bid - buy_px) >= tick:
            hbt.cancel(asset_no, buy_id, True)
            buy_id = 0
        if sell_id != 0 and np.abs(target_ask - sell_px) >= tick:
            hbt.cancel(asset_no, sell_id, True)
            sell_id = 0

        if spread_ticks >= min_spread_ticks:
            state = hbt.state_values(asset_no)
            max_pos = max_inventory_lots * lot
            inv = state.position

            if buy_id == 0 and inv + qty <= max_pos:
                oid = next_order_id
                next_order_id += 1
                res = hbt.submit_buy_order(asset_no, oid, target_bid, qty, GTC, LIMIT, True)
                if res == 0:
                    buy_id = oid
                    buy_px = target_bid
                    buy_ts = now

            if sell_id == 0 and inv - qty >= -max_pos:
                oid = next_order_id
                next_order_id += 1
                res = hbt.submit_sell_order(asset_no, oid, target_ask, qty, GTC, LIMIT, True)
                if res == 0:
                    sell_id = oid
                    sell_px = target_ask
                    sell_ts = now
        else:
            if buy_id != 0:
                hbt.cancel(asset_no, buy_id, True)
                buy_id = 0
            if sell_id != 0:
                hbt.cancel(asset_no, sell_id, True)
                sell_id = 0


@njit
def _alpha_taker_loop(
    hbt,
    recorder,
    signal_ts: np.ndarray,
    signal_vals: np.ndarray,
    step_ns: int,
    buy_threshold: float,
    sell_threshold: float,
    cooldown_ns: int,
    order_size_lots: float,
    max_inventory_lots: float,
    slippage_ticks: float,
    record_every_n_steps: int,
) -> None:
    asset_no = 0
    step_counter = 0
    sig_idx = 0
    sig_val = signal_vals[0] if len(signal_vals) > 0 else 0.0
    last_trade_ts = -cooldown_ns

    recorder.record(hbt)

    while True:
        status = hbt.elapse(step_ns)
        if status != 0:
            break

        step_counter += 1
        if step_counter % record_every_n_steps == 0:
            recorder.record(hbt)

        now = hbt.current_timestamp
        sig_idx, sig_val = _advance_signal(now, signal_ts, signal_vals, sig_idx, sig_val)
        depth = hbt.depth(asset_no)
        tick = depth.tick_size
        lot = depth.lot_size
        qty = max(order_size_lots * lot, lot)
        state = hbt.state_values(asset_no)
        max_pos = max_inventory_lots * lot
        inv = state.position

        if now - last_trade_ts < cooldown_ns:
            continue

        if sig_val >= buy_threshold and inv + qty <= max_pos:
            price = depth.best_ask + slippage_ticks * tick
            hbt.submit_buy_order(asset_no, now, price, qty, GTC, MARKET, True)
            last_trade_ts = now
        elif sig_val <= sell_threshold and inv - qty >= -max_pos:
            price = depth.best_bid - slippage_ticks * tick
            hbt.submit_sell_order(asset_no, now, price, qty, GTC, MARKET, True)
            last_trade_ts = now
