"""Pure economic calculations for the Phase 0.5 funding gate (spec 0.5.5-0.5.7).

All functions are side-effect free so they can be unit tested in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.phase05_funding_gate.common import YEAR_MS

CONSUMPTION_EPS = 1e-12
THRESHOLD_EPS = 1e-12   # float tolerance for funding-threshold comparisons


def meets_threshold(value, threshold):
    """value >= threshold, tolerant of float noise (e.g. a mean of three
    0.0001 rates computing as 9.999999999999999e-05). Works on arrays."""
    return value >= threshold - THRESHOLD_EPS


def below_threshold(value, threshold):
    """value < threshold, consistent with meets_threshold (its exact negation)."""
    return value < threshold - THRESHOLD_EPS


def funding_payment_short(rate: float, qty: float, funding_price: float) -> float:
    """Funding cash flow to a SHORT linear USDT-M perp position.

    Positive rate -> positive value (income to the short).
    Negative rate -> negative value (cost to the short).
    """
    return rate * qty * funding_price


def basis_pnl(qty: float, spot_entry: float, spot_exit: float,
              perp_entry: float, perp_exit: float) -> float:
    """PnL of LONG spot + SHORT perp from price changes only."""
    return qty * ((spot_exit - spot_entry) - (perp_exit - perp_entry))


def leg_fees(qty: float, spot_price: float, perp_price: float,
             spot_fee: float, perp_fee: float) -> float:
    """Fees for one side (entry OR exit) of both legs: two fills."""
    return qty * (spot_price * spot_fee + perp_price * perp_fee)


def round_trip_fees(qty: float, spot_entry: float, perp_entry: float,
                    spot_exit: float, perp_exit: float,
                    spot_fee: float, perp_fee: float) -> float:
    """Four-fill fees: entry spot, entry perp, exit spot, exit perp."""
    return (leg_fees(qty, spot_entry, perp_entry, spot_fee, perp_fee)
            + leg_fees(qty, spot_exit, perp_exit, spot_fee, perp_fee))


def slippage_cost(notional: float, slippage_bps: float, fills: int = 4) -> float:
    """Slippage approximation: fills * bps/10000 * notional."""
    return fills * slippage_bps / 10_000 * notional


def capital_allocation(notional: float, max_leverage: float) -> float:
    """Research capital allocation (spec 0.5.7).

    Spot leg notional + futures margin allocation. This is NOT the
    exchange's required initial or maintenance margin.
    """
    return notional + notional / max_leverage


def margin_consumption(notional: float, max_leverage: float, qty: float,
                       perp_entry: float, max_perp_high: float) -> dict:
    """Margin-consumption diagnostic (spec 0.5.6). NOT a liquidation model."""
    futures_margin = notional / max_leverage
    adverse_move = max(0.0, max_perp_high - perp_entry)
    adverse_pnl = qty * adverse_move
    consumption = adverse_pnl / futures_margin
    return {
        "futures_margin": futures_margin,
        "adverse_pnl": adverse_pnl,
        "remaining_futures_equity": futures_margin - adverse_pnl,
        "consumption": consumption,
        "adverse_move_fraction": adverse_move / perp_entry if perp_entry else 0.0,
    }


def is_consumption_flagged(consumption: float, limit: float) -> bool:
    return consumption >= limit - CONSUMPTION_EPS


def round_trip_cost_fraction(spot_fee: float, perp_fee: float,
                             slippage_bps: float) -> float:
    """Round-trip cost as a fraction of notional (fees + slippage, 4 fills)."""
    return 2 * spot_fee + 2 * perp_fee + 4 * slippage_bps / 10_000


def breakeven_rate(round_trip_fraction: float, median_periods: float | None) -> float | None:
    """Breakeven funding rate per period (spec 0.5.10)."""
    if median_periods is None or not median_periods > 0:
        return None
    return round_trip_fraction / median_periods


def annualized_return(net_pnl: float, capital: float, duration_ms: int) -> float | None:
    """Simple (non-compounded) annualization by calendar duration."""
    if capital <= 0 or duration_ms <= 0:
        return None
    return (net_pnl / capital) * (YEAR_MS / duration_ms)


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough decline of an equity curve that starts at 0."""
    values = np.concatenate([[0.0], np.asarray(equity, dtype=float)])
    peaks = np.maximum.accumulate(values)
    return float((peaks - values).max())


def resolve_funding_prices(funding: pd.DataFrame,
                           perp_bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach funding_price per settlement (spec 0.5.5 convention).

    markPrice from the funding-history record is authoritative. Only when it
    is missing/empty/non-positive is the perp 1h kline open at or immediately
    after the settlement used, and that settlement is flagged as fallback.
    Settlements with no resolvable price are dropped and counted.
    """
    f = funding.sort_values("funding_time").reset_index(drop=True).copy()
    mark = pd.to_numeric(f["mark_price"], errors="coerce")
    has_mark = mark.notna() & (mark > 0)

    bars = perp_bars.sort_values("open_time")
    bar_times = bars["open_time"].to_numpy(np.int64)
    bar_opens = bars["open"].to_numpy(float)
    idx = np.searchsorted(bar_times, f["funding_time"].to_numpy(np.int64), side="left")
    if len(bar_times):
        fallback_price = np.where(idx < len(bar_times),
                                  bar_opens[np.minimum(idx, len(bar_times) - 1)],
                                  np.nan)
    else:
        fallback_price = np.full(len(f), np.nan)

    f["funding_price"] = np.where(has_mark, mark, fallback_price)
    f["fallback"] = (~has_mark).to_numpy()
    unresolved = f["funding_price"].isna()
    dropped = int(unresolved.sum())
    return f[~unresolved].reset_index(drop=True), dropped
