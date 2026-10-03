"""Synthetic data builders for tests. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.phase05_funding_gate.common import HOUR_MS


def make_raw(rates, mark_prices=None, price=100.0, start_ms=0, interval_h=8,
             extra_hours=48, highs=None):
    """Build raw funding/perp/spot frames with constant prices by default."""
    n = len(rates)
    f_times = [start_ms + i * interval_h * HOUR_MS for i in range(n)]
    if mark_prices is None:
        mark_prices = [price] * n
    funding = pd.DataFrame({"funding_time": f_times,
                            "funding_rate": [float(r) for r in rates],
                            "mark_price": mark_prices})
    hours = n * interval_h + extra_hours
    times = [start_ms + k * HOUR_MS for k in range(hours)]
    high = [price] * hours if highs is None else highs
    perp = pd.DataFrame({"open_time": times, "open": price, "high": high,
                         "low": price, "close": price, "volume": 1.0})
    spot = pd.DataFrame({"open_time": times, "open": price, "high": price,
                         "low": price, "close": price, "volume": 1.0})
    return funding, perp, spot


def base_cfg(**overrides):
    cfg = {
        "start_date": "2025-01-01",
        "end_date": "2026-01-01",
        "symbols": ["AAAUSDT"],
        "notional_per_symbol_usdt": 1000,
        "max_leverage": 2.0,
        "diagnostic_margin_consumption_limit": 0.70,
        "fees": {"spot_taker": 0.001, "perp_taker": 0.0005},
        "slippage_bps": {"default": 5},
        "entry_lookback_periods": 3,
        "entry_threshold_grid": [0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005],
        "exit_threshold": 0.0,
        "exit_periods": 2,
        "is_fraction": 0.6,
        "cost_stress_multiplier": 1.5,
        "min_oos_trades": 20,
        "min_profitable_symbols": 3,
        "hurdle_annual_return": 0.05,
        "max_drawdown_pct": 0.05,
    }
    cfg.update(overrides)
    return cfg


def write_symbol_csvs(data_dir, symbol, funding, perp, spot):
    data_dir.mkdir(parents=True, exist_ok=True)
    funding.to_csv(data_dir / f"{symbol}_funding.csv", index=False)
    perp.to_csv(data_dir / f"{symbol}_perp_1h.csv", index=False)
    spot.to_csv(data_dir / f"{symbol}_spot_1h.csv", index=False)


def random_walk_symbol(seed, start_ms, end_ms):
    """Synthetic year of hourly prices and 8h funding with persistent regimes."""
    rng = np.random.default_rng(seed)
    hours = (end_ms - start_ms) // HOUR_MS + 72
    times = start_ms - 24 * HOUR_MS + np.arange(hours) * HOUR_MS
    spot = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, hours)))
    perp = spot * (1 + rng.normal(0.0002, 0.0002, hours))
    perp_df = pd.DataFrame({"open_time": times, "open": perp, "high": perp * 1.002,
                            "low": perp * 0.998, "close": perp, "volume": 1.0})
    spot_df = pd.DataFrame({"open_time": times, "open": spot, "high": spot * 1.002,
                            "low": spot * 0.998, "close": spot, "volume": 1.0})
    f_times = np.arange(start_ms - 7 * 24 * HOUR_MS, end_ms, 8 * HOUR_MS)
    regime = np.repeat(rng.choice([0.0003, 0.0001, -0.00005], size=len(f_times) // 20 + 1),
                       20)[:len(f_times)]
    rates = regime + rng.normal(0, 0.00005, len(f_times))
    idx = np.searchsorted(times, f_times)
    marks = perp[np.minimum(idx, len(perp) - 1)]
    marks = marks.astype(object)
    marks[:5] = ""  # a few older records without markPrice -> fallback
    funding_df = pd.DataFrame({"funding_time": f_times, "funding_rate": rates,
                               "mark_price": marks})
    return funding_df, perp_df, spot_df
