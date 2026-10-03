"""Phase 0.5 funding-arbitrage backtest and go/no-go gate (spec 0.5.3-0.5.9).

Reads data written by download_data.py. No network, no execution capability.

Usage (from repo root):
    python -m research.phase05_funding_gate.backtest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from research.phase05_funding_gate import common as C
from research.phase05_funding_gate import economics as E

MAX_MISSING_RATIO = 0.02            # spec 0.5.2
MIN_SURVIVING_SYMBOLS = 6           # spec 0.5.9
MAX_OOS_FALLBACK_RATIO = 0.10       # spec 0.5.9
MIN_TRADES_PER_PROFITABLE_SYMBOL = 3

TRADE_FIELDS = [
    "symbol", "segment", "threshold", "entry_decision_time", "entry_time",
    "exit_decision_time", "exit_time", "qty", "spot_entry", "perp_entry",
    "spot_exit", "perp_exit", "periods_held", "hours_held", "funding_pnl",
    "basis_pnl", "entry_fees", "exit_fees", "slippage", "net_pnl",
    "forced_exit", "fallback_settlements_held", "max_perp_high",
    "consumption", "adverse_move_fraction", "margin_flagged",
    "allocation_fully_consumed",
]


@dataclass(frozen=True)
class Params:
    lookback: int
    exit_periods: int
    exit_threshold: float
    notional: float
    spot_fee: float
    perp_fee: float
    slippage_bps: float
    leverage: float
    consumption_limit: float


def params_for(cfg: dict, symbol: str) -> Params:
    return Params(
        lookback=int(cfg["entry_lookback_periods"]),
        exit_periods=int(cfg["exit_periods"]),
        exit_threshold=float(cfg["exit_threshold"]),
        notional=float(cfg["notional_per_symbol_usdt"]),
        spot_fee=float(cfg["fees"]["spot_taker"]),
        perp_fee=float(cfg["fees"]["perp_taker"]),
        slippage_bps=C.slippage_bps_for(cfg, symbol),
        leverage=float(cfg["max_leverage"]),
        consumption_limit=float(cfg["diagnostic_margin_consumption_limit"]),
    )


@dataclass
class SymbolData:
    symbol: str
    f_time: np.ndarray
    f_rate: np.ndarray
    f_price: np.ndarray
    f_fallback: np.ndarray
    b_time: np.ndarray
    spot_open: np.ndarray
    spot_close: np.ndarray
    perp_open: np.ndarray
    perp_high: np.ndarray
    perp_close: np.ndarray
    unresolved_settlements: int


# --------------------------------------------------------------------------
# Data loading and quality
# --------------------------------------------------------------------------

def load_raw(data_dir: Path, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = {
        "funding": data_dir / f"{symbol}_funding.csv",
        "perp": data_dir / f"{symbol}_perp_1h.csv",
        "spot": data_dir / f"{symbol}_spot_1h.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files (run download_data first): " + ", ".join(missing)
        )
    return (pd.read_csv(files["funding"]), pd.read_csv(files["perp"]),
            pd.read_csv(files["spot"]))


def prepare_symbol(symbol: str, funding: pd.DataFrame, perp: pd.DataFrame,
                   spot: pd.DataFrame) -> SymbolData:
    perp = perp.drop_duplicates("open_time").sort_values("open_time")
    spot = spot.drop_duplicates("open_time").sort_values("open_time")
    funding = funding.drop_duplicates("funding_time")
    resolved, dropped = E.resolve_funding_prices(funding, perp)

    bars = perp[["open_time", "open", "high", "close"]].rename(
        columns={"open": "perp_open", "high": "perp_high", "close": "perp_close"}
    ).merge(
        spot[["open_time", "open", "close"]].rename(
            columns={"open": "spot_open", "close": "spot_close"}),
        on="open_time", how="inner",
    ).sort_values("open_time").reset_index(drop=True)

    return SymbolData(
        symbol=symbol,
        f_time=resolved["funding_time"].to_numpy(np.int64),
        f_rate=resolved["funding_rate"].to_numpy(float),
        f_price=resolved["funding_price"].to_numpy(float),
        f_fallback=resolved["fallback"].to_numpy(bool),
        b_time=bars["open_time"].to_numpy(np.int64),
        spot_open=bars["spot_open"].to_numpy(float),
        spot_close=bars["spot_close"].to_numpy(float),
        perp_open=bars["perp_open"].to_numpy(float),
        perp_high=bars["perp_high"].to_numpy(float),
        perp_close=bars["perp_close"].to_numpy(float),
        unresolved_settlements=dropped,
    )


def funding_missing_ratio(times: np.ndarray, start_ms: int, end_ms: int) -> float:
    """Estimate missing funding records without assuming a fixed interval.

    Each gap is compared with the rolling median interval around it, so
    symbols that use (or switch between) 1h/4h/8h intervals are handled.
    """
    t = np.sort(times[(times >= start_ms) & (times < end_ms)])
    if len(t) < 2:
        return 1.0
    diffs = np.diff(t).astype(float)
    ref = pd.Series(diffs).rolling(21, center=True, min_periods=1).median().to_numpy()
    missing = float(np.maximum(0, np.round(diffs / ref) - 1).sum())
    missing += max(0.0, round((t[0] - start_ms) / ref[0]) - 1)
    missing += max(0.0, round((end_ms - t[-1]) / ref[-1]) - 1)
    return missing / (len(t) + missing)


def bars_missing_ratio(open_times: np.ndarray, start_ms: int, end_ms: int) -> float:
    expected = (end_ms - start_ms) // C.HOUR_MS
    present = np.unique(open_times[(open_times >= start_ms) & (open_times < end_ms)]).size
    return max(0, expected - present) / expected if expected else 1.0


def check_quality(sd: SymbolData, perp: pd.DataFrame, spot: pd.DataFrame,
                  start_ms: int, end_ms: int) -> dict:
    r_funding = funding_missing_ratio(sd.f_time, start_ms, end_ms)
    r_perp = bars_missing_ratio(perp["open_time"].to_numpy(np.int64), start_ms, end_ms)
    r_spot = bars_missing_ratio(spot["open_time"].to_numpy(np.int64), start_ms, end_ms)
    reasons = []
    for name, ratio in (("funding records", r_funding), ("perp 1h bars", r_perp),
                        ("spot 1h bars", r_spot)):
        if ratio > MAX_MISSING_RATIO:
            reasons.append(f"{name} missing {ratio:.2%} (> {MAX_MISSING_RATIO:.0%})")
    return {
        "symbol": sd.symbol,
        "missing_funding_ratio": r_funding,
        "missing_perp_ratio": r_perp,
        "missing_spot_ratio": r_spot,
        "unresolved_funding_price": sd.unresolved_settlements,
        "included": not reasons,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def _next_bar(bar_times: np.ndarray, t: int) -> int | None:
    """Index of the first common bar whose open time is strictly after t."""
    j = int(np.searchsorted(bar_times, t, side="right"))
    return j if j < len(bar_times) else None


def _close_trade(sd: SymbolData, p: Params, pos: dict, exit_j: int, use_close: bool,
                 forced: bool, exit_decision_time: int | None, segment: str,
                 threshold: float, with_path: bool) -> dict:
    if use_close:
        spot_exit, perp_exit = sd.spot_close[exit_j], sd.perp_close[exit_j]
        exit_time = int(sd.b_time[exit_j]) + C.HOUR_MS
    else:
        spot_exit, perp_exit = sd.spot_open[exit_j], sd.perp_open[exit_j]
        exit_time = int(sd.b_time[exit_j])

    qty = pos["qty"]
    entry_j = pos["entry_j"]
    basis = E.basis_pnl(qty, pos["spot_entry"], spot_exit, pos["perp_entry"], perp_exit)
    entry_fees = E.leg_fees(qty, pos["spot_entry"], pos["perp_entry"], p.spot_fee, p.perp_fee)
    exit_fees = E.leg_fees(qty, spot_exit, perp_exit, p.spot_fee, p.perp_fee)
    slip_entry = E.slippage_cost(p.notional, p.slippage_bps, fills=2)
    slip_exit = E.slippage_cost(p.notional, p.slippage_bps, fills=2)
    net = pos["funding_pnl"] + basis - entry_fees - exit_fees - slip_entry - slip_exit

    max_high = float(sd.perp_high[entry_j:exit_j + 1].max())
    mc = E.margin_consumption(p.notional, p.leverage, qty, pos["perp_entry"], max_high)

    trade = {
        "symbol": sd.symbol,
        "segment": segment,
        "threshold": threshold,
        "entry_decision_time": pos["entry_decision_time"],
        "entry_time": pos["entry_time"],
        "exit_decision_time": exit_decision_time,
        "exit_time": exit_time,
        "qty": qty,
        "spot_entry": pos["spot_entry"],
        "perp_entry": pos["perp_entry"],
        "spot_exit": float(spot_exit),
        "perp_exit": float(perp_exit),
        "periods_held": pos["periods"],
        "hours_held": (exit_time - pos["entry_time"]) / C.HOUR_MS,
        "funding_pnl": pos["funding_pnl"],
        "basis_pnl": basis,
        "entry_fees": entry_fees,
        "exit_fees": exit_fees,
        "slippage": slip_entry + slip_exit,
        "net_pnl": net,
        "forced_exit": forced,
        "fallback_settlements_held": pos["fallback_held"],
        "max_perp_high": max_high,
        "consumption": mc["consumption"],
        "adverse_move_fraction": mc["adverse_move_fraction"],
        "margin_flagged": E.is_consumption_flagged(mc["consumption"], p.consumption_limit),
        "allocation_fully_consumed": mc["consumption"] >= 1.0,
    }

    if with_path:
        # Hourly mark-to-market path for drawdown (bar closes), ending at net.
        entry_costs = entry_fees + slip_entry
        points: dict[int, float] = {}
        sett = pos["settlements"]
        si, accrued = 0, 0.0
        for k in range(entry_j, exit_j):
            t_close = int(sd.b_time[k]) + C.HOUR_MS
            while si < len(sett) and sett[si][0] <= t_close:
                accrued += sett[si][1]
                si += 1
            mtm = qty * ((sd.spot_close[k] - pos["spot_entry"])
                         - (sd.perp_close[k] - pos["perp_entry"]))
            points[t_close] = accrued + mtm - entry_costs
        points[exit_time] = net
        trade["_path"] = points
    return trade


def simulate(sd: SymbolData, p: Params, threshold: float, seg_start: int,
             seg_end: int, segment: str, with_path: bool = False) -> list[dict]:
    """Simulate the positive-funding strategy on one symbol and one segment.

    Decisions occur only at funding settlements inside [seg_start, seg_end)
    and use only rates settled at or before the decision time. The lookback
    may use settlements before seg_start (past data, no look-ahead).
    """
    trades: list[dict] = []
    bt = sd.b_time
    if len(bt) == 0 or len(sd.f_time) == 0:
        return trades

    times, rates = sd.f_time, sd.f_rate
    K, N = p.lookback, p.exit_periods
    first = int(np.searchsorted(times, seg_start, side="left"))
    last = int(np.searchsorted(times, seg_end, side="left"))
    pos: dict | None = None

    for i in range(first, last):
        s = int(times[i])
        rate = float(rates[i])

        if pos is not None:
            # Funding at s is paid only if the position was open before s.
            if s > pos["entry_time"]:
                pay = E.funding_payment_short(rate, pos["qty"], float(sd.f_price[i]))
                pos["funding_pnl"] += pay
                pos["periods"] += 1
                pos["fallback_held"] += int(sd.f_fallback[i])
                pos["settlements"].append((s, pay))
            if i + 1 >= N and bool(np.all(E.below_threshold(rates[i - N + 1:i + 1], p.exit_threshold))):
                j = _next_bar(bt, s)
                if j is None:
                    trades.append(_close_trade(sd, p, pos, len(bt) - 1, True, True, s,
                                               segment, threshold, with_path))
                else:
                    trades.append(_close_trade(sd, p, pos, j, False, False, s,
                                               segment, threshold, with_path))
                pos = None
            continue

        if (i + 1 >= K and rate > 0
                and E.meets_threshold(float(rates[i - K + 1:i + 1].mean()), threshold)):
            j = _next_bar(bt, s)
            if j is None or bt[j] >= seg_end:
                continue
            spot_entry = float(sd.spot_open[j])
            pos = {
                "entry_decision_time": s,
                "entry_j": j,
                "entry_time": int(bt[j]),
                "spot_entry": spot_entry,
                "perp_entry": float(sd.perp_open[j]),
                "qty": p.notional / spot_entry,
                "funding_pnl": 0.0,
                "periods": 0,
                "fallback_held": 0,
                "settlements": [],
            }

    if pos is not None:
        j = int(np.searchsorted(bt, seg_end, side="left"))
        if j < len(bt):
            trades.append(_close_trade(sd, p, pos, j, False, True, None,
                                       segment, threshold, with_path))
        else:
            trades.append(_close_trade(sd, p, pos, len(bt) - 1, True, True, None,
                                       segment, threshold, with_path))
    return trades


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def equity_curve(trades: list[dict]) -> pd.Series:
    paths = [t["_path"] for t in trades if t.get("_path")]
    if not paths:
        return pd.Series(dtype=float)
    index = sorted({ts for path in paths for ts in path})
    total = pd.Series(0.0, index=index)
    for path in paths:
        s = pd.Series(path).sort_index()
        total = total.add(s.reindex(index, method="ffill").fillna(0.0))
    return total


def settlement_counts(sd: SymbolData, seg_start: int, seg_end: int) -> tuple[int, int]:
    mask = (sd.f_time >= seg_start) & (sd.f_time < seg_end)
    return int(mask.sum()), int(sd.f_fallback[mask].sum())


def summarize(trades: list[dict], capital: float, duration_ms: int,
              total_settlements: int, fallback_settlements: int) -> dict:
    n = len(trades)

    def total(key: str) -> float:
        return float(sum(t[key] for t in trades))

    periods = [t["periods_held"] for t in trades]
    hours = [t["hours_held"] for t in trades]
    net = total("net_pnl")
    eq = equity_curve(trades)
    dd = E.max_drawdown(eq.to_numpy()) if len(eq) else 0.0
    return {
        "trades": n,
        "gross_funding": total("funding_pnl"),
        "basis_pnl": total("basis_pnl"),
        "entry_fees": total("entry_fees"),
        "exit_fees": total("exit_fees"),
        "slippage": total("slippage"),
        "net_pnl": net,
        "avg_hold_periods": float(np.mean(periods)) if n else None,
        "median_hold_periods": float(np.median(periods)) if n else None,
        "avg_hold_hours": float(np.mean(hours)) if n else None,
        "median_hold_hours": float(np.median(hours)) if n else None,
        "win_rate": sum(1 for t in trades if t["net_pnl"] > 0) / n if n else None,
        "max_drawdown_usdt": dd,
        "max_drawdown_pct": dd / capital if capital else None,
        "annualized_return": E.annualized_return(net, capital, duration_ms),
        "capital_allocation": capital,
        "margin_flagged": sum(1 for t in trades if t["margin_flagged"]),
        "allocation_fully_consumed": sum(1 for t in trades if t["allocation_fully_consumed"]),
        "max_consumption": max((t["consumption"] for t in trades), default=0.0),
        "forced_exits": sum(1 for t in trades if t["forced_exit"]),
        "settlements": total_settlements,
        "fallback_settlements": fallback_settlements,
        "fallback_settlements_held": int(sum(t["fallback_settlements_held"] for t in trades)),
    }


def select_threshold(datasets: dict[str, SymbolData], params: dict[str, Params],
                     grid: list[float], is_start: int, is_end: int) -> tuple[list[dict], int]:
    """Choose ONE universe-wide threshold using IN-SAMPLE data only (0.5.8)."""
    rows = []
    for thr in grid:
        trades = []
        for sym, sd in datasets.items():
            trades.extend(simulate(sd, params[sym], thr, is_start, is_end, "in_sample"))
        rows.append({
            "threshold": thr,
            "trades": len(trades),
            "net_pnl": float(sum(t["net_pnl"] for t in trades)),
        })
    best = max(range(len(rows)), key=lambda k: (rows[k]["net_pnl"], -k))
    return rows, best


def evaluate_gate(m: dict, cfg: dict) -> dict:
    """Apply spec 0.5.9 gate criteria to OOS metrics."""
    inconclusive = []
    if m["oos_trades"] < cfg["min_oos_trades"]:
        inconclusive.append(
            f"OOS trades {m['oos_trades']} < min_oos_trades {cfg['min_oos_trades']}")
    if m["symbols_included"] < MIN_SURVIVING_SYMBOLS:
        inconclusive.append(
            f"only {m['symbols_included']} symbols survived data checks "
            f"(< {MIN_SURVIVING_SYMBOLS})")
    if m["oos_fallback_ratio"] > MAX_OOS_FALLBACK_RATIO:
        inconclusive.append(
            f"{m['oos_fallback_ratio']:.1%} of OOS settlements used the funding-price "
            f"fallback (> {MAX_OOS_FALLBACK_RATIO:.0%})")

    mult = cfg["cost_stress_multiplier"]
    lower, upper = m["neighbor_oos_net"].get("lower"), m["neighbor_oos_net"].get("upper")
    if lower is None or upper is None:
        neighbor_ok = False
        neighbor_note = "selected threshold is at the grid edge; a neighbour is unavailable"
    else:
        neighbor_ok = lower > 0 and upper > 0
        neighbor_note = f"lower={lower:.2f}, upper={upper:.2f}"

    ann = m["oos_annualized_return"]
    criteria = [
        {"id": 1, "criterion": "Universe OOS net PnL > 0",
         "value": f"{m['oos_net_pnl']:.2f} USDT", "passed": m["oos_net_pnl"] > 0},
        {"id": 2, "criterion": (f">= {cfg['min_profitable_symbols']} symbols with OOS net "
                                f"PnL > 0 and >= {MIN_TRADES_PER_PROFITABLE_SYMBOL} trades"),
         "value": str(m["profitable_symbols"]),
         "passed": m["profitable_symbols"] >= cfg["min_profitable_symbols"]},
        {"id": 3, "criterion": f"Universe OOS net PnL > 0 with fees and slippage x{mult}",
         "value": f"{m['oos_net_pnl_stressed']:.2f} USDT",
         "passed": m["oos_net_pnl_stressed"] > 0},
        {"id": 4, "criterion": "Both neighbouring thresholds: universe OOS net PnL > 0",
         "value": neighbor_note, "passed": neighbor_ok},
        {"id": 5, "criterion": f"OOS annualized return >= {cfg['hurdle_annual_return']:.2%}",
         "value": "n/a" if ann is None else f"{ann:.2%}",
         "passed": ann is not None and ann >= cfg["hurdle_annual_return"]},
        {"id": 6, "criterion": f"OOS max drawdown <= {cfg['max_drawdown_pct']:.2%} of capital",
         "value": f"{m['oos_max_drawdown_pct']:.2%}",
         "passed": m["oos_max_drawdown_pct"] <= cfg["max_drawdown_pct"]},
    ]
    if inconclusive:
        verdict = "INCONCLUSIVE"
    elif all(c["passed"] for c in criteria):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "criteria": criteria,
            "inconclusive_reasons": inconclusive, "action": C.GO_NO_GO_ACTION[verdict]}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def create_run(cfg: dict, reports_dir: Path) -> Path:
    """Pre-register the run: snapshot + hash the config BEFORE any backtest."""
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = C.config_hash(cfg)
    run_dir = reports_dir / f"run_{created}_{digest[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config_snapshot.json").write_text(json.dumps(
        {"config_hash": digest, "code_version": C.CODE_VERSION,
         "created_utc": created, "config": cfg},
        indent=2, default=str), encoding="utf-8")
    return run_dir


def _strip(trade: dict) -> dict:
    return {k: v for k, v in trade.items() if not k.startswith("_")}


def run_backtest(cfg: dict, data_dir: Path, run_dir: Path) -> dict:
    snapshot = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    start_ms, end_ms = C.period_bounds(cfg)
    span_hours = (end_ms - start_ms) // C.HOUR_MS
    split_ms = start_ms + int(round(cfg["is_fraction"] * span_hours)) * C.HOUR_MS

    quality, datasets = [], {}
    for sym in cfg["symbols"]:
        funding, perp, spot = load_raw(data_dir, sym)
        sd = prepare_symbol(sym, funding, perp, spot)
        q = check_quality(sd, perp, spot, start_ms, end_ms)
        quality.append(q)
        if q["included"]:
            datasets[sym] = sd
    params = {sym: params_for(cfg, sym) for sym in datasets}

    grid = list(cfg["entry_threshold_grid"])
    if datasets:
        grid_rows, sel = select_threshold(datasets, params, grid, start_ms, split_ms)
    else:
        grid_rows, sel = [], 0
    threshold = grid[sel]

    segments = {"in_sample": (start_ms, split_ms), "out_of_sample": (split_ms, end_ms)}
    cap = E.capital_allocation(float(cfg["notional_per_symbol_usdt"]),
                               float(cfg["max_leverage"]))
    seg_results: dict[str, dict] = {}
    all_trades: dict[str, list[dict]] = {}
    for seg, (a, b) in segments.items():
        per_symbol, seg_trades = {}, []
        tot_settle = tot_fb = 0
        for sym, sd in datasets.items():
            trades = simulate(sd, params[sym], threshold, a, b, seg, with_path=True)
            n_settle, n_fb = settlement_counts(sd, a, b)
            tot_settle += n_settle
            tot_fb += n_fb
            per_symbol[sym] = summarize(trades, cap, b - a, n_settle, n_fb)
            seg_trades.extend(trades)
        universe = summarize(seg_trades, cap * len(datasets), b - a, tot_settle, tot_fb)
        seg_results[seg] = {"start_ms": a, "end_ms": b,
                            "per_symbol": per_symbol, "universe": universe}
        all_trades[seg] = seg_trades

    oos_trades = all_trades["out_of_sample"]
    neighbors = {}
    for label, idx in (("lower", sel - 1), ("upper", sel + 1)):
        if 0 <= idx < len(grid) and datasets:
            net = 0.0
            for sym, sd in datasets.items():
                net += sum(t["net_pnl"] for t in simulate(
                    sd, params[sym], grid[idx], split_ms, end_ms, "out_of_sample"))
            neighbors[label] = {"threshold": grid[idx], "oos_net_pnl": net}
        else:
            neighbors[label] = None

    mult = float(cfg["cost_stress_multiplier"])
    stressed = float(sum(t["funding_pnl"] + t["basis_pnl"]
                         - mult * (t["entry_fees"] + t["exit_fees"] + t["slippage"])
                         for t in oos_trades))

    breakeven = {}
    for sym, sd in datasets.items():
        p = params[sym]
        rt = E.round_trip_cost_fraction(p.spot_fee, p.perp_fee, p.slippage_bps)
        med = seg_results["out_of_sample"]["per_symbol"][sym]["median_hold_periods"]
        be = E.breakeven_rate(rt, med)
        mask = (sd.f_time >= split_ms) & (sd.f_time < end_ms)
        rates = sd.f_rate[mask]
        breakeven[sym] = {
            "round_trip_cost_fraction": rt,
            "median_hold_periods": med,
            "breakeven_rate_per_period": be,
            "pct_settlements_above_breakeven": (
                float((rates > be).mean()) if be is not None and len(rates) else None),
        }

    oos_per_symbol = seg_results["out_of_sample"]["per_symbol"]
    oos_uni = seg_results["out_of_sample"]["universe"]
    metrics = {
        "oos_trades": oos_uni["trades"],
        "symbols_included": len(datasets),
        "oos_fallback_ratio": (oos_uni["fallback_settlements"] / oos_uni["settlements"]
                               if oos_uni["settlements"] else 0.0),
        "oos_net_pnl": oos_uni["net_pnl"],
        "profitable_symbols": sum(
            1 for s in oos_per_symbol.values()
            if s["net_pnl"] > 0 and s["trades"] >= MIN_TRADES_PER_PROFITABLE_SYMBOL),
        "oos_net_pnl_stressed": stressed,
        "neighbor_oos_net": {k: (v["oos_net_pnl"] if v else None)
                             for k, v in neighbors.items()},
        "oos_annualized_return": oos_uni["annualized_return"],
        "oos_max_drawdown_pct": oos_uni["max_drawdown_pct"] or 0.0,
    }
    gate = evaluate_gate(metrics, cfg)

    flagged = [_strip(t) for seg in all_trades.values() for t in seg if t["margin_flagged"]]
    results = {
        "run_id": run_dir.name,
        "config_hash": snapshot["config_hash"],
        "code_version": snapshot.get("code_version", "0.5.0"),
        "created_utc": snapshot["created_utc"],
        "config": cfg,
        "period": {"start_ms": start_ms, "split_ms": split_ms, "end_ms": end_ms},
        "data_quality": quality,
        "grid": grid_rows,
        "selected_index": sel,
        "selected_threshold": threshold,
        "segments": seg_results,
        "neighbors": neighbors,
        "stress": {"multiplier": mult,
                   "oos_net_pnl": oos_uni["net_pnl"],
                   "oos_net_pnl_stressed": stressed},
        "breakeven": breakeven,
        "margin_flagged_trades": flagged,
        "metrics": metrics,
        "gate": gate,
    }

    with open(run_dir / "trades.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        for seg in ("in_sample", "out_of_sample"):
            for t in all_trades[seg]:
                writer.writerow(_strip(t))
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


def record_run(reports_dir: Path, results: dict) -> list[dict]:
    """Append to runs_index.csv; return previous runs (earlier results kept).

    Older index files without a code_version column are upgraded in place,
    labelling their rows as 0.5.0 (the original method).
    """
    index_path = reports_dir / "runs_index.csv"
    fields = ["run_id", "created_utc", "config_hash", "code_version",
              "selected_threshold", "verdict"]
    previous: list[dict] = []
    if index_path.exists():
        with open(index_path, newline="", encoding="utf-8") as fh:
            previous = list(csv.DictReader(fh))
        for row in previous:
            row.setdefault("code_version", "0.5.0")
            if not row["code_version"]:
                row["code_version"] = "0.5.0"
    rows = previous + [{
        "run_id": results["run_id"], "created_utc": results["created_utc"],
        "config_hash": results["config_hash"],
        "code_version": results.get("code_version", C.CODE_VERSION),
        "selected_threshold": results["selected_threshold"],
        "verdict": results["gate"]["verdict"],
    }]
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return previous


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0.5 funding backtest")
    parser.add_argument("--config", default=None, help="path to phase05.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = C.load_config(args.config)
    except C.ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    data_dir = C.resolve_dir(cfg, "data_dir", "data/phase05")
    reports_dir = C.resolve_dir(cfg, "reports_dir", "reports/phase05")
    run_dir = create_run(cfg, reports_dir)
    try:
        results = run_backtest(cfg, data_dir, run_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    previous = record_run(reports_dir, results)
    (run_dir / "previous_runs.json").write_text(json.dumps(previous, indent=2),
                                                encoding="utf-8")
    print(f"Run: {run_dir}")
    print(f"Selected threshold (in-sample): {results['selected_threshold']}")
    print(f"Verdict: {results['gate']['verdict']} - {results['gate']['action']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
