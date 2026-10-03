"""Build the Phase 0.5 markdown report from a run's results.json (spec 0.5.10).

Usage (from repo root):
    python -m research.phase05_funding_gate.report            # latest run
    python -m research.phase05_funding_gate.report --run-dir reports/phase05/run_...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from research.phase05_funding_gate import common as C

KNOWN_LIMITATIONS = [
    "The 1h bar open is not an executable price; slippage_bps approximates "
    "spread and impact.",
    "No intra-bar execution, latency, or partial-fill modelling.",
    "The margin-consumption check is a conservative diagnostic; liquidation and "
    "maintenance margin are not simulated, and real liquidation would occur "
    "before consumption reaches 1.0.",
    "The fee tier is assumed constant over the period.",
    "Funding PnL uses the 0.5.5 convention; fallback settlements are approximations.",
    "Annualized returns are simple (non-compounded) scalings by calendar duration.",
]


def _num(v, digits=2):
    return "n/a" if v is None else f"{v:,.{digits}f}"


def _pct(v, digits=2):
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def _rate(v):
    return "n/a" if v is None else f"{v * 100:.4f}%"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _segment_tables(seg: dict) -> str:
    rows_pnl, rows_hold = [], []
    entries = list(seg["per_symbol"].items()) + [("UNIVERSE", seg["universe"])]
    for name, s in entries:
        label = f"**{name}**" if name == "UNIVERSE" else name
        rows_pnl.append([label, s["trades"], _num(s["gross_funding"]), _num(s["basis_pnl"]),
                         _num(s["entry_fees"]), _num(s["exit_fees"]), _num(s["slippage"]),
                         _num(s["net_pnl"]), _pct(s["annualized_return"])])
        rows_hold.append([label, _num(s["avg_hold_periods"], 1),
                          _num(s["median_hold_periods"], 1), _num(s["avg_hold_hours"], 1),
                          _num(s["median_hold_hours"], 1), _pct(s["win_rate"], 1),
                          _pct(s["max_drawdown_pct"]), s["margin_flagged"], s["forced_exits"],
                          f"{s['fallback_settlements']}/{s['settlements']}"])
    pnl = table(["Symbol", "Trades", "Gross funding", "Basis PnL", "Entry fees",
                 "Exit fees", "Slippage", "Net PnL", "Ann. return"], rows_pnl)
    hold = table(["Symbol", "Avg hold (periods)", "Median hold (periods)",
                  "Avg hold (h)", "Median hold (h)", "Win rate", "Max DD",
                  "Margin flagged", "Forced exits", "Fallback settlements"], rows_hold)
    return ("PnL in USDT. Ann. return on capital allocation.\n\n" + pnl
            + "\n\n" + hold)


def build_report(results: dict, previous_runs: list[dict] | None = None) -> str:
    cfg = results["config"]
    gate = results["gate"]
    period = results["period"]
    out: list[str] = []

    out.append("# Phase 0.5 Funding Gate Report\n")
    out.append(f"- Run: `{results['run_id']}`")
    out.append(f"- Created: {results['created_utc']}")
    out.append(f"- Config hash (pre-registered before the backtest): "
               f"`{results['config_hash']}`")
    out.append(f"- Code version: {results.get('code_version', '0.5.0')}\n")

    out.append(f"## Verdict: {gate['verdict']}\n")
    out.append(f"**Go/no-go action:** {gate['action']}\n")
    if gate["inconclusive_reasons"]:
        out.append("Inconclusive because:\n")
        out += [f"- {r}" for r in gate["inconclusive_reasons"]]
        out.append("")

    out.append("## Gate criteria (out-of-sample, locked threshold)\n")
    out.append(table(["#", "Criterion", "Value", "Result"],
                     [[c["id"], c["criterion"], c["value"],
                       "PASS" if c["passed"] else "FAIL"] for c in gate["criteria"]]))
    out.append("")

    out.append("## Setup\n")
    out.append(f"- Period: {C.ms_to_iso(period['start_ms'])} to "
               f"{C.ms_to_iso(period['end_ms'])} (end exclusive)")
    out.append(f"- In-sample / OOS split: {C.ms_to_iso(period['split_ms'])} "
               f"(is_fraction = {cfg['is_fraction']})")
    out.append(f"- Selected entry threshold (in-sample only, universe-wide): "
               f"{_rate(results['selected_threshold'])} per period")
    out.append(f"- Entry: mean of last {cfg['entry_lookback_periods']} settled rates >= "
               f"threshold AND last rate > 0. Exit: last {cfg['exit_periods']} rates < "
               f"{_rate(cfg['exit_threshold'])}, or end of segment (forced).")
    out.append(f"- Fees: spot taker {_rate(cfg['fees']['spot_taker'])}, perp taker "
               f"{_rate(cfg['fees']['perp_taker'])}; four fills per round trip.")
    out.append("- Funding-price convention: `markPrice` from the funding-history record "
               "is authoritative; the perp 1h kline open at or after the settlement is "
               "used only when `markPrice` is missing or empty (counted as fallback).")
    out.append("- Capital allocation per symbol = notional + notional / max_leverage = "
               f"{_num(cfg['notional_per_symbol_usdt'] * (1 + 1 / cfg['max_leverage']))} "
               "USDT. This is a research allocation, NOT the exchange's required margin.\n")

    out.append("## Config snapshot\n")
    out.append("```yaml\n" + yaml.safe_dump({"phase05": cfg}, sort_keys=False).rstrip()
               + "\n```\n")

    out.append("## Data quality\n")
    out.append(table(
        ["Symbol", "Missing funding", "Missing perp bars", "Missing spot bars",
         "Unresolved funding price", "Included", "Reason"],
        [[q["symbol"], _pct(q["missing_funding_ratio"]), _pct(q["missing_perp_ratio"]),
          _pct(q["missing_spot_ratio"]), q["unresolved_funding_price"],
          "yes" if q["included"] else "NO", "; ".join(q["reasons"]) or "-"]
         for q in results["data_quality"]]))
    out.append("")

    out.append("## In-sample threshold grid\n")
    out.append(table(["Threshold", "Trades", "Net PnL (USDT)", "Selected"],
                     [[_rate(r["threshold"]), r["trades"], _num(r["net_pnl"]),
                       "<-- selected" if i == results["selected_index"] else ""]
                      for i, r in enumerate(results["grid"])]))
    out.append("")

    out.append("## In-sample results (selected threshold)\n")
    out.append(_segment_tables(results["segments"]["in_sample"]) + "\n")
    out.append("## Out-of-sample results (selected threshold)\n")
    out.append(_segment_tables(results["segments"]["out_of_sample"]) + "\n")

    out.append("## Cost stress (OOS)\n")
    st = results["stress"]
    out.append(f"- Net PnL: {_num(st['oos_net_pnl'])} USDT")
    out.append(f"- Net PnL with fees and slippage x{st['multiplier']}: "
               f"{_num(st['oos_net_pnl_stressed'])} USDT\n")

    out.append("## Neighbouring thresholds (OOS)\n")
    for label in ("lower", "upper"):
        nb = results["neighbors"][label]
        out.append(f"- {label}: " + ("not available (grid edge)" if nb is None else
                   f"{_rate(nb['threshold'])} -> {_num(nb['oos_net_pnl'])} USDT"))
    out.append("")

    out.append("## Breakeven funding rate (OOS)\n")
    out.append("Breakeven per period = round-trip cost fraction / median holding periods.\n")
    out.append(table(
        ["Symbol", "Round-trip cost", "Median hold (periods)", "Breakeven per period",
         "OOS settlements above breakeven"],
        [[sym, _pct(b["round_trip_cost_fraction"], 3), _num(b["median_hold_periods"], 1),
          _rate(b["breakeven_rate_per_period"]), _pct(b["pct_settlements_above_breakeven"], 1)]
         for sym, b in results["breakeven"].items()]))
    out.append("")

    limit = cfg["diagnostic_margin_consumption_limit"]
    out.append("## Margin-consumption diagnostic (not a liquidation model)\n")
    out.append(f"Limit {limit} of the futures allocation at {cfg['max_leverage']}x leverage "
               f"= a perp rise of about {_pct(limit / cfg['max_leverage'], 1)} above entry. "
               "Trades below the limit are not proven safe.\n")
    rows = []
    for seg in ("in_sample", "out_of_sample"):
        for sym, s in results["segments"][seg]["per_symbol"].items():
            rows.append([seg, sym, _num(s["max_consumption"], 3), s["margin_flagged"],
                         s["allocation_fully_consumed"]])
    out.append(table(["Segment", "Symbol", "Max consumption", "Flagged", "Fully consumed"], rows))
    out.append("")
    flagged = results["margin_flagged_trades"]
    if flagged:
        out.append("Flagged trades:\n")
        out.append(table(
            ["Segment", "Symbol", "Entry", "Exit", "Consumption", "Adverse move", "Note"],
            [[t["segment"], t["symbol"], C.ms_to_iso(t["entry_time"]),
              C.ms_to_iso(t["exit_time"]), _num(t["consumption"], 3),
              _pct(t["adverse_move_fraction"], 1),
              "**allocation fully consumed**" if t["allocation_fully_consumed"] else ""]
             for t in flagged]))
    else:
        out.append("No trades flagged.")
    out.append("")

    out.append("## Previous runs\n")
    if previous_runs:
        out.append(table(["Run", "Config hash", "Code version", "Threshold", "Verdict"],
                         [[r["run_id"], r["config_hash"][:12],
                           r.get("code_version") or "0.5.0",
                           r["selected_threshold"], r["verdict"]] for r in previous_runs]))
        this = (results["config_hash"], results.get("code_version", "0.5.0"))
        differs = [r for r in previous_runs
                   if (r["config_hash"], r.get("code_version") or "0.5.0") != this]
        if differs:
            out.append("\nEarlier runs used a different config and/or code version. "
                       "This run is reported separately; earlier results are retained.")
    else:
        out.append("None.")
    out.append("")

    out.append("## Known limitations\n")
    out += [f"- {item}" for item in KNOWN_LIMITATIONS]
    out.append("")
    return "\n".join(out)


def latest_run_dir(reports_dir: Path) -> Path:
    runs = sorted(p for p in reports_dir.glob("run_*") if (p / "results.json").exists())
    if not runs:
        raise FileNotFoundError(f"No completed runs found in {reports_dir}")
    return runs[-1]


def write_report(run_dir: Path) -> Path:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    prev_path = run_dir / "previous_runs.json"
    previous = (json.loads(prev_path.read_text(encoding="utf-8"))
                if prev_path.exists() else [])
    path = run_dir / "report.md"
    path.write_text(build_report(results, previous), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 0.5 report")
    parser.add_argument("--config", default=None, help="path to phase05.yaml")
    parser.add_argument("--run-dir", default=None, help="run directory (default: latest)")
    args = parser.parse_args(argv)
    try:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            cfg = C.load_config(args.config)
            run_dir = latest_run_dir(C.resolve_dir(cfg, "reports_dir", "reports/phase05"))
        path = write_report(run_dir)
    except (C.ConfigError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Report written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
