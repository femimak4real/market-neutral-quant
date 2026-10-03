"""Funding-frequency diagnostic (descriptive only, NOT a gate run).

Explains how often the Phase 0.5 entry conditions were met, per symbol and
per segment, using the cached data and the same IS/OOS split as the gate.
It does not select parameters, does not produce a verdict, and is not
recorded in runs_index.csv.

Usage (from repo root):
    python -m research.phase05_funding_gate.diagnose_funding
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from research.phase05_funding_gate import common as C
from research.phase05_funding_gate import economics as E

DEFAULT_THRESHOLDS = [0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005]
BASELINE_RATE = 0.0001       # Binance's typical neutral rate (0.01% per 8h)
BASELINE_TOL = 1e-8


def segment_stats(times: np.ndarray, rates: np.ndarray, start: int, end: int,
                  lookback: int, thresholds: list[float]) -> dict:
    """Counts for settlements in [start, end); lookback may use earlier data."""
    rolling = pd.Series(rates).rolling(lookback, min_periods=lookback).mean().to_numpy()
    mask = (times >= start) & (times < end) & ~np.isnan(rolling)
    r, m = rates[mask], rolling[mask]
    n = int(mask.sum())
    out = {
        "settlements": n,
        "last_rate_gt_0": int((r > 0).sum()),
        "at_baseline": int((np.abs(r - BASELINE_RATE) <= BASELINE_TOL).sum()),
        "negative": int((r < 0).sum()),
        "mean_rate": float(r.mean()) if n else None,
        "median_rate": float(np.median(r)) if n else None,
        "p90_rate": float(np.percentile(r, 90)) if n else None,
        "mean_ann_pct": float(r.mean() * 3 * 365 * 100) if n else None,
        "mean_k": {},
        "entry_condition": {},
    }
    for thr in thresholds:
        out["mean_k"][thr] = int(E.meets_threshold(m, thr).sum())
        out["entry_condition"][thr] = int((E.meets_threshold(m, thr) & (r > 0)).sum())
    return out


def _pct(count: int, total: int) -> str:
    return f"{count} ({count / total * 100:.1f}%)" if total else "0"


def build_markdown(results: dict, thresholds: list[float], lookback: int,
                   split_iso: str) -> str:
    lines = ["# Funding-frequency diagnostic (descriptive only)", "",
             "Not a gate run. No parameters are selected and no verdict is produced.",
             f"IS/OOS split: {split_iso}. Lookback K = {lookback}.",
             f"'Entry condition' = mean(last {lookback}) >= threshold AND last rate > 0.",
             f"'At baseline' = rate within 1e-8 of {BASELINE_RATE * 100:.4f}%.", ""]
    for seg in ("in_sample", "out_of_sample"):
        lines += [f"## {seg.replace('_', '-').title()}", "",
                  "### Rate distribution", ""]
        hdr = ["Symbol", "Settlements", "Rate > 0", "At baseline", "Negative",
               "Mean", "Median", "P90", "Mean annualized"]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for sym, segs in results.items():
            s = segs[seg]
            n = s["settlements"]
            fmt = lambda v: "n/a" if v is None else f"{v * 100:.4f}%"
            lines.append("| " + " | ".join([
                sym, str(n), _pct(s["last_rate_gt_0"], n), _pct(s["at_baseline"], n),
                _pct(s["negative"], n), fmt(s["mean_rate"]), fmt(s["median_rate"]),
                fmt(s["p90_rate"]),
                "n/a" if s["mean_ann_pct"] is None else f"{s['mean_ann_pct']:.2f}%",
            ]) + " |")
        for key, title in (("mean_k", f"Settlements with mean(last {lookback}) >= threshold"),
                           ("entry_condition", "Settlements meeting the full entry condition")):
            lines += ["", f"### {title}", ""]
            hdr = ["Symbol"] + [f"{t * 100:.4f}%" for t in thresholds]
            lines.append("| " + " | ".join(hdr) + " |")
            lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
            for sym, segs in results.items():
                s = segs[seg]
                lines.append("| " + " | ".join(
                    [sym] + [_pct(s[key][t], s["settlements"]) for t in thresholds]) + " |")
        lines.append("")
    lines += ["Note: entry-condition counts are settlements, not trades. A position, "
              "once open, is held until the exit rule fires, so trades <= entry events.", ""]
    return "\n".join(lines)


def run(cfg: dict, data_dir: Path, out_dir: Path) -> Path:
    start, end = C.period_bounds(cfg)
    span_hours = (end - start) // C.HOUR_MS
    split = start + int(round(cfg["is_fraction"] * span_hours)) * C.HOUR_MS
    thresholds = sorted(set(DEFAULT_THRESHOLDS) | set(cfg["entry_threshold_grid"]))
    lookback = int(cfg["entry_lookback_periods"])

    results = {}
    for sym in cfg["symbols"]:
        path = data_dir / f"{sym}_funding.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (run download_data first)")
        f = (pd.read_csv(path).drop_duplicates("funding_time")
               .sort_values("funding_time").reset_index(drop=True))
        times = f["funding_time"].to_numpy(np.int64)
        rates = f["funding_rate"].to_numpy(float)
        results[sym] = {
            "in_sample": segment_stats(times, rates, start, split, lookback, thresholds),
            "out_of_sample": segment_stats(times, rates, split, end, lookback, thresholds),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"funding_frequency_{stamp}.md"
    path.write_text(build_markdown(results, thresholds, lookback, C.ms_to_iso(split)),
                    encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Funding-frequency diagnostic")
    parser.add_argument("--config", default=None, help="path to phase05.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = C.load_config(args.config)
        data_dir = C.resolve_dir(cfg, "data_dir", "data/phase05")
        out_dir = C.resolve_dir(cfg, "reports_dir", "reports/phase05") / "diagnostics"
        path = run(cfg, data_dir, out_dir)
    except (C.ConfigError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Diagnostic written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
