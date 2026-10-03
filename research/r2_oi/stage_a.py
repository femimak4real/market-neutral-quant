"""R2 Stage A: data feasibility (spec R2.4). Produces FEASIBLE or NOT FEASIBLE.

Stage A never runs Stage B. Stage B is not built; it requires explicit
authorization after this report has been reviewed.

Usage (from repo root):
    python -m research.r2_oi.stage_a
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import requests

from research.phase05_funding_gate import common as C
from research.phase05_funding_gate.backtest import bars_missing_ratio, funding_missing_ratio
from research.phase05_funding_gate.download_data import RegionBlockedError, fetch_funding
from research.r2_oi import archive as A
from research.r2_oi import config as R
from research.r2_oi import universe as U

FEASIBLE = "FEASIBLE"
NOT_FEASIBLE = "NOT FEASIBLE"

# Machine-readable per-symbol exclusion reasons (Stage A data checks).
METRICS_DAY_COVERAGE_BELOW_MIN = "METRICS_DAY_COVERAGE_BELOW_MIN"
METRICS_MEDIAN_ROWS_BELOW_MIN = "METRICS_MEDIAN_ROWS_BELOW_MIN"
METRICS_OI_NULL_OR_ZERO_ABOVE_MAX = "METRICS_OI_NULL_OR_ZERO_ABOVE_MAX"
METRICS_CADENCE_COARSER_THAN_5M = "METRICS_CADENCE_COARSER_THAN_5M"
KLINES_MISSING_ABOVE_MAX = "KLINES_MISSING_ABOVE_MAX"
FUNDING_UNAVAILABLE = "FUNDING_UNAVAILABLE"
FUNDING_MISSING_ABOVE_MAX = "FUNDING_MISSING_ABOVE_MAX"


# --------------------------------------------------------------------------
# Funding (Phase 0.5 downloader conventions, R2 cache location)
# --------------------------------------------------------------------------

def make_funding_fetcher(cache_dir: Path):
    session = requests.Session()
    session.headers["User-Agent"] = "r2-oi-stage-a/0.1"

    def fetch(symbol: str, a_ms: int, b_ms: int) -> pd.DataFrame:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{symbol}_funding.csv"
        meta = cache_dir / f"{symbol}_funding_meta.json"
        if path.exists() and meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("start_ms", 1 << 62) <= a_ms and m.get("end_ms", 0) >= b_ms:
                return pd.read_csv(path)
        df = fetch_funding(session, symbol, a_ms, b_ms)
        df.to_csv(path, index=False)
        meta.write_text(json.dumps({"start_ms": a_ms, "end_ms": b_ms}), encoding="utf-8")
        return df
    return fetch


# --------------------------------------------------------------------------
# Per-symbol evaluation
# --------------------------------------------------------------------------

def metrics_stats(results: list[A.FileResult], days_required: int, sa: dict) -> dict:
    counts = A.status_counts(results)
    rows_per_day, gap_days, insufficient_days, unparseable = [], 0, 0, 0
    times, oi, oiv = [], [], []
    gap_ms = sa["gap_report_minutes"] * 60_000
    for res in results:
        if res.status != A.VALID:
            continue
        try:
            df = A.parse_metrics_csv(A.read_zip_text(res.path))
        except Exception as exc:                      # counted as corrupt
            unparseable += 1
            res.status, res.detail = A.CORRUPT_OR_EMPTY, f"unparseable: {exc}"[:200]
            continue
        n = len(df)
        rows_per_day.append(n)
        if n < sa["min_median_rows_per_day"]:
            insufficient_days += 1
        t = np.sort(df["create_time"].to_numpy(np.int64))
        if n >= 2 and np.diff(t).max() > gap_ms:
            gap_days += 1
        times.append(t)
        oi.append(df["sum_open_interest"].to_numpy(float))
        oiv.append(df["sum_open_interest_value"].to_numpy(float))
    counts[A.VALID] -= unparseable
    counts[A.CORRUPT_OR_EMPTY] += unparseable
    all_t = np.unique(np.concatenate(times)) if times else np.array([], dtype=np.int64)
    oi_all = np.concatenate(oi) if oi else np.array([])
    oiv_all = np.concatenate(oiv) if oiv else np.array([])

    def bad_ratio(arr):
        return float((np.isnan(arr) | (arr <= 0)).mean()) if arr.size else 1.0

    return {
        "days_required": days_required,
        "valid_days": counts[A.VALID],
        "file_missing_days": counts[A.FILE_MISSING],
        "download_failed_days": counts[A.DOWNLOAD_FAILED],
        "corrupt_or_empty_days": counts[A.CORRUPT_OR_EMPTY],
        "day_coverage": counts[A.VALID] / days_required if days_required else 0.0,
        "median_rows_per_day": float(np.median(rows_per_day)) if rows_per_day else 0.0,
        "days_insufficient_rows": insufficient_days,
        "days_excessive_gaps": gap_days,
        "rows": int(oi_all.size),
        "oi_null_or_zero_ratio": bad_ratio(oi_all),
        "oi_value_null_or_zero_ratio": bad_ratio(oiv_all),
        "median_interval_s": (float(np.median(np.diff(all_t))) / 1000
                              if all_t.size >= 2 else None),
    }


def evaluate_symbol(sym: str, listed_months: list[str], cfg: dict, cache: A.ArchiveCache,
                    funding_fetcher, req_start: int, end_ms: int) -> dict:
    sa = cfg["stage_a"]
    end_month = A.ym(end_ms - 1)
    months_set = set(listed_months)
    # Recent unlisted months may simply be unpublished: fall back to daily files.
    prior_month = A.months_between(end_ms - 40 * C.DAY_MS, end_ms)[-2]
    fallback = {m for m in A.months_between(req_start, end_ms)
                if m >= prior_month and m not in months_set}
    df_k, k_results = A.load_klines(cache, sym, req_start, end_ms, months_set,
                                    daily_fallback_months=fallback,
                                    label=f"{sym} klines")
    # Delisting: data ends before end_ms and every file after the last bar is
    # a clean FILE_MISSING. Any failed/corrupt file there blocks the inference.
    delisted, req_end, basis = False, end_ms, None
    if len(df_k):
        tail_start = int(df_k["open_time"].max()) + C.HOUR_MS
        if tail_start < end_ms:
            tail = [r for r in k_results if A.period_start_ms(r.key) >= tail_start]
            if all(r.status == A.FILE_MISSING for r in tail):
                delisted, req_end, basis = True, tail_start, "archive files end"
                k_results = [r for r in k_results if r not in tail]
        # Placeholder bars: zero volume to the end of the data for >= N days.
        vol = df_k["quote_volume"].fillna(0).to_numpy(float)
        traded = np.nonzero(vol > 0)[0]
        if traded.size:
            last_trade_end = int(df_k["open_time"].iloc[traded[-1]]) + C.HOUR_MS
            data_end = int(df_k["open_time"].iloc[-1]) + C.HOUR_MS
            run_ms = data_end - last_trade_end
            if (run_ms >= sa["delisting_zero_volume_days"] * C.DAY_MS
                    and last_trade_end < req_end):
                delisted, req_end = True, last_trade_end
                basis = f"zero-volume bars from {C.ms_to_iso(last_trade_end)}"
    out = {"symbol": sym, "delisted": delisted, "window_start": req_start,
           "window_end": req_end, "delisting_basis": basis,
           "last_archive_month": listed_months[-1],
           "end_month": end_month}

    # Klines
    k_counts = A.status_counts(k_results)
    open_times = df_k["open_time"].to_numpy(np.int64) if len(df_k) else np.array([], np.int64)
    out["klines"] = {
        "files": len(k_results), **{s.lower(): n for s, n in k_counts.items()},
        "missing_ratio": bars_missing_ratio(open_times, req_start, req_end),
    }

    # Metrics
    days = A.days_between(req_start, req_end)
    m_results = list(cache.get_many([A.metrics_key(sym, d) for d in days],
                                    f"{sym} metrics").values())
    out["metrics"] = metrics_stats(m_results, len(days), sa)

    # Funding
    try:
        f = funding_fetcher(sym, req_start, req_end)
        times = pd.to_numeric(f["funding_time"]).to_numpy(np.int64) if len(f) else np.array([], np.int64)
        out["funding"] = {"records": int(len(f)), "error": None,
                          "missing_ratio": funding_missing_ratio(times, req_start, req_end)}
    except (RegionBlockedError, requests.RequestException, RuntimeError, ValueError,
            KeyError) as exc:
        out["funding"] = {"records": 0, "error": f"{exc.__class__.__name__}: {exc}"[:200],
                          "missing_ratio": 1.0}

    reasons = []
    m = out["metrics"]
    if m["day_coverage"] < sa["min_metrics_day_coverage"]:
        reasons.append(METRICS_DAY_COVERAGE_BELOW_MIN)
    if m["median_rows_per_day"] < sa["min_median_rows_per_day"]:
        reasons.append(METRICS_MEDIAN_ROWS_BELOW_MIN)
    if m["oi_null_or_zero_ratio"] > sa["max_oi_null_or_zero_ratio"]:
        reasons.append(METRICS_OI_NULL_OR_ZERO_ABOVE_MAX)
    if m["median_interval_s"] is None or m["median_interval_s"] > sa["max_metrics_interval_seconds"]:
        reasons.append(METRICS_CADENCE_COARSER_THAN_5M)
    if out["klines"]["missing_ratio"] > sa["max_kline_missing_ratio"]:
        reasons.append(KLINES_MISSING_ABOVE_MAX)
    if out["funding"]["error"]:
        reasons.append(FUNDING_UNAVAILABLE)
    elif out["funding"]["missing_ratio"] > sa["max_funding_missing_ratio"]:
        reasons.append(FUNDING_MISSING_ABOVE_MAX)
    out["reasons"] = reasons
    out["included"] = not reasons
    out["_files"] = [vars(r) for r in k_results + m_results]
    return out


def verdict(selected: list[str], evaluations: list[dict], cfg: dict) -> dict:
    included = [e["symbol"] for e in evaluations if e["included"]]
    coarse = [e["symbol"] for e in evaluations
              if METRICS_CADENCE_COARSER_THAN_5M in e["reasons"]]
    reasons = []
    if len(included) < cfg["min_feasible_symbols"]:
        reasons.append(f"{len(included)} symbols passed data checks "
                       f"< min_feasible_symbols {cfg['min_feasible_symbols']}")
    if selected and len(coarse) * 2 > len(selected):
        reasons.append(f"metrics cadence coarser than 5 minutes for {len(coarse)} of "
                       f"{len(selected)} symbols")
    return {"verdict": NOT_FEASIBLE if reasons else FEASIBLE,
            "included": included, "coarse_cadence": coarse, "reasons": reasons}


# --------------------------------------------------------------------------
# Run management and report
# --------------------------------------------------------------------------

def create_run(cfg: dict, reports_dir: Path) -> Path:
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = C.config_hash(cfg)
    base = reports_dir / "stage_a" / f"run_{created}_{digest[:8]}"
    run_dir, n = base, 1
    while run_dir.exists():                 # never overwrite an earlier run
        n += 1
        run_dir = base.with_name(f"{base.name}_{n}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config_snapshot.json").write_text(json.dumps(
        {"config_hash": digest, "code_version": R.R2_CODE_VERSION,
         "created_utc": created, "config": cfg}, indent=2, default=str), encoding="utf-8")
    return run_dir


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.2f}%"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_report(res: dict) -> str:
    v = res["verdict"]
    u = res["universe"]
    sa = res["config"]["stage_a"]
    out = ["# R2 Stage A: Data Feasibility Report", "",
           f"- Run: `{res['run_id']}`",
           f"- Config hash: `{res['config_hash']}`",
           f"- Code version: {res['code_version']}", "",
           f"## Verdict: {v['verdict']}", ""]
    if v["reasons"]:
        out += [f"- {r}" for r in v["reasons"]] + [""]
    if res["download_failures"]:
        out += [f"**WARNING:** {res['download_failures']} files failed to download "
                "(network/HTTP). These are not evidence of missing data. Rerun Stage A "
                "(the cache resumes) before relying on this verdict.", ""]
    out += ["Stage B has not been built or run. It requires explicit authorization "
            "after this report is reviewed.", "",
            "## Setup", "",
            f"- Period: {C.ms_to_iso(res['period'][0])} to {C.ms_to_iso(res['period'][1])} "
            "(end exclusive)",
            f"- Required observation window start: {C.ms_to_iso(res['required_start'])}",
            f"- Listing boundary (min_listing_days): {C.ms_to_iso(u['listing_boundary_ms'])}",
            f"- Pre-start volume window: {C.ms_to_iso(u['prestart_window'][0])} to "
            f"{C.ms_to_iso(u['prestart_window'][1])}",
            f"- Symbols in archive listing: {u['symbols_in_archive']}", "",
            "## Universe construction", ""]
    counts: dict[str, int] = {}
    for row in u["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    out.append(_table(["Status", "Symbols"], sorted(counts.items())))
    out += ["", "Full candidate list: `universe_candidates.csv`.", "",
            "### Selected universe (by 30-day pre-start quote volume)", ""]
    rows_by_sym = {r["symbol"]: r for r in u["rows"]}
    ranked = sorted((r for r in u["rows"] if r["rank"]), key=lambda r: r["rank"])
    out.append(_table(["Rank", "Symbol", "Pre-start quote volume (USDT)", "First archive month",
                       "Last archive month", "Selected"],
                      [[r["rank"], r["symbol"], f"{r['prestart_quote_volume']:,.0f}",
                        r["first_month"], r["last_month"], "yes" if r["selected"] else ""]
                       for r in ranked[:len(u["selected"]) + 5]]))
    excluded = [r for r in u["rows"] if r["status"] in (
        U.INSUFFICIENT_LISTING_HISTORY, U.INSUFFICIENT_PRESTART_VOLUME_DATA,
        U.ARCHIVE_LISTING_FAILED)]
    if excluded:
        out += ["", "### Candidates excluded for data reasons", ""]
        out.append(_table(["Symbol", "Reason", "Detail"],
                          [[r["symbol"], r["status"], r["detail"]] for r in excluded]))

    out += ["", "## Per-symbol data checks", "",
            f"Thresholds: metrics day coverage >= {_pct(sa['min_metrics_day_coverage'])}, "
            f"median rows/day >= {sa['min_median_rows_per_day']}, OI null/zero <= "
            f"{_pct(sa['max_oi_null_or_zero_ratio'])}, cadence <= "
            f"{sa['max_metrics_interval_seconds']}s, klines missing <= "
            f"{_pct(sa['max_kline_missing_ratio'])}, funding missing <= "
            f"{_pct(sa['max_funding_missing_ratio'])}. Gap days (> "
            f"{sa['gap_report_minutes']} min) are reported only.", "",
            "### Metrics (5-minute OI)", ""]
    out.append(_table(
        ["Symbol", "Days req.", "Valid", "File missing", "Download failed",
         "Corrupt/empty", "Coverage", "Median rows/day", "Days < min rows",
         "Days with gaps", "OI null/zero", "OI value null/zero", "Median interval (s)"],
        [[e["symbol"], m["days_required"], m["valid_days"], m["file_missing_days"],
          m["download_failed_days"], m["corrupt_or_empty_days"], _pct(m["day_coverage"]),
          f"{m['median_rows_per_day']:.0f}", m["days_insufficient_rows"],
          m["days_excessive_gaps"], _pct(m["oi_null_or_zero_ratio"]),
          _pct(m["oi_value_null_or_zero_ratio"]),
          "n/a" if m["median_interval_s"] is None else f"{m['median_interval_s']:.0f}"]
         for e in res["evaluations"] for m in [e["metrics"]]]))
    out += ["", "### Klines and funding", ""]
    out.append(_table(
        ["Symbol", "Window", "Data ends early (delisting)", "Kline files", "Kline missing/failed/corrupt",
         "Kline bars missing", "Funding records", "Funding missing", "Funding error"],
        [[e["symbol"], f"{C.ms_to_iso(e['window_start'])[:10]} to "
          f"{C.ms_to_iso(e['window_end'])[:10]}",
          f"yes ({e.get('delisting_basis')})" if e["delisted"] else "",
          e["klines"]["files"],
          f"{e['klines']['file_missing']}/{e['klines']['download_failed']}/"
          f"{e['klines']['corrupt_or_empty']}",
          _pct(e["klines"]["missing_ratio"]), e["funding"]["records"],
          _pct(e["funding"]["missing_ratio"]), e["funding"]["error"] or "-"]
         for e in res["evaluations"]]))
    out += ["", "### Inclusion", ""]
    out.append(_table(["Symbol", "Included", "Exclusion reasons"],
                      [[e["symbol"], "yes" if e["included"] else "NO",
                        ", ".join(e["reasons"]) or "-"] for e in res["evaluations"]]))
    out += ["", f"Included: {len(v['included'])} of {len(u['selected'])} selected "
            f"(minimum {res['config']['min_feasible_symbols']}).", "",
            "## Notes", "",
            "- Missing archive days are recorded, never substituted.",
            f"- Delisting is inferred when archive files end, or when every bar to the "
            f"end of the data has zero volume for >= "
            f"{sa['delisting_zero_volume_days']} days; the window then ends at the "
            "last traded bar.",
            "- The symbol master comes from the archive listing (includes delisted "
            "contracts); today's exchangeInfo is not used.",
            "- Per-file outcomes: `file_manifest.csv`.", ""]
    return "\n".join(out)


def run_stage_a(cfg: dict, transport, funding_fetcher, data_dir: Path,
                reports_dir: Path) -> tuple[Path, dict]:
    run_dir = create_run(cfg, reports_dir)
    snapshot = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    cache = A.ArchiveCache(data_dir / "archive", transport,
                           cfg["stage_a"]["download_workers"])
    start_ms, end_ms = C.period_bounds(cfg)
    req_start = R.required_start_ms(cfg)

    universe = U.build_universe(cfg, cache, transport)
    with open(run_dir / "universe_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(universe["rows"][0].keys())
                                if universe["rows"] else ["symbol"])
        writer.writeheader()
        writer.writerows(universe["rows"])
    (run_dir / "universe.json").write_text(json.dumps(
        {k: v for k, v in universe.items() if k != "rows"}, indent=2), encoding="utf-8")

    evaluations = []
    for sym in universe["selected"]:
        print(f"Stage A checks: {sym}")
        evaluations.append(evaluate_symbol(sym, universe["listed_months"][sym], cfg, cache,
                                           funding_fetcher, req_start, end_ms))

    files = [f for e in evaluations for f in e.pop("_files")]
    with open(run_dir / "file_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["key", "status", "detail", "path", "cached"])
        writer.writeheader()
        writer.writerows(files)
    download_failures = sum(1 for f in files if f["status"] == A.DOWNLOAD_FAILED)

    results = {
        "run_id": run_dir.name, "config_hash": snapshot["config_hash"],
        "code_version": snapshot["code_version"], "created_utc": snapshot["created_utc"],
        "config": cfg, "period": [start_ms, end_ms], "required_start": req_start,
        "universe": universe, "evaluations": evaluations,
        "download_failures": download_failures,
        "verdict": verdict(universe["selected"], evaluations, cfg),
    }
    (run_dir / "stage_a_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    (run_dir / "stage_a_report.md").write_text(build_report(results), encoding="utf-8")

    index = reports_dir / "stage_a" / "runs_index.csv"
    new = not index.exists()
    with open(index, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "created_utc", "config_hash", "code_version",
                        "verdict", "included", "selected", "download_failures"])
        w.writerow([run_dir.name, snapshot["created_utc"], snapshot["config_hash"],
                    snapshot["code_version"], results["verdict"]["verdict"],
                    len(results["verdict"]["included"]), len(universe["selected"]),
                    download_failures])
    return run_dir, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R2 Stage A data feasibility")
    parser.add_argument("--config", default=None, help="path to r2_oi.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = R.load(args.config)
    except C.ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    data_dir = C.resolve_dir(cfg, "data_dir", "data/r2_oi")
    reports_dir = C.resolve_dir(cfg, "reports_dir", "reports/r2_oi")
    try:
        run_dir, results = run_stage_a(cfg, A.HttpTransport(),
                                       make_funding_fetcher(data_dir / "funding"),
                                       data_dir, reports_dir)
    except A.ArchiveBlockedError as exc:
        print(f"ERROR: {exc}")
        return 2
    v = results["verdict"]
    print()
    print(f"Report: {run_dir / 'stage_a_report.md'}")
    print(f"Included {len(v['included'])} of {len(results['universe']['selected'])} "
          f"selected symbols")
    for reason in v["reasons"]:
        print(f"  - {reason}")
    if results["download_failures"]:
        print(f"WARNING: {results['download_failures']} download failures; "
              "rerun to retry before relying on this verdict.")
    print(f"STAGE A: {v['verdict']}")
    print("Stage B is not built. Review the report before authorizing it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
