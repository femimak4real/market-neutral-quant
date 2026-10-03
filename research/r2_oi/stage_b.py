"""R2 Stage B: economic gate (spec R2.5-R2.11). PASS / FAIL / INCONCLUSIVE.

Reads only data cached by a FEASIBLE Stage A run with the same config hash.
Never downloads. A PASS authorizes a written proposal only (R2.0).

Usage (from repo root):
    python -m research.r2_oi.stage_b
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

from research.phase05_funding_gate import common as C
from research.phase05_funding_gate import economics as E
from research.r2_oi import archive as A
from research.r2_oi import config as R
from research.r2_oi import features as F
from research.r2_oi import simulate as SIM

STAGE_B_CODE_VERSION = "r2-stage-b-0.1"
H, DAY = C.HOUR_MS, C.DAY_MS

CONVENTIONS = [
    "Observations older than 1 hour at their query time are treated as missing.",
    "z-scores use the prior W daily values with sample std (ddof=1).",
    "Test and control share eligibility: both z_ret and z_oi must be valid.",
    "Ranking ties are broken alphabetically by symbol.",
    "Fewer than 2 x n_per_side eligible symbols: the target portfolio is flat.",
    "Held same-side positions are not resized (R2.6 step 5); entry notionals "
    "are equal, realized long/short drift is reported.",
    "Funding counts for settlements strictly after entry and strictly before exit.",
    "Delisted symbols close at the close of their last traded bar.",
    "Segments start flat and close all positions at the close of the bar ending "
    "at the segment end. The IS/OOS split is aligned to a UTC day.",
    "Quadrant attribution uses the OI/price change at the entry rebalance.",
]
GATE_ACTION = {
    "PASS": "PASS permits a written proposal for the next phase only. It does not "
            "permit building execution.",
    "FAIL": "STOP. Do not implement production infrastructure.",
    "INCONCLUSIVE": "STOP. Resolve the inconclusive reasons listed in this report.",
}


class StageBRefused(RuntimeError):
    pass


class OfflineTransport:
    """Stage B never downloads: any cache miss is reported, not fetched."""

    def fetch(self, url, params=None):
        return None, b"", "offline: Stage B reads Stage A cache only"


# --------------------------------------------------------------------------
# Stage A linkage
# --------------------------------------------------------------------------

def find_stage_a(reports_dir: Path, cfg_hash: str) -> dict:
    runs = sorted((reports_dir / "stage_a").glob("run_*"))
    for run in reversed(runs):
        path = run / "stage_a_results.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        if res.get("config_hash") == cfg_hash and res["verdict"]["verdict"] == "FEASIBLE":
            return res
    raise StageBRefused(
        "No FEASIBLE Stage A run found for the current config hash. Stage B "
        "refuses to run (R2.0). Run Stage A first, without changing the config.")


# --------------------------------------------------------------------------
# Data loading (cache only)
# --------------------------------------------------------------------------

def load_market(cfg: dict, stage_a: dict, data_dir: Path) -> tuple[list[SIM.SymbolMarket], dict]:
    cache = A.ArchiveCache(data_dir / "archive", OfflineTransport(),
                           cfg["stage_a"]["download_workers"], progress=False)
    _, end_ms = C.period_bounds(cfg)
    listed_all = stage_a["universe"]["listed_months"]
    prior_month = A.months_between(end_ms - 40 * DAY, end_ms)[-2]
    market, problems = [], {}
    for ev in stage_a["evaluations"]:
        if not ev["included"]:
            continue
        sym, ws, we = ev["symbol"], int(ev["window_start"]), int(ev["window_end"])
        listed = set(listed_all[sym])
        fallback = {m for m in A.months_between(ws, end_ms)
                    if m >= prior_month and m not in listed}
        kl, k_res = A.load_klines(cache, sym, ws, end_ms, listed, fallback)
        kl = kl[kl["open_time"] < we].reset_index(drop=True)
        days = A.days_between(ws, we)
        m_res = cache.get_many([A.metrics_key(sym, d) for d in days])
        frames = [A.parse_metrics_csv(A.read_zip_text(r.path))
                  for r in m_res.values() if r.status == A.VALID]
        bad = [r for r in list(k_res) + list(m_res.values())
               if r.status != A.VALID and r.detail != "not in archive listing"]
        if bad:
            problems[sym] = len(bad)
        mt = (pd.concat(frames).drop_duplicates("create_time").sort_values("create_time")
              if frames else pd.DataFrame(columns=["create_time", "sum_open_interest_value"]))
        fpath = data_dir / "funding" / f"{sym}_funding.csv"
        f = pd.read_csv(fpath)
        f = f[(f["funding_time"] >= ws) & (f["funding_time"] < we)]
        f, _ = E.resolve_funding_prices(f, kl[["open_time", "open"]])
        market.append(SIM.SymbolMarket(
            symbol=sym,
            bar_open=kl["open_time"].to_numpy(np.int64),
            open=kl["open"].to_numpy(float), high=kl["high"].to_numpy(float),
            close=kl["close"].to_numpy(float), window_end=we,
            f_time=f["funding_time"].to_numpy(np.int64),
            f_cum=SIM.make_funding_cum(f["funding_time"].to_numpy(np.int64),
                                       f["funding_rate"].to_numpy(float),
                                       f["funding_price"].to_numpy(float)),
            slippage_bps=C.slippage_bps_for(cfg, sym),
            m_time=mt["create_time"].to_numpy(np.int64),
            m_oiv=pd.to_numeric(mt["sum_open_interest_value"]).to_numpy(float)))
    if problems:
        raise StageBRefused(f"Cache is missing files that Stage A validated: {problems}. "
                            "Rerun Stage A (same config) to restore the cache.")
    return market, {}


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------

def evaluate_gate(m: dict, cfg: dict) -> dict:
    inconclusive = []
    if not m.get("stage_a_feasible", False):
        inconclusive.append("Stage A was not FEASIBLE")
    if m["oos_full_rebalances"] < cfg["min_oos_rebalances"]:
        inconclusive.append(f"{m['oos_full_rebalances']} OOS rebalances with full positions "
                            f"< min_oos_rebalances {cfg['min_oos_rebalances']}")
    if m["oos_symbols_eligible"] < cfg["min_feasible_symbols"]:
        inconclusive.append(f"{m['oos_symbols_eligible']} symbols eligible OOS "
                            f"< min_feasible_symbols {cfg['min_feasible_symbols']}")
    lo, hi = m["neighbor_net"].get("lower"), m["neighbor_net"].get("upper")
    if lo is None or hi is None:
        c4, c4v = False, "locked L at grid edge; a neighbour is unavailable"
    else:
        c4, c4v = lo > 0 and hi > 0, f"lower={lo:.2f}, upper={hi:.2f}"
    ann, beta, q = m["oos_annualized_return"], m["oos_beta"], m["perm_quantile_value"]
    criteria = [
        (1, "OI-signal OOS net PnL > 0", f"{m['oos_net']:.2f} USDT", m["oos_net"] > 0),
        (2, "OI-signal OOS net PnL > control OOS net PnL",
         f"{m['oos_net']:.2f} vs {m['oos_control_net']:.2f}",
         m["oos_net"] > m["oos_control_net"]),
        (3, f"OI-signal OOS net PnL > 0 at costs x{cfg['cost_stress_multiplier']}",
         f"{m['oos_net_stressed']:.2f} USDT", m["oos_net_stressed"] > 0),
        (4, "Both grid neighbours of locked L: OOS net PnL > 0", c4v, c4),
        (5, f"OOS annualized return >= {cfg['hurdle_annual_return']:.2%}",
         "n/a" if ann is None else f"{ann:.2%}",
         ann is not None and ann >= cfg["hurdle_annual_return"]),
        (6, f"OOS max drawdown <= {cfg['max_drawdown_pct']:.2%} of capital",
         f"{m['oos_max_dd_pct']:.2%}", m["oos_max_dd_pct"] <= cfg["max_drawdown_pct"]),
        (7, f"abs(OOS BTC beta) <= {cfg['max_abs_beta']}",
         "n/a" if beta is None else f"{beta:.3f}",
         beta is not None and abs(beta) <= cfg["max_abs_beta"]),
        (8, f"OOS net PnL > {cfg['permutation_quantile']:.0%} quantile of "
            f"{cfg['n_permutations']} score permutations",
         "n/a" if q is None else f"{m['oos_net']:.2f} vs {q:.2f}",
         q is not None and m["oos_net"] > q),
    ]
    rows = [{"id": i, "criterion": c, "value": v, "passed": bool(ok)}
            for i, c, v, ok in criteria]
    verdict = ("INCONCLUSIVE" if inconclusive
               else "PASS" if all(r["passed"] for r in rows) else "FAIL")
    return {"verdict": verdict, "criteria": rows, "inconclusive_reasons": inconclusive,
            "action": GATE_ACTION[verdict]}


# --------------------------------------------------------------------------
# Segment statistics
# --------------------------------------------------------------------------

def summarize(res: SIM.SimResult, market, seg: tuple[int, int], capital: float,
              p: SIM.SimParams, cfg: dict, btc_close: np.ndarray | None) -> dict:
    a, b = seg
    lots = res.lots
    bounds = SIM.day_boundaries(a, b)
    eq = SIM.equity_curve(lots, market, bounds)
    daily = np.diff(eq)
    tot = lambda k: float(sum(l[k] for l in lots))
    net = tot("net")
    strat_ret = daily / capital
    beta = se = None
    if btc_close is not None and len(daily) >= 3:
        btc_ret = btc_close[1:] / btc_close[:-1] - 1
        beta, se = SIM.beta_stats(strat_ret, btc_ret)
    months: dict[str, float] = {}
    for t, v in zip(bounds[1:], daily):
        key = A.ym(int(t) - 1)
        months[key] = months.get(key, 0.0) + float(v)
    quad = {name: 0.0 for name in F.QUADRANTS}
    quad_n = {name: 0 for name in F.QUADRANTS}
    for l in lots:
        if l["quadrant"] >= 0:
            quad[F.QUADRANTS[l["quadrant"]]] += l["net"]
            quad_n[F.QUADRANTS[l["quadrant"]]] += 1
    shorts = sorted((l for l in lots if l["side"] < 0 and "consumption" in l),
                    key=lambda l: -l["consumption"])
    entry_turn = tot("entry_notional")
    exit_turn = tot("exit_notional")
    n_reb = len(res.eligible)
    sd = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    return {
        "net": net, "price_pnl": tot("price_pnl"), "funding": tot("funding"),
        "fees": tot("entry_fee") + tot("exit_fee"),
        "slippage": tot("entry_slip") + tot("exit_slip"),
        "equity_end_check": float(eq[-1]) if len(eq) else 0.0,
        "lots": len(lots), "rebalances": n_reb, "full_rebalances": int(res.full.sum()),
        "entry_turnover": entry_turn, "exit_turnover": exit_turn,
        "gross_turnover": entry_turn + exit_turn,
        "avg_turnover_per_rebalance": ((entry_turn + exit_turn) / n_reb / p.gross_notional
                                       if n_reb else None),
        "annualized_return": E.annualized_return(net, capital, b - a),
        "max_dd_usdt": E.max_drawdown(eq), "max_dd_pct": E.max_drawdown(eq) / capital,
        "sharpe_daily_annualized": (float(np.mean(daily) / sd * np.sqrt(365)) if sd > 0
                                    else None),
        "beta": beta, "beta_se": se,
        "monthly": months,
        "long_pnl": float(sum(l["net"] for l in lots if l["side"] > 0)),
        "short_pnl": float(sum(l["net"] for l in lots if l["side"] < 0)),
        "quadrant_pnl": quad, "quadrant_lots": quad_n,
        "breadth": SIM.breadth_stats(res.eligible, p.n_per_side),
        "net_exposure_abs_mean_pct": (float(np.mean(np.abs(res.net_exposure)))
                                      / (p.gross_notional / 2) if n_reb else None),
        "net_exposure_abs_max_pct": (float(np.max(np.abs(res.net_exposure)))
                                     / (p.gross_notional / 2) if n_reb else None),
        "squeeze_worst10": [
            {k: s[k] for k in ("symbol", "entry_time", "exit_time", "entry_price",
                               "max_high", "consumption", "adverse_move")}
            for s in shorts[:10]],
        "squeeze_flagged": sum(1 for s in shorts
                               if E.is_consumption_flagged(
                                   s["consumption"], cfg["diagnostic_margin_consumption_limit"])),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def create_run(cfg: dict, stage_a: dict, reports_dir: Path) -> tuple[Path, dict]:
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    universe = [{"symbol": e["symbol"], "window_start": e["window_start"],
                 "window_end": e["window_end"], "delisted": e["delisted"]}
                for e in stage_a["evaluations"] if e["included"]]
    snapshot = {"config_hash": C.config_hash(cfg), "code_version": STAGE_B_CODE_VERSION,
                "stage_a_run_id": stage_a["run_id"], "created_utc": created,
                "universe": universe, "conventions": CONVENTIONS, "config": cfg}
    snapshot["preregistration_hash"] = C.config_hash(snapshot)
    base = reports_dir / "stage_b" / f"run_{created}_{snapshot['preregistration_hash'][:8]}"
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = base.with_name(f"{base.name}_{n}")
    run_dir.mkdir(parents=True)
    (run_dir / "preregistration.json").write_text(json.dumps(snapshot, indent=2, default=str),
                                                  encoding="utf-8")
    return run_dir, snapshot


def build_matrices(market, times, grid, window):
    out = {}
    for L in grid:
        S = np.full((len(times), len(market)), np.nan)
        Cm, Q = S.copy(), np.full(S.shape, 4, dtype=int)
        for j, m in enumerate(market):
            d_oi, ret = F.raw_features(m.m_time, m.m_oiv, m.bar_open, m.close, times, L)
            sig = F.signals(d_oi, ret, window)
            S[:, j], Cm[:, j] = sig["S"], sig["C"]
            Q[:, j] = F.quadrant_codes(d_oi, ret)
        out[L] = (S, Cm, Q)
    return out


def run_stage_b(cfg: dict, data_dir: Path, reports_dir: Path, verbose: bool = True) -> tuple[Path, dict]:
    stage_a = find_stage_a(reports_dir, C.config_hash(cfg))
    run_dir, snap = create_run(cfg, stage_a, reports_dir)
    log = print if verbose else (lambda *a, **k: None)
    log(f"Stage B pre-registered: {run_dir.name} (Stage A {stage_a['run_id']})")

    market, _ = load_market(cfg, stage_a, data_dir)
    log(f"Loaded {len(market)} symbols from the Stage A cache")
    start, end = C.period_bounds(cfg)
    span_days = (end - start) // DAY
    split = start + int(round(cfg["is_fraction"] * span_days)) * DAY
    W = cfg["zscore_window_days"]
    hour = cfg["rebalance_hour_utc"] * H
    times = np.arange(start - W * DAY, end, DAY, dtype=np.int64) + hour
    times = times[times < end]
    is_rows = (times >= start) & (times < split)
    oos_rows = (times >= split) & (times < end)
    grid = list(cfg["lookback_grid_hours"])
    mats = build_matrices(market, times, grid, W)
    fills = SIM.fill_matrix(market, times)
    p = SIM.SimParams(cfg["n_per_side"], float(cfg["gross_notional_usdt"]),
                      float(cfg["fees"]["perp_taker"]), float(cfg["max_leverage"]))
    capital = p.gross_notional / p.leverage

    def sim(L, which, rows, seg_end, collect=True, rng=None):
        S, Cm, Q = mats[L]
        sc = S if which == "S" else Cm
        return SIM.simulate(market, times[rows], sc[rows], fills[rows], seg_end, p,
                            quadrants=Q[rows], collect=collect, rng=rng)

    log("In-sample L grid...")
    grid_rows = []
    for L in grid:
        r = sim(L, "S", is_rows, split, collect=False)
        grid_rows.append({"L": L, "net": r.net_pnl, "lots": len(r.lots)})
        log(f"  L={L}h  IS net {r.net_pnl:,.2f}")
    sel = max(range(len(grid)), key=lambda i: (grid_rows[i]["net"], -i))
    L = grid[sel]
    log(f"Locked L = {L}h. Out-of-sample run (once)...")

    btc = next((m for m in market if m.symbol == "BTCUSDT"), None)

    def btc_closes(a, b):
        if btc is None:
            return None
        bounds = SIM.day_boundaries(a, b)
        return F.value_at(btc.bar_open + H, btc.close, bounds)

    segs = {"in_sample": (start, split, is_rows), "out_of_sample": (split, end, oos_rows)}
    results_seg, lot_tables = {}, {}
    for seg, (a, b, rows) in segs.items():
        for which in ("S", "C"):
            r = sim(L, which, rows, b)
            label = "oi_signal" if which == "S" else "control"
            results_seg[f"{seg}/{label}"] = summarize(r, market, (a, b), capital, p, cfg,
                                                      btc_closes(a, b))
            lot_tables[f"{seg}_{label}"] = r.lots
            if seg == "out_of_sample" and which == "S":
                oos_res = r
            if seg == "out_of_sample" and which == "C":
                oos_ctrl = r

    neighbors = {}
    for label, i in (("lower", sel - 1), ("upper", sel + 1)):
        neighbors[label] = ({"L": grid[i], "net": sim(grid[i], "S", oos_rows, end,
                                                      collect=False).net_pnl}
                            if 0 <= i < len(grid) else None)

    mult = float(cfg["cost_stress_multiplier"])
    stressed = float(sum(l["price_pnl"] + l["funding"]
                         - mult * (l["entry_fee"] + l["exit_fee"] + l["entry_slip"]
                                   + l["exit_slip"]) for l in oos_res.lots))

    log(f"Permutation test ({cfg['n_permutations']} runs)...")
    rng = np.random.default_rng(cfg["permutation_seed"])
    perms = []
    for i in range(cfg["n_permutations"]):
        perms.append(sim(L, "S", oos_rows, end, collect=False, rng=rng).net_pnl)
        if verbose and (i + 1) % 100 == 0:
            log(f"  {i + 1}/{cfg['n_permutations']}")
    perms = np.array(perms)
    q = float(np.quantile(perms, cfg["permutation_quantile"])) if perms.size else None
    oos_s = results_seg["out_of_sample/oi_signal"]
    elig_oos = (~np.isnan(mats[L][0][oos_rows])) & (fills[oos_rows] >= 0)

    metrics = {
        "stage_a_feasible": True,
        "oos_full_rebalances": oos_s["full_rebalances"],
        "oos_symbols_eligible": int(elig_oos.any(axis=0).sum()),
        "oos_net": oos_s["net"],
        "oos_control_net": results_seg["out_of_sample/control"]["net"],
        "oos_net_stressed": stressed,
        "neighbor_net": {k: (v["net"] if v else None) for k, v in neighbors.items()},
        "oos_annualized_return": oos_s["annualized_return"],
        "oos_max_dd_pct": oos_s["max_dd_pct"],
        "oos_beta": oos_s["beta"],
        "perm_quantile_value": q,
    }
    gate = evaluate_gate(metrics, cfg)
    results = {
        "run_id": run_dir.name, "stage_a_run_id": stage_a["run_id"],
        "config_hash": snap["config_hash"],
        "preregistration_hash": snap["preregistration_hash"],
        "code_version": STAGE_B_CODE_VERSION, "config": cfg,
        "period": {"start": start, "split": split, "end": end},
        "universe": snap["universe"], "grid": grid_rows, "selected_index": sel,
        "locked_L": L, "segments": results_seg, "neighbors": neighbors,
        "stress": {"multiplier": mult, "oos_net": oos_s["net"], "oos_net_stressed": stressed},
        "permutation": {
            "n": int(perms.size), "seed": cfg["permutation_seed"],
            "quantile": cfg["permutation_quantile"], "quantile_value": q,
            "actual_percentile": (float((perms < oos_s["net"]).mean()) if perms.size else None),
            "distribution": {f"p{int(x * 100)}": float(np.quantile(perms, x))
                             for x in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)} if perms.size else {},
        },
        "metrics": metrics, "gate": gate, "conventions": CONVENTIONS,
    }
    for name, lots in lot_tables.items():
        if lots:
            pd.DataFrame(lots).to_csv(run_dir / f"positions_{name}.csv", index=False)
    (run_dir / "stage_b_results.json").write_text(json.dumps(results, indent=2, default=str),
                                                  encoding="utf-8")
    (run_dir / "stage_b_report.md").write_text(build_report(results), encoding="utf-8")
    index = reports_dir / "stage_b" / "runs_index.csv"
    new = not index.exists()
    with open(index, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "stage_a_run_id", "config_hash", "code_version",
                        "locked_L", "verdict"])
        w.writerow([run_dir.name, stage_a["run_id"], snap["config_hash"],
                    STAGE_B_CODE_VERSION, L, gate["verdict"]])
    return run_dir, results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _n(v, d=2):
    return "n/a" if v is None else f"{v:,.{d}f}"


def _pct(v, d=2):
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def _table(h, rows):
    out = ["| " + " | ".join(h) + " |", "|" + "|".join(["---"] * len(h)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def build_report(r: dict) -> str:
    g, cfg, segs = r["gate"], r["config"], r["segments"]
    out = ["# R2 Stage B: Economic Gate Report", "",
           f"- Run: `{r['run_id']}`", f"- Stage A run: `{r['stage_a_run_id']}`",
           f"- Config hash: `{r['config_hash']}`",
           f"- Pre-registration hash: `{r['preregistration_hash']}`",
           f"- Code version: {r['code_version']}", "",
           f"## Verdict: {g['verdict']}", "", f"**Go/no-go action:** {g['action']}", ""]
    if g["inconclusive_reasons"]:
        out += ["Inconclusive because:", ""] + [f"- {x}" for x in g["inconclusive_reasons"]] + [""]
    out += ["## Gate criteria (OOS, locked L)", "",
            _table(["#", "Criterion", "Value", "Result"],
                   [[c["id"], c["criterion"], c["value"], "PASS" if c["passed"] else "FAIL"]
                    for c in g["criteria"]]), "",
            "## Setup", "",
            f"- Period {C.ms_to_iso(r['period']['start'])} to {C.ms_to_iso(r['period']['end'])}; "
            f"IS/OOS split {C.ms_to_iso(r['period']['split'])}",
            f"- Universe: {len(r['universe'])} symbols; daily rebalance at "
            f"{cfg['rebalance_hour_utc']:02d}:00 UTC; {cfg['n_per_side']} long / "
            f"{cfg['n_per_side']} short; gross {cfg['gross_notional_usdt']:,} USDT",
            f"- Capital allocation {cfg['gross_notional_usdt'] / cfg['max_leverage']:,.0f} USDT "
            "(research assumption, not exchange margin)",
            f"- Locked L = {r['locked_L']}h (selected on in-sample OI-signal net PnL only)", "",
            "### In-sample L grid", "",
            _table(["L (h)", "IS net PnL", "Positions", "Selected"],
                   [[x["L"], _n(x["net"]), x["lots"],
                     "<-- locked" if i == r["selected_index"] else ""]
                    for i, x in enumerate(r["grid"])]), "",
            "### Neighbours (OOS, OI signal)", ""]
    for k in ("lower", "upper"):
        nb = r["neighbors"][k]
        out.append(f"- {k}: " + ("not available (grid edge)" if nb is None
                                 else f"L={nb['L']}h -> {_n(nb['net'])} USDT"))
    st = r["stress"]
    out += ["", f"Cost stress (OOS): {_n(st['oos_net'])} -> {_n(st['oos_net_stressed'])} USDT "
            f"at x{st['multiplier']}", "", "## Results", ""]
    order = ["in_sample/oi_signal", "in_sample/control",
             "out_of_sample/oi_signal", "out_of_sample/control"]
    out.append(_table(
        ["Run", "Net", "Price", "Funding", "Fees", "Slippage", "Ann. return", "Max DD",
         "Sharpe (desc.)", "BTC beta (SE)", "Long PnL", "Short PnL"],
        [[k, _n(s["net"]), _n(s["price_pnl"]), _n(s["funding"]), _n(s["fees"]),
          _n(s["slippage"]), _pct(s["annualized_return"]), _pct(s["max_dd_pct"]),
          _n(s["sharpe_daily_annualized"]),
          "n/a" if s["beta"] is None else f"{s['beta']:.3f} ({s['beta_se']:.3f})",
          _n(s["long_pnl"]), _n(s["short_pnl"])] for k in order for s in [segs[k]]]))
    out += ["", "### Turnover and breadth", ""]
    out.append(_table(
        ["Run", "Rebalances", "Full", "Positions", "Entry turnover", "Exit turnover",
         "Gross turnover", "Avg turnover/rebalance", "Breadth median/min/p5",
         "Thin (8-9) rebalances", "Net exposure drift mean/max"],
        [[k, s["rebalances"], s["full_rebalances"], s["lots"], _n(s["entry_turnover"], 0),
          _n(s["exit_turnover"], 0), _n(s["gross_turnover"], 0),
          _pct(s["avg_turnover_per_rebalance"], 1),
          f"{_n(s['breadth']['median'], 0)}/{s['breadth']['min']}/{_n(s['breadth']['p5'], 0)}",
          s["breadth"]["thin_rebalances"],
          f"{_pct(s['net_exposure_abs_mean_pct'], 1)}/{_pct(s['net_exposure_abs_max_pct'], 1)}"]
         for k in order for s in [segs[k]]]))
    out += ["", "### Quadrant attribution (positioning-flow proxies, by entry rebalance)", ""]
    out.append(_table(["Run"] + list(F.QUADRANTS),
                      [[k] + [f"{_n(segs[k]['quadrant_pnl'][q])} ({segs[k]['quadrant_lots'][q]})"
                              for q in F.QUADRANTS] for k in order]))
    out += ["", "The flush proxy (OI down, price down) does not identify liquidations.", "",
            "### Monthly net PnL (OOS)", ""]
    months = sorted(set(segs["out_of_sample/oi_signal"]["monthly"])
                    | set(segs["out_of_sample/control"]["monthly"]))
    out.append(_table(["Month", "OI signal", "Control"],
                      [[mo, _n(segs["out_of_sample/oi_signal"]["monthly"].get(mo)),
                        _n(segs["out_of_sample/control"]["monthly"].get(mo))] for mo in months]))
    pm = r["permutation"]
    out += ["", "## Permutation test (OOS, OI scores shuffled across eligible symbols)", "",
            f"- Runs: {pm['n']}, seed {pm['seed']}",
            f"- Quantiles: " + ", ".join(f"{k} {_n(v)}" for k, v in pm["distribution"].items()),
            f"- Actual OOS net PnL percentile within permutations: {_pct(pm['actual_percentile'], 1)}",
            "", "## Short-squeeze diagnostic (not a liquidation model)", ""]
    for k in ("in_sample/oi_signal", "out_of_sample/oi_signal"):
        s = segs[k]
        out.append(f"**{k}**: {s['squeeze_flagged']} short positions with consumption >= "
                   f"{cfg['diagnostic_margin_consumption_limit']}")
        out.append("")
        out.append(_table(["Symbol", "Entry", "Exit", "Consumption", "Adverse move"],
                          [[x["symbol"], C.ms_to_iso(x["entry_time"]),
                            C.ms_to_iso(x["exit_time"]), _n(x["consumption"], 3),
                            _pct(x["adverse_move"], 1)] for x in s["squeeze_worst10"]]))
        out.append("")
    out += ["## Universe", "",
            _table(["Symbol", "Window end", "Delisted"],
                   [[u["symbol"], C.ms_to_iso(u["window_end"]), "yes" if u["delisted"] else ""]
                    for u in r["universe"]]), "",
            "## Implementation conventions (fixed before results)", ""]
    out += [f"- {c}" for c in r["conventions"]]
    out += ["", "## Known limitations", "",
            "- No liquidation data is used; quadrants are positioning-flow proxies.",
            "- The universe is fixed at start_date; later listings are excluded.",
            "- 1h bar opens are not executable prices; slippage_bps approximates costs.",
            "- Metrics timestamps use a 5-minute guard; publication timing is not verified.",
            "- Funding regime after December 2025 was compressed (Decision 001).",
            "- The period overlaps R1, which studied funding rates, not these signals.",
            "- Daily rebalancing at a fixed hour; no intraday risk management.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R2 Stage B economic gate")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = R.load(args.config)
        run_dir, res = run_stage_b(cfg, C.resolve_dir(cfg, "data_dir", "data/r2_oi"),
                                   C.resolve_dir(cfg, "reports_dir", "reports/r2_oi"))
    except (C.ConfigError, StageBRefused) as exc:
        print(f"ERROR: {exc}")
        return 2
    g = res["gate"]
    print()
    print(f"Report: {run_dir / 'stage_b_report.md'}")
    print(f"Locked L: {res['locked_L']}h")
    for c in g["criteria"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['id']}. {c['criterion']}: {c['value']}")
    for x in g["inconclusive_reasons"]:
        print(f"  [INCONCLUSIVE] {x}")
    print(f"STAGE B: {g['verdict']}")
    print(g["action"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
