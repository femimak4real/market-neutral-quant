"""Stage A end-to-end on a simulated archive covering every exclusion path."""
import csv
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from research.phase05_funding_gate import common as C
from research.r2_oi import archive as A
from research.r2_oi import config as R
from research.r2_oi import stage_a as S
from research.r2_oi import universe as U
from research.r2_oi.tests.fake_archive import FakeArchive, H, ms

START, END = ms(2025, 2, 26), ms(2025, 3, 6)     # end_date 2025-03-05 inclusive
REQ_START = ms(2025, 2, 22)                       # 2 + ceil(12/24)=1 + 1 = 4 days


def cfg(**over):
    c = {
        "start_date": "2025-02-26", "end_date": "2025-03-05", "universe_size": 20,
        "min_listing_days": 40, "min_feasible_symbols": 4,
        "exclude_bases": ["USDC", "FDUSD", "TUSD", "USDP", "DAI"],
        "zscore_window_days": 2, "lookback_grid_hours": [4, 8, 12],
        "rebalance_hour_utc": 0, "n_per_side": 2, "gross_notional_usdt": 8000,
        "max_leverage": 2.0, "diagnostic_margin_consumption_limit": 0.7,
        "fees": {"perp_taker": 0.0005}, "slippage_bps": {"default": 5},
        "is_fraction": 0.6, "cost_stress_multiplier": 1.5, "min_oos_rebalances": 1,
        "hurdle_annual_return": 0.05, "max_drawdown_pct": 0.1, "max_abs_beta": 0.2,
        "n_permutations": 10, "permutation_quantile": 0.95, "permutation_seed": 1,
        "stage_a": {
            "exclude_symbols": ["BTCDOMUSDT"], "prestart_min_bar_coverage": 1.0,
            "min_metrics_day_coverage": 0.98, "min_median_rows_per_day": 280,
            "max_oi_null_or_zero_ratio": 0.01, "max_metrics_interval_seconds": 330,
            "max_kline_missing_ratio": 0.02, "max_funding_missing_ratio": 0.02,
            "gap_report_minutes": 15, "download_workers": 2,
            "delisting_zero_volume_days": 3,
        },
    }
    c.update(over)
    return c


def full_symbol(fake, sym, qv=1000.0, first=ms(2024, 11, 1), last=END + 2 * 86_400_000,
                metrics_last=date(2025, 3, 5), **mkw):
    fake.add_monthly_klines(sym, first, last, qv=qv)
    fake.add_metrics(sym, date(2025, 2, 22), metrics_last, **mkw)


@pytest.fixture
def world():
    f = FakeArchive(page_size=7)
    full_symbol(f, "AAAUSDT", qv=9000)
    full_symbol(f, "BBBUSDT", qv=8000)
    full_symbol(f, "CCCUSDT", qv=500, first=ms(2025, 1, 10))   # boundary month, early enough
    full_symbol(f, "DDDUSDT", qv=500, first=ms(2025, 1, 25))   # boundary month, too late
    full_symbol(f, "EEEUSDT", qv=500, first=ms(2025, 2, 5))    # after boundary month
    f.add_monthly_klines("FFFUSDT", ms(2024, 11, 1), END, qv=500, skip_months={"2025-01"})
    f.add_monthly_klines("GGGUSDT", ms(2024, 11, 1), ms(2025, 1, 20), qv=500)  # gone pre-start
    full_symbol(f, "QQQUSDT", qv=0.0)          # placeholder bars only: not active
    full_symbol(f, "USDCUSDT"); full_symbol(f, "BTCDOMUSDT")
    full_symbol(f, "XYZUSDT_250328"); full_symbol(f, "ETHUSDC")
    # Delisted mid-final-month: data ends 2025-03-03 12:00.
    full_symbol(f, "HHHUSDT", qv=7000, last=ms(2025, 3, 3, 12), metrics_last=date(2025, 3, 3))
    # Mar monthly unpublished; daily files cover the final days.
    f.add_monthly_klines("IIIUSDT", ms(2024, 11, 1), END + 2 * 86_400_000, qv=6000,
                         skip_months={"2025-03"})
    for d in range(1, 8):
        f.add_daily_klines("IIIUSDT", date(2025, 3, d), qv=6000)
    f.add_metrics("IIIUSDT", date(2025, 2, 22), date(2025, 3, 5))
    # Defects
    full_symbol(f, "JJJUSDT", qv=5000)
    del f.files[A.metrics_key("JJJUSDT", date(2025, 3, 2))]          # file missing
    full_symbol(f, "KKKUSDT", qv=4900)
    k = A.metrics_key("KKKUSDT", date(2025, 3, 2))
    f.put(k, "", raw=b"not a zip", checksum=False)                       # corrupt
    full_symbol(f, "LLLUSDT", qv=4800, rows=96, step_min=15)             # coarse cadence
    full_symbol(f, "MMMUSDT", qv=4700, null_every=10)                    # 10% null OI
    full_symbol(f, "NNNUSDT", qv=4600)                                   # funding error
    # Trading stops 2025-03-02; archive keeps writing zero-volume bars (REEF/FTM case).
    full_symbol(f, "PPPUSDT", qv=lambda t: 4400.0 if t < ms(2025, 3, 2) else 0.0)
    full_symbol(f, "OOOUSDT", qv=4500)
    f.override[A.metrics_key("OOOUSDT", date(2025, 3, 2))] = 500         # download fails
    return f


def funding_ok(sym, a, b):
    if sym == "NNNUSDT":
        raise RuntimeError("Invalid symbol")
    times = np.arange(a, b, 8 * H)
    return pd.DataFrame({"funding_time": times, "funding_rate": 0.0001,
                         "mark_price": 100.0})


def run(world, tmp_path, **over):
    c = cfg(**over)
    R.validate(c)
    return S.run_stage_a(c, world, funding_ok, tmp_path / "data", tmp_path / "reports")


def test_required_start_formula():
    assert R.required_start_ms(cfg()) == REQ_START
    real = R.load()
    assert R.required_start_ms(real) == C.date_to_ms("2024-10-01") - 34 * C.DAY_MS


def test_refuses_null_fee():
    with pytest.raises(C.ConfigError, match="perp_taker"):
        R.validate(cfg(fees={"perp_taker": None}))


def test_universe_statuses_and_no_current_exchange_data(world, tmp_path):
    run_dir, res = run(world, tmp_path)
    rows = {r["symbol"]: r for r in res["universe"]["rows"]}
    assert rows["AAAUSDT"]["status"] == U.ELIGIBLE and rows["AAAUSDT"]["rank"] == 1
    assert rows["CCCUSDT"]["status"] == U.ELIGIBLE
    assert rows["DDDUSDT"]["status"] == U.INSUFFICIENT_LISTING_HISTORY
    assert rows["DDDUSDT"]["detail"].startswith("first bar 2025-01-25")
    assert rows["EEEUSDT"]["status"] == U.INSUFFICIENT_LISTING_HISTORY
    assert rows["FFFUSDT"]["status"] == U.INSUFFICIENT_PRESTART_VOLUME_DATA
    assert "/720 hourly bars" in rows["FFFUSDT"]["detail"]
    assert rows["GGGUSDT"]["status"] == U.NOT_ACTIVE_AT_START
    assert rows["QQQUSDT"]["status"] == U.NOT_ACTIVE_AT_START
    assert "zero traded volume" in rows["QQQUSDT"]["detail"]
    assert rows["USDCUSDT"]["status"] == U.EXCLUDED_STABLECOIN_BASE
    assert rows["BTCDOMUSDT"]["status"] == U.EXCLUDED_INDEX_CONTRACT
    assert rows["XYZUSDT_250328"]["status"] == U.EXCLUDED_NOT_USDT_PERPETUAL
    assert rows["ETHUSDC"]["status"] == U.EXCLUDED_NOT_USDT_PERPETUAL
    # Later-delisted contract is a candidate and can be selected.
    assert rows["HHHUSDT"]["status"] == U.ELIGIBLE and rows["HHHUSDT"]["selected"]
    # Only the archive was contacted; never exchangeInfo or other live endpoints.
    assert all(c.startswith((A.FILE_BASE, A.LIST_BASE)) for c in world.calls)
    assert (run_dir / "universe_candidates.csv").exists()


def test_symbol_checks_and_categories(world, tmp_path):
    _, res = run(world, tmp_path)
    ev = {e["symbol"]: e for e in res["evaluations"]}
    assert ev["AAAUSDT"]["included"] and ev["CCCUSDT"]["included"]
    assert ev["AAAUSDT"]["metrics"]["days_required"] == 12
    assert ev["AAAUSDT"]["metrics"]["median_interval_s"] == 300

    j = ev["JJJUSDT"]
    assert S.METRICS_DAY_COVERAGE_BELOW_MIN in j["reasons"]
    assert j["metrics"]["file_missing_days"] == 1

    k = ev["KKKUSDT"]
    assert k["metrics"]["corrupt_or_empty_days"] == 1 and not k["included"]

    l = ev["LLLUSDT"]
    assert S.METRICS_CADENCE_COARSER_THAN_5M in l["reasons"]
    assert S.METRICS_MEDIAN_ROWS_BELOW_MIN in l["reasons"]
    assert l["metrics"]["days_insufficient_rows"] == 12
    assert l["metrics"]["days_excessive_gaps"] == 0      # 15 min is not > 15 min

    m = ev["MMMUSDT"]
    assert S.METRICS_OI_NULL_OR_ZERO_ABOVE_MAX in m["reasons"]
    assert m["metrics"]["oi_null_or_zero_ratio"] == pytest.approx(0.1, abs=0.01)

    assert ev["NNNUSDT"]["reasons"] == [S.FUNDING_UNAVAILABLE]

    o = ev["OOOUSDT"]
    assert o["metrics"]["download_failed_days"] == 1
    assert res["download_failures"] >= 1


def test_delisting_ends_window_without_counting_missing(world, tmp_path):
    _, res = run(world, tmp_path)
    h = {e["symbol"]: e for e in res["evaluations"]}["HHHUSDT"]
    assert h["delisted"] and h["window_end"] == ms(2025, 3, 3, 12)
    assert h["metrics"]["days_required"] == 10
    assert h["klines"]["missing_ratio"] == 0.0 and h["included"]


def test_zero_volume_placeholder_bars_end_window(world, tmp_path):
    _, res = run(world, tmp_path)
    p = {e["symbol"]: e for e in res["evaluations"]}["PPPUSDT"]
    assert p["delisted"] and p["window_end"] == ms(2025, 3, 2)
    assert p["delisting_basis"].startswith("zero-volume")
    assert p["metrics"]["days_required"] == 8          # Feb 22 .. Mar 1
    assert p["included"]


def test_short_zero_volume_run_is_not_delisting(world, tmp_path):
    _, res = run(world, tmp_path, stage_a={**cfg()["stage_a"],
                                           "delisting_zero_volume_days": 30})
    p = {e["symbol"]: e for e in res["evaluations"]}["PPPUSDT"]
    assert not p["delisted"] and p["window_end"] == END


def test_daily_fallback_for_unpublished_month(world, tmp_path):
    _, res = run(world, tmp_path)
    i = {e["symbol"]: e for e in res["evaluations"]}["IIIUSDT"]
    assert i["included"] and not i["delisted"]
    assert i["klines"]["missing_ratio"] == 0.0
    assert any("/daily/klines/IIIUSDT/" in c for c in world.calls)


def test_feasible_and_not_feasible(world, tmp_path):
    _, res = run(world, tmp_path)
    assert res["verdict"]["verdict"] == S.FEASIBLE
    assert set(res["verdict"]["included"]) == {"AAAUSDT", "BBBUSDT", "CCCUSDT",
                                               "HHHUSDT", "IIIUSDT", "PPPUSDT"}
    _, res2 = run(world, tmp_path, min_feasible_symbols=7)
    assert res2["verdict"]["verdict"] == S.NOT_FEASIBLE
    assert "6 symbols passed" in res2["verdict"]["reasons"][0]


def test_cadence_majority_rule():
    evs = [{"symbol": f"S{i}", "included": False,
            "reasons": [S.METRICS_CADENCE_COARSER_THAN_5M]} for i in range(3)]
    evs += [{"symbol": "OK", "included": True, "reasons": []}]
    v = S.verdict(["S0", "S1", "S2", "OK"], evs, cfg(min_feasible_symbols=1))
    assert v["verdict"] == S.NOT_FEASIBLE and "cadence" in v["reasons"][0]


def test_rerun_resumes_from_cache_and_keeps_runs(world, tmp_path):
    run(world, tmp_path)
    world.calls.clear()
    world.override.clear()                                # outage resolved
    run_dir2, res2 = run(world, tmp_path)
    downloaded = [c for c in world.calls
                  if c.startswith(A.FILE_BASE) and not c.endswith(".CHECKSUM")]
    # Only previously non-valid files are requested again.
    assert A.FILE_BASE + A.metrics_key("OOOUSDT", date(2025, 3, 2)) in downloaded
    assert A.FILE_BASE + A.metrics_key("AAAUSDT", date(2025, 3, 1)) not in downloaded
    assert {e["symbol"]: e for e in res2["evaluations"]}["OOOUSDT"]["included"]
    index = list(csv.DictReader(open(tmp_path / "reports" / "stage_a" / "runs_index.csv",
                                     encoding="utf-8")))
    assert len(index) == 2 and index[0]["run_id"] != index[1]["run_id"]


def test_report_and_snapshot(world, tmp_path):
    run_dir, res = run(world, tmp_path)
    snap = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    assert snap["code_version"] == R.R2_CODE_VERSION and len(snap["config_hash"]) == 64
    text = (run_dir / "stage_a_report.md").read_text(encoding="utf-8")
    for needle in ("## Verdict: FEASIBLE", "File missing", "Download failed",
                   "Corrupt/empty", "Days < min rows", "Days with gaps", "OI null/zero",
                   "INSUFFICIENT_PRESTART_VOLUME_DATA", "INSUFFICIENT_LISTING_HISTORY",
                   "WARNING", "Stage B has not been built"):
        assert needle in text, needle
    manifest = list(csv.DictReader(open(run_dir / "file_manifest.csv", encoding="utf-8")))
    assert {r["status"] for r in manifest} >= {A.VALID, A.FILE_MISSING,
                                               A.DOWNLOAD_FAILED, A.CORRUPT_OR_EMPTY}
