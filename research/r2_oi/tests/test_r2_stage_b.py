"""R2 Stage B tests (spec R2.15)."""
import json
import math
from datetime import date

import numpy as np
import pytest

from research.phase05_funding_gate import common as C
from research.r2_oi import features as F
from research.r2_oi import simulate as SIM
from research.r2_oi import stage_a as SA
from research.r2_oi import stage_b as SB
from research.r2_oi.tests.fake_archive import FakeArchive, H, ms
from research.r2_oi.tests.test_r2_stage_a import cfg as stage_a_cfg

DAY = 86_400_000
T0 = ms(2025, 1, 1)


def mk(sym, prices, start=T0, window_end=None, f_times=(), f_rates=(), slip=0.0):
    prices = np.asarray(prices, float)
    bo = start + np.arange(len(prices)) * H
    ft = np.asarray(f_times, np.int64)
    fr = np.asarray(f_rates, float)
    fp = np.interp(ft, bo, prices) if len(ft) else np.array([])
    return SIM.SymbolMarket(sym, bo, prices.copy(), prices * 1.0, prices.copy(),
                            window_end or int(bo[-1] + H), ft,
                            SIM.make_funding_cum(ft, fr, fp), slip)


P = SIM.SimParams(n_per_side=1, gross_notional=1000.0, fee=0.0005, leverage=2.0)


def run_sim(market, scores, times, seg_end, **kw):
    return SIM.simulate(market, times, scores, SIM.fill_matrix(market, times), seg_end, P, **kw)


# ---------------------------------------------------------------- features
def test_point_in_time_metrics_guard_and_bars():
    T = ms(2025, 1, 2)
    ct = np.array([T - 6 * 60_000, T - 4 * 60_000])
    v = np.array([1.0, 999.0])
    assert F.oi_at(ct, v, np.array([T]))[0] == 1.0          # row at T-4m not knowable
    bo = np.array([T - 2 * H, T - H, T])                   # bar at T ends after T
    cl = np.array([10.0, 11.0, 50.0])
    assert F.close_at(bo, cl, np.array([T]))[0] == 11.0
    assert np.isnan(F.oi_at(ct, v, np.array([T + 3 * H]))[0])   # stale > 1h


def test_zscore_uses_prior_values_only():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    z = F.zscore_prior(x, 4)
    assert z[4] == pytest.approx((100 - 2.5) / np.std([1, 2, 3, 4], ddof=1))
    x2 = x.copy(); x2[4] = -50
    z2 = F.zscore_prior(x2, 4)
    assert z2[4] == pytest.approx((-50 - 2.5) / np.std([1, 2, 3, 4], ddof=1))
    assert np.isnan(z[:4]).all()


def test_signal_signs_and_quadrants():
    rng = np.random.default_rng(0)
    d_oi, ret = rng.normal(0, 1, 40), rng.normal(0, 1, 40)
    d_oi[-1], ret[-1] = 5.0, 5.0                            # OI up + price up
    sig = F.signals(d_oi, ret, 30)
    assert sig["S"][-1] < 0 and sig["C"][-1] < 0            # short candidate
    q = F.quadrant_codes(np.array([1, 1, -1, -1, 0]), np.array([1, -1, -1, 1, 1]))
    assert [F.QUADRANTS[i] for i in q] == ["Crowding-long proxy", "Crowding-short proxy",
                                           "Flush proxy", "Covering proxy", "Neutral"]


def test_anti_leakage_funding_does_not_change_scores():
    def market(rate):
        ms_ = [mk(f"S{i}USDT", 100 + np.sin(np.arange(24 * 60) / (5 + i)) * 5,
                  f_times=[T0 + 8 * H * k for k in range(150)], f_rates=[rate] * 150)
               for i in range(4)]
        for j, m in enumerate(ms_):
            m.m_time = m.bar_open.copy()
            m.m_oiv = 1e6 * (1 + 0.1 * np.cos(np.arange(len(m.bar_open)) / (3 + j)))
        return ms_
    times = T0 + np.arange(10, 55) * DAY
    a = SB.build_matrices(market(0.0001), times, [4, 8, 12], 5)
    b = SB.build_matrices(market(-0.01), times, [4, 8, 12], 5)
    for L in (4, 8, 12):
        np.testing.assert_array_equal(a[L][0], b[L][0])
        np.testing.assert_array_equal(a[L][1], b[L][1])


# ---------------------------------------------------------------- simulation
def flat_market(n=4, hours=24 * 6, **kw):
    return [mk(f"{chr(65 + i)}USDT", [100.0] * hours, **kw) for i in range(n)]


def test_dollar_neutral_targets_and_unchanged_portfolio_has_no_fees():
    market = flat_market()
    times = T0 + np.arange(4) * DAY
    scores = np.tile([3.0, 2.0, 1.0, 0.0], (4, 1))
    r = run_sim(market, scores, times, T0 + 4 * DAY)
    assert r.entry_notional[0] == 1000.0                    # 500 long + 500 short
    assert (r.entry_notional[1:] == 0).all() and (r.exit_notional == 0).all()
    longs = [l for l in r.lots if l["side"] > 0]
    shorts = [l for l in r.lots if l["side"] < 0]
    assert sum(l["entry_notional"] for l in longs) == sum(l["entry_notional"] for l in shorts)
    assert {l["symbol"] for l in longs} == {"AUSDT"} and {l["symbol"] for l in shorts} == {"DUSDT"}
    assert sum(l["entry_fee"] + l["exit_fee"] for l in r.lots) == pytest.approx(1000 * 0.0005 * 2)


def test_turnover_hand_example():
    market = flat_market()
    times = T0 + np.arange(2) * DAY
    scores = np.array([[3.0, 2.0, 1.0, 0.0], [2.0, 3.0, 1.0, 0.0]])   # long A -> long B
    r = run_sim(market, scores, times, T0 + 2 * DAY)
    assert r.entry_notional.tolist() == [1000.0, 500.0]
    assert r.exit_notional.tolist() == [0.0, 500.0]


def test_ties_broken_by_symbol():
    market = flat_market()
    r = run_sim(market, np.zeros((1, 4)), np.array([T0]), T0 + DAY)
    assert {l["symbol"]: l["side"] for l in r.lots} == {"AUSDT": 1, "DUSDT": -1}


def test_insufficient_breadth_is_flat():
    market = flat_market()
    scores = np.array([[1.0, np.nan, np.nan, np.nan]])
    r = run_sim(market, scores, np.array([T0]), T0 + DAY)
    assert r.lots == [] and not r.full[0] and r.eligible[0] == 1


def test_funding_sign_and_timing():
    ft = [T0, T0 + 8 * H, T0 + DAY]                       # at decision, during, next decision
    fr = [0.01, 0.001, 0.002]
    market = [mk("AUSDT", [100.0] * 72, f_times=ft, f_rates=fr)] + flat_market(4, 72)[1:]
    times = T0 + np.arange(2) * DAY
    scores = np.array([[3.0, 2.0, 1.0, 0.0], [0.0, 3.0, 2.0, 1.0]])   # A long, then removed
    r = run_sim(market, scores, times, T0 + 2 * DAY)
    a = [l for l in r.lots if l["symbol"] == "AUSDT"][0]
    qty = a["qty"]
    # T0 settlement excluded (before the 01:00 fill); 08:00 and next 00:00 included
    # (removed position accrues until its 01:00 exit fill). Long pays positive funding.
    assert a["funding"] == pytest.approx(-qty * 100 * (0.001 + 0.002))
    assert a["exit_time"] == T0 + DAY + H


def test_delisting_closes_at_last_traded_bar():
    prices = [100.0] * 30 + [120.0] * 10
    a = mk("AUSDT", prices, window_end=T0 + 40 * H)
    market = [a] + flat_market(4, 24 * 4)[1:]
    times = T0 + np.arange(3) * DAY
    scores = np.tile([3.0, 2.0, 1.0, 0.0], (3, 1))
    r = run_sim(market, scores, times, T0 + 3 * DAY)
    lot = [l for l in r.lots if l["symbol"] == "AUSDT"][0]
    assert lot["exit_reason"] == "delisted" and lot["exit_time"] == T0 + 40 * H
    assert lot["exit_price"] == 120.0


def test_equity_curve_ends_at_total_net():
    rng = np.random.default_rng(3)
    market = [mk(f"{c}USDT", 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 24 * 8))),
                 f_times=[T0 + 8 * H * k for k in range(24)],
                 f_rates=rng.normal(0, 0.001, 24), slip=5) for c in "ABCDEF"]
    times = T0 + np.arange(7) * DAY
    scores = rng.normal(size=(7, 6))
    r = run_sim(market, scores, times, T0 + 7 * DAY)
    eq = SIM.equity_curve(r.lots, market, SIM.day_boundaries(T0, T0 + 7 * DAY))
    assert eq[0] == pytest.approx(0.0)
    assert eq[-1] == pytest.approx(r.net_pnl)


def test_permutation_reproducible_and_scores_only():
    rng = np.random.default_rng(5)
    market = [mk(f"{c}USDT", 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 24 * 12))))
              for c in "ABCDEFGH"]
    times = T0 + np.arange(10) * DAY
    scores = rng.normal(size=(10, 8))
    scores[3, 2] = np.nan
    runs = [[run_sim(market, scores, times, T0 + 10 * DAY, collect=False,
                     rng=np.random.default_rng(seed)).net_pnl for _ in range(1)]
            for seed in (1, 1, 2)]
    assert runs[0] == runs[1]
    actual = run_sim(market, scores, times, T0 + 10 * DAY)
    perm = run_sim(market, scores, times, T0 + 10 * DAY, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(actual.eligible, perm.eligible)   # same universe each day


def test_beta_known_and_breadth():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 0.02, 500)
    y = 0.5 * x + rng.normal(0, 0.0005, 500)
    beta, se = SIM.beta_stats(y, x)
    assert beta == pytest.approx(0.5, abs=0.01) and se < 0.01
    b = SIM.breadth_stats(np.array([8, 9, 10, 20, 20]), 4)
    assert b["min"] == 8 and b["median"] == 10 and b["thin_rebalances"] == 2


# ---------------------------------------------------------------- gate
def gate_metrics(**kw):
    m = {"stage_a_feasible": True, "oos_full_rebalances": 290, "oos_symbols_eligible": 18,
         "oos_net": 500.0, "oos_control_net": 300.0, "oos_net_stressed": 200.0,
         "neighbor_net": {"lower": 10.0, "upper": 20.0}, "oos_annualized_return": 0.2,
         "oos_max_dd_pct": 0.05, "oos_beta": 0.05, "perm_quantile_value": 100.0}
    m.update(kw)
    return m


def gcfg():
    return stage_a_cfg(min_oos_rebalances=200, min_feasible_symbols=15)


def test_gate_pass():
    assert SB.evaluate_gate(gate_metrics(), gcfg())["verdict"] == "PASS"


@pytest.mark.parametrize("kw,crit", [
    ({"oos_net": -1.0, "oos_control_net": -5.0, "perm_quantile_value": -10.0}, 1),
    ({"oos_control_net": 600.0}, 2),
    ({"oos_net_stressed": -1.0}, 3), ({"neighbor_net": {"lower": -1.0, "upper": 5.0}}, 4),
    ({"neighbor_net": {"lower": None, "upper": 5.0}}, 4),
    ({"oos_annualized_return": 0.01}, 5), ({"oos_max_dd_pct": 0.2}, 6),
    ({"oos_beta": -0.3}, 7), ({"perm_quantile_value": 900.0}, 8)])
def test_gate_each_criterion_fails(kw, crit):
    g = SB.evaluate_gate(gate_metrics(**kw), gcfg())
    assert g["verdict"] == "FAIL"
    assert [c["id"] for c in g["criteria"] if not c["passed"]] == [crit]


@pytest.mark.parametrize("kw", [{"stage_a_feasible": False}, {"oos_full_rebalances": 10},
                                {"oos_symbols_eligible": 12}])
def test_gate_inconclusive(kw):
    g = SB.evaluate_gate(gate_metrics(**kw), gcfg())
    assert g["verdict"] == "INCONCLUSIVE" and "STOP" in g["action"]


# ---------------------------------------------------------------- end to end
def e2e_world(n=10):
    f = FakeArchive()
    first, last = ms(2024, 11, 1), ms(2025, 4, 3)
    for i in range(n):
        sym = f"Z{chr(65 + i)}USDT"
        price = (lambda k: (lambda t: 100 * (1 + 0.05 * math.sin(t / (H * (17 + k)) + k)
                                             + 0.03 * math.sin(t / (H * (5 + k)) + 2 * k))))(i)
        oi = (lambda k: (lambda t: 1e6 * (1 + 0.1 * math.sin(t / (H * (11 + k)) + 3 * k))))(i)
        f.add_monthly_klines(sym, first, last, qv=1000.0 * (n - i), price=price)
        f.add_metrics(sym, date(2025, 1, 5), date(2025, 3, 31), oi=oi)
    return f


def e2e_cfg(**over):
    c = stage_a_cfg(start_date="2025-02-10", end_date="2025-03-31", universe_size=10,
                    min_feasible_symbols=6, zscore_window_days=5, n_per_side=2,
                    lookback_grid_hours=[4, 8, 12], min_oos_rebalances=5, n_permutations=20)
    c.update(over)
    return c


def funding(sym, a, b):
    import pandas as pd
    t = np.arange(a - a % (8 * H), b, 8 * H)
    t = t[t >= a]
    return pd.DataFrame({"funding_time": t, "funding_rate": 0.0001, "mark_price": 100.0})


def test_stage_b_refuses_without_feasible_stage_a(tmp_path):
    with pytest.raises(SB.StageBRefused):
        SB.run_stage_b(e2e_cfg(), tmp_path / "data", tmp_path / "reports", verbose=False)


def caching_funding(data_dir):
    def fetch(sym, a, b):
        df = funding(sym, a, b)
        (data_dir / "funding").mkdir(parents=True, exist_ok=True)
        df.to_csv(data_dir / "funding" / f"{sym}_funding.csv", index=False)
        return df
    return fetch


def test_stage_b_end_to_end(tmp_path):
    c = e2e_cfg()
    _, a_res = SA.run_stage_a(c, e2e_world(), caching_funding(tmp_path / "data"),
                              tmp_path / "data", tmp_path / "reports")
    assert a_res["verdict"]["verdict"] == "FEASIBLE"
    run_dir, res = SB.run_stage_b(c, tmp_path / "data", tmp_path / "reports", verbose=False)
    assert res["gate"]["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert (run_dir / "preregistration.json").exists()
    assert (run_dir / "stage_b_report.md").read_text(encoding="utf-8").startswith(
        "# R2 Stage B")
    for key, s in res["segments"].items():
        assert s["equity_end_check"] == pytest.approx(s["net"], abs=1e-6), key
    assert res["permutation"]["n"] == 20
    # A config change after Stage A is refused (hash mismatch).
    with pytest.raises(SB.StageBRefused):
        SB.run_stage_b(e2e_cfg(n_per_side=3), tmp_path / "data", tmp_path / "reports",
                       verbose=False)
