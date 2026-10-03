import numpy as np
import pandas as pd
import pytest

from research.phase05_funding_gate import backtest as B
from research.phase05_funding_gate import economics as E
from research.phase05_funding_gate.common import HOUR_MS
from research.phase05_funding_gate.tests.helpers import base_cfg, make_raw

H8 = 8 * HOUR_MS


def zero_cost_params(**kw):
    defaults = dict(lookback=1, exit_periods=2, exit_threshold=0.0, notional=1000.0,
                    spot_fee=0.0, perp_fee=0.0, slippage_bps=0.0, leverage=2.0,
                    consumption_limit=0.7)
    defaults.update(kw)
    return B.Params(**defaults)


def test_markprice_is_authoritative_and_fallback_counted():
    funding, perp, _ = make_raw([0.001, 0.001], mark_prices=[150.0, None], price=100.0)
    resolved, dropped = E.resolve_funding_prices(funding, perp)
    assert dropped == 0
    assert list(resolved["funding_price"]) == [150.0, 100.0]
    assert list(resolved["fallback"]) == [False, True]


def test_empty_string_markprice_uses_fallback():
    funding, perp, _ = make_raw([0.001], mark_prices=[""], price=100.0)
    resolved, _ = E.resolve_funding_prices(funding, perp)
    assert resolved["fallback"].tolist() == [True]
    assert resolved["funding_price"].tolist() == [100.0]


def test_no_lookahead_and_accrual_only_while_open():
    rates = [0.0] * 5 + [0.001] * 6 + [-0.001] * 3
    sd = B.prepare_symbol("T", *make_raw(rates))
    trades = B.simulate(sd, zero_cost_params(), 0.0005, 0, 10**15, "in_sample")
    assert len(trades) == 1
    t = trades[0]
    # Decision only when the positive rate has actually settled (t5).
    assert t["entry_decision_time"] == 5 * H8
    # Fill at the first bar starting after the decision.
    assert t["entry_time"] == 5 * H8 + HOUR_MS
    # Funding at t5 is NOT received (not open before t5); t6..t12 are.
    assert t["periods_held"] == 7
    assert t["funding_pnl"] == pytest.approx(10 * 100 * (5 * 0.001 - 2 * 0.001))
    # Exit decided at t12 (last 2 rates < 0); t13 not accrued.
    assert t["exit_decision_time"] == 12 * H8
    assert t["exit_time"] == 12 * H8 + HOUR_MS
    assert t["net_pnl"] == pytest.approx(t["funding_pnl"])


def test_decision_cannot_see_future_rate():
    # A huge rate settles at t5. With K=3 a look-ahead bug would enter at t4.
    rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01]
    sd = B.prepare_symbol("T", *make_raw(rates))
    trades = B.simulate(sd, zero_cost_params(lookback=3), 0.003, 0, 10**15, "in_sample")
    assert trades and trades[0]["entry_decision_time"] == 5 * H8


def test_fees_and_slippage_reduce_net():
    rates = [0.001] * 4 + [-0.001] * 2
    sd = B.prepare_symbol("T", *make_raw(rates))
    p = zero_cost_params(spot_fee=0.001, perp_fee=0.0005, slippage_bps=5)
    t = B.simulate(sd, p, 0.0005, 0, 10**15, "in_sample")[0]
    expected_costs = 10 * 100 * (2 * 0.001 + 2 * 0.0005) + 4 * 5 / 10_000 * 1000
    assert t["entry_fees"] + t["exit_fees"] + t["slippage"] == pytest.approx(expected_costs)
    assert t["net_pnl"] == pytest.approx(t["funding_pnl"] - expected_costs)


def test_forced_exit_at_segment_end():
    rates = [0.001] * 10
    sd = B.prepare_symbol("T", *make_raw(rates))
    t = B.simulate(sd, zero_cost_params(), 0.0005, 0, 6 * H8, "in_sample")[0]
    assert t["forced_exit"] is True
    assert t["exit_time"] == 6 * H8


def test_threshold_selection_ignores_oos_data():
    rng = np.random.default_rng(1)
    base = list(rng.normal(0.0002, 0.0002, 200))
    split = 120 * H8
    sd_a = B.prepare_symbol("T", *make_raw(base))
    altered = base[:120] + [0.01 if i % 2 else -0.01 for i in range(80)]
    sd_b = B.prepare_symbol("T", *make_raw(altered))
    p = {"T": zero_cost_params(lookback=3)}
    grid = [0.00005, 0.0001, 0.0002, 0.0003]
    rows_a, sel_a = B.select_threshold({"T": sd_a}, p, grid, 0, split)
    rows_b, sel_b = B.select_threshold({"T": sd_b}, p, grid, 0, split)
    assert sel_a == sel_b
    assert rows_a == rows_b


def test_margin_flag_in_simulation():
    rates = [0.001] * 4 + [-0.001] * 2
    funding, perp, spot = make_raw(rates)
    perp.loc[20, "high"] = 136.0   # 36% spike while the position is open
    sd = B.prepare_symbol("T", funding, perp, spot)
    t = B.simulate(sd, zero_cost_params(), 0.0005, 0, 10**15, "in_sample")[0]
    assert t["margin_flagged"] is True
    assert t["consumption"] == pytest.approx(0.72)


def _metrics(**kw):
    m = {"oos_trades": 50, "symbols_included": 9, "oos_fallback_ratio": 0.0,
         "oos_net_pnl": 500.0, "profitable_symbols": 5, "oos_net_pnl_stressed": 200.0,
         "neighbor_oos_net": {"lower": 100.0, "upper": 50.0},
         "oos_annualized_return": 0.08, "oos_max_drawdown_pct": 0.01}
    m.update(kw)
    return m


def test_gate_pass():
    assert B.evaluate_gate(_metrics(), base_cfg())["verdict"] == "PASS"


def test_gate_fail_on_stressed_costs():
    g = B.evaluate_gate(_metrics(oos_net_pnl_stressed=-1.0), base_cfg())
    assert g["verdict"] == "FAIL"
    assert [c["passed"] for c in g["criteria"]][2] is False


def test_gate_fail_on_hurdle():
    assert B.evaluate_gate(_metrics(oos_annualized_return=0.03),
                           base_cfg())["verdict"] == "FAIL"


def test_gate_fail_when_neighbor_unavailable():
    g = B.evaluate_gate(_metrics(neighbor_oos_net={"lower": None, "upper": 10.0}),
                        base_cfg())
    assert g["verdict"] == "FAIL"


def test_gate_inconclusive_few_trades():
    g = B.evaluate_gate(_metrics(oos_trades=5), base_cfg())
    assert g["verdict"] == "INCONCLUSIVE"
    assert "STOP" in g["action"]


def test_gate_inconclusive_few_symbols():
    assert B.evaluate_gate(_metrics(symbols_included=5),
                           base_cfg())["verdict"] == "INCONCLUSIVE"


def test_gate_inconclusive_fallback_ratio():
    assert B.evaluate_gate(_metrics(oos_fallback_ratio=0.2),
                           base_cfg())["verdict"] == "INCONCLUSIVE"


def test_funding_missing_ratio_variable_interval():
    t8 = np.arange(0, 100) * H8
    t4 = t8[-1] + np.arange(1, 101) * 4 * HOUR_MS
    times = np.concatenate([t8, t4])
    end = int(times[-1] + 4 * HOUR_MS)
    assert B.funding_missing_ratio(times, 0, end) < 0.02
    gappy = np.delete(times, np.arange(10, 60))
    assert B.funding_missing_ratio(gappy, 0, end) > 0.02


def test_gate_criterion4_fails_with_negative_neighbors_run_9513951a():
    """Regression: exact values from run_20261001T115430Z_9513951a."""
    g = B.evaluate_gate(_metrics(neighbor_oos_net={"lower": -276.89, "upper": -0.49}),
                        base_cfg())
    crit4 = g["criteria"][3]
    assert crit4["id"] == 4 and crit4["passed"] is False
    assert g["verdict"] == "FAIL"


def test_gate_criterion4_fails_if_either_neighbor_not_positive():
    for lower, upper in ((-1.0, 5.0), (5.0, -1.0), (0.0, 5.0), (5.0, 0.0)):
        g = B.evaluate_gate(_metrics(neighbor_oos_net={"lower": lower, "upper": upper}),
                            base_cfg())
        assert g["criteria"][3]["passed"] is False


def test_report_renders_criterion4_fail_after_json_roundtrip():
    import json
    from research.phase05_funding_gate import report as R
    g = B.evaluate_gate(_metrics(neighbor_oos_net={"lower": -276.89, "upper": -0.49}),
                        base_cfg())
    g = json.loads(json.dumps(g))
    line = [ln for ln in R.table(["#", "Criterion", "Value", "Result"],
            [[c["id"], c["criterion"], c["value"], "PASS" if c["passed"] else "FAIL"]
             for c in g["criteria"]]).splitlines() if ln.startswith("| 4 |")][0]
    assert line.endswith("| FAIL |")


def test_entry_at_threshold_equal_to_baseline_despite_float_noise():
    # mean(0.0001, 0.0003, -0.0001) computes as 9.999999999999999e-05;
    # the last rate is negative here, so use a window ending positive:
    rates = [0.0003, -0.0001, 0.0001, 0.0001, 0.0001, -0.001, -0.001]
    sd = B.prepare_symbol("T", *make_raw(rates))
    t = B.simulate(sd, zero_cost_params(lookback=3), 0.0001, 0, 10**15, "in_sample")
    # Index 2: mean(0.0003,-0.0001,0.0001)=0.0001 (noisy) and last>0 -> enter at t2
    assert t and t[0]["entry_decision_time"] == 2 * H8
