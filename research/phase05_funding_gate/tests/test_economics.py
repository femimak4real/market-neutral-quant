import pytest

from research.phase05_funding_gate import common as C
from research.phase05_funding_gate import economics as E
from research.phase05_funding_gate.tests.helpers import base_cfg


def test_funding_sign_short_receives_positive_rate():
    assert E.funding_payment_short(0.0001, 10, 100) == pytest.approx(0.1)


def test_funding_sign_short_pays_negative_rate():
    assert E.funding_payment_short(-0.0001, 10, 100) == pytest.approx(-0.1)


def test_four_fill_fees():
    # qty 10: entry spot 100, perp 101; exit spot 110, perp 111
    fees = E.round_trip_fees(10, 100, 101, 110, 111, 0.001, 0.0005)
    expected = 10 * (100 * 0.001 + 101 * 0.0005 + 110 * 0.001 + 111 * 0.0005)
    assert fees == pytest.approx(expected)
    assert fees == pytest.approx(E.leg_fees(10, 100, 101, 0.001, 0.0005)
                                 + E.leg_fees(10, 110, 111, 0.001, 0.0005))


def test_slippage_four_fills():
    assert E.slippage_cost(1000, 5) == pytest.approx(2.0)


def test_basis_pnl_hand_example():
    # spot 100 -> 110 (+10), perp 101 -> 110.5 (+9.5): long spot / short perp
    assert E.basis_pnl(2, 100, 110, 101, 110.5) == pytest.approx(2 * (10 - 9.5))


def test_margin_diagnostic_flagged_at_limit():
    m = E.margin_consumption(1000, 2.0, 10, 100, 135)
    assert m["futures_margin"] == pytest.approx(500)
    assert m["adverse_pnl"] == pytest.approx(350)
    assert m["consumption"] == pytest.approx(0.70)
    assert m["remaining_futures_equity"] == pytest.approx(150)
    assert E.is_consumption_flagged(m["consumption"], 0.70)


def test_margin_diagnostic_not_flagged_below_limit():
    m = E.margin_consumption(1000, 2.0, 10, 100, 134)
    assert m["consumption"] == pytest.approx(0.68)
    assert not E.is_consumption_flagged(m["consumption"], 0.70)


def test_margin_diagnostic_price_falling_only():
    m = E.margin_consumption(1000, 2.0, 10, 100, 95)
    assert m["consumption"] == 0.0


def test_capital_allocation():
    assert E.capital_allocation(1000, 2.0) == pytest.approx(1500)


def test_breakeven_rate():
    rt = E.round_trip_cost_fraction(0.001, 0.0005, 5)
    assert rt == pytest.approx(0.002 + 0.001 + 0.002)
    assert E.breakeven_rate(rt, 10) == pytest.approx(rt / 10)
    assert E.breakeven_rate(rt, None) is None


def test_max_drawdown():
    assert E.max_drawdown([5, 3, 8, 2, 4]) == pytest.approx(6)
    assert E.max_drawdown([-2, -1]) == pytest.approx(2)


def test_refuses_null_spot_fee():
    cfg = base_cfg(fees={"spot_taker": None, "perp_taker": 0.0005})
    with pytest.raises(C.ConfigError, match="spot_taker"):
        C.validate_config(cfg)


def test_refuses_null_perp_fee():
    cfg = base_cfg(fees={"spot_taker": 0.001, "perp_taker": None})
    with pytest.raises(C.ConfigError, match="perp_taker"):
        C.validate_config(cfg)


def test_refuses_short_period():
    with pytest.raises(C.ConfigError, match="12 months"):
        C.validate_config(base_cfg(end_date="2025-06-01"))


def test_valid_config_passes():
    C.validate_config(base_cfg())


def test_threshold_tolerates_float_noise_at_baseline():
    import numpy as np
    noisy_mean = np.array([0.0001, 0.0003, -0.0001]).mean()  # 9.999999999999999e-05
    assert noisy_mean < 0.0001                               # raw comparison fails
    assert E.meets_threshold(noisy_mean, 0.0001)
    assert not E.meets_threshold(0.0000999, 0.0001)          # real shortfall still fails


def test_below_threshold_is_exact_negation():
    for v in (-0.0001, 0.0, 1e-15, 0.0001, 0.00009999999999999999):
        for thr in (0.0, 0.0001):
            assert E.below_threshold(v, thr) == (not E.meets_threshold(v, thr))
