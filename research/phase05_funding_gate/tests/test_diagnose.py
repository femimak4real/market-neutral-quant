import numpy as np

from research.phase05_funding_gate import diagnose_funding as D
from research.phase05_funding_gate.common import HOUR_MS
from research.phase05_funding_gate.tests.helpers import (
    base_cfg, random_walk_symbol, write_symbol_csvs)
from research.phase05_funding_gate import common as C


def test_segment_counts_hand_example():
    H8 = 8 * HOUR_MS
    times = np.arange(6) * H8
    rates = np.array([0.0001, 0.0001, 0.0001, 0.0003, -0.0001, 0.0002])
    s = D.segment_stats(times, rates, 0, 10**15, 3, [0.0001, 0.0002])
    # rolling(3) valid from index 2: means 1e-4, 1.667e-4, 1.0e-4, 1.333e-4
    assert s["settlements"] == 4
    assert s["last_rate_gt_0"] == 3
    assert s["at_baseline"] == 1
    assert s["negative"] == 1
    assert s["mean_k"][0.0001] == 4
    assert s["entry_condition"][0.0001] == 3   # index 4 has last rate < 0
    assert s["mean_k"][0.0002] == 0


def test_lookback_may_use_pre_segment_data():
    H8 = 8 * HOUR_MS
    times = np.arange(5) * H8
    rates = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
    s = D.segment_stats(times, rates, 3 * H8, 10**15, 3, [0.0005])
    assert s["settlements"] == 2 and s["entry_condition"][0.0005] == 2


def test_diagnostic_end_to_end(tmp_path):
    symbols = ["S0USDT", "S1USDT"]
    cfg = base_cfg(symbols=symbols)
    start, end = C.period_bounds(cfg)
    for i, sym in enumerate(symbols):
        write_symbol_csvs(tmp_path / "data", sym, *random_walk_symbol(i, start, end))
    path = D.run(cfg, tmp_path / "data", tmp_path / "diag")
    text = path.read_text(encoding="utf-8")
    assert "Out-Of-Sample" in text and "S1USDT" in text
