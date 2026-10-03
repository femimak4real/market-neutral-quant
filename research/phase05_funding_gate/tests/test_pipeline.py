"""End-to-end run on synthetic data files (no network)."""
from research.phase05_funding_gate import backtest as B
from research.phase05_funding_gate import common as C
from research.phase05_funding_gate import report as R
from research.phase05_funding_gate.tests.helpers import (
    base_cfg, random_walk_symbol, write_symbol_csvs)


def test_end_to_end_synthetic(tmp_path):
    symbols = [f"S{i}USDT" for i in range(6)]
    cfg = base_cfg(symbols=symbols, min_oos_trades=5, min_profitable_symbols=2)
    C.validate_config(cfg)
    start, end = C.period_bounds(cfg)
    data_dir = tmp_path / "data"
    for i, sym in enumerate(symbols):
        write_symbol_csvs(data_dir, sym, *random_walk_symbol(i, start, end))

    run_dir = B.create_run(cfg, tmp_path / "reports")
    assert (run_dir / "config_snapshot.json").exists()
    results = B.run_backtest(cfg, data_dir, run_dir)
    B.record_run(tmp_path / "reports", results)

    assert results["gate"]["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert len(results["data_quality"]) == 6
    assert all(q["included"] for q in results["data_quality"])
    assert results["segments"]["out_of_sample"]["universe"]["trades"] > 0
    assert (run_dir / "trades.csv").exists()

    path = R.write_report(run_dir)
    text = path.read_text(encoding="utf-8")
    assert "## Verdict:" in text
    assert results["config_hash"] in text
    assert "Known limitations" in text


def test_runs_index_upgrade_keeps_old_run(tmp_path):
    import csv
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "runs_index.csv").write_text(
        "run_id,created_utc,config_hash,selected_threshold,verdict\n"
        "run_old,20261001T115430Z,9513951a,0.00015,INCONCLUSIVE\n", encoding="utf-8")
    previous = B.record_run(reports, {
        "run_id": "run_new", "created_utc": "x", "config_hash": "abc",
        "code_version": C.CODE_VERSION, "selected_threshold": 0.0001,
        "gate": {"verdict": "FAIL"}})
    assert previous[0]["run_id"] == "run_old"
    rows = list(csv.DictReader(open(reports / "runs_index.csv", encoding="utf-8")))
    assert [r["run_id"] for r in rows] == ["run_old", "run_new"]
    assert rows[0]["code_version"] == "0.5.0"
    assert rows[1]["code_version"] == C.CODE_VERSION
