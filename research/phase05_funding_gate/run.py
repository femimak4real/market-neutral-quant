"""Run the full Phase 0.5 gate: download -> backtest -> report.

Usage (from repo root):
    python -m research.phase05_funding_gate.run
    python -m research.phase05_funding_gate.run --skip-download
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.phase05_funding_gate import backtest, report
from research.phase05_funding_gate import common as C
from research.phase05_funding_gate import download_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0.5 funding gate")
    parser.add_argument("--config", default=None, help="path to phase05.yaml")
    parser.add_argument("--skip-download", action="store_true",
                        help="use already-downloaded data")
    parser.add_argument("--force-download", action="store_true",
                        help="re-download even if cached")
    args = parser.parse_args(argv)

    try:
        cfg = C.load_config(args.config)
    except C.ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2

    data_dir = C.resolve_dir(cfg, "data_dir", "data/phase05")
    reports_dir = C.resolve_dir(cfg, "reports_dir", "reports/phase05")

    if not args.skip_download:
        try:
            download_data.download_all(cfg, data_dir, force=args.force_download)
        except download_data.RegionBlockedError as exc:
            print(f"ERROR: {exc}")
            return 2

    run_dir = backtest.create_run(cfg, reports_dir)
    try:
        results = backtest.run_backtest(cfg, data_dir, run_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    previous = backtest.record_run(reports_dir, results)
    (run_dir / "previous_runs.json").write_text(json.dumps(previous, indent=2),
                                                encoding="utf-8")
    report_path = report.write_report(run_dir)

    gate = results["gate"]
    print()
    print(f"Report: {report_path}")
    print(f"Selected threshold (in-sample): {results['selected_threshold']}")
    for c in gate["criteria"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['criterion']}: {c['value']}")
    for reason in gate["inconclusive_reasons"]:
        print(f"  [INCONCLUSIVE] {reason}")
    print(f"VERDICT: {gate['verdict']}")
    print(gate["action"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
