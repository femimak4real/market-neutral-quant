# Market-Neutral Quant Bot: Phase 0.5 Funding Gate

Go/no-go research gate for positive-funding arbitrage (long spot + short perp)
on Binance. Specification: `docs/phase05_gate.md`.

This code uses **public market data only**. It needs no API keys and cannot
place orders.

## Setup (Windows, from the repo root)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, run this first (current terminal only):
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Before running

Edit `config/phase05.yaml`:

- `fees.spot_taker` / `fees.perp_taker`: your actual taker fees as fractions
  (0.1% = 0.001). The scripts refuse to run if either is null.
- `hurdle_annual_return`: your realistic alternative yield.
- `diagnostic_margin_consumption_limit` and the threshold grid.

Set these BEFORE the first run. Every run snapshots and hashes the config;
reruns with changed values are reported as separate runs.

## Run

Tests first:

```
pytest -v
```

Then the whole gate in one command (download -> backtest -> report):

```
python -m research.phase05_funding_gate.run
```

Or step by step:

```
python -m research.phase05_funding_gate.download_data
python -m research.phase05_funding_gate.backtest
python -m research.phase05_funding_gate.report
```

Funding-frequency diagnostic (descriptive only, no verdict, not indexed):

```
python -m research.phase05_funding_gate.diagnose_funding
```

Writes `reports/phase05/diagnostics/funding_frequency_<timestamp>.md`.

Useful flags: `run --skip-download` (reuse cached data),
`run --force-download` / `download_data --force` (re-download).

## Output

- `data/phase05/`: downloaded CSVs (cached between runs)
- `reports/phase05/run_<timestamp>_<hash>/`
  - `report.md`: the full report and verdict
  - `trades.csv`, `results.json`, `config_snapshot.json`
- `reports/phase05/runs_index.csv`: every run and its verdict

## Verdict

- PASS: Phase 1 may begin.
- FAIL or INCONCLUSIVE: STOP. Do not build production infrastructure.

## If the download fails with HTTP 451 or 403

Binance is blocking requests from your location. Run from a VPS in an allowed
region, or source the same data from https://data.binance.vision.

## Changelog

- 0.5.0: original release (run 9513951a).
- 0.5.1: float-tolerant threshold comparisons (a mean of three 0.0100% rates
  can compute as 0.009999...% and previously failed a >= 0.0100% test);
  code version recorded in every run; regression tests for gate criterion 4;
  funding-frequency diagnostic. Runs made with 0.5.0 are retained and labelled.

# Research Gate R2: OI-conditioned reversal

Specification: `docs/research_gate_r2_oi_positioning.md` (Revision 2).
Config (pre-registered): `config/r2_oi.yaml`.

## Stage A: data feasibility (built)

```
python -m research.r2_oi.stage_a
```

- Lists the data.binance.vision archive (symbol master incl. delisted
  contracts), builds the point-in-time universe, downloads 5-minute OI
  metrics and 1h klines, fetches funding, and applies the R2.4 checks.
- Output: `reports/r2_oi/stage_a/run_<timestamp>_<hash>/stage_a_report.md`
  with verdict FEASIBLE or NOT FEASIBLE, plus `universe_candidates.csv`,
  `file_manifest.csv`, and `stage_a_results.json`.
- The first run downloads roughly 15,000 small files into `data/r2_oi/`.
  It is resumable: if it stops, run it again and valid cached files are
  skipped. Corrupt or empty cached files are re-downloaded.
- If the report shows download failures, rerun before relying on the verdict.

## Stage B: economic gate (authorized)

```
python -m research.r2_oi.stage_b
```

- Refuses to run unless a FEASIBLE Stage A run exists for the same config hash.
- Reads only the Stage A cache (never downloads).
- Pre-registers (config, Stage A run, universe, conventions) before computing.
- Selects L on in-sample data only, runs OOS once for the OI signal and the
  plain-reversal control, then the neighbours, cost stress, beta, and a
  1,000-run score permutation test.
- Output: `reports/r2_oi/stage_b/run_<timestamp>_<hash>/stage_b_report.md`,
  `stage_b_results.json`, `positions_*.csv`, `preregistration.json`.
- Verdict PASS / FAIL / INCONCLUSIVE. A PASS authorizes a written proposal
  only, never execution.

# Liquidation recorder (forward data collection)

Spec: `docs/collector_liquidations_spec.md`. Config: `config/collector.yaml`.
Records only: no API keys, no orders.

```
pip install -r requirements.txt
python -m research.collector.liquidations run      # Ctrl+C to stop
python -m research.collector.liquidations report   # coverage per UTC day
```

Data goes to `data/collector/`: raw events by UTC day, `gaps.csv`,
`sessions.csv`, `heartbeat.json`. Every period without a live connection is
logged as a gap, including PC sleep and time between runs.
