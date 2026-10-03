# Liquidation Recorder Spec (forward data collection)

## Purpose

Record Binance USDⓈ-M liquidation events live, because Binance no longer
publishes historical liquidation data. OI, funding and klines are NOT
collected here: they remain available historically from the archive and REST
API and are fetched by research code when needed.

## Rules

- Records only. No API keys, no orders, no account access.
- Source: the public all-market liquidation stream `!forceOrder@arr`.
- Binance pushes at most the latest liquidation per symbol per 1000 ms
  (snapshot). Recorded data is therefore a **lower bound** on true
  liquidations, and every analysis must say so.
- Every message is stored raw with its local receive time. Nothing is
  filtered, aggregated or corrected at collection time.
- Every period without a live connection is recorded as a gap, with a
  reason. Gaps are never filled or interpolated.

## Storage (`data/collector/`)

- `liquidations/YYYY-MM-DD.jsonl`: one line per message,
  `{"recv_ms": ..., "msg": {...raw...}}`, by UTC receive date. Completed days
  are compressed to `.jsonl.gz`.
- `gaps.csv`: `start_ms, end_ms, seconds, reason`.
- `sessions.csv`: START / CONNECTED / DISCONNECTED / STOP events with detail.
- `heartbeat.json`: last time the recorder was alive.

## Gap detection

- Recorder not running (PC off, process stopped): from the last heartbeat to
  the next start.
- Process paused (system sleep): heartbeat ticks further apart than
  `gap_threshold_seconds`.
- Disconnect (including Binance's 24-hour connection limit): from disconnect
  to reconnect, however short.
- Silent connection: connected but no message for `max_silence_minutes`.
  The gap runs from the last message received to the reconnect. A
  connection is only coverage if data actually arrives.
- Endpoint: Binance routes USDⓈ-M futures streams by path (/public, /market,
  /private). Unrouted legacy URLs can connect but not deliver. `ws_url` must
  be a routed URL confirmed by a live probe.
- Overlapping gaps are merged when computing coverage.

## Coverage report

`python -m research.collector.liquidations report` prints, per UTC day:
events, symbols, uptime %, gap count and longest gap. Coverage is measured
from the first recorder start.

## Usage of the data

Any future study must state its coverage, exclude or flag low-coverage days
by a rule fixed in that study's own pre-registered spec, and treat counts
and notionals as lower bounds.
