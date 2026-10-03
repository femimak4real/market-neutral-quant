"""Liquidation recorder tests: no network (fake WebSocket and clock)."""
import asyncio
import csv
import gzip
import json

import pytest

from research.collector import liquidations as L

D1 = 1_759_276_800_000          # 2025-10-01 00:00 UTC
CFG = {"ws_url": "wss://example/ws", "data_dir": "x", "heartbeat_seconds": 0.01,
       "gap_threshold_seconds": 90, "gzip_completed_days": True,
       "max_silence_minutes": 0.001}


def event(sym, t, q="1", p="100", ap="100", z="1"):
    return json.dumps({"e": "forceOrder", "E": t, "o": {
        "s": sym, "S": "SELL", "o": "LIMIT", "f": "IOC", "q": q, "p": p, "ap": ap,
        "X": "FILLED", "l": z, "z": z, "T": t}})


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class FakeWS:
    """Async context manager yielding scripted messages, then closing or failing."""

    def __init__(self, clock, script, fail=None, hang=False):
        self.clock, self.script, self.fail, self.hang = clock, script, fail, hang

    async def __aenter__(self):
        if self.fail == "connect":
            raise OSError("connection refused")
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for t, raw in self.script:
            self.clock.t = t
            yield raw
            await asyncio.sleep(0)
        if self.fail == "drop":
            raise ConnectionResetError("dropped")
        if self.hang:
            await asyncio.sleep(3600)            # connected, never sends again


def run(rec):
    asyncio.run(rec.run())


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_records_raw_events_and_reconnect_gap(tmp_path):
    clock = Clock(D1)
    conns = iter([
        ([(D1 + 1000, event("BTCUSDT", D1 + 900)), (D1 + 2000, event("ETHUSDT", D1 + 1900))], "drop"),
        ([(D1 + 9000, event("SOLUSDT", D1 + 8900))], None),
    ])

    def connect(url):
        script, fail = next(conns)
        if script and script[0][0] == D1 + 9000:
            clock.t = D1 + 8000                      # reconnected 6 s after the drop
        return FakeWS(clock, script, fail)

    store = L.Store(tmp_path)
    run(L.Recorder(CFG, store, connect=connect, clock=clock, max_connections=2))
    lines = (tmp_path / "liquidations" / "2025-10-01.jsonl").read_text().splitlines()
    assert [json.loads(x)["msg"]["o"]["s"] for x in lines] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert json.loads(lines[0])["recv_ms"] == D1 + 1000
    gaps = rows(tmp_path / "gaps.csv")
    drop = [g for g in gaps if g["reason"].startswith("ConnectionResetError")]
    assert drop and int(drop[0]["start_ms"]) == D1 + 2000 and int(drop[0]["end_ms"]) == D1 + 8000
    events = [r["event"] for r in rows(tmp_path / "sessions.csv")]
    assert events[0] == "START" and events[-1] == "STOP"
    assert events.count("CONNECTED") == 2 and "DISCONNECTED" in events


def test_downtime_between_runs_is_a_gap(tmp_path):
    clock = Clock(D1)
    store = L.Store(tmp_path)
    run(L.Recorder(CFG, store, connect=lambda u: FakeWS(clock, [(D1 + 500, event("A", D1))]),
                   clock=clock, max_connections=1))
    clock.t = D1 + 3 * 3_600_000                     # PC off for ~3 hours
    store2 = L.Store(tmp_path)
    run(L.Recorder(CFG, store2, connect=lambda u: FakeWS(clock, []), clock=clock,
                   max_connections=1))
    gaps = rows(tmp_path / "gaps.csv")
    down = [g for g in gaps if g["reason"].startswith("recorder not running")]
    assert down and int(down[0]["end_ms"]) == D1 + 3 * 3_600_000
    assert float(down[0]["seconds"]) >= 3 * 3600 - 1


def test_failed_connect_is_retried_and_logged(tmp_path):
    clock = Clock(D1)
    attempts = []

    def connect(url):
        attempts.append(1)
        if len(attempts) == 2:
            clock.t += 5000                          # retry succeeds 5 s later
        return FakeWS(clock, [], fail="connect" if len(attempts) == 1 else None)

    rec = L.Recorder(CFG, L.Store(tmp_path), connect=connect, clock=clock, max_connections=2)
    run(rec)
    assert len(attempts) == 2
    gaps = rows(tmp_path / "gaps.csv")
    assert any(g["reason"].startswith("OSError") and float(g["seconds"]) == 5 for g in gaps)


def test_day_rotation_compresses_completed_day(tmp_path):
    clock = Clock(D1)
    script = [(D1 + L.DAY_MS - 1000, event("A", D1)), (D1 + L.DAY_MS + 1000, event("B", D1))]
    run(L.Recorder(CFG, L.Store(tmp_path), connect=lambda u: FakeWS(clock, script),
                   clock=clock, max_connections=1))
    files = sorted(p.name for p in (tmp_path / "liquidations").iterdir())
    assert files == ["2025-10-01.jsonl.gz", "2025-10-02.jsonl"]
    with gzip.open(tmp_path / "liquidations" / "2025-10-01.jsonl.gz", "rt") as fh:
        assert json.loads(fh.readline())["msg"]["o"]["s"] == "A"


def test_sleep_detected_by_heartbeat(tmp_path):
    clock = Clock(D1)
    store = L.Store(tmp_path)
    rec = L.Recorder(CFG, store, clock=clock)

    async def scenario():
        hb = asyncio.create_task(rec._heartbeat_loop())
        await asyncio.sleep(0.03)
        clock.t += 600_000                            # 10 minutes of system sleep
        await asyncio.sleep(0.03)
        rec.stop.set()
        await hb
    asyncio.run(scenario())
    gaps = rows(tmp_path / "gaps.csv")
    assert any(g["reason"].startswith("process paused") and float(g["seconds"]) == 600
               for g in gaps)


def test_coverage_merges_overlapping_gaps_and_counts(tmp_path):
    store = L.Store(tmp_path)
    store.session("START", t=D1)
    store.write_event(D1 + 10, json.loads(event("BTCUSDT", D1)))
    store.write_event(D1 + 20, json.loads(event("ETHUSDT", D1)))
    store.gap(D1 + 3_600_000, D1 + 3 * 3_600_000, "a")
    store.gap(D1 + 2 * 3_600_000, D1 + 4 * 3_600_000, "b")     # overlaps the first
    store.close()
    cov = L.coverage(tmp_path, until_ms=D1 + L.DAY_MS)
    assert len(cov) == 1
    assert cov[0]["uptime"] == pytest.approx(1 - 3 / 24)
    assert cov[0]["events"] == 2 and cov[0]["symbols"] == 2
    assert cov[0]["longest_gap_min"] == pytest.approx(180)


def test_load_events_parses_fields_and_tolerates_torn_line(tmp_path):
    store = L.Store(tmp_path)
    store.write_event(D1, json.loads(event("BTCUSDT", D1, q="2", p="100", ap="101", z="2")))
    store.close()
    with open(tmp_path / "liquidations" / "2025-10-01.jsonl", "a") as fh:
        fh.write('{"recv_ms": 1, "msg": {"o"')                 # crash mid-write
    df = L.load_events(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["notional"] == pytest.approx(202.0)
    assert df.iloc[0]["symbol"] == "BTCUSDT"


def test_array_messages_are_split(tmp_path):
    store = L.Store(tmp_path)
    rec = L.Recorder(CFG, store, clock=Clock(D1))
    rec._handle(json.dumps([json.loads(event("A", D1)), json.loads(event("B", D1))]))
    store.close()
    assert len((tmp_path / "liquidations" / "2025-10-01.jsonl").read_text().splitlines()) == 2


def test_config_loads():
    cfg = L.load_config()
    assert cfg["ws_url"].endswith("!forceOrder@arr") and cfg["gzip_completed_days"]
    assert "/market/" in cfg["ws_url"] or "/public/" in cfg["ws_url"]
    assert cfg["max_silence_minutes"] > 0


def test_silent_connection_is_a_gap_and_reconnects(tmp_path):
    clock = Clock(D1)
    calls = []

    def connect(url):
        calls.append(1)
        if len(calls) == 1:
            return FakeWS(clock, [(D1 + 1000, event("BTCUSDT", D1))], hang=True)
        clock.t = D1 + 20 * 60_000                   # reconnect 20 min later
        return FakeWS(clock, [])

    run(L.Recorder(CFG, L.Store(tmp_path), connect=connect, clock=clock, max_connections=2))
    gaps = rows(tmp_path / "gaps.csv")
    silent = [g for g in gaps if g["reason"].startswith("no data received")]
    assert silent and int(silent[0]["start_ms"]) == D1 + 1000      # from last message
    assert int(silent[0]["end_ms"]) == D1 + 20 * 60_000
    assert len(calls) == 2


def test_data_dir_env_override(tmp_path):
    import os
    cfg = L.load_config()
    old = os.environ.pop("COLLECTOR_DATA_DIR", None)
    try:
        assert L.data_root(cfg) == L.resolve(cfg["data_dir"])
        os.environ["COLLECTOR_DATA_DIR"] = str(tmp_path / "disk")
        assert L.data_root(cfg) == tmp_path / "disk"
    finally:
        os.environ.pop("COLLECTOR_DATA_DIR", None)
        if old is not None:
            os.environ["COLLECTOR_DATA_DIR"] = old
