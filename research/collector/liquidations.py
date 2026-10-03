"""Liquidation recorder for the Binance USDⓈ-M all-market forceOrder stream.

Records only: no API keys, no orders. See docs/collector_liquidations_spec.md.

Usage (from repo root):
    python -m research.collector.liquidations run       # record (Ctrl+C to stop)
    python -m research.collector.liquidations report    # coverage per UTC day
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from research.phase05_funding_gate.common import REPO_ROOT

DAY_MS = 86_400_000
DEFAULT_CONFIG = REPO_ROOT / "config" / "collector.yaml"


def now_ms() -> int:
    return int(time.time() * 1000)


def day_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)["collector"]
    for key in ("ws_url", "data_dir", "heartbeat_seconds", "gap_threshold_seconds",
                "gzip_completed_days", "max_silence_minutes"):
        if key not in cfg:
            raise ValueError(f"collector config missing '{key}'")
    return cfg


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def data_root(cfg: dict) -> Path:
    """COLLECTOR_DATA_DIR (e.g. a Render persistent disk) overrides the config."""
    override = os.environ.get("COLLECTOR_DATA_DIR", "").strip()
    return Path(override) if override else resolve(cfg["data_dir"])


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

class Store:
    def __init__(self, root: Path, gzip_completed: bool = True):
        self.root = Path(root)
        self.liq_dir = self.root / "liquidations"
        self.liq_dir.mkdir(parents=True, exist_ok=True)
        self.gaps_path = self.root / "gaps.csv"
        self.sessions_path = self.root / "sessions.csv"
        self.hb_path = self.root / "heartbeat.json"
        self.gzip_completed = gzip_completed
        self._day: str | None = None
        self._fh = None
        self.events_written = 0

    # events ---------------------------------------------------------------
    def write_event(self, recv_ms: int, msg) -> None:
        d = day_str(recv_ms)
        if d != self._day:
            self._rotate(d)
        self._fh.write(json.dumps({"recv_ms": recv_ms, "msg": msg},
                                  separators=(",", ":")) + "\n")
        self._fh.flush()
        self.events_written += 1

    def _rotate(self, d: str) -> None:
        old = self._day
        if self._fh:
            self._fh.close()
        self._day = d
        self._fh = open(self.liq_dir / f"{d}.jsonl", "a", encoding="utf-8")
        if old and self.gzip_completed:
            self.compress_day(old)

    def compress_day(self, d: str) -> None:
        src = self.liq_dir / f"{d}.jsonl"
        dst = self.liq_dir / f"{d}.jsonl.gz"
        if not src.exists() or src == (self.liq_dir / f"{self._day}.jsonl"):
            return
        if dst.exists():                      # never overwrite: keep both
            dst = self.liq_dir / f"{d}.part{int(time.time())}.jsonl.gz"
        tmp = dst.with_name(dst.name + ".tmp")
        with open(src, "rb") as fi, gzip.open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.replace(tmp, dst)
        src.unlink()

    def compress_past_days(self, today: str) -> None:
        if not self.gzip_completed:
            return
        for p in sorted(self.liq_dir.glob("*.jsonl")):
            d = p.name[:10]
            if d < today:
                self.compress_day(d)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    # logs -----------------------------------------------------------------
    def _append_csv(self, path: Path, header: list[str], row: list) -> None:
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header)
            w.writerow(row)

    def gap(self, start_ms: int, end_ms: int, reason: str) -> None:
        if end_ms <= start_ms:
            return
        self._append_csv(self.gaps_path, ["start_ms", "end_ms", "seconds", "reason"],
                         [start_ms, end_ms, round((end_ms - start_ms) / 1000, 1), reason])

    def session(self, event: str, detail: str = "", t: int | None = None) -> None:
        t = t or now_ms()
        self._append_csv(self.sessions_path, ["time_ms", "time_utc", "event", "detail"],
                         [t, iso(t), event, detail])

    def heartbeat(self, t: int, connected: bool) -> None:
        tmp = self.hb_path.with_name("heartbeat.json.tmp")
        tmp.write_text(json.dumps({"last_alive_ms": t, "last_alive_utc": iso(t),
                                   "connected": connected,
                                   "events_this_session": self.events_written}),
                       encoding="utf-8")
        os.replace(tmp, self.hb_path)

    def last_heartbeat(self) -> int | None:
        try:
            return int(json.loads(self.hb_path.read_text(encoding="utf-8"))["last_alive_ms"])
        except (OSError, ValueError, KeyError):
            return None


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------

def default_connect(url: str):
    import websockets  # imported lazily so tests run without it
    return websockets.connect(url, open_timeout=20, ping_interval=20, ping_timeout=20,
                              max_queue=4096)


class Recorder:
    def __init__(self, cfg: dict, store: Store, connect=None, clock=now_ms,
                 max_connections: int | None = None, verbose: bool = False):
        self.cfg, self.store = cfg, store
        self.connect = connect or default_connect
        self.clock = clock
        self.max_connections = max_connections        # tests only
        self.connected = False
        self.stop = asyncio.Event()
        self.verbose = verbose

    def _say(self, text: str) -> None:
        if self.verbose:
            print(f"[{iso(self.clock())}] {text}", flush=True)

    def _handle(self, raw) -> None:
        recv = self.clock()
        msg = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        for item in (msg if isinstance(msg, list) else [msg]):
            self.store.write_event(recv, item)

    async def _heartbeat_loop(self) -> None:
        interval = self.cfg["heartbeat_seconds"]
        limit = self.cfg["gap_threshold_seconds"] * 1000
        last = self.clock()
        ticks = 0
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            t = self.clock()
            if t - last > limit:
                self.store.gap(last, t, "process paused (system sleep or freeze)")
            self.store.heartbeat(t, self.connected)
            last = t
            ticks += 1
            if ticks % 10 == 0:
                self._say(f"{'connected' if self.connected else 'NOT connected'}, "
                          f"{self.store.events_written} events this session")

    async def run(self) -> None:
        start = self.clock()
        self.store.compress_past_days(day_str(start))
        last_alive = self.store.last_heartbeat()
        if last_alive is not None:
            # From the last proof of life to now: conservative (may overcount by
            # up to one heartbeat interval after a crash).
            self.store.gap(last_alive, start, "recorder not running (PC off, asleep, or stopped)")
        self.store.session("START", f"url={self.cfg['ws_url']}", start)
        self.store.heartbeat(start, False)
        hb = asyncio.create_task(self._heartbeat_loop())
        down_since, reason, backoff, n = start, "startup", 1.0, 0
        try:
            while not self.stop.is_set():
                if self.max_connections is not None and n >= self.max_connections:
                    break
                n += 1
                try:
                    silent = False
                    async with self.connect(self.cfg["ws_url"]) as ws:
                        up = self.clock()
                        self.store.gap(down_since, up, reason)
                        self.store.session("CONNECTED", "", up)
                        self._say("connected to Binance liquidation stream")
                        self.connected, backoff = True, 1.0
                        last_msg = up
                        limit_s = self.cfg["max_silence_minutes"] * 60
                        it = ws.__aiter__()
                        while not self.stop.is_set():
                            try:
                                raw = await asyncio.wait_for(it.__anext__(), timeout=limit_s)
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                silent = True      # connected but no data: not coverage
                                break
                            self._handle(raw)
                            last_msg = self.clock()
                    reason = ("no data received (silent connection)" if silent
                              else "server closed connection")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:              # network errors -> reconnect
                    reason = f"{exc.__class__.__name__}: {exc}"[:200]
                    if not self.connected:
                        self._say(f"connection failed ({reason}); retrying in {backoff:.0f}s")
                if self.connected:
                    down_since = self.clock()
                    if reason.startswith("no data received"):
                        down_since = last_msg      # the silent period is a gap
                    self.store.session("DISCONNECTED", reason, self.clock())
                    self._say(f"disconnected ({reason}); reconnecting")
                self.connected = False
                if self.stop.is_set():
                    break
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60.0)
        finally:
            end = self.clock()
            if not self.connected:
                self.store.gap(down_since, end, reason)
            self.store.session("STOP", f"events this session: {self.store.events_written}", end)
            self.store.heartbeat(end, False)
            self.stop.set()
            await hb
            self.store.close()


# --------------------------------------------------------------------------
# Reading and coverage
# --------------------------------------------------------------------------

def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def read_gaps(root: Path) -> list[tuple[int, int, str]]:
    p = Path(root) / "gaps.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return [(int(r["start_ms"]), int(r["end_ms"]), r["reason"]) for r in csv.DictReader(fh)]


def first_start(root: Path) -> int | None:
    p = Path(root) / "sessions.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as fh:
        starts = [int(r["time_ms"]) for r in csv.DictReader(fh) if r["event"] == "START"]
    return min(starts) if starts else None


def iter_events(root: Path):
    for p in sorted((Path(root) / "liquidations").glob("*.jsonl*")):
        if p.name.endswith(".tmp"):
            continue
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue          # a torn final line after a hard crash


def load_events(root: Path):
    """Parsed events as a DataFrame (lower-bound data; see spec)."""
    import pandas as pd
    rows = []
    for rec in iter_events(root):
        o = rec["msg"].get("o", {})
        q, p, ap, z = (float(o.get(k, 0) or 0) for k in ("q", "p", "ap", "z"))
        rows.append({"recv_ms": rec["recv_ms"], "event_ms": rec["msg"].get("E"),
                     "trade_ms": o.get("T"), "symbol": o.get("s"), "side": o.get("S"),
                     "status": o.get("X"), "qty": q, "price": p, "avg_price": ap,
                     "filled_qty": z, "notional": (ap * z) if ap and z else p * q})
    return pd.DataFrame(rows)


def coverage(root: Path, until_ms: int | None = None) -> list[dict]:
    root = Path(root)
    begin = first_start(root)
    if begin is None:
        return []
    until = until_ms or now_ms()
    gaps = merge([(a, b) for a, b, _ in read_gaps(root)])
    counts: dict[str, int] = {}
    symbols: dict[str, set] = {}
    for rec in iter_events(root):
        d = day_str(rec["recv_ms"])
        counts[d] = counts.get(d, 0) + 1
        symbols.setdefault(d, set()).add(rec["msg"].get("o", {}).get("s"))
    out = []
    day0 = int(datetime.strptime(day_str(begin), "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)
    d = day0
    while d < until:
        lo, hi = max(d, begin), min(d + DAY_MS, until)
        if hi > lo:
            in_day = [(max(a, lo), min(b, hi)) for a, b in gaps if b > lo and a < hi]
            gap_ms = sum(b - a for a, b in in_day)
            out.append({"day": day_str(d), "measured_hours": (hi - lo) / 3_600_000,
                        "uptime": 1 - gap_ms / (hi - lo), "gaps": len(in_day),
                        "longest_gap_min": max((b - a for a, b in in_day), default=0) / 60_000,
                        "events": counts.get(day_str(d), 0),
                        "symbols": len(symbols.get(day_str(d), ()))})
        d += DAY_MS
    return out


def print_report(root: Path) -> None:
    rows = coverage(root)
    if not rows:
        print("No recorder sessions found yet.")
        return
    print(f"{'Day (UTC)':<11} {'Hours':>6} {'Uptime':>8} {'Gaps':>5} {'Longest':>9} "
          f"{'Events':>7} {'Symbols':>8}")
    for r in rows:
        print(f"{r['day']:<11} {r['measured_hours']:>6.1f} {r['uptime'] * 100:>7.1f}% "
              f"{r['gaps']:>5} {r['longest_gap_min']:>7.1f}m {r['events']:>7} {r['symbols']:>8}")
    print("\nCounts are lower bounds: Binance pushes at most one liquidation per symbol "
          "per second.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Binance liquidation recorder (records only)")
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = data_root(cfg)
    if args.command == "report":
        print_report(root)
        return 0
    store = Store(root, cfg["gzip_completed_days"])
    print(f"Recording liquidations to {root / 'liquidations'} (Ctrl+C to stop)")
    try:
        asyncio.run(Recorder(cfg, store, verbose=True).run())
    except KeyboardInterrupt:
        pass
    print(f"Stopped. Events this session: {store.events_written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
