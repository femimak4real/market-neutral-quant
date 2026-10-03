"""Download Phase 0.5 data from Binance PUBLIC REST endpoints (spec 0.5.2).

No API keys. No order placement. Read-only market data only.

Usage (from repo root):
    python -m research.phase05_funding_gate.download_data
    python -m research.phase05_funding_gate.download_data --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import requests

from research.phase05_funding_gate import common as C

FAPI_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"
FUNDING_PATH = "/fapi/v1/fundingRate"
PERP_KLINES_PATH = "/fapi/v1/klines"
SPOT_KLINES_PATH = "/api/v3/klines"

FUNDING_LIMIT = 1000
PERP_KLINE_LIMIT = 1500
SPOT_KLINE_LIMIT = 1000
REQUEST_PAUSE_S = 0.25
MAX_RETRIES = 6

FUNDING_LOOKBACK_MS = 7 * C.DAY_MS    # history before start for the K lookback
KLINE_LOOKBACK_MS = 1 * C.DAY_MS
KLINE_TAIL_MS = 2 * C.DAY_MS          # bars after end for exit fills

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]


class RegionBlockedError(RuntimeError):
    """Binance refused the request based on location (HTTP 451/403)."""


def _get(session: requests.Session, url: str, params: dict):
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  network error ({exc.__class__.__name__}); retrying in {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code in (451, 403):
            raise RegionBlockedError(
                f"HTTP {resp.status_code} from {url}. Binance is blocking requests "
                "from this location. Run from a VPS in an allowed region, or use "
                "the public archive at https://data.binance.vision instead."
            )
        if resp.status_code in (418, 429):
            wait = int(resp.headers.get("Retry-After", 5 * 2 ** attempt))
            print(f"  rate limited (HTTP {resp.status_code}); waiting {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"  server error {resp.status_code}; retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected response from {url}: {str(data)[:200]}")
        return data
    raise RuntimeError(f"Giving up on {url} after {MAX_RETRIES} attempts")


def fetch_funding(session: requests.Session, symbol: str,
                  start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get(session, FAPI_BASE + FUNDING_PATH, {
            "symbol": symbol, "startTime": cursor, "endTime": end_ms - 1,
            "limit": FUNDING_LIMIT,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < FUNDING_LIMIT:
            break
        time.sleep(REQUEST_PAUSE_S)

    df = pd.DataFrame({
        "funding_time": [int(r["fundingTime"]) for r in rows],
        "funding_rate": [float(r["fundingRate"]) for r in rows],
        # markPrice may be absent or "" on older records -> NaN (fallback later)
        "mark_price": pd.to_numeric([r.get("markPrice", "") for r in rows],
                                    errors="coerce"),
    })
    return (df.drop_duplicates("funding_time")
              .sort_values("funding_time")
              .reset_index(drop=True))


def fetch_klines(session: requests.Session, base: str, path: str, symbol: str,
                 start_ms: int, end_ms: int, limit: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get(session, base + path, {
            "symbol": symbol, "interval": "1h", "startTime": cursor,
            "endTime": end_ms - 1, "limit": limit,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + C.HOUR_MS
        if len(batch) < limit:
            break
        time.sleep(REQUEST_PAUSE_S)

    df = pd.DataFrame([r[:6] for r in rows], columns=KLINE_COLUMNS)
    df["open_time"] = df["open_time"].astype("int64")
    for col in KLINE_COLUMNS[1:]:
        df[col] = df[col].astype(float)
    return (df.drop_duplicates("open_time")
              .sort_values("open_time")
              .reset_index(drop=True))


def raw_paths(data_dir: Path, symbol: str) -> dict[str, Path]:
    return {
        "funding": data_dir / f"{symbol}_funding.csv",
        "perp": data_dir / f"{symbol}_perp_1h.csv",
        "spot": data_dir / f"{symbol}_spot_1h.csv",
        "meta": data_dir / f"{symbol}_meta.json",
    }


def _cached_covers(paths: dict[str, Path], start_ms: int, end_ms: int) -> bool:
    if not all(p.exists() for p in paths.values()):
        return False
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return meta.get("start_ms", 1 << 62) <= start_ms and meta.get("end_ms", 0) >= end_ms


def download_symbol(session: requests.Session, symbol: str, data_dir: Path,
                    start_ms: int, end_ms: int, force: bool = False) -> None:
    paths = raw_paths(data_dir, symbol)
    if not force and _cached_covers(paths, start_ms, end_ms):
        print(f"{symbol}: cached, skipping (use --force to re-download)")
        return

    now_ms = int(time.time() * 1000)
    kline_end = min(end_ms + KLINE_TAIL_MS, now_ms)
    funding_end = min(end_ms, now_ms)

    print(f"{symbol}: funding history")
    funding = fetch_funding(session, symbol, start_ms - FUNDING_LOOKBACK_MS, funding_end)
    print(f"{symbol}: perp 1h klines")
    perp = fetch_klines(session, FAPI_BASE, PERP_KLINES_PATH, symbol,
                        start_ms - KLINE_LOOKBACK_MS, kline_end, PERP_KLINE_LIMIT)
    print(f"{symbol}: spot 1h klines")
    spot = fetch_klines(session, SPOT_BASE, SPOT_KLINES_PATH, symbol,
                        start_ms - KLINE_LOOKBACK_MS, kline_end, SPOT_KLINE_LIMIT)

    funding.to_csv(paths["funding"], index=False)
    perp.to_csv(paths["perp"], index=False)
    spot.to_csv(paths["spot"], index=False)
    missing_mark = int(funding["mark_price"].isna().sum())
    paths["meta"].write_text(json.dumps({
        "symbol": symbol, "start_ms": start_ms, "end_ms": end_ms,
        "downloaded_utc": C.ms_to_iso(now_ms),
        "funding_records": len(funding), "funding_missing_mark_price": missing_mark,
        "perp_bars": len(perp), "spot_bars": len(spot),
    }, indent=2), encoding="utf-8")
    print(f"{symbol}: {len(funding)} funding records ({missing_mark} without markPrice), "
          f"{len(perp)} perp bars, {len(spot)} spot bars")


def download_all(cfg: dict, data_dir: Path, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = C.period_bounds(cfg)
    with requests.Session() as session:
        session.headers["User-Agent"] = "phase05-funding-gate/1.0"
        for symbol in cfg["symbols"]:
            download_symbol(session, symbol, data_dir, start_ms, end_ms, force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to phase05.yaml")
    parser.add_argument("--force", action="store_true", help="ignore cached files")
    args = parser.parse_args(argv)
    try:
        cfg = C.load_config(args.config)
        download_all(cfg, C.resolve_dir(cfg, "data_dir", "data/phase05"), args.force)
    except (C.ConfigError, RegionBlockedError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print("Download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
