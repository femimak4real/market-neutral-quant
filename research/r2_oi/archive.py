"""Resumable, idempotent access to the data.binance.vision archive (R2 Stage A).

Every file goes through the same path:

    check local cache --valid--> skip
          |missing/corrupt/empty
          v
    download (+ .CHECKSUM) --> verify sha256 + zip --> atomic write --> VALID
          |404 -> FILE_MISSING   |network/HTTP error -> DOWNLOAD_FAILED
          |bad checksum/zip/empty after retry -> CORRUPT_OR_EMPTY

Results are returned per file so they can be recorded. An interrupted run
resumes from the cache: valid files are never downloaded twice.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research.phase05_funding_gate.common import HOUR_MS

FILE_BASE = "https://data.binance.vision/"
LIST_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

VALID = "VALID"
FILE_MISSING = "FILE_MISSING"
DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
CORRUPT_OR_EMPTY = "CORRUPT_OR_EMPTY"
STATUSES = (VALID, FILE_MISSING, DOWNLOAD_FAILED, CORRUPT_OR_EMPTY)

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "count", "taker_buy_volume",
                 "taker_buy_quote_volume", "ignore"]
METRICS_COLUMNS = ["create_time", "symbol", "sum_open_interest",
                   "sum_open_interest_value", "count_toptrader_long_short_ratio",
                   "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                   "sum_taker_long_short_vol_ratio"]


class ArchiveBlockedError(RuntimeError):
    """The archive refused access (HTTP 403/451) at the listing level."""


@dataclass
class FileResult:
    key: str
    status: str
    detail: str = ""
    path: str | None = None
    cached: bool = False


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------

class HttpTransport:
    """Thread-safe GET with retries. Returns (status|None, body, detail)."""

    def __init__(self, retries: int = 4, timeout: float = 60.0, backoff: float = 1.5):
        self.retries, self.timeout, self.backoff = retries, timeout, backoff
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers["User-Agent"] = "r2-oi-stage-a/0.1"
            self._local.session = s
        return self._local.session

    def fetch(self, url: str, params: dict | None = None) -> tuple[int | None, bytes, str]:
        last: tuple[int | None, bytes, str] = (None, b"", "no attempt")
        for attempt in range(self.retries):
            try:
                r = self._session().get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last = (None, b"", f"{exc.__class__.__name__}: {exc}"[:200])
                time.sleep(self.backoff * 2 ** attempt)
                continue
            if r.status_code in (418, 429) or r.status_code >= 500:
                last = (r.status_code, b"", f"HTTP {r.status_code}")
                wait = float(r.headers.get("Retry-After", self.backoff * 2 ** attempt))
                time.sleep(wait)
                continue
            body = r.content if r.status_code == 200 else b""
            return r.status_code, body, f"HTTP {r.status_code}"
        return last


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------

def _tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def parse_listing(xml_bytes: bytes) -> tuple[list[str], list[str], bool, str | None]:
    root = ET.fromstring(xml_bytes)
    prefixes, keys, truncated, next_marker = [], [], False, None
    for el in root:
        tag = _tag(el)
        if tag == "CommonPrefixes":
            for sub in el:
                if _tag(sub) == "Prefix" and sub.text:
                    prefixes.append(sub.text)
        elif tag == "Contents":
            for sub in el:
                if _tag(sub) == "Key" and sub.text:
                    keys.append(sub.text)
        elif tag == "IsTruncated":
            truncated = (el.text or "").strip().lower() == "true"
        elif tag == "NextMarker":
            next_marker = el.text
    return prefixes, keys, truncated, next_marker


def list_prefix(transport, prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[str]]:
    """List an archive prefix, following pagination."""
    all_prefixes, all_keys, marker = [], [], ""
    while True:
        params = {"prefix": prefix, "marker": marker}
        if delimiter:
            params["delimiter"] = delimiter
        status, body, detail = transport.fetch(LIST_BASE, params)
        if status in (403, 451):
            raise ArchiveBlockedError(
                f"Archive listing refused ({detail}) for prefix {prefix}. "
                "data.binance.vision may be blocked from this location.")
        if status != 200:
            raise RuntimeError(f"Archive listing failed for {prefix}: {detail}")
        prefixes, keys, truncated, next_marker = parse_listing(body)
        all_prefixes += prefixes
        all_keys += keys
        if not truncated:
            break
        marker = next_marker or (keys[-1] if keys else prefixes[-1] if prefixes else "")
        if not marker:
            break
    return all_prefixes, all_keys


# --------------------------------------------------------------------------
# Keys and dates
# --------------------------------------------------------------------------

def ms_to_date(ms: int) -> date:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()


def date_to_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def ym(ms: int) -> str:
    d = ms_to_date(ms)
    return f"{d.year:04d}-{d.month:02d}"


def months_between(a_ms: int, b_ms: int) -> list[str]:
    """Months touched by [a_ms, b_ms)."""
    if b_ms <= a_ms:
        return []
    d, last = ms_to_date(a_ms).replace(day=1), ms_to_date(b_ms - 1).replace(day=1)
    out = []
    while d <= last:
        out.append(f"{d.year:04d}-{d.month:02d}")
        d = (d + timedelta(days=32)).replace(day=1)
    return out


def days_between(a_ms: int, b_ms: int) -> list[date]:
    """Calendar days touched by [a_ms, b_ms)."""
    if b_ms <= a_ms:
        return []
    d, last = ms_to_date(a_ms), ms_to_date(b_ms - 1)
    out = []
    while d <= last:
        out.append(d)
        d += timedelta(days=1)
    return out


def metrics_key(symbol: str, d: date) -> str:
    return f"data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{d:%Y-%m-%d}.zip"


def monthly_kline_key(symbol: str, month: str) -> str:
    return f"data/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"


def daily_kline_key(symbol: str, d: date) -> str:
    return f"data/futures/um/daily/klines/{symbol}/1h/{symbol}-1h-{d:%Y-%m-%d}.zip"


def monthly_kline_prefix(symbol: str) -> str:
    return f"data/futures/um/monthly/klines/{symbol}/1h/"


def months_from_keys(keys: list[str]) -> list[str]:
    out = set()
    for k in keys:
        name = k.rsplit("/", 1)[-1]
        if name.endswith(".zip"):
            out.add(name[:-4][-7:])
    return sorted(out)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _zip_ok(data: bytes) -> tuple[bool, str]:
    if not data:
        return False, "empty file"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                return False, "zip has no members"
            content = zf.read(names[0])
            if not content.strip():
                return False, "zip member is empty"
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        return False, f"bad zip: {exc}"[:200]
    return True, ""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_checksum(body: bytes) -> str | None:
    try:
        token = body.decode("utf-8", "replace").split()[0].strip().lower()
    except IndexError:
        return None
    return token if len(token) == 64 else None


class ArchiveCache:
    def __init__(self, root: Path, transport, workers: int = 8, progress: bool = True):
        self.root = Path(root)
        self.transport = transport
        self.workers = max(1, int(workers))
        self.progress = progress

    def local_path(self, key: str) -> Path:
        return self.root / key

    def _write_atomic(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}_{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _local_valid(self, path: Path) -> bool:
        try:
            data = path.read_bytes()
        except OSError:
            return False
        cks = path.with_name(path.name + ".CHECKSUM")
        if cks.exists():
            expected = _parse_checksum(cks.read_bytes())
            if expected and _sha256(data) != expected:
                return False
        return _zip_ok(data)[0]

    def get(self, key: str) -> FileResult:
        path = self.local_path(key)
        if path.exists():
            if self._local_valid(path):
                return FileResult(key, VALID, "cached", str(path), True)
            path.unlink(missing_ok=True)      # corrupt/empty -> redownload

        expected, checksum_body = None, None
        st, body, _ = self.transport.fetch(FILE_BASE + key + ".CHECKSUM")
        if st == 200 and body:
            expected, checksum_body = _parse_checksum(body), body

        why = ""
        for _ in range(2):                    # one immediate retry on corruption
            st, body, detail = self.transport.fetch(FILE_BASE + key)
            if st == 404:
                return FileResult(key, FILE_MISSING, "HTTP 404")
            if st != 200:
                return FileResult(key, DOWNLOAD_FAILED, detail or f"HTTP {st}")
            if not body:
                why = "empty response"
                continue
            if expected and _sha256(body) != expected:
                why = "checksum mismatch"
                continue
            ok, why = _zip_ok(body)
            if not ok:
                continue
            self._write_atomic(path, body)
            if checksum_body:
                self._write_atomic(path.with_name(path.name + ".CHECKSUM"), checksum_body)
            return FileResult(key, VALID, "downloaded", str(path))
        return FileResult(key, CORRUPT_OR_EMPTY, why)

    def get_many(self, keys: list[str], label: str = "") -> dict[str, FileResult]:
        keys = list(dict.fromkeys(keys))
        results: dict[str, FileResult] = {}
        if not keys:
            return results
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for res in pool.map(self.get, keys):
                results[res.key] = res
                done += 1
                if self.progress and (done % 500 == 0 or done == len(keys)):
                    counts = {s: sum(1 for r in results.values() if r.status == s)
                              for s in STATUSES}
                    print(f"  {label} {done}/{len(keys)} "
                          + " ".join(f"{s.lower()}={n}" for s, n in counts.items() if n))
        return results


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------

def read_zip_text(path: str | Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        return zf.read(names[0]).decode("utf-8", "replace")


def _has_header(text: str) -> bool:
    first = text.lstrip().split("\n", 1)[0].split(",")[0].strip().strip('"')
    return not first.lstrip("-").replace(".", "", 1).isdigit()


def _to_ms(series: pd.Series) -> pd.Series:
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return (dt - epoch) // pd.Timedelta(milliseconds=1)


def parse_klines_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), header=None,
                     skiprows=1 if _has_header(text) else 0, names=KLINE_COLUMNS)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    ot = df["open_time"].astype("int64")
    if len(ot) and ot.median() > 1e14:          # microsecond timestamps
        ot = ot // 1000
    df["open_time"] = ot
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]]


def parse_metrics_csv(text: str) -> pd.DataFrame:
    first = text.lstrip().split("\n", 1)[0].lower()
    if first.startswith("create_time") or first.startswith('"create_time'):
        df = pd.read_csv(io.StringIO(text))
    else:
        df = pd.read_csv(io.StringIO(text), header=None, names=METRICS_COLUMNS)
    if "create_time" not in df.columns:
        raise ValueError("metrics file has no create_time column")
    raw = df["create_time"]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().all() and len(numeric):
        ms = numeric.astype("int64")
        ms = ms // 1000 if ms.median() > 1e14 else ms
    else:
        ms = _to_ms(raw)
    out = pd.DataFrame({
        "create_time": ms,
        "sum_open_interest": pd.to_numeric(df.get("sum_open_interest"), errors="coerce"),
        "sum_open_interest_value": pd.to_numeric(df.get("sum_open_interest_value"),
                                                 errors="coerce"),
    })
    return out.dropna(subset=["create_time"]).astype({"create_time": "int64"})


# --------------------------------------------------------------------------
# Kline range loader
# --------------------------------------------------------------------------

def load_klines(cache: ArchiveCache, symbol: str, a_ms: int, b_ms: int,
                listed_months: set[str], daily_fallback_months: set[str] | None = None,
                label: str = "") -> tuple[pd.DataFrame, list[FileResult]]:
    """Load 1h perp klines for [a_ms, b_ms).

    Months present in the archive listing use the monthly file. Unlisted
    months in `daily_fallback_months` (recent months whose monthly file may
    not be published yet) use daily files. Any other unlisted month is
    recorded as FILE_MISSING without a request.
    """
    results: list[FileResult] = []
    monthly_keys, daily_keys = [], []
    for month in months_between(a_ms, b_ms):
        if month in listed_months:
            monthly_keys.append(monthly_kline_key(symbol, month))
        elif daily_fallback_months and month in daily_fallback_months:
            y, m = map(int, month.split("-"))
            first = date(y, m, 1)
            nxt = (first + timedelta(days=32)).replace(day=1)
            lo, hi = max(a_ms, date_to_ms(first)), min(b_ms, date_to_ms(nxt))
            daily_keys += [daily_kline_key(symbol, d) for d in days_between(lo, hi)]
        else:
            results.append(FileResult(monthly_kline_key(symbol, month), FILE_MISSING,
                                      "not in archive listing"))
    fetched = cache.get_many(monthly_keys + daily_keys, label)
    frames = []
    for res in fetched.values():
        if res.status == VALID:
            try:
                frames.append(parse_klines_csv(read_zip_text(res.path)))
            except (ValueError, KeyError, zipfile.BadZipFile) as exc:
                res = FileResult(res.key, CORRUPT_OR_EMPTY, f"unparseable: {exc}"[:200])
        results.append(res)
    if frames:
        df = (pd.concat(frames).drop_duplicates("open_time").sort_values("open_time"))
        df = df[(df["open_time"] >= a_ms) & (df["open_time"] < b_ms)].reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close",
                                   "volume", "quote_volume"])
    return df, results


def period_start_ms(key: str) -> int:
    """Start of the period covered by a kline archive file (monthly or daily)."""
    tag = key.rsplit("/", 1)[-1].rsplit("-1h-", 1)[-1].removesuffix(".zip")
    if len(tag) == 7:
        y, m = map(int, tag.split("-"))
        return date_to_ms(date(y, m, 1))
    return date_to_ms(date.fromisoformat(tag))


def expected_hours(a_ms: int, b_ms: int) -> int:
    return max(0, (b_ms - a_ms) // HOUR_MS)


def status_counts(results: list[FileResult]) -> dict[str, int]:
    return {s: sum(1 for r in results if r.status == s) for s in STATUSES}
