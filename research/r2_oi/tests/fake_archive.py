"""In-memory stand-in for data.binance.vision (S3 listing + files). No network."""
from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date, datetime, timedelta, timezone

from research.r2_oi import archive as A

H = 3_600_000


def ms(y, m, d, h=0):
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp() * 1000)


def make_zip(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def klines_text(open_times, qv=1000.0, header=True, us=False, price=None):
    lines = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,"
             "taker_buy_volume,taker_buy_quote_volume,ignore"] if header else []
    for t in open_times:
        tt = t * 1000 if us else t
        v = qv(t) if callable(qv) else qv
        if price is None:
            o, hi, lo, c = 100, 101, 99, 100
        else:
            o, c = price(t), price(t + H)
            hi, lo = max(o, c) * 1.002, min(o, c) * 0.998
        lines.append(f"{tt},{o},{hi},{lo},{c},10,{tt + H - 1},{v},5,5,{v / 2},0")
    return "\n".join(lines) + "\n"


def metrics_text(day: date, symbol: str, rows=288, step_min=5, header=True,
                 null_every=0, start_offset_min=5, oi=None):
    lines = ["create_time,symbol,sum_open_interest,sum_open_interest_value,"
             "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
             "count_long_short_ratio,sum_taker_long_short_vol_ratio"] if header else []
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    for i in range(rows):
        t = base + timedelta(minutes=start_offset_min + i * step_min)
        tms = int(t.timestamp() * 1000)
        oiv = 100000.0 if oi is None else oi(tms)
        oi_s = "" if null_every and i % null_every == 0 else "1000.5"
        lines.append(f"{t:%Y-%m-%d %H:%M:%S},{symbol},{oi_s},{oiv},1,1,1,1")
    return "\n".join(lines) + "\n"


class FakeArchive:
    def __init__(self, page_size=1000):
        self.files: dict[str, bytes] = {}
        self.override: dict[str, int] = {}      # key -> HTTP status to return
        self.calls: list[str] = []
        self.page_size = page_size

    def put(self, key: str, text: str, checksum=True, bad_checksum=False, raw=None):
        data = raw if raw is not None else make_zip(key.rsplit("/", 1)[-1][:-4] + ".csv", text)
        self.files[key] = data
        if checksum:
            digest = "0" * 64 if bad_checksum else hashlib.sha256(data).hexdigest()
            self.files[key + ".CHECKSUM"] = f"{digest}  {key.rsplit('/', 1)[-1]}\n".encode()

    def _listing(self, prefix, delimiter, marker):
        keys = sorted(k for k in self.files if k.startswith(prefix))
        prefixes, contents = [], []
        if delimiter:
            seen = set()
            for k in keys:
                rest = k[len(prefix):]
                if delimiter in rest:
                    seen.add(prefix + rest.split(delimiter, 1)[0] + delimiter)
                else:
                    contents.append(k)
            prefixes = sorted(seen)
        else:
            contents = keys
        items = sorted([("p", p) for p in prefixes] + [("k", k) for k in contents],
                       key=lambda x: x[1])
        items = [x for x in items if x[1] > marker]
        page, truncated = items[:self.page_size], len(items) > self.page_size
        ns = "http://s3.amazonaws.com/doc/2006-03-01/"
        parts = [f'<ListBucketResult xmlns="{ns}"><Prefix>{prefix}</Prefix>',
                 f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"]
        if truncated:
            parts.append(f"<NextMarker>{page[-1][1]}</NextMarker>")
        for kind, val in page:
            parts.append(f"<CommonPrefixes><Prefix>{val}</Prefix></CommonPrefixes>"
                         if kind == "p" else f"<Contents><Key>{val}</Key></Contents>")
        parts.append("</ListBucketResult>")
        return "".join(parts).encode()

    def fetch(self, url, params=None):
        self.calls.append(url)
        if url == A.LIST_BASE:
            p = params or {}
            return 200, self._listing(p.get("prefix", ""), p.get("delimiter"),
                                      p.get("marker", "")), "HTTP 200"
        key = url[len(A.FILE_BASE):]
        if key in self.override:
            code = self.override[key]
            return code, b"", f"HTTP {code}"
        if key in self.files:
            return 200, self.files[key], "HTTP 200"
        return 404, b"", "HTTP 404"

    # -- helpers to populate a symbol ------------------------------------
    def add_monthly_klines(self, symbol, first_ms, last_ms_excl, qv=1000.0, skip_months=(),
                           price=None):
        t, by_month = first_ms, {}
        while t < last_ms_excl:
            m = A.ym(t)
            by_month.setdefault(m, []).append(t)
            t += H
        for month, times in by_month.items():
            if month in skip_months:
                continue
            self.put(A.monthly_kline_key(symbol, month), klines_text(times, qv, price=price))

    def add_daily_klines(self, symbol, d: date, qv=1000.0):
        start = ms(d.year, d.month, d.day)
        self.put(A.daily_kline_key(symbol, d), klines_text([start + i * H for i in range(24)], qv))

    def add_metrics(self, symbol, first: date, last_incl: date, **kw):
        d = first
        while d <= last_incl:
            self.put(A.metrics_key(symbol, d), metrics_text(d, symbol, **kw))
            d += timedelta(days=1)
