"""Historical symbol master and point-in-time universe (spec R2.2).

The symbol master comes from the archive listing of USDⓈ-M monthly klines,
which retains delisted contracts. Today's exchangeInfo is never used.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from research.phase05_funding_gate import common as C
from research.r2_oi import archive as A

KLINES_ROOT = "data/futures/um/monthly/klines/"

ELIGIBLE = "ELIGIBLE"
EXCLUDED_NOT_USDT_PERPETUAL = "EXCLUDED_NOT_USDT_PERPETUAL"
EXCLUDED_STABLECOIN_BASE = "EXCLUDED_STABLECOIN_BASE"
EXCLUDED_INDEX_CONTRACT = "EXCLUDED_INDEX_CONTRACT"
NOT_ACTIVE_AT_START = "NOT_ACTIVE_AT_START"
INSUFFICIENT_LISTING_HISTORY = "INSUFFICIENT_LISTING_HISTORY"
INSUFFICIENT_PRESTART_VOLUME_DATA = "INSUFFICIENT_PRESTART_VOLUME_DATA"
ARCHIVE_LISTING_FAILED = "ARCHIVE_LISTING_FAILED"

PRESTART_DAYS = 30


def list_all_symbols(transport) -> list[str]:
    prefixes, _ = A.list_prefix(transport, KLINES_ROOT, "/")
    return sorted({p[len(KLINES_ROOT):].strip("/") for p in prefixes if p.startswith(KLINES_ROOT)})


def static_exclusion(symbol: str, cfg: dict) -> str | None:
    if "_" in symbol or not symbol.endswith("USDT") or symbol == "USDT":
        return EXCLUDED_NOT_USDT_PERPETUAL
    if symbol in set(cfg["stage_a"]["exclude_symbols"]):
        return EXCLUDED_INDEX_CONTRACT
    if symbol[:-4] in set(cfg["exclude_bases"]):
        return EXCLUDED_STABLECOIN_BASE
    return None


def list_months(transport, symbols: list[str], workers: int) -> dict[str, list[str] | Exception]:
    def one(sym):
        try:
            _, keys = A.list_prefix(transport, A.monthly_kline_prefix(sym), None)
            return sym, A.months_from_keys(keys)
        except A.ArchiveBlockedError:
            raise
        except Exception as exc:          # recorded, never silently substituted
            return sym, exc
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(pool.map(one, symbols))


def build_universe(cfg: dict, cache: A.ArchiveCache, transport) -> dict:
    start_ms, _ = C.period_bounds(cfg)
    sa = cfg["stage_a"]
    boundary_ms = start_ms - cfg["min_listing_days"] * C.DAY_MS
    boundary_month = A.ym(boundary_ms)
    start_month = A.ym(start_ms)
    ps_start = start_ms - PRESTART_DAYS * C.DAY_MS
    ps_expected = A.expected_hours(ps_start, start_ms)

    print("Listing archive symbols...")
    symbols = list_all_symbols(transport)
    rows: dict[str, dict] = {}
    to_list = []
    for sym in symbols:
        reason = static_exclusion(sym, cfg)
        rows[sym] = {"symbol": sym, "status": reason or "", "detail": "",
                     "first_month": None, "last_month": None,
                     "prestart_bars": None, "prestart_quote_volume": None,
                     "selected": False, "rank": None}
        if reason is None:
            to_list.append(sym)

    print(f"Listing monthly 1h klines for {len(to_list)} USDT perpetual symbols...")
    months = list_months(transport, to_list, sa["download_workers"])
    listed: dict[str, list[str]] = {}
    for sym in to_list:
        row, m = rows[sym], months[sym]
        if isinstance(m, Exception):
            row.update(status=ARCHIVE_LISTING_FAILED, detail=str(m)[:200])
            continue
        listed[sym] = m
        if not m:
            row.update(status=NOT_ACTIVE_AT_START, detail="no monthly 1h klines")
            continue
        row.update(first_month=m[0], last_month=m[-1])
        if start_month not in m:
            row.update(status=NOT_ACTIVE_AT_START,
                       detail=f"no archive month {start_month}")
        elif m[0] > boundary_month:
            row.update(status=INSUFFICIENT_LISTING_HISTORY,
                       detail=f"first archive month {m[0]} after {boundary_month}")

    # Exact listing check for contracts whose first month is the boundary month.
    boundary_syms = [s for s in listed if not rows[s]["status"]
                     and listed[s][0] == boundary_month]
    if boundary_syms:
        print(f"Checking exact listing time for {len(boundary_syms)} boundary symbols...")
    for sym in boundary_syms:
        y, mo = map(int, boundary_month.split("-"))
        first_day = A.date(y, mo, 1)
        lo = A.date_to_ms(first_day)
        hi = A.date_to_ms((first_day + A.timedelta(days=32)).replace(day=1))
        df, res = A.load_klines(cache, sym, lo, hi, set(listed[sym]),
                                label=f"{sym} listing")
        if df.empty:
            rows[sym].update(status=INSUFFICIENT_LISTING_HISTORY,
                             detail="listing-month file unavailable: "
                                    + ",".join(r.status for r in res))
        elif int(df["open_time"].min()) > boundary_ms:
            rows[sym].update(status=INSUFFICIENT_LISTING_HISTORY,
                             detail=f"first bar {C.ms_to_iso(int(df['open_time'].min()))}")

    candidates = [s for s in listed if not rows[s]["status"]]
    print(f"Loading pre-start volume for {len(candidates)} candidates...")
    for sym in candidates:
        df, res = A.load_klines(cache, sym, ps_start, start_ms, set(listed[sym]),
                                label=f"{sym} pre-start")
        bars = int(df["open_time"].nunique()) if len(df) else 0
        qv_ok = len(df) > 0 and df["quote_volume"].notna().all()
        rows[sym]["prestart_bars"] = bars
        coverage = bars / ps_expected if ps_expected else 0.0
        if coverage < sa["prestart_min_bar_coverage"] - 1e-12 or not qv_ok:
            bad = [f"{r.key.rsplit('/', 1)[-1]}:{r.status}" for r in res if r.status != A.VALID]
            rows[sym].update(
                status=INSUFFICIENT_PRESTART_VOLUME_DATA,
                detail=(f"{bars}/{ps_expected} hourly bars"
                        + ("" if qv_ok else "; quote volume missing")
                        + (f"; files: {', '.join(bad)}" if bad else "")))
            continue
        qv_sum = float(df["quote_volume"].sum())
        if qv_sum <= 0:          # archive placeholder bars: no trading at start
            rows[sym].update(status=NOT_ACTIVE_AT_START,
                             detail="zero traded volume in pre-start window")
            continue
        rows[sym].update(status=ELIGIBLE, prestart_quote_volume=qv_sum)

    eligible = sorted((s for s in rows if rows[s]["status"] == ELIGIBLE),
                      key=lambda s: -rows[s]["prestart_quote_volume"])
    for i, sym in enumerate(eligible, start=1):
        rows[sym]["rank"] = i
        rows[sym]["selected"] = i <= cfg["universe_size"]
    selected = eligible[:cfg["universe_size"]]
    return {
        "start_ms": start_ms,
        "listing_boundary_ms": boundary_ms,
        "prestart_window": [ps_start, start_ms],
        "symbols_in_archive": len(symbols),
        "rows": [rows[s] for s in sorted(rows)],
        "selected": selected,
        "listed_months": {s: listed[s] for s in selected},
    }
