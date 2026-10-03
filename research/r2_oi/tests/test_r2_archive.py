import pytest

from research.r2_oi import archive as A
from research.r2_oi.tests.fake_archive import (
    FakeArchive, H, klines_text, make_zip, metrics_text, ms)
from datetime import date

KEY = "data/futures/um/daily/metrics/AAAUSDT/AAAUSDT-metrics-2025-03-01.zip"


def file_calls(fake, key):
    return sum(1 for c in fake.calls if c == A.FILE_BASE + key)


def test_download_then_cached_without_refetch(tmp_path):
    fake = FakeArchive()
    fake.put(KEY, metrics_text(date(2025, 3, 1), "AAAUSDT"))
    cache = A.ArchiveCache(tmp_path, fake, workers=1, progress=False)
    r1 = cache.get(KEY)
    assert r1.status == A.VALID and not r1.cached
    r2 = cache.get(KEY)
    assert r2.status == A.VALID and r2.cached
    assert file_calls(fake, KEY) == 1


def test_corrupt_local_file_is_redownloaded(tmp_path):
    fake = FakeArchive()
    fake.put(KEY, metrics_text(date(2025, 3, 1), "AAAUSDT"))
    cache = A.ArchiveCache(tmp_path, fake, workers=1, progress=False)
    cache.get(KEY)
    cache.local_path(KEY).write_bytes(b"garbage")
    r = cache.get(KEY)
    assert r.status == A.VALID and not r.cached
    assert file_calls(fake, KEY) == 2


def test_empty_local_file_is_redownloaded(tmp_path):
    fake = FakeArchive()
    fake.put(KEY, metrics_text(date(2025, 3, 1), "AAAUSDT"))
    cache = A.ArchiveCache(tmp_path, fake, workers=1, progress=False)
    cache.local_path(KEY).parent.mkdir(parents=True)
    cache.local_path(KEY).write_bytes(b"")
    assert cache.get(KEY).status == A.VALID


def test_status_categories(tmp_path):
    fake = FakeArchive()
    fake.put("k/bad_checksum.zip", "a,b\n1,2\n", bad_checksum=True)
    fake.put("k/not_zip.zip", "", raw=b"this is not a zip", checksum=False)
    fake.put("k/empty.zip", "", raw=b"", checksum=False)
    fake.put("k/empty_member.zip", "", raw=make_zip("x.csv", ""), checksum=False)
    fake.put("k/server_error.zip", "a\n1\n")
    fake.override["k/server_error.zip"] = 500
    cache = A.ArchiveCache(tmp_path, fake, workers=1, progress=False)
    assert cache.get("k/missing.zip").status == A.FILE_MISSING
    assert cache.get("k/bad_checksum.zip").status == A.CORRUPT_OR_EMPTY
    assert cache.get("k/not_zip.zip").status == A.CORRUPT_OR_EMPTY
    assert cache.get("k/empty.zip").status == A.CORRUPT_OR_EMPTY
    assert cache.get("k/empty_member.zip").status == A.CORRUPT_OR_EMPTY
    r = cache.get("k/server_error.zip")
    assert r.status == A.DOWNLOAD_FAILED and "500" in r.detail
    assert not cache.local_path("k/bad_checksum.zip").exists()


def test_interrupted_batch_resumes_only_failed_files(tmp_path):
    fake = FakeArchive()
    keys = [f"k/f{i}.zip" for i in range(10)]
    for k in keys:
        fake.put(k, "a,b\n1,2\n")
    for k in keys[6:]:
        fake.override[k] = 503            # simulated outage mid-run
    cache = A.ArchiveCache(tmp_path, fake, workers=3, progress=False)
    first = cache.get_many(keys)
    assert sum(r.status == A.VALID for r in first.values()) == 6
    assert sum(r.status == A.DOWNLOAD_FAILED for r in first.values()) == 4
    fake.override.clear()
    fake.calls.clear()
    second = cache.get_many(keys)
    assert all(r.status == A.VALID for r in second.values())
    refetched = {c[len(A.FILE_BASE):] for c in fake.calls if not c.endswith(".CHECKSUM")}
    assert refetched == set(keys[6:])


def test_listing_pagination(tmp_path):
    fake = FakeArchive(page_size=2)
    for sym in ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]:
        fake.put(A.monthly_kline_key(sym, "2025-01"), "x\n1\n")
    prefixes, _ = A.list_prefix(fake, "data/futures/um/monthly/klines/", "/")
    assert len(prefixes) == 5
    _, keys = A.list_prefix(fake, A.monthly_kline_prefix("AAAUSDT"), None)
    assert A.months_from_keys(keys) == ["2025-01"]


def test_listing_blocked_raises():
    class Blocked:
        def fetch(self, url, params=None):
            return 451, b"", "HTTP 451"
    with pytest.raises(A.ArchiveBlockedError):
        A.list_prefix(Blocked(), "data/", "/")


def test_parse_klines_header_noheader_microseconds():
    times = [ms(2025, 1, 1, h) for h in range(3)]
    for text in (klines_text(times), klines_text(times, header=False),
                 klines_text(times, us=True)):
        df = A.parse_klines_csv(text)
        assert df["open_time"].tolist() == times
        assert df["quote_volume"].tolist() == [1000.0] * 3


def test_parse_metrics_header_and_noheader():
    for header in (True, False):
        df = A.parse_metrics_csv(metrics_text(date(2025, 3, 1), "AAAUSDT", rows=3,
                                              header=header))
        assert df["create_time"].tolist() == [ms(2025, 3, 1) + m * 60_000 for m in (5, 10, 15)]
        assert df["sum_open_interest"].tolist() == [1000.5] * 3


def test_parse_metrics_null_oi():
    df = A.parse_metrics_csv(metrics_text(date(2025, 3, 1), "A", rows=4, null_every=2))
    assert df["sum_open_interest"].isna().sum() == 2


def test_period_start_and_ranges():
    assert A.period_start_ms(A.monthly_kline_key("X", "2025-02")) == ms(2025, 2, 1)
    assert A.period_start_ms(A.daily_kline_key("X", date(2025, 2, 3))) == ms(2025, 2, 3)
    assert A.months_between(ms(2025, 1, 30), ms(2025, 3, 1)) == ["2025-01", "2025-02"]
    assert len(A.days_between(ms(2025, 2, 25), ms(2025, 3, 5))) == 8
