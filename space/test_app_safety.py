"""Unit tests for the IBM hardware safety gate."""

from __future__ import annotations

# space/ is not a package; load safety.py by file path.
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_qv_safety", _HERE / "safety.py")
assert spec is not None and spec.loader is not None
_safety = importlib.util.module_from_spec(spec)
sys.modules["_qv_safety"] = _safety
spec.loader.exec_module(_safety)

RateLimiter = _safety.RateLimiter


def _utc(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_rate_limit_blocks_second_call_within_5min() -> None:
    rl = RateLimiter(window_seconds=300, daily_cap=10)
    t0 = _utc(2026, 5, 2, 10, 0)
    v1 = rl.check_and_register(ip="1.1.1.1", now=t0)
    assert v1.allowed
    v2 = rl.check_and_register(ip="1.1.1.1", now=t0 + timedelta(seconds=120))
    assert not v2.allowed
    assert v2.reason == "rate_limited"


def test_rate_limit_allows_after_5min() -> None:
    rl = RateLimiter(window_seconds=300, daily_cap=10)
    t0 = _utc(2026, 5, 2, 10, 0)
    rl.check_and_register(ip="1.1.1.1", now=t0)
    v2 = rl.check_and_register(ip="1.1.1.1", now=t0 + timedelta(seconds=301))
    assert v2.allowed
    assert v2.reason is None


def test_rate_limit_independent_per_ip() -> None:
    rl = RateLimiter(window_seconds=300, daily_cap=10)
    t0 = _utc(2026, 5, 2, 10, 0)
    v1 = rl.check_and_register(ip="1.1.1.1", now=t0)
    v2 = rl.check_and_register(ip="2.2.2.2", now=t0 + timedelta(seconds=10))
    assert v1.allowed
    assert v2.allowed


def test_daily_cap_at_5_blocks_sixth() -> None:
    rl = RateLimiter(window_seconds=0, daily_cap=5)
    t0 = _utc(2026, 5, 2, 10, 0)
    for i in range(5):
        v = rl.check_and_register(ip=f"{i}.0.0.0", now=t0)
        assert v.allowed, f"call {i} should pass"
        assert v.daily_remaining == 5 - i - 1
    v6 = rl.check_and_register(ip="x.0.0.0", now=t0)
    assert not v6.allowed
    assert v6.reason == "daily_cap"
    assert v6.daily_remaining == 0


def test_daily_cap_resets_at_midnight_utc() -> None:
    rl = RateLimiter(window_seconds=0, daily_cap=2)
    t0 = _utc(2026, 5, 2, 23, 30)
    rl.check_and_register(ip="a", now=t0)
    rl.check_and_register(ip="b", now=t0)
    blocked = rl.check_and_register(ip="c", now=t0)
    assert not blocked.allowed
    next_day = _utc(2026, 5, 3, 0, 1)
    after_reset = rl.check_and_register(ip="c", now=next_day)
    assert after_reset.allowed
    assert after_reset.daily_remaining == 1


def test_quota_under_60s_disables_button() -> None:
    rl = RateLimiter(window_seconds=0, daily_cap=10, quota_floor_seconds=60)
    t0 = _utc(2026, 5, 2, 10, 0)
    v = rl.check_and_register(ip="x", now=t0, quota_remaining_seconds=42)
    assert not v.allowed
    assert v.reason == "quota_exceeded"
    # Daily counter is NOT consumed when the quota gate fires.
    assert v.daily_remaining == 10


def test_quota_over_floor_allows_call() -> None:
    rl = RateLimiter(window_seconds=0, daily_cap=10, quota_floor_seconds=60)
    t0 = _utc(2026, 5, 2, 10, 0)
    v = rl.check_and_register(ip="x", now=t0, quota_remaining_seconds=600)
    assert v.allowed


def test_persistence_round_trips_count_across_instances(tmp_path: Path) -> None:
    persist = tmp_path / "quota.json"
    rl1 = RateLimiter(window_seconds=0, daily_cap=5, persist_path=persist)
    t0 = _utc(2026, 5, 2, 10, 0)
    rl1.check_and_register(ip="a", now=t0)
    rl1.check_and_register(ip="b", now=t0)
    raw = json.loads(persist.read_text(encoding="utf-8"))
    assert raw == {"date": "2026-05-02", "count": 2}

    # Fresh instance picks up the same counter.
    rl2 = RateLimiter(window_seconds=0, daily_cap=5, persist_path=persist)
    assert rl2.daily_remaining(now=t0) == 3


def test_persistence_resets_on_new_day(tmp_path: Path) -> None:
    persist = tmp_path / "quota.json"
    persist.write_text(
        json.dumps({"date": "2026-05-01", "count": 4}),
        encoding="utf-8",
    )
    rl = RateLimiter(window_seconds=0, daily_cap=5, persist_path=persist)
    today = _utc(2026, 5, 2, 10, 0)
    v = rl.check_and_register(ip="a", now=today)
    assert v.allowed
    # Counter rolled over: only 1 of today's calls used.
    assert v.daily_remaining == 4


def test_invalid_constructor_args_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        RateLimiter(window_seconds=-1)
    with pytest.raises(ValueError):
        RateLimiter(daily_cap=-1)
