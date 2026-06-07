"""Rate-limit correctness for the public Space gate (audit pass).

Loads ``space/safety.py`` by file path (the Space dir is not a package) and
asserts the documented public-deployment guarantees, in particular the
spec'd scenario: six IBM submissions from the same IP, the sixth rejected.

* daily-cap path: with the per-IP window disabled, the 6th of six same-IP
  requests is rejected with reason ``daily_cap`` (cap = 5, the Space default).
* per-IP path: with the real 5-minute window, a same-IP burst is throttled
  after the first (reason ``rate_limited``), and a different IP is unaffected.

No source is modified; ``safety.py`` is loaded under a private module name so
it does not collide with the Space's own loader.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_SAFETY_PATH = Path(__file__).resolve().parent.parent / "space" / "safety.py"


def _load_safety():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("_qv_safety_audit", _SAFETY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_qv_safety_audit"] = module
    spec.loader.exec_module(module)
    return module


_safety = _load_safety()
RateLimiter = _safety.RateLimiter


def _now() -> datetime:
    return datetime(2026, 6, 6, 12, 0, tzinfo=UTC)


def test_six_requests_same_ip_sixth_rejected_by_daily_cap() -> None:
    # window_seconds=0 isolates the global daily cap from the per-IP throttle.
    rl = RateLimiter(window_seconds=0, daily_cap=5)
    now = _now()
    verdicts = [rl.check_and_register(ip="9.9.9.9", now=now) for _ in range(6)]

    assert all(v.allowed for v in verdicts[:5]), "first five should be allowed"
    sixth = verdicts[5]
    assert not sixth.allowed
    assert sixth.reason == "daily_cap"
    assert sixth.daily_remaining == 0


def test_per_ip_window_throttles_burst_but_not_other_ip() -> None:
    rl = RateLimiter(window_seconds=300, daily_cap=100)
    now = _now()

    first = rl.check_and_register(ip="1.1.1.1", now=now)
    assert first.allowed

    # Same IP, 6 rapid follow-ups within the 5-minute window -> all rejected.
    for offset in range(1, 7):
        v = rl.check_and_register(ip="1.1.1.1", now=now + timedelta(seconds=offset))
        assert not v.allowed
        assert v.reason == "rate_limited"

    # A different IP is independent.
    other = rl.check_and_register(ip="2.2.2.2", now=now + timedelta(seconds=2))
    assert other.allowed

    # After the window elapses the original IP is allowed again.
    later = rl.check_and_register(ip="1.1.1.1", now=now + timedelta(seconds=301))
    assert later.allowed


def test_quota_floor_blocks_without_consuming_daily_counter() -> None:
    rl = RateLimiter(window_seconds=0, daily_cap=5, quota_floor_seconds=60)
    now = _now()
    v = rl.check_and_register(ip="3.3.3.3", now=now, quota_remaining_seconds=30)
    assert not v.allowed
    assert v.reason == "quota_exceeded"
    assert v.daily_remaining == 5  # cap not consumed when the quota gate fires


@pytest.mark.parametrize("bad", [{"window_seconds": -1}, {"daily_cap": -1}])
def test_invalid_constructor_args_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        RateLimiter(**bad)
