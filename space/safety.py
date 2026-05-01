"""Rate-limit, daily-cap, and quota-gating helpers for the IBM hardware path.

The Space is exposed publicly. Without protection, a script could burn
through the project's IBM Quantum free-tier budget (~10 minutes / month)
in seconds. Three layers of defence:

1. **Per-IP rate limit** — at most one IBM submission per ``window_seconds``
   per visitor IP. The IP is read from ``x-forwarded-for`` (set by HF's
   reverse proxy); the client_host fallback covers local dev.
2. **Global daily cap** — at most ``daily_cap`` IBM submissions per UTC
   day across all visitors. Persisted to a JSON file when a writable
   directory is configured (HF Persistent Storage at ``/data`` when
   enabled), in-memory otherwise.
3. **Quota guard** — when the IBM monthly remaining seconds drops below
   ``quota_floor_seconds``, every submission is blocked regardless of
   the per-IP / daily counters. The remaining-seconds value is supplied
   by the caller (the Space module probes ``service.usage()`` once an
   hour and caches the result).

Verdicts are returned as a dataclass; the Space module turns the verdict
into a Gradio output payload. Keeping the logic separate from Gradio
makes the unit tests trivial.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

SafetyReason = Literal["rate_limited", "daily_cap", "quota_exceeded"]


@dataclass(frozen=True)
class SafetyVerdict:
    """Outcome of a :meth:`RateLimiter.check_and_register` call."""

    allowed: bool
    reason: SafetyReason | None
    detail: str
    daily_remaining: int
    daily_cap: int


class RateLimiter:
    """Per-IP + daily + quota gate for the IBM hardware path.

    Thread-safe via a single lock; the contention is negligible because
    each submission is sub-millisecond.
    """

    def __init__(
        self,
        *,
        window_seconds: int = 300,
        daily_cap: int = 5,
        quota_floor_seconds: int = 60,
        persist_path: Path | None = None,
    ) -> None:
        if window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")
        if daily_cap < 0:
            raise ValueError("daily_cap must be >= 0")
        self._window = window_seconds
        self._cap = daily_cap
        self._quota_floor = quota_floor_seconds
        self._persist_path = persist_path
        self._lock = threading.Lock()

        # Per-IP last-allowed timestamp (epoch seconds).
        self._last_ip: dict[str, float] = {}

        # Daily counter — loaded lazily.
        self._day: date | None = None
        self._count: int = 0
        self._load_persisted()

    # -- public API -------------------------------------------------------

    def check_and_register(
        self,
        *,
        ip: str,
        now: datetime,
        quota_remaining_seconds: float | None = None,
    ) -> SafetyVerdict:
        """Decide whether ``ip`` may submit an IBM job at ``now``.

        The function is **commit-on-allow**: a successful return updates
        the per-IP timestamp and bumps the daily counter. Callers MUST
        proceed with the IBM submission when ``allowed=True``.
        """
        with self._lock:
            self._roll_day_if_needed(now)

            if quota_remaining_seconds is not None and quota_remaining_seconds < self._quota_floor:
                return SafetyVerdict(
                    allowed=False,
                    reason="quota_exceeded",
                    detail=(
                        f"IBM monthly quota is exhausted: "
                        f"{quota_remaining_seconds:.0f}s remaining "
                        f"(floor={self._quota_floor}s). Resets on the 1st of next month UTC."
                    ),
                    daily_remaining=max(0, self._cap - self._count),
                    daily_cap=self._cap,
                )

            if self._count >= self._cap:
                return SafetyVerdict(
                    allowed=False,
                    reason="daily_cap",
                    detail=(
                        f"Daily limit of {self._cap} IBM runs reached. Resets at midnight UTC."
                    ),
                    daily_remaining=0,
                    daily_cap=self._cap,
                )

            last = self._last_ip.get(ip)
            if last is not None and (now.timestamp() - last) < self._window:
                wait = self._window - int(now.timestamp() - last)
                return SafetyVerdict(
                    allowed=False,
                    reason="rate_limited",
                    detail=(
                        f"Per-IP rate limit: wait {wait}s before submitting "
                        "another IBM job (5 minutes between runs per visitor)."
                    ),
                    daily_remaining=max(0, self._cap - self._count),
                    daily_cap=self._cap,
                )

            # Allow + commit.
            self._last_ip[ip] = now.timestamp()
            self._count += 1
            self._persist()
            return SafetyVerdict(
                allowed=True,
                reason=None,
                detail="ok",
                daily_remaining=max(0, self._cap - self._count),
                daily_cap=self._cap,
            )

    def daily_remaining(self, now: datetime) -> int:
        """Read-only counter inspector for the UI badge."""
        with self._lock:
            self._roll_day_if_needed(now)
            return max(0, self._cap - self._count)

    def daily_cap(self) -> int:
        return self._cap

    # -- internal ---------------------------------------------------------

    def _roll_day_if_needed(self, now: datetime) -> None:
        today = now.astimezone(UTC).date()
        if self._day != today:
            self._day = today
            self._count = 0
            # Stale per-IP timestamps fade naturally: window_seconds is
            # short relative to a day, so we leave them alone.

    def _load_persisted(self) -> None:
        if self._persist_path is None:
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        try:
            self._day = date.fromisoformat(str(raw.get("date")))
            self._count = int(raw.get("count", 0))
        except (TypeError, ValueError):
            self._day = None
            self._count = 0

    def _persist(self) -> None:
        if self._persist_path is None or self._day is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps({"date": self._day.isoformat(), "count": self._count}),
                encoding="utf-8",
            )
        except OSError:
            # Persistent storage unavailable; degrade silently to in-memory.
            self._persist_path = None


def default_persist_path() -> Path | None:
    """Return ``/data/qverify_quota.json`` when HF Persistent Storage is
    mounted; ``None`` otherwise (in-memory only)."""
    candidate = Path("/data")
    if candidate.is_dir():
        try:
            test = candidate / ".qv_write_test"
            test.write_text("x")
            test.unlink()
        except OSError:
            return None
        return candidate / "qverify_quota.json"
    return None
