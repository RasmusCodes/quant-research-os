"""Clocks and timestamp helpers.

All registry timestamps are UTC ISO-8601 strings ending in ``Z``. The clock is injectable so tests
and the scripted demo can replay a research timeline deterministically (the demo labels its
timestamps as simulated).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

UTC = dt.timezone.utc


class Clock:
    """Source of 'now' for the registry."""

    simulated = False

    def now(self) -> dt.datetime:  # pragma: no cover - interface
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> dt.datetime:
        return dt.datetime.now(UTC)


class SimulatedClock(Clock):
    """A clock that only moves when told to; every reading advances it by ``tick``.

    The automatic tick keeps event order strictly increasing even when many records are written at
    the same simulated moment.
    """

    simulated = True

    def __init__(self, start: str | dt.datetime, tick_seconds: float = 1.0):
        self._now = parse_iso(start) if isinstance(start, str) else _aware(start)
        self._tick = dt.timedelta(seconds=tick_seconds)

    def now(self) -> dt.datetime:
        current = self._now
        self._now = self._now + self._tick
        return current

    def set(self, when: str | dt.datetime) -> None:
        target = parse_iso(when) if isinstance(when, str) else _aware(when)
        if target < self._now:
            raise ValueError("A simulated clock cannot move backwards")
        self._now = target

    def advance(self, **kwargs: float) -> None:
        self._now = self._now + dt.timedelta(**kwargs)


def _aware(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_iso(value: dt.datetime) -> str:
    value = _aware(value)
    text = value.isoformat(timespec="microseconds" if value.microsecond else "seconds")
    return text.replace("+00:00", "Z")


def parse_iso(text: str) -> dt.datetime:
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    value = dt.datetime.fromisoformat(text)
    return _aware(value)


def date_str(value: Any) -> str | None:
    """Normalise a date-like value to ``YYYY-MM-DD`` (or ``None``)."""
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%d")


def timestamp_str(value: Any) -> str | None:
    """Normalise a timestamp-like value to ISO-8601 (date only when the time is midnight)."""
    if value is None or value == "":
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    if ts == ts.normalize():
        return ts.strftime("%Y-%m-%d")
    return ts.isoformat()
