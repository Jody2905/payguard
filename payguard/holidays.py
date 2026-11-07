"""Public holidays in St. Vincent and the Grenadines, for the off-hours rule."""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _easter(y: int) -> date:
    # Anonymous Gregorian algorithm
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def _first_weekday(y: int, m: int, wd: int, n: int = 1) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(wd - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _monday_on_or_after(d: date) -> date:
    return d + timedelta(days=(0 - d.weekday()) % 7)


@lru_cache(maxsize=None)
def holidays(year: int) -> dict[date, str]:
    e = _easter(year)
    h = {
        date(year, 1, 1): "New Year's Day",
        date(year, 3, 14): "National Heroes' Day",
        e - timedelta(days=2): "Good Friday",
        e + timedelta(days=1): "Easter Monday",
        date(year, 5, 1): "Labour Day",
        e + timedelta(days=50): "Whit Monday",
        # Carnival (Vincy Mas) dates are proclaimed each year; usually the Monday from 7 July on
        _monday_on_or_after(date(year, 7, 7)): "Carnival Monday",
        _monday_on_or_after(date(year, 7, 7)) + timedelta(days=1): "Carnival Tuesday",
        _first_weekday(year, 8, 0, 1): "Emancipation Day",
        date(year, 10, 27): "Independence Day",
        date(year, 12, 25): "Christmas Day",
        date(year, 12, 26): "Boxing Day",
    }
    # A holiday on Sunday is observed on Monday
    for d, name in list(h.items()):
        if d.weekday() == 6:
            h.setdefault(d + timedelta(days=1), name + " (observed)")
    return h


def is_holiday(when) -> str | None:
    d = when.date() if hasattr(when, "date") else when
    return holidays(d.year).get(d)
