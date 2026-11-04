"""SILSO (Royal Observatory of Belgium) daily sunspot number reference and MAE."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime

import httpx

from backend.config import get_settings

log = logging.getLogger(__name__)
_CACHE: dict[str, tuple[float, dict[date, float]]] = {}
_TTL_S = 6 * 3600
_FAIL_TTL_S = 15 * 60


def parse_silso_csv(text: str) -> dict[date, float]:
    """Parse SILSO ``SN_d_tot_V2.0`` CSV (``Y;M;D;frac;SN;std;nobs;def``)."""
    out: dict[date, float] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 5:
            continue
        try:
            d = date(int(parts[0]), int(parts[1]), int(parts[2]))
            sn = float(parts[4])
        except ValueError:
            continue
        if sn >= 0:
            out[d] = sn
    return out


def fetch_silso_daily(url: str | None = None, timeout: float = 20.0) -> dict[date, float]:
    url = url or get_settings().silso_url
    cached = _CACHE.get(url)
    if cached and time.time() - cached[0] < (_TTL_S if cached[1] else _FAIL_TTL_S):
        return cached[1]
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        data = parse_silso_csv(resp.text)
    except httpx.HTTPError as exc:
        log.warning("SILSO fetch failed: %s", exc)
        if cached and cached[1]:
            return cached[1]
        _CACHE[url] = (time.time(), {})
        return {}
    _CACHE[url] = (time.time(), data)
    return data


def rolling_mae(observations: Iterable[tuple[datetime, float]], reference: dict[date, float], window: int | None = None) -> float | None:
    """MAE between daily-mean pipeline R and SILSO over the last ``window`` matched days."""
    window = window or get_settings().mae_window
    daily: dict[date, list[float]] = defaultdict(list)
    for ts, r in observations:
        daily[ts.date()].append(r)
    matched = sorted(d for d in daily if d in reference)[-window:]
    if not matched:
        return None
    errors = [abs(sum(daily[d]) / len(daily[d]) - reference[d]) for d in matched]
    return float(sum(errors) / len(errors))
