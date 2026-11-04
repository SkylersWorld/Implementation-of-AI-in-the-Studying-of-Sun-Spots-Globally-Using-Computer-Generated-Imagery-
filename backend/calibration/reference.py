"""International daily sunspot number from manual entries, SILSO and NOAA SWPC.

Remote series are cached in the ``reference_sn`` table so comparisons and MAE keep
working offline. ``source="auto"`` resolves each date as manual > silso > swpc.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.calibration.silso import parse_silso_csv
from backend.config import get_settings
from backend.database import ReferenceSN

log = logging.getLogger(__name__)

REMOTE_SOURCES = ("silso", "swpc")
SOURCES = ("manual",) + REMOTE_SOURCES
PRIORITY = {"manual": 0, "silso": 1, "swpc": 2}
SYNC_TTL_S = 6 * 3600
FAIL_TTL_S = 15 * 60
SYNC_SINCE = date(1990, 1, 1)
_last_sync: dict[str, tuple[float, bool]] = {}


def parse_swpc_json(data: list[dict]) -> dict[date, float]:
    """Parse SWPC ``swpc_observed_ssn.json`` (``[{"Obsdate": ISO, "swpc_ssn": n}, ...]``)."""
    out: dict[date, float] = {}
    for row in data:
        try:
            d = date.fromisoformat(str(row["Obsdate"])[:10])
            v = float(row["swpc_ssn"])
        except (KeyError, TypeError, ValueError):
            continue
        if v >= 0:
            out[d] = v
    return out


def fetch_remote(source: str, timeout: float = 20.0) -> dict[date, float]:
    s = get_settings()
    url = {"silso": s.silso_url, "swpc": s.swpc_url}[source]
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return parse_silso_csv(resp.text) if source == "silso" else parse_swpc_json(resp.json())


def _upsert(db: Session, source: str, values: dict[date, float]) -> int:
    existing = {r.date: r for r in db.scalars(select(ReferenceSN).where(ReferenceSN.source == source))}
    changed = 0
    for d, v in values.items():
        if d < SYNC_SINCE:
            continue
        row = existing.get(d)
        if row is None:
            db.add(ReferenceSN(date=d, source=source, value=v))
            changed += 1
        elif row.value != v:
            row.value = v
            changed += 1
    db.commit()
    return changed


def sync(db: Session, sources: tuple[str, ...] = REMOTE_SOURCES, force: bool = False) -> dict[str, str]:
    """Refresh remote sources into the DB (rate-limited unless ``force``)."""
    status: dict[str, str] = {}
    for src in sources:
        last = _last_sync.get(src)
        if not force and last and time.time() - last[0] < (SYNC_TTL_S if last[1] else FAIL_TTL_S):
            status[src] = "cached" if last[1] else "unavailable"
            continue
        try:
            changed = _upsert(db, src, fetch_remote(src))
            _last_sync[src] = (time.time(), True)
            status[src] = f"ok ({changed} updated)"
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("Reference source %s unavailable: %s", src, exc)
            _last_sync[src] = (time.time(), False)
            status[src] = f"unavailable: {exc.__class__.__name__}"
    return status


def get_reference(
    db: Session, start: date | None = None, end: date | None = None, source: str = "auto", auto_sync: bool = True
) -> dict[date, tuple[float, str]]:
    """``{date: (value, source)}`` for the requested source, or best-available for ``auto``."""
    if source not in SOURCES + ("auto",):
        raise ValueError(f"source must be one of {SOURCES + ('auto',)}")
    if auto_sync and get_settings().silso_enabled and source in REMOTE_SOURCES + ("auto",):
        sync(db, REMOTE_SOURCES if source == "auto" else (source,))
    stmt = select(ReferenceSN)
    if source != "auto":
        stmt = stmt.where(ReferenceSN.source == source)
    if start:
        stmt = stmt.where(ReferenceSN.date >= start)
    if end:
        stmt = stmt.where(ReferenceSN.date <= end)
    out: dict[date, tuple[float, str]] = {}
    for row in db.scalars(stmt):
        cur = out.get(row.date)
        if cur is None or PRIORITY[row.source] < PRIORITY[cur[1]]:
            out[row.date] = (row.value, row.source)
    return out


def set_manual(db: Session, d: date, value: float) -> ReferenceSN:
    row = db.scalars(select(ReferenceSN).where(ReferenceSN.date == d, ReferenceSN.source == "manual")).first()
    if row is None:
        row = ReferenceSN(date=d, source="manual", value=value)
        db.add(row)
    else:
        row.value = value
    db.commit()
    return row


def delete_manual(db: Session, d: date) -> bool:
    row = db.scalars(select(ReferenceSN).where(ReferenceSN.date == d, ReferenceSN.source == "manual")).first()
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True
