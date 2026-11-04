"""Fetch the latest SDO/HMI continuum (and LOS magnetogram) FITS via SunPy Fido."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.config import get_settings

log = logging.getLogger(__name__)

PHYSOBS = {"continuum": "intensity", "magnetogram": "LOS_magnetic_field"}


def fetch_latest(
    kind: str = "continuum",
    hot_folder: Path | None = None,
    lookback_minutes: int | None = None,
    end: datetime | None = None,
) -> list[Path]:
    """Download the most recent HMI ``kind`` image in ``[end - lookback, end]`` into the hot folder.

    ``end`` defaults to now; pass a past time to backfill.
    """
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    s = get_settings()
    hot_folder = Path(hot_folder or s.hot_folder)
    hot_folder.mkdir(parents=True, exist_ok=True)
    end = end or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(minutes=lookback_minutes or s.hmi_lookback_minutes)

    result = Fido.search(
        a.Time(start.replace(tzinfo=None), end.replace(tzinfo=None)),
        a.Instrument("HMI"),
        a.Physobs(PHYSOBS[kind]),
    )
    if not len(result) or not len(result[0]):
        log.info("No HMI %s data between %s and %s", kind, start, end)
        return []
    latest = result[0][-1:]
    files = Fido.fetch(latest, path=str(hot_folder / f"hmi_{'ic' if kind == 'continuum' else 'm'}_{{file}}"), progress=False)
    paths = [Path(f) for f in files]
    log.info("Fetched HMI %s: %s", kind, [p.name for p in paths])
    return paths


def fetch_hourly_bundle(hot_folder: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    for kind in ("continuum", "magnetogram"):
        try:
            paths += fetch_latest(kind, hot_folder)
        except Exception as exc:
            log.error("HMI %s fetch failed: %s", kind, exc)
    return paths
