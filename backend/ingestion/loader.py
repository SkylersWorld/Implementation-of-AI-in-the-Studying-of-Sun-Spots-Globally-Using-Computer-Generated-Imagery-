"""Load FITS (via SunPy/Astropy), PNG/JPEG/TIFF or NumPy images into arrays."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.model.dataset import load_array

log = logging.getLogger(__name__)
FITS_EXTS = {".fits", ".fit", ".fts", ".fz"}


@dataclass
class LoadedImage:
    data: np.ndarray
    timestamp: datetime | None = None
    measurement: str | None = None  # e.g. "continuum", "magnetogram"


def _measurement_from_name(name: str) -> str | None:
    n = name.lower()
    if "_m_" in n or "magnetogram" in n:
        return "magnetogram"
    if "ic_" in n or "continuum" in n or "intensity" in n:
        return "continuum"
    return None


def _parse_date(value: object) -> datetime | None:
    from astropy.time import Time

    text = str(value).strip().replace("_TAI", "").replace("_", "T").replace(".", "-", 2)
    for candidate in (str(value).strip(), text):
        try:
            return Time(candidate.rstrip("Z")).to_datetime().replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def load_fits(path: Path) -> LoadedImage:
    try:
        import sunpy.map

        smap = sunpy.map.Map(str(path))
        measurement = str(smap.measurement).lower() if smap.measurement is not None else None
        if measurement and "magnet" in measurement:
            measurement = "magnetogram"
        elif measurement and ("continuum" in measurement or "intensity" in measurement):
            measurement = "continuum"
        ts = smap.date.to_datetime().replace(tzinfo=timezone.utc) if smap.date is not None else None
        return LoadedImage(np.asarray(smap.data, dtype=np.float32), ts, measurement or _measurement_from_name(path.name))
    except Exception as exc:  # non-solar or malformed headers: fall back to raw astropy
        log.debug("sunpy.map failed for %s (%s); falling back to astropy", path, exc)
        from astropy.io import fits

        with fits.open(path) as hdul:
            hdu = next(h for h in hdul if h.data is not None)
            data = np.asarray(hdu.data, dtype=np.float32)
            date_obs = hdu.header.get("DATE-OBS") or hdu.header.get("T_OBS")
        ts = _parse_date(date_obs) if date_obs else None
        return LoadedImage(data, ts, _measurement_from_name(path.name))


def load_image(path: str | Path) -> LoadedImage:
    path = Path(path)
    if path.suffix.lower() in FITS_EXTS:
        return load_fits(path)
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return LoadedImage(load_array(path), ts, _measurement_from_name(path.name) or "continuum")
