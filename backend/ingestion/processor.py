"""Turn an image into a calibrated, alert-checked Observation row."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.alerts.rules import Snapshot, check_and_alert
from backend.calibration import calibrate
from backend.calibration.reference import get_reference
from backend.calibration.silso import rolling_mae
from backend.config import get_settings
from backend.database import Observation, SessionLocal
from backend.ingestion.loader import FITS_EXTS, LoadedImage, load_image
from backend.model.dataset import IMAGE_EXTS
from backend.model.inference import get_segmenter

log = logging.getLogger(__name__)


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def recent_history(db: Session, before: datetime, limit: int) -> list[Observation]:
    rows = db.scalars(
        select(Observation).where(Observation.timestamp < before).order_by(Observation.timestamp.desc()).limit(limit)
    ).all()
    return list(reversed(rows))


def process_image(
    db: Session,
    image: np.ndarray,
    timestamp: datetime | None = None,
    source: str = "upload",
    filename: str | None = None,
    magnetogram: np.ndarray | None = None,
    seeing_arcsec: float | None = None,
    send_alerts: bool = True,
) -> Observation:
    s = get_settings()
    timestamp = _aware(timestamp or datetime.now(timezone.utc))
    result = get_segmenter().analyze(image, magnetogram)

    history = recent_history(db, timestamp, s.interpolation_window)
    hist_pairs = [(_aware(o.timestamp), o.r_corrected) for o in history]
    cal = calibrate(result.wolf_raw, result.r_corrected, result.confidence, timestamp, hist_pairs, seeing_arcsec)

    mae = None
    if s.silso_enabled:
        reference = {d: v for d, (v, _) in get_reference(db).items()}
        if reference:
            mae = rolling_mae(hist_pairs + [(timestamp, cal.r_corrected)], reference)

    obs = Observation(
        timestamp=timestamp,
        source=source,
        filename=filename,
        groups=result.groups,
        spots=result.spots,
        wolf_raw=result.wolf_raw,
        r_corrected=cal.r_corrected,
        sigma=cal.sigma,
        confidence=result.confidence,
        delta_class=result.delta_class,
        mae=mae,
        seeing_arcsec=seeing_arcsec,
        interpolated=cal.interpolated,
    )
    db.add(obs)
    db.commit()
    db.refresh(obs)
    log.info(
        "Observation %s: g=%d s=%d R=%.1f±%.1f (%s)",
        obs.id, obs.groups, obs.spots, obs.r_corrected, obs.sigma, result.method,
    )

    if send_alerts:
        prev = history[-1] if history else None
        check_and_alert(
            Snapshot(timestamp, obs.r_corrected, obs.delta_class, obs.groups, obs.spots),
            Snapshot(_aware(prev.timestamp), prev.r_corrected, prev.delta_class, prev.groups, prev.spots) if prev else None,
        )
    return obs


def _pair_magnetogram(cont: LoadedImage, mags: list[tuple[Path, LoadedImage]], tolerance: timedelta) -> tuple[Path, LoadedImage] | None:
    if not mags or cont.timestamp is None:
        return mags[0] if len(mags) == 1 else None
    best = min(mags, key=lambda m: abs((m[1].timestamp or cont.timestamp) - cont.timestamp))
    if best[1].timestamp and abs(best[1].timestamp - cont.timestamp) > tolerance:
        return None
    return best


def process_hot_folder(
    hot_folder: Path | None = None,
    processed_folder: Path | None = None,
    seeing_arcsec: float | None = None,
) -> list[Observation]:
    """Process every image in the hot folder, then move inputs to the processed folder."""
    s = get_settings()
    hot = Path(hot_folder or s.hot_folder)
    done = Path(processed_folder or s.processed_folder)
    hot.mkdir(parents=True, exist_ok=True)
    done.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in hot.iterdir() if p.is_file() and p.suffix.lower() in FITS_EXTS | IMAGE_EXTS)
    loaded: list[tuple[Path, LoadedImage]] = []
    for p in files:
        try:
            loaded.append((p, load_image(p)))
        except Exception as exc:
            log.error("Failed to load %s: %s", p, exc)
    mags = [x for x in loaded if x[1].measurement == "magnetogram"]
    conts = [x for x in loaded if x[1].measurement != "magnetogram"]

    observations: list[Observation] = []
    with SessionLocal() as db:
        for path, img in sorted(conts, key=lambda x: x[1].timestamp or datetime.min.replace(tzinfo=timezone.utc)):
            mag = _pair_magnetogram(img, mags, timedelta(minutes=15))
            source = "hmi" if path.suffix.lower() in FITS_EXTS else "ground"
            observations.append(
                process_image(db, img.data, img.timestamp, source, path.name, mag[1].data if mag else None, seeing_arcsec)
            )
    for path, _ in loaded:
        shutil.move(str(path), done / path.name)
    return observations
