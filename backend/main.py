"""FastAPI application exposing the Solar AI Pipeline."""
from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.alerts.rules import Snapshot, cme_risk, drdt_per_hour
from backend.calibration.reference import get_reference
from backend.calibration.seeing import measure_seeing
from backend.calibration.silso import rolling_mae
from backend.config import get_settings
from backend.database import Observation, get_db, init_db
from backend.hardware.mount import MountClient, sun_radec
from backend.ingestion.loader import load_image
from backend.ingestion.processor import process_hot_folder, process_image
from backend.model.inference import get_segmenter
from backend.schemas import (
    MetricsOut,
    MountSyncIn,
    MountSyncOut,
    ObservationCreate,
    ObservationOut,
    SeeingIn,
    SeeingOut,
)
from backend.stations import router as stations_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("solar-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler = None
    if get_settings().scheduler_enabled:
        from backend.ingestion.scheduler import start_background

        scheduler = start_background()
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(title=get_settings().app_name, version="1.1.0", lifespan=lifespan)
app.include_router(stations_router)


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


@app.get("/health")
def health() -> dict:
    seg = get_segmenter()
    return {"status": "ok", "model": "unet" if seg.model is not None else "classical-fallback", "time": datetime.now(timezone.utc)}


@app.get("/observations", response_model=list[ObservationOut])
def list_observations(
    limit: int = Query(500, ge=1, le=10000),
    since: datetime | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(Observation).order_by(Observation.timestamp.desc()).limit(limit)
    if since:
        stmt = stmt.where(Observation.timestamp >= since)
    return list(reversed(db.scalars(stmt).all()))


@app.get("/observations/latest", response_model=ObservationOut)
def latest_observation(db: Session = Depends(get_db)):
    obs = db.scalars(select(Observation).order_by(Observation.timestamp.desc()).limit(1)).first()
    if obs is None:
        raise HTTPException(404, "No observations yet")
    return obs


@app.get("/observations/{obs_id}", response_model=ObservationOut)
def get_observation(obs_id: int, db: Session = Depends(get_db)):
    obs = db.get(Observation, obs_id)
    if obs is None:
        raise HTTPException(404, "Observation not found")
    return obs


@app.post("/observations", response_model=ObservationOut, status_code=201)
def create_observation(payload: ObservationCreate, db: Session = Depends(get_db)):
    data = payload.model_dump()
    data["timestamp"] = data["timestamp"] or datetime.now(timezone.utc)
    obs = Observation(**data)
    db.add(obs)
    db.commit()
    db.refresh(obs)
    return obs


async def _load_upload(upload: UploadFile, tmpdir: Path):
    path = tmpdir / Path(upload.filename or "upload.fits").name
    path.write_bytes(await upload.read())
    try:
        return load_image(path)
    except Exception as exc:
        raise HTTPException(400, f"Could not read image {upload.filename}: {exc}") from exc


@app.post("/infer", response_model=ObservationOut, status_code=201)
async def infer(
    image: UploadFile = File(..., description="Continuum image: FITS, PNG/JPEG/TIFF or .npy"),
    magnetogram: UploadFile | None = File(None, description="Optional co-aligned LOS magnetogram for delta detection"),
    seeing_arcsec: float | None = Form(None),
    timestamp: datetime | None = Form(None),
    source: str = Form("upload"),
    db: Session = Depends(get_db),
):
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        img = await _load_upload(image, tmpdir)
        mag = await _load_upload(magnetogram, tmpdir) if magnetogram and magnetogram.filename else None
    return process_image(
        db,
        img.data,
        timestamp or img.timestamp,
        source,
        image.filename,
        mag.data if mag else None,
        seeing_arcsec,
    )


@app.post("/ingest/process", response_model=list[ObservationOut])
def ingest_process(seeing_arcsec: float | None = None):
    """Process everything currently in the hot folder."""
    return process_hot_folder(seeing_arcsec=seeing_arcsec)


@app.post("/ingest/run", status_code=202)
def ingest_run(background: BackgroundTasks) -> dict:
    """Trigger the hourly HMI fetch + process job immediately (async)."""
    from backend.ingestion.scheduler import hourly_job

    background.add_task(hourly_job)
    return {"status": "scheduled"}


@app.post("/calibration/seeing", response_model=SeeingOut)
def calibration_seeing(payload: SeeingIn):
    try:
        m = measure_seeing(payload.samples)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return SeeingOut(scintillation_index=m.scintillation_index, seeing_arcsec=m.seeing_arcsec, fried_parameter_cm=m.fried_parameter_cm)


@app.get("/metrics", response_model=MetricsOut)
def metrics(db: Session = Depends(get_db)):
    rows = list(reversed(db.scalars(select(Observation).order_by(Observation.timestamp.desc()).limit(1000)).all()))
    if not rows:
        return MetricsOut(count=0, latest_r=None, rolling_mae=None, drdt_per_hour=None, cme_risk="green")
    latest = rows[-1]
    snap = lambda o: Snapshot(_aware(o.timestamp), o.r_corrected, o.delta_class, o.groups, o.spots)  # noqa: E731
    rate = drdt_per_hour(snap(rows[-2]), snap(latest)) if len(rows) > 1 else None
    mae = latest.mae
    if mae is None:
        reference = {d: v for d, (v, _) in get_reference(db).items()}
        mae = rolling_mae([(_aware(o.timestamp), o.r_corrected) for o in rows], reference) if reference else None
    return MetricsOut(
        count=len(rows),
        latest_r=latest.r_corrected,
        rolling_mae=mae,
        drdt_per_hour=rate,
        cme_risk=cme_risk(latest.delta_class, rate),
    )


@app.get("/metrics/silso")
def silso_reference(days: int = Query(90, ge=1, le=3650), db: Session = Depends(get_db)) -> list[dict]:
    """International daily R (best available: manual > SILSO > NOAA SWPC) for the last ``days``."""
    data = get_reference(db, start=date.today() - timedelta(days=days))
    return [{"date": d.isoformat(), "sn": v, "source": s} for d, (v, s) in sorted(data.items())]


@app.post("/hardware/sync", response_model=MountSyncOut)
def hardware_sync(payload: MountSyncIn | None = None):
    """Sync the mount to the given RA/Dec, or to the current Sun position if omitted."""
    if payload and payload.ra_deg is not None and payload.dec_deg is not None:
        ra, dec = payload.ra_deg, payload.dec_deg
    else:
        ra, dec = sun_radec()
    try:
        with MountClient() as client:
            reply = client.sync(ra, dec)
    except OSError as exc:
        raise HTTPException(502, f"Mount unreachable: {exc}") from exc
    return MountSyncOut(ok=True, ra_deg=ra, dec_deg=dec, response=reply)
