"""Station (country) daily counts and comparison with the international sunspot number."""
from __future__ import annotations

import csv
import io
from dataclasses import asdict
from datetime import date, datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.calibration import reference as ref
from backend.calibration.local import INTERPOLATION_METHODS, Count, compare
from backend.database import DailyCount, Station, get_db
from backend.schemas import (
    ComparisonOut,
    CountIn,
    CountOut,
    ReferenceIn,
    ReferenceOut,
    StationIn,
    StationOut,
)

router = APIRouter()
DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y")


def parse_date(text: str) -> date:
    text = text.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date {text!r}; use YYYY-MM-DD or DD.MM.YY")


def _station_out(db: Session, st: Station) -> StationOut:
    n = db.scalar(select(func.count()).select_from(DailyCount).where(DailyCount.station_id == st.id)) or 0
    return StationOut.model_validate({**{c: getattr(st, c) for c in StationIn.model_fields}, "id": st.id, "n_counts": n})


def _get_station(db: Session, station_id: int) -> Station:
    st = db.get(Station, station_id)
    if st is None:
        raise HTTPException(404, "Station not found")
    return st


def upsert_count(db: Session, station: Station, payload: CountIn) -> DailyCount:
    row = db.scalars(select(DailyCount).where(DailyCount.station_id == station.id, DailyCount.date == payload.date)).first()
    data = payload.model_dump(exclude={"international_r"})
    if row is None:
        row = DailyCount(station_id=station.id, **data)
        db.add(row)
    else:
        for k, v in data.items():
            setattr(row, k, v)
    db.commit()
    if payload.international_r is not None:
        ref.set_manual(db, payload.date, payload.international_r)
    db.refresh(row)
    return row


@router.get("/countries", response_model=list[str])
def list_countries(db: Session = Depends(get_db)):
    return list(db.scalars(select(Station.country).distinct().order_by(Station.country)))


@router.get("/stations", response_model=list[StationOut])
def list_stations(country: str | None = None, db: Session = Depends(get_db)):
    stmt = select(Station).order_by(Station.country, Station.name)
    if country:
        stmt = stmt.where(Station.country == country)
    return [_station_out(db, st) for st in db.scalars(stmt)]


@router.post("/stations", response_model=StationOut, status_code=201)
def create_station(payload: StationIn, db: Session = Depends(get_db)):
    if db.scalars(select(Station).where(Station.name == payload.name)).first():
        raise HTTPException(409, "A station with this name already exists")
    st = Station(**payload.model_dump())
    db.add(st)
    db.commit()
    db.refresh(st)
    return _station_out(db, st)


@router.get("/stations/{station_id}", response_model=StationOut)
def get_station(station_id: int, db: Session = Depends(get_db)):
    return _station_out(db, _get_station(db, station_id))


@router.patch("/stations/{station_id}", response_model=StationOut)
def update_station(station_id: int, payload: StationIn, db: Session = Depends(get_db)):
    st = _get_station(db, station_id)
    for k, v in payload.model_dump().items():
        setattr(st, k, v)
    db.commit()
    return _station_out(db, st)


@router.delete("/stations/{station_id}", status_code=204)
def delete_station(station_id: int, db: Session = Depends(get_db)):
    db.delete(_get_station(db, station_id))
    db.commit()


@router.get("/stations/{station_id}/counts", response_model=list[CountOut])
def list_counts(station_id: int, start: date | None = None, end: date | None = None, db: Session = Depends(get_db)):
    _get_station(db, station_id)
    stmt = select(DailyCount).where(DailyCount.station_id == station_id).order_by(DailyCount.date)
    if start:
        stmt = stmt.where(DailyCount.date >= start)
    if end:
        stmt = stmt.where(DailyCount.date <= end)
    return list(db.scalars(stmt))


@router.post("/stations/{station_id}/counts", response_model=CountOut, status_code=201)
def add_count(station_id: int, payload: CountIn, db: Session = Depends(get_db)):
    """Create or replace the count for ``payload.date``."""
    return upsert_count(db, _get_station(db, station_id), payload)


@router.delete("/stations/{station_id}/counts/{day}", status_code=204)
def delete_count(station_id: int, day: date, db: Session = Depends(get_db)):
    row = db.scalars(select(DailyCount).where(DailyCount.station_id == station_id, DailyCount.date == day)).first()
    if row is None:
        raise HTTPException(404, "No count for that date")
    db.delete(row)
    db.commit()


def import_csv_text(db: Session, station: Station, text: str) -> int:
    """Import ``date,g,s[,k][,ri][,seeing][,notes]`` rows (header required)."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("CSV is empty")
    cols = {c.strip().lower(): c for c in reader.fieldnames}
    missing = {"date", "g", "s"} - cols.keys()
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    def val(row, key, cast):
        raw = row.get(cols[key], "") if key in cols else ""
        raw = (raw or "").strip()
        return cast(raw) if raw else None

    n = 0
    for i, row in enumerate(reader, start=2):
        try:
            payload = CountIn(
                date=parse_date(row[cols["date"]]),
                groups=val(row, "g", int),
                spots=val(row, "s", int),
                k=val(row, "k", float),
                international_r=val(row, "ri", float),
                seeing_arcsec=val(row, "seeing", float),
                notes=val(row, "notes", str),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Row {i}: {exc}") from exc
        upsert_count(db, station, payload)
        n += 1
    return n


@router.post("/stations/{station_id}/counts/import")
async def import_counts(station_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    st = _get_station(db, station_id)
    try:
        n = import_csv_text(db, st, (await file.read()).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"imported": n}


@router.get("/stations/{station_id}/compare", response_model=ComparisonOut)
def compare_station(
    station_id: int,
    start: date | None = None,
    end: date | None = None,
    source: str = Query("auto", description="auto | manual | silso | swpc"),
    interpolation: str = Query("linear", description="linear | spline | none"),
    db: Session = Depends(get_db),
):
    """Daily local vs calibrated vs international R with % differences and summary stats."""
    st = _get_station(db, station_id)
    if interpolation not in INTERPOLATION_METHODS:
        raise HTTPException(422, f"interpolation must be one of {INTERPOLATION_METHODS}")
    counts = [Count(c.date, c.groups, c.spots, c.k) for c in sorted(st.counts, key=lambda c: c.date)]
    try:
        reference = ref.get_reference(db, source=source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    rows, summary = compare(counts, reference, st.k_factor, interpolation, start, end)
    # Days in range the station didn't cover still show the international value.
    if start and end:
        covered = {r.date for r in rows}
        extra = [
            {"date": d, "observed": False, "groups": None, "spots": None, "k": None, "wolf_raw": None, "r_local": None,
             "r_calibrated": None, "r_international": v, "reference_source": s, "pct_diff_local": None, "pct_diff_calibrated": None}
            for d, (v, s) in reference.items() if start <= d <= end and d not in covered
        ]
        row_dicts = sorted([r.as_dict() for r in rows] + extra, key=lambda r: r["date"])
    else:
        row_dicts = [r.as_dict() for r in rows]
    return ComparisonOut(
        station=_station_out(db, st), reference_source=source, interpolation=interpolation, rows=row_dicts, summary=asdict(summary)
    )


@router.get("/reference", response_model=list[ReferenceOut])
def list_reference(start: date | None = None, end: date | None = None, source: str = "auto", db: Session = Depends(get_db)):
    try:
        data = ref.get_reference(db, start, end, source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return [ReferenceOut(date=d, value=v, source=s) for d, (v, s) in sorted(data.items())]


@router.put("/reference/{day}", response_model=ReferenceOut)
def put_reference(day: date, payload: ReferenceIn, db: Session = Depends(get_db)):
    row = ref.set_manual(db, day, payload.value)
    return ReferenceOut(date=row.date, value=row.value, source=row.source)


@router.delete("/reference/{day}", status_code=204)
def delete_reference(day: date, db: Session = Depends(get_db)):
    if not ref.delete_manual(db, day):
        raise HTTPException(404, "No manual reference for that date")


@router.post("/reference/refresh")
def refresh_reference(db: Session = Depends(get_db)) -> dict:
    return ref.sync(db, force=True)
