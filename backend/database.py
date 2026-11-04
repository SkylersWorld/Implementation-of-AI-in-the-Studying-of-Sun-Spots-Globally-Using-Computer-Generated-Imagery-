"""SQLAlchemy engine, session factory, and ORM models."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from backend.config import get_settings


class Base(DeclarativeBase):
    pass


class Observation(Base):
    """One processed solar disk observation."""

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc)
    )
    source: Mapped[str] = mapped_column(String(64), default="hmi")
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)

    groups: Mapped[int] = mapped_column(Integer, default=0)
    spots: Mapped[int] = mapped_column(Integer, default=0)
    wolf_raw: Mapped[float] = mapped_column(Float, default=0.0)
    r_corrected: Mapped[float] = mapped_column(Float, default=0.0)
    sigma: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    delta_class: Mapped[bool] = mapped_column(Boolean, default=False)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)

    seeing_arcsec: Mapped[float | None] = mapped_column(Float, nullable=True)
    interpolated: Mapped[bool] = mapped_column(Boolean, default=False)


class Station(Base):
    """A ground observing station run by a user (name, country, location, default k)."""

    __tablename__ = "stations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    country: Mapped[str] = mapped_column(String(64), index=True)
    city: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    k_factor: Mapped[float] = mapped_column(Float, default=1.0)

    counts: Mapped[list["DailyCount"]] = relationship(back_populates="station", cascade="all, delete-orphan")


class DailyCount(Base):
    """One day's visual sunspot count at a station: groups g, spots s, observer coefficient k."""

    __tablename__ = "daily_counts"
    __table_args__ = (UniqueConstraint("station_id", "date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[int] = mapped_column(ForeignKey("stations.id", ondelete="CASCADE"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    groups: Mapped[int] = mapped_column(Integer)
    spots: Mapped[int] = mapped_column(Integer)
    k: Mapped[float | None] = mapped_column(Float, nullable=True)
    seeing_arcsec: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(String(512), nullable=True)

    station: Mapped[Station] = relationship(back_populates="counts")


class SolarDay(Base):
    """The AI's count of the real SDO/HMI full-disk image for one date (shared by all stations)."""

    __tablename__ = "solar_days"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | done | failed
    obs_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    method: Mapped[str | None] = mapped_column(String(16), nullable=True)
    groups: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wolf_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    r_ai: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error: Mapped[str | None] = mapped_column(String(512), nullable=True)


class ReferenceSN(Base):
    """International daily sunspot number for one date from one source (manual / silso / swpc)."""

    __tablename__ = "reference_sn"
    __table_args__ = (UniqueConstraint("date", "source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    source: Mapped[str] = mapped_column(String(16))
    value: Mapped[float] = mapped_column(Float)


def _make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


engine = _make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def configure(url: str) -> None:
    """Rebind the engine/session factory (used by tests and CLIs)."""
    global engine
    engine = _make_engine(url)
    SessionLocal.configure(bind=engine)


def init_db() -> None:
    if engine.url.get_backend_name() == "sqlite" and engine.url.database:
        from pathlib import Path

        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
