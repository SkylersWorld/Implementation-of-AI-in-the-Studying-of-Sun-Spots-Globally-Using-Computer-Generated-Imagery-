"""SQLAlchemy engine, session factory, and ORM models."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

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
