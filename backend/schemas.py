"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ObservationBase(BaseModel):
    timestamp: datetime | None = None
    source: str = "manual"
    filename: str | None = None
    groups: int = Field(0, ge=0)
    spots: int = Field(0, ge=0)
    wolf_raw: float = 0.0
    r_corrected: float = 0.0
    sigma: float = 0.0
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    delta_class: bool = False
    mae: float | None = None
    seeing_arcsec: float | None = None
    interpolated: bool = False


class ObservationCreate(ObservationBase):
    pass


class ObservationOut(ObservationBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime


class SeeingIn(BaseModel):
    samples: list[float] = Field(..., min_length=8, description="Photometer intensity samples")


class SeeingOut(BaseModel):
    scintillation_index: float
    seeing_arcsec: float
    fried_parameter_cm: float


class MountSyncIn(BaseModel):
    ra_deg: float | None = Field(None, ge=0, lt=360)
    dec_deg: float | None = Field(None, ge=-90, le=90)


class MountSyncOut(BaseModel):
    ok: bool
    ra_deg: float
    dec_deg: float
    response: str


class MetricsOut(BaseModel):
    count: int
    latest_r: float | None
    rolling_mae: float | None
    drdt_per_hour: float | None
    cme_risk: str
