"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import date as date_type
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


class StationIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    country: str = Field(..., min_length=1, max_length=64)
    city: str | None = None
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    k_factor: float = Field(1.0, gt=0)


class StationOut(StationIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    n_counts: int = 0


class CountIn(BaseModel):
    date: date_type
    groups: int = Field(..., ge=0)
    spots: int = Field(..., ge=0)
    k: float | None = Field(None, gt=0, description="Observer coefficient; defaults to the station k_factor")
    seeing_arcsec: float | None = Field(None, ge=0)
    notes: str | None = None
    international_r: float | None = Field(None, ge=0, description="Optional international R to store as a manual reference")


class CountOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    station_id: int
    date: date_type
    groups: int
    spots: int
    k: float | None
    seeing_arcsec: float | None
    notes: str | None


class ReferenceIn(BaseModel):
    value: float = Field(..., ge=0)


class ReferenceOut(BaseModel):
    date: date_type
    value: float
    source: str


class ComparisonRowOut(BaseModel):
    date: date_type
    observed: bool
    groups: int | None
    spots: int | None
    k: float | None
    wolf_raw: float | None
    r_local: float | None
    r_calibrated: float | None
    r_international: float | None
    reference_source: str | None
    pct_diff_local: float | None
    pct_diff_calibrated: float | None


class ComparisonSummaryOut(BaseModel):
    k_fit: float | None
    k_fit_sigma: float | None
    n_fit: int
    n_observed: int
    n_interpolated: int
    mean_r_local: float | None
    std_r_local: float | None
    mean_r_calibrated: float | None
    mean_r_international: float | None
    mae_local: float | None
    mae_calibrated: float | None
    rmse_local: float | None
    rmse_calibrated: float | None
    mean_pct_diff_local: float | None
    mean_pct_diff_calibrated: float | None
    notes: list[str] = []


class ComparisonOut(BaseModel):
    station: StationOut
    reference_source: str
    interpolation: str
    rows: list[ComparisonRowOut]
    summary: ComparisonSummaryOut
