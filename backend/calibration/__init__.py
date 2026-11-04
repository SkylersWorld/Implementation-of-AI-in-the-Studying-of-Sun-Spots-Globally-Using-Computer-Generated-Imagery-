"""Seeing-aware calibration with spline interpolation fallback."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from backend.calibration.interpolation import spline_estimate
from backend.calibration.seeing import estimate_sigma
from backend.config import get_settings


@dataclass
class Calibrated:
    r_corrected: float
    sigma: float
    interpolated: bool


def calibrate(
    wolf_raw: float,
    r_corrected: float,
    confidence: float,
    timestamp: datetime,
    history: Sequence[tuple[datetime, float]] = (),
    seeing_arcsec: float | None = None,
) -> Calibrated:
    """Attach sigma; if it exceeds the threshold, replace R with a spline estimate from history."""
    s = get_settings()
    sigma = estimate_sigma(wolf_raw, r_corrected, confidence, seeing_arcsec)
    if sigma <= s.sigma_threshold or not history:
        return Calibrated(r_corrected, sigma, False)
    recent = list(history)[-s.interpolation_window :]
    est = spline_estimate([t for t, _ in recent], [v for _, v in recent], timestamp)
    if est is None:
        return Calibrated(r_corrected, sigma, False)
    value, spline_sigma = est
    return Calibrated(value, spline_sigma, True)
