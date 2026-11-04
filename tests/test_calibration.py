from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from backend.calibration import calibrate
from backend.calibration.interpolation import spline_estimate
from backend.calibration.seeing import (
    ScintillationSensor,
    estimate_sigma,
    fried_parameter_cm,
    scintillation_index,
    seeing_from_scintillation,
)
from backend.calibration.silso import parse_silso_csv, rolling_mae

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_scintillation_index():
    samples = [1.0, 1.02, 0.98, 1.0] * 10
    si = scintillation_index(samples)
    assert si == pytest.approx(np.var(samples) / np.mean(samples) ** 2)
    with pytest.raises(ValueError):
        scintillation_index([0.0, 0.0])


def test_seeing_monotonic_and_fried():
    assert seeing_from_scintillation(1e-4) < seeing_from_scintillation(4e-4)
    assert fried_parameter_cm(1.0) == pytest.approx(10.1, rel=0.01)


def test_sensor_simulated():
    calm = ScintillationSensor.simulated(0.002, seed=1).measure()
    rough = ScintillationSensor.simulated(0.03, seed=1).measure()
    assert rough.seeing_arcsec > calm.seeing_arcsec


def test_sigma_grows_with_seeing_and_low_confidence():
    base = estimate_sigma(50, 57.3, 0.95)
    assert estimate_sigma(50, 57.3, 0.95, seeing_arcsec=5) > base
    assert estimate_sigma(50, 57.3, 0.5) > base


def test_spline_estimate_recovers_smooth_curve():
    times = [T0 + timedelta(hours=h) for h in range(12)]
    values = [100 + 5 * np.sin(h / 3) for h in range(12)]
    value, sigma = spline_estimate(times, values, T0 + timedelta(hours=5.5))
    assert value == pytest.approx(100 + 5 * np.sin(5.5 / 3), abs=0.2)
    assert sigma < 1.0


def test_spline_requires_two_points():
    assert spline_estimate([T0], [1.0], T0) is None


def test_calibrate_falls_back_when_sigma_high():
    history = [(T0 + timedelta(hours=h), 80.0 + h) for h in range(10)]
    ts = T0 + timedelta(hours=10)
    good = calibrate(60, 68.8, 0.97, ts, history)
    assert not good.interpolated and good.r_corrected == 68.8
    bad = calibrate(60, 68.8, 0.3, ts, history, seeing_arcsec=6)
    assert bad.interpolated
    assert bad.r_corrected == pytest.approx(90.0, abs=1.0)


def test_silso_parse_and_mae():
    csv = "2026;01;01;2026.001;  100;  5.0;  30;0\n2026;01;02;2026.004;   -1; -1.0;   0;0\n2026;01;03;2026.007;  120;  6.0;  31;0\n"
    ref = parse_silso_csv(csv)
    assert ref == {date(2026, 1, 1): 100.0, date(2026, 1, 3): 120.0}
    obs = [(T0, 90.0), (T0 + timedelta(hours=1), 110.0), (T0 + timedelta(days=2), 130.0)]
    assert rolling_mae(obs, ref, window=10) == pytest.approx((0 + 10) / 2)
