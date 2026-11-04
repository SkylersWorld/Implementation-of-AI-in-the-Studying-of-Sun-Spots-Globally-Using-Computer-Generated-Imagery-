"""Reproduces the numbers in the TSU internship report (Tbilisi, Aug-Sep 2026)."""
from datetime import date

import pytest

from backend.calibration.local import Count, compare, fit_k, interpolate_daily, pct_diff

TSU = [
    ("2026-08-21", 2, 5, 1.4, 112), ("2026-08-24", 6, 28, 1.5, 91), ("2026-09-02", 2, 3, 1.5, 52),
    ("2026-09-07", 4, 10, 1.5, 97), ("2026-09-08", 7, 10, 1.2, 98), ("2026-09-11", 6, 7, 1.4, 101),
    ("2026-09-14", 2, 3, 1.5, 105), ("2026-09-16", 7, 15, 1.5, 114), ("2026-09-17", 7, 10, 1.5, 118),
]
COUNTS = [Count(date.fromisoformat(d), g, s, k) for d, g, s, k, _ in TSU]
REF = {date.fromisoformat(d): (float(ri), "manual") for d, *_, ri in TSU}


def test_report_mean_and_std():
    rows, summary = compare(COUNTS, REF, method="none")
    assert [r.r_local for r in rows] == pytest.approx([35, 132, 34.5, 75, 96, 93.8, 34.5, 127.5, 120])
    assert summary.mean_r_local == pytest.approx(83.14, abs=0.01)
    assert summary.std_r_local == pytest.approx(38.16, abs=0.01)


def test_report_linear_interpolation():
    rows, summary = compare(COUNTS, REF, method="linear")
    by_day = {r.date: r for r in rows}
    assert len(rows) == 28 and summary.n_interpolated == 19
    assert by_day[date(2026, 8, 22)].r_local == pytest.approx(67.33, abs=0.01)
    assert by_day[date(2026, 8, 25)].r_local == pytest.approx(121.17, abs=0.01)
    assert by_day[date(2026, 9, 3)].r_local == pytest.approx(42.60, abs=0.01)
    assert by_day[date(2026, 9, 15)].r_local == pytest.approx(81.0, abs=0.01)
    assert not by_day[date(2026, 8, 22)].observed


def test_k_fit_and_percent_difference():
    k, sigma, n = fit_k([(10 * g + s, ri) for _, g, s, _, ri in TSU])
    assert k == pytest.approx(1.4546, abs=1e-3) and n == 9 and sigma > 0
    rows, summary = compare(COUNTS, REF, method="none")
    first = rows[0]
    assert first.pct_diff_local == pytest.approx((35 - 112) / 112 * 100)
    assert first.r_calibrated == pytest.approx(k * 25)
    assert summary.mae_calibrated is not None and summary.mae_local is not None
    assert pct_diff(10, 0) is None


def test_spline_and_range_clipping():
    obs = {date(2026, 1, d): float(v) for d, v in [(1, 10), (3, 30), (6, 60), (9, 90), (12, 120)]}
    series = interpolate_daily(obs, "spline", start=date(2025, 12, 1), end=date(2026, 1, 5))
    assert min(series) == date(2026, 1, 1) and max(series) == date(2026, 1, 5)
    assert series[date(2026, 1, 2)][0] == pytest.approx(20, abs=1.0) and series[date(2026, 1, 2)][1]
    with pytest.raises(ValueError):
        interpolate_daily(obs, "cubic")
