"""Local (station) Wolf number calibration against the international sunspot number.

For a station day with g groups and s spots:

    W        = 10 g + s                    raw Wolf number
    R_local  = k_obs * W                   observer's own coefficient (as recorded)
    R_cal    = k_fit * W                   k_fit = sum(Ri*W) / sum(W^2) (least squares through origin)
    %diff    = (R - Ri) / Ri * 100

Missing days are filled by linear interpolation between the neighbouring observed
days a < d < b:  R_d = R_a + (d - a) / (b - a) * (R_b - R_a)   (or a cubic spline).
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np
from scipy.interpolate import CubicSpline

INTERPOLATION_METHODS = ("linear", "spline", "none")


@dataclass
class Count:
    date: date
    groups: int
    spots: int
    k: float | None = None


@dataclass
class ComparisonRow:
    date: date
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

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonSummary:
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
    notes: list[str] = field(default_factory=list)


def wolf_raw(groups: int, spots: int) -> float:
    return float(10 * groups + spots)


def pct_diff(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0:
        return None
    return (value - reference) / reference * 100.0


def fit_k(pairs: Iterable[tuple[float, float]]) -> tuple[float | None, float | None, int]:
    """Least-squares k for Ri ≈ k·W through the origin. Returns ``(k, sigma_k, n)``."""
    pts = [(w, ri) for w, ri in pairs if w is not None and ri is not None]
    n = len(pts)
    sww = sum(w * w for w, _ in pts)
    if n == 0 or sww == 0:
        return None, None, n
    k = sum(w * ri for w, ri in pts) / sww
    if n < 2:
        return k, None, n
    resid = sum((ri - k * w) ** 2 for w, ri in pts)
    return k, math.sqrt(resid / (n - 1) / sww), n


def interpolate_daily(
    observed: dict[date, float], method: str = "linear", start: date | None = None, end: date | None = None
) -> dict[date, tuple[float, bool]]:
    """Daily series ``{date: (value, interpolated)}`` between the first and last observed day.

    Days outside the observed span are never extrapolated.
    """
    if method not in INTERPOLATION_METHODS:
        raise ValueError(f"method must be one of {INTERPOLATION_METHODS}")
    if not observed:
        return {}
    days = sorted(observed)
    first, last = days[0], days[-1]
    lo, hi = max(start or first, first), min(end or last, last)
    out: dict[date, tuple[float, bool]] = {}
    if method == "none":
        return {d: (observed[d], False) for d in days if lo <= d <= hi}

    x = np.array([(d - first).days for d in days], dtype=float)
    y = np.array([observed[d] for d in days], dtype=float)
    spline = CubicSpline(x, y, bc_type="natural") if method == "spline" and len(days) >= 4 else None
    d = lo
    while d <= hi:
        if d in observed:
            out[d] = (observed[d], False)
        else:
            t = (d - first).days
            val = float(spline(t)) if spline is not None else float(np.interp(t, x, y))
            out[d] = (max(val, 0.0), True)
        d += timedelta(days=1)
    return out


def _mean(xs: Sequence[float]) -> float | None:
    return float(statistics.fmean(xs)) if xs else None


def compare(
    counts: Sequence[Count],
    reference: dict[date, tuple[float, str]],
    station_k: float = 1.0,
    method: str = "linear",
    start: date | None = None,
    end: date | None = None,
) -> tuple[list[ComparisonRow], ComparisonSummary]:
    """Compare a station's daily counts with the international sunspot number.

    ``k_fit`` is fitted on *all* supplied observed days that have a reference value,
    then applied to every row; summary statistics cover the observed days in range.
    """
    by_day = {c.date: c for c in counts}
    w_obs = {d: wolf_raw(c.groups, c.spots) for d, c in by_day.items()}
    r_obs = {d: (c.k if c.k is not None else station_k) * w_obs[d] for d, c in by_day.items()}
    k_fit, k_sigma, n_fit = fit_k((w_obs[d], reference[d][0]) for d in w_obs if d in reference)

    r_series = interpolate_daily(r_obs, method, start, end)
    w_series = interpolate_daily(w_obs, method, start, end)

    rows: list[ComparisonRow] = []
    for d in sorted(r_series):
        r_local, interpolated = r_series[d]
        w = w_series[d][0]
        c = by_day.get(d)
        ref_val, ref_src = reference.get(d, (None, None))
        r_cal = k_fit * w if k_fit is not None else None
        rows.append(
            ComparisonRow(
                date=d,
                observed=not interpolated,
                groups=c.groups if c else None,
                spots=c.spots if c else None,
                k=(c.k if c.k is not None else station_k) if c else None,
                wolf_raw=w,
                r_local=r_local,
                r_calibrated=r_cal,
                r_international=ref_val,
                reference_source=ref_src,
                pct_diff_local=pct_diff(r_local, ref_val),
                pct_diff_calibrated=pct_diff(r_cal, ref_val),
            )
        )

    obs_rows = [r for r in rows if r.observed]
    matched = [r for r in obs_rows if r.r_international is not None]
    local_err = [r.r_local - r.r_international for r in matched]
    cal_err = [r.r_calibrated - r.r_international for r in matched if r.r_calibrated is not None]
    local_vals = [r.r_local for r in obs_rows]
    notes = []
    if n_fit < 3:
        notes.append("k_fit uses fewer than 3 days with an international value; treat it as provisional.")
    summary = ComparisonSummary(
        k_fit=k_fit,
        k_fit_sigma=k_sigma,
        n_fit=n_fit,
        n_observed=len(obs_rows),
        n_interpolated=len(rows) - len(obs_rows),
        mean_r_local=_mean(local_vals),
        std_r_local=float(statistics.pstdev(local_vals)) if len(local_vals) > 1 else None,
        mean_r_calibrated=_mean([r.r_calibrated for r in obs_rows if r.r_calibrated is not None]),
        mean_r_international=_mean([r.r_international for r in matched]),
        mae_local=_mean([abs(e) for e in local_err]),
        mae_calibrated=_mean([abs(e) for e in cal_err]),
        rmse_local=math.sqrt(_mean([e * e for e in local_err])) if local_err else None,
        rmse_calibrated=math.sqrt(_mean([e * e for e in cal_err])) if cal_err else None,
        mean_pct_diff_local=_mean([r.pct_diff_local for r in matched if r.pct_diff_local is not None]),
        mean_pct_diff_calibrated=_mean([r.pct_diff_calibrated for r in matched if r.pct_diff_calibrated is not None]),
        notes=notes,
    )
    return rows, summary
