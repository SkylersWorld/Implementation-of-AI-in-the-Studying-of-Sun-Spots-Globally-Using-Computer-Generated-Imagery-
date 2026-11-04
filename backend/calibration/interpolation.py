"""Cubic-spline statistical interpolation fallback for unreliable observations."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from scipy.interpolate import CubicSpline


def _epoch(t: datetime) -> float:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


def spline_estimate(
    times: Sequence[datetime], values: Sequence[float], target: datetime
) -> tuple[float, float] | None:
    """Estimate R at ``target`` from history with a natural cubic spline.

    Returns ``(value, sigma)`` where sigma is the RMS leave-one-out residual of
    the spline over the history, or ``None`` if there are fewer than 2 points.
    Falls back to linear interpolation with fewer than 4 points.
    """
    if len(times) != len(values):
        raise ValueError("times and values must have equal length")
    pairs = sorted({_epoch(t): float(v) for t, v in zip(times, values)}.items())
    if len(pairs) < 2:
        return None
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    tx = _epoch(target)

    if len(x) < 4:
        value = float(np.interp(tx, x, y)) if x[0] <= tx <= x[-1] else float(y[-1])
        return max(value, 0.0), float(np.std(y))

    spline = CubicSpline(x, y, bc_type="natural", extrapolate=True)
    value = float(spline(tx))
    residuals = []
    for i in range(1, len(x) - 1):
        xs, ys = np.delete(x, i), np.delete(y, i)
        if len(xs) >= 4:
            residuals.append(float(CubicSpline(xs, ys, bc_type="natural")(x[i])) - y[i])
        else:
            residuals.append(float(np.interp(x[i], xs, ys)) - y[i])
    sigma = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float(np.std(y))
    return max(value, 0.0), sigma
