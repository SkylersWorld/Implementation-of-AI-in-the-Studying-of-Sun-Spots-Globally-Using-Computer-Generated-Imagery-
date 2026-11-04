"""Scintillation seeing sensor (SHABAR-style photometer) and uncertainty model.

The scintillation index of a solar photometer, sigma_I^2 = var(I) / mean(I)^2,
is mapped to an equivalent seeing FWHM with an empirical linear relation
``seeing = a + b * (100 * sqrt(sigma_I^2))`` whose coefficients should be fitted
per-site against a DIMM. Seeing then degrades confidence in the sunspot count.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from backend.config import get_settings

ARCSEC_PER_RAD = 206264.806


@dataclass
class SeeingMeasurement:
    scintillation_index: float
    seeing_arcsec: float
    fried_parameter_cm: float


def scintillation_index(samples: Sequence[float]) -> float:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size < 2:
        raise ValueError("Need at least 2 samples")
    mean = arr.mean()
    if mean <= 0:
        raise ValueError("Mean intensity must be positive")
    return float(arr.var() / mean**2)


def seeing_from_scintillation(si: float, a: float | None = None, b: float | None = None) -> float:
    s = get_settings()
    a = s.seeing_coeff_a if a is None else a
    b = s.seeing_coeff_b if b is None else b
    return float(a + b * 100.0 * math.sqrt(max(si, 0.0)))


def fried_parameter_cm(seeing_arcsec: float, wavelength_nm: float = 500.0) -> float:
    """r0 = 0.98 * lambda / FWHM."""
    if seeing_arcsec <= 0:
        return float("inf")
    return float(0.98 * wavelength_nm * 1e-9 / (seeing_arcsec / ARCSEC_PER_RAD) * 100.0)


def measure_seeing(samples: Sequence[float]) -> SeeingMeasurement:
    si = scintillation_index(samples)
    seeing = seeing_from_scintillation(si)
    return SeeingMeasurement(si, seeing, fried_parameter_cm(seeing))


class ScintillationSensor:
    """Reads photometer samples from a pluggable source.

    ``reader`` returns ``n`` intensity samples. Use :meth:`from_serial` for a
    real photodiode/ADC on a serial port (one float per line), or pass any
    callable (e.g. a simulator) for testing.
    """

    def __init__(self, reader: Callable[[int], Sequence[float]], n_samples: int = 256) -> None:
        self.reader = reader
        self.n_samples = n_samples

    @classmethod
    def from_serial(cls, port: str, baudrate: int = 115200, n_samples: int = 256) -> "ScintillationSensor":
        import serial  # pyserial, optional dependency

        conn = serial.Serial(port, baudrate=baudrate, timeout=1)

        def read(n: int) -> list[float]:
            out: list[float] = []
            while len(out) < n:
                line = conn.readline().strip()
                if line:
                    out.append(float(line))
            return out

        return cls(read, n_samples)

    @classmethod
    def simulated(cls, rms_fraction: float = 0.01, seed: int | None = None, n_samples: int = 256) -> "ScintillationSensor":
        rng = np.random.default_rng(seed)
        return cls(lambda n: (1.0 + rng.normal(0, rms_fraction, n)).tolist(), n_samples)

    def measure(self) -> SeeingMeasurement:
        return measure_seeing(self.reader(self.n_samples))


def estimate_sigma(
    wolf_raw: float,
    r_corrected: float,
    confidence: float,
    seeing_arcsec: float | None = None,
    k: float | None = None,
) -> float:
    """1-sigma uncertainty of R combining counting, model and seeing terms in quadrature."""
    s = get_settings()
    k = s.wolf_k_factor if k is None else k
    counting = k * math.sqrt(max(wolf_raw, 0.0))
    model = r_corrected * (1.0 - min(max(confidence, 0.0), 1.0))
    seeing = r_corrected * s.seeing_sigma_per_arcsec * seeing_arcsec if seeing_arcsec else 0.0
    return float(math.sqrt(counting**2 + model**2 + seeing**2))
