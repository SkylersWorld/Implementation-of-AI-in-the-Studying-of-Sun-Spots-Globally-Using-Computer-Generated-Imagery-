"""Sunspot segmentation inference, group/spot counting and Wolf number.

R = k * (10 g + s), with k = 1.146 by default (configurable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from backend.config import get_settings
from backend.model.dataset import normalize
from backend.model.unet import UNet

log = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    groups: int
    spots: int
    wolf_raw: float
    r_corrected: float
    confidence: float
    delta_class: bool
    mask: np.ndarray = field(repr=False)
    probability: np.ndarray = field(repr=False)
    method: str = "unet"


def wolf_number(groups: int, spots: int, k: float = 1.146) -> tuple[float, float]:
    """Return ``(raw, corrected)`` Wolf numbers."""
    raw = float(10 * groups + spots)
    return raw, k * raw


def count_features(mask: np.ndarray, min_area: int = 4, merge_radius: int = 12) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Count sunspots ``s`` and sunspot groups ``g`` in a binary mask.

    Spots are 8-connected components of at least ``min_area`` pixels. Groups are
    formed by merging spots whose dilated footprints (``merge_radius`` px) touch.
    Returns ``(groups, spots, spot_labels, group_labels)``.
    """
    structure = np.ones((3, 3), dtype=bool)
    labels, n = ndimage.label(mask > 0, structure=structure)
    if n:
        areas = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        keep = np.isin(labels, np.flatnonzero(areas >= min_area) + 1)
        labels, n = ndimage.label(keep, structure=structure)
    if n == 0:
        empty = np.zeros_like(labels)
        return 0, 0, empty, empty

    if merge_radius > 0:
        yy, xx = np.ogrid[-merge_radius : merge_radius + 1, -merge_radius : merge_radius + 1]
        disk = xx**2 + yy**2 <= merge_radius**2
        grown = ndimage.binary_dilation(labels > 0, structure=disk)
    else:
        grown = labels > 0
    group_regions, _ = ndimage.label(grown, structure=structure)
    group_labels = np.where(labels > 0, group_regions, 0)
    groups = len(np.unique(group_labels[group_labels > 0]))
    return groups, n, labels, group_labels


def detect_delta(spot_labels: np.ndarray, magnetogram: np.ndarray | None, b_threshold: float = 100.0, min_px: int = 2) -> bool:
    """Mount Wilson delta configuration: opposite-polarity umbrae in one penumbra.

    A spot component (shared penumbra) containing at least ``min_px`` pixels of
    each polarity above ``b_threshold`` Gauss is flagged. Without a co-aligned
    magnetogram the classification cannot be made and ``False`` is returned.
    """
    if magnetogram is None or spot_labels.max() == 0:
        return False
    if magnetogram.shape != spot_labels.shape:
        t = torch.from_numpy(magnetogram.astype(np.float32))[None, None]
        magnetogram = F.interpolate(t, size=spot_labels.shape, mode="nearest")[0, 0].numpy()
    idx = np.arange(1, spot_labels.max() + 1)
    pos = ndimage.sum_labels(magnetogram > b_threshold, spot_labels, index=idx)
    neg = ndimage.sum_labels(magnetogram < -b_threshold, spot_labels, index=idx)
    return bool(np.any((pos >= min_px) & (neg >= min_px)))


def disk_mask(image: np.ndarray, frac: float = 0.1) -> np.ndarray:
    return image > frac * float(image.max() if image.size else 0.0)


def radial_background(image: np.ndarray, disk: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Limb-darkening background from the median radial profile; returns ``(background, r/R)``."""
    yy, xx = np.indices(image.shape)
    area = disk.sum()
    cy, cx = (yy[disk].mean(), xx[disk].mean()) if area else ((image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2)
    radius = max(np.sqrt(area / np.pi), 1.0)
    r = np.hypot(yy - cy, xx - cx) / radius
    bins = np.minimum((r * n_bins).astype(int), n_bins)
    profile = np.ones(n_bins + 1, dtype=np.float32)
    centers = (np.arange(n_bins + 1) + 0.5) / n_bins
    valid = np.zeros(n_bins + 1, dtype=bool)
    for b in np.unique(bins[disk]):
        profile[b] = np.median(image[disk & (bins == b)])
        valid[b] = True
    if valid.any():
        profile = np.interp(centers, centers[valid], profile[valid]).astype(np.float32)
    return profile[bins], r


def classical_probability(image: np.ndarray, contrast: float = 0.85, max_r: float = 0.97) -> np.ndarray:
    """Limb-darkening-aware threshold fallback when no trained weights exist."""
    disk = disk_mask(image)
    background, r = radial_background(image, disk)
    ratio = np.divide(image, background, out=np.ones_like(image), where=background > 1e-3)
    prob = np.clip((1.0 - ratio) / (2 * (1.0 - contrast)), 0.0, 1.0)
    return np.where(disk & (r < max_r), prob, 0.0).astype(np.float32)


class SunspotSegmenter:
    """Wraps a trained U-Net checkpoint; falls back to classical thresholding."""

    def __init__(self, weights: Path | str | None = None, device: str | None = None, input_size: int | None = None) -> None:
        s = get_settings()
        self.settings = s
        self.input_size = input_size or s.model_input_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: UNet | None = None
        path = Path(weights) if weights else s.model_weights
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
            self.model = UNet(**config).to(self.device).eval()
            self.model.load_state_dict(state)
            self.input_size = int(ckpt.get("input_size", self.input_size)) if isinstance(ckpt, dict) else self.input_size
            log.info("Loaded U-Net weights from %s", path)
        else:
            log.warning("No weights at %s; using classical threshold fallback", path)

    def _resize(self, arr: np.ndarray, size: int, mode: str = "bilinear") -> np.ndarray:
        t = torch.from_numpy(arr.astype(np.float32))[None, None]
        kwargs = {"align_corners": False, "antialias": True} if mode == "bilinear" else {}
        return F.interpolate(t, size=(size, size), mode=mode, **kwargs)[0, 0].numpy()

    @torch.no_grad()
    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        img = self._resize(normalize(image), self.input_size)
        if self.model is None:
            return classical_probability(img)
        x = torch.from_numpy(img)[None, None].to(self.device)
        return torch.sigmoid(self.model(x))[0, 0].cpu().numpy()

    def analyze(self, image: np.ndarray, magnetogram: np.ndarray | None = None) -> InferenceResult:
        s = self.settings
        prob = self.predict_proba(image)
        mask = prob >= s.seg_threshold
        groups, spots, spot_labels, _ = count_features(mask, s.min_spot_area_px, s.group_merge_radius_px)
        raw, corrected = wolf_number(groups, spots, s.wolf_k_factor)
        if spots:
            confidence = float(prob[spot_labels > 0].mean())
        else:
            disk = disk_mask(self._resize(normalize(image), self.input_size))
            confidence = float(1.0 - prob[disk].mean()) if disk.any() else 0.0
        return InferenceResult(
            groups=groups,
            spots=spots,
            wolf_raw=raw,
            r_corrected=corrected,
            confidence=confidence,
            delta_class=detect_delta(spot_labels, magnetogram),
            mask=mask,
            probability=prob,
            method="unet" if self.model is not None else "classical",
        )


_segmenter: SunspotSegmenter | None = None


def get_segmenter() -> SunspotSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = SunspotSegmenter()
    return _segmenter


def reset_segmenter() -> None:
    global _segmenter
    _segmenter = None
