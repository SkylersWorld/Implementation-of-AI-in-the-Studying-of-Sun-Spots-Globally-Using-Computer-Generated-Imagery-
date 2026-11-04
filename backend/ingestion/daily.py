"""Daily full-disk analysis: fetch the real SDO/HMI continuum image for a date, let the AI
count sunspot groups g and spots s on it, and render an annotated PNG.

The image is taken near local solar noon of the station (12:00 UTC − longitude/15 h).
Source order: JSOC/VSO FITS via SunPy Fido, then the Helioviewer screenshot API.
"""
from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from backend.config import get_settings
from backend.model.dataset import normalize
from backend.model.inference import count_features, get_segmenter, wolf_number

log = logging.getLogger(__name__)


@dataclass
class SolarImage:
    data: np.ndarray  # solar north up, east left, row 0 at top
    obs_time: datetime
    source: str


@dataclass
class DayAnalysis:
    groups: int
    spots: int
    wolf_raw: float
    r_ai: float
    confidence: float
    method: str
    png: bytes


def target_time(day: date, longitude: float | None = None) -> datetime:
    """UTC time of local solar noon at ``longitude`` (12:00 UTC if unknown)."""
    hours = 12.0 - (longitude or 0.0) / 15.0
    return datetime.combine(day, time(0), tzinfo=timezone.utc) + timedelta(hours=hours)


def _orient_fits(data: np.ndarray, header) -> np.ndarray:
    data = np.nan_to_num(np.asarray(data, dtype=np.float32))[::-1]  # FITS row 0 is the bottom
    if abs(((float(header.get("CROTA2", 0.0)) + 180) % 360) - 180) > 90:  # HMI is stored upside down
        data = data[::-1, ::-1]
    return np.ascontiguousarray(data)


def fetch_hmi_fits(when: datetime, window_min: int = 45) -> SolarImage:
    from astropy.io import fits
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    start, end = (when - timedelta(minutes=window_min)).replace(tzinfo=None), (when + timedelta(minutes=window_min)).replace(tzinfo=None)
    result = Fido.search(a.Time(start, end), a.Instrument("HMI"), a.Physobs("intensity"))
    if not len(result) or not len(result[0]):
        raise LookupError(f"No HMI continuum between {start} and {end}")
    from astropy.time import Time

    table = result[0]
    i = int(np.argmin(np.abs((Time(table["Start Time"]) - Time(when.replace(tzinfo=None))).sec)))
    best = table[i : i + 1]
    with tempfile.TemporaryDirectory() as tmp:
        files = Fido.fetch(best, path=str(Path(tmp) / "{file}"), progress=False)
        if not files:
            raise LookupError("HMI download failed")
        with fits.open(files[0]) as hdul:
            hdu = next(h for h in hdul if h.data is not None)
            data, header = hdu.data, hdu.header
            data = _orient_fits(data, header)
            obs = header.get("T_OBS") or header.get("DATE-OBS")
    obs_time = when
    if obs:
        try:
            obs_time = datetime.fromisoformat(str(obs).replace("_TAI", "").replace(".", "-", 2).replace("_", "T")[:19]).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return SolarImage(data, obs_time, "SDO/HMI continuum FITS (JSOC)")


def fetch_helioviewer(when: datetime, size: int = 2048) -> SolarImage:
    s = get_settings()
    params = {
        "date": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "imageScale": 2.0 * 1024 / size * 1.2,
        "layers": "[SDO,HMI,continuum,1,100]",
        "x0": 0, "y0": 0, "width": size, "height": size, "display": "true", "watermark": "false",
    }
    resp = httpx.get(s.helioviewer_url, params=params, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("L")
    return SolarImage(np.asarray(img, dtype=np.float32), when, "SDO/HMI continuum (Helioviewer)")


def fetch_day_image(day: date, longitude: float | None = None) -> SolarImage:
    when = target_time(day, longitude)
    errors = []
    for fetch in (fetch_hmi_fits, fetch_helioviewer):
        try:
            return fetch(when)
        except Exception as exc:  # network / archive errors: try the next source
            log.warning("%s failed for %s: %s", fetch.__name__, when, exc)
            errors.append(f"{fetch.__name__}: {exc}")
    raise LookupError("; ".join(errors))


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.asarray(Image.fromarray(img.astype(np.float32), mode="F").resize((size, size), Image.BILINEAR))


def intensity_ratio(img: np.ndarray, n_bins: int = 96, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Image divided by its limb-darkening profile (median per radius, from a subsample); returns ``(ratio, r/R)``."""
    disk = img > 0.1 * float(img.max() or 1.0)
    yy, xx = np.indices(img.shape, dtype=np.float32)
    area = float(disk.sum()) or 1.0
    cy, cx = float(yy[disk].mean()), float(xx[disk].mean())
    r = np.hypot(yy - cy, xx - cx) / max(np.sqrt(area / np.pi), 1.0)
    sub, rs = img[::step, ::step], r[::step, ::step]
    bins = np.minimum((rs * n_bins).astype(int), n_bins)
    centers = (np.arange(n_bins + 1) + 0.5) / n_bins
    profile = np.full(n_bins + 1, np.nan, dtype=np.float32)
    for b in range(n_bins):
        vals = sub[(bins == b) & (sub > 0.1)]
        if vals.size > 10:
            profile[b] = np.median(vals)
    ok = ~np.isnan(profile)
    background = np.interp(r, centers[ok], profile[ok]) if ok.any() else np.ones_like(img)
    ratio = np.divide(img, background, out=np.ones_like(img), where=background > 1e-3)
    return np.where(disk, ratio, 1.0).astype(np.float32), r


def count_spots_and_groups(
    ratio: np.ndarray, r: np.ndarray, penumbra: float, umbra: float, min_area: int, merge_radius: int, max_r: float = 0.96
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Spots = umbrae (each umbra in a penumbra counts; a pore without umbra counts once); groups = merged clusters."""
    inside = r < max_r
    _, n_comp, comp_labels, group_labels = count_features((ratio < penumbra) & inside, min_area, merge_radius)
    if n_comp == 0:
        return 0, 0, comp_labels, group_labels
    umbra_labels, n_umbra = ndimage.label((ratio < umbra) & inside & (comp_labels > 0), structure=np.ones((3, 3), bool))
    per_comp = np.ones(n_comp + 1, dtype=int)
    if n_umbra:
        sizes = ndimage.sum_labels(np.ones_like(umbra_labels), umbra_labels, np.arange(1, n_umbra + 1))
        owners = ndimage.maximum(comp_labels, umbra_labels, np.arange(1, n_umbra + 1)).astype(int)
        for owner, size in zip(owners, sizes):
            if size >= 2:
                per_comp[owner] += 1
        per_comp = np.maximum(per_comp - 1, 1)
    spots = int(per_comp[1:].sum())
    groups = len(np.unique(group_labels[group_labels > 0]))
    return groups, spots, comp_labels, group_labels


def analyze_disk(image: np.ndarray, size: int | None = None) -> DayAnalysis:
    """Count g and s on a full-disk continuum image and draw them."""
    s = get_settings()
    size = size or s.daily_analysis_size
    img = normalize(_resize(normalize(image), size))
    scale = size / 512
    min_area = max(2, int(round(s.min_spot_area_px * scale / 2)))
    merge = int(round(s.daily_group_radius_px * scale))
    seg = get_segmenter()
    if seg.model is not None:
        prob = _resize(seg.predict_proba(img), size)
        groups, spots, spot_labels, group_labels = count_features(prob >= s.seg_threshold, min_area, merge)
        confidence = float(prob[spot_labels > 0].mean()) if spots else 1.0
        method = "unet"
    else:
        ratio, r = intensity_ratio(img)
        groups, spots, spot_labels, group_labels = count_spots_and_groups(
            ratio, r, s.daily_contrast, s.daily_umbra_contrast, min_area, merge
        )
        depth = np.clip((1 - ratio[spot_labels > 0]) / (1 - s.daily_umbra_contrast), 0, 1)
        confidence = float(depth.mean()) if spots else 1.0
        method = "classical"
    raw, r_ai = wolf_number(groups, spots, s.wolf_k_factor)
    return DayAnalysis(groups, spots, raw, r_ai, confidence, method, render(img, spot_labels, group_labels))


def render(img: np.ndarray, spot_labels: np.ndarray, group_labels: np.ndarray, out_size: int = 1024) -> bytes:
    """Grey disk, spot outlines in red, numbered yellow circles around each group."""
    grey = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([grey] * 3, axis=-1)
    edge = (spot_labels > 0) & ~ndimage.binary_erosion(spot_labels > 0)
    rgb[ndimage.binary_dilation(edge, iterations=max(1, img.shape[0] // 1024))] = (255, 40, 40)
    pil = Image.fromarray(rgb).resize((out_size, out_size), Image.LANCZOS)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=max(12, out_size // 48))
    f = out_size / img.shape[0]
    ids = [i for i in np.unique(group_labels) if i]
    for n, gid in enumerate(sorted(ids, key=lambda i: ndimage.center_of_mass(group_labels == i)[1]), start=1):
        ys, xs = np.nonzero(group_labels == gid)
        cy, cx = ys.mean() * f, xs.mean() * f
        r = max(14.0, 0.5 * max(np.ptp(ys), np.ptp(xs)) * f + 10)
        n_spots = len(np.unique(spot_labels[group_labels == gid]))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 210, 0), width=3)
        draw.text((cx + r + 3, cy - r), f"G{n}: {n_spots}", fill=(255, 210, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def process_day(day: date, longitude: float | None = None, force: bool = False):
    """Fetch, analyse and store the AI count for ``day``; returns the ``SolarDay`` row."""
    from sqlalchemy import select

    from backend.database import SessionLocal, SolarDay

    s = get_settings()
    with SessionLocal() as db:
        row = db.scalars(select(SolarDay).where(SolarDay.date == day)).first()
        if row is not None and row.status == "done" and not force:
            return row
        if row is None:
            row = SolarDay(date=day)
            db.add(row)
        row.status, row.error = "pending", None
        db.commit()
        try:
            img = fetch_day_image(day, longitude)
            result = analyze_disk(img.data)
            folder = Path(s.daily_image_folder)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{day.isoformat()}.png"
            path.write_bytes(result.png)
            row.status, row.obs_time, row.source, row.method = "done", img.obs_time, img.source, result.method
            row.groups, row.spots, row.wolf_raw, row.r_ai = result.groups, result.spots, result.wolf_raw, result.r_ai
            row.confidence, row.image_path = result.confidence, str(path)
        except Exception as exc:
            log.exception("Daily analysis failed for %s", day)
            row.status, row.error = "failed", str(exc)[:500]
        db.commit()
        db.refresh(row)
        return row


def process_days(days: list[date], longitude: float | None = None, force: bool = False) -> None:
    for d in sorted(set(days)):
        process_day(d, longitude, force)
