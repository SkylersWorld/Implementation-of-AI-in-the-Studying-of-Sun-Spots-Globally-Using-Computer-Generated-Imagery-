import numpy as np
import pytest

from backend.model.cgi import render_disk
from backend.model.inference import SunspotSegmenter, count_features, detect_delta, wolf_number


def _blob(mask, cy, cx, r=2):
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1


def test_wolf_number():
    raw, corrected = wolf_number(3, 12)
    assert raw == 42
    assert corrected == pytest.approx(1.146 * 42)


def test_count_groups_and_spots():
    mask = np.zeros((200, 200), np.uint8)
    for cx in (20, 28, 36):  # one group of 3 spots
        _blob(mask, 30, cx)
    for cx in (150, 158):  # second group of 2 spots
        _blob(mask, 150, cx)
    mask[100, 100] = 1  # noise, below min area
    g, s, labels, groups = count_features(mask, min_area=4, merge_radius=6)
    assert (g, s) == (2, 5)
    assert labels.max() == 5


def test_empty_mask():
    g, s, *_ = count_features(np.zeros((32, 32)), 4, 5)
    assert (g, s) == (0, 0)


def test_delta_detection():
    mask = np.zeros((64, 64), np.uint8)
    _blob(mask, 32, 32, r=5)
    _, _, labels, _ = count_features(mask, 4, 0)
    mag = np.zeros((64, 64), np.float32)
    mag[30:33, 28:31] = 800
    assert not detect_delta(labels, mag)
    mag[31:34, 33:36] = -900
    assert detect_delta(labels, mag)
    assert not detect_delta(labels, None)


def test_classical_fallback_detects_cgi_spots(tmp_path):
    rng = np.random.default_rng(3)
    image, mask, n_groups, _ = render_disk(256, n_groups=3, rng=rng, noise=0.0)
    seg = SunspotSegmenter(weights=tmp_path / "missing.pt", device="cpu", input_size=256)
    result = seg.analyze(image)
    assert result.method == "classical"
    assert result.spots >= 1
    assert 1 <= result.groups <= n_groups + 1
    assert result.r_corrected == pytest.approx(1.146 * (10 * result.groups + result.spots))
    assert 0.0 <= result.confidence <= 1.0


def test_quiet_sun_has_zero_count(tmp_path):
    image, *_ = render_disk(256, n_groups=0, rng=np.random.default_rng(0), noise=0.0)
    seg = SunspotSegmenter(weights=tmp_path / "missing.pt", device="cpu", input_size=256)
    result = seg.analyze(image)
    assert (result.groups, result.spots) == (0, 0)
