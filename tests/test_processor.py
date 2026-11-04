import numpy as np
from astropy.io import fits
from PIL import Image

from backend.database import Observation, SessionLocal
from backend.ingestion.processor import process_hot_folder
from backend.model.cgi import render_disk


def test_hot_folder_png_and_fits(tmp_path):
    hot, done = tmp_path / "hot", tmp_path / "done"
    hot.mkdir()
    image, *_ = render_disk(256, n_groups=2, rng=np.random.default_rng(1), noise=0.0)
    Image.fromarray((image * 255).astype(np.uint8)).save(hot / "frame.png")
    hdu = fits.PrimaryHDU((image * 60000).astype(np.float32))
    hdu.header["DATE-OBS"] = "2026-03-01T10:00:00"
    hdu.writeto(hot / "hmi_ic_45s_test.fits")

    obs = process_hot_folder(hot, done)
    assert len(obs) == 2
    assert {o.source for o in obs} == {"ground", "hmi"}
    assert not any(hot.iterdir())
    assert sorted(p.name for p in done.iterdir()) == ["frame.png", "hmi_ic_45s_test.fits"]
    with SessionLocal() as db:
        assert db.query(Observation).count() == 2


def test_fits_date_parsing():
    from backend.ingestion.loader import _parse_date

    assert _parse_date("2024-05-10T12:04:52.30").isoformat().startswith("2024-05-10T12:04:52")
    assert _parse_date("2024.05.10_12:05:15_TAI").isoformat().startswith("2024-05-10T12:05:15")
    assert _parse_date("garbage") is None
