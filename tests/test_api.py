import io
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.model.cgi import render_disk


def _png(n_groups=3, seed=0) -> bytes:
    image, *_ = render_disk(256, n_groups=n_groups, rng=np.random.default_rng(seed), noise=0.0)
    buf = io.BytesIO()
    Image.fromarray((image * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_empty_state():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/observations").json() == []
        assert c.get("/observations/latest").status_code == 404
        assert c.get("/metrics").json()["cme_risk"] == "green"


def test_create_and_list_observations():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with TestClient(app) as c:
        for i, (r, delta) in enumerate([(50, False), (80, True)]):
            resp = c.post("/observations", json={
                "timestamp": (t0 + timedelta(hours=i)).isoformat(), "groups": 3, "spots": 10,
                "wolf_raw": 40, "r_corrected": r, "sigma": 2, "confidence": 0.9, "delta_class": delta,
            })
            assert resp.status_code == 201
        rows = c.get("/observations").json()
        assert [r["r_corrected"] for r in rows] == [50, 80]
        assert c.get(f"/observations/{rows[0]['id']}").json()["r_corrected"] == 50
        m = c.get("/metrics").json()
        assert m["count"] == 2 and m["drdt_per_hour"] == 30 and m["cme_risk"] == "red"


def test_infer_png_upload():
    with TestClient(app) as c:
        resp = c.post(
            "/infer",
            files={"image": ("disk.png", _png(), "image/png")},
            data={"seeing_arcsec": "1.5", "timestamp": "2026-02-01T12:00:00Z"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["spots"] >= 1 and body["groups"] >= 1
        assert abs(body["wolf_raw"] - (10 * body["groups"] + body["spots"])) < 1e-9
        assert body["seeing_arcsec"] == 1.5
        assert body["sigma"] > 0


def test_infer_rejects_garbage():
    with TestClient(app) as c:
        resp = c.post("/infer", files={"image": ("x.png", b"not an image", "image/png")})
        assert resp.status_code == 400


def test_seeing_endpoint():
    with TestClient(app) as c:
        resp = c.post("/calibration/seeing", json={"samples": [1.0, 1.01, 0.99, 1.0] * 4})
        assert resp.status_code == 200
        assert resp.json()["seeing_arcsec"] > 0


def test_hardware_sync_unreachable(monkeypatch):
    from backend.config import get_settings

    monkeypatch.setattr(get_settings(), "mount_port", 1)
    with TestClient(app) as c:
        resp = c.post("/hardware/sync", json={"ra_deg": 10, "dec_deg": 5})
        assert resp.status_code == 502
