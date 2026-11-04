from datetime import date, datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.calibration.local import AICount, Count, compare
from backend.config import get_settings
from backend.ingestion import daily
from backend.main import app
from backend.model.cgi import render_disk


def _disk(seed=3, n_groups=3):
    img, *_ = render_disk(512, n_groups=n_groups, rng=np.random.default_rng(seed), noise=0.0)
    return img


def test_target_time_uses_local_solar_noon():
    assert daily.target_time(date(2026, 9, 7), 45.0) == datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)
    assert daily.target_time(date(2026, 9, 7)).hour == 12


def test_analyze_disk_counts_and_renders():
    quiet = daily.analyze_disk(_disk(n_groups=0), size=512)
    assert (quiet.groups, quiet.spots) == (0, 0)
    active = daily.analyze_disk(_disk(n_groups=3), size=512)
    assert active.groups >= 1 and active.spots >= active.groups
    assert active.r_ai == pytest.approx(get_settings().wolf_k_factor * (10 * active.groups + active.spots))
    assert active.png[:8] == b"\x89PNG\r\n\x1a\n"


def test_compare_calibrates_against_ai():
    counts = [Count(date(2026, 9, d), 2, 5, 1.0) for d in (1, 2)]
    ai = {date(2026, 9, 1): AICount(3, 10, 50.0), date(2026, 9, 2): AICount(3, 10, 50.0)}
    ref = {date(2026, 9, 1): (40.0, "swpc")}
    rows, summary = compare(counts, ref, ai=ai, method="none")
    assert summary.calibration_reference == "ai" and summary.k_fit == pytest.approx(2.0)
    assert rows[0].r_calibrated == pytest.approx(50.0) and rows[0].pct_diff_ai == pytest.approx(25.0)
    assert summary.mae_ai == pytest.approx(10.0)
    _, no_ai = compare(counts, ref, method="none")
    assert no_ai.calibration_reference == "international"


def test_count_submission_triggers_ai_and_serves_image(monkeypatch):
    img = _disk(n_groups=2)
    calls = []

    def fake_fetch(day, longitude=None):
        calls.append((day, longitude))
        return daily.SolarImage(img, daily.target_time(day, longitude), "test")

    monkeypatch.setattr(daily, "fetch_day_image", fake_fetch)
    monkeypatch.setattr(get_settings(), "daily_auto_analyze", True)
    monkeypatch.setattr(get_settings(), "daily_analysis_size", 512)
    with TestClient(app) as c:
        st = c.post("/stations", json={"name": "Home", "country": "Georgia", "longitude": 44.8, "k_factor": 1.2}).json()
        c.post(f"/stations/{st['id']}/counts", json={"date": "2026-09-07", "groups": 2, "spots": 6})
        assert calls == [(date(2026, 9, 7), 44.8)]
        day = c.get("/solar-days/2026-09-07").json()
        assert day["status"] == "done" and day["groups"] >= 1 and day["source"] == "test"
        resp = c.get("/solar-days/2026-09-07/image")
        assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
        row = c.get(f"/stations/{st['id']}/compare").json()["rows"][0]
        assert row["r_ai"] == pytest.approx(day["r_ai"]) and row["ai_groups"] == day["groups"]
        c.post(f"/stations/{st['id']}/counts", json={"date": "2026-09-07", "groups": 3, "spots": 6})
        assert len(calls) == 1  # already analysed


def test_failed_fetch_is_recorded(monkeypatch):
    def boom(day, longitude=None):
        raise LookupError("archive down")

    monkeypatch.setattr(daily, "fetch_day_image", boom)
    row = daily.process_day(date(2026, 9, 8))
    assert row.status == "failed" and "archive down" in row.error
    with TestClient(app) as c:
        assert c.get("/solar-days/2026-09-08/image").status_code == 404
