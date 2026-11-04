from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.calibration import reference as ref
from backend.main import app

CSV = (Path(__file__).resolve().parent.parent / "data" / "samples" / "tsu_tbilisi_2026.csv").read_bytes()


@pytest.fixture
def client():
    ref._last_sync.clear()
    with TestClient(app) as c:
        yield c


def _station(c, **kw):
    body = {"name": "TSU Observatory", "country": "Georgia", "city": "Tbilisi", "k_factor": 1.5, **kw}
    resp = c.post("/stations", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_station_crud_and_countries(client):
    st = _station(client)
    _station(client, name="Kanzelhöhe", country="Austria", city=None)
    assert client.post("/stations", json={"name": "TSU Observatory", "country": "Georgia"}).status_code == 409
    assert client.get("/countries").json() == ["Austria", "Georgia"]
    assert [s["name"] for s in client.get("/stations", params={"country": "Georgia"}).json()] == ["TSU Observatory"]
    assert client.get(f"/stations/{st['id']}").json()["n_counts"] == 0


def test_import_and_compare_reproduces_report(client):
    st = _station(client)
    resp = client.post(f"/stations/{st['id']}/counts/import", files={"file": ("tsu.csv", CSV, "text/csv")})
    assert resp.json() == {"imported": 9}
    data = client.get(f"/stations/{st['id']}/compare", params={"interpolation": "none"}).json()
    s = data["summary"]
    assert round(s["mean_r_local"], 2) == 83.14 and round(s["std_r_local"], 2) == 38.16
    assert round(s["k_fit"], 3) == 1.455
    first = data["rows"][0]
    assert first["r_international"] == 112 and first["reference_source"] == "manual"
    assert round(first["pct_diff_local"], 2) == round((35 - 112) / 112 * 100, 2)

    day = client.get(f"/stations/{st['id']}/compare", params={"start": "2026-08-22", "end": "2026-08-22"}).json()["rows"]
    assert len(day) == 1 and not day[0]["observed"] and round(day[0]["r_local"], 2) == 67.33


def test_manual_count_and_reference(client):
    st = _station(client)
    r = client.post(f"/stations/{st['id']}/counts", json={"date": "2026-10-01", "groups": 3, "spots": 9, "international_r": 50})
    assert r.status_code == 201 and r.json()["k"] is None
    client.post(f"/stations/{st['id']}/counts", json={"date": "2026-10-01", "groups": 4, "spots": 9})
    counts = client.get(f"/stations/{st['id']}/counts").json()
    assert len(counts) == 1 and counts[0]["groups"] == 4
    row = client.get(f"/stations/{st['id']}/compare").json()["rows"][0]
    assert row["r_local"] == 1.5 * 49 and row["r_international"] == 50
    assert client.put("/reference/2026-10-01", json={"value": 60}).json()["value"] == 60
    assert client.get("/reference", params={"source": "manual"}).json()[0]["value"] == 60
    assert client.delete("/reference/2026-10-01").status_code == 204
    assert client.delete(f"/stations/{st['id']}/counts/2026-10-01").status_code == 204


def test_bad_csv(client):
    st = _station(client)
    resp = client.post(f"/stations/{st['id']}/counts/import", files={"file": ("x.csv", b"day,g\n1,2\n", "text/csv")})
    assert resp.status_code == 400 and "missing" in resp.json()["detail"]


def test_reference_priority_and_fallback(client, monkeypatch):
    from backend.config import get_settings

    monkeypatch.setattr(get_settings(), "silso_enabled", True)

    def fake_get(url, **kw):
        if "sidc" in url:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json=[{"Obsdate": "2026-09-16T00:00:00", "swpc_ssn": 23},
                                         {"Obsdate": "2026-09-17T00:00:00", "swpc_ssn": 11}],
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(ref.httpx, "get", fake_get)
    client.put("/reference/2026-09-16", json={"value": 114})
    rows = {r["date"]: r for r in client.get("/reference").json()}
    assert rows["2026-09-16"] == {"date": "2026-09-16", "value": 114, "source": "manual"}
    assert rows["2026-09-17"]["source"] == "swpc"
    status = client.post("/reference/refresh").json()
    assert status["silso"].startswith("unavailable") and status["swpc"].startswith("ok")
