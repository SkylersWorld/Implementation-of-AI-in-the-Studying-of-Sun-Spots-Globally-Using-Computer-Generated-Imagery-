import json
from datetime import datetime, timedelta, timezone

import httpx

from backend.alerts import notifier
from backend.alerts.rules import Snapshot, cme_risk, drdt_per_hour, evaluate
from backend.config import get_settings

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_cme_risk_levels():
    assert cme_risk(False, 1.0, 20) == "green"
    assert cme_risk(True, None, 20) == "amber"
    assert cme_risk(False, -25.0, 20) == "amber"
    assert cme_risk(True, 30.0, 20) == "red"


def test_drdt_and_rules():
    prev = Snapshot(T0, 50.0, False)
    cur = Snapshot(T0 + timedelta(minutes=30), 70.0, True, 3, 10)
    assert drdt_per_hour(prev, cur) == 40.0
    msgs = evaluate(cur, prev)
    assert len(msgs) == 2
    assert "Delta-class" in msgs[0] and "dR/dt" in msgs[1]
    assert evaluate(Snapshot(T0, 50, False), None) == []


def test_slack_and_twilio_payloads(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "slack_webhook_url", "https://hooks.slack.test/abc")
    monkeypatch.setattr(s, "twilio_account_sid", "AC123")
    monkeypatch.setattr(s, "twilio_auth_token", "tok")
    monkeypatch.setattr(s, "twilio_from_number", "+100")
    monkeypatch.setattr(s, "twilio_to_number", "+200")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert notifier.send_slack("hello", client)
    assert notifier.send_sms("hi", client)
    assert json.loads(seen[0].content) == {"text": "hello"}
    assert "/Accounts/AC123/Messages.json" in str(seen[1].url)
    assert b"Body=hi" in seen[1].content
    assert seen[1].headers["authorization"].startswith("Basic ")


def test_channels_disabled_without_config():
    assert notifier.send_slack("x") is False
    assert notifier.send_sms("x") is False
