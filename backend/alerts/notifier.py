"""Outbound notification channels (Slack incoming webhook, Twilio SMS)."""
from __future__ import annotations

import logging

import httpx

from backend.config import get_settings

log = logging.getLogger(__name__)
TWILIO_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


def send_slack(text: str, client: httpx.Client | None = None) -> bool:
    url = get_settings().slack_webhook_url
    if not url:
        return False
    try:
        resp = (client or httpx).post(url, json={"text": text}, timeout=10)
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        log.error("Slack alert failed: %s", exc)
        return False


def send_sms(text: str, client: httpx.Client | None = None) -> bool:
    s = get_settings()
    if not all([s.twilio_account_sid, s.twilio_auth_token, s.twilio_from_number, s.twilio_to_number]):
        return False
    try:
        resp = (client or httpx).post(
            TWILIO_URL.format(sid=s.twilio_account_sid),
            data={"From": s.twilio_from_number, "To": s.twilio_to_number, "Body": text[:1600]},
            auth=(s.twilio_account_sid, s.twilio_auth_token),
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        log.error("Twilio alert failed: %s", exc)
        return False


def broadcast(text: str) -> dict[str, bool]:
    return {"slack": send_slack(text), "sms": send_sms(text)}
