"""Alert rules and CME-risk traffic light."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from backend.alerts import notifier
from backend.config import get_settings


@dataclass
class Snapshot:
    timestamp: datetime
    r_corrected: float
    delta_class: bool
    groups: int = 0
    spots: int = 0


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def drdt_per_hour(previous: Snapshot | None, current: Snapshot) -> float | None:
    if previous is None:
        return None
    hours = (_aware(current.timestamp) - _aware(previous.timestamp)).total_seconds() / 3600.0
    if hours <= 0:
        return None
    return (current.r_corrected - previous.r_corrected) / hours


def cme_risk(delta_class: bool, drdt: float | None, threshold: float | None = None) -> str:
    """green: quiet; amber: delta region OR fast dR/dt; red: both."""
    threshold = get_settings().drdt_threshold if threshold is None else threshold
    fast = drdt is not None and abs(drdt) >= threshold
    if delta_class and fast:
        return "red"
    if delta_class or fast:
        return "amber"
    return "green"


def evaluate(current: Snapshot, previous: Snapshot | None = None) -> list[str]:
    s = get_settings()
    messages: list[str] = []
    ts = _aware(current.timestamp).strftime("%Y-%m-%d %H:%M UTC")
    if current.delta_class:
        messages.append(
            f":warning: Delta-class region detected at {ts} (R={current.r_corrected:.1f}, "
            f"g={current.groups}, s={current.spots}). Elevated flare/CME risk."
        )
    rate = drdt_per_hour(previous, current)
    if rate is not None and abs(rate) >= s.drdt_threshold:
        messages.append(f":chart_with_upwards_trend: Rapid sunspot number change at {ts}: dR/dt = {rate:+.1f} /h.")
    return messages


def check_and_alert(current: Snapshot, previous: Snapshot | None = None) -> list[str]:
    messages = evaluate(current, previous)
    for msg in messages:
        notifier.broadcast(msg)
    return messages
