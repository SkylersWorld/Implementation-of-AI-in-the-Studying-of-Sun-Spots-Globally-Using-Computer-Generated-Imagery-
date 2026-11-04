"""Telescope mount sync over a TCP socket (LX200-style command stub).

Many mounts / bridges (INDI, ASCOM Alpaca-to-LX200, SkySafari WiFi) accept
LX200 serial commands over TCP. This client sets target RA/Dec and issues a
sync (``:CM#``). Adapt :meth:`MountClient.sync` for other protocols.
"""
from __future__ import annotations

import socket
from datetime import datetime, timezone

from backend.config import get_settings


def format_ra(ra_deg: float) -> str:
    total = round((ra_deg % 360.0) / 15.0 * 3600) % (24 * 3600)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_dec(dec_deg: float) -> str:
    sign = "-" if dec_deg < 0 else "+"
    total = round(abs(dec_deg) * 3600)
    d, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{d:02d}*{m:02d}:{s:02d}"


def sun_radec(when: datetime | None = None) -> tuple[float, float]:
    """Apparent geocentric RA/Dec of the Sun in degrees."""
    from astropy.coordinates import get_sun
    from astropy.time import Time

    sun = get_sun(Time(when or datetime.now(timezone.utc)))
    return float(sun.ra.deg), float(sun.dec.deg)


class MountClient:
    def __init__(self, host: str | None = None, port: int | None = None, timeout: float | None = None) -> None:
        s = get_settings()
        self.host = host or s.mount_host
        self.port = port or s.mount_port
        self.timeout = timeout or s.mount_timeout_s
        self._sock: socket.socket | None = None

    def __enter__(self) -> "MountClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def send(self, command: str) -> str:
        if self._sock is None:
            raise ConnectionError("Mount not connected")
        self._sock.sendall(command.encode("ascii"))
        # LX200 set-commands reply with a bare '0'/'1'; others are '#'-terminated.
        data = b""
        while not data.endswith(b"#"):
            try:
                chunk = self._sock.recv(256)
            except socket.timeout:
                break
            if not chunk:
                break
            data += chunk
            if data in (b"0", b"1"):
                break
        return data.decode("ascii", errors="replace")

    def sync(self, ra_deg: float, dec_deg: float) -> str:
        """Set target coordinates then sync the mount model to them."""
        replies = [
            self.send(f":Sr{format_ra(ra_deg)}#"),
            self.send(f":Sd{format_dec(dec_deg)}#"),
            self.send(":CM#"),
        ]
        return "|".join(replies)


def sync_to_sun(when: datetime | None = None, client: MountClient | None = None) -> tuple[float, float, str]:
    ra, dec = sun_radec(when)
    client = client or MountClient()
    with client:
        reply = client.sync(ra, dec)
    return ra, dec, reply
