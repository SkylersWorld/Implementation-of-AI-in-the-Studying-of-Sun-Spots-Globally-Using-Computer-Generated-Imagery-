import socket
import threading

from backend.hardware.mount import MountClient, format_dec, format_ra, sun_radec


def test_formatting():
    assert format_ra(0) == "00:00:00"
    assert format_ra(187.5) == "12:30:00"
    assert format_dec(-23.5) == "-23*30:00"
    assert format_dec(45.25) == "+45*15:00"


def test_sun_radec_in_range():
    ra, dec = sun_radec()
    assert 0 <= ra < 360 and -23.5 <= dec <= 23.5


def test_sync_against_fake_mount():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    received = []

    def serve():
        conn, _ = server.accept()
        with conn:
            buf = b""
            while True:
                data = conn.recv(256)
                if not data:
                    break
                buf += data
                while b"#" in buf:
                    cmd, buf = buf.split(b"#", 1)
                    received.append(cmd.decode())
                    conn.sendall(b" M31 EX GAL MAG 3.5 SZ178.0'#" if cmd == b":CM" else b"1")

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    with MountClient("127.0.0.1", port, timeout=2) as client:
        reply = client.sync(187.5, -23.5)
    server.close()
    assert received == [":Sr12:30:00", ":Sd-23*30:00", ":CM"]
    assert reply.startswith("1|1|")
