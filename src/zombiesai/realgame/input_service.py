"""One virtual mouse and keyboard for the whole session, owned by a small background service.

Destroying a uinput device while World at War has focus unplugs a mouse under the compositor, and the first
live runs showed what that costs: after quitting the player, the human's own clicks jumped the view -- the
same unconfined-pointer symptom windowed mode had, most likely the pointer lock being dropped with the
device. So the device is not the player's to create and destroy. This service creates it once and keeps it
until the desktop session ends (or `stop()`), and each player run borrows it over a Unix socket.

The service never sends anything of its own accord except one thing: when a client disconnects -- finished,
killed, crashed -- it releases every key and button that client left down. A player that dies mid-press must
not leave W held in the game.

Wire format, one message per event: struct "=Bii" (op, a, b). op 1 = key (a = evdev code, b = 1 down / 0 up),
2 = move (a = dx, b = dy), 3 = sync.
"""

import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from zombiesai.realgame.uinput import DEVICE_NAME, KEY_CODES, UinputDevice, code_for

_MESSAGE = struct.Struct("=Bii")
OP_KEY, OP_MOVE, OP_SYNC = 1, 2, 3


def socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime) / "zombiesai-input.sock"


def serve(path: Path | None = None, device=None, *, say=print) -> None:
    """Run the service until killed. `device` defaults to a real uinput device registering every key the
    decoder knows, so any bindings a player run uses are already there."""
    path = Path(path or socket_path())
    device = device or UinputDevice(KEY_CODES.keys())
    if path.exists():
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path, 0o600)
    server.listen(1)
    say(f"virtual input {DEVICE_NAME!r} ready at {path}")
    try:
        while True:
            client, _ = server.accept()
            with client:
                handle(client, device)
    finally:
        server.close()
        path.unlink(missing_ok=True)
        device.close()


def handle(client, device) -> None:
    """Relay one client's events to the device; on disconnect, release whatever it left held."""
    held: set[int] = set()
    buffer = b""
    try:
        while chunk := client.recv(4096):
            buffer += chunk
            usable = len(buffer) - len(buffer) % _MESSAGE.size
            for offset in range(0, usable, _MESSAGE.size):
                op, a, b = _MESSAGE.unpack_from(buffer, offset)
                if op == OP_KEY:
                    device.key_code(a, bool(b))
                    held.add(a) if b else held.discard(a)
                elif op == OP_MOVE:
                    device.move(a, b)
                elif op == OP_SYNC:
                    device.sync()
            buffer = buffer[usable:]
    except OSError:
        pass
    finally:
        for code in sorted(held):
            device.key_code(code, False)
        if held:
            device.sync()


class RemoteSink:
    """The dispatcher's sink, forwarding to the service. `close()` only hangs up: the device stays."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or socket_path())
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(str(self.path))

    def _send(self, op: int, a: int, b: int) -> None:
        self.sock.sendall(_MESSAGE.pack(op, a, b))

    def key(self, code: str, down: bool, t: float = 0.0) -> None:
        self._send(OP_KEY, code_for(code), 1 if down else 0)

    def move(self, dx: int, dy: int, t: float = 0.0) -> None:
        self._send(OP_MOVE, int(dx), int(dy))

    def sync(self) -> None:
        self._send(OP_SYNC, 0, 0)

    def describe(self) -> dict:
        return {"kind": "uinput-service", "socket": str(self.path), "name": DEVICE_NAME}

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def running(path: Path | None = None) -> bool:
    path = Path(path or socket_path())
    if not path.exists():
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(path))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def ensure_running(path: Path | None = None, timeout_s: float = 5.0) -> Path:
    """Start the service detached if it is not already up, and wait for its socket. It outlives the caller
    on purpose: that is the whole point."""
    path = Path(path or socket_path())
    if running(path):
        return path
    subprocess.Popen(
        [sys.executable, "-m", "zombiesai.realgame.input_service"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if running(path):
            return path
        time.sleep(0.1)
    raise RuntimeError(f"the virtual input service did not come up at {path}")


def stop(path: Path | None = None) -> bool:
    """Stop a running service, which destroys the device -- do it with the game closed."""
    import signal

    path = Path(path or socket_path())
    result = subprocess.run(["pgrep", "-f", "zombiesai.realgame.input_service"], capture_output=True, text=True)
    pids = [int(p) for p in result.stdout.split() if int(p) != os.getpid()]
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    return bool(pids)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    serve()
