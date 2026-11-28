"""A one-way, lossy feed from a running loop to whatever is drawing it: Unix datagrams, newest wins.

The recorder, the gamepad bridge and the actors each know something worth drawing live -- how long the recording
has run, where the sticks are, what the policy thought of its choices -- and each runs a loop that must never
wait for a picture. So they publish, and an overlay subscribes:

* **A publisher costs one non-blocking `sendto` a message and nothing when nobody listens.** No socket at the
  path, a reader that went away, a full buffer: the message is dropped and the loop never hears of it.
* **A subscriber only ever wants the latest.** `Subscriber.poll()` drains everything that arrived and keeps the
  newest message of each `kind`, so a slow overlay skips messages rather than falling behind.
* **Stdlib only.** The live recording overlay runs under the system Python (it needs GTK, which the project's
  venv does not have), and imports this file from the checkout.

The sockets live in `$XDG_RUNTIME_DIR/zombiesai/`: `record.sock` for the recorder and the pad bridge,
`brain-60.sock` for the actor playing on X display :60, and so on (`bus_path`).
"""

import json
import os
import socket
import time
from pathlib import Path

MAX_DATAGRAM = 60_000  # well under the default Unix datagram limit; a message is a few hundred bytes


def bus_dir() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "zombiesai"


def bus_path(name: str) -> Path:
    """The socket for feed `name` ("record", or "brain" + a display like ":60" -> brain-60.sock)."""
    return bus_dir() / f"{name.replace(':', '-')}.sock"


def brain_bus(display: str) -> Path:
    """Where the actor playing on X display `display` publishes what its policy thought each step."""
    return bus_path(f"brain{display}")


class Publisher:
    """Fire-and-forget JSON datagrams to `path`. Every failure is swallowed: the drawing is optional, the loop
    publishing it is not."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sent = self.dropped = 0

    def send(self, message: dict) -> bool:
        if self.sock is None:
            return False
        try:
            data = json.dumps(message, separators=(",", ":")).encode()
            if len(data) > MAX_DATAGRAM:
                self.dropped += 1
                return False
            self.sock.sendto(data, self.path)
        except (OSError, TypeError, ValueError):  # no listener, gone, buffer full, unserialisable
            self.dropped += 1
            return False
        self.sent += 1
        return True

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None


class Subscriber:
    """Binds `path` (replacing a stale socket left by a reader that died) and keeps the newest message of each
    `kind`, with the monotonic time it arrived."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(str(self.path))
        self.inode = os.stat(self.path).st_ino
        self.sock.setblocking(False)
        self.latest: dict[str, dict] = {}
        self.arrived: dict[str, float] = {}

    def fileno(self) -> int:
        return self.sock.fileno()

    def poll(self, now: float | None = None) -> bool:
        """Drain what has arrived. True if anything did."""
        got = False
        now = time.monotonic() if now is None else now
        while True:
            try:
                data = self.sock.recv(MAX_DATAGRAM + 1)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            try:
                message = json.loads(data)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            kind = str(message.get("kind", ""))
            self.latest[kind] = message
            self.arrived[kind] = now
            got = True
        return got

    def get(self, kind: str, max_age_s: float | None = None, now: float | None = None) -> dict | None:
        """The newest `kind` message, or None if there is none or it is older than `max_age_s`."""
        message = self.latest.get(kind)
        if message is None or max_age_s is None:
            return message
        now = time.monotonic() if now is None else now
        return message if now - self.arrived[kind] <= max_age_s else None

    def close(self) -> None:
        """Let go of the socket, and remove it only if it is still ours: a reader started in this one's place
        (a viewer relaunched, the next recording's overlay) may already have bound a fresh one at the path."""
        self.sock.close()
        try:
            if os.stat(self.path).st_ino == self.inode:
                self.path.unlink()
        except OSError:
            pass
