"""How a worker finds the learner without being told where it is: the learner announces itself on the LAN.

Every couple of seconds while it listens for workers, the learner (`FleetServer`) sends a small UDP broadcast,
a **beacon**, to port BEACON_PORT on every network it is on: where to reach it, which run it is training, the
commit and spec version a worker must match, when the run started and when the beacon was sent. A worker started
without `--learner` listens on that port until it hears one (`BeaconListener`), and goes there. Nothing pins the
address once found: a worker that loses its learner listens again, so a run continued on another PC
(`train_rl.py <checkpoint.pt> --listen`) is found the same way.

A beacon is signed -- an HMAC over all of it, keyed by the fleet's token -- and that is the whole of its
security: a worker sends the token with every request to wherever a beacon points it, so a beacon anyone on the
LAN could forge would hand them the token. The address is inside the signed part, never taken from the packet's
source, so a genuine beacon replayed from another machine still points at the real learner. The send time stops
an old beacon from being replayed for long; it is compared with a generous allowance, because a PC whose clock is
off should be told so, not left waiting in silence (`BeaconError.kind == "stale"`).

    learner                                           worker
    FleetServer.start() ── Announcer ── UDP 47861 ──► BeaconListener ── choose() ── FleetClient.point_at()
                           every 2 s,   broadcast     verify_beacon()   --run, our commit, newest
                           signed
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import socket
import struct
import threading
import time
from dataclasses import asdict, dataclass

BEACON_PORT = 47861  # UDP; the learner's HTTP is on 47860 (rl/fleet.py DEFAULT_PORT)
BEACON_INTERVAL_S = 2.0
MAGIC = b"zombiesai-learner/1"
# How far a beacon's send time may sit from the listener's clock. Replaying a genuine beacon can only point a
# worker at the real learner (the address is signed), so this need not be tight; it keeps yesterday's beacons
# out, and a PC whose clock is minutes off is told so rather than left waiting.
MAX_SKEW_S = 300.0
MAX_BEACON = 1024  # bytes: a real one is ~300


@dataclass(frozen=True)
class Beacon:
    host: str  # where the learner's HTTP listens, as seen from the network the beacon went out on
    port: int
    run: str  # the run directory's name: what `fleet_worker.py --run` picks by
    sha: str  # the learner's commit and spec: a worker can tell before hello whether it would be refused
    spec_version: str
    started: float  # when the learner started listening (unix time): the newest run wins
    t: float  # when this beacon was sent
    name: str = ""  # the learner's host name, for people reading a log

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class BeaconError(ValueError):
    """Why a packet on the beacon port was not taken. `kind`: "foreign" (not a beacon at all -- someone else's
    program on the port), "token" (a learner signing with another fleet token), "stale" (signed, but sent too
    far from now) or "malformed" (signed, but not what a learner sends)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _mac(token: str, body: bytes) -> bytes:
    return hmac.new(token.encode(), MAGIC + b"\n" + body, hashlib.sha256).hexdigest().encode()


def encode_beacon(beacon: Beacon, token: str) -> bytes:
    """`MAGIC <hex HMAC-SHA256> <JSON body>`: the magic so other programs' packets are told apart at a glance,
    the MAC over the magic and the exact body bytes."""
    if not token:
        raise ValueError("a beacon is signed with the fleet token: there is none")
    body = json.dumps(asdict(beacon), sort_keys=True, separators=(",", ":")).encode()
    return MAGIC + b" " + _mac(token, body) + b" " + body


def verify_beacon(data: bytes, token: str, *, now: float | None = None, max_skew_s: float = MAX_SKEW_S) -> Beacon:
    """The beacon in `data`, if a learner holding `token` sent it lately; BeaconError otherwise. The MAC is
    checked before the body is parsed: nothing a stranger sends is interpreted."""
    if len(data) > MAX_BEACON or not data.startswith(MAGIC + b" "):
        raise BeaconError("foreign", "not a ZombiesAI beacon")
    mac, sep, body = data[len(MAGIC) + 1:].partition(b" ")
    if not sep or len(mac) != 64:
        raise BeaconError("foreign", "not a ZombiesAI beacon")
    if not hmac.compare_digest(mac, _mac(token, body)):
        raise BeaconError("token", "a learner's beacon signed with another fleet token")
    try:
        fields = json.loads(body)
        beacon = Beacon(host=fields["host"], port=fields["port"], run=fields["run"], sha=fields["sha"],
                        spec_version=fields["spec_version"], started=fields["started"], t=fields["t"],
                        name=fields.get("name", ""))
    except (ValueError, KeyError, TypeError) as error:
        raise BeaconError("malformed", f"a signed beacon that does not parse: {error}") from None
    for key in ("host", "run", "sha", "spec_version", "name"):
        if not isinstance(getattr(beacon, key), str) or len(getattr(beacon, key)) > 255:
            raise BeaconError("malformed", f"a beacon's {key} is not a short string")
    if not beacon.host or isinstance(beacon.port, bool) or not isinstance(beacon.port, int) \
            or not 0 < beacon.port < 65536:
        raise BeaconError("malformed", f"a beacon pointing at {beacon.host!r}:{beacon.port!r}")
    for key in ("started", "t"):
        value = getattr(beacon, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            raise BeaconError("malformed", f"a beacon's {key} is not a time")
    now = time.time() if now is None else now
    if abs(now - beacon.t) > max_skew_s:
        raise BeaconError("stale", f"a beacon sent {now - beacon.t:+.0f}s from this PC's clock: an old one "
                                   "replayed, or one of the two clocks is wrong (is NTP on?)")
    return beacon


def choose(beacons: list[Beacon], *, run: str | None = None, sha: str | None = None,
           spec_version: str | None = None) -> Beacon | None:
    """Which learner to join when several are announcing: only `run` if one is named; of the rest, one on our
    commit and spec before one that would refuse us; then the newest run. A learner that will refuse is still
    chosen when it is all there is, so its hello can say why."""
    candidates = [b for b in beacons if run is None or b.run == run]
    if not candidates:
        return None
    return max(candidates, key=lambda b: ((sha is None or b.sha == sha) and
                                          (spec_version is None or b.spec_version == spec_version),
                                          b.started, b.t))


# ------------------------------------------------------------------------------------------------ the learner


SIOCGIFFLAGS, SIOCGIFADDR, SIOCGIFBRDADDR = 0x8913, 0x8915, 0x8919
IFF_UP, IFF_BROADCAST, IFF_LOOPBACK = 0x1, 0x2, 0x8


def broadcast_addresses() -> list[tuple[str, str]]:
    """(this PC's IPv4 address, that network's broadcast address) for every interface that is up and can
    broadcast: the LAN, but also a second NIC or a VM bridge. Linux ioctls, no dependency; [] where they are
    not available, and the caller falls back to 255.255.255.255 (the default route's network only)."""
    try:
        import fcntl
    except ImportError:
        return []
    out = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for _, name in socket.if_nameindex():
            request = struct.pack("256s", name.encode()[:15])
            try:
                flags = struct.unpack("H", fcntl.ioctl(s.fileno(), SIOCGIFFLAGS, request)[16:18])[0]
                if flags & IFF_LOOPBACK or not flags & IFF_UP or not flags & IFF_BROADCAST:
                    continue
                address = socket.inet_ntoa(fcntl.ioctl(s.fileno(), SIOCGIFADDR, request)[20:24])
                broadcast = socket.inet_ntoa(fcntl.ioctl(s.fileno(), SIOCGIFBRDADDR, request)[20:24])
            except OSError:  # no IPv4 address on it
                continue
            if broadcast != "0.0.0.0":
                out.append((address, broadcast))
    return out


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


class Announcer:
    """The learner's side: a daemon thread sending a fresh beacon to every target every `interval_s`.

    `make(host)` builds the beacon for one target, `host` being the address that reaches the learner from that
    target's network. Where the learner listens decides the targets: on a loopback address, only this machine
    (nothing else could reach it); on one address, that address's networks; on all of them (0.0.0.0, the usual
    `--listen :47860`), every network this PC is on, each beacon naming this PC's own address there. A send that
    fails (no network yet, a cable out) is skipped: the next tick tries again."""

    def __init__(self, make, token: str, *, bound_host: str, port: int = BEACON_PORT,
                 interval_s: float = BEACON_INTERVAL_S, targets: list[str] | None = None):
        if not token:
            raise ValueError("a beacon is signed with the fleet token: there is none")
        self.make, self.token, self.bound_host, self.port, self.interval_s = make, token, bound_host, port, interval_s
        self.fixed_targets = targets
        self.sent = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="fleet-beacon", daemon=True)

    def targets(self) -> list[tuple[str, str | None]]:
        """(where to send, the host to advertise there -- None: whatever address the kernel sends from)."""
        if self.fixed_targets is not None:
            return [(t, None if self.bound_host in ("", "0.0.0.0") else self.bound_host) for t in self.fixed_targets]
        if _is_loopback(self.bound_host):
            return [("127.0.0.1", "127.0.0.1")]
        found = broadcast_addresses()
        if self.bound_host not in ("", "0.0.0.0"):
            return [(brd, self.bound_host) for addr, brd in found if addr == self.bound_host] \
                or [("255.255.255.255", self.bound_host)]
        return [(brd, addr) for addr, brd in found] or [("255.255.255.255", None)]

    def send_once(self, targets: list[tuple[str, str | None]]) -> int:
        sent = 0
        for target, host in targets:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    s.connect((target, self.port))  # picks the source address: what the worker should dial
                    s.send(encode_beacon(self.make(host or s.getsockname()[0]), self.token))
                    sent += 1
            except OSError:
                continue
        self.sent += sent
        return sent

    def _loop(self) -> None:
        targets, refreshed = self.targets(), time.monotonic()
        while not self._stop.is_set():
            if time.monotonic() - refreshed > 30.0:  # a new DHCP lease, a cable plugged in
                targets, refreshed = self.targets(), time.monotonic()
            self.send_once(targets)
            self._stop.wait(self.interval_s)

    def start(self) -> "Announcer":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_s + 1.0)


# ------------------------------------------------------------------------------------------------ the worker


class BeaconListener:
    """The worker's side: a UDP socket on the beacon port, kept for the worker's life. `listen()` gathers the
    learners heard within a window; what was turned away is kept in `rejected` (by source address) so a worker
    can say "a learner at 10.0.0.5 signs with another token" instead of waiting in silence."""

    def __init__(self, token: str, *, port: int = BEACON_PORT, bind: str = ""):
        self.token = token
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # a second listener on the PC is fine
        self.sock.bind((bind, port))
        self.port = self.sock.getsockname()[1]
        self.rejected: dict[str, BeaconError] = {}

    def listen(self, timeout_s: float, *, settle_s: float = BEACON_INTERVAL_S + 0.5, stop=None,
               clock=time.monotonic) -> list[Beacon]:
        """Every learner heard (its latest beacon each), returning `settle_s` after the first -- long enough to
        hear the others announcing, so `choose()` sees them all -- or at `timeout_s`, or when `stop` is set."""
        heard: dict[tuple[str, int], Beacon] = {}
        deadline = clock() + timeout_s
        while not (stop is not None and stop.is_set()):
            now = clock()
            if now >= deadline:
                break
            self.sock.settimeout(min(0.5, deadline - now))
            try:
                data, (source, _) = self.sock.recvfrom(MAX_BEACON + 1)
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                break
            try:
                beacon = verify_beacon(data, self.token)
            except BeaconError as error:
                if error.kind != "foreign":
                    self.rejected[source] = error
                continue
            self.rejected.pop(source, None)
            if not heard:
                deadline = min(deadline, clock() + settle_s)
            heard[(beacon.host, beacon.port)] = beacon
        return list(heard.values())

    def close(self) -> None:
        self.sock.close()
