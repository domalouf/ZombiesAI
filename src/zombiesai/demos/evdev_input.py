"""Linux raw input: the same device counts Win32 Raw Input gives, read straight off the kernel.

`/dev/input/event*` carries exactly what the hardware reported -- `REL_X`/`REL_Y` are mouse counts, not
cursor positions, so they survive a game that captures and re-centres the pointer, and they are the unit
`realgame/uinput.py` emits back. Reading them needs no library and no message loop: open the device, read
24-byte records, decode.

Two details decide whether the timestamps are usable:

* **`EVIOCSCLOCKID` puts the device on `CLOCK_MONOTONIC`**, which is what `time.monotonic()` reads and what
  the capture thread stamps frames with. Without it the kernel reports wall-clock time, and any NTP step
  during a recording silently shifts part of your labels.
* **Key auto-repeat is dropped.** Holding W produces a stream of repeat events; treating them as presses
  would turn one press into thirty, which the `button` head would read as thirty reloads.

Requires read access to the device nodes: `sudo usermod -aG input $USER` and a re-login, or a udev rule.
See docs/linux.md.
"""

import fcntl
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

EV_SYN, EV_KEY, EV_REL, EV_MSC = 0x00, 0x01, 0x02, 0x04
REL_X, REL_Y, REL_HWHEEL, REL_WHEEL = 0x00, 0x01, 0x06, 0x08
KEY_UP, KEY_DOWN, KEY_REPEAT = 0, 1, 2
EV_REP = 0x14

# _IOW('E', 0xa0, int): tells the kernel to timestamp this device's events with CLOCK_MONOTONIC.
EVIOCSCLOCKID = 0x400445A0
CLOCK_MONOTONIC = 1

# struct input_event on 64-bit Linux: struct timeval (two longs), type, code, value.
_EVENT = struct.Struct("=qqHHi")
EVENT_SIZE = _EVENT.size

BUTTON_NAMES = {0x110: "mouse1", 0x111: "mouse2", 0x112: "mouse3", 0x113: "mouse4", 0x114: "mouse5"}
# The first 58 codes are the PC scancode order, so the table is the keyboard read left to right, top to
# bottom. These are physical positions, not letters -- which is what a game binds, and what makes the
# labels layout-independent.
_SCANCODE_ORDER = (
    "escape 1 2 3 4 5 6 7 8 9 0 minus equal backspace tab "
    "q w e r t y u i o p leftbrace rightbrace enter ctrl "
    "a s d f g h j k l semicolon apostrophe grave shift backslash "
    "z x c v b n m comma dot slash shift kpasterisk alt space capslock"
).split()
KEY_NAMES = {code: name for code, name in enumerate(_SCANCODE_ORDER, start=1)}
KEY_NAMES.update({59 + i: f"f{i + 1}" for i in range(10)})
KEY_NAMES.update({97: "ctrl", 100: "alt", 103: "up", 105: "left", 106: "right", 108: "down"})


def key_name(code: int) -> str:
    return KEY_NAMES.get(code) or BUTTON_NAMES.get(code) or f"key{code}"


def decode_events(buffer: bytes, offset: float = 0.0) -> list[dict]:
    """A read() from an event device -> log events in the recorder's format.

    `offset` is added to every kernel timestamp; it is zero once the device is on CLOCK_MONOTONIC, and the
    realtime-to-monotonic difference when the ioctl was refused. Mouse motion is reported per axis, so a
    diagonal flick arrives as two records and is summed by the caller, not here.
    """
    events = []
    for start in range(0, len(buffer) - EVENT_SIZE + 1, EVENT_SIZE):
        seconds, micros, kind, code, value = _EVENT.unpack_from(buffer, start)
        t = seconds + micros * 1e-6 + offset
        if kind == EV_REL:
            if code == REL_X:
                events.append({"t": t, "type": "mouse", "dx": int(value), "dy": 0})
            elif code == REL_Y:
                events.append({"t": t, "type": "mouse", "dx": 0, "dy": int(value)})
        elif kind == EV_KEY and value in (KEY_UP, KEY_DOWN):  # value 2 is auto-repeat, not a press
            name = key_name(code)
            events.append(
                {"t": t, "type": "button" if code in BUTTON_NAMES else "key", "code": name, "down": value == KEY_DOWN}
            )
    return events


@dataclass(frozen=True)
class InputDevice:
    path: str
    name: str
    handlers: tuple[str, ...]
    ev: int

    @property
    def is_mouse(self) -> bool:
        return "mouse" in " ".join(self.handlers) and bool(self.ev & (1 << EV_REL))

    @property
    def is_keyboard(self) -> bool:
        return "kbd" in self.handlers and bool(self.ev & (1 << EV_KEY)) and bool(self.ev & (1 << EV_REP))


def parse_devices(text: str) -> list[InputDevice]:
    """Parse /proc/bus/input/devices. Reading the text beats ioctl probing: it is one file, it names every
    device, and it can be captured from a real machine and replayed in a test."""
    devices = []
    name, handlers, ev = "", (), 0
    for line in text.splitlines() + [""]:
        line = line.strip()
        if not line:
            event = next((h for h in handlers if h.startswith("event")), None)
            if event:
                devices.append(InputDevice(f"/dev/input/{event}", name, handlers, ev))
            name, handlers, ev = "", (), 0
        elif line.startswith("N: Name="):
            name = line.partition("=")[2].strip('"')
        elif line.startswith("H: Handlers="):
            handlers = tuple(line.partition("=")[2].split())
        elif line.startswith("B: EV="):
            ev = int(line.partition("=")[2], 16)
    return devices


def find_devices(exclude: tuple[str, ...] = ()) -> list[InputDevice]:
    """Every mouse and keyboard the kernel knows about, minus any whose name contains an excluded string.

    The exclusion matters once the agent is playing: our own virtual device is an input device too, and a
    recorder that reads it back would log the agent's actions as if a human had made them.
    """
    text = Path("/proc/bus/input/devices").read_text()
    devices = [d for d in parse_devices(text) if d.is_mouse or d.is_keyboard]
    return [d for d in devices if not any(e.lower() in d.name.lower() for e in exclude)]


class EvdevInput:
    """Reads raw input from the device nodes, behind the recorder's drain()/close() interface.

    Opened read-only and never grabbed: the game must keep receiving the same input the recorder sees.
    """

    def __init__(self, devices=None, *, exclude: tuple[str, ...] = ("zombiesai",)):
        """`devices` may be InputDevice objects or plain /dev/input/event* paths; the default is every
        mouse and keyboard except our own virtual one."""
        self.devices = list(devices) if devices is not None else find_devices(exclude)
        if not self.devices:
            raise RuntimeError(
                "no mouse or keyboard found under /proc/bus/input/devices -- "
                "check permissions on /dev/input/event* (see docs/linux.md)"
            )
        self.fds: dict[int, str] = {}
        self.offset = 0.0
        self.monotonic = True
        for device in self.devices:
            path = device if isinstance(device, str) else device.path
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError as error:
                self.close()
                raise PermissionError(
                    f"cannot read {path}: add yourself to the 'input' group (see docs/linux.md)"
                ) from error
            try:
                fcntl.ioctl(fd, EVIOCSCLOCKID, struct.pack("=i", CLOCK_MONOTONIC))
            except OSError:
                # Old kernel or an odd driver: fall back to converting wall-clock stamps once, up front.
                self.monotonic = False
                self.offset = time.monotonic() - time.time()
            self.fds[fd] = path

    def drain(self, start: float = 0.0, end: float = 0.0) -> list[dict]:
        """Everything the devices reported since the last call, merged and in time order."""
        events: list[dict] = []
        for fd in self.fds:
            while True:
                try:
                    chunk = os.read(fd, EVENT_SIZE * 512)
                except BlockingIOError:
                    break
                except OSError:  # device unplugged mid-recording; keep the rest of the session
                    break
                if not chunk:
                    break
                events.extend(decode_events(chunk, self.offset))
                if len(chunk) < EVENT_SIZE * 512:
                    break
        events.sort(key=lambda e: e["t"])
        return events

    def describe(self) -> dict:
        return {
            "kind": "evdev",
            "devices": [d if isinstance(d, str) else {"path": d.path, "name": d.name} for d in self.devices],
            "monotonic_clock": self.monotonic,
        }

    def close(self) -> None:
        for fd in list(getattr(self, "fds", {})):
            os.close(fd)
        self.fds = {}
