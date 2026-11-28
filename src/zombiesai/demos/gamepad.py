"""Record a demo with a gamepad: the pad drives a virtual mouse and keyboard, and the recording logs exactly that.

The policy's actions are mouse-style -- a turn in degrees per decision, keys held or not -- so a stick has to
become those somewhere. Doing it here, rather than letting the game read the pad natively, keeps every label
exact: the game is sent mouse counts, the recorder logs the very counts it was sent, and `--counts-per-degree`
turns them into degrees as it does for a real mouse. No model of the game's own stick response, no second
calibration, and the game's controller aim assist never enters the data.

    # Which pads are plugged in, and what each control is called (nothing is sent to the game):
    uv run python -m zombiesai.demos.gamepad --watch

    # Record with one (scripts/record_demo.py; record_waw.sh and record_lan.sh pass it through):
    scripts/record_lan.sh 20 "pad, camping the help room" --pad

How a pad becomes input (all of it set in `configs/pad_bindings.json`, which every recording stamps into
its clip.json under `gamepad`):

* **The right stick looks.** A radial deadzone, a response curve (`exponent`: 2 is gentle near the centre and
  full speed at the rim), and a top turn rate in degrees a second, times the recording's counts per degree: a
  stick at full tilt for one second turns `max_yaw_deg_s` degrees, whatever the mouse sensitivity. Sub-count
  remainders are carried to the next tick, so slow aim is not rounded away.
* **The left stick walks**, digitally, since the action space's forward and strafe are: past `press` it
  holds W/A/S/D, back under `release` it lets go (the gap stops a stick resting near the line from chattering).
* **Triggers and buttons press keys**, named the way `configs/waw_bindings.json` names them ("mouse1", "r",
  "shift"), so the recorder's bindings turn them into controls with nothing new to learn. A pad button may
  also press the mark key (F8), so a player on the couch can still flag a menu.
* **`toggles`** give console-style sprint: a click of L3 holds shift until the next click or until the stick
  stops pushing forward.

The bridge runs on its own thread at `rate_hz` (250 by default), on absolute monotonic deadlines like the
recorder, and never catches up: a late tick is simply the next tick. It grabs the pad (EVIOCGRAB) so the game
cannot also read it and turn twice. A grab covers one evdev node, not the pad's hidraw node or its sibling
devices, so the launchers also start the game with every node of every pad masked (`pad_nodes`, passed to
`realgame.instances.offline_wrapper`), and the recorder warns when a Wine process holds one anyway
(`game_holders`: a pad plugged in after the game started). The virtual
device is `zombiesai-pad-bridge`, which `EvdevInput` skips like every `zombiesai` device: its events reach the
log straight from the bridge (`PadInput`), each exactly once, stamped with the monotonic time it was sent.

A pad that is unplugged mid-recording costs nothing but itself: every key the bridge held is released (and
logged released), the mouse and keyboard carry on, and the pad is picked up again if it comes back.

The pad's state is published for the live recording overlay (viz/live_bus.py, `record.sock`, kind "pad").
"""

import argparse
import errno
import fcntl
import json
import math
import os
import stat
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from zombiesai.demos.evdev_input import (
    _EVENT,
    BUTTON_NAMES,
    EV_KEY,
    EV_SYN,
    EVENT_SIZE,
    KEY_DOWN,
    KEY_UP,
    WHEEL_NAMES,
    EvdevInput,
    find_devices,
    key_name,
)

EV_ABS = 0x03
BRIDGE_NAME = "zombiesai-pad-bridge"
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "pad_bindings.json"

ABS_CODES = {
    "ABS_X": 0x00, "ABS_Y": 0x01, "ABS_Z": 0x02, "ABS_RX": 0x03, "ABS_RY": 0x04, "ABS_RZ": 0x05,
    "ABS_THROTTLE": 0x06, "ABS_RUDDER": 0x07, "ABS_WHEEL": 0x08, "ABS_GAS": 0x09, "ABS_BRAKE": 0x0A,
    "ABS_HAT0X": 0x10, "ABS_HAT0Y": 0x11,
}
ABS_HAT0X, ABS_HAT0Y = ABS_CODES["ABS_HAT0X"], ABS_CODES["ABS_HAT0Y"]
BTN_TRIGGER = 0x120  # the first joystick button: generic pads with no gamepad mapping start here
BTN_SOUTH = 0x130
# The kernel's gamepad codes (Documentation/input/gamepad.rst) by position, not by printed letter -- which
# is how one config serves an Xbox pad and a DualSense. Drivers disagree about north and west, though: xpad
# reports the Xbox X button (the left one) as 0x133, which the spec calls north. `--watch` shows which.
PAD_BUTTONS = {
    0x130: "south", 0x131: "east", 0x132: "c", 0x133: "north", 0x134: "west", 0x135: "z",
    0x136: "lb", 0x137: "rb", 0x138: "lt", 0x139: "rt", 0x13A: "select", 0x13B: "start", 0x13C: "mode",
    0x13D: "l3", 0x13E: "r3",
    0x220: "dpad_up", 0x221: "dpad_down", 0x222: "dpad_left", 0x223: "dpad_right",
}
# Names the bridge makes up from axes, so a config binds them like any button.
STICK_BUTTONS = ("left_stick_up", "left_stick_down", "left_stick_left", "left_stick_right")
PSEUDO_BUTTONS = ("lt", "rt", "dpad_up", "dpad_down", "dpad_left", "dpad_right") + STICK_BUTTONS

# ioctls. EVIOCGABS(abs) = _IOR('E', 0x40 + abs, struct input_absinfo): six 32-bit ints.
_ABSINFO = struct.Struct("=6i")
EVIOCGRAB = 0x40044590  # _IOW('E', 0x90, int)


def EVIOCGABS(code: int) -> int:
    return (2 << 30) | (_ABSINFO.size << 16) | (ord("E") << 8) | (0x40 + code)


def button_name(code: int) -> str:
    return PAD_BUTTONS.get(code) or f"btn_{code:#x}"


class PadError(RuntimeError):
    """No usable pad, or no way to drive the virtual device: said before anyone has played a minute."""


class PadLost(OSError):
    """The pad went away (unplugged, out of battery)."""


# ------------------------------------------------------------------------------------------------ config


@dataclass(frozen=True)
class LookConfig:
    deadzone: float = 0.12
    exponent: float = 2.0
    max_yaw_deg_s: float = 240.0
    max_pitch_deg_s: float = 80.0
    invert_pitch: bool = False


@dataclass(frozen=True)
class Threshold:
    press: float
    release: float


@dataclass(frozen=True)
class PadConfig:
    rate_hz: float = 250.0
    look: LookConfig = field(default_factory=LookConfig)
    move_deadzone: float = 0.2
    move: Threshold = Threshold(0.5, 0.4)
    trigger: Threshold = Threshold(0.35, 0.25)
    axes: dict | str = "auto"
    buttons: dict[str, str] = field(default_factory=dict)
    toggles: dict[str, str] = field(default_factory=dict)
    publish_hz: float = 60.0

    @classmethod
    def from_dict(cls, raw: dict) -> "PadConfig":
        look = LookConfig(**raw.get("look", {}))
        move = raw.get("move", {})
        trig = raw.get("trigger", {})
        config = cls(
            rate_hz=float(raw.get("rate_hz", 250.0)),
            look=look,
            move_deadzone=float(move.get("deadzone", 0.2)),
            move=Threshold(float(move.get("press", 0.5)), float(move.get("release", 0.4))),
            trigger=Threshold(float(trig.get("press", 0.35)), float(trig.get("release", 0.25))),
            axes=raw.get("axes", "auto"),
            buttons={str(k).lower(): str(v).lower() for k, v in raw.get("buttons", {}).items() if v},
            toggles={str(k).lower(): str(v).lower() for k, v in raw.get("toggles", {}).items()},
            publish_hz=float(raw.get("publish_hz", 60.0)),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PadConfig":
        path = Path(path) if path else DEFAULT_CONFIG
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise PadError(f"cannot read the pad config {path}: {error}") from None
        return cls.from_dict(raw)

    def validate(self) -> None:
        from zombiesai.realgame.uinput import code_for

        wheel = {name for name in WHEEL_NAMES.values()}
        for button, code in self.buttons.items():
            if code in wheel:
                raise PadError(f"pad button {button!r} -> {code!r}: the virtual device has no wheel; bind a key")
            try:
                code_for(code)
            except KeyError as error:
                raise PadError(f"pad button {button!r}: {error.args[0]}") from None
        for button, until in self.toggles.items():
            if button not in self.buttons:
                raise PadError(f"toggle {button!r} is not bound to anything in 'buttons'")
            if until and until not in PSEUDO_BUTTONS + tuple(PAD_BUTTONS.values()):
                raise PadError(f"toggle {button!r}: {until!r} is not a pad button or stick direction")
        if not 0 <= self.look.deadzone < 1 or not 0 <= self.move_deadzone < 1:
            raise PadError("deadzones must be in [0, 1)")
        if self.look.exponent <= 0 or self.rate_hz <= 0:
            raise PadError("look.exponent and rate_hz must be positive")

    def as_dict(self) -> dict:
        """What a recording is stamped with: everything that decided how the pad became counts and keys."""
        return {
            "rate_hz": self.rate_hz,
            "look": dict(vars(self.look)),
            "move": {"deadzone": self.move_deadzone, "press": self.move.press, "release": self.move.release},
            "trigger": {"press": self.trigger.press, "release": self.trigger.release},
            "axes": self.axes,
            "buttons": dict(self.buttons),
            "toggles": dict(self.toggles),
            "publish_hz": self.publish_hz,
        }

    def spec_warnings(self, decision_hz: float = 15.0) -> list[str]:
        """Turn rates the action space cannot hold at full tilt: those steps get clamped and flagged."""
        from zombiesai import spec

        out = []
        for name, rate, limit in (("max_yaw_deg_s", self.look.max_yaw_deg_s, max(spec.YAW_BINS_DEG)),
                                  ("max_pitch_deg_s", self.look.max_pitch_deg_s, max(spec.PITCH_BINS_DEG))):
            if rate / decision_hz > limit:
                out.append(f"look.{name} {rate:g} is {rate / decision_hz:.1f} deg a decision at full tilt; the "
                           f"widest bin is {limit:g}, so full-tilt steps are clamped and flagged "
                           f"(keep it under {limit * decision_hz:g})")
        return out


# ------------------------------------------------------------------------------------------------ the maths


def normalize_stick(raw: int, minimum: int, maximum: int) -> float:
    """An axis's raw value -> [-1, 1] about its centre, as the device reports its own range."""
    half = (maximum - minimum) / 2
    if half <= 0:
        return 0.0
    return max(-1.0, min(1.0, (raw - (minimum + maximum) / 2) / half))


def normalize_trigger(raw: int, minimum: int, maximum: int) -> float:
    span = maximum - minimum
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (raw - minimum) / span))


def shape_stick(x: float, y: float, deadzone: float, exponent: float = 1.0) -> tuple[float, float]:
    """A radial deadzone, the remaining travel rescaled to [0, 1] and raised to `exponent`, direction kept.

    Radial rather than per axis: a per-axis deadzone snaps a slow diagonal onto the axes, which is exactly
    the small aim correction the half-degree bins are there to record."""
    r = math.hypot(x, y)
    if r <= deadzone or r == 0.0:
        return 0.0, 0.0
    m = ((min(r, 1.0) - deadzone) / (1.0 - deadzone)) ** exponent
    return x / r * m, y / r * m


class LookIntegrator:
    """The right stick -> whole mouse counts per tick, with the fraction carried, so nothing is rounded away:
    a stick held at a rate of 0.3 counts a tick still sends a count every third or fourth tick."""

    def __init__(self, look: LookConfig, counts_per_degree: float):
        self.look = look
        self.cpd = counts_per_degree
        self.carry = [0.0, 0.0]

    def step(self, x: float, y: float, dt: float) -> tuple[int, int]:
        """`x` right, `y` up, both already shaped. Returns (dx, dy) counts: dy positive is down, as a mouse."""
        sx, sy = shape_stick(x, y, self.look.deadzone, self.look.exponent)
        up = -1.0 if self.look.invert_pitch else 1.0
        want = (sx * self.look.max_yaw_deg_s * dt * self.cpd, -up * sy * self.look.max_pitch_deg_s * dt * self.cpd)
        out = []
        for i, w in enumerate(want):
            total = self.carry[i] + w
            whole = math.trunc(total)
            self.carry[i] = total - whole
            out.append(int(whole))
        if not sx:
            self.carry[0] = 0.0  # stick back in the deadzone: a leftover fraction must not drift the view later
        if not sy:
            self.carry[1] = 0.0
        return out[0], out[1]


class Hysteresis:
    def __init__(self, threshold: Threshold):
        self.threshold = threshold
        self.on = False

    def feed(self, value: float) -> bool:
        self.on = value >= (self.threshold.release if self.on else self.threshold.press)
        return self.on


# ------------------------------------------------------------------------------------------------ the device


@dataclass(frozen=True)
class AbsInfo:
    value: int
    minimum: int
    maximum: int
    fuzz: int = 0
    flat: int = 0
    resolution: int = 0


@dataclass(frozen=True)
class PadInfo:
    path: str
    name: str
    abs_codes: frozenset[int]
    key_codes: frozenset[int]


def parse_bitmap(text: str) -> frozenset[int]:
    """A /proc/bus/input/devices bitmap ("B: KEY=..."): space-separated hex longs, most significant first."""
    bits = set()
    words = text.split()
    for j, word in enumerate(reversed(words)):
        value = int(word, 16)
        i = 0
        while value:
            if value & 1:
                bits.add(j * 64 + i)
            value >>= 1
            i += 1
    return frozenset(bits)


def parse_pads(text: str) -> list[PadInfo]:
    """Every gamepad in a /proc/bus/input/devices listing: absolute X/Y axes and either the gamepad buttons or
    the generic joystick ones. LED controllers and the like that only claim a js handler have neither."""
    pads = []
    block: dict[str, str] = {}
    for line in text.splitlines() + [""]:
        line = line.strip()
        if not line:
            handlers = block.get("H", "").split()
            event = next((h for h in handlers if h.startswith("event")), None)
            ev = int(block.get("EV", "0"), 16)
            if event and ev & (1 << EV_ABS) and ev & (1 << EV_KEY):
                abs_codes = parse_bitmap(block.get("ABS", ""))
                key_codes = parse_bitmap(block.get("KEY", ""))
                has_buttons = BTN_SOUTH in key_codes or BTN_TRIGGER in key_codes
                name = block.get("N", "")
                if has_buttons and {0, 1} <= abs_codes and "zombiesai" not in name.lower():
                    pads.append(PadInfo(f"/dev/input/{event}", name, abs_codes, key_codes))
            block = {}
        elif line.startswith("N: Name="):
            block["N"] = line.partition("=")[2].strip('"')
        elif line.startswith("H: Handlers="):
            block["H"] = line.partition("=")[2]
        elif line.startswith("B: "):
            key, _, value = line[3:].partition("=")
            block[key] = value
    return pads


def find_pads() -> list[PadInfo]:
    return parse_pads(Path("/proc/bus/input/devices").read_text())


# Device nodes a game could read a pad through, by the name sysfs lists them under (/dev/input/<name>).
INPUT_NODE_PREFIXES = ("event", "js", "mouse")


def pad_nodes(pads: list[PadInfo] | None = None, sys_root: str | Path = "/sys") -> list[str]:
    """Every device node a game could read these pads through, not only the event node the bridge grabs.

    EVIOCGRAB covers one evdev node. The device behind a pad has more: its hidraw node, which Proton's winebus
    and SDL open past any grab (a DualSense was read that way on 2026-11-27, with PROTON_ENABLE_HIDRAW=0 and
    SDL_JOYSTICK_HIDAPI=0 both set), and the nodes of its sibling input devices (a DualSense is four: the pad,
    its motion sensors, its touchpad and its headset jack). This is all of them, for each pad: the hidraw
    nodes of the device the pad's input device hangs off (a HID device, or a USB interface for xpad), and the
    event/js/mouse nodes of every input device under it. `realgame.instances.offline_wrapper(mask=...)` hides
    them from the game. A pad sysfs does not know (gone since `find_pads`) contributes nothing."""
    pads = find_pads() if pads is None else pads
    sys_root = Path(sys_root)
    nodes: set[str] = set()
    for pad in pads:
        try:
            device = (sys_root / "class" / "input" / Path(pad.path).name / "device").resolve(strict=True)
            parent = (device / "device").resolve(strict=True)
        except OSError:
            continue
        hidraw = parent / "hidraw"
        if hidraw.is_dir():
            nodes.update(f"/dev/{p.name}" for p in hidraw.iterdir() if p.name.startswith("hidraw"))
        inputs = {p.resolve() for p in (parent / "input").glob("input*")} if (parent / "input").is_dir() else set()
        for input_dir in inputs | {device}:
            try:
                names = [p.name for p in input_dir.iterdir()]
            except OSError:
                continue
            nodes.update(f"/dev/input/{n}" for n in names if n.startswith(INPUT_NODE_PREFIXES))
    return sorted(nodes)


def game_holders(nodes: list[str], proc: str | Path = "/proc") -> list[tuple[int, str, str]]:
    """(pid, program, node) for each Wine process -- anything whose name or argv[0] ends in .exe -- that has one
    of `nodes` open: a game reading the pad itself, past the bridge. Only Wine counts: Steam keeps a pad's
    hidraw open and the compositor its touchpad, and neither is the game. Compared by device number, so a node
    the game opened under another path (a by-id link, a node masked to /dev/null in its namespace) is matched
    for what it really is. Processes we may not look into are skipped."""
    wanted = {}
    for node in nodes:
        try:
            wanted[os.stat(node).st_rdev] = node
        except OSError:
            continue
    held = []
    for pid_dir in Path(proc).iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            comm = (pid_dir / "comm").read_text().strip()
            argv0 = (pid_dir / "cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
            fds = list((pid_dir / "fd").iterdir())
        except OSError:
            continue
        if not (comm.lower().endswith(".exe") or argv0.lower().endswith(".exe")):
            continue
        for fd in fds:
            try:
                st = os.stat(fd)
            except OSError:
                continue
            if stat.S_ISCHR(st.st_mode) and st.st_rdev in wanted:
                program = argv0.replace("\\", "/").rsplit("/", 1)[-1] or comm
                held.append((int(pid_dir.name), program, wanted[st.st_rdev]))
    return sorted(set(held))


def resolve_axes(axes: dict | str, abs_codes: frozenset[int]) -> dict[str, int]:
    """Which axis is which. "auto" covers the two layouts drivers use: xpad, hid-playstation and most XInput
    pads put the right stick on RX/RY and the triggers on Z/RZ; many DirectInput pads put the right stick on
    Z/RZ and have digital triggers (BTN_TL2/TR2), or analog ones on BRAKE/GAS."""
    if isinstance(axes, dict):
        out = {}
        for role, name in axes.items():
            code = ABS_CODES.get(str(name).upper()) if not isinstance(name, int) else name
            if code is None:
                raise PadError(f"axes.{role}: unknown axis {name!r} (use one of {', '.join(ABS_CODES)})")
            out[role] = code
        return out
    c = ABS_CODES
    if c["ABS_RX"] in abs_codes and c["ABS_RY"] in abs_codes:
        out = {"left_x": c["ABS_X"], "left_y": c["ABS_Y"], "right_x": c["ABS_RX"], "right_y": c["ABS_RY"]}
        if c["ABS_Z"] in abs_codes and c["ABS_RZ"] in abs_codes:
            out.update(left_trigger=c["ABS_Z"], right_trigger=c["ABS_RZ"])
    elif c["ABS_Z"] in abs_codes and c["ABS_RZ"] in abs_codes:
        out = {"left_x": c["ABS_X"], "left_y": c["ABS_Y"], "right_x": c["ABS_Z"], "right_y": c["ABS_RZ"]}
    else:
        out = {"left_x": c["ABS_X"], "left_y": c["ABS_Y"]}
    if "left_trigger" not in out and c["ABS_BRAKE"] in abs_codes and c["ABS_GAS"] in abs_codes:
        out.update(left_trigger=c["ABS_BRAKE"], right_trigger=c["ABS_GAS"])
    return out


class PadDevice:
    """One gamepad's event node, opened non-blocking and grabbed, with each axis's range read off the device."""

    def __init__(self, info: PadInfo, grab: bool = True):
        self.info = info
        try:
            self.fd = os.open(info.path, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError:
            raise PadError(f"cannot read {info.path}: add yourself to the 'input' group (docs/linux.md)") from None
        except OSError as error:
            raise PadError(f"cannot open {info.path}: {error}") from None
        self.grabbed = False
        if grab:
            try:
                fcntl.ioctl(self.fd, EVIOCGRAB, 1)
                self.grabbed = True
            except OSError as error:
                os.close(self.fd)
                raise PadError(f"cannot grab {info.path} ({error}); is another program (Steam) holding it?") from None
        self.absinfo: dict[int, AbsInfo] = {}
        for code in info.abs_codes:
            buf = bytearray(_ABSINFO.size)
            try:
                fcntl.ioctl(self.fd, EVIOCGABS(code), buf)
            except OSError:
                continue
            self.absinfo[code] = AbsInfo(*_ABSINFO.unpack(buf))

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def path(self) -> str:
        return self.info.path

    def read(self) -> list[tuple[int, int, int]]:
        """Everything pending, as (type, code, value). Raises PadLost when the device is gone."""
        out = []
        while True:
            try:
                chunk = os.read(self.fd, EVENT_SIZE * 256)
            except BlockingIOError:
                break
            except OSError as error:
                if error.errno in (errno.ENODEV, errno.EIO, errno.EBADF):
                    raise PadLost(error.errno, f"{self.name} went away") from None
                raise
            if not chunk:
                raise PadLost(errno.ENODEV, f"{self.name} went away")
            out.extend(decode_raw(chunk))
            if len(chunk) < EVENT_SIZE * 256:
                break
        return out

    def close(self) -> None:
        if getattr(self, "fd", None) is None:
            return
        try:
            if self.grabbed:
                fcntl.ioctl(self.fd, EVIOCGRAB, 0)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = None


def decode_raw(buffer: bytes) -> list[tuple[int, int, int]]:
    out = []
    for start in range(0, len(buffer) - EVENT_SIZE + 1, EVENT_SIZE):
        _s, _us, kind, code, value = _EVENT.unpack_from(buffer, start)
        if kind != EV_SYN:
            out.append((kind, code, value))
    return out


# ------------------------------------------------------------------------------------------------ the bridge


class PadState:
    """What the pad is doing now: raw axis values and the buttons down, folded from its events."""

    def __init__(self, absinfo: dict[int, AbsInfo], axes: dict[str, int]):
        self.absinfo = absinfo
        self.axes = axes
        self.raw = {code: info.value for code, info in absinfo.items()}
        self.down: set[str] = set()

    def apply(self, events) -> None:
        for kind, code, value in events:
            if kind == EV_ABS:
                self.raw[code] = value
            elif kind == EV_KEY and value in (KEY_UP, KEY_DOWN):  # 2 is auto-repeat
                (self.down.add if value == KEY_DOWN else self.down.discard)(button_name(code))

    def _stick(self, code: int | None) -> float:
        if code is None or code not in self.absinfo:
            return 0.0
        info = self.absinfo[code]
        return normalize_stick(self.raw.get(code, info.value), info.minimum, info.maximum)

    def stick(self, side: str) -> tuple[float, float]:
        """(x right, y up) in [-1, 1], before any deadzone. evdev's Y is down-positive; this is up."""
        return self._stick(self.axes.get(f"{side}_x")), -self._stick(self.axes.get(f"{side}_y"))

    def trigger(self, side: str) -> float | None:
        code = self.axes.get(f"{side}_trigger")
        if code is None or code not in self.absinfo:
            return None
        info = self.absinfo[code]
        return normalize_trigger(self.raw.get(code, info.value), info.minimum, info.maximum)

    def hat(self) -> set[str]:
        out = set()
        x, y = self.raw.get(ABS_HAT0X, 0), self.raw.get(ABS_HAT0Y, 0)
        if x < 0:
            out.add("dpad_left")
        elif x > 0:
            out.add("dpad_right")
        if y < 0:
            out.add("dpad_up")
        elif y > 0:
            out.add("dpad_down")
        return out


def _kind(code: str) -> str:
    return "button" if code in BUTTON_NAMES.values() else "key"


class PadBridge:
    """Pad -> virtual mouse and keyboard, one `tick()` at a time; `start()` runs ticks on a thread.

    `sink` is anything with key(code, down) / move(dx, dy) / sync() -- a `UinputDevice`, or a fake in tests.
    Every event sent to it is also kept, as a log event, for `take()`; `close()` releases what is held."""

    def __init__(self, pad, config: PadConfig, counts_per_degree: float, sink, *, publisher=None,
                 clock=time.monotonic, reopen=None):
        self.pad = pad
        self.device_name = pad.name
        self.config = config
        self.sink = sink
        self.clock = clock
        self.publisher = publisher
        self.reopen = reopen  # () -> a new pad or None, tried once a second while the pad is lost
        self.look = LookIntegrator(config.look, counts_per_degree)
        self._set_pad(pad)
        self.emitting: set[str] = set()  # canonical key/button codes the bridge holds down now
        self.toggled: set[str] = set()  # toggle pad buttons latched on
        self._toggle_was_down: set[str] = set()
        self.log: list[dict] = []
        self.lock = threading.Lock()
        self.last_tick: float | None = None
        self.last_publish = float("-inf")
        self.lost_at: float | None = None
        self.losses = 0
        self.thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: Exception | None = None

    def _set_pad(self, pad) -> None:
        self.pad = pad
        self.axes = resolve_axes(self.config.axes, frozenset(pad.absinfo))
        self.state = PadState(pad.absinfo, self.axes)
        self.move = {name: Hysteresis(self.config.move) for name in STICK_BUTTONS}
        self.triggers = {name: Hysteresis(self.config.trigger) for name in ("lt", "rt")}

    # -- one tick

    def active_buttons(self) -> set[str]:
        """Every pad button (real or made up from an axis) that is down now."""
        down = set(self.state.down) | self.state.hat()
        x, y = shape_stick(*self.state.stick("left"), self.config.move_deadzone)
        for name, value in (("left_stick_up", y), ("left_stick_down", -y), ("left_stick_left", -x),
                            ("left_stick_right", x)):
            if self.move[name].feed(value):
                down.add(name)
        for side, name in (("left", "lt"), ("right", "rt")):
            value = self.state.trigger(side)
            if value is not None and self.triggers[name].feed(value):
                down.add(name)
        return down

    def wanted_codes(self, down: set[str]) -> set[str]:
        from zombiesai.realgame.uinput import code_for

        for button, until in self.config.toggles.items():
            pressed = button in down
            if pressed and button not in self._toggle_was_down:  # a click flips it
                (self.toggled.discard if button in self.toggled else self.toggled.add)(button)
            if until and until not in down:
                self.toggled.discard(button)
            if pressed:
                self._toggle_was_down.add(button)
            else:
                self._toggle_was_down.discard(button)
        held = (down - set(self.config.toggles)) | self.toggled
        return {key_name(code_for(self.config.buttons[b])) for b in held if b in self.config.buttons}

    def _emit_keys(self, wanted: set[str], t: float) -> bool:
        changed = False
        for code in sorted(self.emitting - wanted):  # releases first: a remap never has two keys down at once
            self.sink.key(code, False)
            self._note({"t": t, "type": _kind(code), "code": code, "down": False})
            changed = True
        for code in sorted(wanted - self.emitting):
            self.sink.key(code, True)
            self._note({"t": t, "type": _kind(code), "code": code, "down": True})
            changed = True
        self.emitting = set(wanted)
        return changed

    def _note(self, event: dict) -> None:
        with self.lock:
            self.log.append(event)

    def tick(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        dt = 1.0 / self.config.rate_hz if self.last_tick is None else min(now - self.last_tick, 0.1)
        self.last_tick = now
        if self.pad is None:
            self._try_reopen(now)
            return
        try:
            self.state.apply(self.pad.read())
        except PadLost:
            self._lose(now)
            return
        changed = self._emit_keys(self.wanted_codes(self.active_buttons()), now)
        dx, dy = self.look.step(*self.state.stick("right"), dt)
        if dx or dy:
            self.sink.move(dx, dy)
            self._note({"t": now, "type": "mouse", "dx": dx, "dy": dy})
            changed = True
        if changed:
            self.sink.sync()
        self._publish(now)

    def _lose(self, now: float) -> None:
        self.release(now)
        self.losses += 1
        self.lost_at = now
        try:
            self.pad.close()
        except OSError:
            pass
        self.pad = None
        print(f"  WARNING gamepad {self.device_name} lost; mouse and keyboard still recording", flush=True)

    def _try_reopen(self, now: float) -> None:
        if self.reopen is None or (self.lost_at is not None and now - self.lost_at < 1.0):
            return
        self.lost_at = now
        try:
            pad = self.reopen()
        except PadError:
            pad = None
        if pad is not None:
            self._set_pad(pad)
            self.look.carry = [0.0, 0.0]
            print(f"  gamepad {pad.name} back", flush=True)

    def release(self, now: float | None = None) -> None:
        """Let go of every key the bridge holds, and log it."""
        now = self.clock() if now is None else now
        self.toggled.clear()
        self._toggle_was_down.clear()
        if self.emitting and self._emit_keys(set(), now):
            self.sink.sync()

    def message(self, now: float) -> dict:
        def r(v):
            return round(float(v), 3)

        lx, ly = shape_stick(*self.state.stick("left"), self.config.move_deadzone)
        rx, ry = shape_stick(*self.state.stick("right"), self.config.look.deadzone)
        held = sorted(self.state.down | self.state.hat() | {n for n, h in self.triggers.items() if h.on})
        return {
            "kind": "pad", "t": now, "device": self.device_name,
            "sticks": {"left": [r(lx), r(ly)], "right": [r(rx), r(ry)]},
            "triggers": {side: r(self._trigger_value(side, name in held))
                         for side, name in (("left", "lt"), ("right", "rt"))},
            "buttons": held,
            "emitting": sorted(self.emitting),
        }

    def _trigger_value(self, side: str, held: bool) -> float:
        """An analog trigger's travel; a digital one (BTN_TL2/TR2) is 0 or 1."""
        value = self.state.trigger(side)
        return (1.0 if held else 0.0) if value is None else value

    def _publish(self, now: float) -> None:
        if self.publisher is None or now - self.last_publish < 1.0 / self.config.publish_hz:
            return
        self.last_publish = now
        self.publisher.send(self.message(now))

    # -- for the recorder

    def take(self) -> list[dict]:
        """The events sent since the last call, oldest first."""
        with self.lock:
            out, self.log = self.log, []
        return out

    def start(self) -> "PadBridge":
        self.thread = threading.Thread(target=self._run, name="pad-bridge", daemon=True)
        self.thread.start()
        return self

    def _run(self) -> None:
        period = 1.0 / self.config.rate_hz
        t0 = self.clock()
        k = 0
        try:
            while not self._stop.is_set():
                self.tick()
                k += 1
                now = self.clock()
                if t0 + k * period < now:  # late: skip to the next deadline, never catch up
                    k = math.ceil((now - t0) / period)
                self._stop.wait(max(0.0, t0 + k * period - self.clock()))
        except Exception as error:  # noqa: BLE001 -- a dead bridge must not leave keys down in a live game
            self.error = error
            print(f"  WARNING gamepad bridge stopped: {error!r}", flush=True)
        finally:
            try:
                self.release()
            except OSError:
                pass

    def describe(self) -> dict:
        return {"device": self.device_name, "path": getattr(self.pad, "path", None),
                "axes": {role: next((n for n, c in ABS_CODES.items() if c == code), code)
                         for role, code in self.axes.items()},
                "virtual_device": BRIDGE_NAME, "config": self.config.as_dict(),
                "counts_per_degree": self.look.cpd}

    def close(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        else:
            try:
                self.release()
            except OSError:
                pass
        if self.pad is not None:
            self.pad.close()
        if self.publisher is not None:
            self.publisher.close()
        close = getattr(self.sink, "close", None)
        if close is not None:
            close()


class PadInput:
    """The recorder's input source with a pad bridged in: the real devices' events and the bridge's, merged in
    time order. The bridge's virtual device is a `zombiesai` device, which `EvdevInput` never reads, so every
    event the bridge sent reaches the log once -- from here."""

    def __init__(self, inputs, bridge: PadBridge):
        self.inputs = inputs
        self.bridge = bridge

    @property
    def devices(self):
        return self.inputs.devices

    @property
    def monotonic(self) -> bool:
        return self.inputs.monotonic

    def drain(self, start: float = 0.0, end: float = 0.0) -> list[dict]:
        events = self.inputs.drain(start, end) + self.bridge.take()
        events.sort(key=lambda e: e["t"])
        return events

    def describe(self) -> dict:
        return {**self.inputs.describe(), "gamepad": self.bridge.describe()}

    def close(self) -> None:
        try:
            self.bridge.close()
        finally:
            self.inputs.close()


# ------------------------------------------------------------------------------------------------ wiring


def choose_pad(path: str | None = None, pads: list[PadInfo] | None = None) -> PadInfo:
    pads = find_pads() if pads is None else pads
    if path:
        match = next((p for p in pads if p.path == path), None)
        if match is None:
            raise PadError(f"{path} is not a gamepad (no X/Y axes and gamepad buttons); "
                           f"pads found: {', '.join(f'{p.path} {p.name!r}' for p in pads) or 'none'}")
        return match
    if not pads:
        raise PadError("no gamepad found under /proc/bus/input/devices -- plug one in (or pair it) and check "
                       "`uv run python -m zombiesai.demos.gamepad --watch`")
    if len(pads) > 1:
        print(f"  several pads; using {pads[0].path} {pads[0].name!r} (pick another with --pad /dev/input/eventN)")
    return pads[0]


class PadSession:
    """What record_demo.py opens for `--pad`: the pad, grabbed, and a bridge driving a fresh virtual device."""

    def __init__(self, path: str | None, config_path: str | Path | None, counts_per_degree: float,
                 bindings: dict[str, str] | None = None, *, say=print):
        from zombiesai.demos.inputs import DEFAULT_BINDINGS
        from zombiesai.realgame.uinput import UinputDevice
        from zombiesai.viz.live_bus import Publisher, bus_path

        self.config = PadConfig.load(config_path)
        for warning in self.config.spec_warnings():
            say(f"warning: {warning}")
        bindings = bindings if bindings is not None else DEFAULT_BINDINGS
        loose = sorted(f"{b} -> {c}" for b, c in self.config.buttons.items() if c not in bindings and c != "f8")
        if loose:
            say(f"  pad buttons the recorder does not label (the game still gets them): {', '.join(loose)}")
        info = choose_pad(path)
        if not os.access("/dev/uinput", os.W_OK):
            raise PadError("cannot write /dev/uinput, so the pad has nothing to drive: install the udev rule in "
                           "docs/linux.md (step 3) and log in again")
        pad = PadDevice(info)
        try:
            sink = UinputDevice(set(self.config.buttons.values()), name=BRIDGE_NAME)
        except (OSError, RuntimeError) as error:
            pad.close()
            raise PadError(f"cannot create the virtual device: {error}") from None
        name = info.name

        def reopen():
            again = next((p for p in find_pads() if p.name == name), None)
            return PadDevice(again) if again is not None else None

        self.bridge = PadBridge(pad, self.config, counts_per_degree, sink,
                                publisher=Publisher(bus_path("record")), reopen=reopen)
        self.exclude_path = info.path
        axes = ", ".join(f"{k} {v}" for k, v in self.bridge.describe()["axes"].items())
        say(f"gamepad {info.name!r} at {info.path}, grabbed; {axes}")
        self.info = info

    def game_warning(self) -> str | None:
        """What to tell the player when the game holds this pad itself -- reading it natively as well as through
        the bridge, so presses can land twice and the view turn by itself. The launchers mask every pad node
        from the game, so this means the pad came after the game did. Ask once the game's window is up: Wine
        opens pads while it boots, which can be after the bridge is."""
        held = game_holders(pad_nodes([self.info]))
        if not held:
            return None
        who = ", ".join(sorted({f"{program} (pid {pid}) has {node}" for pid, program, node in held}))
        return (f"the game can read the pad itself, past the bridge: {who}. Its presses may land twice. Quit the "
                f"game and launch it again with the pad plugged in, so it starts with the pad hidden from it.")

    def devices(self) -> list:
        """The mice and keyboards for EvdevInput, without the pad (grabbed, it would read nothing anyway)."""
        return [d for d in find_devices(("zombiesai",)) if d.path != self.exclude_path]

    def wrap(self, inputs: EvdevInput) -> PadInput:
        self.bridge.start()
        return PadInput(inputs, self.bridge)

    def annotation(self) -> dict:
        return self.bridge.describe()


# ------------------------------------------------------------------------------------------------ --watch


def watch(path: str | None) -> None:
    """Print each pad's controls by name as they change, grabbing nothing and sending nothing."""
    pads = find_pads()
    if not pads:
        raise SystemExit("no gamepad found under /proc/bus/input/devices")
    for p in pads:
        print(f"{p.path}  {p.name!r}  axes: {resolve_axes('auto', p.abs_codes)}")
    info = choose_pad(path, pads)
    pad = PadDevice(info, grab=False)
    state = PadState(pad.absinfo, resolve_axes("auto", frozenset(pad.absinfo)))
    print(f"watching {info.name!r}; press buttons and move the sticks (ctrl-c to stop)")
    last = None
    try:
        while True:
            state.apply(pad.read())
            lx, ly = state.stick("left")
            rx, ry = state.stick("right")
            lt, rt = state.trigger("left"), state.trigger("right")
            line = (f"left {lx:+.2f},{ly:+.2f}  right {rx:+.2f},{ry:+.2f}  "
                    f"lt {lt if lt is None else round(lt, 2)}  rt {rt if rt is None else round(rt, 2)}  "
                    f"buttons {' '.join(sorted(state.down | state.hat())) or '-'}")
            if line != last:
                print(line, flush=True)
                last = line
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="list gamepads and show what each control is called")
    parser.add_argument("--watch", action="store_true", help="print the pad's controls live")
    parser.add_argument("--pad", help="/dev/input/eventN (default: the first pad)")
    args = parser.parse_args()
    if args.watch:
        watch(args.pad)
        return
    pads = find_pads()
    for p in pads:
        print(f"{p.path}  {p.name!r}  axes: {resolve_axes('auto', p.abs_codes)}")
    if not pads:
        print("no gamepad found")


if __name__ == "__main__":
    main()
