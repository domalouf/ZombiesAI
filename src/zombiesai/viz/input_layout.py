"""What the input display draws, and where: one layout for the live overlay (cairo) and the demo video (ASS).

The player's left hand and mouse, the way a key-overlay shows them: the block of keys around WASD, a mouse with
its three buttons and wheel, and a look meter (which way, and how fast, the mouse is turning the view). A key
bound to a control in the recording's bindings carries that control's name under its letter and lights in the
palette's amber; a key that is not a control (space, ctrl, c...) still lights when pressed, in a dim grey, since
the game sees it and the labels do not.

Everything is in key units (`u`): a key cap is 1 x 1, and `PANEL_W` x `PANEL_H` is the whole panel. Stdlib only:
the live overlay runs under the system Python (scripts/record_overlay.py).
"""

import math
import time
from dataclasses import dataclass

# The Twitch overlay's palette (viz/stream_overlay.html, realgame/viewer.py), as RGB hex.
CHALK, CHALK_DIM, SANGUINE, AMBER, GOOD, INK = "E8E4D8", "AAB2AA", "D0593F", "DDB04D", "0CA30C", "101010"
# Short names for the controls, to fit under a key cap.
CONTROL_SHORT = {"forward": "fwd", "back": "back", "left": "left", "right": "right", "sprint": "sprint",
                 "fire": "fire", "ads": "aim", "use": "use", "reload": "reload", "melee": "melee",
                 "grenade": "nade", "swap": "swap"}
KEY_LABELS = {"grave": "`", "tab": "Tab", "capslock": "Caps", "shift": "Shift", "ctrl": "Ctrl", "alt": "Alt",
              "space": "Space", "escape": "Esc"}
WHEEL_FLASH_S = 0.15  # how long a wheel notch stays lit
LOOK_WINDOW_S = 0.1  # the look meter's speed is the mouse's over this much of the recent past
LOOK_FULL_DEG_S = 360.0  # a turn this fast reaches the meter's rim


@dataclass(frozen=True)
class Cap:
    code: str  # the evdev name the log uses: "w", "shift", "mouse1", "wheelup"
    label: str
    x: float
    y: float
    w: float = 1.0
    h: float = 1.0


GAP = 0.12  # between caps
_ROWS = (  # (row, x offset, keys) -- the left-hand block of a keyboard, staggered by each row's wide first key
    (0.0, 0.0, ("grave", "1", "2", "3", "4", "5")),
    (1.0, 0.0, (("tab", 1.5), "q", "w", "e", "r", "t")),
    (2.0, 0.0, (("capslock", 1.75), "a", "s", "d", "f", "g")),
    (3.0, 0.0, (("shift", 2.25), "z", "x", "c", "v", "b")),
    (4.0, 0.0, (("ctrl", 1.5), ("alt", 1.25), ("space", 3.6))),
)


def _keyboard() -> list[Cap]:
    caps = []
    for row, x0, keys in _ROWS:
        x = x0
        for key in keys:
            code, w = key if isinstance(key, tuple) else (key, 1.0)
            caps.append(Cap(code, KEY_LABELS.get(code, code.upper()), x, row * (1 + GAP), w))
            x += w + GAP
    return caps


KEYBOARD = tuple(_keyboard())
KEYBOARD_W = max(c.x + c.w for c in KEYBOARD)
MOUSE_X = KEYBOARD_W + 0.5  # the mouse, right of the keys
MOUSE = (
    Cap("mouse1", "L", MOUSE_X, 0.0, 1.0, 1.6),
    Cap("mouse3", "M", MOUSE_X + 1.0 + GAP, 0.0, 0.5, 1.6),
    Cap("mouse2", "R", MOUSE_X + 1.5 + 2 * GAP, 0.0, 1.0, 1.6),
)
MOUSE_W = 2.5 + 2 * GAP
WHEEL = (Cap("wheelup", "▲", MOUSE_X + 1.0 + GAP, -0.75, 0.5, 0.6),
         Cap("wheeldown", "▼", MOUSE_X + 1.0 + GAP, 1.75, 0.5, 0.6))
# The look meter: a circle under the mouse buttons, centre and radius.
LOOK_CX, LOOK_CY, LOOK_R = MOUSE_X + MOUSE_W / 2, 3.45, 1.05
PANEL_W = MOUSE_X + MOUSE_W
PANEL_H = 4.0 + 1.0 + 4 * GAP  # five rows
ALL_CAPS = KEYBOARD + MOUSE + WHEEL


def control_of(bindings: dict[str, str]) -> dict[str, str]:
    """Code -> the short name of the control it drives, for the caps that are bound."""
    return {str(code).lower(): CONTROL_SHORT.get(control, control) for code, control in bindings.items()}


def look_vector(dx: float, dy: float, seconds: float, counts_per_degree: float) -> tuple[float, float, float]:
    """Mouse counts over `seconds` -> (x, y, speed): the meter's unit-circle direction scaled to [0, 1] by how
    fast the turn is (`LOOK_FULL_DEG_S` at the rim, square-root so small aim corrections still show), and the
    speed in degrees a second. Screen y is down, as the mouse's own is."""
    if seconds <= 0 or counts_per_degree <= 0:
        return 0.0, 0.0, 0.0
    vx, vy = dx / counts_per_degree / seconds, dy / counts_per_degree / seconds
    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        return 0.0, 0.0, 0.0
    reach = math.sqrt(min(1.0, speed / LOOK_FULL_DEG_S))
    return vx / speed * reach, vy / speed * reach, speed


class InputState:
    """What is held right now, folded from the recorder's log events (`demos/evdev_input.decode_events`), for a
    live display: the codes down, when each wheel notch came, and the recent mouse motion for the look meter."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.down: set[str] = set()
        self.flashed: dict[str, float] = {}
        self.motion: list[tuple[float, int, int]] = []

    def feed(self, events) -> bool:
        """Fold `events` in. True if anything visible may have changed."""
        changed = False
        for event in events:
            kind, code = event.get("type"), str(event.get("code", "")).lower()
            if kind == "mouse":
                self.motion.append((self.clock(), int(event.get("dx", 0)), int(event.get("dy", 0))))
                changed = True
            elif kind in ("key", "button"):
                if code.startswith("wheel"):
                    if event.get("down"):
                        self.flashed[code] = self.clock()
                        changed = True
                elif event.get("down"):
                    changed |= code not in self.down
                    self.down.add(code)
                else:
                    changed |= code in self.down
                    self.down.discard(code)
        return changed

    def lit(self, now: float | None = None) -> set[str]:
        now = self.clock() if now is None else now
        return self.down | {code for code, t in self.flashed.items() if now - t < WHEEL_FLASH_S}

    def look(self, counts_per_degree: float, now: float | None = None) -> tuple[float, float, float]:
        now = self.clock() if now is None else now
        self.motion = [m for m in self.motion if now - m[0] <= LOOK_WINDOW_S]
        dx = sum(m[1] for m in self.motion)
        dy = sum(m[2] for m in self.motion)
        return look_vector(dx, dy, LOOK_WINDOW_S, counts_per_degree)

    def release_all(self) -> None:
        self.down.clear()
