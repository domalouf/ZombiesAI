"""Watch the running instances live, in a grid on one Hyprland workspace.

Each instance draws into an X server of its own (realgame/instances.py), which under the headless Weston host
nothing on the desktop ever shows. But an X server's root window can be read by any X client, so a viewer here
is a read-only copy of that picture, opened as an ordinary window on your desktop. It sends nothing back -- no
input, no focus -- so the games cannot tell they are watched.

**Only whole frames.** How a frame reaches the X server depends on the stack. With `-glamor` (DRI3) it is one
GPU copy, a single damage rectangle. With `-shm` there is no DRI3, so NVIDIA's Vulkan falls back to reading the
frame back and sending it as plain PutImage requests: a 2560x1440 frame lands as four strips of 409 rows, ~65 ms
from first to last, which also caps the game at ~11 fps. A viewer that grabs on a clock (`ffplay -f x11grab`)
catches most of those frames with some strips new and some old, torn into four bands. So a viewer here listens
for XDamage and grabs only once damage reaches the bottom row: the server handles one request at a time, so that
grab sees the whole frame, on either stack. At most `fps` of them a second are passed on -- a glamor instance
delivers 60, more than watching needs -- as BGRX straight from the shared segment, to mpv, which shows each as
it comes (`--untimed`) and scales it on the GPU.

The instances are found from their X servers' own command lines (`Xwayland :60 -geometry 2560x1440 ...`; the
desktop's rootless Xwayland has no `-geometry`), not from fleet.json, so this works from any checkout and for
either host. The viewers are launched through Hyprland with rules that put them on one workspace, floating, in
a grid in display order. Floating rather than tiled: Omarchy's dwindle splits four windows into one half and
three quarters-of-a-half, in whatever order they happened to map. They live on after the script exits.
"""

import argparse
import ctypes
import json
import math
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

from zombiesai.demos.x11_capture import X11Grabber, _bind, _load
from zombiesai.realgame.instances import hyprland_dispatch, hyprland_exec

TITLE_PREFIX = "zombiesai-view"  # every viewer's window title starts with this
GAP = 10  # between the viewers and around them, like Omarchy's gaps_out
# How the viewer processes are found again: `<python> -m zombiesai.realgame.viewer --watch :60 ...`.
_PROCESS = r"^\S+ -m zombiesai\.realgame\.viewer --watch :"
_SERVER = re.compile(r"\bXwayland (:\d+)\b.*\s-geometry (\d+)x(\d+)\b")


@dataclass(frozen=True)
class Screen:
    display: str  # ":60"
    width: int
    height: int

    @property
    def number(self) -> int:
        return int(self.display[1:])

    @property
    def title(self) -> str:
        return f"{TITLE_PREFIX}{self.display}"


@dataclass(frozen=True)
class Box:
    x: int  # monitor-local, as Hyprland's `move` rule takes them
    y: int
    w: int
    h: int


def running_screens(*, run=subprocess.run) -> list[Screen]:
    """The instances' X servers that are up now, in display order."""
    result = run(["pgrep", "-a", "Xwayland"], capture_output=True, text=True)
    screens = {}
    for line in result.stdout.splitlines():
        match = _SERVER.search(line)
        if match:
            screens[match[1]] = Screen(match[1], int(match[2]), int(match[3]))
    return sorted(screens.values(), key=lambda s: s.number)


def usable_area(workspace: str, *, run=subprocess.run) -> Box:
    """The part of the workspace's monitor not taken by bars: the monitor it is on already, else the focused
    one, which is where Hyprland will create it."""
    def query(what):
        return json.loads(run(["hyprctl", what, "-j"], capture_output=True, text=True).stdout or "[]")

    home = next((w["monitor"] for w in query("workspaces") if w["name"] == workspace), None)
    monitors = query("monitors")
    monitor = next((m for m in monitors if m["name"] == home), None) or next(m for m in monitors if m["focused"])
    w, h = monitor["width"] / monitor["scale"], monitor["height"] / monitor["scale"]
    if monitor["transform"] % 2:
        w, h = h, w
    left, top, right, bottom = monitor["reserved"]
    return Box(left, top, int(w) - left - right, int(h) - top - bottom)


def grid(n: int, area: Box, aspect: float, gap: int = GAP) -> list[Box]:
    """`n` boxes of the game's aspect, row by row, as large as fits, the whole grid centred in `area`."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = (area.w - gap * (cols + 1)) / cols
    cell_h = (area.h - gap * (rows + 1)) / rows
    w = int(min(cell_w, cell_h * aspect))
    h = int(w / aspect)
    x0 = area.x + (area.w - cols * w - (cols - 1) * gap) // 2
    y0 = area.y + (area.h - rows * h - (rows - 1) * gap) // 2
    return [Box(x0 + (i % cols) * (w + gap), y0 + (i // cols) * (h + gap), w, h) for i in range(n)]


def viewer_command(screen: Screen, fps: float = 30) -> str:
    """What Hyprland runs for one viewer: this module, with the venv's own Python, so no `uv` is needed."""
    return (f"{sys.executable} -m zombiesai.realgame.viewer --watch {screen.display} --title {screen.title} "
            f"--fps {fps:g}")


def show(screens: list[Screen], *, workspace: str = "9", fps: float = 30, focus: bool = True,
         run=subprocess.run, say=print) -> int:
    """Lay out a viewer for every screen on `workspace`, then (if `focus`) switch there. Any viewers already
    open are closed first, so rerunning after more instances come up lays them all out afresh. Returns how many
    Hyprland launched."""
    close(run=run)
    if not screens:
        return 0
    boxes = grid(len(screens), usable_area(workspace, run=run), screens[0].width / screens[0].height)
    opened = 0
    for screen, box in zip(screens, boxes):
        rules = f"workspace {workspace} silent; float; size {box.w} {box.h}; move {box.x} {box.y}"
        if hyprland_exec(viewer_command(screen, fps), rules, run=run):
            opened += 1
        else:
            say(f"  {screen.display}: Hyprland refused to launch the viewer")
    if focus:
        hyprland_dispatch(f'hl.dsp.focus({{ workspace = "{workspace}" }})', ["workspace", workspace], run=run)
    return opened


def close(*, run=subprocess.run) -> None:
    """Close every viewer. Their mpv windows follow: mpv exits when its input ends."""
    run(["pkill", "-f", _PROCESS], capture_output=True)


# ------------------------------------------------------------------------------------------------ one viewer


class FrameGate:
    """Decides when to grab. The root holds a whole frame once damage reaches the bottom row, and stops holding
    one when the next frame's first strip lands; anything that never reaches the bottom (a menu redrawing one
    corner) is let through after `settle_s`, so no change is ever lost. And no grab comes sooner than
    `min_interval_s` after the last: a whole frame that arrives early waits for its turn, it is not dropped --
    unless a newer one starts to land meanwhile, and then that one is waited for instead."""

    def __init__(self, height: int, settle_s: float = 0.25, min_interval_s: float = 0.0):
        self.height = height
        self.settle_s = settle_s
        self.min_interval_s = min_interval_s
        self.since: float | None = None  # when the first damage not yet shown arrived
        self.complete = False  # a whole frame has landed since the last grab
        self.last = float("-inf")  # when the last grab was

    def damaged(self, y: int, h: int, now: float) -> None:
        if self.since is None:
            self.since = now
        self.complete = y + h >= self.height  # a strip above the bottom: the next frame has begun

    def due(self, now: float) -> bool:
        ready = self.complete or (self.since is not None and now - self.since >= self.settle_s)
        return ready and now - self.last >= self.min_interval_s

    def wait_s(self, now: float) -> float:
        """How long to sleep before `due` could change without new damage."""
        if self.since is None:
            return self.settle_s
        until = max(self.last + self.min_interval_s, self.since + self.settle_s if not self.complete else now)
        return max(0.0, until - now)

    def shown(self, now: float) -> None:
        self.since, self.complete, self.last = None, False, now


class _XRectangle(ctypes.Structure):
    _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short), ("width", ctypes.c_ushort),
                ("height", ctypes.c_ushort)]


class _DamageNotify(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
                ("display", ctypes.c_void_p), ("drawable", ctypes.c_ulong), ("damage", ctypes.c_ulong),
                ("level", ctypes.c_int), ("more", ctypes.c_int), ("timestamp", ctypes.c_ulong),
                ("area", _XRectangle), ("geometry", _XRectangle)]


def mpv_command(width: int, height: int, title: str) -> list[str]:
    """mpv reading raw frames on stdin and showing each the moment it arrives. BGRX, the X server's own
    layout, goes to the GPU as it is: RGB would cost a conversion on each side of the pipe. --no-config: your
    own mpv.conf is for films, and a watch-later or resume setting has no business here."""
    return ["mpv", "--no-config", "--really-quiet", "--profile=low-latency", "--untimed", "--cache=no",
            "--demuxer=rawvideo", f"--demuxer-rawvideo-w={width}", f"--demuxer-rawvideo-h={height}",
            "--demuxer-rawvideo-mp-format=bgr0", "--demuxer-rawvideo-fps=60", "--no-audio", "--osc=no",
            "--osd-level=0", "--scale=bilinear", f"--title={title}", "-"]


def watch(display: str, title: str, fps: float = 30) -> None:
    """Pipe whole frames of `display`, at most `fps` a second, to an mpv window until either side goes away."""
    grabber = X11Grabber(display=display)  # the root: in a rootful Xwayland, that is the game
    width, height = grabber.size
    x11, xdamage = _load("X11", "libX11.so.6"), _load("Xdamage", "libXdamage.so.1")
    void, cint, ulong = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong
    pending = _bind(x11, "XPending", cint, [void])
    next_event = _bind(x11, "XNextEvent", cint, [void, void])
    connection = _bind(x11, "XConnectionNumber", cint, [void])
    query = _bind(xdamage, "XDamageQueryExtension", cint, [void, ctypes.POINTER(cint), ctypes.POINTER(cint)])
    create = _bind(xdamage, "XDamageCreate", ulong, [void, ulong, cint])

    event_base, error_base = cint(), cint()
    if not query(grabber.display, ctypes.byref(event_base), ctypes.byref(error_base)):
        raise SystemExit(f"{display} has no DAMAGE extension")
    create(grabber.display, grabber.root, 0)  # XDamageReportRawRectangles: every rectangle, as it lands
    fd = connection(grabber.display)
    event = ctypes.create_string_buffer(192)  # sizeof(XEvent)
    notify = _DamageNotify.from_buffer(event)
    gate = FrameGate(height, min_interval_s=1 / fps if fps else 0.0)

    player = subprocess.Popen(mpv_command(width, height, title), stdin=subprocess.PIPE, bufsize=0)
    try:
        player.stdin.write(np.ascontiguousarray(grabber.grab_bgrx()).data)  # something to show before any damage
        while player.poll() is None:
            # Xlib may already hold events it read during the last grab's round trip; the socket would not say.
            if not pending(grabber.display):
                select.select([fd], [], [], gate.wait_s(time.monotonic()))
            while pending(grabber.display):
                next_event(grabber.display, event)
                if notify.type == event_base.value:
                    gate.damaged(notify.area.y, notify.area.height, time.monotonic())
            now = time.monotonic()
            if gate.due(now):
                player.stdin.write(np.ascontiguousarray(grabber.grab_bgrx()).data)
                gate.shown(now)  # when the grab began: the write's own time must not stretch the interval
    except BrokenPipeError:  # the window was closed
        pass
    finally:
        if player.poll() is None:
            player.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="one viewer: stream an instance's X server to an mpv window")
    parser.add_argument("--watch", required=True, metavar="DISPLAY", help="e.g. :60")
    parser.add_argument("--title", default=None)
    parser.add_argument("--fps", type=float, default=30, help="at most this many frames a second (0: all)")
    args = parser.parse_args()
    watch(args.watch, args.title or f"{TITLE_PREFIX}{args.watch}", args.fps)


if __name__ == "__main__":
    main()
