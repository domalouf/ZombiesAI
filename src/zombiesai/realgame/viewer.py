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

**What the agent sees**, inset in the top right corner (`--no-agent-view` leaves it out): each shown frame made
into the policy's 128x72 observation by the code the actors run (demos/frames.py, as demos/capture.py calls it),
then blown up with whole pixels so every one of them can be seen. It is the picture the policy gets, not the
exact frame it acted on: the actor grabs on its own 15 Hz deadlines, and stacks several such frames.

Drawn over the picture by mpv, through its IPC socket (`Overlay`), not into it:

- **the stream's numbers** in the top left, as the Twitch overlay shows them (viz/stream.py's `summarize` over
  the run's episodes.jsonl): best round, average round, accuracy, average survival -- and average points, the
  points a game earned beyond the 500 it starts with, and average kills (realgame/hud_reward.py says how a kill
  is told from a hit, and why the count is a lower bound).
- **what the reward reads**: the HUD regions the env crops for its reward -- points and round -- outlined,
  and what the HUD parser reads in them right now. The policy never sees these crops; the reward does.

**Sound** (`--sink`): the instance's own PulseAudio null sink, the one the agent hears, looped to your default
output by `pw-loopback` for as long as the viewer is open.
"""

import argparse
import ctypes
import json
import math
import os
import re
import select
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zombiesai.demos import frames as fr
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


def viewer_command(screen: Screen, fps: float = 30, agent_view: bool = True, run_dir: str | None = None,
                   sink: str | None = None) -> str:
    """What Hyprland runs for one viewer: this module, with the venv's own Python, so no `uv` is needed."""
    command = (f"{sys.executable} -m zombiesai.realgame.viewer --watch {screen.display} --title {screen.title} "
               f"--fps {fps:g}")
    if not agent_view:
        command += " --no-agent-view"
    if run_dir:
        command += f" --run {Path(run_dir).resolve()}"
    if sink:
        command += f" --sink {sink}"
    return command


def show(screens: list[Screen], *, workspace: str = "9", fps: float = 30, agent_view: bool = True,
         run_dir: str | None = None, sinks: dict[str, str] | None = None, focus: bool = True,
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
        command = viewer_command(screen, fps, agent_view, run_dir, (sinks or {}).get(screen.display))
        if hyprland_exec(command, rules, run=run):
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


class AgentInset:
    """Draws the agent's view of a frame into the frame's top right corner (see the module docstring).

    The crop box is found on the first frame and kept, as the capture keeps it. The inset is `share` of the
    picture's width, rounded to a whole number of screen pixels per observation pixel."""

    MARGIN = 32  # screen pixels between the inset's border and the picture's edges
    BORDER = 4

    def __init__(self, height: int, width: int, share: float = 0.25):
        self.scale = max(1, round(width * share / fr.FRAME_W))
        self.h, self.w = fr.FRAME_H * self.scale, fr.FRAME_W * self.scale
        self.y = self.MARGIN + self.BORDER
        self.x = width - self.MARGIN - self.BORDER - self.w
        self.box: tuple[int, int, int, int] | None = None

    @property
    def bottom(self) -> int:
        """The first screen row below the inset's border."""
        return self.y + self.h + self.BORDER

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """`frame` (H, W, 4) BGRX with the inset drawn in: in place when the array allows it (a grab's shared
        segment, which the next grab overwrites anyway), else on a copy."""
        if self.box is None:
            self.box = fr.crop_box(*frame.shape[:2], fr.detect_bars(frame[..., fr.BGRX]), "crop")
        seen = fr.to_policy_frame(frame, self.box, "crop", channels=fr.BGRX)  # RGB, read before drawing
        out = frame if frame.flags.writeable else frame.copy()
        b = self.BORDER
        out[self.y - b:self.bottom, self.x - b:self.x + self.w + b, :3] = 255
        out[self.y:self.y + self.h, self.x:self.x + self.w, :3] = (
            seen[..., ::-1].repeat(self.scale, axis=0).repeat(self.scale, axis=1))
        return out


def mpv_command(width: int, height: int, title: str, ipc: str | None = None) -> list[str]:
    """mpv reading raw frames on stdin and showing each the moment it arrives. BGRX, the X server's own
    layout, goes to the GPU as it is: RGB would cost a conversion on each side of the pipe. --no-config: your
    own mpv.conf is for films, and a watch-later or resume setting has no business here. With `ipc`, mpv
    listens on that socket for the overlay (`Overlay`)."""
    osd = [f"--input-ipc-server={ipc}"] if ipc else ["--osd-level=0"]
    return ["mpv", "--no-config", "--really-quiet", "--profile=low-latency", "--untimed", "--cache=no",
            "--demuxer=rawvideo", f"--demuxer-rawvideo-w={width}", f"--demuxer-rawvideo-h={height}",
            "--demuxer-rawvideo-mp-format=bgr0", "--demuxer-rawvideo-fps=60", "--no-audio", "--osc=no",
            *osd, "--scale=bilinear", f"--title={title}", "-"]


# ------------------------------------------------------------------------------------------------ the overlay

# The Twitch overlay's palette (viz/stream_overlay.html), as ASS colours: &HBBGGRR&.
CHALK, CHALK_DIM, SANGUINE, AMBER, GOOD = "&HD8E4E8&", "&HAAB2AA&", "&H3F59D0&", "&H4DB0DD&", "&H0CA30C&"
STATUS_TEXT = {1: "\u2013", 2: "?", 3: "\u2026"}  # ABSENT, UNREADABLE, TRANSITION (hud/parse.py); OK shows the value
LIVE_S = 300  # a run whose episodes.jsonl changed this recently is training now


def _ass(text: str) -> str:
    """Text that ASS will show as written: braces and backslashes would be read as override tags."""
    return text.replace("\\", "\u29f5").replace("{", "(").replace("}", ")")


def _text(x: float, y: float, align: int, size: float, colour: str, text: str, bold: bool = False) -> str:
    return (f"{{\\an{align}\\pos({x:.0f},{y:.0f})\\fs{size:.0f}\\b{int(bold)}\\bord0\\shad1\\4c&H000000&\\4a&H60&"
            f"\\1c{colour}}}{_ass(text)}")


def _rect(x: float, y: float, w: float, h: float, colour: str, alpha: str = "00") -> str:
    return (f"{{\\an7\\pos({x:.0f},{y:.0f})\\p1\\bord0\\shad0\\1c{colour}\\1a&H{alpha}&}}"
            f"m 0 0 l {w:.0f} 0 {w:.0f} {h:.0f} 0 {h:.0f}{{\\p0}}")


def _outline(box: tuple[int, int, int, int], colour: str, width: float) -> str:
    left, top, w, h = box
    return (f"{{\\an7\\pos({left},{top})\\p1\\bord{width:.0f}\\shad0\\1a&HFF&\\3c{colour}}}"
            f"m 0 0 l {w} 0 {w} {h} 0 {h}{{\\p0}}")


def _duration(seconds) -> str:
    if seconds is None:
        return "\u2013"
    s = round(seconds)
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def stats_panel(stats: dict | None, run: str | None, live: bool, height: int) -> list[str]:
    """The Twitch overlay's panel (viz/stream_overlay.html), in the top left, sized as it is on a 1080p canvas
    and scaled to this one."""
    k = height / 1080
    x0 = y0 = 28 * k
    col = 128 * k
    w, h = 19 * k + 6 * col - 4 * k, 124 * k
    stats = stats or {}
    head = "ZOMBIES AI \u00b7 PPO" + (f" \u00b7 {run}" if run else "")
    out = [_rect(x0, y0, w, h, "&H0E0E0E&", "40"), _rect(x0, y0, 3 * k, h, SANGUINE),
           _text(x0 + 19 * k, y0 + 11 * k, 7, 13.5 * k, CHALK_DIM, head, bold=True)]
    if live:
        out.append(_text(x0 + w - 17 * k, y0 + 11 * k, 9, 13.5 * k, GOOD, "\u25cf TRAINING LIVE", bold=True))
    acc = stats.get("accuracy")
    values = (
        ("Best round", "\u2013" if stats.get("best_round") is None else str(stats["best_round"])),
        ("Avg round", "\u2013" if stats.get("avg_round") is None else f"{stats['avg_round']:.1f}"),
        ("Accuracy", "\u2013" if acc is None else f"{100 * acc:.{1 if acc < 0.1 else 0}f}%"),
        ("Avg survival", _duration(stats.get("avg_survival_s"))),
        ("Avg points", "\u2013" if stats.get("avg_points") is None else f"{stats['avg_points']:,.0f}"),
        ("Avg kills", "\u2013" if stats.get("avg_kills") is None else f"{stats['avg_kills']:.1f}"),
    )
    for i, (label, value) in enumerate(values):
        x = x0 + 19 * k + i * col
        out.append(_text(x, y0 + 32 * k, 7, 14 * k, CHALK_DIM, label, bold=True))
        out.append(_text(x, y0 + 49 * k, 7, 41 * k, CHALK, value, bold=True))
    games = stats.get("games") or 0
    foot = (f"Averages over the last {stats.get('window')} of {games:,} games" if games
            else "Waiting for the first game\u2026")
    out.append(_text(x0 + 19 * k, y0 + 98 * k, 7, 13.5 * k, CHALK_DIM, foot))
    return out


def hud_marks(boxes: dict[str, tuple[int, int, int, int]], reading, height: int) -> list[str]:
    """The reward's HUD regions outlined, each with what the parser reads in it."""
    k = height / 1080
    out = []

    def value(v, status, unit=""):
        return f"{v}{unit}" if status == 0 else STATUS_TEXT.get(status, "?")

    for name, (left, top, w, h) in boxes.items():
        out.append(_outline((left, top, w, h), AMBER, 2 * k))
        if name == "points_ammo":
            text = "reward reads: " + ("\u2013" if reading is None else value(reading.points, reading.points_status, " pts"))
            out.append(_text(left + w - 8 * k, top - 6 * k, 3, 18 * k, AMBER, text, bold=True))
        elif name == "round":
            text = "round " + ("\u2013" if reading is None else value(reading.round, reading.round_status))
            out.append(_text(left + 8 * k, top - 6 * k, 1, 18 * k, AMBER, text, bold=True))
    return out


def inset_label(inset: "AgentInset", height: int) -> list[str]:
    k = height / 1080
    return [_text(inset.x + inset.w, inset.bottom + 8 * k, 9, 18 * k, CHALK,
                  f"what the agent sees \u00b7 {fr.FRAME_W}\u00d7{fr.FRAME_H}", bold=True)]


class HudView:
    """The HUD crops the env's capture takes for the reward (realgame/instances.py: `Instance.capture`), cut and
    parsed the same way, from the frame the viewer shows."""

    REGIONS = ("points_ammo", "round")

    def __init__(self, height: int, width: int):
        from zombiesai.demos.hud_crops import HUD_REGIONS, region_box
        from zombiesai.hud.parse import HudParser

        self.regions = {k: HUD_REGIONS[k] for k in self.REGIONS}
        self.boxes = {k: region_box(height, width, v) for k, v in self.regions.items()}
        self.scale = min(1.0, 720.0 / height)
        self.parser = HudParser()

    def read(self, frame: np.ndarray):
        from zombiesai.demos.hud_crops import crop_regions

        return self.parser.parse(crop_regions(frame, self.regions, self.scale, channels=fr.BGRX))


class RunStats:
    """The stream's four numbers for one run, re-read every `every_s` (viz/stream.py)."""

    def __init__(self, run_dir: str | None, every_s: float = 5.0):
        from zombiesai.viz.stream import EPISODES, EpisodeTail

        self.path = Path(run_dir) / EPISODES if run_dir else None
        self.name = Path(run_dir).name if run_dir else None
        self.tail, self.every_s = EpisodeTail(), every_s
        self.next, self.stats, self.live = 0.0, None, False

    def poll(self, now: float) -> bool:
        """Re-read if due. True when the numbers changed."""
        if self.path is None or now < self.next:
            return False
        from zombiesai.viz.stream import summarize

        self.next = now + self.every_s
        before = (self.stats, self.live)
        self.stats = summarize(self.tail.read(self.path))
        try:
            self.live = time.time() - self.path.stat().st_mtime < LIVE_S
        except OSError:
            self.live = False
        return (self.stats, self.live) != before


class Overlay:
    """mpv's OSD over its JSON IPC socket: one `osd-overlay` of ASS events, in the picture's own pixels, replaced
    whenever it changes. Connects lazily (mpv makes the socket a moment after it starts), drains mpv's replies so
    they never back up, and goes quiet rather than failing if mpv goes away: the picture matters more."""

    def __init__(self, path: str, width: int, height: int):
        self.path, self.width, self.height = path, width, height
        self.sock: socket.socket | None = None
        self.shown: str | None = None

    def _connect(self) -> bool:
        if self.sock is not None:
            return True
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.path)
        except OSError:
            return False
        sock.setblocking(False)
        self.sock = sock
        self._send({"command": ["disable_event", "all"]})
        return True

    def _send(self, message: dict) -> None:
        try:
            self.sock.sendall((json.dumps(message) + "\n").encode())
            while True:
                if not self.sock.recv(65536):
                    break
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            self.sock, self.shown = None, None

    def show(self, events: list[str]) -> None:
        data = "\n".join(events)
        if data == self.shown or not self._connect():
            return
        self._send({"command": {"name": "osd-overlay", "id": 1, "format": "ass-events", "data": data,
                                "res_x": self.width, "res_y": self.height}})
        if self.sock is not None:
            self.shown = data


# ------------------------------------------------------------------------------------------------ sound


def loopback_command(sink: str, title: str) -> list[str]:
    """The instance's sink, as the agent hears it (its monitor), played on your default output."""
    return ["pw-loopback", "--name", title.replace(":", "-"), "--latency", "60",
            "--capture-props", f"target.object={sink} stream.capture.sink=true node.dont-reconnect=true",
            "--playback-props", f'media.name="{title}"']  # quoted: a bare ':' ends PipeWire's key


def _die_with_parent() -> None:
    """In the child, before exec: a SIGTERM when the viewer exits however it exits, so no sound outlives it."""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class DamageFeed:
    """The root of one X server, and the damage that lands on it: `wait()` blocks until there is some (or the
    timeout passes) and returns it as (y, height) rows. `grabber` reads the pixels."""

    def __init__(self, display: str):
        self.grabber = X11Grabber(display=display)  # the root: in a rootful Xwayland, that is the game
        self.width, self.height = self.grabber.size
        x11, xdamage = _load("X11", "libX11.so.6"), _load("Xdamage", "libXdamage.so.1")
        void, cint, ulong = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong
        self._pending = _bind(x11, "XPending", cint, [void])
        self._next_event = _bind(x11, "XNextEvent", cint, [void, void])
        connection = _bind(x11, "XConnectionNumber", cint, [void])
        query = _bind(xdamage, "XDamageQueryExtension", cint, [void, ctypes.POINTER(cint), ctypes.POINTER(cint)])
        create = _bind(xdamage, "XDamageCreate", ulong, [void, ulong, cint])

        self._event_base, error_base = cint(), cint()
        if not query(self.grabber.display, ctypes.byref(self._event_base), ctypes.byref(error_base)):
            raise SystemExit(f"{display} has no DAMAGE extension")
        create(self.grabber.display, self.grabber.root, 0)  # XDamageReportRawRectangles: each, as it lands
        self._fd = connection(self.grabber.display)
        self._event = ctypes.create_string_buffer(192)  # sizeof(XEvent)
        self._notify = _DamageNotify.from_buffer(self._event)

    def wait(self, timeout_s: float) -> list[tuple[int, int]]:
        display = self.grabber.display
        # Xlib may already hold events it read during the last grab's round trip; the socket would not say.
        if not self._pending(display):
            select.select([self._fd], [], [], timeout_s)
        rects = []
        while self._pending(display):
            self._next_event(display, self._event)
            if self._notify.type == self._event_base.value:
                rects.append((self._notify.area.y, self._notify.area.height))
        return rects


def watch(display: str, title: str, fps: float = 30, agent_view: bool = True, run_dir: str | None = None,
          sink: str | None = None) -> None:
    """Pipe whole frames of `display`, at most `fps` a second, to an mpv window until either side goes away,
    with the overlay drawn over them and, with `sink`, the game's sound."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # `close()` sends it: run the cleanup below
    feed = DamageFeed(display)
    gate = FrameGate(feed.height, min_interval_s=1 / fps if fps else 0.0)
    inset = AgentInset(feed.height, feed.width) if agent_view else None
    hud = HudView(feed.height, feed.width)
    stats = RunStats(run_dir)
    ipc = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / f"{title.replace(':', '-')}.sock"
    ipc.unlink(missing_ok=True)
    overlay = Overlay(str(ipc), feed.width, feed.height)
    player = subprocess.Popen(mpv_command(feed.width, feed.height, title, str(ipc)), stdin=subprocess.PIPE,
                              bufsize=0)
    sound = (subprocess.Popen(loopback_command(sink, title), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              preexec_fn=_die_with_parent) if sink else None)
    static = inset_label(inset, feed.height) if inset is not None else []

    def frame() -> memoryview:
        grabbed = feed.grabber.grab_bgrx()
        reading = hud.read(grabbed)  # before the inset is drawn into it
        stats.poll(time.monotonic())
        overlay.show(stats_panel(stats.stats, stats.name, stats.live, feed.height)
                     + hud_marks(hud.boxes, reading, feed.height) + static)
        return np.ascontiguousarray(inset.draw(grabbed) if inset is not None else grabbed).data

    try:
        player.stdin.write(frame())  # something before any damage
        while player.poll() is None:
            for y, h in feed.wait(gate.wait_s(time.monotonic())):
                gate.damaged(y, h, time.monotonic())
            now = time.monotonic()
            if gate.due(now):
                player.stdin.write(frame())
                gate.shown(now)  # when the grab began: the write's own time must not stretch the interval
    except BrokenPipeError:  # the window was closed
        pass
    finally:
        for child in (player, sound):
            if child is not None and child.poll() is None:
                child.terminate()
        ipc.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="one viewer: stream an instance's X server to an mpv window")
    parser.add_argument("--watch", required=True, metavar="DISPLAY", help="e.g. :60")
    parser.add_argument("--title", default=None)
    parser.add_argument("--fps", type=float, default=30, help="at most this many frames a second (0: all)")
    parser.add_argument("--agent-view", action=argparse.BooleanOptionalAction, default=True,
                        help="inset the policy's 128x72 view of each frame (default: on)")
    parser.add_argument("--run", default=None, help="a run directory, for the stream's numbers")
    parser.add_argument("--sink", default=None, help="the instance's audio sink, to play its sound")
    args = parser.parse_args()
    watch(args.watch, args.title or f"{TITLE_PREFIX}{args.watch}", args.fps, args.agent_view, args.run, args.sink)


if __name__ == "__main__":
    main()
