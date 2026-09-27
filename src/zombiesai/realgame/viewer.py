"""Watch the running instances live, in a grid on one Hyprland workspace.

Each instance draws into an X server of its own (realgame/instances.py), which under the headless Weston host
nothing on the desktop ever shows. But an X server's root window can be read by any X client, so a viewer here
is just `ffplay -f x11grab -i :60`: a read-only copy of the picture, opened as an ordinary window on your
desktop. It sends nothing back -- no input, no focus -- so the games cannot tell they are watched.

The instances are found from their X servers' own command lines (`Xwayland :60 -geometry 2560x1440 ...`; the
desktop's rootless Xwayland has no `-geometry`), not from fleet.json, so this works from any checkout and for
either host. The viewers are launched through Hyprland with rules that put them on one workspace, floating, in
a grid in display order. Floating rather than tiled: Omarchy's dwindle splits four windows into one half and
three quarters-of-a-half, in whatever order they happened to map. They live on after the script exits.
"""

import json
import math
import re
import subprocess
from dataclasses import dataclass

from zombiesai.realgame.instances import hyprland_dispatch, hyprland_exec

TITLE_PREFIX = "zombiesai-view"  # every viewer's window title starts with this; it is how they are found again
GAP = 10  # between the viewers and around them, like Omarchy's gaps_out
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


def viewer_command(screen: Screen, box: Box, *, fps: int = 15) -> str:
    """ffplay grabbing `screen` with as little latency as it allows, scaled once to its window's size: a 1440p
    frame is ~14 MB, a lot to upload for a window a quarter of the monitor."""
    scale = f"-vf scale={box.w}:{box.h}:flags=fast_bilinear " if box.w < screen.width else ""
    return ("ffplay -loglevel quiet -an -sn -fflags nobuffer -flags low_delay -framedrop "
            f"-f x11grab -draw_mouse 0 -framerate {fps} -video_size {screen.width}x{screen.height} "
            f"{scale}-window_title {screen.title} -i {screen.display}")


def show(screens: list[Screen], *, workspace: str = "9", fps: int = 15, focus: bool = True, run=subprocess.run,
         say=print) -> int:
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
        if hyprland_exec(viewer_command(screen, box, fps=fps), rules, run=run):
            opened += 1
        else:
            say(f"  {screen.display}: Hyprland refused to launch the viewer")
    if focus:
        hyprland_dispatch(f'hl.dsp.focus({{ workspace = "{workspace}" }})', ["workspace", workspace], run=run)
    return opened


def close(*, run=subprocess.run) -> None:
    run(["pkill", "-f", f"ffplay .*-window_title {TITLE_PREFIX}:"], capture_output=True)
