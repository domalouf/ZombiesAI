"""The game's own settings, read from World at War's config.cfg and stamped into every recording.

A look label is mouse counts divided by `counts_per_degree`, and that number is a property of the in-game
sensitivity (`sensitivity` x `m_yaw` degrees per count, for this engine). Change the sensitivity between
sessions and every older recording's labels are only recoverable if the recording says what it was made at.
The same goes for the field of view (what a degree of turn looks like on screen), the resolution (where the
HUD crops sit), and the key bindings (what a key press meant). So the recorder copies them into clip.json.

WaW writes config.cfg when it exits, not when a setting changes: a value changed in the menus during the
session being recorded shows up in the *next* recording's copy. The file's modification time is kept so that
lag is visible rather than silent.
"""

import os
import re
import sys
import time
from pathlib import Path

WAW_STEAM_APP_ID = 10090
# The settings worth keeping with a recording; everything else in config.cfg is graphics and sound.
KEPT_DVARS = (
    "sensitivity",
    "m_yaw",
    "m_pitch",
    "m_filter",
    "cl_mouseAccel",
    "input_viewSensitivity",
    "cg_fov",
    "cg_fovscale",
    "r_mode",
    "r_fullscreen",
    "r_aspectRatio",
    "com_maxfps",
)
# How far the counts_per_degree given on the command line may sit from the one the config implies before the
# recorder says so. Calibration measures what the engine really does, so a small gap is expected; a large one
# is a wrong number or a changed sensitivity.
CPD_MISMATCH = 0.10

_SETA = re.compile(r'^\s*seta?\s+(\S+)\s+"(.*)"\s*$', re.IGNORECASE)
_BIND = re.compile(r'^\s*bind\s+(\S+)\s+"(.*)"\s*$', re.IGNORECASE)


def parse_config(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """(dvars, binds) from config.cfg text. Keys are as written; dvar lookups should go through `dvar()`."""
    dvars, binds = {}, {}
    for line in text.splitlines():
        if m := _SETA.match(line):
            dvars[m[1]] = m[2]
        elif m := _BIND.match(line):
            binds[m[1]] = m[2]
    return dvars, binds


def dvar(dvars: dict[str, str], name: str) -> str | None:
    """Dvar names are case-insensitive in the engine and written in mixed case in the file."""
    lowered = {k.lower(): v for k, v in dvars.items()}
    return lowered.get(name.lower())


def candidate_configs(home: Path | None = None) -> list[Path]:
    """Every WaW single-player profile config this machine has, Proton prefixes on Linux and the native path
    on Windows. Multiplayer keeps its own profile elsewhere; Nacht is single-player."""
    home = home or Path.home()
    roots = []
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.append(Path(local) / "Activision" / "CoDWaW")
    else:
        for steam in (home / ".steam" / "steam", home / ".local" / "share" / "Steam"):
            roots.append(
                steam / "steamapps" / "compatdata" / str(WAW_STEAM_APP_ID) / "pfx" / "drive_c" / "users"
                / "steamuser" / "AppData" / "Local" / "Activision" / "CoDWaW"
            )
    found = {p.resolve(): p for root in roots for p in root.glob("players/profiles/*/config.cfg")}
    return sorted(found.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def read_settings(path: str | Path | None = None, *, home: Path | None = None) -> dict | None:
    """The settings a recording should carry, or None when no config can be found (a sim recording, a
    machine without the game). Picks the most recently written profile when there are several."""
    if path is None:
        found = candidate_configs(home)
        if not found:
            return None
        path = found[0]
    path = Path(path)
    dvars, binds = parse_config(path.read_text(errors="replace"))
    kept = {name: dvar(dvars, name) for name in KEPT_DVARS}
    mtime = path.stat().st_mtime
    return {
        "config_path": str(path),
        "config_written_unix": mtime,
        "config_age_s": time.time() - mtime,
        "dvars": {k: v for k, v in kept.items() if v is not None},
        "binds": binds,
        "implied_counts_per_degree": implied_counts_per_degree(kept),
    }


def implied_counts_per_degree(dvars: dict[str, str | None]) -> float | None:
    """Mouse counts per degree of yaw the config implies: the engine turns `sensitivity * m_yaw` degrees per
    count (with mouse acceleration off). None if either is missing or unusable."""
    try:
        per_count = float(dvars.get("sensitivity")) * float(dvars.get("m_yaw"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return 1.0 / per_count if per_count > 0 else None


def settings_warnings(settings: dict | None, counts_per_degree: float) -> list[str]:
    """Things about the game's settings worth saying before anyone plays twenty minutes on them."""
    if settings is None:
        return ["couldn't find World at War's config.cfg, so this recording won't record the game's settings"]
    out = []
    dvars = settings["dvars"]
    implied = settings["implied_counts_per_degree"]
    if implied and abs(counts_per_degree / implied - 1.0) > CPD_MISMATCH:
        out.append(
            f"--counts-per-degree {counts_per_degree:g} but the game's sensitivity {dvars.get('sensitivity')} x "
            f"m_yaw {dvars.get('m_yaw')} implies {implied:.2f}; every look label would be scaled "
            f"{implied / counts_per_degree:.2f}x wrong"
        )
    if dvars.get("m_filter", "0") not in ("0", "0.0"):
        out.append("mouse smoothing is on (m_filter), so turns are smeared across frames; set m_filter 0")
    if dvars.get("cl_mouseAccel", "0") not in ("0", "0.0"):
        out.append("in-game mouse acceleration is on (cl_mouseAccel); counts no longer map linearly to degrees")
    ads = next((cmd for key, cmd in settings["binds"].items() if key.upper() == "MOUSE2"), None)
    if ads is not None and "toggleads" in ads.lower():
        out.append('right-click is toggle-ADS; the recorder labels aim as held -- bind MOUSE2 "+speed_throw"')
    return out
