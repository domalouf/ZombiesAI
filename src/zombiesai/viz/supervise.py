"""The Training Room, live, on localhost: every run's curves, what the running trainers are printing, and the game
instances -- with the switch for their viewer windows (realgame/viewer.py), and buttons to start a run and to
stop one gracefully.

    uv run python scripts/supervise.py            # http://127.0.0.1:8765

Runs are read from `runs/` in the main checkout and in every worktree (a trainer writes where it was started),
with the same code as the static dashboard (viz/dashboard.py). A trainer is found by its process
(`scripts/train_*.py`); what it prints is read from wherever its stdout goes, when that is a file.

Each game gets a monitor process while the page is open: it counts whole frames off XDamage (nearly free) and
keeps a small thumbnail of the latest whole frame. They start with the first look at the page and stop
`idle_s` after the last, so a closed dashboard costs nothing. A monitor is a process of its own because Xlib
ends whatever process loses its X server, and an instance going down must not take the dashboard with it.

A run is started as the fleet's runs have been (`start_run`): scripts/train_rl.py in the chosen checkout, with
its src/ first on PYTHONPATH, printing to runs/<name>.log, in a session of its own so the dashboard can come and
go. Stopping sends SIGINT to the learner alone -- Ctrl-C's path, which stops the actors through their stop event
(every key released) and writes the checkpoint; a stop that has not finished in 30 s can be forced.

The machine itself -- CPU, memory, GPU, temperatures, disks, network and the busiest processes, as btop shows
them -- is sampled every 2 s for as long as the server runs (viz/system.py), so the last half hour is on the page
the moment it opens.

It listens on 127.0.0.1 only, answers only requests addressed to it by name (no DNS rebinding), and takes
commands only with an `X-Supervise` header, which no other page can send without a preflight it would refuse.
"""

import argparse
import ast
import json
import math
import os
import re
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from zombiesai.viz.dashboard import SERIES_COLORS, build_dashboard, collect_runs, dashboard_html
from zombiesai.viz.system import SystemSampler

HERE = Path(__file__).parent
GAME_TITLES = ("Plutonium", "Call of Duty")
_XWAYLAND = re.compile(r"^(\d+) Xwayland (:\d+)\b(.*)$")


# ------------------------------------------------------------------------------------------------ runs


def trees(repo: Path, *, run=subprocess.run) -> list[tuple[str, Path]]:
    """(label, path) for the main checkout ("main") and every worktree (its directory's name)."""
    result = run(["git", "-C", str(repo), "worktree", "list", "--porcelain"], capture_output=True, text=True)
    paths = [Path(line[len("worktree "):]) for line in result.stdout.splitlines() if line.startswith("worktree ")]
    return [("main" if i == 0 else path.name, path) for i, path in enumerate(paths or [repo])]


def run_roots(repo: Path, *, run=subprocess.run) -> list[tuple[str, Path]]:
    """(label, runs dir) for the main checkout and every worktree that has a runs/ directory."""
    return [(label, tree / "runs") for label, tree in trees(repo, run=run) if (tree / "runs").is_dir()]


def build_payload(roots: list[tuple[str, Path]], *, stale_after: float = 600.0, now: float | None = None) -> dict:
    """The static dashboard's payload over several runs dirs. A run is named `<tree>/<run>` once more than one
    tree has runs, so rl1 in two worktrees stays two runs."""
    now = time.time() if now is None else now
    # The frame (gates, series, time) from a root with no runs: /dev/null is never a directory.
    payload = build_dashboard(Path(os.devnull), stale_after=stale_after, now=now)
    runs, paths = [], {}
    many = sum(1 for _, root in roots if any(root.iterdir())) > 1
    for label, root in roots:
        for r in collect_runs(root, stale_after=stale_after, now=now):
            path = str((root / r["name"]).resolve())
            if many:
                r["name"] = f"{label}/{r['name']}"
            paths[path] = r["name"]
            runs.append(r)
    runs.sort(key=lambda r: r["updated"], reverse=True)
    for i, r in enumerate(runs):  # the slots, over all of them: colour follows the run
        r["color"], r["slot"] = SERIES_COLORS[i % len(SERIES_COLORS)], i
    payload.update(runs=runs, root=", ".join(f"{label}/runs" for label, _ in roots) or "runs/")
    payload["run_paths"] = paths
    return payload


def _proc_start_epoch(pid: int) -> float | None:
    try:
        ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return time.time() - uptime + ticks / os.sysconf("SC_CLK_TCK")


def _tail(path: Path, lines: int = 40, max_bytes: int = 32768) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    return text.replace("\r", "\n").splitlines()[-lines:]


def live_trainers(run_paths: dict[str, str], proc: Path = Path("/proc")) -> list[dict]:
    """Every `scripts/train_*.py` process: which run it writes, how long it has run, and the tail of its stdout."""
    found = []
    for entry in proc.iterdir() if proc.is_dir() else []:
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        script = next((a for a in argv if re.search(r"(^|/)scripts/train_\w+\.py$", a)), None)
        if script is None:
            continue
        try:
            cwd = Path(os.readlink(entry / "cwd"))
            out = os.readlink(entry / "fd" / "1")
        except OSError:
            continue
        run_dir = None
        if "--out" in argv[:-1]:
            run_dir = str((cwd / argv[argv.index("--out") + 1]).resolve())
        log = Path(out) if out.startswith("/") and Path(out).is_file() else None
        started = _proc_start_epoch(int(entry.name))
        env_flag = argv[argv.index("--env") + 1] if "--env" in argv[:-1] else None
        found.append({
            "pid": int(entry.name),
            "script": Path(script).name,
            "args": " ".join(argv[argv.index(script) + 1:]),
            "tree": cwd.name,
            "cwd": str(cwd),
            "sim": env_flag == "sim",  # a sim run plays no game, so it does not hold the fleet
            "run": run_paths.get(run_dir) if run_dir else None,
            "elapsed_s": None if started is None else round(time.time() - started, 1),
            "log": str(log) if log else None,
            "tail": _tail(log) if log else [],
        })
    return sorted(found, key=lambda t: t["pid"])


# ------------------------------------------------------------------------------------------------ games


def screens_with_stack(*, run=subprocess.run) -> list[dict]:
    """The instances' X servers (see realgame/viewer.running_screens), with how frames reach each: `gpu`
    (glamor: whole frames) or `copy` (-shm: read back and sent in strips, ~11 fps at 1440p)."""
    from zombiesai.realgame.viewer import running_screens

    lines = run(["pgrep", "-a", "Xwayland"], capture_output=True, text=True).stdout.splitlines()
    flags = {m[2]: m[3] for line in lines if (m := _XWAYLAND.match(line))}
    return [{"display": s.display, "number": s.number, "width": s.width, "height": s.height,
             "stack": "gpu" if "-glamor" in flags.get(s.display, "") else "copy"}
            for s in running_screens(run=run)]


def open_viewers(*, run=subprocess.run) -> list[int]:
    out = run(["pgrep", "-af", r"^\S+ -m zombiesai\.realgame\.viewer --watch :"], capture_output=True,
              text=True).stdout
    return sorted({int(m[1]) for m in re.finditer(r"--watch :(\d+)", out)})


def png(rgb) -> bytes:
    """An 8-bit RGB PNG of an (H, W, 3) uint8 array: stdlib zlib, no imaging library needed for a thumbnail."""
    h, w, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def thumbnail(bgrx, width: int = 320):
    """Downscale a BGRX frame to about `width` wide, RGB: every other pixel of a 2x-wider grid, then 2x2 means --
    a box filter at a fraction of a full one's cost."""
    import numpy as np

    h, w, _ = bgrx.shape
    step = max(1, w // (width * 2))
    sub = bgrx[::step, ::step, 2::-1]
    sh, sw = sub.shape[0] // 2 * 2, sub.shape[1] // 2 * 2
    small = sub[:sh, :sw].reshape(sh // 2, 2, sw // 2, 2, 3).mean(axis=(1, 3))
    return np.ascontiguousarray(small.astype(np.uint8))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def monitor(display: str, out_dir: Path, *, thumb_every_s: float = 2.0) -> None:
    """One game's monitor (a process of its own): whole frames per second from XDamage, the game window's title,
    and -- unless `thumbs-off` exists in `out_dir` -- a thumbnail of a whole frame every `thumb_every_s`."""
    from zombiesai.demos.x11_capture import list_windows
    from zombiesai.realgame.viewer import DamageFeed, FrameGate

    feed = DamageFeed(display)
    gate = FrameGate(feed.height, min_interval_s=thumb_every_s)
    number = display.lstrip(":")
    status_path, thumb_path = out_dir / f"{number}.json", out_dir / f"{number}.png"
    parent = os.getppid()
    frames: deque[float] = deque()
    game, thumb_at, next_status, next_look = None, None, 0.0, 0.0
    started = time.monotonic()
    while os.getppid() == parent:  # the server went away: so do we
        now = time.monotonic()
        for y, h in feed.wait(min(0.5, gate.wait_s(now))):
            now = time.monotonic()
            gate.damaged(y, h, now)
            if y + h >= feed.height:
                frames.append(now)
        now = time.monotonic()
        while frames and frames[0] < now - 2.0:
            frames.popleft()
        if (thumb_at is None or gate.due(now)) and not (out_dir / "thumbs-off").exists():
            _write_atomic(thumb_path, png(thumbnail(feed.grabber.grab_bgrx())))
            gate.shown(now)
            thumb_at = time.time()
        if now >= next_look:
            titles = [w.title for w in list_windows(display)]
            game = next((t for t in titles if any(g in t for g in GAME_TITLES)), None)
            next_look = now + 5.0
        if now >= next_status:
            span = min(2.0, now - started)  # a monitor that just started has not watched for 2 s yet
            _write_atomic(status_path, json.dumps({
                "fps": round(len(frames) / span, 1) if span >= 0.5 else None,
                "game": game, "thumb_at": thumb_at, "at": time.time(),
            }).encode())
            next_status = now + 1.0


# ------------------------------------------------------------------------------------------------ the machine

# When a reading earns a mark, per the hardware in this PC. A Ryzen 7000 is built to run up to 95 C under load
# (Tctl), so warm is normal there; an RTX 5070 starts to slow itself near 90 C. Busy CPU: the benchmark of
# 2026-09-27 had agents starting to miss their 15 Hz deadlines past ~85% of all threads (8 games).
LIMITS = {  # warning, serious, critical
    "cpu_temp": (85, 90, 95), "gpu_temp": (80, 85, 90), "ssd_temp": (70, 75, 80), "ram_temp": (70, 80, 85),
    "cpu_busy": (85, 95, 101), "vram": (85, 92, 97), "ram": (85, 92, 97),
}
RECOMMENDED_GAMES = 6  # the most this PC ran with every agent keeping time (benchmark, 2026-09-27)


def level(key: str, value) -> str:
    if value is None:
        return "info"
    warning, serious, critical = LIMITS[key]
    return "critical" if value >= critical else "serious" if value >= serious else "warning" if value >= warning else "good"


def hwmon_temps(root: Path = Path("/sys/class/hwmon")) -> dict:
    """The temperatures worth a tile: the CPU's Tctl and die (k10temp), the SSD (nvme), the RAM (spd5118)."""
    found = {}
    for hw in root.iterdir() if root.is_dir() else []:
        try:
            name = (hw / "name").read_text().strip()
        except OSError:
            continue
        for sensor in hw.glob("temp*_input"):
            label_file = sensor.with_name(sensor.name.replace("_input", "_label"))
            try:
                label = label_file.read_text().strip() if label_file.exists() else ""
                celsius = int(sensor.read_text()) / 1000
            except (OSError, ValueError):
                continue
            key = {("k10temp", "Tctl"): "cpu", ("k10temp", "Tccd1"): "cpu_die", ("nvme", "Composite"): "ssd"}.get(
                (name, label), "ram" if name == "spd5118" else None)
            if key and key not in found:
                found[key] = round(celsius, 1)
    return found


def gpu_status(*, run=subprocess.run) -> dict | None:
    fields = "name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,fan.speed," \
             "clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown"
    try:
        out = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True,
                  text=True, timeout=5).stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None
    parts = [x.strip() for x in out.split(",")]

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    return {"name": parts[0], "temp": num(parts[1]), "busy": num(parts[2]), "vram_used": num(parts[3]),
            "vram_total": num(parts[4]), "power": num(parts[5]), "power_limit": num(parts[6]), "fan": num(parts[7]),
            "throttling": "Active" in parts[8:10]}


def _cpu_ticks() -> tuple[int, int]:
    values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
    return sum(values), values[3] + values[4]


def ram_status() -> dict:
    info = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    total, available = info["MemTotal"] / 1048576, info["MemAvailable"] / 1048576
    return {"used": round(total - available, 1), "total": round(total, 1)}


# ------------------------------------------------------------------------------------------------ starting runs

TRAINER = Path("scripts") / "train_rl.py"
RL_CONFIG = Path("src") / "zombiesai" / "rl" / "parallel_ppo.py"
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,47}")
FORCE_AFTER_S = 30.0  # a graceful stop gets this long before the page offers to force one
MAX_TIME_LIMIT_MIN = 7 * 24 * 60
# RLConfig fields that are not settings in the form: the form's own fields set them, or train_rl.py has no flag.
# listen has a flag, but train_rl.py exits at once with it unless ZOMBIES_FLEET_TOKEN is set, which a run started
# from here has only if the dashboard itself was started with it: a run on several PCs is started by hand, as its
# workers on the other PCs are (docs/rl.md, "Several PCs").
NOT_SETTINGS = {"init", "env", "n_actors", "fleet_root", "sim", "listen"}
LEARNING = {"lr", "ent_coef", "kl_coef", "kl_decay", "kl_min", "target_kl", "clip_coef", "gamma", "gae_lambda",
            "update_epochs", "minibatch_size", "batch_steps", "segment_steps", "vf_coef", "max_grad_norm",
            "max_policy_lag", "critic_warmup_updates", "reward_scale"}


def python_for(tree: Path, main: Path) -> Path:
    """The interpreter a run in `tree` uses: the tree's own .venv, else the main checkout's. Either way the tree's
    src/ goes first on PYTHONPATH (see start_run), which is how the fleet's runs have been started: a worktree
    without a venv of its own borrows main's packages, not main's code."""
    for base in (tree, main):
        if (base / ".venv" / "bin" / "python").exists():
            return base / ".venv" / "bin" / "python"
    return Path(sys.executable)


def checkpoints(tree: Path) -> list[dict]:
    """What a run can start from: RL checkpoints (to continue) and BC policies (to fine-tune), newest first. An RL
    checkpoint carries its run's config.json, so the form can offer to continue with the same settings."""
    found = []
    for filename, kind in (("checkpoint.pt", "RL"), ("bc.pt", "BC")):
        for f in (tree / "runs").glob(f"*/{filename}"):
            config = None
            if kind == "RL":
                try:
                    config = json.loads((f.parent / "config.json").read_text())
                except (OSError, ValueError):
                    pass
            found.append({"path": str(f.relative_to(tree)), "run": f.parent.name, "kind": kind,
                          "updated": f.stat().st_mtime, "config": config})
    return sorted(found, key=lambda c: c["updated"], reverse=True)


def trainer_settings(tree: Path) -> list[dict]:
    """RLConfig's fields as the form offers them, read from the tree's own source with `ast` (no torch, no import
    of anything), so a setting added to the trainer appears here without the dashboard changing. Each has the
    flag train_rl.py makes of it, a kind, the code's default, the field's comment as a hint, and a group."""
    try:
        source = (tree / RL_CONFIG).read_text()
        module = ast.parse(source)
    except (OSError, SyntaxError):
        return []
    lines = source.splitlines()
    config = next((n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "RLConfig"), None)
    settings = []
    for node in config.body if config else []:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None):
            continue
        name = node.target.id
        if name in NOT_SETTINGS:
            continue
        try:
            default = ast.literal_eval(node.value)
        except ValueError:  # field(default_factory=...): not something a flag sets
            continue
        annotation = ast.unparse(node.annotation)
        kind = {"bool": "bool", "int": "int", "str": "str"}.get(annotation, "float" if "float" in annotation else "str")
        line = lines[node.lineno - 1]
        settings.append({
            "name": name, "flag": "--" + name.replace("_", "-"), "kind": kind, "default": default,
            "hint": line.split("#", 1)[1].strip() if "#" in line else "",
            "group": "length" if name == "total_steps" else "learning" if name in LEARNING else "run",
        })
    return settings


def _coerce(kind: str, raw):
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        word = str(raw).strip().lower()
        if word in ("true", "1", "yes", "on"):
            return True
        if word in ("false", "0", "no", "off"):
            return False
        raise ValueError
    if kind in ("int", "float"):
        value = float(raw)
        if not math.isfinite(value) or (kind == "int" and not value.is_integer()):
            raise ValueError
        return int(value) if kind == "int" else value
    value = str(raw)
    if not value or any(c in value for c in "\0\n\r"):
        raise ValueError
    return value


def setting_flags(settings: list[dict], values: dict) -> tuple[list[str], str | None]:
    """train_rl.py flags for the values that differ from the code's defaults, each checked against its kind."""
    known = {s["name"]: s for s in settings}
    flags = []
    for name, raw in values.items():
        setting = known.get(name)
        if setting is None:
            return [], f"{name} is not a setting of this checkout's trainer"
        try:
            value = _coerce(setting["kind"], raw)
        except (TypeError, ValueError):
            return [], f"{name}: {raw!r} is not {'an' if setting['kind'] == 'int' else 'a'} {setting['kind']}"
        if value == setting["default"]:
            continue
        if setting["kind"] == "bool":
            flags.append(setting["flag"] if value else "--no-" + setting["flag"][2:])
        else:
            flags += [setting["flag"], str(value)]
    return flags, None


def recent_speeds(tree: Path) -> dict:
    """How fast this tree's newest runs went, per game, for the form's estimate of how long a run takes: the last
    logged steps per second of the newest real-game run and of the newest NachtSim run."""
    speeds = {}
    logs = sorted((tree / "runs").glob("*/metrics.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    for log in logs[:12]:
        try:
            config = json.loads((log.parent / "config.json").read_text())
            last = json.loads(_tail(log, 1)[-1])
        except (OSError, ValueError, IndexError):
            continue
        if config.get("listen"):
            # Trained on several PCs (rl/fleet.py): its sps counts every machine's steps, not n_actors' alone, and
            # actors_alive is only the count at its last update while the sps is the whole run's average, with
            # machines joining and leaving in between. The runs started here use this PC alone, so only those
            # say how fast one would go.
            continue
        env = "real" if config.get("env") == "real-waw" else "sim" if config.get("env") == "nacht-render" else None
        games, sps = config.get("n_actors"), last.get("sps")
        if env and env not in speeds and isinstance(games, int) and games > 0 and isinstance(sps, (int, float)) and sps > 0:
            speeds[env] = {"run": log.parent.name, "sps": sps, "games": games, "per_game": sps / games}
    return speeds


def next_run_name(tree: Path) -> str:
    numbers = [int(m[1]) for d in (tree / "runs").glob("rl*") if (m := re.fullmatch(r"rl(\d+)", d.name))]
    return f"rl{max(numbers, default=0) + 1}"


def fleet_of(tree: Path, *, screens=None) -> dict | None:
    """The tree's fleet (runs/instances/fleet.json) and how many of its games are up right now."""
    try:
        fleet = json.loads((tree / "runs" / "instances" / "fleet.json").read_text())
    except (OSError, ValueError):
        return None
    if screens is None:
        from zombiesai.realgame.viewer import running_screens

        screens = running_screens()
    base, n = int(fleet.get("display_base", 60)), int(fleet.get("n", 0))
    return {"n": n, "display_base": base, "running": sum(1 for s in screens if base <= s.number < base + n)}


def launch_options(repo: Path, trainers: list[dict]) -> dict:
    options = []
    for label, tree in trees(repo):
        if not (tree / TRAINER).exists() or not checkpoints(tree):  # nothing to start from: not a place to run
            continue
        busy = [t["run"] or f"pid {t['pid']}" for t in trainers if t["cwd"] == str(tree) and not t["sim"]]
        options.append({"label": label, "checkpoints": checkpoints(tree)[:40], "next": next_run_name(tree),
                        "fleet": fleet_of(tree), "busy": busy, "settings": trainer_settings(tree),
                        "speeds": recent_speeds(tree)})
    newest = max(options, key=lambda o: o["checkpoints"][0]["updated"] if o["checkpoints"] else 0, default=None)
    return {"trees": options, "default": newest["label"] if newest else None}


def start_run(repo: Path, tree_label: str, init: str, actors: int, name: str, extra: str,
              trainers: list[dict], *, settings: dict | None = None, env: str = "real",
              sim_hardness: float | None = None) -> dict:
    """Start `scripts/train_rl.py` in a tree, as a session of its own (the dashboard can stop and start again
    without taking the run with it), writing runs/<name>/ and printing to runs/<name>.log. Refuses what would
    fail or collide: a fleet with too few games up, a fleet another trainer is already playing, a name in use.
    `settings` are RLConfig values by field name; only those that differ from the code's defaults become flags."""
    tree = dict(trees(repo)).get(tree_label)
    if tree is None or not (tree / TRAINER).exists():
        return {"error": f"no {TRAINER} in {tree_label!r}"}
    if not RUN_NAME.fullmatch(name):
        return {"error": "a run name is letters, digits, '-', '_' and '.', starting with a letter or digit"}
    if (tree / "runs" / name).exists():
        return {"error": f"runs/{name} already exists in {tree_label}"}
    if init not in {c["path"] for c in checkpoints(tree)}:
        return {"error": f"{init} is not a checkpoint in {tree_label}/runs"}
    if not 1 <= actors <= 16:
        return {"error": "between 1 and 16 games"}
    try:
        extra_args = shlex.split(extra)
    except ValueError as error:
        return {"error": f"more options: {error}"}
    if any(a in ("--out", "--actors") or a.startswith(("--out=", "--actors=")) for a in extra_args):
        return {"error": "set the name and the number of games in their own fields"}
    if env not in ("real", "sim"):
        return {"error": "the environment is real or sim"}
    flags, problem = setting_flags(trainer_settings(tree), settings or {})
    if problem:
        return {"error": problem}
    if env == "sim":
        flags = ["--env", "sim"] + (["--sim-hardness", str(float(sim_hardness))] if sim_hardness is not None else []) + flags
    sim = env == "sim" or ("--env" in extra_args[:-1] and extra_args[extra_args.index("--env") + 1] == "sim")
    if not sim:
        fleet = fleet_of(tree)
        if fleet is None:
            return {"error": f"{tree_label} has no fleet (runs/instances/fleet.json): scripts/instances.py up"}
        if fleet["running"] < actors:
            return {"error": f"{fleet['running']} of the fleet's games are running and the run needs {actors}: "
                             f"start them with scripts/instances.py up --n {max(actors, fleet['n'])}"}
        busy = [t for t in trainers if t["cwd"] == str(tree) and not t["sim"]]
        if busy:
            return {"error": f"{busy[0]['run'] or 'a run'} (pid {busy[0]['pid']}) is already playing this fleet"}

    python = python_for(tree, dict(trees(repo))["main"])
    argv = [str(python), "-u", str(TRAINER), init, "--actors", str(actors), "--out", f"runs/{name}", *flags,
            *extra_args]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tree / "src"), env.get("PYTHONPATH")]))
    log = tree / "runs" / f"{name}.log"
    with open(log, "ab") as out:
        out.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} started from the dashboard: "
                  f"{shlex.join(argv[1:])}\n".encode())
        out.flush()
        proc = subprocess.Popen(argv, cwd=tree, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 3.0  # an argument it rejects, a missing fleet: those fail in the first seconds
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.1)
    if proc.poll() is not None:
        return {"error": f"it exited straight away (code {proc.returncode})", "tail": _tail(log, 12), "proc": proc}
    return {"ok": True, "pid": proc.pid, "run": name, "log": str(log), "proc": proc, "cwd": str(tree),
            "command": shlex.join(argv[2:])}


def fleet_command(repo: Path, tree_label: str, action: str, n: int | None = None) -> subprocess.Popen:
    """scripts/instances.py `action` in that checkout, started as its runs are (its src/ first on PYTHONPATH), in
    a session of its own, printing to runs/instances/<action>.log."""
    tree = dict(trees(repo))[tree_label]
    argv = [str(python_for(tree, dict(trees(repo))["main"])), "-u", "scripts/instances.py", action]
    if action == "up" and n:
        argv += ["--n", str(n)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tree / "src"), env.get("PYTHONPATH")]))
    log = tree / "runs" / "instances" / f"{action}.log"
    with open(log, "ab") as out:
        out.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} from the dashboard: {shlex.join(argv[2:])}\n".encode())
        out.flush()
        return subprocess.Popen(argv, cwd=tree, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)


def descendants(pid: int, proc: Path = Path("/proc")) -> list[int]:
    """Every process below `pid`: a trainer's actors, and theirs."""
    children: dict[int, list[int]] = {}
    for entry in proc.iterdir():
        if entry.name.isdigit():
            try:
                ppid = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(ppid, []).append(int(entry.name))
    found, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


class Supervisor:
    """What the page asks for: runs, trainers, games, and the viewer switch. Monitors run only while it asks."""

    def __init__(self, repo: Path, state_dir: Path, *, idle_s: float = 20.0, stale_after: float = 600.0,
                 system: SystemSampler | None = None):
        self.repo, self.state_dir, self.idle_s, self.stale_after = repo, state_dir, idle_s, stale_after
        self.system = system
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "thumbs-off").unlink(missing_ok=True)
        self.monitors: dict[str, subprocess.Popen] = {}
        self.viewer = {"workspace": "9", "fps": 30}
        self.last_seen = 0.0
        self._lock = threading.Lock()
        self._payload, self._payload_at = None, 0.0
        self.stopping: dict[int, float] = {}  # trainer pid -> when a graceful stop was asked for
        self.fleet_ops: dict[str, dict] = {}  # tree label -> the instances.py up/down it is running
        self._cpu_prev = _cpu_ticks()
        self._history: list[tuple[float, dict]] = []  # (time, readings) for the last 10 minutes' peaks
        self.started: list[subprocess.Popen] = []  # runs started here, reaped when they end
        # Time limits, on disk so a dashboard that restarts still keeps them: pid -> stop_at, cwd, run, log.
        self.deadlines_path = self.state_dir / "deadlines.json"
        try:
            self.deadlines: dict[str, dict] = json.loads(self.deadlines_path.read_text())
        except (OSError, ValueError):
            self.deadlines = {}
        threading.Thread(target=self._janitor, daemon=True).start()

    # -- runs
    def payload(self) -> dict:
        with self._lock:
            if self._payload is None or time.monotonic() - self._payload_at > 5.0:
                self._payload = build_payload(run_roots(self.repo), stale_after=self.stale_after)
                self._payload_at = time.monotonic()
            return self._payload

    def live(self) -> dict:
        trainers = live_trainers(self.payload()["run_paths"])
        alive = {t["pid"] for t in trainers}
        self.stopping = {pid: at for pid, at in self.stopping.items() if pid in alive}
        runs = {r["name"]: r for r in self.payload()["runs"]}
        for t in trainers:
            at = self.stopping.get(t["pid"])
            t["stopping_s"] = None if at is None else round(time.time() - at, 1)
            t["can_force"] = at is not None and time.time() - at >= FORCE_AFTER_S
            deadline = self.deadlines.get(str(t["pid"]))
            t["stop_at"] = deadline["stop_at"] if deadline and deadline["cwd"] == t["cwd"] else None
            run = runs.get(t["run"]) if t["run"] else None
            t["progress"] = None if run is None else {k: run.get(k) for k in ("progress", "progress_text", "eta_s")}
        return {"trainers": trainers}

    # -- the machine
    def machine(self) -> dict:
        """Temperatures and load now, and the peak of each over the last 10 minutes of looking."""
        total, idle = _cpu_ticks()
        before_total, before_idle = self._cpu_prev
        self._cpu_prev = (total, idle)
        busy = None if total == before_total else round(100 * (1 - (idle - before_idle) / (total - before_total)), 1)
        temps, gpu, ram = hwmon_temps(), gpu_status(), ram_status()
        now = {"cpu_temp": temps.get("cpu"), "cpu_busy": busy, "gpu_temp": gpu and gpu["temp"],
               "gpu_busy": gpu and gpu["busy"], "ssd_temp": temps.get("ssd"), "ram_temp": temps.get("ram"),
               "vram": gpu and gpu["vram_total"] and round(100 * gpu["vram_used"] / gpu["vram_total"], 1),
               "ram": round(100 * ram["used"] / ram["total"], 1)}
        t = time.time()
        self._history = [(at, r) for at, r in self._history if t - at <= 600] + [(t, now)]
        peaks = {k: max((r[k] for _, r in self._history if r.get(k) is not None), default=None) for k in now}
        levels = {k: level(k, v) for k, v in now.items() if k in LIMITS}
        return {"now": now, "peak": peaks, "levels": levels, "gpu": gpu, "ram": ram,
                "cpu_die": temps.get("cpu_die"), "threads": os.cpu_count(), "limits": LIMITS}

    # -- the fleet
    def fleets(self) -> dict:
        trainers = self.live()["trainers"]
        out = []
        for label, tree in trees(self.repo):
            fleet = fleet_of(tree)
            if fleet is None:
                continue
            op = self.fleet_ops.get(label)
            if op and op["proc"].poll() is not None:
                op["ended"] = op.get("ended") or time.time()
                op["code"] = op["proc"].returncode
            status = None
            if op:
                status = {"action": op["action"], "running": op["proc"].poll() is None, "code": op.get("code"),
                          "since_s": round(time.time() - op["started"]), "tail": _tail(op["log"], 3)}
                if not status["running"] and time.time() - op["ended"] > 60:  # an old result stops being news
                    self.fleet_ops.pop(label, None)
                    status = None
            busy = [t["run"] or f"pid {t['pid']}" for t in trainers if t["cwd"] == str(tree) and not t["sim"]]
            out.append({"label": label, **fleet, "busy": busy, "op": status})
        return {"fleets": out, "recommended": RECOMMENDED_GAMES}

    def fleet_action(self, label: str, action: str, n: int | None = None) -> dict:
        """Start (`up`, to `n` games) or stop (`down`) a checkout's fleet with its own scripts/instances.py. Refuses to
        stop games a run is playing -- that would wreck the run -- and to start one operation over another."""
        fleet = next((f for f in self.fleets()["fleets"] if f["label"] == label), None)
        if fleet is None:
            return {"error": f"{label!r} has no fleet"}
        if action not in ("up", "down"):
            return {"error": "up or down"}
        if fleet["op"] and fleet["op"]["running"]:
            return {"error": f"the fleet is already {'starting' if fleet['op']['action'] == 'up' else 'stopping'}"}
        if action == "down" and fleet["busy"]:
            return {"error": f"{fleet['busy'][0]} is playing these games: stop it first"}
        if action == "up" and not (n and 1 <= n <= 16):
            return {"error": "between 1 and 16 games"}
        tree = dict(trees(self.repo))[label]
        proc = fleet_command(self.repo, label, action, n)
        self.started.append(proc)
        self.fleet_ops[label] = {"action": action, "proc": proc, "started": time.time(),
                                 "log": tree / "runs" / "instances" / f"{action}.log"}
        return {"ok": True}

    # -- starting and stopping runs
    def launch(self) -> dict:
        self._payload = None  # a run that just ended or started changes the names and checkpoints
        return launch_options(self.repo, self.live()["trainers"])

    def start(self, tree: str, init: str, actors: int, name: str, extra: str, *, settings: dict | None = None,
              env: str = "real", sim_hardness: float | None = None, stop_after_min: float = 0.0) -> dict:
        """Start a run; with `stop_after_min`, stop it gracefully (as Ctrl-C) that many minutes in, whichever of
        that and its step budget comes first."""
        if not 0 <= stop_after_min <= MAX_TIME_LIMIT_MIN:
            return {"error": f"a time limit is between 0 (none) and {MAX_TIME_LIMIT_MIN} minutes"}
        result = start_run(self.repo, tree, init, actors, name, extra, self.live()["trainers"], settings=settings,
                           env=env, sim_hardness=sim_hardness)
        if "proc" in result:
            self.started.append(result.pop("proc"))
        if result.get("ok") and stop_after_min > 0:
            result["stop_at"] = time.time() + stop_after_min * 60
            self.deadlines[str(result["pid"])] = {"stop_at": result["stop_at"], "cwd": result["cwd"],
                                                  "run": name, "log": result["log"]}
            self._save_deadlines()
        self._payload = None
        return result

    def _save_deadlines(self) -> None:
        _write_atomic(self.deadlines_path, json.dumps(self.deadlines).encode())

    def enforce_deadlines(self) -> None:
        """Stop, gracefully, every run whose time limit has passed; forget the limits of runs that have ended (or
        whose pid now belongs to something else)."""
        if not self.deadlines:
            return
        trainers = {t["pid"]: t for t in live_trainers({})}
        changed = False
        for key, deadline in list(self.deadlines.items()):
            pid = int(key)
            trainer = trainers.get(pid)
            if trainer is None or trainer["cwd"] != deadline["cwd"]:
                del self.deadlines[key]
                changed = True
            elif time.time() >= deadline["stop_at"] and pid not in self.stopping:
                with open(deadline["log"], "a") as log:  # O_APPEND: safe beside the trainer's own writes
                    log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} time limit reached: the dashboard is "
                              f"stopping it, as Ctrl-C would\n")
                os.kill(pid, signal.SIGINT)
                self.stopping[pid] = time.time()
        if changed:
            self._save_deadlines()

    def stop(self, pid: int, force: bool = False) -> dict:
        """Ask a trainer to stop as Ctrl-C would -- SIGINT to the learner alone, which stops its actors (every key
        released) and writes its checkpoint. `force`, once that has had FORCE_AFTER_S, kills it and everything
        below it: no checkpoint beyond the last periodic one."""
        trainer = next((t for t in self.live()["trainers"] if t["pid"] == pid), None)
        if trainer is None:
            return {"error": f"no trainer with pid {pid}"}
        if not force:
            os.kill(pid, signal.SIGINT)
            self.stopping.setdefault(pid, time.time())
            return {"ok": True}
        if not trainer["can_force"]:
            return {"error": f"ask it to stop first; forcing is offered {FORCE_AFTER_S:.0f} s after"}
        for victim in [*descendants(pid), pid]:
            try:
                os.kill(victim, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return {"ok": True}

    def runs(self) -> dict:
        """The payload, with "running" meaning a trainer process is writing the run right now -- not only that
        its metrics are recent, which is all the static page can know."""
        payload = dict(self.payload())
        writing = {t["run"] for t in self.live()["trainers"]}
        payload["runs"] = [dict(r, status="stopped") if r["status"] == "running" and r["name"] not in writing else r
                           for r in payload["runs"]]
        return payload

    # -- the machine in detail (viz/system.py): what btop shows, with its last half hour
    def system_payload(self, since: float = 0.0) -> dict:
        if self.system is None:
            return {"specs": None, "now": None, "history": [], "history_s": 0}
        return self.system.payload(since)

    # -- games
    def games(self) -> dict:
        self.last_seen = time.monotonic()
        screens = screens_with_stack()
        self._ensure_monitors([s["display"] for s in screens])
        watched = open_viewers()
        for s in screens:
            s.update(fps=None, game=None, thumb_at=None, monitor=False, watched=s["number"] in watched)
            try:
                status = json.loads((self.state_dir / f"{s['number']}.json").read_text())
            except (OSError, ValueError):
                continue
            if time.time() - status["at"] < 5.0:
                s.update(fps=status["fps"], game=status["game"], thumb_at=status["thumb_at"], monitor=True)
        return {"screens": screens, "viewer": self.viewer, "thumbs": not (self.state_dir / "thumbs-off").exists()}

    def _ensure_monitors(self, displays: list[str]) -> None:
        with self._lock:
            for display in displays:
                proc = self.monitors.get(display)
                if proc is None or proc.poll() is not None:
                    self.monitors[display] = subprocess.Popen(
                        [sys.executable, "-m", "zombiesai.viz.supervise", "--monitor", display,
                         "--state", str(self.state_dir)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_monitors(self) -> None:
        with self._lock:
            for proc in self.monitors.values():
                if proc.poll() is None:
                    proc.terminate()
            for proc in self.monitors.values():
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self.monitors.clear()

    def _janitor(self) -> None:
        while True:
            time.sleep(5.0)
            if self.monitors and time.monotonic() - self.last_seen > self.idle_s:
                self.stop_monitors()
            self.started = [p for p in self.started if p.poll() is None]  # reap runs that ended
            try:
                self.enforce_deadlines()
            except OSError:
                pass  # a log that went away, a pid that ended between looking and signalling: next time

    def set_thumbs(self, on: bool) -> None:
        flag = self.state_dir / "thumbs-off"
        flag.unlink(missing_ok=True) if on else flag.touch()

    # -- viewers
    def show(self, displays: list[int], workspace: str, fps: float, focus: bool) -> dict:
        from zombiesai.realgame.viewer import close, running_screens, show

        self.viewer = {"workspace": workspace, "fps": fps}
        screens = [s for s in running_screens() if s.number in displays]
        if screens:
            show(screens, workspace=workspace, fps=fps, focus=focus, say=lambda *a: None)
        else:
            close()
        return {"ok": True}

    def goto(self, workspace: str) -> dict:
        from zombiesai.realgame.instances import hyprland_dispatch

        ok = hyprland_dispatch(f'hl.dsp.focus({{ workspace = "{workspace}" }})', ["workspace", workspace])
        return {"ok": ok}


# ------------------------------------------------------------------------------------------------ server


def page(supervisor: Supervisor) -> str:
    return dashboard_html(
        supervisor.runs(),
        head='<link rel="icon" href="data:,">\n',
        body_before=(HERE / "supervise_panel.html").read_text(),
        script_after="".join((HERE / f).read_text() for f in
                             ("supervise_live.html", "system_view.html", "supervise_system.html")),
    )


class Handler(BaseHTTPRequestHandler):
    supervisor: Supervisor
    port: int

    def log_message(self, *args):  # quiet: the page polls every couple of seconds
        pass

    def _allowed_host(self) -> bool:
        return self.headers.get("Host", "") in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _send(self, body: bytes, kind: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(value, allow_nan=False).encode(), "application/json", status)

    def do_GET(self):
        if not self._allowed_host():
            return self._send(b"wrong host", "text/plain", HTTPStatus.FORBIDDEN)
        path, _, query = self.path.partition("?")
        s = self.supervisor
        if path == "/":
            return self._send(page(s).encode(), "text/html; charset=utf-8")
        if path == "/api/system":  # ?since=<epoch>: only the history the page does not have yet
            since = re.fullmatch(r"since=(\d+(?:\.\d+)?)", query)
            return self._json(s.system_payload(float(since[1]) if since else 0.0))
        if path == "/api/ping":  # how a second launch knows one is already serving (scripts/supervise.py)
            return self._json({"app": "zombiesai-supervise"})
        if path == "/api/runs":
            return self._json(s.runs())
        if path == "/api/live":
            return self._json(s.live())
        if path == "/api/games":
            return self._json(s.games())
        if path == "/api/launch":
            return self._json(s.launch())
        if path == "/api/machine":
            return self._json(s.machine())
        if path == "/api/fleet":
            return self._json(s.fleets())
        if m := re.fullmatch(r"/thumb/(\d+)\.png", path):
            try:
                return self._send((s.state_dir / f"{m[1]}.png").read_bytes(), "image/png")
            except OSError:
                return self._send(b"", "image/png", HTTPStatus.NOT_FOUND)
        self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if not self._allowed_host() or self.headers.get("X-Supervise") != "1":
            return self._send(b"forbidden", "text/plain", HTTPStatus.FORBIDDEN)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        except ValueError:
            return self._json({"error": "bad json"}, HTTPStatus.BAD_REQUEST)
        s = self.supervisor
        workspace = str(body.get("workspace", s.viewer["workspace"]))
        if not re.fullmatch(r"\d{1,2}|name:[\w-]{1,32}", workspace):
            return self._json({"error": "bad workspace"}, HTTPStatus.BAD_REQUEST)
        if self.path == "/api/viewers":
            displays = [int(d) for d in body.get("displays", []) if str(d).isdigit()]
            fps = float(body.get("fps", s.viewer["fps"]))
            return self._json(s.show(displays, workspace, max(0.0, min(fps, 120.0)), bool(body.get("focus"))))
        if self.path == "/api/goto":
            return self._json(s.goto(workspace))
        if self.path == "/api/thumbs":
            s.set_thumbs(bool(body.get("on", True)))
            return self._json({"ok": True})
        if self.path == "/api/start":
            try:
                actors = int(body.get("actors", 0))
            except (TypeError, ValueError):
                return self._json({"error": "games must be a number"}, HTTPStatus.BAD_REQUEST)
            settings = body.get("settings") or {}
            try:
                stop_after_min = float(body.get("stop_after_min") or 0)
                hardness = None if body.get("sim_hardness") in (None, "") else float(body["sim_hardness"])
            except (TypeError, ValueError):
                return self._json({"error": "the time limit and the sim hardness are numbers"}, HTTPStatus.BAD_REQUEST)
            if not isinstance(settings, dict) or (hardness is not None and not 0 <= hardness <= 1):
                return self._json({"error": "bad settings"}, HTTPStatus.BAD_REQUEST)
            result = s.start(str(body.get("tree", "")), str(body.get("init", "")), actors,
                             str(body.get("name", "")).strip(), str(body.get("extra", "")), settings=settings,
                             env=str(body.get("env", "real")), sim_hardness=hardness, stop_after_min=stop_after_min)
            return self._json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT)
        if self.path == "/api/fleet":
            try:
                n = None if body.get("n") in (None, "") else int(body["n"])
            except (TypeError, ValueError):
                return self._json({"error": "games must be a number"}, HTTPStatus.BAD_REQUEST)
            result = s.fleet_action(str(body.get("fleet", "")), str(body.get("action", "")), n)
            return self._json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT)
        if self.path == "/api/stop":
            try:
                pid = int(body.get("pid"))
            except (TypeError, ValueError):
                return self._json({"error": "pid must be a number"}, HTTPStatus.BAD_REQUEST)
            result = s.stop(pid, force=bool(body.get("force")))
            return self._json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT)
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def serve(repo: Path, port: int, state_dir: Path, *, idle_s: float = 20.0, ready=None) -> None:
    system = SystemSampler().start()
    supervisor = Supervisor(repo, state_dir, idle_s=idle_s, system=system)
    handler = type("BoundHandler", (Handler,), {"supervisor": supervisor, "port": port})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    if ready:
        ready(f"http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    finally:
        supervisor.stop_monitors()
        system.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="one game's monitor (started by the supervise server)")
    parser.add_argument("--monitor", required=True, metavar="DISPLAY")
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    monitor(args.monitor, args.state)


if __name__ == "__main__":
    main()
