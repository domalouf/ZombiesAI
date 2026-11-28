"""The dashboard app's side of the Training Room server: recording, the demo library, BC training, presets, the
desktop game, a light summary of every run, and the events worth a notification.

scripts/supervise.py serves the browser page and, through the endpoints below, the native app (app/, `zombies`).
Everything here is wired into viz/supervise.py's `Supervisor` as `supervisor.control`; the page's own logic stays
there. Same rules as the rest of the server: 127.0.0.1 only, the Host header checked, and a command (POST) only
with an `X-Supervise: 1` header.

API
---
GET  /api/ping                 {"app": "zombiesai-supervise"}: how a launcher knows one is already serving
GET  /api/runs                 every run, as the browser page draws it (big: every series, every note)
GET  /api/summary              every run, light: status, progress, headline, up to 4 series, preset, log path;
                               and the live trainers. What the app polls.
GET  /api/live                 the trainer processes: run, pid, elapsed, the tail of their output
GET  /api/launch               what a run can start from (checkpoints per checkout) and the trainer's settings
POST /api/start                start an RL run: {tree, init, actors, name, settings, extra, env, stop_after_min,
                               preset?}; writes runs/<name>.launch.json when given a preset
POST /api/stop                 {pid, force?}: Ctrl-C a trainer (force: kill it, 30 s after asking)
GET  /api/machine              temperatures and load now, and their 10-minute peaks
GET  /api/system?since=<t>     the machine in detail (viz/system.py), with its history
GET  /api/fleet                the fleets per checkout and how many games are up
POST /api/fleet                {fleet, action: up|down, n?}
GET  /api/games                the running game screens; GET /thumb/<display>.png a thumbnail of one
POST /api/viewers, /api/goto, /api/thumbs   the viewer windows on a Hyprland workspace
GET  /api/record               the recording: active, its status off the live bus, log tail, the last one saved,
                               and whether the desktop game is up
POST /api/record               {minutes, notes, pad: ""|"auto"|"/dev/input/eventN", overlay, audio}: record_lan.sh
POST /api/record/stop          SIGINT to the recorder (a clean stop: labels, audio and checks still written)
POST /api/desktop              {action: up|down}: scripts/desktop_game.py, in the background
GET  /api/demos                every recorded session in data/demos (clip.json only), today's minutes against the
                               daily goal, the last 7 days
POST /api/demos/render         {name}: scripts/render_inputs.py on it, in the background
GET  /api/bc                   the next free bc_real<N>, train_bc.py's settings, the demos it can train on
POST /api/bc                   {name, demos, settings, extra, preset?}: start scripts/train_bc.py
GET  /api/presets              saved run settings: [{name, kind: rl|bc, created, request}]
POST /api/presets              {name, kind, request} (same name overwrites); POST /api/presets/delete {name}
GET  /api/events?since=<seq>   what happened since: runs that ended (finished, stopped, crashed), recordings
                               saved, renders and BC runs done; each also a desktop notification

Every long job (a recording, a BC run, a render, the desktop game going up) runs in a session of its own, so the
server can stop and start again without taking it along.
"""

import ast
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path

DEMOS = Path("data") / "demos"
GOAL_MIN = 20.0  # the user's daily recording goal, in minutes of play
MAX_POINTS = 300
RECORDER = re.compile(r"(^|/)scripts/record_demo\.py$")
PAD = re.compile(r"/dev/input/event\d{1,3}")
BC_TRAINER = Path("scripts") / "train_bc.py"
BC_CONFIG = Path("src") / "zombiesai" / "demos" / "bc.py"
# train_bc.py's flags that are not settings in the form: the form's own fields set them (the demos, the name), or
# they name paths a form has no business typing.
BC_NOT_SETTINGS = {"clips", "out", "val_clips", "audio_cache"}
# The series worth a small chart, per kind of run, after its headline: the first three that exist.
SUMMARY_SERIES = {
    "ppo": ("round_reached_mean", "length_mean", "entropy", "kl_bc", "explained_variance", "sps"),
    "bc": ("val.nll", "loss.policy", "val.accuracy.mean", "loss.total"),
    "idm": ("loss.total", "val.accuracy.mean"),
}
EVENT_RING = 200


def default_presets_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "zombiesai" / "presets.json"


def write_atomic(path: Path, data: bytes) -> None:
    """Write beside the target, then rename over it: a reader sees the old file or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def tail(path: Path | None, lines: int = 25, max_bytes: int = 32768) -> list[str]:
    if path is None:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    return [line for line in text.replace("\r", "\n").splitlines() if line.strip()][-lines:]


# ------------------------------------------------------------------------------------------------ the summary


def downsample(x: list, y: list, n: int = MAX_POINTS) -> tuple[list, list]:
    """At most `n` points, evenly spaced through the series, always keeping the last (what a chart ends on)."""
    if len(x) <= n:
        return list(x), list(y)
    step = (len(x) - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)})
    return [x[i] for i in idx], [y[i] for i in idx]


def read_launch(run_path: Path) -> dict | None:
    """What the dashboard recorded when it started this run (runs/<name>.launch.json), if it did."""
    try:
        return json.loads(run_path.with_name(run_path.name + ".launch.json").read_text())
    except (OSError, ValueError):
        return None


def write_launch(runs_dir: Path, name: str, preset: str | None, request: dict, origin: str = "the dashboard app") -> Path:
    path = runs_dir / f"{name}.launch.json"
    request = {k: v for k, v in request.items() if k != "preset"}
    write_atomic(path, json.dumps({"preset": preset or None, "request": request, "started": time.time(),
                                   "origin": origin}, indent=2).encode())
    return path


def summarize_run(run: dict, path: Path | None, tree: str) -> dict:
    """One run of the full payload (viz/dashboard.py `read_run`), light enough to poll."""
    head_key = run.get("headline")
    series = run.get("series") or {}
    headline = None
    if head_key and head_key in series:
        s = series[head_key]
        headline = {"key": head_key, "label": s.get("label"), "last": s.get("last"), "best": s.get("best"),
                    "trend": (s.get("trend") or {}).get("verdict")}
    keys = [head_key] if head_key in series else []
    keys += [k for k in SUMMARY_SERIES.get(run.get("kind"), ()) if k in series and k not in keys][:3]
    if len(keys) < 4:  # a kind without a list, or one whose favourites are missing: the first others logged
        keys += [k for k in series if k not in keys][:4 - len(keys)]
    small = {}
    for key in keys:
        x, y = downsample(series[key].get("x") or [], series[key].get("y") or [])
        small[key] = {"label": series[key].get("label") or key, "x": x, "y": y}
    launch = read_launch(path) if path else None
    log = path.with_name(path.name + ".log") if path else None
    return {
        "name": run.get("name"), "path": str(path) if path else None, "tree": tree, "kind": run.get("kind"),
        "env": run.get("env"), "status": run.get("status"), "progress": run.get("progress"),
        "progress_text": run.get("progress_text"), "elapsed_s": run.get("elapsed_s"), "eta_s": run.get("eta_s"),
        "updated": run.get("updated"), "headline": headline, "series": small,
        "notes": [n["text"] for n in run.get("notes") or [] if n.get("level") in ("critical", "serious")],
        "preset": (launch or {}).get("preset"),
        "log": str(log) if log is not None and log.exists() else None,
    }


def summary(payload: dict, roots: list[tuple[str, Path]], trainers: list[dict]) -> dict:
    """The app's view of every run: `payload` is the Supervisor's (status already corrected for live trainers)."""
    by_name = {name: Path(path) for path, name in payload.get("run_paths", {}).items()}
    out = []
    for run in payload.get("runs", []):
        path = by_name.get(run["name"])
        tree = next((label for label, root in roots if path is not None and path.parent == root.resolve()),
                    roots[0][0] if roots else "main")
        out.append(summarize_run(run, path, tree))
    return {"runs": out, "trainers": trainers}


# ------------------------------------------------------------------------------------------------ demos


def read_demo(path: Path) -> dict | None:
    """One session's card, from its clip.json alone: never the frames."""
    try:
        manifest = json.loads((path / "clip.json").read_text())
    except (OSError, ValueError):
        return None
    summary_ = manifest.get("summary") or {}
    steps = manifest.get("n_steps") or 0
    seconds = summary_.get("seconds", steps / 15)
    idle = summary_.get("not_playing_seconds") or 0.0
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    created = manifest.get("created_unix") or source.get("recorded_unix") or (path / "clip.json").stat().st_mtime
    gamepad = manifest.get("gamepad")
    video = path / "inputs.mp4"
    return {
        "name": path.name, "path": str(path.resolve()), "created": created,
        "minutes": round(seconds / 60, 2), "play_minutes": round((seconds - idle) / 60, 2),
        "not_playing_minutes": round(idle / 60, 2), "notes": summary_.get("notes", ""),
        "scene": "scene" in ((manifest.get("hud") or {}).get("shapes") or {}),
        "audio": isinstance(manifest.get("audio"), dict),
        "gamepad": (gamepad.get("device") if isinstance(gamepad, dict) else None),
        "stale_rate": summary_.get("stale_rate"), "overrun_rate": summary_.get("overrun_rate"),
        "capture_lost": summary_.get("capture_lost"), "status": manifest.get("status", "open"),
        "video": str(video.resolve()) if video.exists() else None,
        "size_mb": round(dir_size(path) / 1e6, 1),
    }


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def week(demos: list[dict], today: date, days: int = 7) -> list[dict]:
    """Play minutes recorded on each of the last `days` days, today last, by the session's start (local time)."""
    totals: dict[str, float] = {}
    for d in demos:
        day = datetime.fromtimestamp(d["created"]).date().isoformat()
        totals[day] = totals.get(day, 0.0) + d["play_minutes"]
    return [{"date": (today - timedelta(days=i)).isoformat(),
             "minutes": round(totals.get((today - timedelta(days=i)).isoformat(), 0.0), 1)}
            for i in range(days - 1, -1, -1)]


class DemoLibrary:
    """data/demos as cards, each re-read only when its clip.json or its rendered video changes."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, tuple[tuple, dict]] = {}

    def demos(self) -> list[dict]:
        found = []
        for path in sorted(self.root.glob("demo_*")) if self.root.is_dir() else []:
            try:
                key = ((path / "clip.json").stat().st_mtime, (path / "inputs.mp4").exists())
            except OSError:
                continue
            cached = self._cache.get(path.name)
            if cached is None or cached[0] != key:
                card = read_demo(path)
                if card is None:
                    continue
                cached = (key, card)
                self._cache[path.name] = cached
            found.append(dict(cached[1]))
        return sorted(found, key=lambda d: d["created"], reverse=True)

    def payload(self, rendering: set[str], today: date | None = None) -> dict:
        demos = self.demos()
        for d in demos:
            d["rendering"] = d["name"] in rendering
        days = week(demos, today or date.today())
        return {"root": str(self.root.resolve()), "demos": demos, "today_min": days[-1]["minutes"],
                "goal_min": GOAL_MIN, "week": days,
                "total_play_min": round(sum(d["play_minutes"] for d in demos), 1)}


# ------------------------------------------------------------------------------------------------ BC settings


def _literal(node, config: dict):
    """A default as written in train_bc.py: a literal, or BCConfig.<field> looked up in BCConfig's source."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "BCConfig":
        return config.get(node.attr)
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def bc_config_defaults(tree: Path) -> dict:
    try:
        module = ast.parse((tree / BC_CONFIG).read_text())
    except (OSError, SyntaxError):
        return {}
    cls = next((n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "BCConfig"), None)
    out = {}
    for node in cls.body if cls else []:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


BC_GROUPS = {"epochs": "length", "lr": "learning", "batch_size": "learning", "focal_gamma": "learning",
             "class_balance": "learning", "class_balance_power": "learning", "hidden": "learning",
             "val_fraction": "learning", "min_confidence": "learning", "correction_weight": "learning"}


def bc_settings(tree: Path) -> list[dict]:
    """train_bc.py's flags as the form offers them, read with `ast` (no torch), in the same shape as the RL
    trainer's settings (viz/supervise.py `trainer_settings`). A flag taking several values is kind "list" (its
    value is the numbers separated by spaces); a store_true flag is a bool defaulting to false."""
    try:
        module = ast.parse((tree / BC_TRAINER).read_text())
    except (OSError, SyntaxError):
        return []
    config = bc_config_defaults(tree)
    settings = []
    for node in ast.walk(module):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args and isinstance(node.args[0], ast.Constant)):
            continue
        flag = node.args[0].value
        if not str(flag).startswith("--"):
            continue
        name = flag[2:].replace("-", "_")
        if name in BC_NOT_SETTINGS:
            continue
        kw = {k.arg: k.value for k in node.keywords}
        action = kw.get("action") and _literal(kw["action"], config)
        hint = _literal(kw["help"], config) if "help" in kw else ""
        if action == "store_true":
            kind, default = "bool", False
        else:
            typ = kw["type"].id if isinstance(kw.get("type"), ast.Name) else "str"
            kind = {"int": "int", "float": "float"}.get(typ, "str")
            default = _literal(kw["default"], config) if "default" in kw else None
            if "nargs" in kw:
                kind = "list"
                default = " ".join(str(v) for v in default) if isinstance(default, (list, tuple)) else ""
        settings.append({"name": name, "flag": flag, "kind": kind, "default": default,
                         "hint": " ".join(str(hint or "").split()),
                         "group": BC_GROUPS.get(name, "run")})
    return settings


def bc_flags(settings: list[dict], values: dict) -> tuple[list[str], str | None]:
    """train_bc.py flags for the values that differ from its defaults, each checked against its kind."""
    known = {s["name"]: s for s in settings}
    flags: list[str] = []
    for name, raw in values.items():
        s = known.get(name)
        if s is None:
            return [], f"{name} is not a setting of train_bc.py"
        kind = s["kind"]
        try:
            if kind == "bool":
                value = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("true", "1", "yes", "on")
                if value:
                    flags.append(s["flag"])
                continue
            if kind == "list":
                parts = str(raw).replace(",", " ").split()
                if " ".join(parts) == (s["default"] or ""):
                    continue
                [int(p) for p in parts]
                if parts:
                    flags += [s["flag"], *parts]
                continue
            if kind in ("int", "float"):
                number = float(raw)
                if kind == "int" and not number.is_integer():
                    raise ValueError
                value = int(number) if kind == "int" else number
            else:
                value = str(raw)
                if not value or any(c in value for c in "\0\n\r"):
                    raise ValueError
        except (TypeError, ValueError):
            return [], f"{name}: {raw!r} is not {'an' if kind == 'int' else 'a'} {kind}"
        if value != s["default"]:
            flags += [s["flag"], str(value)]
    return flags, None


def next_bc_name(runs_dir: Path) -> str:
    numbers = [int(m[1]) for d in runs_dir.glob("bc_real*") if (m := re.fullmatch(r"bc_real(\d+)", d.name))]
    return f"bc_real{max(numbers, default=0) + 1}"


# ------------------------------------------------------------------------------------------------ presets


class PresetStore:
    """Named run settings, in one JSON file written atomically."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_presets_path()

    def list(self) -> list[dict]:
        try:
            presets = json.loads(self.path.read_text()).get("presets", [])
        except (OSError, ValueError, AttributeError):
            return []
        return [p for p in presets if isinstance(p, dict) and "name" in p]

    def save(self, name: str, kind: str, request: dict) -> dict:
        name = str(name).strip()
        if not name or len(name) > 60 or any(c in name for c in "\0\n\r"):
            return {"error": "a preset needs a name (60 characters at most)"}
        if kind not in ("rl", "bc"):
            return {"error": "a preset is for an rl run or a bc run"}
        if not isinstance(request, dict):
            return {"error": "a preset's request is an object"}
        presets = [p for p in self.list() if p["name"] != name]
        preset = {"name": name, "kind": kind, "created": time.time(),
                  "request": {k: v for k, v in request.items() if k != "preset"}}
        presets.append(preset)
        presets.sort(key=lambda p: p["name"].lower())
        write_atomic(self.path, json.dumps({"presets": presets}, indent=2).encode())
        return {"ok": True, "preset": preset}

    def delete(self, name: str) -> dict:
        presets = self.list()
        kept = [p for p in presets if p["name"] != name]
        if len(kept) == len(presets):
            return {"error": f"no preset called {name!r}"}
        write_atomic(self.path, json.dumps({"presets": kept}, indent=2).encode())
        return {"ok": True}


# ------------------------------------------------------------------------------------------------ processes


def _cmdline(entry: Path) -> list[str]:
    try:
        return [a.decode(errors="replace") for a in (entry / "cmdline").read_bytes().split(b"\0") if a]
    except OSError:
        return []


def find_recorders(proc: Path = Path("/proc")) -> list[dict]:
    """Every recorder: a Python interpreter running scripts/record_demo.py -- not the `uv run` that started it, so a
    SIGINT reaches the one process that turns it into a clean stop, once."""
    from zombiesai.viz.supervise import _proc_start_epoch

    found = []
    for entry in proc.iterdir() if proc.is_dir() else []:
        if not entry.name.isdigit():
            continue
        argv = _cmdline(entry)
        if len(argv) < 2 or not re.search(r"python[\d.]*$", Path(argv[0]).name):
            continue
        script = next((a for a in argv[1:3] if RECORDER.search(a)), None)
        if script is None:
            continue
        log = None
        try:
            out = os.readlink(entry / "fd" / "1")
            log = out if out.startswith("/") and Path(out).is_file() else None
        except OSError:
            pass
        found.append({"pid": int(entry.name), "argv": argv, "started": _proc_start_epoch(int(entry.name)),
                      "log": log})
    return sorted(found, key=lambda r: r["pid"])


def classify_end(lines: list[str]) -> str:
    """How a trainer's output ended: "crashed" (a traceback or a fatal error), "stopped" (interrupted, as by
    Ctrl-C or the dashboard's time limit), or "finished"."""
    text = "\n".join(lines[-40:])
    if re.search(r"^Traceback \(most recent call last\)|^\w*(Error|Exception)\b|^Killed\b|^Segmentation fault",
                 text, re.M):
        return "crashed"
    if re.search(r"interrupted|SIGINT|stopping it, as Ctrl-C|KeyboardInterrupt|time limit reached", text, re.I):
        return "stopped"
    if re.search(r"^checkpoint: |^next: scripts/eval_bc", text, re.M):
        return "finished"
    return "stopped"


def notify_send(title: str, text: str, level: str = "info") -> None:
    """A desktop notification, never waited on, and nothing at all where there is no notify-send."""
    if shutil.which("notify-send") is None:
        return
    urgency = {"bad": "critical", "warn": "normal"}.get(level, "low")
    try:
        subprocess.Popen(["notify-send", "--app-name", "ZombiesAI", "--urgency", urgency, title, text],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError:
        pass


class Events:
    """The last EVENT_RING things that happened, numbered, each also a notification."""

    def __init__(self, notify=notify_send, clock=time.time):
        self.notify, self.clock = notify, clock
        self.ring: deque[dict] = deque(maxlen=EVENT_RING)
        self.seq = 0
        self._lock = threading.Lock()

    def add(self, level: str, title: str, text: str = "", *, notify: bool = True) -> dict:
        with self._lock:
            self.seq += 1
            event = {"seq": self.seq, "t": self.clock(), "level": level, "title": title, "text": text}
            self.ring.append(event)
        if notify and self.notify is not None:
            try:
                self.notify(title, text, level)
            except Exception:  # noqa: BLE001 -- a notification never breaks the server
                pass
        return event

    def since(self, seq: int) -> dict:
        with self._lock:
            return {"events": [e for e in self.ring if e["seq"] > seq], "seq": self.seq}


# ------------------------------------------------------------------------------------------------ control


class Control:
    """The app's endpoints over one checkout (`repo`, the main one): jobs it started, what it last saw running,
    and the events in between. `popen`, `kill`, `notify` and `proc` are injectable for tests."""

    def __init__(self, supervisor, *, presets: Path | None = None, popen=subprocess.Popen, kill=os.kill,
                 notify=notify_send, proc: Path = Path("/proc"), subscribe: bool = True):
        self.supervisor = supervisor
        self.repo = Path(supervisor.repo)
        self.popen, self.kill, self.proc = popen, kill, proc
        self.events = Events(notify)
        self.presets = PresetStore(presets)
        self.library = DemoLibrary(self.repo / DEMOS)
        self.state_path = Path(supervisor.state_dir) / "control.json"
        try:
            self.state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            self.state = {}
        self.jobs: dict[str, dict] = {}  # "render:<demo>", "desktop", "record", "bc:<name>" -> {proc, ...}
        self._subscribe = subscribe
        self._bus = None
        self._seen_trainers: dict[int, dict] | None = None
        self._seen_recorders: set[int] | None = None
        self._recorder_logs: dict[int, str] = {}
        self._recorder_started: dict[int, float] = {}
        self._lock = threading.Lock()
        self._summary, self._summary_at = None, 0.0

    # -- the background watch
    def start(self, every_s: float = 5.0) -> "Control":
        def loop():
            while True:
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 -- one bad look must not end the watch
                    pass
                time.sleep(every_s)

        threading.Thread(target=loop, daemon=True, name="control-watch").start()
        return self

    def _save_state(self) -> None:
        write_atomic(self.state_path, json.dumps(self.state).encode())

    def tick(self) -> None:
        """Notice what ended since the last look: trainers, the recording, and the jobs started here."""
        from zombiesai.viz.supervise import live_trainers

        trainers = {t["pid"]: t for t in live_trainers({}, self.proc)}
        if self._seen_trainers is not None:
            for pid, t in self._seen_trainers.items():
                if pid in trainers:
                    continue
                what = classify_end(tail(Path(t["log"]), 40) if t.get("log") else [])
                name = run_name(t)
                level = {"crashed": "bad", "finished": "good"}.get(what, "warn")
                self.events.add(level, f"{name} {what}", f"{t['script']} (pid {pid}) in {t['tree']}"
                                + (f": {tail(Path(t['log']), 1)[0]}" if t.get("log") and tail(Path(t['log']), 1) else ""))
        self._seen_trainers = trainers

        found = find_recorders(self.proc)
        recorders = {r["pid"] for r in found}
        self._recorder_logs.update({r["pid"]: r["log"] for r in found if r["log"]})
        self._recorder_started.update({r["pid"]: r["started"] for r in found if r["started"]})
        if self._seen_recorders is not None and self._seen_recorders - recorders and not recorders:
            newest = next(iter(self.library.demos()), None)
            gone = sorted(self._seen_recorders - recorders)
            started = min((self._recorder_started.get(pid) or time.time() for pid in gone), default=time.time())
            if newest is not None and newest["created"] < started - 5:  # gave up before writing anything
                newest = None
            log = (self.state.get("current") or {}).get("log") or self._recorder_logs.get(gone[-1])
            self.state["last"] = {"name": newest["name"] if newest else None,
                                  "path": newest["path"] if newest else None, "ended": time.time(), "log": log}
            self.state.pop("current", None)
            self._save_state()
            if newest:
                self.events.add("good", "Recording saved",
                                f"{newest['name']}: {newest['play_minutes']:.1f} min of play")
            else:
                self.events.add("warn", "Recording ended",
                                "nothing was recorded (the game window never came up?)")
        self._seen_recorders = recorders

        for key, job in list(self.jobs.items()):
            code = job["proc"].poll()
            if code is None:
                continue
            del self.jobs[key]
            if job.get("silent"):  # its own event comes from what it did (a recording: "Recording saved")
                continue
            ok = code == 0
            title = job["done"] if ok else job["failed"]
            last = tail(job.get("log"), 1)
            self.events.add("good" if ok else "bad", title, last[0] if last else f"exit code {code}")

    # -- the summary
    def summary(self) -> dict:
        from zombiesai.viz.supervise import run_roots

        with self._lock:
            if self._summary is None or time.monotonic() - self._summary_at > 5.0:
                s = self.supervisor
                self._summary = summary(s.runs(), run_roots(self.repo), s.live()["trainers"])
                self._summary_at = time.monotonic()
            return self._summary

    # -- recording
    def _status(self) -> dict | None:
        if self._bus is None and self._subscribe:
            from zombiesai.viz.live_bus import Subscriber, bus_path

            try:
                self._bus = Subscriber(bus_path("record-status"))
            except OSError:
                self._subscribe = False
        if self._bus is None:
            return None
        self._bus.poll()
        return self._bus.get("recorder", max_age_s=5.0)

    def desktop_running(self) -> bool:
        try:
            pid = int((self.repo / "runs" / "desktop" / "game.pid").read_text())
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False

    def record_payload(self) -> dict:
        recorders = find_recorders(self.proc)
        recorder = recorders[0] if recorders else None
        # The request is known only for a recording started here, while the `uv run` it became is alive: one
        # started from a terminal shows as active, with no request.
        current = self.state.get("current") if recorder else None
        if current and not pid_alive(current.get("wrapper")):
            current = None
        log = (current or {}).get("log") or (recorder or {}).get("log")
        desktop = self.jobs.get("desktop")
        return {
            "active": recorder is not None, "pid": recorder["pid"] if recorder else None,
            "started": recorder["started"] if recorder else None,
            "request": (current or {}).get("request"),
            "status": self._status() if recorder else None,
            "log": log, "log_tail": tail(Path(log)) if log else [],
            "last": self.state.get("last"),
            "game": {"running": self.desktop_running(), "op": desktop["action"] if desktop else None},
        }

    def record(self, body: dict) -> dict:
        if find_recorders(self.proc) or ("record" in self.jobs and self.jobs["record"]["proc"].poll() is None):
            return {"error": "a recording is already running: stop it first"}
        try:
            minutes = float(body.get("minutes", 20))
        except (TypeError, ValueError):
            return {"error": "minutes must be a number"}
        if not 1 <= minutes <= 120:
            return {"error": "a recording is between 1 and 120 minutes"}
        notes = str(body.get("notes", "")).strip()
        if any(c in notes for c in "\0\n\r") or len(notes) > 300:
            return {"error": "notes are one line of at most 300 characters"}
        pad = str(body.get("pad") or "")
        if pad not in ("", "auto") and not PAD.fullmatch(pad):
            return {"error": "pad is empty, 'auto' or /dev/input/eventN"}
        overlay, audio = bool(body.get("overlay", True)), bool(body.get("audio", True))
        env = dict(os.environ, ZOMBIESAI_OVERLAY="1" if overlay else "0", ZOMBIESAI_AUDIO="1" if audio else "0")
        env.pop("ZOMBIESAI_PAD", None)
        if pad:
            env["ZOMBIESAI_PAD"] = pad
        logs = self.repo / "runs" / "record"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{time.strftime('%Y%m%d-%H%M%S')}.log"
        argv = ["scripts/record_lan.sh", f"{minutes:g}", notes or "recorded from the dashboard"]
        with open(log, "ab") as out:
            out.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} from the dashboard: {shlex.join(argv)} "
                      f"(overlay {overlay}, audio {audio}, pad {pad or 'none'})\n".encode())
            out.flush()
            proc = self.popen(argv, cwd=self.repo, env=env, stdin=subprocess.DEVNULL, stdout=out,
                              stderr=subprocess.STDOUT, start_new_session=True)
        request = {"minutes": minutes, "notes": notes, "pad": pad, "overlay": overlay, "audio": audio}
        self.jobs["record"] = {"proc": proc, "silent": True, "log": log}
        self.state["current"] = {"request": request, "log": str(log), "wrapper": proc.pid, "started": time.time()}
        self._save_state()
        self.events.add("info", "Recording starting",
                        f"{minutes:g} min: switch to the game (it starts after a countdown)", notify=False)
        return {"ok": True, "pid": proc.pid, "log": str(log)}

    def stop_recording(self) -> dict:
        recorders = find_recorders(self.proc)
        if not recorders:
            return {"error": "no recording is running"}
        for r in recorders:
            self.kill(r["pid"], signal.SIGINT)
        return {"ok": True, "pids": [r["pid"] for r in recorders]}

    def desktop(self, action: str) -> dict:
        if action not in ("up", "down"):
            return {"error": "up or down"}
        job = self.jobs.get("desktop")
        if job and job["proc"].poll() is None:
            return {"error": f"the game is already {'starting' if job['action'] == 'up' else 'stopping'}"}
        if action == "down" and find_recorders(self.proc):
            return {"error": "a recording is running on the game: stop it first"}
        log = self.repo / "runs" / "desktop" / f"{action}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        python = self.repo / ".venv" / "bin" / "python"
        argv = [str(python if python.exists() else "python3"), "-u", "scripts/desktop_game.py", action]
        with open(log, "ab") as out:
            proc = self.popen(argv, cwd=self.repo, env=dict(os.environ), stdin=subprocess.DEVNULL, stdout=out,
                              stderr=subprocess.STDOUT, start_new_session=True)
        self.jobs["desktop"] = {"proc": proc, "action": action, "log": log,
                                "done": "Game is up" if action == "up" else "Game closed",
                                "failed": f"The game failed to {'start' if action == 'up' else 'stop'}"}
        return {"ok": True}

    # -- demos
    def demos_payload(self) -> dict:
        rendering = {key.split(":", 1)[1] for key, job in self.jobs.items()
                     if key.startswith("render:") and job["proc"].poll() is None}
        return self.library.payload(rendering)

    def render(self, name: str) -> dict:
        path = self.repo / DEMOS / str(name)
        if not re.fullmatch(r"demo_\d{4,}", str(name)) or not (path / "clip.json").exists():
            return {"error": f"no demo called {name!r}"}
        key = f"render:{name}"
        if key in self.jobs and self.jobs[key]["proc"].poll() is None:
            return {"error": f"{name} is already rendering"}
        python = self.repo / ".venv" / "bin" / "python"
        log = path / "render_inputs.log"
        with open(log, "wb") as out:
            proc = self.popen([str(python if python.exists() else "python3"), "-u", "scripts/render_inputs.py",
                               str(path)], cwd=self.repo, env=dict(os.environ), stdin=subprocess.DEVNULL,
                              stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        self.jobs[key] = {"proc": proc, "log": log, "done": f"{name} rendered",
                          "failed": f"Rendering {name} failed"}
        return {"ok": True}

    # -- BC
    def bc_payload(self) -> dict:
        return {"next": next_bc_name(self.repo / "runs"), "settings": bc_settings(self.repo),
                "demos": [d["name"] for d in self.library.demos()]}

    def start_bc(self, body: dict, trainers: list[dict]) -> dict:
        from zombiesai.viz.supervise import RUN_NAME

        name = str(body.get("name") or "").strip()
        if not RUN_NAME.fullmatch(name):
            return {"error": "a run name is letters, digits, '-', '_' and '.', starting with a letter or digit"}
        runs = self.repo / "runs"
        if (runs / name).exists():
            return {"error": f"runs/{name} already exists"}
        busy = [t for t in trainers if t["script"] == "train_bc.py" and t["cwd"] == str(self.repo)]
        if busy:
            return {"error": f"a BC run is already training (pid {busy[0]['pid']})"}
        available = {d["name"] for d in self.library.demos()}
        demos = [str(d) for d in body.get("demos") or []]
        unknown = [d for d in demos if d not in available]
        if unknown:
            return {"error": f"no demo called {unknown[0]!r}"}
        if not available:
            return {"error": "there are no demos to train on: record some first"}
        settings = body.get("settings") or {}
        if not isinstance(settings, dict):
            return {"error": "bad settings"}
        flags, problem = bc_flags(bc_settings(self.repo), settings)
        if problem:
            return {"error": problem}
        try:
            extra = shlex.split(str(body.get("extra") or ""))
        except ValueError as error:
            return {"error": f"more options: {error}"}
        if any(a == "--out" or a.startswith("--out=") for a in extra):
            return {"error": "set the name in its own field"}
        clips = [str(DEMOS / d) for d in demos] if demos else [str(DEMOS)]
        python = self.repo / ".venv" / "bin" / "python"
        argv = [str(python if python.exists() else "python3"), "-u", str(BC_TRAINER), *clips,
                "--out", f"runs/{name}", *flags, *extra]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self.repo / "src"), env.get("PYTHONPATH")]))
        runs.mkdir(parents=True, exist_ok=True)
        log = runs / f"{name}.log"
        with open(log, "ab") as out:
            out.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} started from the dashboard app: "
                      f"{shlex.join(argv[1:])}\n".encode())
            out.flush()
            proc = self.popen(argv, cwd=self.repo, env=env, stdin=subprocess.DEVNULL, stdout=out,
                              stderr=subprocess.STDOUT, start_new_session=True)
        self.supervisor.started.append(proc)
        write_launch(runs, name, body.get("preset"), body)  # the request, so the app can show what a run used
        self.supervisor._payload = None
        return {"ok": True, "pid": proc.pid, "run": name, "log": str(log), "command": shlex.join(argv[2:])}

    def note_launch(self, tree_label: str, name: str, body: dict) -> None:
        """After /api/start succeeds: remember the request (and its preset) beside the run."""
        from zombiesai.viz.supervise import trees

        tree = dict(trees(self.repo)).get(tree_label)
        if tree is not None and (tree / "runs").is_dir():
            write_launch(tree / "runs", name, body.get("preset"), body)


def pid_alive(pid) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_name(trainer: dict) -> str:
    if trainer.get("run"):
        return trainer["run"]
    args = trainer.get("args", "").split()
    if "--out" in args[:-1]:
        return Path(args[args.index("--out") + 1]).name
    return trainer.get("script", "a trainer")
