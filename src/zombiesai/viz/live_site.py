"""The Training Room on domalouf.com, live: this PC pushes what the page shows, and nothing reaches in.

Three files, written to a staging directory and rsynced (over one reused SSH connection) to the site's
zombies/training/live/:

    machine.json  every 5 s   the machine as viz/system.py samples it, and the runs training right now
    runs.json     every 60 s  every run's curves: the payload the static training page is built from
    stream.json   every 5 s   the stream overlay's numbers and its page's details (viz/stream.py)

That is while something trains. With nothing training, the training PC only looks for a trainer every few
seconds (a read of /proc) and reports every 5 minutes, its sampler asleep in between; a trainer starting, or the
last one ending, is reported at once. machine.json and stream.json carry `every_s`, how soon the next report is
due, so the pages tell "not training" from "offline".

The other gaming PCs, which play for the learner (rl/fleet.py), push one file each beside them:

    machine-<id>.json  every 5 s  that machine, and what its worker is doing: waiting, or playing for which run

The page itself (scripts/dashboard.py --site ... --live) is the static training page with the machines and
"Training now" added, polling these; the stream's page and overlay (zombies/live/) poll stream.json. All are
public, so all go through the same scrub as the site build: no paths, no command lines, no host name, no
process ids -- a process is its name and its load, and a run is its name. A machine is the label it was given.
"""

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from zombiesai.viz.dashboard import public_payload, write_dashboard_site
from zombiesai.viz.stream import StreamFeed
from zombiesai.viz.supervise import build_payload, live_trainers, run_roots
from zombiesai.viz.system import SystemSampler

SITE_LINKS = [("← domalouf.com", "/"), ("Live stream", "/zombies/live/"),
              ("Code on GitHub", "https://github.com/domalouf/ZombiesAI")]
LIVE_INTRO = (
    "How the reinforcement-learning agent's training is going, live from the gaming PC it trains on: the machine "
    "as it works, the runs training right now, and every run its trainers have written -- PPO on the real game, "
    "behavioural cloning from human play, and the inverse dynamics model."
)
PROC_FIELDS = ("name", "cpu", "rss", "mem_pct", "threads")
# A worker machine's id is its file name on the site and nothing else: short, lower case, no dots or slashes.
MACHINE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
LEARNER_LABEL = "Training PC"
# What a worker's status may say in public (rl/fleet.py writes more, for the person at that machine).
WORKER_FIELDS = ("state", "run", "actors", "alive", "sent", "dropped", "version", "episodes", "last_round",
                 "best_round")
WORKER_STALE_S = 30.0  # the worker rewrites its status every 5-10 s; older than this, it is not running
WARMUP_S = 10.0  # an idle learner wakes its sampler this long before reporting: a sample to take rates from


def machine_id(value: str) -> str:
    if not MACHINE_ID.fullmatch(value or ""):
        raise ValueError(f"machine id {value!r}: 1-32 of a-z, 0-9 and -, starting with a letter or digit")
    return value


def worker_file(mid: str) -> str:
    return f"machine-{machine_id(mid)}.json"


def worker_status(path: Path, now: float) -> dict:
    """The worker's status file (rl/fleet.py), as the public page may show it."""
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"state": "not running"}
    if not isinstance(status, dict) or now - float(status.get("at") or 0) > WORKER_STALE_S:
        return {"state": "not running"}
    out = {k: status[k] for k in WORKER_FIELDS if k in status}
    if out.get("state") not in ("waiting", "playing", "refused", "stopped"):
        out["state"] = "not running"
    if out.get("run") is not None:
        out["run"] = Path(str(out["run"])).name
    return out


def downsample(points: list[dict], step_s: float) -> list[dict]:
    """Means over `step_s` buckets: half an hour at 2 s is 900 points, which a phone polling every 5 s need not
    fetch. A bucket's time is its last sample's; a key missing from every sample in it stays None."""
    out, bucket, edge = [], [], None
    for p in points:
        if edge is not None and p["t"] >= edge:
            out.append(_mean(bucket))
            bucket = []
        if not bucket:
            edge = (p["t"] // step_s + 1) * step_s
        bucket.append(p)
    if bucket:
        out.append(_mean(bucket))
    return out


def _mean(bucket: list[dict]) -> dict:
    row = {"t": bucket[-1]["t"]}
    for key in bucket[0]:
        if key == "t":
            continue
        values = [p[key] for p in bucket if p.get(key) is not None]
        row[key] = round(sum(values) / len(values), 2) if values else None
    return row


def public_system(payload: dict, step_s: float = 10.0) -> dict:
    """viz/system.py's payload with nothing that identifies the machine or says where anything lives on it."""
    specs = {k: v for k, v in (payload.get("specs") or {}).items() if k not in ("host", "kernel")}
    now = payload.get("now")
    if now is not None:
        now = dict(now)
        now["procs"] = [{k: p.get(k) for k in PROC_FIELDS} for p in now["procs"]]
        now["mounts"] = [{k: v for k, v in m.items() if k != "device"} for m in now["mounts"]]
        now["gpus"] = [{k: v for k, v in g.items() if k != "at"} for g in now["gpus"]]
    return {"specs": specs, "now": now, "history": downsample(payload.get("history") or [], step_s),
            "history_s": payload.get("history_s")}


def live_runs(payload: dict, trainers: list[dict]) -> dict:
    """supervise.build_payload's runs with "running" meaning a trainer is writing the run now, and without
    run_paths -- absolute paths keyed by run, the one part of it that is purely local."""
    writing = {t["run"] for t in trainers}
    runs = [dict(r, status="stopped") if r["status"] == "running" and r["name"] not in writing else r
            for r in payload["runs"]]
    return dict({k: v for k, v in payload.items() if k != "run_paths"}, runs=runs, live=True)


def public_runs(payload: dict, trainers: list[dict]) -> dict:
    """Every run, scrubbed as the site's training page is (dashboard.public_payload): runs.json."""
    out = public_payload(live_runs(payload, trainers))
    out["intro"], out["links"] = LIVE_INTRO, [list(link) for link in SITE_LINKS]
    return out


def write_live_page(repo: Path, out_dir: Path, description: str, machines: list[str] = ()) -> Path:
    """The training page, live: the runs as they stand now baked in (what a visitor sees if the PC is off), and
    the machines and Training now filled from live/*.json. out_dir is the site's zombies/training/. `machines`
    are the worker PCs' ids, whose live/machine-<id>.json the page also reads (a static site has no listing)."""
    payload = build_payload(run_roots(repo))
    here = Path(__file__).parent
    ids = [machine_id(m) for m in machines]
    return write_dashboard_site(
        live_runs(payload, live_trainers(payload["run_paths"])), out_dir, LIVE_INTRO, SITE_LINKS, description,
        body_before=(here / "live_panel.html").read_text(),
        script_after=f"<script>window.LIVE_MACHINES = {json.dumps(ids)};</script>\n"
                     + (here / "system_view.html").read_text() + (here / "live_public.html").read_text(),
    )


def training_now(trainers: list[dict], runs: dict) -> list[dict]:
    """The runs being trained this minute: how far along, how long it has run, and the number it is judged by."""
    by_name = {r["name"]: r for r in runs["runs"]}
    out = []
    for t in trainers:
        run = by_name.get(t["run"]) if t["run"] else None
        head = run["series"].get(run["headline"] or "") if run else None
        out.append({
            "run": t["run"], "script": t["script"], "elapsed_s": t["elapsed_s"],
            "kind": run and run["kind"], "env": run and run["env"], "color": run and run["color"],
            "progress": run and run["progress"], "progress_text": run and run["progress_text"],
            "eta_s": run and run["eta_s"], "steps_per_s": run and run["stats"].get("sps"),
            "headline": head and {"label": head["label"], "last": head["last"], "best": head["best"],
                                  "trend": (head.get("trend") or {}).get("verdict")},
        })
    return out


def _write(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    os.replace(tmp, path)


class LivePublisher:
    """Writes machine.json, runs.json and stream.json to `out_dir` and, with a `dest`, rsyncs them there.
    `stream_run` pins the run the stream's numbers come from (by its public name); otherwise it follows training.

    With `worker` (a machine id), this is one of the PCs that play for the learner: it writes and pushes only its
    own machine-<id>.json, with what its worker is doing read from `worker_status` -- never runs.json or
    stream.json, which are the learner's.

    The learner with nothing training reports only every `idle_every_s` (0: every tick), the sampler paused between;
    `every_s` is how often tick() is called, which the pages are told while something trains."""

    def __init__(self, repo: Path, out_dir: Path, dest: str | None, *, sampler: SystemSampler,
                 runs_every_s: float = 60.0, stream_run: str | None = None, ssh: str = "ssh", run=subprocess.run,
                 say=print, label: str | None = None, worker: str | None = None, worker_status: Path | None = None,
                 every_s: float = 5.0, idle_every_s: float = 300.0):
        self.repo, self.out_dir, self.dest, self.sampler = repo, out_dir, dest, sampler
        self.every_s, self.idle_every_s = every_s, idle_every_s
        self.worker = machine_id(worker) if worker else None
        self.label = (label or (self.worker or LEARNER_LABEL))[:40]
        self.worker_status = worker_status
        self.stream = StreamFeed(stream_run)
        self.runs_every_s, self.run, self.say = runs_every_s, run, say
        self.out_dir.mkdir(parents=True, exist_ok=True)
        control = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "zombiesai-live-%C"
        # One SSH connection, kept open between pushes: a handshake every 5 s is most of the cost otherwise.
        self.ssh = (f"{ssh} -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=auto "
                    f"-o ControlPath={shlex.quote(str(control))} -o ControlPersist=300")
        self.runs: dict | None = None
        self.runs_at = 0.0
        self.run_paths: dict[str, str] = {}  # a run's directory -> its public name, as of the last runs.json
        self.training = False  # whether the last report had anything training
        self.reported_at: float | None = None
        self.failing: str | None = "not tried yet"  # None is pushing fine; a string is why it is not

    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if self.worker:
            machine = public_system(self.sampler.payload())
            machine.update(at=now, label=self.label, role="worker",
                           fleet=worker_status(self.worker_status, now) if self.worker_status else
                           {"state": "not running"})
            _write(self.out_dir / worker_file(self.worker), machine)
            if self.dest:
                self.push()
            return
        # Reading /proc is cheap; reading every run's metrics is not, so the names come from the last runs.json.
        trainers = live_trainers(self.run_paths)
        training = bool(trainers)
        if self.idle_every_s and not training and not self.training and self.reported_at is not None:
            due_in = self.reported_at + self.idle_every_s - now
            if due_in > 0:
                if due_in <= WARMUP_S:
                    self.sampler.resume()
                return
        # A trainer starting is named, and one ending shows its last curves, as soon as it happens.
        if self.runs is None or training != self.training or now - self.runs_at >= self.runs_every_s:
            payload = build_payload(run_roots(self.repo))
            self.run_paths = payload["run_paths"]
            trainers = live_trainers(self.run_paths)
            self.runs, self.runs_at = public_runs(payload, trainers), now
            self.stream.choose(self.run_paths, trainers)
            _write(self.out_dir / "runs.json", self.runs)
        every_s = self.every_s if training or not self.idle_every_s else self.idle_every_s
        machine = public_system(self.sampler.payload())
        machine["at"], machine["every_s"] = now, every_s
        machine["runs_at"] = self.runs_at
        machine["training"] = training_now(trainers, self.runs)
        machine["label"], machine["role"] = self.label, "learner"
        _write(self.out_dir / "machine.json", machine)
        _write(self.out_dir / "stream.json", {**self.stream.payload(self.runs, trainers, now), "every_s": every_s})
        self.training, self.reported_at = training, now
        if training or not self.idle_every_s:
            self.sampler.resume()
        else:
            self.sampler.pause()
        if self.dest:
            self.push()

    def push(self) -> bool:
        files = [f"--include={worker_file(self.worker)}"] if self.worker else ["--include=*.json"]
        argv = ["rsync", "-a", "--timeout=20", "-e", self.ssh, *files, "--exclude=*", f"{self.out_dir}/", self.dest]
        try:
            result = self.run(argv, capture_output=True, text=True, timeout=60)
            problem = None if result.returncode == 0 else (result.stderr.strip().splitlines() or ["rsync failed"])[-1]
        except (OSError, subprocess.TimeoutExpired) as error:
            problem = str(error)
        if problem != self.failing:  # say when it starts failing and when it recovers, not every 5 s
            self.say(f"{time.strftime('%H:%M:%S')}  " + (f"push failing: {problem}" if problem else
                                                         f"pushing to {self.dest}"))
            self.failing = problem
        return problem is None
