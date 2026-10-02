"""The Training Room on domalouf.com, live: this PC pushes what the page shows, and nothing reaches in.

Three files, written to a staging directory and rsynced (over one reused SSH connection) to the site's
zombies/training/live/:

    machine.json  every 5 s   the machine as viz/system.py samples it, and the runs training right now
    runs.json     every 60 s  every run's curves: the payload the static training page is built from
    stream.json   every 5 s   the stream overlay's numbers and its page's details (viz/stream.py)

The page itself (scripts/dashboard.py --site ... --live) is the static training page with the machine and
"Training now" added, polling the first two; the stream's page and overlay (zombies/live/) poll the third. All
are public, so all go through the same scrub as the site build: no paths, no command lines, no host name, no
process ids -- a process is its name and its load, and a run is its name.
"""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from zombiesai.viz.dashboard import public_payload, write_dashboard_site
from zombiesai.viz.stream import StreamFeed
from zombiesai.viz.supervise import build_payload, live_trainers, run_roots
from zombiesai.viz.system import SystemSampler

SITE_LINKS = [("← domalouf.com", "/"), ("The agent playing", "/zombies/"), ("Live stream", "/zombies/live/"),
              ("Code on GitHub", "https://github.com/domalouf/ZombiesAI")]
LIVE_INTRO = (
    "How the reinforcement-learning agent's training is going, live from the gaming PC it trains on: the machine "
    "as it works, the runs training right now, and every run its trainers have written -- PPO on NachtSim and the "
    "real game, behavioural cloning from human play, and the inverse dynamics model."
)
PROC_FIELDS = ("name", "cpu", "rss", "mem_pct", "threads")


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


def write_live_page(repo: Path, out_dir: Path, description: str) -> Path:
    """The training page, live: the runs as they stand now baked in (what a visitor sees if the PC is off), and
    the machine and Training now filled from live/*.json. out_dir is the site's zombies/training/."""
    payload = build_payload(run_roots(repo))
    here = Path(__file__).parent
    return write_dashboard_site(
        live_runs(payload, live_trainers(payload["run_paths"])), out_dir, LIVE_INTRO, SITE_LINKS, description,
        body_before=(here / "live_panel.html").read_text(),
        script_after=(here / "system_view.html").read_text() + (here / "live_public.html").read_text(),
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
    `stream_run` pins the run the stream's numbers come from (by its public name); otherwise it follows training."""

    def __init__(self, repo: Path, out_dir: Path, dest: str | None, *, sampler: SystemSampler,
                 runs_every_s: float = 60.0, stream_run: str | None = None, ssh: str = "ssh", run=subprocess.run,
                 say=print):
        self.repo, self.out_dir, self.dest, self.sampler = repo, out_dir, dest, sampler
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
        self.failing: str | None = "not tried yet"  # None is pushing fine; a string is why it is not

    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if self.runs is None or now - self.runs_at >= self.runs_every_s:
            payload = build_payload(run_roots(self.repo))
            self.run_paths = payload["run_paths"]
            writing = live_trainers(self.run_paths)
            self.runs, self.runs_at = public_runs(payload, writing), now
            self.stream.choose(self.run_paths, writing)
            _write(self.out_dir / "runs.json", self.runs)
        # Reading /proc is cheap; reading every run's metrics is not, so the names come from the last runs.json.
        trainers = live_trainers(self.run_paths)
        machine = public_system(self.sampler.payload())
        machine["at"] = now
        machine["runs_at"] = self.runs_at
        machine["training"] = training_now(trainers, self.runs)
        _write(self.out_dir / "machine.json", machine)
        _write(self.out_dir / "stream.json", self.stream.payload(self.runs, trainers, now))
        if self.dest:
            self.push()

    def push(self) -> bool:
        argv = ["rsync", "-a", "--timeout=20", "-e", self.ssh, "--include=*.json", "--exclude=*",
                f"{self.out_dir}/", self.dest]
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
