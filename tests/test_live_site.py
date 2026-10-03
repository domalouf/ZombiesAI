import json
import re
import subprocess
from collections import Counter

import pytest

from zombiesai.viz import live_site
from zombiesai.viz.live_site import LivePublisher, downsample, public_runs, public_system, training_now

SYSTEM = {
    "specs": {"host": "DomPC-mk3", "kernel": "7.2.5", "os": "Omarchy", "cpu": "Ryzen", "gpus": []},
    "now": {
        "t": 10.0, "cpu": {"pct": 50.0}, "gpus": [{"name": "RTX", "temp": 60, "at": 9.9}],
        "procs": [{"pid": 4242, "name": "python", "user": "dgm", "cpu": 150.0, "rss": 1, "mem_pct": 0.1,
                   "threads": 7, "state": "R", "command": "python scripts/train_rl.py --out /home/dgm/x"}],
        "mounts": [{"mount": "/", "device": "/dev/mapper/root", "pct": 7.0}],
    },
    "history": [{"t": float(t), "cpu": float(t), "gpu": None} for t in range(0, 30, 2)],
    "history_s": 1800.0,
}


def test_the_public_machine_says_nothing_about_where_things_live():
    pub = public_system(SYSTEM, step_s=10)
    text = json.dumps(pub)
    assert "DomPC" not in text and "/home" not in text and "/dev/" not in text and "dgm" not in text
    assert pub["now"]["procs"] == [{"name": "python", "cpu": 150.0, "rss": 1, "mem_pct": 0.1, "threads": 7}]
    assert pub["specs"]["cpu"] == "Ryzen" and pub["now"]["gpus"] == [{"name": "RTX", "temp": 60}]
    assert SYSTEM["now"]["procs"][0]["command"]  # the sampler's own payload is untouched


def test_history_is_bucket_means_that_keep_gaps_as_gaps():
    rows = downsample([{"t": float(t), "cpu": float(t), "gpu": None} for t in range(0, 30, 2)], step_s=10)
    assert [r["t"] for r in rows] == [8.0, 18.0, 28.0]
    assert [r["cpu"] for r in rows] == [4.0, 14.0, 24.0] and all(r["gpu"] is None for r in rows)


def run(name, status="running", kind="ppo"):
    return {"name": name, "status": status, "kind": kind, "env": "real-waw", "color": "#3987e5", "progress": 0.5,
            "progress_text": "update 1 of 2", "eta_s": 60.0, "stats": {"sps": 40}, "headline": "return_mean",
            "series": {"return_mean": {"label": "Return", "last": -3.0, "best": -1.0, "trend": {"verdict": "flat"}}},
            "config": {"clips": "/home/dgm/clips/a.npz", "lr": 0.001}, "last_row": {"step": 5}}


def test_runs_json_is_scrubbed_and_running_means_a_trainer_is_writing_it():
    payload = {"runs": [run("rl8"), run("rl7")], "run_paths": {"/home/dgm/runs/rl8": "rl8"}, "root": "main/runs"}
    trainers = [{"run": "rl8", "script": "train_rl.py", "elapsed_s": 100.0, "args": "--out /home/dgm/runs/rl8"}]
    out = public_runs(payload, trainers)
    assert "run_paths" not in out and "/home" not in json.dumps(out)
    assert [r["status"] for r in out["runs"]] == ["running", "stopped"] and out["live"] is True
    assert out["runs"][0]["config"] == {"lr": 0.001} and out["links"][0][1] == "/"
    [now] = training_now(trainers, out)
    assert now["run"] == "rl8" and now["headline"] == {"label": "Return", "last": -3.0, "best": -1.0, "trend": "flat"}
    assert "args" not in now and now["steps_per_s"] == 40


class FakeSampler:
    paused = False

    def payload(self, since=0.0):
        return SYSTEM

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


def test_a_tick_writes_both_files_and_pushes_them_saying_only_when_that_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(live_site, "build_payload", lambda roots: {"runs": [run("rl8")], "run_paths": {}})
    monkeypatch.setattr(live_site, "run_roots", lambda repo: [])
    monkeypatch.setattr(live_site, "live_trainers", lambda paths: [])
    calls, said, codes = [], [], iter([0, 23, 23, 0])

    def fake_rsync(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, next(codes), stderr="rsync: connection refused\n")

    pub = LivePublisher(tmp_path, tmp_path / "out", "lts:", sampler=FakeSampler(), run=fake_rsync, say=said.append,
                        idle_every_s=0)
    for t in (100.0, 105.0, 110.0, 170.0):
        pub.tick(now=t)
    machine = json.loads((tmp_path / "out" / "machine.json").read_text())
    assert machine["at"] == 170.0 and machine["runs_at"] == 170.0 and machine["training"] == []
    assert machine["label"] == "Training PC" and machine["role"] == "learner"
    assert json.loads((tmp_path / "out" / "runs.json").read_text())["runs"][0]["name"] == "rl8"
    stream = json.loads((tmp_path / "out" / "stream.json").read_text())
    assert stream["at"] == 170.0 and stream["run"] is None and stream["stats"]["games"] == 0  # no run has played
    assert calls[0][:2] == ["rsync", "-a"] and calls[0][-2:] == [f"{tmp_path / 'out'}/", "lts:"]
    assert "ControlMaster=auto" in calls[0][calls[0].index("-e") + 1]
    assert [s.split("  ", 1)[1] for s in said] == ["pushing to lts:", "push failing: rsync: connection refused",
                                                   "pushing to lts:"]


def test_with_nothing_training_it_reports_every_few_minutes_and_sleeps_between(tmp_path, monkeypatch):
    built, trainers = [], []
    monkeypatch.setattr(live_site, "build_payload",
                        lambda roots: built.append(1) or {"runs": [run("rl8")], "run_paths": {"/r/rl8": "rl8"}})
    monkeypatch.setattr(live_site, "run_roots", lambda repo: [])
    monkeypatch.setattr(live_site, "live_trainers", lambda paths: list(trainers))
    pushed = []
    sampler = FakeSampler()
    pub = LivePublisher(tmp_path, tmp_path / "out", "lts:", sampler=sampler, say=lambda m: None,
                        run=lambda argv, **kw: pushed.append(argv) or subprocess.CompletedProcess(argv, 0, stderr=""))
    machine = lambda: json.loads((tmp_path / "out" / "machine.json").read_text())

    pub.tick(now=100.0)  # the first report goes out at once, then the sampler sleeps
    assert len(pushed) == 1 and machine()["every_s"] == 300.0 and sampler.paused
    stream = json.loads((tmp_path / "out" / "stream.json").read_text())
    assert stream["every_s"] == 300.0 and stream["at"] == 100.0
    for t in (105.0, 200.0, 385.0):
        pub.tick(now=t)
    assert len(pushed) == 1 and sampler.paused
    pub.tick(now=395.0)  # due in 5 s: wake the sampler so the report has rates
    assert len(pushed) == 1 and not sampler.paused
    pub.tick(now=400.0)
    assert len(pushed) == 2 and machine()["at"] == 400.0 and sampler.paused

    trainers.append({"run": "rl8", "script": "train_rl.py", "elapsed_s": 1.0})
    runs_before = len(built)
    pub.tick(now=405.0)  # a trainer started: reported at once, runs.json rebuilt to name it, every 5 s from now
    assert len(pushed) == 3 and len(built) == runs_before + 1 and not sampler.paused
    assert machine()["every_s"] == 5.0 and machine()["training"][0]["run"] == "rl8"
    pub.tick(now=410.0)
    assert len(pushed) == 4 and len(built) == runs_before + 1

    trainers.clear()
    pub.tick(now=415.0)  # it ended: said at once, with its final curves, and back to sleep
    assert len(pushed) == 5 and len(built) == runs_before + 2 and sampler.paused
    assert machine()["training"] == [] and machine()["every_s"] == 300.0
    pub.tick(now=420.0)
    assert len(pushed) == 5


def test_a_worker_says_only_what_it_is_doing_and_goes_quiet_when_it_stops(tmp_path):
    status = tmp_path / "status.json"
    assert live_site.worker_status(status, now=100.0) == {"state": "not running"}  # never written
    status.write_text(json.dumps({"at": 95.0, "state": "playing", "run": "/home/dgm/runs/fleet/rl5", "actors": 4,
                                  "alive": 3, "sent": 12, "reason": "commit abc on DomPC-mk3"}))
    out = live_site.worker_status(status, now=100.0)
    assert out == {"state": "playing", "run": "rl5", "actors": 4, "alive": 3, "sent": 12}
    assert live_site.worker_status(status, now=95.0 + live_site.WORKER_STALE_S + 1) == {"state": "not running"}
    status.write_text(json.dumps({"at": 99.0, "state": "<script>"}))
    assert live_site.worker_status(status, now=100.0)["state"] == "not running"


def test_a_worker_pc_pushes_its_own_file_and_never_the_learners(tmp_path, monkeypatch):
    monkeypatch.setattr(live_site, "build_payload", lambda roots: pytest.fail("a worker has no runs to publish"))
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"at": 100.0, "state": "waiting", "actors": 4}))
    calls = []

    def fake_rsync(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stderr="")

    pub = LivePublisher(tmp_path, tmp_path / "out", "lts:", sampler=FakeSampler(), run=fake_rsync, say=lambda m: None,
                        worker="rig2", label="Gaming PC 2", worker_status=status)
    pub.tick(now=101.0)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["machine-rig2.json"]
    machine = json.loads((tmp_path / "out" / "machine-rig2.json").read_text())
    assert machine["label"] == "Gaming PC 2" and machine["role"] == "worker" and machine["at"] == 101.0
    assert machine["fleet"] == {"state": "waiting", "actors": 4} and "DomPC" not in json.dumps(machine)
    assert "--include=machine-rig2.json" in calls[0] and "--include=*.json" not in calls[0]
    with pytest.raises(ValueError):
        LivePublisher(tmp_path, tmp_path / "x", None, sampler=FakeSampler(), worker="../machine")


def test_the_live_page_knows_which_machines_to_read(tmp_path, monkeypatch):
    monkeypatch.setattr(live_site, "build_payload", lambda roots: {"runs": [run("rl8")], "run_paths": {}})
    monkeypatch.setattr(live_site, "run_roots", lambda repo: [])
    monkeypatch.setattr(live_site, "live_trainers", lambda paths: [])
    html = live_site.write_live_page(tmp_path, tmp_path / "site", "d", machines=["rig2", "rig3"]).read_text()
    assert 'window.LIVE_MACHINES = ["rig2", "rig3"];' in html and "machine-${id}.json" in html
    alone = live_site.write_live_page(tmp_path, tmp_path / "solo", "d").read_text()
    assert "window.LIVE_MACHINES = [];" in alone
    with pytest.raises(ValueError):
        live_site.write_live_page(tmp_path, tmp_path / "bad", "d", machines=['x"];alert(1)//'])


TOP_LEVEL = re.compile(r"^(?:async\s+function\*?|function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)", re.M)


def top_level_names(html: str) -> list[str]:
    """The names declared at column 0 of every inline script: on one page they all share a single global scope."""
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert scripts
    return [name for script in scripts for name in TOP_LEVEL.findall(script)]


def test_the_live_pages_scripts_never_declare_one_name_twice(tmp_path, monkeypatch):
    # The dashboard, the machine view and the live additions are three scripts in one global scope: a const or
    # let declared twice is a SyntaxError that kills the page, and a function declared twice silently replaces
    # the first (the live page's machine cards once replaced the dashboard's per-machine charts).
    monkeypatch.setattr(live_site, "build_payload", lambda roots: {"runs": [run("rl8")], "run_paths": {}})
    monkeypatch.setattr(live_site, "run_roots", lambda repo: [])
    monkeypatch.setattr(live_site, "live_trainers", lambda paths: [])
    html = live_site.write_live_page(tmp_path, tmp_path / "site", "d", machines=["rig2"]).read_text()
    names = top_level_names(html)
    assert {"machineCard", "liveMachineCard", "runTile", "tile", "refreshData", "renderSystem"} <= set(names)
    assert [name for name, count in Counter(names).items() if count > 1] == []
