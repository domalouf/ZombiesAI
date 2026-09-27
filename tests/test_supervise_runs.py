import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from zombiesai.viz import supervise
from zombiesai.viz.supervise import Supervisor, descendants, launch_options, live_trainers, start_run

# Stands in for scripts/train_rl.py: writes metrics while it runs and, on Ctrl-C (SIGINT), says so and writes its
# checkpoint -- what the real trainer does.
FAKE_TRAINER = """\
import json, pathlib, sys, time
out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True)
print(f"writing {out}", flush=True)
try:
    i = 0
    while True:
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"update": i, "step": 100 * i}) + "\\n")
        print(f"upd {i}", flush=True)
        i += 1
        time.sleep(0.1)
except KeyboardInterrupt:
    print("interrupted: stopping the actors", flush=True)
    (out / "checkpoint.pt").write_bytes(b"")
    print(f"checkpoint: {out / 'checkpoint.pt'}", flush=True)
"""


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "train_rl.py").write_text(FAKE_TRAINER)
    (tmp_path / "runs" / "rl1").mkdir(parents=True)
    (tmp_path / "runs" / "rl1" / "checkpoint.pt").write_bytes(b"")
    (tmp_path / "runs" / "bc_real2").mkdir()
    (tmp_path / "runs" / "bc_real2" / "bc.pt").write_bytes(b"")
    return tmp_path


def wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_the_form_offers_the_checkpoints_the_next_name_and_the_fleet(repo):
    (repo / "runs" / "instances").mkdir()
    (repo / "runs" / "instances" / "fleet.json").write_text(json.dumps({"n": 4, "display_base": 190}))
    [tree] = launch_options(repo, [])["trees"]
    assert tree["label"] == "main" and tree["next"] == "rl2"
    assert {c["path"] for c in tree["checkpoints"]} == {"runs/rl1/checkpoint.pt", "runs/bc_real2/bc.pt"}
    assert tree["fleet"] == {"n": 4, "display_base": 190, "running": 0} and tree["busy"] == []


def test_a_run_that_would_fail_or_collide_is_refused_before_anything_starts(repo):
    ok = dict(init="runs/rl1/checkpoint.pt", actors=4, name="rl2", extra="", trainers=[])

    def refused(**changes):
        return start_run(repo, "main", **{**ok, **changes}).get("error", "")

    assert "letters, digits" in refused(name="../evil")
    assert "already exists" in refused(name="rl1")
    assert "not a checkpoint" in refused(init="/etc/passwd")
    assert "between 1 and 16" in refused(actors=0)
    assert "own fields" in refused(extra="--out runs/elsewhere")
    assert "no fleet" in refused()
    (repo / "runs" / "instances").mkdir()
    (repo / "runs" / "instances" / "fleet.json").write_text(json.dumps({"n": 4, "display_base": 190}))
    assert "0 of the fleet's games are running" in refused()
    busy = [{"pid": 1, "run": "rl9", "cwd": str(repo), "sim": False}]
    assert "nowhere" in start_run(repo, "nowhere", **ok)["error"]  # not a checkout of this repo
    supervise_fleet = supervise.fleet_of
    try:
        supervise.fleet_of = lambda tree, screens=None: {"n": 4, "display_base": 190, "running": 4}
        assert "already playing this fleet" in refused(trainers=busy)
    finally:
        supervise.fleet_of = supervise_fleet
    assert not (repo / "runs" / "rl2").exists() and not (repo / "runs" / "rl2.log").exists()


def test_a_started_run_is_found_and_a_graceful_stop_leaves_its_checkpoint(repo, tmp_path_factory):
    supervisor = Supervisor(repo, tmp_path_factory.mktemp("state"))
    started = supervisor.start("main", "runs/rl1/checkpoint.pt", 2, "rl2", "--env sim")
    assert started.get("ok"), started
    pid, log = started["pid"], Path(started["log"])
    try:
        assert log == repo / "runs" / "rl2.log" and "started from the dashboard" in log.read_text()
        assert wait_for(lambda: "upd 3" in log.read_text())
        [trainer] = [t for t in supervisor.live()["trainers"] if t["pid"] == pid]
        assert trainer["run"] == "rl2" and trainer["sim"] and trainer["tree"] == repo.name
        assert trainer["stopping_s"] is None and not trainer["can_force"]
        assert "args" in trainer and "--actors 2 --out runs/rl2 --env sim" in trainer["args"]

        assert "ask it to stop first" in supervisor.stop(pid, force=True)["error"]
        assert supervisor.stop(pid) == {"ok": True}
        assert wait_for(lambda: not any(t["pid"] == pid for t in live_trainers({})))
        text = log.read_text()
        assert "interrupted: stopping the actors" in text and "checkpoint:" in text
        assert (repo / "runs" / "rl2" / "checkpoint.pt").exists()
        assert supervisor.stop(pid)["error"].startswith("no trainer")
    finally:
        subprocess.run(["kill", "-9", str(pid)], capture_output=True)


def test_the_run_gets_its_own_checkouts_code_and_outlives_the_dashboard(repo, tmp_path_factory):
    started = start_run(repo, "main", "runs/rl1/checkpoint.pt", 1, "rl3", "--env sim", [])
    proc = started["proc"]
    try:
        environ = Path(f"/proc/{proc.pid}/environ").read_bytes().split(b"\0")
        pythonpath = next(e for e in environ if e.startswith(b"PYTHONPATH=")).decode()
        assert pythonpath.split("=", 1)[1].split(":")[0] == str(repo / "src")
        assert Path(f"/proc/{proc.pid}/cwd").resolve() == repo.resolve()
        assert os.getsid(proc.pid) == proc.pid  # a session of its own: a dashboard restart does not reach it
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)


def test_every_process_below_a_trainer_is_found_for_a_forced_stop():
    shell = subprocess.Popen(["sh", "-c", "sleep 30 & sleep 30 & wait"])
    try:
        assert wait_for(lambda: len(descendants(shell.pid)) >= 2)
        assert all(Path(f"/proc/{p}/cmdline").read_bytes().startswith(b"sleep") for p in descendants(shell.pid))
    finally:
        for p in descendants(shell.pid):
            subprocess.run(["kill", str(p)], capture_output=True)
        shell.wait(timeout=5)
