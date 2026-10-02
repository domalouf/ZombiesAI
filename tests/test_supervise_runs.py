import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from zombiesai.viz import supervise
from zombiesai.viz.supervise import (Supervisor, descendants, launch_options, live_trainers, recent_speeds,
                                     setting_flags, start_run, trainer_settings)

# A trainer's RLConfig as the dashboard reads it: parsed, never imported.
FAKE_RL_CONFIG = '''\
from dataclasses import dataclass, field


@dataclass
class RLConfig:
    init: str = ""  # the BC checkpoint RL starts from
    env: str = "real"
    n_actors: int = 4
    total_steps: int = 2_000_000
    lr: float = 5e-5
    target_kl: float | None = 0.03  # stop an update's epochs early past 1.5x this
    seed: int = 0
    bindings: str = "configs/waw_bindings.json"
    hear: bool = True  # a checkpoint trained with audio hears its own instance's sink
    listen: str = ""  # host:port other PCs' workers send their games to
    sim: dict = field(default_factory=dict)
'''

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
    (tmp_path / "src" / "zombiesai" / "rl").mkdir(parents=True)
    (tmp_path / "src" / "zombiesai" / "rl" / "parallel_ppo.py").write_text(FAKE_RL_CONFIG)
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


def test_the_settings_are_the_trainers_own_fields_with_their_defaults_and_comments(repo):
    settings = {s["name"]: s for s in trainer_settings(repo)}
    # not init/env/sim, and not listen: a run started here has no fleet token, and train_rl.py would refuse it
    assert list(settings) == ["total_steps", "lr", "target_kl", "seed", "bindings", "hear"]
    assert "not a setting" in setting_flags(list(settings.values()), {"listen": ":47860"})[1]
    assert settings["total_steps"] == {"name": "total_steps", "flag": "--total-steps", "kind": "int",
                                       "default": 2_000_000, "hint": "", "group": "length"}
    assert settings["lr"]["kind"] == "float" and settings["lr"]["group"] == "learning"
    assert settings["target_kl"]["kind"] == "float" and settings["target_kl"]["hint"].startswith("stop an update")
    assert settings["hear"]["kind"] == "bool" and settings["bindings"]["kind"] == "str"


def test_only_changed_settings_become_flags_and_bad_values_are_refused(repo):
    settings = trainer_settings(repo)
    flags, problem = setting_flags(settings, {"total_steps": "500000", "lr": 5e-5, "hear": False, "seed": "3",
                                              "bindings": "configs/waw_bindings.json"})
    assert problem is None and flags == ["--total-steps", "500000", "--no-hear", "--seed", "3"]
    assert setting_flags(settings, {"total_steps": "2e6"}) == ([], None)  # the default, however written
    assert "not an int" in setting_flags(settings, {"seed": "1.5"})[1]
    assert "not a float" in setting_flags(settings, {"lr": "fast"})[1]
    assert "not a float" in setting_flags(settings, {"lr": "nan"})[1]
    assert "not a setting" in setting_flags(settings, {"fleet_root": "/tmp"})[1]
    assert "not a str" in setting_flags(settings, {"bindings": "a\nb"})[1]


def test_the_estimate_uses_the_newest_runs_speed_per_game(repo):
    for name, env, sps, games in (("rl1", "real-waw", 50, 4), ("sim1", "nacht-render", 400, 8)):
        (repo / "runs" / name).mkdir(exist_ok=True)
        (repo / "runs" / name / "config.json").write_text(json.dumps({"env": env, "n_actors": games}))
        (repo / "runs" / name / "metrics.jsonl").write_text(json.dumps({"update": 1, "sps": sps}) + "\n")
    speeds = recent_speeds(repo)
    assert speeds["real"] == {"run": "rl1", "sps": 50, "games": 4, "per_game": 12.5}
    assert speeds["sim"]["per_game"] == 50.0


def test_the_estimate_skips_a_run_trained_on_several_pcs(repo):
    # rl2 is newer, but its 200 steps/s came from this PC's 4 games and another PC's 12: not 50 a game here.
    for name, config, row in (("rl1", {"env": "real-waw", "n_actors": 4}, {"sps": 50}),
                              ("rl2", {"env": "real-waw", "n_actors": 4, "listen": "192.168.1.20:47860"},
                               {"sps": 200, "actors_alive": 16}),
                              ("rl3", {"env": "real-waw", "n_actors": 0, "listen": ":47860"},
                               {"sps": 150, "actors_alive": 12})):
        (repo / "runs" / name).mkdir(exist_ok=True)
        (repo / "runs" / name / "config.json").write_text(json.dumps(config))
        (repo / "runs" / name / "metrics.jsonl").write_text(json.dumps({"update": 1, **row}) + "\n")
    now = time.time()
    for age, name in enumerate(("rl3", "rl2", "rl1")):
        os.utime(repo / "runs" / name / "metrics.jsonl", (now - 10 * age, now - 10 * age))
    assert recent_speeds(repo)["real"] == {"run": "rl1", "sps": 50, "games": 4, "per_game": 12.5}


def test_a_run_starts_with_its_settings_as_flags_and_stops_itself_at_its_time_limit(repo, tmp_path_factory):
    state = tmp_path_factory.mktemp("state")
    supervisor = Supervisor(repo, state)
    started = supervisor.start("main", "runs/rl1/checkpoint.pt", 2, "rl4", "", env="sim", sim_hardness=0.25,
                               settings={"total_steps": 500000, "lr": 5e-5, "hear": False}, stop_after_min=0.05)
    assert started.get("ok"), started
    pid, log = started["pid"], Path(started["log"])
    try:
        assert started["command"].endswith("--out runs/rl4 --env sim --sim-hardness 0.25 --total-steps 500000 --no-hear")
        assert abs(started["stop_at"] - (time.time() + 3)) < 2
        [trainer] = [t for t in supervisor.live()["trainers"] if t["pid"] == pid]
        assert trainer["stop_at"] == started["stop_at"]
        # A dashboard that restarts keeps the limit: it is on disk.
        assert json.loads((state / "deadlines.json").read_text())[str(pid)]["run"] == "rl4"
        again = Supervisor(repo, state)
        assert wait_for(lambda: again.enforce_deadlines() or "time limit reached" in log.read_text(), 10)
        assert wait_for(lambda: not any(t["pid"] == pid for t in live_trainers({})))
        text = log.read_text()
        assert "interrupted: stopping the actors" in text and (repo / "runs" / "rl4" / "checkpoint.pt").exists()
        again.enforce_deadlines()
        assert json.loads((state / "deadlines.json").read_text()) == {}  # forgotten once the run has ended
    finally:
        subprocess.run(["kill", "-9", str(pid)], capture_output=True)


def test_a_time_limit_must_be_sane_and_settings_must_belong_to_the_trainer(repo, tmp_path_factory):
    supervisor = Supervisor(repo, tmp_path_factory.mktemp("state"))
    common = dict(tree="main", init="runs/rl1/checkpoint.pt", actors=1, name="rl5", extra="", env="sim")
    assert "time limit" in supervisor.start(**common, stop_after_min=-1)["error"]
    assert "time limit" in supervisor.start(**common, stop_after_min=60 * 24 * 8)["error"]
    assert "not a setting" in supervisor.start(**common, settings={"init": "x"})["error"]
    assert "real or sim" in supervisor.start(**{**common, "env": "moon"})["error"]
    assert not (repo / "runs" / "rl5").exists()
