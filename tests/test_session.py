import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from zombiesai import session
from zombiesai.demos import bc
from zombiesai.rl.parallel_ppo import StopRequest
from zombiesai.session import Proc, classify, default_init, processes, saved

REPO = Path(__file__).resolve().parents[1]
TREE = Path("/home/me/ZombiesAI")
PY = "/home/me/ZombiesAI/.venv/bin/python"


def procs(*ps: Proc) -> dict[int, Proc]:
    return {p.pid: p for p in ps}


def kind(p: Proc, *others: Proc) -> str | None:
    return classify(p, procs(p, *others), [TREE])


def test_what_is_ours_is_told_by_command_line_environment_and_parent():
    trainer = Proc(100, 1, [PY, "-u", "scripts/train_rl.py", "runs/rl5/checkpoint.pt", "--out", "runs/rl6"],
                   cwd=str(TREE))
    actor = Proc(101, 100, [PY, "-c", "from multiprocessing.spawn import spawn_main; spawn_main()",
                            "--multiprocessing-fork"], cwd=str(TREE))
    tracker = Proc(102, 100, [PY, "-c", "from multiprocessing.resource_tracker import main;main(5)"],
                   cwd=str(TREE))
    orphan = Proc(103, 1, actor.argv, cwd=str(TREE))
    init = Proc(1, 0, ["/sbin/init"])
    assert kind(trainer) == "trainer"
    assert kind(actor, trainer) == "actor"
    assert kind(tracker, trainer) is None  # goes with its trainer
    assert kind(orphan, init) == "actor"  # a killed trainer's, left playing
    assert kind(Proc(5, 1, [PY, "scripts/fleet_worker.py"])) == "worker"
    assert kind(Proc(6, 1, [PY, "-u", "scripts/supervise.py", "--no-open"])) == "dashboard"
    assert kind(Proc(7, 6, [PY, "-m", "zombiesai.viz.supervise", "--monitor", ":60"])) == "dashboard"
    assert kind(Proc(8, 1, [PY, "-m", "zombiesai.realgame.viewer", "--watch", ":60"])) == "viewer"
    assert kind(Proc(9, 1, ["Xwayland", ":60", "-geometry", "2560x1440", "-glamor", "gl", "-noreset"])) \
        == "game display"
    assert kind(Proc(10, 1, ["weston", "--backend=headless", "--socket=zombiesai-60"])) == "game display"
    assert kind(Proc(11, 1, ["C:\\Plutonium\\bin\\plutonium-bootstrapper-win32.exe"],
                     env={"PULSE_SINK": "zombiesai_2"})) == "game"
    assert kind(Proc(12, 1, ["wineserver"], env={"WINEPREFIX": f"{TREE}/runs/instances/i0/prefix/pfx"})) == "game"


def test_nothing_else_is_touched():
    bc_trainer = Proc(200, 1, [PY, "scripts/train_bc.py", "data/clips"], cwd=str(TREE))
    loader = Proc(201, 200, [PY, "-c", "from multiprocessing.spawn import spawn_main; spawn_main()"], cwd=str(TREE))
    pytest_run = Proc(202, 1, [PY, "-m", "pytest"], cwd=str(TREE))
    test_child = Proc(203, 202, loader.argv, cwd=str(TREE))
    assert kind(bc_trainer) is None
    assert kind(loader, bc_trainer) is None
    assert kind(test_child, pytest_run) is None
    assert kind(Proc(204, 1, [PY, "-u", "scripts/publish_live.py"])) is None  # the site's reporter stays
    assert kind(Proc(205, 1, ["Xwayland", ":0", "-rootless", "-core", "-listenfd", "55"])) is None  # the desktop's
    assert kind(Proc(206, 1, ["wine", "game.exe"], env={"WINEPREFIX": "/home/me/Games/pfx"})) is None
    assert kind(Proc(207, 1, ["python", "-c", "import multiprocessing"], cwd="/elsewhere")) is None


def test_processes_reads_a_proc_tree(tmp_path):
    entry = tmp_path / "4242"
    entry.mkdir()
    (entry / "cmdline").write_bytes(b"python\0scripts/train_rl.py\0--out\0runs/rl9\0")
    (entry / "stat").write_text("4242 (python (x)) S 77 4242 4242 0")
    (entry / "environ").write_bytes(b"PULSE_SINK=zombiesai_0\0HOME=/home/me\0")
    os.symlink(tmp_path, entry / "cwd")
    (tmp_path / "self").mkdir()
    found = processes(tmp_path)
    p = found[4242]
    assert p.ppid == 77 and p.script == "train_rl.py" and p.flag("--out") == "runs/rl9"
    assert p.env["PULSE_SINK"] == "zombiesai_0" and p.cwd == str(tmp_path)
    assert session.run_dir_of(p) == (tmp_path / "runs" / "rl9").resolve()


def write_ckpt(tree: Path, run: str, name: str, config: dict | None, mtime: float) -> None:
    d = tree / "runs" / run
    d.mkdir(parents=True)
    (d / name).write_bytes(b"")
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    os.utime(d / name, (mtime, mtime))


def test_start_continues_the_newest_real_game_run(tmp_path):
    assert default_init(tmp_path) == "fresh"
    write_ckpt(tmp_path, "bc1", "bc.pt", None, 1000)
    assert default_init(tmp_path) == "runs/bc1/bc.pt"
    write_ckpt(tmp_path, "rl4", "checkpoint.pt", {"algorithm": "ppo-finetune", "env": "real-waw"}, 2000)
    write_ckpt(tmp_path, "rl5", "checkpoint.pt", {"algorithm": "ppo-finetune", "env": "real-waw"}, 3000)
    write_ckpt(tmp_path, "rl-synth", "checkpoint.pt", {"algorithm": "ppo-finetune", "env": "synthetic"}, 4000)
    write_ckpt(tmp_path, "ppo-nacht-state", "checkpoint.pt", {"env": "nacht-state"}, 5000)
    assert default_init(tmp_path) == "runs/rl5/checkpoint.pt"


def test_saved_says_what_a_run_has_on_disk(tmp_path):
    run = tmp_path / "rl6"
    assert saved(run) == "rl6: no checkpoint.pt"
    (run / "best").mkdir(parents=True)
    (run / "checkpoint.pt").write_bytes(b"")
    (run / "metrics.jsonl").write_text(json.dumps({"update": 3, "step": 1536}) + "\n" +
                                       json.dumps({"update": 4, "step": 2048}) + "\n")
    (run / "episodes.jsonl").write_text("{}\n{}\n{}\n")
    (run / "best" / "best.json").write_text(json.dumps({"round": 2, "points": 1090}))
    line = saved(run, now=(run / "checkpoint.pt").stat().st_mtime + 4)
    assert line == ("rl6: checkpoint.pt written 4 s ago, 4 updates, 2,048 steps, 3 games, "
                    "best game round 2, 1090 points (best/best.mp4)")


def test_a_stop_request_finishes_the_step_and_never_cuts_the_save_short():
    said = []
    before = signal.getsignal(signal.SIGTERM)
    stop = StopRequest(said.append).install()
    try:
        os.kill(os.getpid(), signal.SIGTERM)  # `kill`: no longer the end of the learner, a request
        time.sleep(0.05)
        assert stop.requested and "SIGTERM: stopping after this step" in said[-1]
        with pytest.raises(KeyboardInterrupt):  # asked twice: the step is interrupted, as Ctrl-C always did
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)
        stop.saving()
        os.kill(os.getpid(), signal.SIGHUP)  # while the checkpoint is written: noted, nothing else
        time.sleep(0.05)
        assert "still stopping" in said[-1]
    finally:
        stop.uninstall()
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.fixture
def bc_checkpoint(tmp_path):
    config = bc.BCConfig(frame_offsets=(0, 1, 3), hidden=32)
    torch.manual_seed(0)
    path = tmp_path / "bc.pt"
    bc.save(path, bc.build_net(config), config, 0, {}, {"fire_duty": 0.1})
    return path


def test_stop_saves_a_running_trainer_and_leaves_nothing_behind(bc_checkpoint, tmp_path, monkeypatch):
    """A real trainer (a synthetic rehearsal, two actors) stopped by `./zai stop`: it writes its checkpoint after
    its last update, its actors go with it, and stop says what was saved."""
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    argv = [sys.executable, "-u", str(REPO / "scripts" / "train_rl.py"), str(bc_checkpoint), "--env", "synthetic",
            "--actors", "2", "--total-steps", "100000000", "--segment-steps", "48", "--batch-steps", "384",
            "--critic-warmup-updates", "1", "--minibatch-size", "96", "--device", "cpu", "--out", "runs/rlt"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))}
    log = open(repo / "runs" / "rlt.log", "wb")
    trainer = subprocess.Popen(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        metrics = repo / "runs" / "rlt" / "metrics.jsonl"
        deadline = time.monotonic() + 180
        while not (metrics.exists() and metrics.read_text().count("\n") >= 2):
            assert trainer.poll() is None, (repo / "runs" / "rlt.log").read_text()
            assert time.monotonic() < deadline, "the rehearsal never got to two updates"
            time.sleep(0.5)
        actors = session.descendants(trainer.pid)
        assert actors

        # Only this test's processes are visible to stop: never a real run or game on the PC running the tests.
        mine = {trainer.pid, *actors}
        real = session.processes
        monkeypatch.setattr(session, "processes", lambda proc=Path("/proc"): {
            pid: p for pid, p in real(proc).items() if pid in mine})
        monkeypatch.setattr("zombiesai.realgame.instances.take_down", lambda root, say=print: 0)
        monkeypatch.setattr(session, "remove_sinks", lambda: 0)
        said = []
        # The trainer is our child here, so it lingers as a zombie until reaped; stop() counts that as gone.
        assert session.stop(repo, timeout_s=170, say=said.append) == 0
        assert trainer.wait(timeout=10) == 0
    finally:
        if trainer.poll() is None:
            os.killpg(trainer.pid, signal.SIGKILL)
        log.close()

    text = (repo / "runs" / "rlt.log").read_text()
    assert "SIGINT: stopping after this step" in text and "interrupted: checkpoint written" in text
    run = repo / "runs" / "rlt"
    assert (run / "checkpoint.pt").stat().st_mtime >= metrics.stat().st_mtime
    meta = torch.load(run / "checkpoint.pt", weights_only=False)
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert meta["rl"]["updates"] == rows[-1]["update"]  # every update it logged is in the checkpoint
    assert not [pid for pid in actors if session.alive(pid)]
    assert any(line.strip().startswith("rlt: checkpoint.pt written") for line in said)
    assert said[-1] == "all stopped"
