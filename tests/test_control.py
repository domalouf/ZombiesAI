import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pytest

from zombiesai.viz import control, supervise
from zombiesai.viz.control import (
    Control,
    DemoLibrary,
    Events,
    PresetStore,
    bc_flags,
    bc_settings,
    classify_end,
    downsample,
    find_recorders,
    next_bc_name,
    summary,
    week,
    write_launch,
)
from zombiesai.viz.supervise import Handler, Supervisor, build_payload

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ fakes


class FakeProc:
    _next = 900_000

    def __init__(self, argv, **kwargs):
        FakeProc._next += 1
        self.pid, self.argv, self.kwargs, self.returncode = FakeProc._next, argv, kwargs, None

    def poll(self):
        return self.returncode


class Popen:
    def __init__(self):
        self.calls: list[FakeProc] = []

    def __call__(self, argv, **kwargs):
        proc = FakeProc(argv, **kwargs)
        self.calls.append(proc)
        return proc


class FakeSupervisor:
    def __init__(self, repo: Path, state_dir: Path):
        self.repo, self.state_dir, self.started, self._payload = repo, state_dir, [], None
        state_dir.mkdir(parents=True, exist_ok=True)


def fake_process(proc: Path, pid: int, argv: list[str], cwd: Path, log: Path | None = None) -> None:
    entry = proc / str(pid)
    (entry / "fd").mkdir(parents=True)
    (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    os.symlink(cwd, entry / "cwd")
    if log is not None:
        os.symlink(log, entry / "fd" / "1")


def make_control(tmp_path, **kwargs) -> tuple[Control, Popen, list, list]:
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True, exist_ok=True)
    popen, killed, notes = Popen(), [], []
    ctl = Control(FakeSupervisor(repo, tmp_path / "state"), presets=tmp_path / "presets.json", popen=popen,
                  kill=lambda pid, sig: killed.append((pid, sig)), notify=lambda *a: notes.append(a),
                  proc=kwargs.pop("proc", tmp_path / "proc"), subscribe=False, **kwargs)
    (tmp_path / "proc").mkdir(exist_ok=True)
    return ctl, popen, killed, notes


def write_demo(root: Path, name: str, created: float, seconds: float = 600, idle: float = 60, **extra) -> Path:
    path = root / name
    path.mkdir(parents=True)
    manifest = {"created_unix": created, "status": "closed", "n_steps": int(seconds * 15),
                "summary": {"seconds": seconds, "not_playing_seconds": idle, "notes": f"{name} notes",
                            "stale_rate": 0.01, "overrun_rate": 0.0, "capture_lost": None},
                "hud": {"shapes": {"scene": [360, 640, 3]}}, "audio": {"rate": 48000}, **extra}
    (path / "clip.json").write_text(json.dumps(manifest))
    (path / "frames.u8").write_bytes(b"\0" * 1000)
    return path


# ------------------------------------------------------------------------------------------------ summary


def test_downsampling_keeps_the_last_point_and_at_most_n():
    x, y = list(range(1000)), [v * 2 for v in range(1000)]
    sx, sy = downsample(x, y, 300)
    assert len(sx) <= 300 and sx[0] == 0 and sx[-1] == 999 and sy[-1] == 1998
    assert downsample([1, 2], [3, 4], 300) == ([1, 2], [3, 4])


def write_ppo_run(root: Path, name: str, rows: int = 900) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"env": "real-waw", "total_steps": rows * 100}))
    (run / "metrics.jsonl").write_text("".join(
        json.dumps({"update": i, "step": 100 * i, "return_mean": i / 10, "round_reached_mean": 1 + i / 1000,
                    "length_mean": 50.0, "entropy": 2.0, "sps": 40}) + "\n" for i in range(rows)))
    return run


def test_the_summary_is_light_and_carries_the_headline_series_preset_and_log(tmp_path):
    runs = tmp_path / "repo" / "runs"
    write_ppo_run(runs, "rl1")
    (runs / "rl1.log").write_text("hello\n")
    write_launch(runs, "rl1", "overnight", {"init": "runs/bc/bc.pt", "actors": 6, "preset": "overnight"})
    payload = build_payload([("main", runs)])
    out = summary(payload, [("main", runs)], trainers=[{"pid": 1}])
    assert out["trainers"] == [{"pid": 1}]
    run = out["runs"][0]
    assert run["name"] == "rl1" and run["tree"] == "main" and run["kind"] == "ppo"
    assert run["path"] == str((runs / "rl1").resolve())
    assert run["headline"]["key"] == "return_mean" and run["headline"]["last"] == pytest.approx(89.9, abs=0.5)
    assert list(run["series"])[:2] == ["return_mean", "round_reached_mean"] and len(run["series"]) <= 4
    assert all(len(s["x"]) <= 300 and len(s["x"]) == len(s["y"]) for s in run["series"].values())
    assert run["preset"] == "overnight"
    assert run["log"] == str((runs / "rl1.log").resolve()) or run["log"].endswith("runs/rl1.log")
    launch = json.loads((runs / "rl1.launch.json").read_text())
    assert "preset" not in launch["request"] and launch["request"]["actors"] == 6
    assert len(json.dumps(out)) < 150_000


# ------------------------------------------------------------------------------------------------ demos


def ts(y, m, d, h=12) -> float:
    return datetime(y, m, d, h).timestamp()


def test_the_demo_library_reads_clip_json_and_counts_the_week(tmp_path):
    root = tmp_path / "demos"
    write_demo(root, "demo_0000", ts(2026, 11, 20), seconds=1200, idle=0)
    write_demo(root, "demo_0001", ts(2026, 11, 27, 9), seconds=600, idle=60)
    write_demo(root, "demo_0002", ts(2026, 11, 27, 20), seconds=300, idle=0,
               gamepad={"device": "Xbox Wireless Controller"})
    (root / "demo_0002" / "inputs.mp4").write_bytes(b"x")
    library = DemoLibrary(root)
    out = library.payload({"demo_0001"}, today=date(2026, 11, 27))
    names = [d["name"] for d in out["demos"]]
    assert names == ["demo_0002", "demo_0001", "demo_0000"]  # newest first
    newest, middle = out["demos"][0], out["demos"][1]
    assert newest["gamepad"] == "Xbox Wireless Controller" and newest["video"].endswith("demo_0002/inputs.mp4")
    assert middle["minutes"] == 10 and middle["play_minutes"] == 9 and middle["not_playing_minutes"] == 1
    assert middle["rendering"] and not newest["rendering"]
    assert middle["scene"] and middle["audio"] and middle["status"] == "closed" and middle["size_mb"] >= 0
    assert out["today_min"] == 14  # 9 + 5 minutes of play, both today
    assert [d["date"] for d in out["week"]] == [f"2026-11-{d}" for d in range(21, 28)]
    assert out["week"][-1]["minutes"] == 14 and sum(d["minutes"] for d in out["week"]) == 14  # the 20th is out
    assert out["goal_min"] == 20 and out["total_play_min"] == 34


def test_a_demo_is_reread_only_when_it_changes(tmp_path):
    root = tmp_path / "demos"
    path = write_demo(root, "demo_0000", ts(2026, 11, 27))
    library = DemoLibrary(root)
    assert library.demos()[0]["video"] is None
    (path / "inputs.mp4").write_bytes(b"x")  # a render finishing changes the card without touching clip.json
    assert library.demos()[0]["video"] is not None
    manifest = json.loads((path / "clip.json").read_text())
    manifest["summary"]["notes"] = "edited"
    (path / "clip.json").write_text(json.dumps(manifest))
    os.utime(path / "clip.json", (time.time() + 5, time.time() + 5))
    assert library.demos()[0]["notes"] == "edited"


def test_week_buckets_by_local_day_of_the_start():
    days = week([{"created": ts(2026, 11, 27, 23), "play_minutes": 20}, {"created": ts(2026, 11, 26, 1),
                                                                         "play_minutes": 5}], date(2026, 11, 27), 2)
    assert days == [{"date": "2026-11-26", "minutes": 5}, {"date": "2026-11-27", "minutes": 20}]


# ------------------------------------------------------------------------------------------------ presets


def test_presets_save_overwrite_and_delete_atomically(tmp_path):
    store = PresetStore(tmp_path / "cfg" / "presets.json")
    assert store.list() == []
    assert store.save("overnight", "rl", {"actors": 6, "preset": "x"})["ok"]
    assert store.save("quick", "bc", {"settings": {"epochs": 4}})["ok"]
    assert store.save("overnight", "rl", {"actors": 7})["ok"]  # the same name replaces it
    presets = store.list()
    assert [p["name"] for p in presets] == ["overnight", "quick"]
    assert presets[0]["request"] == {"actors": 7} and presets[0]["kind"] == "rl"
    assert not list((tmp_path / "cfg").glob("*.tmp")) and not list((tmp_path / "cfg").glob(".*tmp"))
    assert "error" in store.save("", "rl", {}) and "error" in store.save("x", "dqn", {})
    assert store.delete("quick")["ok"] and [p["name"] for p in store.list()] == ["overnight"]
    assert "error" in store.delete("nope")


# ------------------------------------------------------------------------------------------------ BC


def test_bc_settings_come_from_train_bcs_own_parser():
    settings = {s["name"]: s for s in bc_settings(REPO)}
    assert {"clips", "out", "val_clips", "audio_cache"}.isdisjoint(settings)
    assert settings["epochs"] == {**settings["epochs"], "flag": "--epochs", "kind": "int", "default": 10,
                                  "group": "length"}
    assert settings["lr"]["kind"] == "float" and settings["lr"]["default"] == pytest.approx(3e-4)
    assert settings["prev_actions"]["kind"] == "bool" and settings["prev_actions"]["default"] is False
    assert settings["frame_offsets"]["kind"] == "list"
    assert settings["focal_gamma"]["hint"].startswith("focal loss exponent")
    assert all(set(s) == {"name", "flag", "kind", "default", "hint", "group"} for s in settings.values())


def test_bc_flags_only_name_what_differs_and_check_kinds():
    settings = bc_settings(REPO)
    flags, problem = bc_flags(settings, {"epochs": 10, "lr": "0.001", "prev_actions": True, "hud_view": False,
                                         "frame_offsets": "0 1 2 4"})
    assert problem is None
    assert flags == ["--lr", "0.001", "--prev-actions", "--frame-offsets", "0", "1", "2", "4"]
    assert bc_flags(settings, {"epochs": 2.5})[1] == "epochs: 2.5 is not an int"
    assert "not a setting" in bc_flags(settings, {"nope": 1})[1]


def test_the_next_bc_name_counts_past_the_highest(tmp_path):
    for name in ("bc_real1", "bc_real4", "bc_other", "rl9"):
        (tmp_path / name).mkdir()
    assert next_bc_name(tmp_path) == "bc_real5"
    assert next_bc_name(tmp_path / "missing") == "bc_real1"


def test_a_bc_run_starts_with_its_demos_flags_and_a_launch_note(tmp_path):
    ctl, popen, _, _ = make_control(tmp_path)
    repo = ctl.repo
    write_demo(repo / "data" / "demos", "demo_0000", ts(2026, 11, 27))
    write_demo(repo / "data" / "demos", "demo_0001", ts(2026, 11, 27))
    (repo / "scripts").mkdir()
    (repo / "scripts" / "train_bc.py").write_text((REPO / "scripts" / "train_bc.py").read_text())
    (repo / "src" / "zombiesai" / "demos").mkdir(parents=True)
    (repo / "src" / "zombiesai" / "demos" / "bc.py").write_text((REPO / "src/zombiesai/demos/bc.py").read_text())
    body = {"name": "bc_real5", "demos": ["demo_0001"], "settings": {"epochs": 4}, "preset": "quick"}
    result = ctl.start_bc(body, trainers=[])
    assert result["ok"], result
    argv = popen.calls[0].argv
    assert argv[2:] == ["scripts/train_bc.py", "data/demos/demo_0001", "--out", "runs/bc_real5", "--epochs", "4"]
    assert popen.calls[0].kwargs["start_new_session"] and popen.calls[0].kwargs["cwd"] == repo
    assert json.loads((repo / "runs" / "bc_real5.launch.json").read_text())["preset"] == "quick"
    assert (repo / "runs" / "bc_real5.log").read_text().startswith("# ")
    (repo / "runs" / "bc_real5").mkdir()
    assert "already exists" in ctl.start_bc(body, trainers=[])["error"]
    assert "no demo" in ctl.start_bc({**body, "name": "bc_x", "demos": ["demo_9"]}, trainers=[])["error"]
    busy = [{"pid": 5, "script": "train_bc.py", "cwd": str(repo)}]
    assert "already training" in ctl.start_bc({**body, "name": "bc_y"}, trainers=busy)["error"]
    everything = ctl.start_bc({"name": "bc_z", "demos": []}, trainers=[])
    assert everything["ok"] and popen.calls[-1].argv[3] == "data/demos"


# ------------------------------------------------------------------------------------------------ recording


def test_the_recorder_is_the_python_process_not_its_uv_wrapper(tmp_path):
    proc = tmp_path / "proc"
    log = tmp_path / "rec.log"
    log.write_text("")
    fake_process(proc, 100, ["uv", "run", "python", "scripts/record_demo.py", "--wait"], tmp_path)
    fake_process(proc, 101, ["/repo/.venv/bin/python3", "scripts/record_demo.py", "--wait"], tmp_path, log)
    fake_process(proc, 102, ["/usr/bin/python3", "scripts/record_overlay.py"], tmp_path)
    found = find_recorders(proc)
    assert [r["pid"] for r in found] == [101] and found[0]["log"] == str(log)


def test_recording_starts_record_lan_with_its_switches_and_refuses_a_second(tmp_path):
    ctl, popen, killed, _ = make_control(tmp_path)
    assert "between 1 and 120" in ctl.record({"minutes": 0})["error"]
    assert "pad" in ctl.record({"minutes": 5, "pad": "/etc/passwd"})["error"]
    assert "one line" in ctl.record({"minutes": 5, "notes": "a\nb"})["error"]
    result = ctl.record({"minutes": 20, "notes": "trains", "pad": "auto", "overlay": False})
    assert result["ok"]
    call = popen.calls[-1]
    assert call.argv == ["scripts/record_lan.sh", "20", "trains"]
    env = call.kwargs["env"]
    assert env["ZOMBIESAI_OVERLAY"] == "0" and env["ZOMBIESAI_AUDIO"] == "1" and env["ZOMBIESAI_PAD"] == "auto"
    assert call.kwargs["start_new_session"] and Path(result["log"]).parent == ctl.repo / "runs" / "record"
    assert "already running" in ctl.record({"minutes": 5})["error"]  # its wrapper is still going
    assert "no recording" in ctl.stop_recording()["error"]  # nothing is recording yet (no record_demo.py)
    fake_process(tmp_path / "proc", 4242, ["/x/python", "scripts/record_demo.py"], ctl.repo)
    assert ctl.stop_recording() == {"ok": True, "pids": [4242]}
    assert killed == [(4242, signal.SIGINT)]


def test_the_record_payload_says_active_and_what_was_asked(tmp_path, monkeypatch):
    ctl, popen, _, _ = make_control(tmp_path)
    assert ctl.record_payload()["active"] is False
    ctl.record({"minutes": 10, "notes": "x"})
    monkeypatch.setattr(control, "pid_alive", lambda pid: True)
    fake_process(tmp_path / "proc", 4243, ["/x/python", "scripts/record_demo.py"], ctl.repo)
    out = ctl.record_payload()
    assert out["active"] and out["pid"] == 4243 and out["request"]["minutes"] == 10
    assert out["log"] and out["log_tail"][0].startswith("# ") and out["game"] == {"running": False, "op": None}
    assert set(out) == {"active", "pid", "started", "request", "status", "log", "log_tail", "last", "game"}


# ------------------------------------------------------------------------------------------------ events


@pytest.mark.parametrize("lines, verdict", [
    (["upd 3 | ...", "Traceback (most recent call last):", '  File "x"', "ValueError: bad"], "crashed"),
    (["SIGINT: stopping after this step", "interrupted: checkpoint written", "checkpoint: runs/rl/c.pt"], "stopped"),
    (["  action inertia: ok", "", "next: scripts/eval_bc.py to score it"], "finished"),
    (["writing runs/rl21", "checkpoint: runs/rl21/checkpoint.pt"], "finished"),
    (["upd 4 | step 100"], "stopped"),
    ([], "stopped"),
])
def test_how_a_trainer_ended_is_read_from_its_output(lines, verdict):
    assert classify_end(lines) == verdict


def test_events_are_numbered_kept_in_a_ring_and_notified():
    notes = []
    events = Events(notify=lambda *a: notes.append(a), clock=lambda: 7.0)
    for i in range(205):
        events.add("info", f"e{i}", notify=i == 0)
    out = events.since(200)
    assert out["seq"] == 205 and [e["title"] for e in out["events"]] == [f"e{i}" for i in range(200, 205)]
    assert len(events.since(0)["events"]) == 200 and notes == [("e0", "", "info")]


def test_the_watch_reports_trainers_that_ended_recordings_saved_and_jobs_done(tmp_path):
    ctl, popen, _, notes = make_control(tmp_path)
    proc = tmp_path / "proc"
    log = ctl.repo / "runs" / "rl21.log"
    log.write_text("upd 1\nTraceback (most recent call last):\nRuntimeError: CUDA out of memory\n")
    fake_process(proc, 777, ["python", "-u", "scripts/train_rl.py", "x", "--out", "runs/rl21"], ctl.repo, log)
    fake_process(proc, 778, ["/v/python", "scripts/record_demo.py"], ctl.repo)
    ctl.tick()  # the first look only learns what is running
    assert ctl.events.since(0)["events"] == []
    import shutil

    shutil.rmtree(proc / "777")
    shutil.rmtree(proc / "778")
    write_demo(ctl.repo / "data" / "demos", "demo_0006", time.time(), seconds=1200, idle=120)
    ctl.render("demo_0006")
    popen.calls[-1].returncode = 0
    ctl.tick()
    events = ctl.events.since(0)["events"]
    titles = [e["title"] for e in events]
    assert "rl21 crashed" in titles and "Recording saved" in titles and "demo_0006 rendered" in titles
    crashed = next(e for e in events if e["title"] == "rl21 crashed")
    assert crashed["level"] == "bad" and "CUDA out of memory" in crashed["text"]
    assert next(e for e in events if e["title"] == "Recording saved")["text"].startswith("demo_0006: 18.0 min")
    assert len(notes) == 3
    assert ctl.record_payload()["last"]["name"] == "demo_0006"


def test_a_recording_that_wrote_nothing_is_not_reported_as_the_last_demo(tmp_path):
    ctl, _, _, _ = make_control(tmp_path)
    write_demo(ctl.repo / "data" / "demos", "demo_0000", time.time() - 86400)
    ctl._seen_recorders, ctl._recorder_started = {5}, {5: time.time() - 60}
    ctl._seen_trainers = {}
    ctl.tick()
    assert ctl.events.since(0)["events"][-1]["title"] == "Recording ended"


# ------------------------------------------------------------------------------------------------ the endpoints


@pytest.fixture
def server(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    write_demo(repo / "data" / "demos", "demo_0000", time.time())
    monkeypatch.setattr(control, "default_presets_path", lambda: tmp_path / "presets.json")
    sup = Supervisor(repo, tmp_path / "state")
    sup.control.popen = Popen()
    sup.control._subscribe = False
    httpd = supervise.ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"supervisor": sup, "port": 0}))
    httpd.RequestHandlerClass.port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path):
        return json.load(urllib.request.urlopen(base + path))

    def post(path, body, header=True):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"X-Supervise": "1"} if header else {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, (json.loads(e.read() or b"{}") if e.code != 403 else {})

    yield get, post, sup
    httpd.shutdown()


def test_the_app_endpoints_answer_and_commands_need_the_header(server):
    get, post, sup = server
    assert set(get("/api/summary")) == {"runs", "trainers"}
    assert get("/api/demos")["demos"][0]["name"] == "demo_0000"
    assert set(get("/api/record")) >= {"active", "game", "last"}
    assert set(get("/api/bc")) == {"next", "settings", "demos"}
    assert get("/api/events?since=0") == {"events": [], "seq": 0}
    assert post("/api/presets", {"name": "p", "kind": "rl", "request": {"actors": 3}}, header=False)[0] == 403
    assert post("/api/presets", {"name": "p", "kind": "rl", "request": {"actors": 3}})[0] == 200
    assert get("/api/presets")["presets"][0]["request"] == {"actors": 3}
    assert post("/api/presets/delete", {"name": "p"})[0] == 200 and get("/api/presets") == {"presets": []}
    status, body = post("/api/record", {"minutes": 500})
    assert status == 409 and "between" in body["error"]
    assert post("/api/record/stop", {})[0] == 409
    assert post("/api/desktop", {"action": "sideways"})[0] == 409
    assert post("/api/demos/render", {"name": "../../etc"})[0] == 409
    status, body = post("/api/demos/render", {"name": "demo_0000"})
    assert status == 200 and get("/api/demos")["demos"][0]["rendering"]
    assert post("/api/bc", {"name": "bad name!"})[0] == 409
