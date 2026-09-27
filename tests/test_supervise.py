import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np

from zombiesai.viz import supervise
from zombiesai.viz.supervise import Handler, build_payload, live_trainers, png, run_roots, thumbnail


def write_run(root: Path, name: str, rows: int = 3) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"env": "real-waw", "total_steps": 1000}))
    (run / "checkpoint.pt").write_bytes(b"")
    (run / "metrics.jsonl").write_text("".join(
        json.dumps({"update": i, "step": 100 * i, "return_mean": -15.0 + i, "sps": 40}) + "\n" for i in range(rows)))
    return run


def test_every_worktree_with_runs_is_a_root_and_the_main_checkout_is_called_main(tmp_path):
    main, tree = tmp_path / "repo", tmp_path / "repo" / ".claude" / "worktrees" / "rl"
    (main / "runs").mkdir(parents=True)
    (tree / "runs").mkdir(parents=True)
    (tmp_path / "bare").mkdir()
    listing = f"worktree {main}\nHEAD abc\n\nworktree {tree}\nHEAD def\n\nworktree {tmp_path / 'bare'}\n"
    roots = run_roots(main, run=lambda *a, **k: subprocess.CompletedProcess([], 0, stdout=listing))
    assert roots == [("main", main / "runs"), ("rl", tree / "runs")]


def test_runs_from_two_trees_keep_their_own_names_and_one_colour_each(tmp_path):
    a, b = tmp_path / "a" / "runs", tmp_path / "b" / "runs"
    write_run(a, "rl1")
    write_run(b, "rl1")
    payload = build_payload([("main", a), ("rl", b)])
    names = sorted(r["name"] for r in payload["runs"])
    assert names == ["main/rl1", "rl/rl1"]
    assert len({r["color"] for r in payload["runs"]}) == 2 and sorted(r["slot"] for r in payload["runs"]) == [0, 1]
    assert payload["run_paths"][str((b / "rl1").resolve())] == "rl/rl1"
    alone = build_payload([("main", a)])
    assert [r["name"] for r in alone["runs"]] == ["rl1"]  # one tree: no prefix


def test_a_trainer_is_found_with_its_run_its_age_and_the_tail_of_its_output(tmp_path):
    proc, repo = tmp_path / "proc", tmp_path / "repo"
    run = write_run(repo / "runs", "rl9")
    log = repo / "runs" / "rl9.log"
    log.write_text("".join(f"line {i}\n" for i in range(100)))
    me = proc / "4242"
    (me / "fd").mkdir(parents=True)
    (me / "cmdline").write_bytes(b"\0".join([b"python", b"-u", b"scripts/train_rl.py", b"--out", b"runs/rl9", b""]))
    os.symlink(repo, me / "cwd")
    os.symlink(log, me / "fd" / "1")
    other = proc / "4243"
    other.mkdir()
    (other / "cmdline").write_bytes(b"python\0-c\0from multiprocessing.spawn import spawn_main\0")
    (proc / "self").mkdir()

    [t] = live_trainers({str(run.resolve()): "rl9"}, proc=proc)
    assert t["pid"] == 4242 and t["run"] == "rl9" and t["script"] == "train_rl.py" and t["args"] == "--out runs/rl9"
    assert t["tail"][-1] == "line 99" and len(t["tail"]) == 40 and t["tree"] == "repo"


def test_the_thumbnail_is_a_small_rgb_png_that_decodes_back_to_its_pixels():
    frame = np.zeros((1440, 2560, 4), np.uint8)
    frame[:, :, 2] = 200  # BGRX: red
    small = thumbnail(frame)
    assert small.shape == (180, 320, 3) and (small[..., 0] == 200).all() and (small[..., 2] == 0).all()
    data = png(small)
    assert data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR"
    idat = data.index(b"IDAT")
    size = int.from_bytes(data[idat - 4:idat], "big")
    rows = zlib.decompress(data[idat + 4:idat + 4 + size])
    assert len(rows) == 180 * (1 + 320 * 3) and rows[0] == 0 and rows[1:4] == bytes([200, 0, 0])


class FakeSupervisor:
    def __init__(self, state_dir):
        self.state_dir, self.viewer, self.shown = state_dir, {"workspace": "9", "fps": 30}, []

    def games(self):
        return {"screens": [], "viewer": self.viewer, "thumbs": True}

    def show(self, displays, workspace, fps, focus):
        self.shown.append((displays, workspace, fps, focus))
        return {"ok": True}


def test_commands_need_the_header_and_the_right_host_and_a_sane_workspace(tmp_path):
    fake = FakeSupervisor(tmp_path)
    server = supervise.ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"supervisor": fake, "port": 0}))
    port = server.server_address[1]
    Handler_ = server.RequestHandlerClass
    Handler_.port = port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(body, headers):
        req = urllib.request.Request(base + "/api/viewers", data=json.dumps(body).encode(), headers=headers)
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert post({"displays": [60]}, {}) == 403  # no header: what any other page could send
        assert post({"displays": [60], "workspace": "9; rm -rf"}, {"X-Supervise": "1"}) == 400
        assert post({"displays": [60, "x"], "workspace": "7", "fps": 999}, {"X-Supervise": "1"}) == 200
        assert fake.shown == [([60], "7", 120.0, False)]
        req = urllib.request.Request(base + "/api/games", headers={"Host": "evil.example"})
        try:
            urllib.request.urlopen(req)
            raise AssertionError("a request for another host was answered")
        except urllib.error.HTTPError as e:
            assert e.code == 403  # DNS rebinding: a page on another name must not reach this
        assert json.load(urllib.request.urlopen(base + "/api/games"))["viewer"]["workspace"] == "9"
        assert json.load(urllib.request.urlopen(base + "/api/ping")) == {"app": "zombiesai-supervise"}
    finally:
        server.shutdown()
