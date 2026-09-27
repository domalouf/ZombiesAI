"""The Training Room, live, on localhost: every run's curves, what the running trainers are printing, and the game
instances -- with the switch for their viewer windows (realgame/viewer.py).

    uv run python scripts/supervise.py            # http://127.0.0.1:8765

Runs are read from `runs/` in the main checkout and in every worktree (a trainer writes where it was started),
with the same code as the static dashboard (viz/dashboard.py). A trainer is found by its process
(`scripts/train_*.py`); what it prints is read from wherever its stdout goes, when that is a file.

Each game gets a monitor process while the page is open: it counts whole frames off XDamage (nearly free) and
keeps a small thumbnail of the latest whole frame. They start with the first look at the page and stop
`idle_s` after the last, so a closed dashboard costs nothing. A monitor is a process of its own because Xlib
ends whatever process loses its X server, and an instance going down must not take the dashboard with it.

It listens on 127.0.0.1 only, answers only requests addressed to it by name (no DNS rebinding), and takes
commands only with an `X-Supervise` header, which no other page can send without a preflight it would refuse.
"""

import argparse
import json
import os
import re
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

HERE = Path(__file__).parent
GAME_TITLES = ("Plutonium", "Call of Duty")
_XWAYLAND = re.compile(r"^(\d+) Xwayland (:\d+)\b(.*)$")


# ------------------------------------------------------------------------------------------------ runs


def run_roots(repo: Path, *, run=subprocess.run) -> list[tuple[str, Path]]:
    """(label, runs dir) for the main checkout and every worktree that has a runs/ directory."""
    result = run(["git", "-C", str(repo), "worktree", "list", "--porcelain"], capture_output=True, text=True)
    trees = [Path(line[len("worktree "):]) for line in result.stdout.splitlines() if line.startswith("worktree ")]
    trees = trees or [repo]
    roots = []
    for i, tree in enumerate(trees):
        if (tree / "runs").is_dir():
            roots.append(("main" if i == 0 else tree.name, tree / "runs"))
    return roots


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
        found.append({
            "pid": int(entry.name),
            "script": Path(script).name,
            "args": " ".join(argv[argv.index(script) + 1:]),
            "tree": cwd.name,
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


class Supervisor:
    """What the page asks for: runs, trainers, games, and the viewer switch. Monitors run only while it asks."""

    def __init__(self, repo: Path, state_dir: Path, *, idle_s: float = 20.0, stale_after: float = 600.0):
        self.repo, self.state_dir, self.idle_s, self.stale_after = repo, state_dir, idle_s, stale_after
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "thumbs-off").unlink(missing_ok=True)
        self.monitors: dict[str, subprocess.Popen] = {}
        self.viewer = {"workspace": "9", "fps": 30}
        self.last_seen = 0.0
        self._lock = threading.Lock()
        self._payload, self._payload_at = None, 0.0
        threading.Thread(target=self._janitor, daemon=True).start()

    # -- runs
    def payload(self) -> dict:
        with self._lock:
            if self._payload is None or time.monotonic() - self._payload_at > 5.0:
                self._payload = build_payload(run_roots(self.repo), stale_after=self.stale_after)
                self._payload_at = time.monotonic()
            return self._payload

    def live(self) -> dict:
        return {"trainers": live_trainers(self.payload()["run_paths"])}

    def runs(self) -> dict:
        """The payload, with "running" meaning a trainer process is writing the run right now -- not only that
        its metrics are recent, which is all the static page can know."""
        payload = dict(self.payload())
        writing = {t["run"] for t in self.live()["trainers"]}
        payload["runs"] = [dict(r, status="stopped") if r["status"] == "running" and r["name"] not in writing else r
                           for r in payload["runs"]]
        return payload

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
        script_after=(HERE / "supervise_live.html").read_text(),
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
        path = self.path.split("?", 1)[0]
        s = self.supervisor
        if path == "/":
            return self._send(page(s).encode(), "text/html; charset=utf-8")
        if path == "/api/ping":  # how a second launch knows one is already serving (scripts/supervise.py)
            return self._json({"app": "zombiesai-supervise"})
        if path == "/api/runs":
            return self._json(s.runs())
        if path == "/api/live":
            return self._json(s.live())
        if path == "/api/games":
            return self._json(s.games())
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
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def serve(repo: Path, port: int, state_dir: Path, *, idle_s: float = 20.0, ready=None) -> None:
    supervisor = Supervisor(repo, state_dir, idle_s=idle_s)
    handler = type("BoundHandler", (Handler,), {"supervisor": supervisor, "port": port})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    if ready:
        ready(f"http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    finally:
        supervisor.stop_monitors()


def main() -> None:
    parser = argparse.ArgumentParser(description="one game's monitor (started by the supervise server)")
    parser.add_argument("--monitor", required=True, metavar="DISPLAY")
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    monitor(args.monitor, args.state)


if __name__ == "__main__":
    main()
