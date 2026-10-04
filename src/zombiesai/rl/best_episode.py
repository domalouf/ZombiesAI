"""The run's best game, kept as a video: `<run>/best/best.mp4`, with what it scored in `best.json`.

Which game is best is only known once it is over, so every actor films every game and, when one ends, keeps it
if it beats the run's best so far and throws it away otherwise. Best is the highest round; a tie goes to the
most points, then the most kills; a game that only ties the best does not replace it. The points and kills are
the game-over scoreboard's when it was read (`end_points` counts the 500 a game starts with, `end_kills` is the
game's own count), else the HUD's: the 500 plus every settled gain, and the gains taken for kills.

The film is the game's own picture, not the policy's 128x72: each grab the env makes on its decision deadline,
cut to every other pixel at 1440p (720p, `VIDEO_HEIGHT`) by the capture, at the decision rate. That cut costs
~1 ms of the actor's 66 ms tick (`video_frame`); the encoding is ffmpeg's, on NVENC where the machine has it,
and is fed from a thread that drops a frame rather than ever stall the game. A rehearsal's synthetic games have
no picture behind them, so they film their observations instead.

Actors on one machine share the run's directory, so `keep_if_best` decides under a file lock. A fleet's other
PCs keep the best of their own games, in their own copy of the run (`<out_root>/<run>/best/`, rl/fleet.py).
A game cut short by the actor stopping is not a finished game, and is thrown away.

Nothing here may cost the run a game: every failure -- no ffmpeg, an encoder that dies, a full disk -- is said
and costs that game's film, never the actor.
"""

import fcntl
import functools
import json
import os
import queue as queue_mod
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.hud.track import START_POINTS

VIDEO_HEIGHT = 720  # the film's height at most: a 1440p game is filmed at 1280x720
BEST_DIR = "best"
QUEUE_FRAMES = 45  # 3 s of frames waiting for the encoder before they are dropped


def episode_rank(summary: dict) -> tuple[int, int, int]:
    """(round, points, kills): the order games are ranked in, higher first."""
    points = summary.get("end_points")
    if points is None:
        points = START_POINTS + int(summary.get("points_gained") or 0)
    kills = summary.get("end_kills")
    if kills is None:
        kills = int(summary.get("kills") or 0)
    return int(summary.get("round_reached") or 0), int(points), int(kills)


def video_frame(frame: np.ndarray, height: int = VIDEO_HEIGHT) -> np.ndarray:
    """A BGRX grab cut to every n-th pixel, no taller than `height`, with even sides (yuv420p needs them).

    Each pixel is copied as one uint32 rather than four bytes: 0.9 ms at 1440p against 9 ms."""
    step = max(1, -(-frame.shape[0] // height))
    pixels = frame.view(np.uint32)[::step, ::step, 0]
    h, w = pixels.shape
    out = np.empty((h - h % 2, w - w % 2), np.uint32)
    np.copyto(out, pixels[: out.shape[0], : out.shape[1]])
    return out.view(np.uint8).reshape(*out.shape, 4)


@functools.lru_cache(maxsize=None)
def encoder_args(width: int, height: int) -> tuple[str, ...]:
    """NVENC where this machine can open it at this size (it has a minimum size, and needs an NVIDIA GPU),
    else x264 at a preset one core keeps up with."""
    probe = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", f"color=s={width}x{height}:d=0.2",
         "-c:v", "h264_nvenc", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    if probe.returncode == 0:
        return ("-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "27")
    return ("-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "2")


class VideoRecorder:
    """Frames in, an H.264 mp4 out, through an ffmpeg fed from its own thread. `add` never blocks: a frame the
    encoder is too far behind to take is dropped and counted."""

    def __init__(self, path: Path, shape: tuple[int, ...], fps: float = spec.DECISION_HZ):
        height, width, channels = shape
        self.path, self.shape = Path(path), tuple(shape)
        self.encoder = encoder_args(width, height)
        self.frames = self.dropped = 0
        self.failed: str | None = None
        self._log = self.path.with_suffix(".log")
        with open(self._log, "wb") as log:
            self._proc = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
                 "-pix_fmt", {3: "rgb24", 4: "bgr0"}[channels], "-s", f"{width}x{height}", "-r", f"{fps:g}",
                 "-i", "-", *self.encoder, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, bufsize=0)
        self._queue: queue_mod.Queue = queue_mod.Queue(maxsize=QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._feed, name=f"video {self.path.name}", daemon=True)
        self._thread.start()

    def add(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape:  # the window came back at another size: not a frame of this film
            self.dropped += 1
            return
        try:
            self._queue.put_nowait(frame)
        except queue_mod.Full:
            self.dropped += 1

    def _feed(self) -> None:
        while (frame := self._queue.get()) is not None:
            if self.failed:
                continue  # drain, so close() never waits on a full queue
            try:
                self._proc.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
                self.frames += 1
            except OSError as error:
                self.failed = f"the encoder stopped taking frames ({error})"

    def close(self) -> bool:
        """Finish the file. Whether it is a whole video of every frame the encoder was given."""
        self._queue.put(None)
        self._thread.join()
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            code = self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            code = "timeout"
        if code != 0 and not self.failed:
            self.failed = f"ffmpeg exited {code}"
        if self.failed:
            tail = self._log.read_text(errors="replace").strip()[-300:] if self._log.exists() else ""
            self.failed += f": {tail}" if tail else ""
        elif not self.frames:
            self.failed = "no frames"
        self._log.unlink(missing_ok=True)
        return self.failed is None

    def abort(self) -> None:
        self._proc.kill()
        self._queue.put(None)  # the feeder fails on the dead pipe and drains to this
        self._thread.join()
        self._proc.wait()
        self._log.unlink(missing_ok=True)
        self.path.unlink(missing_ok=True)


def read_best(best_dir: Path) -> dict | None:
    try:
        return json.loads((Path(best_dir) / "best.json").read_text())
    except (OSError, ValueError):
        return None


def keep_if_best(best_dir: Path, video: Path, summary: dict, **about) -> bool:
    """Make `video` the run's best game if it outranks the one kept so far, else delete it. Whether it was kept."""
    best_dir = Path(best_dir)
    rank = episode_rank(summary)
    with open(best_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read_best(best_dir)
        if current is not None and tuple(current.get("rank", ())) >= rank:
            video.unlink(missing_ok=True)
            return False
        os.replace(video, best_dir / "best.mp4")
        record = {"rank": list(rank), "round": rank[0], "points": rank[1], "kills": rank[2], "video": "best.mp4",
                  "host": socket.gethostname(), "recorded_unix": time.time(), **about, "summary": summary}
        tmp = best_dir / "best.json.tmp"
        tmp.write_text(json.dumps(record, indent=2, default=str))
        tmp.replace(best_dir / "best.json")
        return True


class BestEpisodeRecorder:
    """One actor's films: `start` a game, `add` each step's frame, `finish` it with its summary, which hands it
    to a thread to encode the tail and offer it as the best -- so the next game's reset is not kept waiting."""

    def __init__(self, run_dir: Path, actor: int, say=print):
        self.dir = Path(run_dir) / BEST_DIR
        self.actor, self.say = actor, say
        self._video: VideoRecorder | None = None
        self._episode = 0
        self._broken = False  # this game's film failed: the rest of it goes unfilmed
        self._pending: list[threading.Thread] = []
        self.enabled = shutil.which("ffmpeg") is not None
        if not self.enabled:
            say("no ffmpeg on PATH: the run's best game will not be filmed")

    def start(self, episode: int) -> None:
        self.abort()
        self._episode, self._broken = episode, False

    def add(self, frame: np.ndarray | None) -> None:
        if not self.enabled or self._broken or frame is None:
            return
        try:
            if self._video is None:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._video = VideoRecorder(self.dir / f".a{self.actor}_ep{self._episode:05d}.mp4", frame.shape)
            self._video.add(frame)
        except Exception as error:  # noqa: BLE001 -- a film is never worth an actor
            self.say(f"filming game {self._episode} failed, it goes unfilmed: {error}")
            self._broken = True
            self.abort()

    def finish(self, summary: dict) -> None:
        video, self._video = self._video, None
        if video is None:
            return
        self._pending = [t for t in self._pending if t.is_alive()]
        thread = threading.Thread(target=self._settle, args=(video, summary, self._episode), daemon=True)
        thread.start()
        self._pending.append(thread)

    def _settle(self, video: VideoRecorder, summary: dict, episode: int) -> None:
        try:
            if not video.close():
                self.say(f"game {episode}'s film is unusable ({video.failed}); it cannot be the best")
                video.path.unlink(missing_ok=True)
                return
            if keep_if_best(self.dir, video.path, summary, actor=self.actor, episode=episode, frames=video.frames,
                            dropped_frames=video.dropped, encoder=video.encoder[1]):
                r, p, k = episode_rank(summary)
                self.say(f"new best game: round {r}, {p} points, {k} kills -> {self.dir / 'best.mp4'}")
        except Exception as error:  # noqa: BLE001
            self.say(f"keeping game {episode}'s film failed: {error}")
            video.path.unlink(missing_ok=True)

    def abort(self) -> None:
        """Throw away the game being filmed (it did not finish)."""
        video, self._video = self._video, None
        if video is not None:
            try:
                video.abort()
            except Exception:  # noqa: BLE001
                pass

    def close(self, timeout_s: float = 120.0) -> None:
        """Abort the unfinished game and wait for finished ones to be settled."""
        self.abort()
        deadline = time.monotonic() + timeout_s
        for thread in self._pending:
            thread.join(max(0.0, deadline - time.monotonic()))
