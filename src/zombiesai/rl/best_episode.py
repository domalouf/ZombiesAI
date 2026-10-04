"""The run's best game, kept as a video: `<run>/best/best.mp4`, with what it scored in `best.json`.

Which game is best is only known once it is over, so every actor films every game and, when one ends, keeps it
if it beats the run's best so far and throws it away otherwise. Best is the highest round; a tie goes to the
most kills, then the most points; a game that only ties the best does not replace it. The points and kills are
the game-over scoreboard's when it was read (`end_points` counts the 500 a game starts with, `end_kills` is the
game's own count), else the HUD's: the 500 plus every settled gain, and the gains taken for kills.

The film is the game's own picture, not the policy's 128x72: each grab the env makes on its decision deadline,
cut to every other pixel at 1440p (720p, `VIDEO_HEIGHT`) by the capture, at the decision rate. That cut costs
~1 ms of the actor's 66 ms tick (`video_frame`); the encoding is ffmpeg's, on NVENC where the machine has it,
and is fed from a thread that drops a frame rather than ever stall the game. A rehearsal's synthetic games have
no picture behind them, so they film their observations instead.

The film has the game's sound when the actor has a sink of its own to capture: picture and sound are lined up
on `time.monotonic()`, the frames by their grab times (`VideoRecorder`) and the sound by its capture's
arrival stamps (`add_sound`), so the two stay together through slow ticks, dropped frames and audio dropouts.

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
# The game's sound reaches its sink later than its picture reaches the X server: measured on the 2080 Ti PC
# (2026-10-04, Plutonium, 16 fire taps at random phase, the screen grabbed ~700 times a second and the sink
# captured with arrival stamps), a tap showed on screen after 34 ms and was heard after 57 ms -- the sound
# ~25 ms behind (mean 23, median 27).
GAME_SOUND_LAG_S = 0.025
# How much later than the frame grabbed beside it a sound is put in the film. A frame stays on screen for the
# whole of its slot but shows the instant it was grabbed, so on average the picture is half a slot behind what
# was happening; the game's own sound lag already covers most of that half, and this is the rest. Checked on
# the film of the agent's own shots (fire is pressed at the start of a tick): the muzzle flash lands in the
# grab 52 ms after the press, the gunshot in the film 46 + 8 = 54 ms after it.
AV_DELAY_S = 0.5 / spec.DECISION_HZ - GAME_SOUND_LAG_S
SOUND_BITRATE = "160k"
ALIGN_BLOCK_S = 10.0  # the sound is lined up with the film this many seconds at a time, to bound memory


def episode_rank(summary: dict) -> tuple[int, int, int]:
    """(round, kills, points): the order games are ranked in, higher first."""
    points = summary.get("end_points")
    if points is None:
        points = START_POINTS + int(summary.get("points_gained") or 0)
    kills = summary.get("end_kills")
    if kills is None:
        kills = int(summary.get("kills") or 0)
    return int(summary.get("round_reached") or 0), int(kills), int(points)


def stored_rank(record: dict) -> tuple[int, int, int]:
    """A best.json's rank, read from its named numbers: a run's best kept before kills outranked points has
    its `rank` list in the old (round, points, kills) order."""
    return int(record.get("round") or 0), int(record.get("kills") or 0), int(record.get("points") or 0)


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
    encoder is too far behind to take is dropped and counted.

    A frame given its grab time goes in the slot of a fixed `fps` grid that time falls in, counted from the
    first frame's (`t0`): a gap -- a dropped frame, a slow tick -- repeats the frame before it, and a second
    grab in a slot already written is dropped. So frame k of the film is what was on screen at `t0 + k/fps`,
    however unevenly the grabs came, which is what lets a sound track recorded on the same clock line up with
    it (`add_sound`). A frame with no time takes the next slot."""

    def __init__(self, path: Path, shape: tuple[int, ...], fps: float = spec.DECISION_HZ):
        height, width, channels = shape
        self.path, self.shape, self.fps = Path(path), tuple(shape), float(fps)
        self.encoder = encoder_args(width, height)
        self.frames = self.dropped = self.repeated = 0
        self.slots = 0  # frames in the film, repeats included
        self.t0: float | None = None  # the first frame's grab time: the film's time zero on the monotonic clock
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

    def add(self, frame: np.ndarray, t: float | None = None) -> None:
        """One frame, grabbed at monotonic time `t` (None: whenever, the slot after the last)."""
        if frame.shape != self.shape:  # the window came back at another size: not a frame of this film
            self.dropped += 1
            return
        try:
            self._queue.put_nowait((frame, t))
        except queue_mod.Full:
            self.dropped += 1

    def _feed(self) -> None:
        last = None
        while (item := self._queue.get()) is not None:
            if self.failed:
                continue  # drain, so close() never waits on a full queue
            frame, t = item
            if t is None:
                slot = self.slots
            else:
                if self.t0 is None:
                    self.t0 = t
                slot = round((t - self.t0) * self.fps)
            if slot < self.slots:  # a second grab in a slot already written
                self.dropped += 1
                continue
            try:
                while self.slots < slot and last is not None:
                    self._proc.stdin.write(last)
                    self.repeated += 1
                    self.slots += 1
                last = memoryview(np.ascontiguousarray(frame)).cast("B")
                self._proc.stdin.write(last)
                self.frames += 1
                self.slots = max(self.slots, slot) + 1
            except OSError as error:
                self.failed = f"the encoder stopped taking frames ({error})"

    @property
    def seconds(self) -> float:
        return self.slots / self.fps

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


def add_sound(video_path: Path, t0: float, seconds: float, sound_dir: Path, meta: dict,
              delay_s: float = AV_DELAY_S) -> dict:
    """Give the film at `video_path` the sound `demos.audio.AudioRecorder` captured into `sound_dir`, in place.

    The film's frame k is what was on screen at `t0 + k/fps` (VideoRecorder), and the capture's index maps its
    samples onto the same monotonic clock (`demos.audio.load_audio`: arrival stamps, jitter stripped by their
    lower envelope, the graph's fixed latency taken off). So the track is built sample by sample from that
    map -- sample m of the film is what was playing at `t0 + m/rate - delay_s` -- which also keeps it in step
    through a dropout (silence there) and the sound card's crystal drifting from the clock. What it adds to
    best.json: whether there is sound, and how much of the film's length was heard."""
    from zombiesai.demos.audio import SAMPLE_FORMAT, load_audio

    clip = load_audio(sound_dir, meta)
    n = int(round(seconds * clip.rate)) if clip is not None else 0
    if not n:
        return {"sound": "none captured"}
    block = int(ALIGN_BLOCK_S * clip.rate)
    aligned, heard = Path(sound_dir) / "aligned.s16", 0
    with open(aligned, "wb") as out:
        for first in range(0, n, block):
            m = np.arange(first, min(n, first + block))
            s = clip.sample_at(t0 - delay_s + m / clip.rate)
            ok = s >= 0
            pcm = np.zeros((len(m), clip.channels), "<i2")
            pcm[ok] = clip.samples[s[ok]]
            heard += int(ok.sum())
            out.write(pcm.tobytes())
    if not heard:
        return {"sound": "none captured"}
    muxed = Path(video_path).with_suffix(".av.mp4")
    done = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video_path),
         "-f", SAMPLE_FORMAT, "-ar", str(clip.rate), "-ac", str(clip.channels), "-i", str(aligned),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", SOUND_BITRATE,
         "-movflags", "+faststart", str(muxed)],
        capture_output=True, timeout=300)
    if done.returncode != 0:
        muxed.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited {done.returncode}: {done.stderr.decode(errors='replace')[-300:]}")
    os.replace(muxed, video_path)
    return {"sound": "aac", "sound_heard": round(heard / n, 4), "av_delay_s": delay_s}


def read_best(best_dir: Path) -> dict | None:
    try:
        return json.loads((Path(best_dir) / "best.json").read_text())
    except (OSError, ValueError):
        return None


def could_be_best(best_dir: Path, summary: dict) -> bool:
    """Whether this game would replace the run's best as it stands -- only worth asking to skip work on a game
    that cannot; `keep_if_best` decides for real, under the lock."""
    current = read_best(best_dir)
    return current is None or stored_rank(current) < episode_rank(summary)


def keep_if_best(best_dir: Path, video: Path, summary: dict, **about) -> bool:
    """Make `video` the run's best game if it outranks the one kept so far, else delete it. Whether it was kept."""
    best_dir = Path(best_dir)
    rank = episode_rank(summary)
    with open(best_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read_best(best_dir)
        if current is not None and stored_rank(current) >= rank:
            video.unlink(missing_ok=True)
            return False
        os.replace(video, best_dir / "best.mp4")
        record = {"rank": list(rank), "round": rank[0], "kills": rank[1], "points": rank[2], "video": "best.mp4",
                  "host": socket.gethostname(), "recorded_unix": time.time(), **about, "summary": summary}
        tmp = best_dir / "best.json.tmp"
        tmp.write_text(json.dumps(record, indent=2, default=str))
        tmp.replace(best_dir / "best.json")
        return True


class BestEpisodeRecorder:
    """One actor's films: `start` a game, `add` each step's frame, `finish` it with its summary, which hands it
    to a thread to encode the tail and offer it as the best -- so the next game's reset is not kept waiting.

    With `sound` -- a function returning a fresh capture stream (`demos.audio.PulseMonitorStream` on the
    instance's own sink) -- each game's sound is captured beside its film from `start`, and a finished game
    that could be the best gets it lined up and added (`add_sound`) before it is offered. A sound that fails
    costs the film its sound, never the film."""

    def __init__(self, run_dir: Path, actor: int, say=print, sound=None):
        self.dir = Path(run_dir) / BEST_DIR
        self.actor, self.say, self.sound = actor, say, sound
        self._video: VideoRecorder | None = None
        self._sound = None  # (AudioRecorder, its directory) for the game being filmed
        self._episode = 0
        self._broken = False  # this game's film failed: the rest of it goes unfilmed
        self._pending: list[threading.Thread] = []
        self.enabled = shutil.which("ffmpeg") is not None
        if not self.enabled:
            say("no ffmpeg on PATH: the run's best game will not be filmed")

    def start(self, episode: int) -> None:
        self.abort()
        self._episode, self._broken = episode, False
        if self.enabled and self.sound is not None:
            self._start_sound(episode)

    def _start_sound(self, episode: int) -> None:
        from zombiesai.demos.audio import AudioRecorder

        where = self.dir / f".a{self.actor}_ep{episode:05d}.sound"
        try:
            where.mkdir(parents=True, exist_ok=True)
            recorder = AudioRecorder(self.sound(), compress=False)
            recorder.start(where)
            self._sound = (recorder, where)
        except Exception as error:  # noqa: BLE001
            self.say(f"cannot capture the games' sound, the films will be silent: {error}")
            self.sound = None  # it would fail the same way every game
            shutil.rmtree(where, ignore_errors=True)

    def add(self, frame: np.ndarray | None, t: float | None = None) -> None:
        """A step's frame, grabbed at monotonic time `t` (None: no time, so the film cannot be given sound)."""
        if not self.enabled or self._broken or frame is None:
            return
        try:
            if self._video is None:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._video = VideoRecorder(self.dir / f".a{self.actor}_ep{self._episode:05d}.mp4", frame.shape)
            self._video.add(frame, t)
        except Exception as error:  # noqa: BLE001 -- a film is never worth an actor
            self.say(f"filming game {self._episode} failed, it goes unfilmed: {error}")
            self._broken = True
            self.abort()

    def finish(self, summary: dict) -> None:
        video, self._video = self._video, None
        sound, self._sound = self._sound, None
        if video is None:
            _drop_sound(sound)
            return
        self._pending = [t for t in self._pending if t.is_alive()]
        thread = threading.Thread(target=self._settle, args=(video, sound, summary, self._episode), daemon=True)
        thread.start()
        self._pending.append(thread)

    def _settle(self, video: VideoRecorder, sound, summary: dict, episode: int) -> None:
        try:
            sound_meta = sound[0].stop() if sound is not None else None
            if not video.close():
                self.say(f"game {episode}'s film is unusable ({video.failed}); it cannot be the best")
                video.path.unlink(missing_ok=True)
                return
            about = {}
            if sound is not None and could_be_best(self.dir, summary):
                if video.t0 is None:
                    about = {"sound": "none (the frames had no grab times to line it up with)"}
                else:
                    try:
                        about = add_sound(video.path, video.t0, video.seconds, sound[1], sound_meta)
                    except Exception as error:  # noqa: BLE001
                        self.say(f"game {episode}'s sound could not be added, it stays silent: {error}")
                        about = {"sound": f"failed: {str(error)[:200]}"}
            if keep_if_best(self.dir, video.path, summary, actor=self.actor, episode=episode, frames=video.frames,
                            dropped_frames=video.dropped, repeated_frames=video.repeated,
                            encoder=video.encoder[1], **about):
                r, k, p = episode_rank(summary)
                heard = " with sound" if about.get("sound") == "aac" else ""
                self.say(f"new best game: round {r}, {p} points, {k} kills -> {self.dir / 'best.mp4'}{heard}")
        except Exception as error:  # noqa: BLE001
            self.say(f"keeping game {episode}'s film failed: {error}")
            video.path.unlink(missing_ok=True)
        finally:
            _drop_sound(sound)

    def abort(self) -> None:
        """Throw away the game being filmed (it did not finish)."""
        video, self._video = self._video, None
        sound, self._sound = self._sound, None
        _drop_sound(sound)
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


def _drop_sound(sound) -> None:
    """Stop a game's sound capture (if it still runs) and delete what it captured."""
    if sound is None:
        return
    recorder, where = sound
    try:
        recorder.stop()  # closing twice is harmless: a settled game's was stopped before its sound was added
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(where, ignore_errors=True)
