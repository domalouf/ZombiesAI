"""Game audio kept beside a screen recording, on the same clock as the frames and the input log.

This exists because audio is the plan's best cheap upgrade to the damage detector (PLAN.md, risk 11): the
hurt grunt, the heartbeat and the growls of a zombie behind you are information the ~65 degree view simply
does not carry. A recording without audio can never be given it afterwards, so screen recordings keep the
raw stream, and the features a policy hears are computed from it later (`demos/hearing.py`: a stereo
log-mel of the half second before each frame, the same function for training and for live play).

**Alignment is the whole job.** Samples are only useful if step k's frame -- grabbed at `t0 + k/15` on
`time.monotonic()` -- maps to the samples that were playing at that moment. A sound card counts samples on
its own crystal, not on the monotonic clock, so this module never trusts the sample count alone:

* Each chunk read off the capture pipe is **stamped with `time.monotonic()` on arrival** and indexed by the
  running sample count at its end: `(sample_end, t_mono)`, one record per read, appended to
  `audio_index.bin` as it happens.
* Arrival is always *late* -- by a fixed pipeline latency plus scheduling jitter that is never negative. So at
  load time each chunk's clock offset `t_mono - sample_end / rate` is replaced by the **minimum over the next
  second of chunks** (the lower envelope): the chunks that happened to be read promptly define the clock,
  and jitter drops out. The envelope is local rather than a single line fitted to the whole file, so it
  follows crystal drift (50 ppm is 60 ms over a 20-minute recording) and survives a dropout, where the
  samples jump forward in time and the offset steps up with them.
* The fixed part of the latency -- the audio graph's buffering before the pipe, which no arrival time can
  see -- is a constant in `clip.json` (`latency_s`), applied when loading, so a better measurement later
  corrects every old recording without re-recording any of them (the same bargain as `requantize`).

Storage, per clip (see `demos/clips.py`):

    audio.s16        raw interleaved s16le PCM, appended while recording (crash-tolerant like frames.u8)
    audio.flac       the same samples, losslessly compressed after a clean stop when ffmpeg is present;
                     the raw file is deleted only once the FLAC decodes back to the same sample count
    audio_index.bin  (sample_end int64, t_mono float64) per chunk read, append-only
    clip.json        ["audio"]: rate, channels, format, device, backend, latency_s, t0_mono, chunk stats

Capture is pluggable: `AudioRecorder` takes any *stream* with `open()`, `read() -> (bytes, t_mono)`,
`describe()` and `close()`. On Linux that is `PulseMonitorStream`: `parec` on the default sink's monitor,
which works on PipeWire (through pipewire-pulse) and on plain PulseAudio alike. Only a `.monitor` source is
ever opened -- never a microphone. **Windows is not implemented**: a WASAPI loopback stream (for instance
through `pyaudiowpatch` or `soundcard`) would read the default render endpoint in loopback mode and stamp
each buffer with `time.monotonic()`, which on Windows is QueryPerformanceCounter -- the clock the Raw Input
log already uses -- and everything downstream of `read()` would be unchanged.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_FORMAT = "s16le"
_BYTES_PER_SAMPLE = 2
FRAGMENT_MS = 10  # what parec is asked to deliver per read; smaller means less jitter to strip, more records
# The latency no arrival stamp can see: the graph's buffering between the sink mixing a sample and the pipe
# handing it over. On this machine the game runs PipeWire at a 256-sample quantum (5.3 ms) and the monitor
# reports 0 us configured, so one quantum is the estimate. It is an estimate, not a measurement -- see
# docs/demos.md -- and it lives in clip.json so a better number can be applied to old recordings.
DEFAULT_LATENCY_S = 0.005
ENVELOPE_WINDOW_S = 1.0

PCM_FILE = "audio.s16"
FLAC_FILE = "audio.flac"
INDEX_FILE = "audio_index.bin"
INDEX_DTYPE = np.dtype([("sample_end", "<i8"), ("t_mono", "<f8")])


# --------------------------------------------------------------------------------------------- capture


class PulseMonitorStream:
    """Linux: the monitor of an output sink, read as raw PCM from `parec`.

    `parec` rather than `pw-record` because it speaks the pulse protocol, which PipeWire serves through
    pipewire-pulse and plain PulseAudio serves natively -- one command for either. It also resamples to the
    rate asked for, so the file format is fixed however the sink is configured (this one runs s32le).

    The whole output is recorded, so anything else playing -- a video, a notification, voice chat -- is in
    the recording too. The game under Proton opens several streams (four here: three float stereo, one 8 kHz
    mono), so single-app capture would mean one `parec --monitor-stream=<sink-input index>` per stream and a
    mix; PipeWire could instead route the game to its own null sink with a loopback to the speakers. Neither
    is worth it until something else playing actually spoils a recording.
    """

    def __init__(self, device: str | None = None, *, rate: int = SAMPLE_RATE, channels: int = CHANNELS,
                 fragment_ms: int = FRAGMENT_MS, latency_s: float = DEFAULT_LATENCY_S,
                 stream_name: str = "demo recording"):
        self.device = device
        self.stream_name = stream_name
        self.rate, self.channels = rate, channels
        self.fragment_ms = fragment_ms
        self.latency_s = latency_s
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def default_monitor() -> str:
        """The default *sink's* monitor. Deliberately not the default source, which is usually a microphone."""
        sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, check=True)
        return sink.stdout.strip() + ".monitor"

    def command(self) -> list[str]:
        if not self.device or not self.device.endswith(".monitor"):
            # The one rule that keeps this a game recorder and not a room recorder.
            raise ValueError(f"refusing to record {self.device!r}: only a sink's .monitor source is allowed")
        return [
            "parec", f"--device={self.device}", "--raw", f"--format={SAMPLE_FORMAT}", f"--rate={self.rate}",
            f"--channels={self.channels}", f"--latency-msec={self.fragment_ms}", "--client-name=zombiesai",
            f"--stream-name={self.stream_name}",
        ]

    def open(self) -> None:
        if shutil.which("parec") is None:
            raise RuntimeError("parec not found; install pipewire-pulse (or pulseaudio) utilities for --audio")
        if self.device is None:
            self.device = self.default_monitor()
        self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self) -> tuple[bytes, float]:
        """Whatever the pipe holds (at most 64 KiB), stamped the instant the read returns. b"" at the end."""
        data = os.read(self._proc.stdout.fileno(), 1 << 16)
        return data, time.monotonic()

    def describe(self) -> dict:
        return {"backend": "pulse-monitor", "tool": "parec", "device": self.device,
                "fragment_ms": self.fragment_ms, "latency_s": self.latency_s}

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


class AudioRecorder:
    """Drains a stream on a background thread into `audio.s16` + `audio_index.bin` in a clip directory.

    A thread rather than the decision loop because audio arrives every 10 ms and the loop wakes every 67:
    reading it from the loop would stamp it up to a decision late, which is exactly the error the index
    exists to avoid. The thread only appends to its own two files, so the recorder never waits on it.
    """

    def __init__(self, stream, *, compress: bool = True):
        self.stream = stream
        self.compress = compress
        self.path: Path | None = None
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._n_samples = 0
        self._n_chunks = 0
        self._first_t: float | None = None
        self._last_t: float | None = None

    @property
    def rate(self) -> int:
        return getattr(self.stream, "rate", SAMPLE_RATE)

    @property
    def channels(self) -> int:
        return getattr(self.stream, "channels", CHANNELS)

    def start(self, clip_dir: str | Path) -> dict:
        self.path = Path(clip_dir)
        self._pcm = open(self.path / PCM_FILE, "ab", buffering=0)
        self._index = open(self.path / INDEX_FILE, "ab", buffering=0)
        self.stream.open()
        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        self._thread.start()
        return self.meta()

    def _run(self) -> None:
        frame_bytes = self.channels * _BYTES_PER_SAMPLE
        carry = b""
        try:
            while True:
                data, t = self.stream.read()
                if not data:
                    return
                # A pipe read can end mid-sample; only whole frames are written and counted, so the sample
                # count in the index is always the number of frames in the file before that point.
                data = carry + data
                whole = len(data) - len(data) % frame_bytes
                carry = data[whole:]
                if not whole:
                    continue
                # PCM first, then the record that points at it: a crash between the two leaves unindexed
                # samples (trimmed on load), never an index entry pointing past the data.
                self._pcm.write(data[:whole])
                self._n_samples += whole // frame_bytes
                self._index.write(np.array([(self._n_samples, t)], dtype=INDEX_DTYPE).tobytes())
                self._n_chunks += 1
                if self._first_t is None:
                    self._first_t = t
                self._last_t = t
        except Exception as exc:  # a dead audio server must not take the recording down with it
            self._error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> dict:
        """Stop capture, close the files, compress if possible, and return the manifest entry."""
        self.stream.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self._error = self._error or "capture thread did not stop"
        self._pcm.close()
        self._index.close()
        meta = self.meta()
        if self.compress and self._error is None and self._n_samples:
            meta.update(compress_to_flac(self.path, self.rate, self.channels, self._n_samples))
        return meta

    def meta(self) -> dict:
        described = self.stream.describe() if hasattr(self.stream, "describe") else {}
        return {
            "rate": self.rate,
            "channels": self.channels,
            "format": SAMPLE_FORMAT,
            "file": PCM_FILE,
            "index": INDEX_FILE,
            "latency_s": described.get("latency_s", DEFAULT_LATENCY_S),
            "source": described,
            "n_samples": self._n_samples,
            "n_chunks": self._n_chunks,
            "seconds": self._n_samples / self.rate,
            "mean_chunk_ms": 1000 * self._n_samples / self.rate / max(self._n_chunks, 1),
            "wall_seconds": (self._last_t - self._first_t) if self._first_t is not None else 0.0,
            "error": self._error,
        }


def compress_to_flac(clip_dir: Path, rate: int, channels: int, n_samples: int) -> dict:
    """Losslessly shrink the raw PCM (~11.5 MB/min) to FLAC, keeping the raw file unless the result checks out.

    After the recording rather than during it, so a crash mid-session still leaves a readable raw file -- a
    FLAC stream cut off by a crash has no clean ending to trust.
    """
    if shutil.which("ffmpeg") is None:
        return {"compression": "none (no ffmpeg)"}
    raw, flac = clip_dir / PCM_FILE, clip_dir / FLAC_FILE
    fmt = ["-f", SAMPLE_FORMAT, "-ar", str(rate), "-ac", str(channels)]
    done = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *fmt, "-i", str(raw), "-c:a", "flac", str(flac)],
        capture_output=True,
    )
    if done.returncode != 0 or len(decode_flac(flac, channels)) != n_samples:
        flac.unlink(missing_ok=True)
        return {"compression": f"none (ffmpeg failed: {done.stderr.decode(errors='replace')[-200:]})"}
    raw.unlink()
    return {"file": FLAC_FILE, "compression": "flac", "bytes": flac.stat().st_size}


def decode_flac(path: Path, channels: int) -> np.ndarray:
    done = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", SAMPLE_FORMAT, "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(done.stdout, dtype="<i2").reshape(-1, channels)


# --------------------------------------------------------------------------------------------- loading


def chunk_offsets(index: np.ndarray, rate: int, window_s: float = ENVELOPE_WINDOW_S) -> np.ndarray:
    """Per chunk, the monotonic time of sample 0 as that chunk's neighbourhood best estimates it.

    `t_mono - sample_end / rate` would be that time exactly if chunks arrived with no delay. They arrive
    late by jitter that is never negative, so the promptest chunk in the next `window_s` is the best
    witness. The window looks *forward* so that after a dropout (the offset steps up) no chunk borrows an
    offset from before it; the chunks just before a dropout see a shorter window, which only costs them a
    little jitter.
    """
    offsets = index["t_mono"] - index["sample_end"] / rate
    if len(offsets) < 2:
        return offsets
    per_chunk = np.median(np.diff(index["sample_end"])) / rate
    width = int(min(len(offsets), max(1, np.ceil(window_s / max(per_chunk, 1e-6)))))
    padded = np.concatenate([offsets, np.full(width - 1, np.inf)])
    return np.lib.stride_tricks.sliding_window_view(padded, width).min(axis=1)


@dataclass
class ClipAudio:
    """A clip's audio plus the map between its sample numbers and the recorder's monotonic clock."""

    samples: np.ndarray  # (N, channels) int16
    rate: int
    chunk_start: np.ndarray  # first sample of each chunk
    chunk_end: np.ndarray  # one past its last sample
    chunk_origin: np.ndarray  # monotonic time of sample 0 according to this chunk, latency removed
    meta: dict
    # Sample number of samples[0]. Always 0 for a recording; a live ring buffer (demos/hearing.py) keeps only
    # the last seconds but numbers them as the stream did, so its times come out of the same arithmetic.
    first_sample: int = 0

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate

    def time_of(self, sample) -> np.ndarray:
        """Monotonic time at which a sample was playing (float array like `sample`)."""
        sample = np.asarray(sample)
        i = np.clip(np.searchsorted(self.chunk_end, sample, side="right"), 0, len(self.chunk_end) - 1)
        return self.chunk_origin[i] + sample / self.rate

    def sample_at(self, t) -> np.ndarray:
        """The sample playing at monotonic time `t`, or -1 where nothing was captured (before, after, or in a
        dropout). Vectorised over `t`."""
        t = np.asarray(t, dtype=np.float64)
        starts = self.chunk_origin + self.chunk_start / self.rate
        i = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(starts) - 1)
        s = np.floor((t - self.chunk_origin[i]) * self.rate + 1e-6).astype(np.int64)
        ok = (s >= self.chunk_start[i]) & (s < self.chunk_end[i]) & (t >= starts[0]) & (s >= self.first_sample)
        return np.where(ok, s, -1)

    def window(self, t_end: float, seconds: float) -> np.ndarray:
        """The `seconds` of audio that had played by monotonic time `t_end`, (n, channels) int16.

        Uncaptured stretches come back as silence, so every window has the same shape -- what a log-mel
        feature wants -- and a caller that cares can check `sample_at` itself.
        """
        n = int(round(seconds * self.rate))
        s = self.sample_at(t_end + (np.arange(n) - n) / self.rate)
        out = np.zeros((n, self.channels), dtype=self.samples.dtype)
        ok = s >= 0
        out[ok] = self.samples[s[ok] - self.first_sample]
        return out


def load_audio(clip_dir: str | Path, meta: dict) -> ClipAudio | None:
    """A clip's audio, trimmed to what the index vouches for. Tolerates a crash mid-recording: a torn final
    sample or index record is dropped, and samples written after the last index record are unmapped."""
    clip_dir = Path(clip_dir)
    rate, channels = int(meta["rate"]), int(meta["channels"])
    raw, flac = clip_dir / PCM_FILE, clip_dir / FLAC_FILE
    if raw.exists():
        n = raw.stat().st_size // (_BYTES_PER_SAMPLE * channels)  # a torn final frame is not a sample
        samples = np.memmap(raw, dtype="<i2", mode="r", shape=(n, channels)) if n else np.zeros((0, channels), "<i2")
    elif flac.exists():
        samples = decode_flac(flac, channels)
    else:
        return None
    index_bytes = (clip_dir / INDEX_FILE).read_bytes() if (clip_dir / INDEX_FILE).exists() else b""
    index = np.frombuffer(index_bytes[: len(index_bytes) - len(index_bytes) % INDEX_DTYPE.itemsize], INDEX_DTYPE)
    index = index[index["sample_end"] <= len(samples)]
    if not len(index):
        return None
    samples = samples[: int(index["sample_end"][-1])]
    ends = index["sample_end"].astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    origin = chunk_offsets(index, rate) - float(meta.get("latency_s", DEFAULT_LATENCY_S))
    return ClipAudio(samples, rate, starts, ends, origin, meta)


def describe_audio(clip_dir: str | Path) -> str:
    """One line for the end of a recording: how much was captured, and whether it kept pace."""
    meta = json.loads((Path(clip_dir) / "clip.json").read_text()).get("audio")
    if not meta:
        return "no audio"
    if meta.get("error"):
        return f"audio stopped early: {meta['error']}"
    size = meta.get("bytes") or (Path(clip_dir) / PCM_FILE).stat().st_size
    pace = meta["seconds"] / meta["wall_seconds"] if meta.get("wall_seconds") else float("nan")
    return (
        f"{meta['seconds']:.1f} s of audio from {meta['source'].get('device')}, "
        f"{meta.get('compression', 'raw')} {size / 2**20:.1f} MiB, "
        f"{meta['mean_chunk_ms']:.1f} ms chunks, sample clock {pace:.4f}x wall"
    )
