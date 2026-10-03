"""What the policy hears: a stereo log-mel of the half second of game audio that ends at each frame.

In Nacht you hear a zombie -- a growl, footsteps, a barricade being torn down -- before you see it, and most
deaths come from behind, outside the ~65 degree view. `demos/audio.py` keeps that sound on the frames' clock;
this module turns it into something a network can read, and does it **one way for training and for play**:

* `features_at(audio, t_end)` is the only feature function. Training calls it once per step on a recording
  (`clip_features`, cached on disk because a 20-minute clip is ~18k calls), and the live player calls it every
  tick on a ring buffer of the last two seconds (`LiveAudio`). Both hand it a `ClipAudio` -- the recording's,
  or one built over the ring with the stream's own sample numbers -- so the window, the clock map and the
  filterbank are the same code, and a test checks the two agree to the bit on the same PCM.

The feature (`AudioFeatureConfig`, stored in every checkpoint that uses it):

* **Causal.** The window ends at the frame's time -- what the player had heard when they chose the step's
  action -- and no sample after it goes in. Live, samples arrive ~10-15 ms after they play, so the window
  ends at the newest sample that has arrived rather than zero-filling a tail training never saw.
* **Half a second, not the plan's 200 ms.** 25 frames of 1024-sample (21 ms) Hann windows every 20 ms,
  24064 samples = 0.501 s. A growl or a plank being ripped off lasts about that long and a 200 ms window
  catches fragments of it; footsteps come a few per second, so a shorter window often holds none. The vision
  stack is 4 frames (0.2 s); the audio is the longer memory. 20 ms frames keep onsets (a gunshot, a hit)
  sharp to within a third of a decision.
* **Stereo kept apart**, `(2, 25, 64)` -- left and right as two input channels, because the level difference
  between them is the only direction cue there is. 64 mel bands (Slaney scale, unit-peak triangles) from 0 to
  16 kHz; above that game audio carries little.
* **Log power** in dB relative to full scale, floored at -100 dB, then `(dB - offset) / scale` with fixed
  constants chosen from the recordings so values sit around [-3, 3]. Fixed rather than fitted per dataset,
  so a checkpoint and the live player cannot disagree about normalisation.

The spec's reserved `audio` observation `(2, 64)` is left as it is: changing it would change SPEC_VERSION and
refuse every labelled recording. The shape a policy actually reads is `AudioFeatureConfig.shape`, recorded in
its checkpoint.
"""

import functools
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from zombiesai.demos.audio import CHANNELS, DEFAULT_LATENCY_S, INDEX_DTYPE, ClipAudio, chunk_offsets

_BYTES_PER_SAMPLE = 2
STATS_TICKS = 9000  # LiveAudio's lag and cost statistics cover the last ten minutes at 15 Hz, not the whole run
DEFAULT_CACHE = Path.home() / ".cache" / "zombiesai" / "audio_features"
_CACHE_VERSION = 1  # bump when features_at changes in a way AudioFeatureConfig does not capture


@dataclass(frozen=True)
class AudioFeatureConfig:
    rate: int = 48_000
    n_fft: int = 1024  # 21.3 ms analysis window
    hop: int = 960  # 20 ms between frames
    n_frames: int = 25  # 0.5 s of audio per step
    n_mels: int = 64
    fmin: float = 0.0
    fmax: float = 16_000.0
    floor_db: float = -100.0
    offset_db: float = -55.0  # (dB - offset) / scale: mean ~0, spread ~1 on the recordings (-52 +/- 15 dB)
    scale_db: float = 15.0

    @property
    def window_samples(self) -> int:
        return self.n_fft + (self.n_frames - 1) * self.hop

    @property
    def window_s(self) -> float:
        return self.window_samples / self.rate

    @property
    def shape(self) -> tuple[int, int, int]:
        return (CHANNELS, self.n_frames, self.n_mels)


def feature_config(saved: dict | None) -> AudioFeatureConfig | None:
    """The feature config a checkpoint was trained with (None for one that does not hear)."""
    return None if not saved else AudioFeatureConfig(**saved)


# ------------------------------------------------------------------------------------------ the feature


def _hz_to_mel(f):
    """Slaney's mel scale: linear below 1 kHz, logarithmic above (librosa's default)."""
    f = np.asarray(f, dtype=np.float64)
    linear = f / (200.0 / 3)
    log = 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log, linear)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    linear = m * (200.0 / 3)
    log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
    return np.where(m >= 15.0, log, linear)


_FILTERBANKS: dict[tuple, np.ndarray] = {}


def mel_filterbank(config: AudioFeatureConfig) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangles, peak 1, evenly spaced on the mel scale."""
    key = (config.rate, config.n_fft, config.n_mels, config.fmin, config.fmax)
    if key not in _FILTERBANKS:
        bins = np.fft.rfftfreq(config.n_fft, 1.0 / config.rate)
        edges = _mel_to_hz(np.linspace(_hz_to_mel(config.fmin), _hz_to_mel(config.fmax), config.n_mels + 2))
        lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
        up = (bins[None] - lo) / (mid - lo)
        down = (hi - bins[None]) / (hi - mid)
        bank = np.maximum(0.0, np.minimum(up, down))
        bank.setflags(write=False)
        _FILTERBANKS[key] = bank
    return _FILTERBANKS[key]


@functools.cache
def _frame_plan(config: AudioFeatureConfig) -> tuple[np.ndarray, float]:
    """The constant parts of `log_mel`, made once per config: the window and its power normalisation."""
    hann = np.hanning(config.n_fft + 1)[:-1]  # periodic
    hann.setflags(write=False)
    return hann, (2.0 / hann.sum()) ** 2


def log_mel(window: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    """(window_samples, 2) int16 PCM -> (2, n_frames, n_mels) float32. The last frame ends on the last sample."""
    window = np.asarray(window)
    if window.shape != (config.window_samples, CHANNELS):
        raise ValueError(f"expected ({config.window_samples}, {CHANNELS}) samples, got {window.shape}")
    x = window.T.astype(np.float64) / 32768.0
    hann, scale = _frame_plan(config)
    # Frame f is samples [f * hop, f * hop + n_fft): a strided view of x, not a gathered copy of it.
    frames = np.lib.stride_tricks.sliding_window_view(x, config.n_fft, axis=-1)[:, :: config.hop]
    spectrum = np.fft.rfft(frames * hann, axis=-1)
    # Mean-square units: a full-scale sine lands at 0.5 (-3 dB) in its bin.
    power = (spectrum.real**2 + spectrum.imag**2) * scale / 2.0
    mel = power @ mel_filterbank(config).T
    db = 10.0 * np.log10(np.maximum(mel, 10.0 ** (config.floor_db / 10.0)))
    return ((db - config.offset_db) / config.scale_db).astype(np.float32)


def silence(config: AudioFeatureConfig) -> np.ndarray:
    """The feature of a window with nothing in it: what a clip without audio is padded with."""
    return log_mel(np.zeros((config.window_samples, CHANNELS), dtype=np.int16), config)


def features_at(audio: ClipAudio, t_end: float, config: AudioFeatureConfig) -> np.ndarray:
    """The feature for the audio that had played by monotonic time `t_end` -- the one function both training
    and the live player call. Uncaptured stretches are silence, exactly as `ClipAudio.window` gives them."""
    if audio.rate != config.rate or audio.channels != CHANNELS:
        raise ValueError(f"audio is {audio.rate} Hz x {audio.channels}, features want {config.rate} Hz x {CHANNELS}")
    return log_mel(audio.window(float(t_end), config.window_samples / config.rate), config)


# ------------------------------------------------------------------------------------------ training side


def _cache_key(clip, config: AudioFeatureConfig) -> str:
    meta = clip.manifest.get("audio") or {}
    files = {}
    for name in (meta.get("file"), meta.get("index"), "audio.s16", "audio.flac"):
        path = clip.path / name if name else None
        if path is not None and path.exists():
            stat = path.stat()
            files[name] = [stat.st_size, stat.st_mtime_ns]
    ident = {
        "version": _CACHE_VERSION,
        "path": str(Path(clip.path).resolve()),
        "n_steps": clip.n_steps,
        "t0": repr(float(clip.step_time(0))),
        "dt": repr(float(clip.step_time(1) - clip.step_time(0))),
        "latency_s": meta.get("latency_s"),
        "files": files,
        "features": asdict(config),
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:20]


def clip_features(clip, config: AudioFeatureConfig, cache_dir: str | Path | None = DEFAULT_CACHE):
    """(T, 2, n_frames, n_mels) float32, one feature per step of the clip, or None if it has no audio.

    Computed once per clip and feature config and kept in `cache_dir` (never beside the clip: recordings are
    read-only data), keyed on the clip's path, its audio files' sizes and times, its step clock and the
    config -- anything that would change a value. `cache_dir=None` computes in memory."""
    if clip.audio() is None or clip.n_steps == 0:
        return None
    if cache_dir is None:
        return _compute(clip, config, np.empty((clip.n_steps, *config.shape), dtype=np.float32))
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{Path(clip.path).name}-{_cache_key(clip, config)}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(clip.n_steps, *config.shape))
    _compute(clip, config, out)
    out.flush()
    del out
    tmp.replace(path)
    return np.load(path, mmap_mode="r")


def _compute(clip, config: AudioFeatureConfig, out: np.ndarray) -> np.ndarray:
    audio = clip.audio()
    times = clip.step_time(np.arange(clip.n_steps))
    for k, t in enumerate(times):
        out[k] = features_at(audio, float(t), config)
    return out


# ------------------------------------------------------------------------------------------ live side


class LiveAudio:
    """The game audio as it plays: a background thread drains a capture stream into a ring of the last few
    seconds, and `observe(t)` returns the same feature `clip_features` would have for a frame grabbed at t.

    The stream is `demos.audio.PulseMonitorStream` in the game (the default sink's *monitor*, never a
    microphone -- the stream refuses anything else) or any object with `open()`, `read() -> (bytes, t_mono)`,
    `describe()` and `close()`, which is how the tests feed it. Chunks are stamped on arrival and the clock
    map is `chunk_offsets` over the ring, the lower envelope `load_audio` applies to a recording.

    **Never blocks the decision loop.** The thread blocks on the pipe; `observe` only copies references to
    the ring's chunks under a lock and does ~1 ms of numpy. A stream that has died or gone quiet for
    `stale_s` makes `observe` answer "no audio" (a mask of 0), which the network was trained to read from the
    clips recorded without sound -- the policy falls back to its eyes rather than to stale noise.

    **Pauses do not reset it.** The frame stack is reset on a pause because frames stop being captured and the
    old ones go stale; audio keeps flowing through standby and pauses, and the recordings the policy learned
    from never stop it either -- the first window after a pause (or after a menu, in a recording) holds the
    half second that really played. Gaps in the capture are silence, in both.
    """

    def __init__(self, stream, config: AudioFeatureConfig | None = None, *, keep_s: float = 2.0,
                 stale_s: float = 0.25, clock=time.monotonic):
        self.stream = stream
        self.config = config or AudioFeatureConfig()
        rate = getattr(stream, "rate", self.config.rate)
        if rate != self.config.rate or getattr(stream, "channels", CHANNELS) != CHANNELS:
            raise ValueError(f"stream is {rate} Hz, features want {self.config.rate} Hz x {CHANNELS}")
        self.keep = int(keep_s * self.config.rate)
        self.stale_s = stale_s
        self.clock = clock
        self._lock = threading.Lock()
        self._chunks: list[tuple[np.ndarray, int, float]] = []  # (samples, sample_end, t_mono)
        self._n_samples = 0
        self._last_t: float | None = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.latency_s = DEFAULT_LATENCY_S
        self._silence = silence(self.config)
        self.counts = {"observed": 0, "no_audio": 0}
        self._lag: deque[float] = deque(maxlen=STATS_TICKS)
        self._cost: deque[float] = deque(maxlen=STATS_TICKS)

    def start(self) -> "LiveAudio":
        self.stream.open()
        described = self.stream.describe() if hasattr(self.stream, "describe") else {}
        self.latency_s = float(described.get("latency_s", DEFAULT_LATENCY_S))
        self._thread = threading.Thread(target=self._run, name="audio-hearing", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        frame_bytes = CHANNELS * _BYTES_PER_SAMPLE
        carry = b""
        try:
            while True:
                data, t = self.stream.read()
                if not data:
                    self.error = self.error or "stream ended"
                    return
                data = carry + data
                whole = len(data) - len(data) % frame_bytes
                carry = data[whole:]
                if whole:
                    self.feed(np.frombuffer(data[:whole], dtype="<i2").reshape(-1, CHANNELS), t)
        except Exception as exc:  # a dead audio server must not take the player down with it
            self.error = f"{type(exc).__name__}: {exc}"

    def feed(self, samples: np.ndarray, t_mono: float) -> None:
        """Append one chunk that arrived at `t_mono` (the thread does this; tests may call it directly)."""
        with self._lock:
            self._n_samples += len(samples)
            self._chunks.append((samples, self._n_samples, float(t_mono)))
            self._last_t = float(t_mono)
            # Whole chunks only, so every retained sample still belongs to an index record.
            while len(self._chunks) > 1 and self._n_samples - self._chunks[0][1] >= self.keep:
                self._chunks.pop(0)

    def healthy(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        return self.error is None and self._last_t is not None and now - self._last_t <= self.stale_s

    def snapshot(self) -> ClipAudio | None:
        """The ring as a `ClipAudio` numbered like the stream, so `features_at` reads it like a recording."""
        with self._lock:
            chunks = list(self._chunks)
        if not chunks:
            return None
        samples = np.concatenate([c[0] for c in chunks])
        index = np.array([(c[1], c[2]) for c in chunks], dtype=INDEX_DTYPE)
        ends = index["sample_end"].astype(np.int64)
        first = int(ends[0] - len(chunks[0][0]))
        starts = np.concatenate([[first], ends[:-1]])
        origin = chunk_offsets(index, self.config.rate) - self.latency_s
        return ClipAudio(samples, self.config.rate, starts, ends, origin, {"live": True}, first_sample=first)

    def observe(self, t_frame: float) -> tuple[np.ndarray, float]:
        """(feature, has_audio) for a frame grabbed at monotonic time `t_frame`."""
        started = time.perf_counter()
        self.counts["observed"] += 1
        audio = self.snapshot() if self.healthy() else None
        if audio is None:
            self.counts["no_audio"] += 1
            return self._silence, 0.0
        # Nothing after the newest sample has arrived yet; end there rather than on a silent tail.
        t_end = min(float(t_frame), float(audio.time_of(audio.chunk_end[-1])))
        feature = log_mel(ring_window(audio, t_end, self.config.window_samples), self.config)
        self._lag.append(float(t_frame) - t_end)
        self._cost.append(time.perf_counter() - started)
        return feature, 1.0

    def stats(self) -> dict:
        lag, cost = np.array(self._lag or [np.nan]), np.array(self._cost or [np.nan])
        return {
            **self.counts,
            "error": self.error,
            "lag_ms_median": float(np.median(lag) * 1e3),
            "lag_ms_p95": float(np.percentile(lag, 95) * 1e3),
            "cost_ms_median": float(np.median(cost) * 1e3),
            "cost_ms_p95": float(np.percentile(cost, 95) * 1e3),
        }

    def close(self) -> None:
        self.stream.close()
        if self._thread is not None:
            self._thread.join(timeout=2)


@functools.cache
def _window_offsets(n: int, rate: int) -> np.ndarray:
    offsets = (np.arange(n) - n) / rate
    offsets.setflags(write=False)
    return offsets


def ring_window(audio: ClipAudio, t_end: float, n: int) -> np.ndarray:
    """`audio.window(t_end, n / rate)`, value for value, at a fraction of its cost -- what `LiveAudio` hands
    `log_mel` every tick, so `observe` stays the feature `features_at` computes (the parity test checks it).

    The window's n sample times are evenly spaced and so sorted, and the chunk starts on a stream are sorted
    too; then which chunk each time falls in is one sorted merge of the ~200 starts into the times, not n
    binary searches, and a window wholly inside captured audio is one gather, not a masked one. Each time's
    sample number is the same expression as `ClipAudio.sample_at`. Anything else (starts out of order, a
    window reaching past what was captured) takes `ClipAudio.window` itself.
    """
    t = t_end + _window_offsets(n, audio.rate)
    starts = audio.chunk_origin + audio.chunk_start / audio.rate
    if len(starts) > 1 and not (starts[1:] >= starts[:-1]).all():
        return audio.window(t_end, n / audio.rate)
    m = len(starts)
    # i = searchsorted(starts, t, "right") - 1, by counting for each time how many starts it has reached.
    reached = np.searchsorted(t, starts, side="left")
    i = np.repeat(np.arange(-1, m), np.diff(np.concatenate(([0], reached, [n]))))
    np.clip(i, 0, m - 1, out=i)
    s = np.floor((t - audio.chunk_origin[i]) * audio.rate + 1e-6).astype(np.int64)
    ok = (s >= audio.chunk_start[i]) & (s < audio.chunk_end[i]) & (t >= starts[0]) & (s >= audio.first_sample)
    if not ok.all():
        return audio.window(t_end, n / audio.rate)
    return audio.samples[s - audio.first_sample]
