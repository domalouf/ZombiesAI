"""Game audio beside a recording: no sound card here, so a fake stream feeds PCM whose every sample says which
sample it is, arriving at timestamps the test chose. Alignment is then checkable to the sample."""

import json
import shutil
import time

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import audio as audiomod
from zombiesai.demos.audio import INDEX_DTYPE, AudioRecorder, ClipAudio, PulseMonitorStream, chunk_offsets
from zombiesai.demos.clips import load_clip
from zombiesai.demos.inputs import InputConfig
from zombiesai.demos.recorder import RecorderConfig, record
from zombiesai.synthetic import SyntheticSource

RATE = 48_000
CHUNK = 480  # 10 ms, what parec delivers at --latency-msec=10


def labelled_pcm(start: int, n: int) -> np.ndarray:
    """Sample i carries (i mod 2^15) on the left and its negation on the right: content that is its own index."""
    i = (np.arange(start, start + n) % 32768).astype(np.int16)
    return np.stack([i, -i], axis=1)


def sample_id(frames: np.ndarray) -> np.ndarray:
    return frames[:, 0].astype(np.int64)


def samples_apart(a, b) -> int:
    """Distance between two sample ids, which wrap at 2^15."""
    d = (int(a) - int(b)) % 32768
    return min(d, 32768 - d)


class FakeStream:
    """Plays `seconds` of labelled PCM. Sample s is produced at `origin + s / (RATE * (1 + drift))` and each
    chunk arrives `latency + jitter` after its last sample. `drop` = (after_sample, n_lost) models a dropout:
    samples lost in the audio server, so the sample count falls behind the clock. Reads may be split at odd
    byte offsets to exercise torn frames."""

    rate, channels = RATE, 2

    def __init__(self, seconds=3.0, origin=None, latency=0.004, jitter_ms=4.0, drift=0.0, drop=None, split=False,
                 seed=0):
        self.seconds, self.origin, self.latency = seconds, origin, latency
        self.jitter_ms, self.drift, self.drop, self.split = jitter_ms, drift, drop, split
        self.rng = np.random.default_rng(seed)
        self.reads = []

    def produced_at(self, s):
        """True monotonic time sample `s` (in the file's numbering) was playing."""
        s = np.asarray(s, dtype=np.float64)
        if self.drop:
            s = s + np.where(s >= self.drop[0], self.drop[1], 0)
        return self.origin + s / (RATE * (1 + self.drift))

    def open(self):
        if self.origin is None:
            self.origin = time.monotonic()
        n = int(self.seconds * RATE)
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            data = labelled_pcm(start, end - start).tobytes()
            # Jitter is never negative, and a fair share of chunks arrive promptly -- as measured on PipeWire.
            jitter = self.rng.exponential(self.jitter_ms / 1000) * (self.rng.random() > 0.3)
            t = float(self.produced_at(end - 1) + 1 / RATE + self.latency + jitter)
            if self.split:
                cut = 1 + int(self.rng.integers(0, len(data) - 1))
                self.reads += [(data[:cut], t), (data[cut:], t)]
            else:
                self.reads.append((data, t))
        self.reads.append((b"", float("inf")))

    def read(self):
        return self.reads.pop(0)

    def describe(self):
        return {"backend": "fake", "device": "fake.monitor", "latency_s": self.latency}

    def close(self):
        pass


def recorded_audio(tmp_path, stream, compress=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    recorder = AudioRecorder(stream, compress=compress)
    recorder.start(tmp_path)
    meta = recorder.stop()
    return audiomod.load_audio(tmp_path, meta), meta


def test_every_sample_arrives_and_is_indexed_even_when_reads_tear_a_frame(tmp_path):
    audio, meta = recorded_audio(tmp_path, FakeStream(seconds=1.0, origin=100.0, split=True))
    assert meta["n_samples"] == RATE and meta["error"] is None
    np.testing.assert_array_equal(np.asarray(audio.samples), labelled_pcm(0, RATE))
    assert (np.diff(audio.chunk_end) > 0).all() and audio.chunk_end[-1] == RATE


def test_a_monotonic_time_maps_to_the_sample_playing_then_despite_jitter(tmp_path):
    stream = FakeStream(seconds=3.0, origin=100.0)
    audio, _ = recorded_audio(tmp_path, stream)
    t = 100.0 + np.linspace(0.05, 2.9, 200)
    truth = np.floor((t - 100.0) * RATE)
    # Up to 4 ms of jitter on every read is stripped to the promptest chunk: well under a millisecond left.
    assert np.abs(audio.sample_at(t) - truth).max() <= RATE * 0.001


def test_a_step_window_ends_at_the_frame_and_holds_consecutive_samples(tmp_path):
    audio, _ = recorded_audio(tmp_path, FakeStream(seconds=2.0, origin=50.0))
    window = audio.window(51.0, 0.2)
    ids = sample_id(window)
    assert window.shape == (int(0.2 * RATE), 2)
    assert (np.diff(ids) == 1).all()  # a copy of the recording, not a resampling of it
    assert samples_apart(ids[-1], RATE - 1) <= RATE * 0.001  # the last one is the sample playing at t = 51.0


def test_crystal_drift_is_followed_rather_than_accumulated(tmp_path):
    """50 ppm is 3 ms a minute; a single sample-count clock would be 60 ms out by the end of a 20 minute session."""
    stream = FakeStream(seconds=20.0, origin=0.0, drift=500e-6)  # exaggerated so 20 s shows it
    audio, _ = recorded_audio(tmp_path, stream)
    s = np.array([RATE, 10 * RATE, 19 * RATE])
    naive = s / RATE
    assert np.abs(audio.time_of(s) - stream.produced_at(s)).max() < 0.001
    assert np.abs(naive - stream.produced_at(s)).max() > 0.009


def test_a_dropout_is_silence_and_the_audio_after_it_stays_aligned(tmp_path):
    lost = RATE // 4
    stream = FakeStream(seconds=2.0, origin=10.0, drop=(RATE - RATE % CHUNK, lost))
    audio, _ = recorded_audio(tmp_path, stream)
    after = 10.0 + 1.6  # well past the gap, which spans [~1.0, 1.25) s
    expected = np.floor((after - 10.0) * RATE) - lost
    assert abs(int(audio.sample_at(after)) - expected) <= RATE * 0.001
    inside = 10.0 + (RATE - RATE % CHUNK) / RATE + 0.1
    assert audio.sample_at(inside) == -1
    assert not audio.window(inside, 0.05).any()


def test_times_outside_the_recording_are_silence_not_the_nearest_sample(tmp_path):
    audio, _ = recorded_audio(tmp_path, FakeStream(seconds=1.0, origin=10.0))
    assert audio.sample_at(9.0) == -1 and audio.sample_at(12.0) == -1
    window = audio.window(10.1, 0.2)  # half before the first sample
    assert not window[: int(0.1 * RATE) - 100].any() and window[-1].any()


def test_a_crash_mid_recording_loses_only_the_torn_tail(tmp_path):
    audio, meta = recorded_audio(tmp_path, FakeStream(seconds=1.0, origin=10.0))
    pcm, index = tmp_path / audiomod.PCM_FILE, tmp_path / audiomod.INDEX_FILE
    with open(pcm, "r+b") as f:
        f.truncate(pcm.stat().st_size - 1001)  # the process died mid-write: a torn sample
    with open(index, "r+b") as f:
        f.truncate(index.stat().st_size - 5)  # and mid-record in the index
    torn = audiomod.load_audio(tmp_path, meta)
    n = len(torn.samples)
    assert 0 < RATE - n < 2 * CHUNK  # trimmed back to the last chunk both files vouch for
    assert n == torn.chunk_end[-1] and n % CHUNK == 0
    np.testing.assert_array_equal(np.asarray(torn.samples), labelled_pcm(0, n))
    assert abs(int(torn.sample_at(10.5)) - int(audio.sample_at(10.5))) <= RATE * 0.001


def test_an_empty_index_means_no_audio_rather_than_an_error(tmp_path):
    (tmp_path / audiomod.PCM_FILE).write_bytes(b"\0" * 400)
    (tmp_path / audiomod.INDEX_FILE).write_bytes(b"")
    assert audiomod.load_audio(tmp_path, {"rate": RATE, "channels": 2}) is None


def test_the_envelope_uses_the_promptest_chunk_nearby_and_never_one_from_before_a_dropout():
    index = np.zeros(6, dtype=INDEX_DTYPE)
    index["sample_end"] = np.arange(1, 7) * RATE // 100
    offsets = np.array([0.003, 0.0, 0.004, 0.502, 0.500, 0.501])
    index["t_mono"] = offsets + index["sample_end"] / RATE
    got = chunk_offsets(index, RATE, window_s=0.02)
    np.testing.assert_allclose(got, [0.0, 0.0, 0.004, 0.5, 0.5, 0.501])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_compression_is_lossless_and_replaces_the_raw_file_only_once_it_checks_out(tmp_path):
    raw, raw_meta = recorded_audio(tmp_path / "raw", FakeStream(seconds=1.0, origin=10.0))
    flac, meta = recorded_audio(tmp_path / "flac", FakeStream(seconds=1.0, origin=10.0), compress=True)
    assert meta["compression"] == "flac" and meta["file"] == audiomod.FLAC_FILE
    assert not (tmp_path / "flac" / audiomod.PCM_FILE).exists()
    np.testing.assert_array_equal(flac.samples, np.asarray(raw.samples))
    np.testing.assert_array_equal(flac.window(10.5, 0.2), raw.window(10.5, 0.2))


def test_only_a_monitor_is_ever_opened_never_a_microphone():
    with pytest.raises(ValueError, match="monitor"):
        PulseMonitorStream("alsa_input.usb-Blue_Yeti-00.analog-stereo").command()
    with pytest.raises(ValueError):
        PulseMonitorStream(None).command()
    command = PulseMonitorStream("alsa_output.hdmi-stereo.monitor").command()
    assert command[0] == "parec" and "--device=alsa_output.hdmi-stereo.monitor" in command
    assert "--raw" in command and "--format=s16le" in command and "--latency-msec=10" in command


def test_a_stream_that_dies_mid_recording_is_noted_and_keeps_what_it_had(tmp_path):
    class Dying(FakeStream):
        def read(self):
            if len(self.reads) < 50:
                raise OSError("pipewire went away")
            return super().read()

    audio, meta = recorded_audio(tmp_path, Dying(seconds=1.0, origin=10.0))
    assert "pipewire went away" in meta["error"]
    assert 0 < len(audio.samples) < RATE


# ------------------------------------------------------------------------------------ recorder integration

CONFIG = InputConfig(counts_per_degree=10.0)


def synthetic_recording(tmp_path, steps=40, audio=None):
    source = SyntheticSource(seed=0, max_steps=steps)
    config = RecorderConfig(max_steps=steps, realtime=False, input=CONFIG)
    path = record(source, source, tmp_path / "demo", config, stop=lambda: source.done, progress_every=0, audio=audio)
    return load_clip(path)


def test_a_recording_with_audio_maps_each_step_to_the_sound_playing_at_its_frame(tmp_path):
    # Audio that started playing 0.5 s before the recorder -- as it does, since capture opens before t0.
    stream = FakeStream(seconds=4.0, origin=time.monotonic() - 0.5)
    clip = synthetic_recording(tmp_path, audio=AudioRecorder(stream, compress=False))
    manifest = json.loads((clip.path / "clip.json").read_text())
    meta = manifest["audio"]
    assert meta["rate"] == RATE and meta["channels"] == 2 and meta["source"]["device"] == "fake.monitor"
    assert meta["t0_mono"] == manifest["summary"]["t0_mono"] and meta["latency_s"] == stream.latency
    t0 = manifest["summary"]["t0_mono"]
    for k in (0, 1, 7, clip.n_steps - 1):
        window = clip.audio_for_step(k, window_s=0.2)
        assert window.shape == (int(0.2 * RATE), 2)
        playing = np.floor((t0 + k / spec.DECISION_HZ - stream.origin) * RATE) % 32768
        assert samples_apart(sample_id(window)[-1], playing) <= RATE * 0.001


def test_a_recording_that_never_closed_still_knows_where_step_zero_was(tmp_path):
    stream = FakeStream(seconds=4.0, origin=time.monotonic() - 0.5)
    clip = synthetic_recording(tmp_path, audio=AudioRecorder(stream, compress=False))
    manifest = json.loads((clip.path / "clip.json").read_text())
    del manifest["summary"]
    (clip.path / "clip.json").write_text(json.dumps(manifest))
    reopened = load_clip(clip.path)
    np.testing.assert_array_equal(reopened.audio_for_step(3), clip.audio_for_step(3))


def test_recording_without_audio_is_unchanged(tmp_path):
    clip = synthetic_recording(tmp_path)
    assert "audio" not in clip.manifest
    assert clip.audio() is None and clip.audio_for_step(0) is None
    assert not list(clip.path.glob("audio*"))
    assert sorted(p.name for p in clip.path.iterdir()) == ["clip.json", "frames.u8", "inputs.jsonl", "labels.npz"]


def test_an_audio_stream_that_will_not_open_stops_the_recording_before_it_starts(tmp_path):
    class Broken(FakeStream):
        def open(self):
            raise RuntimeError("parec not found")

    with pytest.raises(RuntimeError, match="parec"):
        synthetic_recording(tmp_path, audio=AudioRecorder(Broken()))


def test_clip_audio_reports_its_shape():
    audio = ClipAudio(labelled_pcm(0, RATE), RATE, np.array([0]), np.array([RATE]), np.array([0.0]), {})
    assert audio.channels == 2 and audio.seconds == 1.0
