"""The policy's hearing: one feature function for training and play, causal, and harmless when absent.

No sound card and no live stream here: PCM comes from the same labelled fake stream test_audio.py uses, fed
to the recorder (training's side) and to `LiveAudio` (the player's side)."""

import threading
import time
from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_audio import RATE, FakeStream, recorded_audio, synthetic_recording

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.demos.agent import BCAgent
from zombiesai.demos.audio import AudioRecorder, ClipAudio
from zombiesai.demos.clips import clip_span
from zombiesai.demos.dataset import ClipDataset, DataConfig
from zombiesai.demos.hearing import (
    AudioFeatureConfig,
    LiveAudio,
    clip_features,
    features_at,
    log_mel,
    mel_filterbank,
    silence,
)
from zombiesai.rl.encoders import PixelActorCritic

CFG = AudioFeatureConfig()


def noise_audio(seconds=3.0, seed=0) -> ClipAudio:
    """Stereo noise with one chunk per 10 ms, sample 0 playing at t=100 s."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    samples = rng.integers(-8000, 8000, size=(n, 2)).astype(np.int16)
    ends = np.arange(480, n + 1, 480)
    starts = np.concatenate([[0], ends[:-1]])
    return ClipAudio(samples, RATE, starts, ends, np.full(len(ends), 100.0), {})


# ------------------------------------------------------------------------------------------ the feature


def test_the_feature_has_the_configured_shape_and_every_mel_band_sees_some_spectrum():
    assert CFG.window_samples == 24064 and abs(CFG.window_s - 0.5) < 0.01
    assert (mel_filterbank(CFG).sum(axis=1) > 0).all()
    f = features_at(noise_audio(), 101.0, CFG)
    assert f.shape == CFG.shape == (2, 25, 64) and f.dtype == np.float32
    assert np.isfinite(f).all()


def test_the_feature_is_deterministic():
    audio = noise_audio()
    a, b = features_at(audio, 101.3, CFG), features_at(audio, 101.3, CFG)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, features_at(noise_audio(), 101.3, CFG))


def test_the_feature_is_causal_nothing_after_the_frame_goes_in():
    audio = noise_audio()
    t = 101.5
    last = int(round((t - 100.0) * RATE)) - 1  # the last sample playing before t
    before = features_at(audio, t, CFG)
    later = ClipAudio(audio.samples.copy(), RATE, audio.chunk_start, audio.chunk_end, audio.chunk_origin, {})
    later.samples[last + 1 :] = 32000  # everything after the frame, as loud as it gets
    np.testing.assert_array_equal(features_at(later, t, CFG), before)
    later.samples[last] = 32000  # ...but the sample just before it is heard, in the newest frame only
    changed = features_at(later, t, CFG) != before
    assert changed[:, -1].any() and not changed[:, :-1].any()


def test_left_and_right_stay_apart():
    audio = noise_audio()
    audio.samples[:, 1] = 0  # sound on the left only
    f = features_at(audio, 101.0, CFG)
    np.testing.assert_array_equal(f[1], silence(CFG)[1])
    assert (f[0] > f[1]).mean() > 0.9


def test_log_mel_refuses_a_window_of_the_wrong_length():
    with pytest.raises(ValueError):
        log_mel(np.zeros((9600, 2), np.int16), CFG)


# ------------------------------------------------------------------------------------------ training side


@pytest.fixture(scope="module")
def heard_clip(tmp_path_factory):
    stream = FakeStream(seconds=4.0, origin=time.monotonic() - 0.5)
    return synthetic_recording(tmp_path_factory.mktemp("heard"), audio=AudioRecorder(stream, compress=False))


@pytest.fixture(scope="module")
def silent_clip(tmp_path_factory):
    return synthetic_recording(tmp_path_factory.mktemp("silent"))


def test_clip_features_are_the_feature_function_at_each_steps_frame(heard_clip):
    features = clip_features(heard_clip, CFG, cache_dir=None)
    assert features.shape == (heard_clip.n_steps, *CFG.shape)
    for k in (0, 5, heard_clip.n_steps - 1):
        np.testing.assert_array_equal(features[k], features_at(heard_clip.audio(), heard_clip.step_time(k), CFG))


def test_clip_features_are_cached_outside_the_clip_and_reused(heard_clip, tmp_path):
    before = sorted(p.name for p in heard_clip.path.iterdir())
    first = clip_features(heard_clip, CFG, cache_dir=tmp_path / "cache")
    cached = list((tmp_path / "cache").glob("*.npy"))
    assert len(cached) == 1
    assert sorted(p.name for p in heard_clip.path.iterdir()) == before  # nothing written beside the recording
    mtime = cached[0].stat().st_mtime_ns
    second = clip_features(heard_clip, CFG, cache_dir=tmp_path / "cache")
    assert cached[0].stat().st_mtime_ns == mtime  # read back, not recomputed
    np.testing.assert_array_equal(first, second)
    other = clip_features(heard_clip, AudioFeatureConfig(n_mels=32), cache_dir=tmp_path / "cache")
    assert other.shape[-1] == 32 and len(list((tmp_path / "cache").glob("*.npy"))) == 2


def test_a_span_of_a_clip_hears_what_those_steps_heard(heard_clip):
    span = clip_span(heard_clip, 10, 30)
    assert span.n_steps == 20
    np.testing.assert_array_equal(
        clip_features(span, CFG, None), clip_features(heard_clip, CFG, None)[10:30]
    )
    np.testing.assert_array_equal(span.actions, heard_clip.actions[10:30])


def test_a_clip_without_audio_trains_masked_rather_than_faked(heard_clip, silent_clip):
    assert clip_features(silent_clip, CFG, None) is None
    data = ClipDataset([heard_clip, silent_clip], before=1, audio=lambda c: clip_features(c, CFG, None),
                       audio_config=CFG)
    rows = data.index
    batch = data.batch(rows)
    heard = rows[:, 0] == 0
    assert batch["audio"].shape == (len(rows), *CFG.shape)
    np.testing.assert_array_equal(batch["audio_mask"], heard.astype(np.float32))
    assert (batch["audio"][~heard] == silence(CFG)).all()
    # Gain jitter moves only what was heard.
    jittered = data.batch(rows, np.random.default_rng(0), augment=True)
    assert (jittered["audio"][~heard] == silence(CFG)).all()
    assert not np.array_equal(jittered["audio"][heard], batch["audio"][heard])


def test_a_dataset_without_hearing_is_unchanged(heard_clip):
    batch = ClipDataset([heard_clip], before=1).batch(np.array([[0, 3]]))
    assert "audio" not in batch and "audio_mask" not in batch


# ------------------------------------------------------------------------------------------ the network


def test_masked_audio_contributes_nothing_whatever_it_holds():
    torch.manual_seed(0)
    net = PixelActorCritic(frame_stack=2, hidden=32, audio_shape=CFG.shape, audio_dim=16).eval()
    pixels = torch.randint(0, 255, (3, 2, *spec.PIXELS_SHAPE), dtype=torch.uint8)
    mask = torch.zeros(3)
    a, _, _ = net(pixels, None, torch.randn(3, *CFG.shape), mask)
    b, _, _ = net(pixels, None, torch.randn(3, *CFG.shape), mask)
    torch.testing.assert_close(a, b)
    c, _, _ = net(pixels, None, torch.randn(3, *CFG.shape), torch.ones(3))
    assert not torch.allclose(a, c)
    with pytest.raises(ValueError, match="hears"):
        net(pixels)


def test_a_deaf_network_is_the_network_it_always_was():
    torch.manual_seed(0)
    net = PixelActorCritic(frame_stack=2, hidden=32)
    assert not any(k.startswith(("audio_encoder", "mixer")) for k in net.state_dict())
    pixels = torch.randint(0, 255, (2, 2, *spec.PIXELS_SHAPE), dtype=torch.uint8)
    torch.testing.assert_close(net.features(pixels), net.encoder(pixels))
    vector = PixelActorCritic(frame_stack=2, hidden=32, vector_dim=5)
    assert {k for k in vector.state_dict() if k.startswith("mixer")} == {"mixer.0.weight", "mixer.0.bias"}
    assert vector.mixer[0].in_features == 37


def test_a_checkpoint_from_before_audio_loads_and_plays_as_it_did(tmp_path):
    """Old checkpoints carry neither `use_audio` in their config nor `audio_features`: they load deaf, with
    the same weights, and give the same answers."""
    config = bc.BCConfig(frame_stack=2, hidden=32, device="cpu")
    torch.manual_seed(0)
    net = bc.build_net(config).eval()
    old_config = {k: v for k, v in asdict(config).items() if k not in ("use_audio", "audio_dim")}
    torch.save({"kind": "bc", "model": net.state_dict(), "config": old_config, "obs_keys": ["pixels"],
                "spec_version": spec.SPEC_VERSION, "epoch": 1, "val": {}, "human_behaviour": {}},
               tmp_path / "old.pt")
    loaded, loaded_config, meta = bc.load(tmp_path / "old.pt")
    assert not loaded_config.use_audio and loaded.audio_encoder is None and "audio_features" not in meta
    pixels = torch.randint(0, 255, (2, 2, *spec.PIXELS_SHAPE), dtype=torch.uint8)
    torch.testing.assert_close(loaded(pixels)[0], net(pixels)[0])
    agent = BCAgent(tmp_path / "old.pt")
    assert agent.audio_features is None
    agent.act({"pixels": np.zeros(spec.PIXELS_SHAPE, np.uint8), "audio": np.zeros(CFG.shape, np.float32)})


REAL_OLD = "/home/dgm/Projects/ZombiesAI/runs/bc_real1/bc.pt"


def test_the_real_pre_audio_checkpoint_still_loads():
    import os

    if not os.path.exists(REAL_OLD):
        pytest.skip("no local pre-audio checkpoint")
    net, config, meta = bc.load(REAL_OLD)
    assert not config.use_audio and net.audio_encoder is None and isinstance(net.mixer, torch.nn.Identity)


def test_a_policy_that_hears_trains_saves_and_plays_with_or_without_sound(heard_clip, silent_clip, tmp_path):
    config = bc.BCConfig(frame_stack=2, hidden=32, audio_dim=16, epochs=1, batch_size=8, max_batches_per_epoch=2,
                         device="cpu", use_audio=True)
    checkpoint = bc.train([heard_clip, silent_clip], config, tmp_path / "run", val_clips=[heard_clip],
                          audio_cache=tmp_path / "cache")
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["audio_features"] == asdict(CFG) and "audio" in saved["obs_keys"]
    assert "nll" in saved["val"] and np.isfinite(saved["val"]["nll"]["mean"])
    agent = BCAgent(checkpoint)
    assert agent.audio_features == CFG
    frame = np.zeros(spec.PIXELS_SHAPE, np.uint8)
    agent.act({"pixels": frame, "audio": clip_features(heard_clip, CFG, None)[5], "audio_mask": 1.0})
    agent.act({"pixels": frame})  # a dead stream: played deaf


# ------------------------------------------------------------------------------------------ live side


class OpenStream(FakeStream):
    """The fake stream, still open after its last chunk: the next read waits, as a quiet pipe would."""

    def open(self):
        super().open()
        self.reads.pop()  # no end-of-stream
        self.total = sum(len(data) for data, _ in self.reads) // 4
        self.closed = threading.Event()

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        self.closed.wait()
        return b"", float("inf")

    def close(self):
        self.closed.set()


def recorded_and_live(tmp_path, *, jitter_ms, seconds=4.0):
    """The same PCM with the same arrival stamps, once through the recorder and once through LiveAudio."""
    audio, _ = recorded_audio(tmp_path, FakeStream(seconds=seconds, origin=100.0, jitter_ms=jitter_ms, seed=3))
    stream = OpenStream(seconds=seconds, origin=100.0, jitter_ms=jitter_ms, seed=3)
    live = LiveAudio(stream, CFG, keep_s=2.0, clock=lambda: 100.0 + seconds)
    live.start()
    deadline = time.monotonic() + 5
    while live._n_samples < stream.total and time.monotonic() < deadline:
        time.sleep(0.001)
    return audio, live


def test_live_features_equal_the_training_features_for_the_same_audio(tmp_path):
    """Train/serve parity to the bit: the ring buffer (which has already dropped the first half of the
    stream) numbers samples as the stream did and maps them with the same clock envelope."""
    audio, live = recorded_and_live(tmp_path, jitter_ms=4.0)
    ring = live.snapshot()
    assert ring.first_sample > 0 and len(ring.samples) < len(audio.samples)
    for t in (102.6, 102.75, 102.8):  # windows whose clock envelope is complete in the ring
        feature, heard = live.observe(t)
        assert heard == 1.0
        np.testing.assert_array_equal(feature, features_at(audio, t, CFG))


def test_live_features_at_the_newest_audio_match_when_arrival_is_prompt(tmp_path):
    audio, live = recorded_and_live(tmp_path, jitter_ms=0.0)
    end = float(live.snapshot().time_of(live.snapshot().chunk_end[-1]))
    feature, _ = live.observe(end)
    np.testing.assert_array_equal(feature, features_at(audio, end, CFG))


def test_live_windows_end_at_the_newest_sample_that_has_arrived(tmp_path):
    _, live = recorded_and_live(tmp_path, jitter_ms=0.0)
    ring = live.snapshot()
    newest = float(ring.time_of(ring.chunk_end[-1]))
    feature, heard = live.observe(newest + 0.05)  # the frame is 50 ms ahead of the audio that has arrived
    np.testing.assert_array_equal(feature, features_at(ring, newest, CFG))  # no silent tail
    assert heard == 1.0 and abs(live.stats()["lag_ms_median"] - 50.0) < 1.0


def test_a_dead_or_stalled_stream_means_no_audio_not_stale_audio(tmp_path):
    _, live = recorded_and_live(tmp_path, jitter_ms=0.0)
    assert live.observe(103.0)[1] == 1.0
    live.close()  # the pipe ends
    assert live.error == "stream ended"
    feature, heard = live.observe(103.0)
    assert heard == 0.0 and (feature == silence(CFG)).all()
    stalled = LiveAudio(FakeStream(), CFG, clock=lambda: 50.0, stale_s=0.25)
    stalled.feed(np.zeros((480, 2), np.int16), 49.0)
    assert stalled.observe(50.0)[1] == 0.0  # nothing for a second
    stalled.feed(np.zeros((480, 2), np.int16), 49.9)
    assert stalled.observe(50.0)[1] == 1.0


def test_observing_never_waits_on_the_stream():
    class Blocking(FakeStream):
        def open(self):
            self.gate = threading.Event()

        def read(self):
            self.gate.wait()  # a pipe with nothing in it
            return b"", 0.0

    stream = Blocking()
    live = LiveAudio(stream, CFG).start()
    live.feed(np.zeros((CFG.window_samples, 2), np.int16), time.monotonic())
    started = time.perf_counter()
    for _ in range(10):
        live.observe(time.monotonic())
    assert (time.perf_counter() - started) / 10 < 0.05
    stream.gate.set()
    live.close()


def test_the_ring_keeps_only_whole_recent_chunks():
    live = LiveAudio(FakeStream(), CFG, keep_s=0.1, clock=lambda: 1.0)
    for i in range(100):
        live.feed(np.full((480, 2), i, np.int16), 0.01 * (i + 1))
    ring = live.snapshot()
    assert len(ring.samples) == ring.chunk_end[-1] - ring.first_sample
    assert int(0.1 * RATE) <= len(ring.samples) < int(0.1 * RATE) + 480
    assert ring.samples[0, 0] == ring.first_sample // 480


def test_the_player_hands_the_policy_what_it_heard_when_the_frame_was_grabbed():
    from test_play import Clock, Focus, Hands, Screen, TimedDispatcher

    from zombiesai.realgame.play import HumanWatch, PlayConfig, play

    class Ears:
        def __init__(self):
            self.times = []

        def observe(self, t):
            self.times.append(t)
            return np.full(CFG.shape, len(self.times), np.float32), 1.0

        def stats(self):
            return {"observed": len(self.times)}

    class Listener:
        def __init__(self):
            self.heard = []

        def act(self, obs):
            self.heard.append((obs["audio"][0, 0, 0], obs["audio_mask"]))
            return np.asarray(spec.NEUTRAL_ACTION)

        def reset(self):
            pass

    clock, ears, agent = Clock(), Ears(), Listener()
    config = PlayConfig(max_seconds=0.5)
    summary = play(Screen(), agent, TimedDispatcher(clock), focus=Focus(), human=HumanWatch(Hands(), config),
                   config=config, clock=clock, say=lambda _: None, hearing=ears)
    assert len(agent.heard) == summary["acted"] == len(ears.times)
    assert [h for h, _ in agent.heard] == list(range(1, len(ears.times) + 1))
    assert np.allclose(np.diff(ears.times), 1 / spec.DECISION_HZ)  # once per tick, at the frame
    assert summary["hearing"] == {"observed": len(ears.times)}
