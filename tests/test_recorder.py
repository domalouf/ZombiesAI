import json

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.agents.scripted import ScriptedAgent
from zombiesai.demos.capture import ClipPlayback, ReplayInput, SimSource
from zombiesai.demos.clips import (
    FLAG_BAD_STEP,
    FLAG_CLIP_START,
    FLAG_NOT_PLAYING,
    load_clip,
)
from zombiesai.demos.inputs import InputConfig, read_log, synthesize
from zombiesai.demos.recorder import RecorderConfig, quality_report, record, requantize, wait_to_start
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.sim.render import FOV_DEG

CONFIG = InputConfig(counts_per_degree=10.0)


def recorded(tmp_path, steps=120, seed=0):
    source = SimSource(ScriptedAgent(), seed=seed, hardness=0.4, max_steps=steps)
    config = RecorderConfig(max_steps=steps, realtime=False, input=CONFIG, notes="test")
    path = record(source, source, tmp_path / "demo", config, stop=lambda: source.done, progress_every=0)
    return load_clip(path), source


def true_actions(steps, seed=0, hardness=0.4):
    env = NachtSim(SimConfig(hardness=hardness, max_steps=steps, obs_profile="render"))
    obs, _ = env.reset(seed=seed)
    agent = ScriptedAgent()
    agent.reset()
    out = []
    for _ in range(steps):
        action = np.asarray(agent.act({**obs, "state": env.state()}))
        out.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return np.array(out, dtype=np.uint8)


def test_a_recording_labels_every_frame_with_what_the_player_actually_did(tmp_path):
    clip, _ = recorded(tmp_path)
    expected = true_actions(120)[: clip.n_steps]
    np.testing.assert_array_equal(clip.actions, expected)
    assert clip.label_source == "input_log"
    assert not (clip.flags & FLAG_BAD_STEP).any()


def test_the_frame_is_the_one_the_player_was_looking_at_when_they_acted(tmp_path):
    """Off-by-one in the pairing is the bug that looks like 'the model can't aim' for a week."""
    clip, _ = recorded(tmp_path)
    env = NachtSim(SimConfig(hardness=0.4, max_steps=120, obs_profile="render"))
    obs, _ = env.reset(seed=0)
    agent = ScriptedAgent()
    agent.reset()
    for step in range(min(clip.n_steps, 20)):
        np.testing.assert_array_equal(clip.frames[step], obs["pixels"])
        action = np.asarray(agent.act({**obs, "state": env.state()}))
        np.testing.assert_array_equal(clip.actions[step], action.astype(np.uint8))
        obs, *_ = env.step(action)


def test_the_raw_input_log_is_kept_and_reproduces_the_labels(tmp_path):
    clip, _ = recorded(tmp_path)
    events = read_log(clip.path / "inputs.jsonl")
    assert events and {e["type"] for e in events} <= {"key", "button", "mouse"}
    relabelled = requantize(clip.path, CONFIG)
    np.testing.assert_array_equal(relabelled[: clip.n_steps], clip.actions)


def test_requantizing_at_a_different_sensitivity_rescales_the_look(tmp_path):
    clip, _ = recorded(tmp_path)
    turned = clip.actions[:, spec.YAW] != spec.YAW_BINS_DEG.index(0.0)
    halved = requantize(clip.path, InputConfig(counts_per_degree=CONFIG.counts_per_degree * 4))
    # Same counts read as a quarter of the rotation: turns shrink towards the middle bin, never grow.
    assert turned.any()
    before = np.abs(np.array(spec.YAW_BINS_DEG)[clip.actions[:, spec.YAW]])
    after = np.abs(np.array(spec.YAW_BINS_DEG)[halved[: clip.n_steps, spec.YAW]])
    assert (after <= before + 1e-6).all() and after.sum() < before.sum()


def test_the_manifest_records_where_the_frames_came_from(tmp_path):
    clip, _ = recorded(tmp_path)
    manifest = json.loads((clip.path / "clip.json").read_text())
    assert manifest["source"]["kind"] == "sim"
    assert manifest["config"]["recorder"]["input"]["counts_per_degree"] == 10.0
    assert manifest["summary"]["notes"] == "test" and "t0_mono" in manifest["summary"]


def test_the_recorder_runs_the_same_on_replayed_frames_and_a_replayed_log(tmp_path):
    """The plan's FakeCapture/FakeInput path: the whole recording loop, no game and no mouse."""
    clip, _ = recorded(tmp_path)
    dt = 1.0 / spec.DECISION_HZ
    actions = [spec.make_action(forward=1, yaw=6.0), spec.make_action(fire=1), spec.make_action(button="reload")]
    events = [e for k, a in enumerate(actions) for e in synthesize(a, k * dt, dt, CONFIG)]
    path = record(
        ClipPlayback(clip.path),
        ReplayInput(events),
        tmp_path / "replayed",
        RecorderConfig(max_steps=len(actions), realtime=False, input=CONFIG),
        progress_every=0,
    )
    replayed = load_clip(path)
    np.testing.assert_array_equal(replayed.actions, np.array(actions, dtype=np.uint8)[: replayed.n_steps])
    np.testing.assert_array_equal(replayed.frames[0], clip.frames[0])


def test_a_finished_recording_reports_whether_it_is_worth_keeping(tmp_path):
    clip, _ = recorded(tmp_path, steps=300)
    report = quality_report(clip)
    assert report["steps"] == clip.n_steps and report["overrun_rate"] == 0.0
    assert 0.0 <= report["mean_confidence"] <= 1.0
    assert report["clamped_looks"] == 0.0
    flow = report["yaw_flow"]
    # The scripted agent turns, the frames move with it, and the response is immediate in the sim's own
    # timing here -- so the best-fitting lag is the one the recorder's pairing implies.
    assert flow["verdict"] == "ok" and flow["lag"] == 0 and flow["rank_correlation"] > 0.5
    # ...and the pixels per degree give back the renderer's lens, to within the integer-pixel search.
    assert flow["fov_deg"] == pytest.approx(FOV_DEG, abs=6.0)
    assert set(report["behaviour"]) >= {"fire_duty", "abs_yaw_deg_per_s"}


def test_the_quality_check_expects_the_sims_own_input_latency(tmp_path):
    """A sim episode can draw a latency of a decision or two; that is the lag its recording should peak at,
    and the manifest says so rather than the check calling it misaligned."""
    source = SimSource(ScriptedAgent(), seed=2, hardness=0.4, max_steps=300)
    assert source.describe()["latency_steps"] == 1
    config = RecorderConfig(max_steps=300, realtime=False, input=CONFIG)
    clip = load_clip(record(source, source, tmp_path / "demo", config, stop=lambda: source.done, progress_every=0))
    flow = quality_report(clip)["yaw_flow"]
    assert flow["expected_lag"] == 1 and flow["lag"] == 1 and flow["verdict"] == "ok"


def test_the_quality_check_catches_labels_that_belong_to_another_recording(tmp_path):
    clip, _ = recorded(tmp_path, steps=300)
    rng = np.random.default_rng(0)
    clip.labels["yaw_deg"] = rng.permutation(clip.labels["yaw_deg"])
    assert quality_report(clip)["yaw_flow"]["verdict"] == "no_signal"


def test_the_quality_check_catches_a_log_one_decision_out_of_step(tmp_path):
    clip, _ = recorded(tmp_path, steps=300)
    yaw = np.asarray(clip.labels["yaw_deg"])
    for steps in (-1, 1):
        clip.labels["yaw_deg"] = np.roll(yaw, steps)
        assert quality_report(clip)["yaw_flow"]["verdict"] == "misaligned"


def test_the_quality_check_catches_a_counts_per_degree_off_by_two(tmp_path):
    clip, _ = recorded(tmp_path, steps=300)
    yaw = np.asarray(clip.labels["yaw_deg"])
    for scale in (0.5, 2.0):
        clip.labels["yaw_deg"] = yaw * scale
        flow = quality_report(clip)["yaw_flow"]
        assert flow["verdict"] == "wrong_scale", flow["reasons"]


class FakeClock:
    """Time that passes only when the code under test sleeps, so the wait loop is tested without waiting."""

    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def probe_after(polls: int, then=None):
    """A game window that becomes capturable on the `polls`-th question -- and, if `then` is given, whose
    answers after that come from `then` instead of a steady yes."""
    answers = iter([False] * polls + list(then or []))
    return lambda: next(answers, True)


def waited(probe, **kwargs):
    clock, said, notes, discards = FakeClock(), [], [], []
    kwargs = {"timeout": 60.0, "countdown": 3.0, "poll": 0.5, **kwargs}
    ready = wait_to_start(
        probe, discard=lambda: discards.append(clock.now), say=said.append, notify=notes.append,
        clock=clock, sleep=clock.sleep, **kwargs,
    )
    return ready, clock, said, notes, discards


def test_recording_waits_for_the_game_window_and_then_counts_down():
    ready, clock, said, notes, _ = waited(probe_after(4))
    assert ready
    assert any("waiting for the game window" in line for line in said)
    assert [line.strip() for line in said if line.strip()[0].isdigit()] == ["3...\a", "2...\a", "1...\a"]
    assert notes == ["recording in 3s"]
    assert clock.now == 4 * 0.5 + 3.0  # four polls of waiting, then the countdown, and not a moment more


def test_a_window_already_on_screen_goes_straight_to_the_countdown():
    ready, clock, said, _, _ = waited(lambda: True, countdown=2.0)
    assert ready and clock.now == 2.0
    assert not any("waiting" in line for line in said)


def test_waiting_gives_up_after_the_timeout():
    ready, clock, said, notes, _ = waited(lambda: False, timeout=10.0)
    assert not ready
    assert 10.0 <= clock.now <= 10.5
    assert notes == [], "nothing should announce a recording that is not going to happen"


def test_a_window_that_leaves_during_the_countdown_sends_it_back_to_waiting():
    # Capturable at the first question, gone when the countdown ends, back two polls later.
    ready, clock, said, notes, _ = waited(probe_after(0, then=[True, False, False, False]))
    assert ready
    assert any("left the screen" in line for line in said)
    assert notes == ["recording in 3s", "recording in 3s"]


def test_a_zero_countdown_starts_the_moment_the_window_appears():
    ready, clock, said, notes, _ = waited(probe_after(2), countdown=0.0)
    assert ready and clock.now == 2 * 0.5 and notes == []


def test_the_input_backlog_is_discarded_right_up_to_the_start():
    ready, clock, _, _, discards = waited(probe_after(3))
    assert ready and discards and discards[-1] == clock.now


class BackloggedInput:
    """Like EvdevInput: `drain()` returns everything since the last call, whatever window it is asked for."""

    def __init__(self, events):
        self.events = list(events)

    def drain(self, start: float = 0.0, end: float = 0.0) -> list[dict]:
        out, self.events = self.events, []
        return out

    def close(self) -> None:
        pass


class BlankScreen:
    def read(self):
        return np.zeros(spec.PIXELS_SHAPE, dtype=np.uint8)

    def close(self) -> None:
        pass


def first_yaw(tmp_path, name, *, discard: bool) -> float:
    # A minute of reaching for the mouse and alt-tabbing, stamped long before the recording starts.
    backlog = [{"t": -60.0 + i * 0.01, "type": "mouse", "dx": 40, "dy": 0} for i in range(500)]
    inputs = BackloggedInput(backlog)
    if discard:
        assert wait_to_start(lambda: True, timeout=1.0, countdown=0.0, discard=inputs.drain, say=lambda _: None)
    config = RecorderConfig(max_steps=2, realtime=False, input=CONFIG)
    path = record(BlankScreen(), inputs, tmp_path / name, config, progress_every=0)
    return float(load_clip(path).labels["yaw_deg"][0])


def test_input_from_before_the_recording_does_not_become_the_first_label(tmp_path):
    """InputFolder clamps early events into the first decision, so without the discard the whole wait
    arrives as one impossible flick -- which is what this checks the discard prevents."""
    assert abs(first_yaw(tmp_path, "backlogged", discard=False)) > 100
    assert first_yaw(tmp_path, "discarded", discard=True) == 0.0


def marked_recording(tmp_path, presses=(3.3, 6.4), n=12, extra=()):
    """A replayed recording with the mark key tapped at the given times (in decisions), plus any extra events."""
    clip, _ = recorded(tmp_path)
    dt = 1.0 / spec.DECISION_HZ
    actions = [spec.make_action(forward=1, yaw=6.0 if k % 2 else 0.0) for k in range(n)]
    events = [e for k, a in enumerate(actions) for e in synthesize(a, k * dt, dt, CONFIG)]
    for at in presses:
        events += [
            {"t": at * dt, "type": "key", "code": "f8", "down": True},
            {"t": (at + 0.1) * dt, "type": "key", "code": "f8", "down": False},
        ]
    events += [{**e, "t": e["t"] * dt} for e in extra]
    path = record(
        ClipPlayback(clip.path),
        ReplayInput(events, origin=0.0),
        tmp_path / "marked",
        RecorderConfig(max_steps=n, realtime=False, input=CONFIG),
        progress_every=0,
    )
    return load_clip(path), np.array(actions, dtype=np.uint8)


def test_the_mark_key_flags_every_step_whose_window_was_not_play(tmp_path, capsys):
    """Pressed during step 3's window, pressed again during step 6's: 3..6 are out, play resumes at 7."""
    clip, _ = marked_recording(tmp_path)
    marked = np.flatnonzero(clip.flags & FLAG_NOT_PLAYING)
    np.testing.assert_array_equal(marked, [3, 4, 5, 6])
    np.testing.assert_array_equal(np.flatnonzero(clip.flags & FLAG_CLIP_START), [0, 7])
    summary = clip.manifest["summary"]
    assert summary["not_playing_steps"] == 4 and summary["mark_key"] == "f8"
    out = capsys.readouterr().out
    assert "NOT PLAYING" in out and "playing again" in out


def test_a_held_mark_key_repeating_is_one_toggle_not_many(tmp_path):
    """Windows Raw Input repeats the make code while a key is held; that must not flicker the marking."""
    held = [{"t": 3.3 + 0.05 * i, "type": "key", "code": "f8", "down": True} for i in range(12)]
    held.append({"t": 4.0, "type": "key", "code": "f8", "down": False})
    clip, _ = marked_recording(tmp_path, presses=(), extra=held)
    np.testing.assert_array_equal(np.flatnonzero(clip.flags & FLAG_NOT_PLAYING), np.arange(3, 12))


def test_the_mark_key_is_never_an_action_label(tmp_path):
    clip, actions = marked_recording(tmp_path)
    np.testing.assert_array_equal(clip.actions, actions[: clip.n_steps])
    assert any(e.get("code") == "f8" for e in read_log(clip.path / "inputs.jsonl"))  # kept in the raw log


def test_usable_leaves_out_the_steps_marked_not_playing(tmp_path):
    clip, _ = marked_recording(tmp_path)
    np.testing.assert_array_equal(clip.usable(), (clip.flags & FLAG_NOT_PLAYING) == 0)
    assert clip.usable().sum() == clip.n_steps - 4


def test_requantizing_rederives_the_marking_from_the_log(tmp_path):
    clip, _ = marked_recording(tmp_path)
    before = clip.flags.copy()
    requantize(clip.path, CONFIG)
    np.testing.assert_array_equal(load_clip(clip.path).flags, before)
    # Recorded with the wrong key in mind: re-applying under another key removes the marks entirely...
    requantize(clip.path, InputConfig(counts_per_degree=10.0, mark_key="f7"))
    again = load_clip(clip.path)
    assert not (again.flags & FLAG_NOT_PLAYING).any()
    np.testing.assert_array_equal(np.flatnonzero(again.flags & FLAG_CLIP_START), [0])
    # ...and going back puts them where the recorder had them.
    requantize(clip.path, CONFIG)
    np.testing.assert_array_equal(load_clip(clip.path).flags, before)


def test_frame_stacks_do_not_reach_back_across_a_marked_stretch(tmp_path):
    from zombiesai.demos.dataset import ClipDataset

    clip, _ = marked_recording(tmp_path)
    data = ClipDataset([clip], before=3)
    assert 7 in data.index[:, 1] and not set(data.index[:, 1]) & {3, 4, 5, 6}
    resumed = data.frames(np.array([[0, 7], [0, 8]]))
    for tap in resumed[0]:
        np.testing.assert_array_equal(tap, clip.frames[7])
    np.testing.assert_array_equal(resumed[1][:3], clip.frames[[7, 7, 7]])
    np.testing.assert_array_equal(
        data.prev_actions(0, 7), spec.encode_prev_actions([spec.NEUTRAL_ACTION] * spec.PREV_ACTION_HISTORY)
    )


def test_the_quality_check_reports_marked_time_and_describes_only_play(tmp_path):
    clip, _ = marked_recording(tmp_path)
    report = quality_report(clip)
    assert report["not_playing_steps"] == 4
    assert report["behaviour"]["steps"] == clip.n_steps - 4


class InterruptedAfter:
    """A screen that is closed on the recorder (Ctrl-C, or SIGTERM/SIGHUP turned into one) after n frames."""

    def __init__(self, n):
        self.n = n

    def read(self):
        self.n -= 1
        if self.n < 0:
            raise KeyboardInterrupt
        return np.zeros(spec.PIXELS_SHAPE, dtype=np.uint8)

    def close(self):
        pass


def test_an_interrupted_recording_still_closes_with_its_labels(tmp_path):
    config = RecorderConfig(max_steps=50, realtime=False, input=CONFIG)
    try:
        record(InterruptedAfter(6), ReplayInput([]), tmp_path / "demo", config, progress_every=0)
    except KeyboardInterrupt:
        pass
    clip = load_clip(tmp_path / "demo")
    assert clip.manifest["status"] == "closed" and clip.labelled and clip.n_steps == 5


def test_a_recording_that_never_closed_is_requantized_against_its_own_start(tmp_path):
    """Killed outright, a clip has no summary -- but the t0 written at the start keeps the labels on the
    recorder's decision boundaries instead of the first event's timestamp."""
    import json

    clip, _ = recorded(tmp_path)
    manifest = json.loads((clip.path / "clip.json").read_text())
    assert manifest["t0_mono"] == manifest["summary"]["t0_mono"]
    before = clip.actions.copy()
    manifest["summary"] = {}
    (clip.path / "clip.json").write_text(json.dumps(manifest))
    requantize(clip.path, CONFIG)
    np.testing.assert_array_equal(load_clip(clip.path).actions, before)


def test_every_toggle_of_the_mark_key_is_announced_to_the_player(tmp_path):
    announced = []
    clip, _ = recorded(tmp_path)
    dt = 1.0 / spec.DECISION_HZ
    events = [
        {"t": t * dt, "type": "key", "code": "f8", "down": down}
        for t, down in ((2.3, True), (2.4, False), (5.3, True), (5.4, False))
    ]
    record(
        ClipPlayback(clip.path), ReplayInput(events, origin=0.0), tmp_path / "announced",
        RecorderConfig(max_steps=8, realtime=False, input=CONFIG), progress_every=0, on_mark=announced.append,
    )
    assert announced == [False, True]
