import json

import numpy as np

from zombiesai import spec
from zombiesai.agents.scripted import ScriptedAgent
from zombiesai.demos.capture import ClipPlayback, ReplayInput, SimSource
from zombiesai.demos.clips import FLAG_BAD_STEP, load_clip
from zombiesai.demos.inputs import InputConfig, read_log, synthesize
from zombiesai.demos.recorder import RecorderConfig, quality_report, record, requantize
from zombiesai.sim.nacht_sim import NachtSim, SimConfig

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
    assert flow["lag"] == 0 and flow["correlation"] > 0.5
    assert set(report["behaviour"]) >= {"fire_duty", "abs_yaw_deg_per_s"}


def test_the_quality_check_catches_labels_that_belong_to_another_recording(tmp_path):
    clip, _ = recorded(tmp_path, steps=300)
    rng = np.random.default_rng(0)
    clip.labels["yaw_deg"] = rng.permutation(clip.labels["yaw_deg"])
    scrambled = quality_report(clip)["yaw_flow"]["correlation"]
    assert not (scrambled > 0.5)  # NaN or low: either way it does not pass for aligned
