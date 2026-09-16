import json

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.agents.random_agent import RandomAgent
from zombiesai.demos import clips as clipmod
from zombiesai.rollout import run_episode
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.store.episode_store import EpisodeWriter, episode_dir


def frame(value: int) -> np.ndarray:
    return np.full(spec.PIXELS_SHAPE, value, dtype=np.uint8)


def write_clip(path, n=10, labelled=True):
    writer = clipmod.ClipWriter(path, source={"kind": "test"}, label_source="input_log" if labelled else "none")
    space = spec.factored_action_space()
    space.seed(0)
    actions = [space.sample() for _ in range(n)]
    for i, action in enumerate(actions):
        writer.add(frame(i), action if labelled else None, confidence=0.5 + i / (2 * n), yaw_deg=float(i))
    writer.close(summary={"seconds": n / spec.DECISION_HZ})
    return np.array(actions, dtype=np.uint8)


def test_round_trip(tmp_path):
    actions = write_clip(tmp_path / "clip")
    clip = clipmod.load_clip(tmp_path / "clip")
    assert clip.n_steps == 10 and clip.labelled and clip.label_source == "input_log"
    np.testing.assert_array_equal(clip.actions, actions)
    np.testing.assert_array_equal(clip.frames[3], frame(3))
    assert clip.manifest["spec_version"] == spec.SPEC_VERSION
    assert clip.flags[0] & clipmod.FLAG_CLIP_START


def test_unlabelled_clips_load_and_report_no_labels(tmp_path):
    write_clip(tmp_path / "video", labelled=False)
    clip = clipmod.load_clip(tmp_path / "video")
    assert not clip.labelled and clip.actions is None
    assert clip.usable().all()  # nothing is known to be bad yet
    with pytest.raises(FileNotFoundError):
        clipmod.load_clip(tmp_path / "video", require_labels=True)


def test_a_clip_opened_unlabelled_refuses_actions(tmp_path):
    writer = clipmod.ClipWriter(tmp_path / "c", source={"kind": "test"}, label_source="none")
    with pytest.raises(ValueError):
        writer.add(frame(0), spec.make_action())


def test_a_labelled_clip_refuses_a_step_without_an_action(tmp_path):
    """One unlabelled step would shift every later label by one, and nothing downstream could tell."""
    writer = clipmod.ClipWriter(tmp_path / "c", source={"kind": "test"}, label_source="input_log")
    writer.add(frame(0), spec.make_action())
    with pytest.raises(ValueError):
        writer.add(frame(1))
    writer.close()
    assert clipmod.load_clip(tmp_path / "c").n_steps == 1


def test_a_misshaped_frame_is_refused(tmp_path):
    writer = clipmod.ClipWriter(tmp_path / "c", source={"kind": "test"}, label_source="none")
    with pytest.raises(ValueError):
        writer.add(np.zeros((84, 84, 3), np.uint8))


def test_frames_are_refused_under_a_different_frame_contract(tmp_path):
    write_clip(tmp_path / "clip")
    path = tmp_path / "clip" / "clip.json"
    manifest = json.loads(path.read_text())
    manifest["frame_contract"]["pixels_shape"] = [84, 84, 1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(spec.SpecMismatchError):
        clipmod.load_clip(tmp_path / "clip")


def test_labels_are_refused_under_a_different_spec(tmp_path):
    write_clip(tmp_path / "clip")
    path = tmp_path / "clip" / "clip.json"
    manifest = json.loads(path.read_text())
    manifest["spec_version"] = "0000000000000000"
    path.write_text(json.dumps(manifest))
    with pytest.raises(spec.SpecMismatchError):
        clipmod.load_clip(tmp_path / "clip")


def test_attach_labels_overwrites_and_records_where_they_came_from(tmp_path):
    write_clip(tmp_path / "video", labelled=False)
    clip = clipmod.load_clip(tmp_path / "video")
    actions = np.zeros((clip.n_steps, len(spec.ACTION_NVEC)), dtype=np.uint8)
    confidence = np.linspace(0.0, 1.0, clip.n_steps, dtype=np.float32)
    clipmod.attach_labels(clip.path, actions, confidence, label_source="idm", detail={"checkpoint": "x"})
    relabelled = clipmod.load_clip(clip.path)
    assert relabelled.labelled and relabelled.label_source == "idm"
    assert relabelled.manifest["labels"]["checkpoint"] == "x"
    assert relabelled.usable(min_confidence=0.5).sum() == (confidence >= 0.5).sum()


def test_out_of_range_pseudo_labels_are_refused(tmp_path):
    write_clip(tmp_path / "video", labelled=False)
    bad = np.zeros((10, len(spec.ACTION_NVEC)), dtype=np.uint8)
    bad[0, spec.YAW] = len(spec.YAW_BINS_DEG)
    with pytest.raises(ValueError):
        clipmod.attach_labels(tmp_path / "video", bad, np.ones(10), label_source="idm")


def test_monte_carlo_returns_discount_backwards():
    rewards = np.array([1.0, 0.0, 2.0])
    np.testing.assert_allclose(clipmod.monte_carlo_returns(rewards, 0.5), [1.5, 1.0, 2.0])


def test_an_episode_reads_as_a_clip_with_value_and_auxiliary_targets(tmp_path):
    env = NachtSim(SimConfig(max_steps=120, obs_profile="render"))
    writer = EpisodeWriter(episode_dir(tmp_path, 0))
    run_episode(env, RandomAgent(0), seed=1, writer=writer)
    clip = clipmod.clip_from_episode(episode_dir(tmp_path, 0))
    assert clip.labelled and clip.n_steps > 0
    assert clip.frames.shape[1:] == spec.PIXELS_SHAPE
    assert clip.extra("mc_return") is not None and clip.extra("aux_damage") is not None
    assert set(np.unique(clip.extra("aux_dpoints"))) <= set(range(len(clipmod.DPOINTS_EDGES) + 1))
    assert clip.label_source == "agent"


def test_iter_clips_finds_every_clip_under_a_root(tmp_path):
    write_clip(tmp_path / "a" / "one")
    write_clip(tmp_path / "b" / "two")
    assert len(list(clipmod.iter_clips(tmp_path))) == 2
