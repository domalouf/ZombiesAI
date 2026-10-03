import json

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import clips as clipmod


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


def test_a_clip_serves_the_value_and_auxiliary_targets_its_source_wrote(tmp_path):
    """A recording with rewards can hand the value head Monte-Carlo returns and the auxiliary heads their targets;
    video has none of them, so a trainer asks with extra() and gets None for a column the clip lacks."""
    returns = clipmod.monte_carlo_returns(np.array([0.6, 0.0, -1.0, 0.1]), 0.9)
    writer = clipmod.ClipWriter(tmp_path / "c", source={"kind": "test"}, label_source="agent")
    for k, gained in enumerate((60, 0, 0, 10)):
        writer.add(frame(k), spec.make_action(), extras={
            "mc_return": np.float32(returns[k]),
            "aux_dpoints": np.uint8(np.digitize(gained, clipmod.DPOINTS_EDGES)),
            "aux_damage": np.uint8(k == 2),
        })
    clip = clipmod.load_clip(writer.close())
    np.testing.assert_allclose(clip.extra("mc_return"), returns, rtol=1e-6)
    assert clip.extra("aux_dpoints").tolist() == [2, 0, 0, 1] and clip.extra("aux_damage").tolist() == [0, 0, 1, 0]
    assert clip.extra("not_a_column") is None and clip.label_source == "agent" and clip.usable().all()


def test_iter_clips_finds_every_clip_under_a_root(tmp_path):
    write_clip(tmp_path / "a" / "one")
    write_clip(tmp_path / "b" / "two")
    assert len(list(clipmod.iter_clips(tmp_path))) == 2


def test_extra_label_columns_are_fixed_by_the_first_step_and_can_be_amended_before_close(tmp_path):
    writer = clipmod.ClipWriter(tmp_path / "c", source={"kind": "test"}, label_source="play")
    frame = np.zeros(spec.PIXELS_SHAPE, np.uint8)
    action = np.asarray(spec.NEUTRAL_ACTION)
    writer.add(frame, action, extras={"actor": np.uint8(clipmod.ACTOR_HUMAN), "t_mono": np.float64(1e5 + 0.5)})
    with pytest.raises(ValueError, match="label columns"):
        writer.add(frame, action)  # a step without the column would shift every later value
    with pytest.raises(ValueError, match="overwrite"):
        writer.add(frame, action, extras={"flags": 0})
    writer.add(frame, action, extras={"actor": np.uint8(clipmod.ACTOR_HUMAN), "t_mono": np.float64(1e5 + 0.6)})
    writer.amend(1, actor=clipmod.ACTOR_HUMAN_IDLE)
    with pytest.raises(IndexError):
        writer.amend(2, actor=clipmod.ACTOR_HUMAN_IDLE)
    writer.close()
    clip = clipmod.load_clip(tmp_path / "c")
    np.testing.assert_array_equal(clip.extra("actor"), [clipmod.ACTOR_HUMAN, clipmod.ACTOR_HUMAN_IDLE])
    assert clip.extra("actor").dtype == np.uint8 and clip.extra("t_mono")[1] == 1e5 + 0.6
    np.testing.assert_array_equal(clip.usable(), [True, False])
