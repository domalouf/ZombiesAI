import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import clips as clipmod
from zombiesai.demos.dataset import (
    ClipDataset,
    DataConfig,
    augment_frames,
    class_weights,
    majority_baseline,
    split_clips,
)


def make_clip(tmp_path, name, n=12, seed=0, confidences=None):
    rng = np.random.default_rng(seed)
    writer = clipmod.ClipWriter(tmp_path / name, source={"kind": "test"}, label_source="input_log")
    for i in range(n):
        frame = np.full(spec.PIXELS_SHAPE, i, dtype=np.uint8)
        action = rng.integers(0, spec.ACTION_NVEC)
        writer.add(frame, action, confidence=1.0 if confidences is None else float(confidences[i]))
    writer.close()
    return clipmod.load_clip(tmp_path / name)


def test_frame_stacks_clamp_at_the_clip_start(tmp_path):
    clip = make_clip(tmp_path, "clip")
    data = ClipDataset([clip], before=3)
    first = data.frames(np.array([[0, 0]]))[0]
    assert first.shape == (4, *spec.PIXELS_SHAPE)
    assert [int(f[0, 0, 0]) for f in first] == [0, 0, 0, 0]  # nothing before the clip is invented
    middle = data.frames(np.array([[0, 5]]))[0]
    assert [int(f[0, 0, 0]) for f in middle] == [2, 3, 4, 5]  # oldest first, current last


def test_a_window_with_a_future_drops_the_steps_that_have_none(tmp_path):
    clip = make_clip(tmp_path, "clip", n=12)
    assert len(ClipDataset([clip], before=2, after=2)) == 10
    assert len(ClipDataset([clip], before=2, after=2, clamp_edges=True)) == 12


def test_low_confidence_steps_are_skipped(tmp_path):
    confidences = np.linspace(0.0, 1.0, 12)
    clip = make_clip(tmp_path, "clip", confidences=confidences)
    data = ClipDataset([clip], DataConfig(min_confidence=0.5))
    assert len(data) == int((confidences >= 0.5).sum())


def test_batches_carry_actions_weights_and_prev_actions(tmp_path):
    clip = make_clip(tmp_path, "clip")
    data = ClipDataset([clip], before=3)
    batch = data.batch(data.index[:4])
    assert batch["pixels"].shape == (4, 4, *spec.PIXELS_SHAPE)
    assert batch["action"].shape == (4, len(spec.ACTION_NVEC))
    assert batch["weight"].shape == (4,)
    assert batch["prev_actions"].shape == (4, spec.PREV_ACTION_HISTORY * spec.ACT_ENC_DIM)
    # The first step of a clip has no history, so it gets the neutral action rather than a neighbour's.
    np.testing.assert_array_equal(
        data.prev_actions(0, 0), spec.encode_prev_actions([spec.NEUTRAL_ACTION] * spec.PREV_ACTION_HISTORY)
    )
    np.testing.assert_array_equal(
        data.prev_actions(0, 2), spec.encode_prev_actions([tuple(clip.actions[1]), tuple(clip.actions[0])])
    )


def test_augmentation_shifts_a_whole_stack_together(tmp_path):
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 255, size=(2, 4, *spec.PIXELS_SHAPE), dtype=np.uint8)
    out = augment_frames(stack, DataConfig(shift_px=4, brightness=0.0), np.random.default_rng(3))
    assert out.shape == stack.shape and out.dtype == np.uint8
    # Whatever shift a sample got, every frame in it got the same one: the differences between frames of the
    # stack must survive unchanged, or the augmentation fabricates camera motion.
    for sample_in, sample_out in zip(stack, out):
        deltas_in = np.diff(sample_in.astype(int), axis=0)
        deltas_out = np.diff(sample_out.astype(int), axis=0)
        assert np.abs(deltas_out).sum() == pytest.approx(np.abs(deltas_in).sum(), rel=0.35)


def test_augmentation_off_is_the_identity(tmp_path):
    stack = np.random.default_rng(0).integers(0, 255, size=(2, 4, *spec.PIXELS_SHAPE), dtype=np.uint8)
    out = augment_frames(stack, DataConfig(shift_px=0, brightness=0.0), np.random.default_rng(0))
    np.testing.assert_array_equal(out, stack)


def test_validation_is_split_by_clip_not_by_step(tmp_path):
    clips = [make_clip(tmp_path, f"c{i}", seed=i) for i in range(10)]
    train, val = split_clips(clips, 0.2, seed=0)
    assert len(train) == 8 and len(val) == 2
    assert not {c.path for c in train} & {c.path for c in val}
    assert split_clips(clips[:1], 0.2)[1] == []


def test_class_weights_lift_the_rare_button_classes():
    actions = np.zeros((1000, len(spec.ACTION_NVEC)), dtype=np.int64)
    actions[:5, spec.BUTTON] = spec.BUTTONS.index("reload")
    weights = class_weights(actions)
    button = weights[spec.BUTTON]
    assert button[spec.BUTTONS.index("reload")] > button[spec.BUTTONS.index("none")]
    assert button[spec.BUTTONS.index("melee")] == 0.0  # never seen, never weighted
    assert majority_baseline(actions)[spec.BUTTON] == pytest.approx(0.995)
