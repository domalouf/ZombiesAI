import numpy as np
import pytest
import torch

from zombiesai import spec
from zombiesai.rl.encoders import PixelActorCritic, PixelEncoder, stack_to_nchw, vector_dim, vector_from_obs


def frames(batch=2, stack=4):
    return torch.randint(0, 256, (batch, stack, *spec.PIXELS_SHAPE), dtype=torch.uint8)


def test_stacking_folds_frames_into_channels():
    x = frames(2, 4)
    nchw = stack_to_nchw(x)
    assert nchw.shape == (2, 12, *spec.PIXELS_SHAPE[:2])
    # Channel 3 of the output is the red plane of the second frame, not a reshuffle of the first.
    torch.testing.assert_close(nchw[:, 3], x[:, 1, :, :, 0])
    assert stack_to_nchw(x[:, 0]).shape == (2, 3, *spec.PIXELS_SHAPE[:2])


def test_the_encoder_takes_uint8_and_normalizes_it():
    encoder = PixelEncoder(12, out_dim=64)
    out = encoder(frames())
    assert out.shape == (2, 64) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_the_policy_produces_one_categorical_per_head():
    net = PixelActorCritic(frame_stack=4, hidden=64)
    logits, value, aux = net(frames())
    assert logits.shape == (2, sum(spec.ACTION_NVEC)) and value.shape == (2,) and aux == {}
    action = net.dist(frames()).sample()
    assert action.shape == (2, len(spec.ACTION_NVEC))
    assert (action < torch.tensor(spec.ACTION_NVEC)).all()


def test_auxiliary_heads_are_optional_and_named():
    net = PixelActorCritic(frame_stack=2, hidden=32, aux_heads={"aux_damage": 2})
    _, _, aux = net(frames(stack=2))
    assert set(aux) == {"aux_damage"} and aux["aux_damage"].shape == (2, 2)


def test_vector_observations_are_mixed_in_and_required_when_configured():
    keys = ("pixels", "prev_actions")
    net = PixelActorCritic(frame_stack=4, vector_dim=vector_dim(keys), hidden=32)
    assert vector_dim(keys) == spec.PREV_ACTION_HISTORY * spec.ACT_ENC_DIM
    vector = torch.zeros(2, vector_dim(keys))
    assert net(frames(), vector)[0].shape == (2, sum(spec.ACTION_NVEC))
    with pytest.raises(ValueError):
        net(frames())


def test_vector_from_obs_follows_the_configured_key_order():
    obs = {"hud": np.ones(spec.HUD_DIM, np.float32), "prev_actions": np.zeros(64, np.float32)}
    row = vector_from_obs(obs, ("pixels", "hud", "prev_actions"))
    assert row.shape == (spec.HUD_DIM + 64,) and row[: spec.HUD_DIM].all() and not row[spec.HUD_DIM :].any()
    assert vector_from_obs(obs, ("pixels",)) is None


def test_gradients_reach_the_first_convolution():
    net = PixelActorCritic(frame_stack=4, hidden=32)
    logits, value, _ = net(frames())
    (logits.sum() + value.sum()).backward()
    first = next(p for p in net.encoder.conv.parameters())
    assert first.grad is not None and torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0
