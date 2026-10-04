"""The policy's look at the HUD corner (demos/hud_crops.py `hud_view`): what it keeps of the screen, the network
branch that reads it, and every path it travels -- env, synthetic env, segments over the wire, BC's dataset --
plus turning an existing policy into one that has it without changing what it does."""

from pathlib import Path

import numpy as np
import pytest
import torch

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.demos.clips import ClipWriter, load_clip
from zombiesai.demos.dataset import ClipDataset
from zombiesai.demos.frames import area_resize
from zombiesai.demos.hearing import AudioFeatureConfig
from zombiesai.demos.hud_crops import HUD_VIEW_SHAPE, POINTS_AMMO_SHAPE, hud_view
from zombiesai.rl.encoders import PixelActorCritic
from zombiesai.rl.fleet import decode_segment, encode_segment
from zombiesai.rl.segments import Segment

FIXTURES = Path(__file__).parent / "fixtures" / "hud"


@pytest.fixture(scope="module")
def live():
    """Real points_ammo crops of the agents' games (test_hud_parse.py LIVE says what each shows)."""
    with np.load(FIXTURES / "live_crops.npz") as z:
        return z["points_ammo"]


def loaded_ticks(view: np.ndarray) -> int:
    """Magazine ticks drawn bright in a view: the row they sit on, against the rows above it."""
    row = view[56:58, 10:40, 0].astype(int).mean(0) - view[52:54, 10:40, 0].astype(int).mean(0)
    return int((row > 20).sum())


def test_the_view_keeps_the_corner_legible_and_every_magazine_tick(live):
    full, empty, half = hud_view(live[0]), hud_view(live[1]), hud_view(live[2])  # a Colt with 8, 0 and 4 loaded
    assert full.shape == HUD_VIEW_SHAPE and full.dtype == np.uint8
    assert (loaded_ticks(full), loaded_ticks(empty), loaded_ticks(half)) == (8, 0, 4)
    assert not hud_view(None).any()


def test_a_crop_from_another_resolution_is_resized_to_the_reference_first(live):
    bigger = area_resize(live[0], POINTS_AMMO_SHAPE[0] * 2, POINTS_AMMO_SHAPE[1] * 2)  # a 4K capture's crop
    assert np.abs(hud_view(bigger).astype(int) - hud_view(live[0]).astype(int)).max() <= 2


def tiny(**kw) -> PixelActorCritic:
    torch.manual_seed(0)
    return PixelActorCritic(frame_stack=2, hidden=32, **kw)


def batch(n=3, audio=None):
    rng = np.random.default_rng(1)
    pixels = torch.from_numpy(rng.integers(0, 256, (n, 2, *spec.PIXELS_SHAPE), dtype=np.uint8))
    views = torch.from_numpy(rng.integers(0, 256, (n, *HUD_VIEW_SHAPE), dtype=np.uint8))
    heard = None if audio is None else torch.from_numpy(rng.normal(0, 1, (n, *audio)).astype(np.float32))
    return pixels, views, heard


def test_the_hud_branch_feeds_the_mixer_and_a_network_without_it_ignores_it():
    looks = tiny(hud_shape=HUD_VIEW_SHAPE, hud_dim=16)
    pixels, views, _ = batch()
    out = looks(pixels, hud=views)[0]
    assert out.shape == (3, sum(spec.ACTION_NVEC))
    assert not torch.allclose(out, looks(pixels, hud=torch.zeros_like(views))[0])  # what it sees there matters
    with pytest.raises(ValueError):
        looks(pixels)
    blind = tiny()
    assert torch.equal(blind(pixels)[0], blind(pixels, hud=views)[0])  # the frozen prior: same batch, no branch


@pytest.mark.parametrize("audio", [False, True])
def test_adding_the_view_to_a_trained_policy_changes_nothing_it_does(audio):
    features = AudioFeatureConfig() if audio else None
    config = bc.BCConfig(frame_offsets=(0, 2), hidden=32, audio_dim=16, hud_dim=16, use_audio=audio)
    torch.manual_seed(3)
    net = bc.build_net(config, features).eval()
    with torch.no_grad():  # weights as training leaves them, not as initialised
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    new, new_config = bc.with_hud_view(net, config, features)
    assert new_config.use_hud_view and "hud_view" in new_config.obs_keys and new.hud_encoder is not None
    pixels, views, heard = batch(audio=features.shape if audio else None)
    mask = torch.ones(3) if audio else None
    new.eval()
    with torch.no_grad():
        before = net(pixels, None, heard, mask)
        after = new(pixels, None, heard, mask, views)
        other = new(pixels, None, heard, mask, torch.zeros_like(views))
    for b, a, o in zip(before[:2], after[:2], other[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
        torch.testing.assert_close(o, b, rtol=0, atol=1e-5)  # whatever the corner shows, until it is trained
    with pytest.raises(ValueError):
        bc.with_hud_view(new, new_config, features)


def segment(n=5, views=True) -> Segment:
    rng = np.random.default_rng(0)
    return Segment(actor=1, version=3, context=2, frames=rng.integers(0, 256, (2 + n + 1, *spec.PIXELS_SHAPE), np.uint8),
                   actions=np.zeros((n, len(spec.ACTION_NVEC)), np.int64), logp=np.zeros(n, np.float32),
                   rewards=np.zeros(n, np.float32), bad=np.zeros(n, bool), terminated=False,
                   hud_view=rng.integers(0, 256, (n + 1, *HUD_VIEW_SHAPE), np.uint8) if views else None)


def test_segments_carry_the_views_and_the_learner_refuses_ones_that_do_not_fit():
    seg = segment()
    got = decode_segment(encode_segment(seg), 1, context=2, audio_shape=None, hud_view=True)
    assert np.array_equal(got.hud_view, seg.hud_view)
    with pytest.raises(ValueError, match="no HUD views"):
        decode_segment(encode_segment(segment(views=False)), 1, context=2, audio_shape=None, hud_view=True)
    with pytest.raises(ValueError, match="does not look"):
        decode_segment(encode_segment(seg), 1, context=2, audio_shape=None, hud_view=False)
    short = segment()
    short.hud_view = short.hud_view[:-1]
    with pytest.raises(ValueError, match="HUD views"):
        decode_segment(encode_segment(short), 1, context=2, audio_shape=None, hud_view=True)


def test_the_synthetic_world_draws_its_magazine_where_the_real_one_is():
    from zombiesai.synthetic import SyntheticActorEnv

    env = SyntheticActorEnv(0)
    obs, _ = env.reset()
    assert obs["hud_view"].shape == HUD_VIEW_SHAPE and loaded_ticks(obs["hud_view"]) == env.world.config.mag_size
    fire = spec.make_action(fire=1)
    for _ in range(40):
        obs, *_ = env.step(fire)
        if env.world.mag < env.world.config.mag_size:
            break
    assert loaded_ticks(obs["hud_view"]) == env.world.mag < env.world.config.mag_size


def test_bc_batches_carry_each_steps_view_from_the_clip(tmp_path, live):
    path = tmp_path / "clip"
    writer = ClipWriter(path, source={"kind": "test"}, label_source="input_log")
    for k in range(12):
        writer.add(np.zeros(spec.PIXELS_SHAPE, np.uint8), spec.make_action(), hud={"points_ammo": live[k % 3]})
    writer.close()
    data = ClipDataset([load_clip(path)], offsets=(-1, 0), hud_view=True)
    rows = data.index[:6]
    views = data.batch(rows)["hud_view"]
    assert views.shape == (6, *HUD_VIEW_SHAPE)
    for (_, t), v in zip(rows, views):
        assert np.array_equal(v, hud_view(live[t % 3]))
    assert "hud_view" not in ClipDataset([load_clip(path)], offsets=(-1, 0)).batch(rows)
