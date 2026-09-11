import numpy as np
import pytest

from zombiesai import spec

# Changing anything hashed in spec.py must be a deliberate act: update this value in the same commit,
# knowing that every checkpoint and episode recorded under the old version stops loading.
GOLDEN_SPEC_VERSION = "bea67a6feccbf23a"


def test_spec_version_is_pinned():
    assert spec.SPEC_VERSION == GOLDEN_SPEC_VERSION


def test_space_shapes():
    assert spec.ACTION_NVEC == (3, 3, 9, 5, 2, 2, 2, 6)
    assert spec.ACT_ENC_DIM == 32
    assert int(np.prod(spec.ACTION_NVEC)) == 19_440
    assert spec.HUD_DIM == 16
    assert spec.STATE_DIM == 50
    state = spec.observation_space("state")
    assert state["state"].shape == (50,)
    assert state["hud"].shape == (16,)
    assert state["prev_actions"].shape == (64,)
    render = spec.observation_space("render")
    assert render["pixels"].shape == (72, 128, 3) and render["pixels"].dtype == np.uint8
    assert "state" not in render.spaces


def test_audio_is_an_opt_in_key():
    assert "audio" not in spec.observation_space("render").spaces
    assert spec.observation_space("render", audio=True)["audio"].shape == (2, 64)


def test_neutral_action():
    assert spec.NEUTRAL_ACTION == tuple(spec.make_action().tolist())
    assert spec.NEUTRAL_ACTION == (1, 1, 4, 2, 0, 0, 0, 0)


def test_compact_profile_round_trips():
    assert spec.N_COMPACT == 48
    assert len(set(spec.COMPACT_ACTIONS)) == 48
    assert spec.COMPACT_ACTIONS[0] == spec.NEUTRAL_ACTION
    for i in range(spec.N_COMPACT):
        assert spec.project_to_compact(spec.expand_compact(i)) == i


def test_batch_projection_matches_scalar():
    rng = np.random.default_rng(0)
    actions = rng.integers(0, spec.ACTION_NVEC, size=(500, len(spec.ACTION_NVEC)))
    batch = spec.project_to_compact_batch(actions)
    assert batch.tolist() == [spec.project_to_compact(a) for a in actions]


def test_projection_keeps_buttons():
    reload_anywhere = spec.make_action(forward=1, yaw=14, button="reload")
    assert spec.COMPACT_ACTIONS[spec.project_to_compact(reload_anywhere)] == tuple(
        spec.make_action(button="reload").tolist()
    )


@pytest.mark.parametrize("bad", [(1, 1, 4, 2, 0, 0, 0), (1, 1, 9, 2, 0, 0, 0, 0), (1.0, 1, 4, 2, 0, 0, 0, 0)])
def test_invalid_actions_rejected(bad):
    with pytest.raises(ValueError):
        spec.action_tuple(np.array(bad))


def test_hud_encode_decode_round_trip():
    raw = spec.hud_raw(
        {
            "round": 7,
            "points": 4520,
            "mag_ammo": 15,
            "reserve_ammo": 120,
            "grenades": 2,
            "damage_flash": 0.4,
            "time_since_damage": 3.0,
            "hud_confidence": 0.97,
            "round_transition": 0,
            "downed": 0,
            "prompt_price": 1200,
            "prompt_door": 0,
            "prompt_weapon": 1,
            "prompt_box": 0,
            "prompt_repair": 0,
            "time_in_round": 45.0,
        }
    )
    enc = spec.encode_hud(raw)
    assert enc.dtype == np.float32 and enc.shape == (16,)
    np.testing.assert_allclose(spec.decode_hud(enc), raw, rtol=1e-5)


def test_hud_encoding_is_bounded():
    enc = spec.encode_hud(np.full(spec.HUD_DIM, 1e12))
    assert enc.max() == spec.HUD_CLIP
    assert spec.encode_hud(np.full(spec.HUD_DIM, -5.0)).min() == 0.0


def test_hud_raw_requires_every_field():
    with pytest.raises(KeyError):
        spec.hud_raw({"round": 1})


def test_prev_actions_one_hot():
    a = spec.make_action(forward=1, yaw=-30, fire=1, button="use")
    enc = spec.encode_prev_actions([tuple(a.tolist()), spec.NEUTRAL_ACTION])
    assert enc.shape == (64,)
    assert enc.sum() == 2 * len(spec.ACTION_NVEC)
    first = enc[: spec.ACT_ENC_DIM]
    offsets = np.concatenate(([0], np.cumsum(spec.ACTION_NVEC)[:-1]))
    assert np.flatnonzero(first).tolist() == (a + offsets).tolist()


def test_mismatched_versions_are_a_hard_error():
    spec.require_spec_version(spec.SPEC_VERSION, "here")
    with pytest.raises(spec.SpecMismatchError):
        spec.require_spec_version("0000000000000000", "an old episode")


def test_state_never_exposes_hidden_health():
    assert not any("hp" in f or "health" in f for f in spec.STATE_FIELDS)
