"""The synthetic stand-in for the game: the few things it must get right for the pipeline around it to be tested."""

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos.clips import load_clip
from zombiesai.demos.inputs import InputConfig
from zombiesai.demos.recorder import RecorderConfig, quality_report, record
from zombiesai.reward import REWARD_TERMS
from zombiesai.synthetic import (
    FOV_DEG,
    PX_PER_DEG,
    ScriptedPlayer,
    SyntheticActorEnv,
    SyntheticConfig,
    SyntheticSource,
    SyntheticWorld,
    scripted_actions,
)


def test_a_turn_drags_the_picture_by_the_lens():
    world = SyntheticWorld(SyntheticConfig(), seed=4)
    world.bearing = world.distance = world.health_z = np.zeros(0)  # an empty room: only the walls move
    world.to_spawn = 0
    before = world.render()
    world.step(spec.make_action(yaw=6.0))
    after = world.render()
    shift = round(6.0 * PX_PER_DEG)
    # Turning right moves the scene left by the lens's pixels per degree, give or take the rounding of the yaw.
    assert any(np.array_equal(after[:, : -s], before[:, s:]) for s in (shift - 1, shift, shift + 1))


def test_latency_holds_an_action_back_by_whole_decisions():
    world = SyntheticWorld(SyntheticConfig(latency_steps=2), seed=1)
    yaw = world.yaw
    world.step(spec.make_action(yaw=14.0))
    world.step(spec.make_action())
    assert world.yaw == pytest.approx(yaw)
    world.step(spec.make_action())
    assert world.yaw == pytest.approx((yaw + 14.0) % 360.0)


def test_the_scripted_player_plays_a_whole_game_and_the_summary_has_what_the_learner_reads():
    world, player = SyntheticWorld(seed=0), ScriptedPlayer(0)
    info, done = {}, False
    while not done:
        _, reward, terminated, truncated, info = world.step(player.act(world))
        assert np.isfinite(reward)
        done = terminated or truncated
    summary = info["episode"]
    assert summary["reason"] == "death" and summary["round_reached"] >= 3
    assert 0 < summary["hits"] <= summary["shots"]
    assert set(summary["reward_term_sums"]) == set(REWARD_TERMS)
    assert {"return", "length", "seconds", "repair_share", "max_term_share", "points_gained"} <= set(summary)


def test_the_same_seed_is_the_same_game():
    a, b = SyntheticWorld(seed=7), SyntheticWorld(seed=7)
    pa, pb = ScriptedPlayer(7), ScriptedPlayer(7)
    for _ in range(200):
        a.step(pa.act(a))
        b.step(pb.act(b))
    np.testing.assert_array_equal(a.render(), b.render())


@pytest.mark.parametrize("latency", [0, 1])
def test_a_recording_of_it_passes_the_recorders_own_checks(tmp_path, latency):
    """Labels equal to what was played, look and image motion agreeing at the world's latency, and the lens the
    flow check infers is the one it was drawn with -- the checks a real demo has to pass."""
    source = SyntheticSource(seed=2, max_steps=300, latency_steps=latency)
    config = RecorderConfig(max_steps=300, realtime=False, input=InputConfig(counts_per_degree=10.0))
    clip = load_clip(record(source, source, tmp_path / "demo", config, stop=lambda: source.done, progress_every=0))
    np.testing.assert_array_equal(clip.actions, scripted_actions(300, seed=2, latency_steps=latency)[: clip.n_steps])
    assert clip.manifest["source"]["kind"] == "synthetic"
    flow = quality_report(clip)["yaw_flow"]
    assert flow["verdict"] == "ok" and flow["lag"] == latency == flow["expected_lag"]
    assert flow["fov_deg"] == pytest.approx(FOV_DEG, abs=3.0)
    assert all(len(np.unique(clip.actions[:, h])) > 1 for h in (spec.YAW, spec.FIRE, spec.BUTTON))


def test_the_actor_env_hears_when_asked_and_starts_each_episode_on_a_new_seed():
    env = SyntheticActorEnv(3, audio_shape=(2, 25, 64))
    first, _ = env.reset()
    assert first["pixels"].shape == spec.PIXELS_SHAPE and first["pixels"].dtype == np.uint8
    assert first["audio"].shape == (2, 25, 64) and first["audio_mask"] == 1.0
    obs, _, _, _, info = env.step(spec.make_action(fire=1))
    assert info["bad"] is False and "audio" in obs
    second, _ = env.reset()
    assert not np.array_equal(first["pixels"], second["pixels"])
    assert "audio" not in SyntheticActorEnv(3).reset()[0]
