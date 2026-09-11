import json

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.agents.random_agent import RandomAgent
from zombiesai.rollout import run_episode
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.store.episode_store import (
    FLAG_TERMINATED,
    FLAG_TRUNCATED,
    EpisodeWriter,
    episode_dir,
    load_episode,
)


def recorded_episode(tmp_path, max_steps=300):
    env = NachtSim(SimConfig(max_steps=max_steps))
    writer = EpisodeWriter(episode_dir(tmp_path, 0), config={"max_steps": max_steps}, flush_every=64)
    summary = run_episode(env, RandomAgent(0), seed=0, writer=writer)
    return summary, load_episode(episode_dir(tmp_path, 0))


def test_round_trip(tmp_path):
    summary, ep = recorded_episode(tmp_path)
    assert ep.complete and ep.n_steps == summary["steps"]
    assert ep.manifest["spec_version"] == spec.SPEC_VERSION
    assert ep.manifest["summary"]["steps"] == summary["steps"]
    assert ep.meta["action"].shape == (ep.n_steps, len(spec.ACTION_NVEC))
    assert ep.meta["hud"].shape == (ep.n_steps, spec.HUD_DIM)
    assert ep.meta["state"].shape == (ep.n_steps, spec.STATE_DIM)
    assert ep.meta["reward"].sum() == pytest.approx(summary["return"], rel=1e-4)
    assert ep.final_obs is not None and ep.final_obs["state"].shape == (spec.STATE_DIM,)
    last = int(ep.meta["flags"][-1])
    assert last & (FLAG_TERMINATED | FLAG_TRUNCATED)
    assert not any(int(f) & (FLAG_TERMINATED | FLAG_TRUNCATED) for f in ep.meta["flags"][:-1])
    assert not list(ep.path.glob("meta_part_*.npz"))


def test_spec_mismatch_refuses_to_load(tmp_path):
    recorded_episode(tmp_path)
    path = episode_dir(tmp_path, 0) / "spec.json"
    manifest = json.loads(path.read_text())
    manifest["spec_version"] = "0000000000000000"
    path.write_text(json.dumps(manifest))
    with pytest.raises(spec.SpecMismatchError):
        load_episode(episode_dir(tmp_path, 0))


def test_crashed_episode_loads_flushed_chunks(tmp_path):
    env = NachtSim(SimConfig(max_steps=10**9))
    writer = EpisodeWriter(episode_dir(tmp_path, 1), flush_every=50)
    obs, _ = env.reset(seed=0)
    agent = RandomAgent(0)
    for _ in range(130):
        action = agent.act(obs)
        next_obs, r, term, trunc, info = env.step(action)
        writer.add_step(obs, action, r, info, term, trunc)
        obs = next_obs
        if term:
            break
    # no close(): the process "died" mid-episode
    ep = load_episode(episode_dir(tmp_path, 1))
    assert not ep.complete
    assert ep.n_steps in (50, 100)


def test_frames_are_stored_once_per_step(tmp_path):
    writer = EpisodeWriter(episode_dir(tmp_path, 2), hud_crop_shape=(4, 6))
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(5, *spec.PIXELS_SHAPE), dtype=np.uint8)
    for k in range(5):
        obs = {"pixels": frames[k], "hud": np.zeros(spec.HUD_DIM, np.float32)}
        info = {"reward_terms": np.zeros(8), "dt": 1 / 15, "hud_crops": np.full((4, 6), k, np.uint8)}
        writer.add_step(obs, spec.make_action(), 0.0, info, False, k == 4)
    writer.close()
    ep = load_episode(episode_dir(tmp_path, 2))
    np.testing.assert_array_equal(ep.frames, frames)
    assert ep.hud_crops.shape == (5, 4, 6) and ep.hud_crops[3].max() == 3


def test_refuses_to_overwrite(tmp_path):
    EpisodeWriter(episode_dir(tmp_path, 3)).close()
    with pytest.raises(FileExistsError):
        EpisodeWriter(episode_dir(tmp_path, 3))
