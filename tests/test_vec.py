import functools

import numpy as np

from zombiesai import spec
from zombiesai.rl.envs import make_env
from zombiesai.rl.vec import BatchedSubprocVecEnv


def test_batched_workers_match_gymnasium_same_step_autoreset():
    fns = [functools.partial(make_env, "nacht-state", {"max_steps": 3}) for _ in range(3)]
    envs = BatchedSubprocVecEnv(fns, num_workers=2)
    try:
        assert envs.single_action_space.shape == (len(spec.ACTION_NVEC),)
        obs, _ = envs.reset(seed=7)
        assert obs["state"].shape == (3, spec.STATE_DIM)
        noop = np.tile(spec.NEUTRAL_ACTION, (3, 1))
        for _ in range(2):
            obs, rewards, terms, truncs, info = envs.step(noop)
            assert not (terms | truncs).any() and info == {}
        obs, rewards, terms, truncs, info = envs.step(noop)
        assert truncs.all() and not terms.any()
        assert info["_final_obs"].all()
        assert info["final_obs"][2]["state"].shape == (spec.STATE_DIM,)
        np.testing.assert_array_equal(info["final_info"]["round_reached"], [1, 1, 1])
        assert info["final_info"]["_round_reached"].all()
        # The returned observation already belongs to the next episode, so its step counter restarted.
        obs, *_ = envs.step(noop)
        assert obs["state"].shape == (3, spec.STATE_DIM)
    finally:
        envs.close()


def test_reset_seeds_each_env_distinctly():
    fns = [functools.partial(make_env, "nacht-state") for _ in range(4)]
    envs = BatchedSubprocVecEnv(fns, num_workers=2)
    try:
        a, _ = envs.reset(seed=3)
        b, _ = envs.reset(seed=3)
        np.testing.assert_array_equal(a["state"], b["state"])
        assert len({row.tobytes() for row in a["state"]}) == 4
    finally:
        envs.close()
