"""Environments for the from-scratch algorithms: classic-control correctness checks plus NachtSim, vectorized."""

import functools
import os
from dataclasses import dataclass, field

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from zombiesai.rl.vec import BatchedSubprocVecEnv


@dataclass(frozen=True)
class EnvPreset:
    gym_id: str | None  # None means NachtSim
    solved_at: float | None = None  # mean return over 100 episodes that counts as solved
    episode_metrics: tuple[str, ...] = ()  # numeric end-of-episode info keys worth logging
    asynchronous: bool = False  # worker processes pay off only when a step costs more than the IPC
    ppo: dict = field(default_factory=dict)


PRESETS = {
    "cartpole": EnvPreset(
        "CartPole-v1",
        solved_at=475.0,
        ppo=dict(total_steps=500_000, num_envs=4, rollout_steps=128),
    ),
    "lunarlander": EnvPreset(
        "LunarLander-v3",
        solved_at=200.0,
        ppo=dict(
            total_steps=1_500_000,
            num_envs=16,
            rollout_steps=1024,
            num_minibatches=256,
            gamma=0.999,
            gae_lambda=0.98,
            lr=3e-4,
        ),
    ),
    "nacht-state": EnvPreset(
        None,
        episode_metrics=("round_reached", "repair_share", "max_term_share"),
        asynchronous=True,
        ppo=dict(
            total_steps=20_000_000,
            num_envs=16,
            rollout_steps=128,
            num_minibatches=8,
            gamma=0.995,
            gae_lambda=0.95,
            lr=3e-4,
            hidden=(128, 128),
        ),
    ),
}


class SlimInfo(gym.Wrapper):
    """Drops per-step info except the listed end-of-episode keys, so vector envs don't ship it across processes."""

    def __init__(self, env: gym.Env, keep: tuple[str, ...]):
        super().__init__(env)
        self.keep = keep

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ended = terminated or truncated
        return obs, reward, terminated, truncated, {k: info[k] for k in self.keep if ended and k in info}


def make_env(name: str, sim_kwargs: dict | None = None) -> gym.Env:
    preset = PRESETS[name]
    if preset.gym_id is None:
        from zombiesai.sim.nacht_sim import NachtSim, SimConfig

        env = NachtSim(SimConfig(**(sim_kwargs or {})))
    else:
        env = gym.make(preset.gym_id)
    return SlimInfo(env, preset.episode_metrics)


def make_vector_env(
    name: str,
    num_envs: int,
    asynchronous: bool | None = None,
    sim_kwargs: dict | None = None,
    num_workers: int | None = None,
):
    """Same-step autoreset: a finished env's last observation (to bootstrap truncation) is in info["final_obs"]."""
    fns = [functools.partial(make_env, name, sim_kwargs) for _ in range(num_envs)]
    if PRESETS[name].asynchronous if asynchronous is None else asynchronous:
        return BatchedSubprocVecEnv(fns, num_workers or os.cpu_count() or 1)
    return SyncVectorEnv(fns, autoreset_mode=AutoresetMode.SAME_STEP)
