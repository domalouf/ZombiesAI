"""Run one episode of an agent in an env, optionally recording it to the episode store."""

import gymnasium as gym
import numpy as np

from zombiesai.store.episode_store import EpisodeWriter

SUMMARY_KEYS = ("round_reached", "rounds_survived", "repair_share", "max_term_share", "gain_clips", "reward_term_sums")


def run_episode(
    env: gym.Env,
    agent,
    *,
    seed: int | None = None,
    options: dict | None = None,
    writer: EpisodeWriter | None = None,
) -> dict:
    obs, _ = env.reset(seed=seed, options=options)
    agent.reset()
    total, steps = 0.0, 0
    while True:
        action = np.asarray(agent.act(obs))
        next_obs, reward, terminated, truncated, info = env.step(action)
        if writer is not None:
            writer.add_step(obs, action, reward, info, terminated, truncated)
        total += reward
        steps += 1
        obs = next_obs
        if terminated or truncated:
            break
    summary = {"return": total, "steps": steps, "terminated": terminated, "truncated": truncated}
    summary.update({k: info[k] for k in SUMMARY_KEYS if k in info})
    if writer is not None:
        writer.close(summary=summary, final_obs=obs)
    return summary
