"""Subprocess vector env where each worker steps several envs, so one IPC round trip carries many env steps."""

import multiprocessing as mp
from collections.abc import Callable

import gymnasium as gym
import numpy as np


def _worker(conn, env_fns: list[Callable[[], gym.Env]]) -> None:
    envs = [fn() for fn in env_fns]
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                obs, rewards, terms, truncs, finals = [], [], [], [], []
                for j, (env, action) in enumerate(zip(envs, data)):
                    o, r, term, trunc, info = env.step(action)
                    if term or trunc:
                        finals.append((j, o, info))
                        o, _ = env.reset()
                    obs.append(o)
                    rewards.append(r)
                    terms.append(term)
                    truncs.append(trunc)
                conn.send((obs, rewards, terms, truncs, finals))
            elif cmd == "reset":
                conn.send([env.reset(seed=s)[0] for env, s in zip(envs, data)])
            elif cmd == "spaces":
                conn.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "close":
                return
    finally:
        for env in envs:
            env.close()
        conn.close()


class BatchedSubprocVecEnv:
    """Same-step autoreset with gymnasium's info layout (final_obs / final_info plus _masks), so callers can't tell."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], num_workers: int):
        self.num_envs = len(env_fns)
        self._groups = [g for g in np.array_split(np.arange(self.num_envs), min(num_workers, self.num_envs)) if len(g)]
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for group in self._groups:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, [env_fns[i] for i in group]), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self._conns[0].send(("spaces", None))
        self.single_observation_space, self.single_action_space = self._conns[0].recv()

    @staticmethod
    def _stack(obs: list):
        if isinstance(obs[0], dict):
            return {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        return np.stack(obs)

    def reset(self, seed: int | None = None):
        for conn, group in zip(self._conns, self._groups):
            conn.send(("reset", [None if seed is None else seed + int(i) for i in group]))
        return self._stack([o for conn in self._conns for o in conn.recv()]), {}

    def step(self, actions: np.ndarray):
        for conn, group in zip(self._conns, self._groups):
            conn.send(("step", actions[group]))
        obs, rewards, terms, truncs, info = [], [], [], [], {}
        n = self.num_envs
        for conn, group in zip(self._conns, self._groups):
            o, r, te, tr, finals = conn.recv()
            obs += o
            rewards += r
            terms += te
            truncs += tr
            for j, final_obs, final_info in finals:
                i = int(group[j])
                if not info:
                    info = {
                        "final_obs": np.full(n, None, dtype=object),
                        "_final_obs": np.zeros(n, bool),
                        "final_info": {},
                    }
                info["final_obs"][i] = final_obs
                info["_final_obs"][i] = True
                for k, v in final_info.items():
                    if k not in info["final_info"]:
                        info["final_info"][k] = np.zeros(n)
                        info["final_info"]["_" + k] = np.zeros(n, bool)
                    info["final_info"][k][i] = v
                    info["final_info"]["_" + k][i] = True
        return self._stack(obs), np.array(rewards), np.array(terms), np.array(truncs), info

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
