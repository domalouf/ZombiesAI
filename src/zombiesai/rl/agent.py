"""Checkpoints, and a trained policy behind the reset()/act(obs) interface that rollouts, evals, and replays use."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zombiesai import spec
from zombiesai.rl.networks import ActorCritic, ObsFlattener


def save_checkpoint(path: Path, net: ActorCritic, flat: ObsFlattener, discrete: bool, cfg, step: int) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "kind": "mlp",
            "model": net.state_dict(),
            "nvec": list(net.nvec),
            "hidden": list(cfg.hidden),
            "obs_keys": list(flat.keys) if flat.keys else None,
            "obs_dim": flat.dim,
            "discrete": discrete,
            "env": cfg.env,
            "step": step,
            "spec_version": spec.SPEC_VERSION,
            "config": asdict(cfg),
        },
        tmp,
    )
    tmp.replace(path)


class PolicyAgent:
    def __init__(self, checkpoint: str | Path, deterministic: bool = False):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        spec.require_spec_version(ck["spec_version"], str(checkpoint))
        self.env = ck["env"]
        self.net = ActorCritic(ck["obs_dim"], tuple(ck["nvec"]), tuple(ck["hidden"]))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.keys = tuple(ck["obs_keys"]) if ck["obs_keys"] else None
        self.discrete = ck["discrete"]
        self.deterministic = deterministic
        self.step = ck["step"]
        self.obs_profile = "render" if self.keys and "pixels" in self.keys else "state"

    def reset(self) -> None:
        pass

    def act(self, obs):
        if self.keys is None:
            row = np.asarray(obs, dtype=np.float32).ravel()
        else:
            row = np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self.keys])
        with torch.no_grad():
            dist = self.net.dist(torch.from_numpy(row)[None])
            action = (dist.mode() if self.deterministic else dist.sample())[0].numpy()
        return int(action[0]) if self.discrete else action


def load_agent(checkpoint: str | Path, *, deterministic: bool = False, **kwargs):
    """Any checkpoint this project writes, behind reset()/act(obs): the vector-observation PPO policy, or a
    BC policy that plays from pixels. Scripts take a path and do not care which they were handed."""
    kind = torch.load(checkpoint, map_location="cpu", weights_only=True).get("kind", "mlp")
    if kind == "bc":
        from zombiesai.demos.agent import BCAgent

        return BCAgent(checkpoint, deterministic=deterministic, **kwargs)
    if kind == "mlp":
        return PolicyAgent(checkpoint, deterministic=deterministic, **kwargs)
    raise ValueError(f"{checkpoint} holds a {kind!r} checkpoint, which is not a playable policy")
