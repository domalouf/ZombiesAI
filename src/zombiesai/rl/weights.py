"""How a policy version gets from the learner to its actors: `weights.pt`, replaced atomically by the learner and
reloaded by each actor between segments, never inside a tick."""

from pathlib import Path

import torch
from torch import nn


def publish(path: Path, net: nn.Module, version: int) -> None:
    tmp = path.with_suffix(".tmp")
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"version": version, "model": state}, tmp)
    tmp.replace(path)


class WeightFollower:
    """The actor's side: reload `weights.pt` when it changes. ~3 ms for the BC net, done between segments."""

    def __init__(self, path: Path):
        self.path, self.version, self._stamp = path, -1, None

    def poll(self, net: nn.Module) -> int:
        try:
            stamp = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return self.version
        if stamp != self._stamp:
            try:
                blob = torch.load(self.path, map_location="cpu", weights_only=True)
            except (EOFError, RuntimeError, OSError):
                return self.version  # caught mid-replace; next poll
            net.load_state_dict(blob["model"])
            self.version, self._stamp = int(blob["version"]), stamp
        return self.version
