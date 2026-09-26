"""A trained pixel policy behind the same reset()/act(obs) interface the baselines and PPO use.

The frame stack lives here rather than in the environment, and it is built exactly the way the training
loader builds it -- most recent frame last, the first frame repeated when there is no history yet -- because
any disagreement between the two shows up as a policy that plays worse than its validation accuracy says it
should, and nothing points at the cause.
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.rl.distributions import FactoredCategorical


class BCAgent:
    """Plays from pixels alone (plus prev-actions if the checkpoint was trained with them)."""

    obs_profile = "render"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        deterministic: bool = False,
        device: str = "cpu",
        temperature: float = 1.0,
    ):
        self.device = torch.device(device)
        self.net, self.config, self.meta = bc.load(checkpoint, self.device)
        self.deterministic = deterministic
        self.temperature = temperature
        self.env = "nacht-render"
        self.step = int(self.meta.get("epoch", 0))
        self._frames: deque[np.ndarray] = deque(maxlen=self.config.frame_stack)
        self._history = [spec.NEUTRAL_ACTION] * spec.PREV_ACTION_HISTORY
        # The probability-weighted mean turn, in degrees, from the last act(). Sampling a look bin afresh every
        # 67 ms jumps between 0, +6 and -2 degrees even when the policy is sure of a gentle turn; the mean
        # moves as smoothly as the policy's beliefs do, and is not limited to the nine bins.
        self.last_look: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        self._frames.clear()
        self._history = [spec.NEUTRAL_ACTION] * spec.PREV_ACTION_HISTORY

    def stack(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.uint8)
        if not self._frames:
            self._frames.extend([frame] * self.config.frame_stack)
        else:
            self._frames.append(frame)
        return np.stack(self._frames)

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        if "pixels" not in obs:
            raise KeyError("a BC policy needs pixel observations: build the sim with obs_profile='render'")
        pixels = torch.from_numpy(self.stack(obs["pixels"])[None]).to(self.device)
        vector = None
        if self.config.use_prev_actions:
            encoded = spec.encode_prev_actions(self._history)[None]
            vector = torch.from_numpy(encoded).to(self.device)
        logits, _, _ = self.net(pixels, vector)
        dist = FactoredCategorical(logits / (1.0 if self.deterministic else self.temperature), spec.ACTION_NVEC)
        action = dist.mode()[0] if self.deterministic else dist.sample()[0]
        self.last_look = tuple(
            float((dist.log_probs[0, head, : len(bins)].exp().cpu().numpy() * np.asarray(bins)).sum())
            for head, bins in ((spec.YAW, spec.YAW_BINS_DEG), (spec.PITCH, spec.PITCH_BINS_DEG))
        )
        action = action.cpu().numpy().astype(np.int64)
        self._history = [tuple(int(v) for v in action)] + self._history[:-1]
        return action
