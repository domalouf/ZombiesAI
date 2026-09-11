"""Factored categorical policy: one independent categorical per action head."""

import functools

import torch


# Finite, not -inf: exp() still underflows to exactly 0, but p * log(p) and its gradient stay 0 * finite
# instead of 0 * -inf = NaN.
PAD_LOGIT = -1e9


@functools.cache
def _layout(nvec: tuple[int, ...]) -> tuple[torch.Tensor, int]:
    """Where each head's logits land in a (heads, max_n) grid; the rest is padding."""
    width = max(nvec)
    return torch.tensor([h * width + j for h, n in enumerate(nvec) for j in range(n)]), width


class FactoredCategorical:
    """Independent per-head categoricals: log-probs and entropies add, so each head gets its own gradient term."""

    def __init__(self, logits: torch.Tensor, nvec: tuple[int, ...]):
        # All heads share one padded tensor so each op below runs once, not once per head.
        index, width = _layout(tuple(nvec))
        grid = logits.new_full((*logits.shape[:-1], len(nvec) * width), PAD_LOGIT)
        grid = grid.index_copy(-1, index.to(logits.device), logits)
        self.log_probs = torch.log_softmax(grid.view(*logits.shape[:-1], len(nvec), width), dim=-1)

    def sample(self) -> torch.Tensor:
        # Gumbel-max: argmax(log p + Gumbel noise) is an exact sample from the categorical.
        u = torch.rand_like(self.log_probs).clamp_min(1e-12)
        return (self.log_probs - torch.log(-torch.log(u))).argmax(-1)

    def mode(self) -> torch.Tensor:
        return self.log_probs.argmax(-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)

    def entropy(self) -> torch.Tensor:
        return -(self.log_probs.exp() * self.log_probs).sum((-2, -1))
