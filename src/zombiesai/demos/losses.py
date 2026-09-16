"""Losses for factored-action supervision, shared by behavioural cloning and the inverse dynamics model."""

import torch
from torch import nn

from zombiesai import spec


def split_heads(logits: torch.Tensor, nvec=spec.ACTION_NVEC) -> tuple[torch.Tensor, ...]:
    return torch.split(logits, list(nvec), dim=-1)


def head_predictions(logits: torch.Tensor, nvec=spec.ACTION_NVEC) -> torch.Tensor:
    """(B, n_heads) argmax action, the greedy decode used for accuracy reports."""
    return torch.stack([h.argmax(-1) for h in split_heads(logits, nvec)], dim=-1)


def head_confidence(logits: torch.Tensor, nvec=spec.ACTION_NVEC) -> torch.Tensor:
    """(B, n_heads) probability the network assigned to the value it predicted."""
    return torch.stack([h.softmax(-1).max(-1).values for h in split_heads(logits, nvec)], dim=-1)


def factored_cross_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    nvec=spec.ACTION_NVEC,
    class_weights: list[torch.Tensor] | None = None,
    sample_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum of one weighted cross-entropy per action head.

    Factorisation is what makes 110k human decisions worth having: each one supervises eight heads instead of
    being a single draw from a 29,160-way categorical. Class weights and the focal term exist for one head in
    particular -- `button` is 'none' about 95% of the time, and unweighted cross-entropy answers that by never
    pressing anything (PLAN.md, "Demonstrations and BC").
    """
    total = logits.new_zeros(())
    per_head: dict[str, torch.Tensor] = {}
    for head, chunk in enumerate(split_heads(logits, nvec)):
        target = actions[:, head]
        weight = None if class_weights is None else class_weights[head].to(chunk.device)
        loss = nn.functional.cross_entropy(chunk, target, weight=weight, reduction="none")
        if focal_gamma:
            with torch.no_grad():
                p_true = chunk.softmax(-1).gather(-1, target[:, None]).squeeze(-1)
            loss = loss * (1.0 - p_true).clamp_min(0.0) ** focal_gamma
        if sample_weights is not None:
            loss = loss * sample_weights
            loss = loss.sum() / sample_weights.sum().clamp_min(1e-6)
        else:
            loss = loss.mean()
        per_head[spec.ACTION_HEADS[head]] = loss.detach()
        total = total + loss
    return total, per_head
