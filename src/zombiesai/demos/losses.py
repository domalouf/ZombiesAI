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


class FocalInverse(nn.Module):
    """A policy trained with `factored_cross_entropy`'s focal term and class weights, sampled at the rates the
    person it learned from had.

    Both are right for what BC is scored on -- balanced accuracy, the argmax -- and wrong for being sampled
    from: the focal term flattens every head. Where the human pressed a button on 5% of decisions, the
    focal-optimal policy says ~19%; it fires on ~25% where the human fired on 12%, sprints on ~31% where they
    sprinted on 20%. RL samples that, and is KL-anchored to it.

    The flattening is exact and invertible. For one head, the loss's minimiser q given the label distribution p
    satisfies p_c w_c f'(q_c) = const, with f(q) = -(1 - q)^gamma log q, so

        p_c  ∝  1 / (w_c g(q_c)),    g(q) = (1 - q)^gamma / q - gamma (1 - q)^(gamma - 1) log q

    and these are the logits of p, up to a constant per head: log q - log h(q) - log w, with
    h(q) = q g(q) = (1 - q)^(gamma - 1) ((1 - q) - gamma q log q). gamma 0 and no weights is the identity.
    A class the weights gave nothing (never demonstrated) gets nothing. No parameters and no persistent
    buffers: attached to a network, it leaves its state_dict as it was."""

    def __init__(self, nvec=spec.ACTION_NVEC, focal_gamma: float = 0.0, class_weights=None):
        super().__init__()
        self.nvec, self.gamma = tuple(int(n) for n in nvec), float(focal_gamma)
        if class_weights is None:
            log_w = torch.zeros(sum(self.nvec))
        else:
            w = torch.cat([torch.as_tensor(h, dtype=torch.float32) for h in class_weights])
            log_w = torch.where(w > 0, w.clamp_min(1e-30).log(), torch.tensor(1e4))
        self.register_buffer("log_w", log_w, persistent=False)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        g, out = self.gamma, []
        for chunk, log_w in zip(split_heads(logits.float(), self.nvec), torch.split(self.log_w, self.nvec)):
            log_q = chunk.log_softmax(-1)
            # 1 - q from log q, floored: a head sure to 1 - 1e-7 is as sure as fp32 can say.
            one_minus_q = (-torch.expm1(log_q)).clamp_min(1e-7)
            log_h = (g - 1.0) * one_minus_q.log() + (one_minus_q - g * log_q.exp() * log_q).log()
            out.append(log_q - log_h - log_w)
        return torch.cat(out, dim=-1)
