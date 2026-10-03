"""The PPO update over a batch of segments: GAE per segment, the clipped surrogate with a KL anchor to the
behavioural prior, a critic warm-up, and reward scaling (rl/parallel_ppo.py has the whole loop)."""

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zombiesai import spec
from zombiesai.rl.config import RLConfig
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.rl.segments import Segment


def compute_gae(rewards, values, last_value: float, terminated: bool, gamma: float, lam: float):
    """GAE over one segment: no episode boundary inside it, and the end bootstraps unless it terminated."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    next_value = 0.0 if terminated else float(last_value)
    gae = 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_value = values[t]
    return adv, adv + np.asarray(values, dtype=np.float32)


class RunningStd:
    """Running variance of the discounted return, per actor stream, for reward scaling."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.count, self.mean, self.m2 = 1e-4, 0.0, 1.0
        self.ret: dict[int, float] = {}

    def update(self, actor: int, rewards: np.ndarray, terminated: bool) -> None:
        ret = self.ret.get(actor, 0.0)
        for r in rewards:
            ret = ret * self.gamma + float(r)
            self.count += 1
            delta = ret - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (ret - self.mean)
        self.ret[actor] = 0.0 if terminated else ret

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.m2 / self.count, 1e-8)))


def kl_to_reference(logits: torch.Tensor, ref_logits: torch.Tensor, nvec) -> torch.Tensor:
    """(B,) KL(pi || pi_ref), summed over the factored heads."""
    p = FactoredCategorical(logits, nvec).log_probs
    q = FactoredCategorical(ref_logits, nvec).log_probs
    return (p.exp() * (p - q)).sum((-2, -1))


def root_prior(path: str | Path) -> str:
    """The behavioural-cloning checkpoint an RL checkpoint's chain of continuations started from."""
    seen = set()
    path = str(path)
    while path not in seen:
        seen.add(path)
        rl = torch.load(path, map_location="cpu", weights_only=False).get("rl") or {}
        if not rl:
            return path
        if rl.get("reference"):
            return rl["reference"]
        path = str(rl["init"])
    raise ValueError(f"checkpoint chain loops at {path}")


class Learner:
    """The PPO update over a batch of segments, on the learner's device. Separate from the process plumbing
    so it can be tested (and driven) in-process."""

    def __init__(self, config: RLConfig, device: torch.device):
        from zombiesai.demos import bc

        self.config, self.device = config, device
        self.net, self.bc_config, self.meta = bc.load(config.init, device)
        self.net.train()
        # Continuing from an RL checkpoint picks up where it stopped: the KL anchor is still the behavioural
        # prior the chain started from (not the checkpoint itself, or every restart would let the policy drift
        # further), and the update count -- so the critic warm-up is not redone -- and the anchor's weight carry
        # on.
        previous = self.meta.get("rl") or {}
        self.reference = previous.get("reference") or (root_prior(previous["init"]) if previous else config.init)
        if previous:
            self.ref, _, _ = bc.load(self.reference, device)
            self.ref.eval()
        else:
            self.ref = copy.deepcopy(self.net).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.net.critic.parameters(), lr=config.lr * 10, eps=1e-5)
        self.offsets = self.bc_config.offsets
        self.uses_audio = self.bc_config.use_audio
        # One step's audio feature, as the actors build it (_actor_loop) and the network takes it; None if deaf.
        self.audio_shape = None
        if self.uses_audio:
            from zombiesai.demos.hearing import AudioFeatureConfig, feature_config

            self.audio_shape = (feature_config(self.meta.get("audio_features")) or AudioFeatureConfig()).shape
        self.scaler = RunningStd(config.gamma)
        self.updates = int(previous.get("updates", 0))
        self.kl_coef = float(previous.get("kl_coef", config.kl_coef))

    def _forward(self, net, pixels, audio=None, mask=None):
        return net(pixels, None, audio, mask)

    def _gather(self, segments: list[Segment], rows: np.ndarray, which: np.ndarray):
        """Stacked pixels (and audio) of rows (segment, t) -- t may be n for the final observation."""
        pixels, audio, masks = [], [], []
        for s_idx, t in zip(which, rows):
            seg = segments[s_idx]
            idx = seg.context + t - np.asarray(self.offsets)
            pixels.append(seg.frames[idx])
            if self.uses_audio:
                audio.append(seg.audio[t])
                masks.append(seg.audio_mask[t])
        out = [torch.from_numpy(np.stack(pixels)).to(self.device)]
        if self.uses_audio:
            out += [torch.from_numpy(np.stack(audio)).to(self.device), torch.from_numpy(np.asarray(masks)).to(self.device)]
        else:
            out += [None, None]
        return out

    @torch.no_grad()
    def _values(self, segments: list[Segment]) -> list[np.ndarray]:
        """Current-network values of every observation, final one included: (n + 1,) per segment."""
        self.net.eval()
        out = []
        for s_idx, seg in enumerate(segments):
            rows = np.arange(seg.n + 1)
            values = []
            for chunk in np.array_split(rows, max(1, len(rows) // 256)):
                px, au, mk = self._gather(segments, chunk, np.full(len(chunk), s_idx))
                values.append(self._forward(self.net, px, au, mk)[1].float().cpu().numpy())
            out.append(np.concatenate(values))
        self.net.train()
        return out

    def update(self, segments: list[Segment]) -> dict:
        c = self.config
        for seg in segments:
            if c.reward_scale:
                self.scaler.update(seg.actor, seg.rewards, seg.terminated)
        scale = self.scaler.std if c.reward_scale else 1.0
        values = self._values(segments)
        adv, ret, which, rows, old_logp, actions, valid = [], [], [], [], [], [], []
        for s_idx, (seg, v) in enumerate(zip(segments, values)):
            a, r = compute_gae(seg.rewards / scale, v[:-1], v[-1], seg.terminated, c.gamma, c.gae_lambda)
            adv.append(a)
            ret.append(r)
            which.append(np.full(seg.n, s_idx))
            rows.append(np.arange(seg.n))
            old_logp.append(seg.logp)
            actions.append(seg.actions)
            valid.append(~seg.bad)
        adv, ret = np.concatenate(adv), np.concatenate(ret)
        which, rows = np.concatenate(which), np.concatenate(rows)
        old_logp, actions, valid = np.concatenate(old_logp), np.concatenate(actions), np.concatenate(valid)
        pred = np.concatenate([v[:-1] for v in values])
        usable = np.flatnonzero(valid)
        stats: dict[str, list[float]] = {k: [] for k in
                                         ("policy_loss", "value_loss", "entropy", "clipfrac", "kl_ref")}
        warmup = self.updates < c.critic_warmup_updates
        epoch_kl: list[float] = []
        for _ in range(c.update_epochs):
            epoch_kl = []
            perm = np.random.permutation(usable)
            for start in range(0, len(perm), c.minibatch_size):
                mb = perm[start : start + c.minibatch_size]
                if len(mb) < 2:
                    continue
                px, au, mk = self._gather(segments, rows[mb], which[mb])
                target = torch.from_numpy(ret[mb]).to(self.device)
                if warmup:
                    with torch.no_grad():
                        h = self.net.features(px, None, au, mk)
                    value = self.net.critic(h).squeeze(-1)
                    value_loss = 0.5 * ((value - target) ** 2).mean()
                    self.critic_opt.zero_grad(set_to_none=True)
                    value_loss.backward()
                    self.critic_opt.step()
                    stats["value_loss"].append(value_loss.item())
                    continue
                logits, value, _ = self._forward(self.net, px, au, mk)
                with torch.no_grad():
                    ref_logits = self._forward(self.ref, px, au, mk)[0]
                dist = FactoredCategorical(logits, spec.ACTION_NVEC)
                act = torch.from_numpy(actions[mb]).to(self.device)
                log_ratio = dist.log_prob(act) - torch.from_numpy(old_logp[mb]).to(self.device)
                ratio = log_ratio.exp()
                a = torch.from_numpy(adv[mb]).to(self.device)
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - c.clip_coef, 1 + c.clip_coef)).mean()
                value_loss = 0.5 * ((value - target) ** 2).mean()
                entropy = dist.entropy().mean()
                kl_ref = kl_to_reference(logits, ref_logits, spec.ACTION_NVEC).mean()
                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy + self.kl_coef * kl_ref
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    epoch_kl.append(((ratio - 1) - log_ratio).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > c.clip_coef).float().mean().item())
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["kl_ref"].append(kl_ref.item())
            if not warmup and c.target_kl is not None and epoch_kl and np.mean(epoch_kl) > 1.5 * c.target_kl:
                break
        self.updates += 1
        if not warmup:
            self.kl_coef = max(c.kl_min, self.kl_coef * c.kl_decay)
        out = {k: float(np.mean(v)) for k, v in stats.items() if v}
        out["approx_kl"] = float(np.mean(epoch_kl)) if epoch_kl else 0.0
        var = float(np.var(ret[usable])) if len(usable) else 0.0
        out["explained_variance"] = 1.0 - float(np.var(ret[usable] - pred[usable])) / var if var > 0 else None
        out.update({"warmup": warmup, "kl_coef": self.kl_coef, "reward_scale": scale,
                    "bad_step_frac": float(1.0 - valid.mean()) if len(valid) else 0.0,
                    "batch_steps": int(len(valid))})
        return out

    def checkpoint(self, path: Path, step: int, extra: dict | None = None) -> None:
        """A BC-format checkpoint (everything that plays a BC policy plays this) with an `rl` section."""
        tmp = path.with_suffix(".tmp")
        blob = {k: v for k, v in self.meta.items() if k not in ("model",)}
        blob.update({
            "kind": "bc",
            "model": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "rl": {"init": self.config.init, "reference": self.reference, "kl_coef": self.kl_coef,
                   "updates": self.updates, "step": step, "env": self.config.env_name,
                   "config": asdict(self.config), **(extra or {})},
        })
        torch.save(blob, tmp)
        tmp.replace(path)
