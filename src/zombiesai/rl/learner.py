"""The PPO update over a batch of segments: GAE per segment, the clipped surrogate with a KL anchor to the
behavioural prior, a critic warm-up, and reward scaling (rl/parallel_ppo.py has the whole loop).

How the update spends the GPU, because at 4096 decisions a batch the bookkeeping cost as much as the learning:

* **Every frame crosses to the GPU once per update.** A segment's frames (uint8, each stored once) and its
  audio features go up in one copy; each step's stack is a row of precomputed frame indices (the
  `context + t - offsets` arithmetic of `segments.stack_indices`), and a minibatch is gathered on the device.
  Stacking on the CPU meant copying every frame `len(offsets)` times per epoch and uploading it again.
* **Big batches where nothing learns:** values of every observation, and the frozen reference policy's logits
  of every step, are computed once per update in chunks of `EVAL_CHUNK`. The reference does not change during
  an update, so running it on every minibatch of every epoch computed the same logits three times. With no
  KL anchor (`kl_coef` 0: a fresh start, `prepare_init`) there is no reference network at all.
* **Mixed precision** (`RLConfig.amp`, `encoders.precision`): the network runs under autocast, the losses and
  the action distribution in fp32. In fp32 the arithmetic is the one it always was.
"""

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter
from torch import nn

from zombiesai import spec
from zombiesai.demos.hud_crops import HUD_VIEW_SHAPE, hud_view
from zombiesai.rl.config import RLConfig
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.rl.encoders import precision
from zombiesai.rl.segments import Segment

# Observations per forward pass where no gradient is kept (values, reference logits): a 7-frame stack of 512
# is ~100 MB of uint8 on the device, and large enough that the GPU is busy rather than waiting on launches.
EVAL_CHUNK = 512


def segment_gae(rewards, values, last_values, terminated, gamma: float, lam: float):
    """GAE of many segments at once: each a list entry, no episode boundary inside one, and each end
    bootstrapping from its last value unless it terminated. Returns (advantages, returns), float32, every
    segment's steps in order, concatenated.

    The recursion runs backwards in time, so the segments are laid side by side, padded after their ends with
    zero TD errors (which keep a zero advantage zero until a segment's real last step), and one first-order
    filter runs along all of them in reverse."""
    lengths = np.array([len(r) for r in rewards])
    if not len(lengths) or lengths.max() == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    steps = np.arange(lengths.max())
    inside = steps[None, :] < lengths[:, None]
    r = np.zeros(inside.shape)
    v = np.zeros(inside.shape)
    v_next = np.zeros(inside.shape)
    for i, n in enumerate(lengths):
        if n:
            r[i, :n] = rewards[i]
            v[i, :n] = values[i]
            v_next[i, : n - 1] = v[i, 1:n]
            v_next[i, n - 1] = 0.0 if terminated[i] else float(last_values[i])
    delta = np.where(inside, r + gamma * v_next - v, 0.0)
    adv = lfilter([1.0], [1.0, -gamma * lam], delta[:, ::-1], axis=1)[:, ::-1]
    adv = adv[inside].astype(np.float32)
    return adv, adv + v[inside].astype(np.float32)


def compute_gae(rewards, values, last_value: float, terminated: bool, gamma: float, lam: float):
    """GAE over one segment: no episode boundary inside it, and the end bootstraps unless it terminated."""
    return segment_gae([np.asarray(rewards)], [np.asarray(values)], [last_value], [terminated], gamma, lam)


class RunningStd:
    """Running variance of the discounted return, per actor stream, for reward scaling."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.count, self.mean, self.m2 = 1e-4, 0.0, 1.0
        self.ret: dict[int, float] = {}

    def update(self, actor: int, rewards: np.ndarray, terminated: bool) -> None:
        rewards = np.asarray(rewards, dtype=np.float64)
        ret = self.ret.get(actor, 0.0)
        if len(rewards):
            # The stream's discounted return after each reward, ret_t = gamma * ret_{t-1} + r_t, as one filter;
            # then the batch's mean and spread merged into the running ones (Chan et al.), which is what
            # Welford's one-at-a-time update adds up to.
            returns = lfilter([1.0], [1.0, -self.gamma], rewards, zi=[self.gamma * ret])[0]
            n, mean = len(returns), float(returns.mean())
            total, delta = self.count + n, mean - self.mean
            self.mean += delta * n / total
            self.m2 += float(((returns - mean) ** 2).sum()) + delta**2 * self.count * n / total
            self.count = total
            ret = float(returns[-1])
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


# The history a policy started from nothing gets: half a second, dense where motion is read.
FRESH_OFFSETS = (0, 1, 2, 4, 8)
# A policy started from nothing would pick every look bin alike: a mean 12 degrees a decision, ~175 degrees a
# second of random jerking, flicks of 30 as often as no turn. Instead it starts with a turn of d degrees
# exp(-|d| / scale) as likely as none: ~2 degrees a decision of yaw, mostly level pitch. Only the starting
# point -- the bins are all still there and PPO moves the odds from here.
FRESH_LOOK_SCALE_DEG = {spec.YAW: 4.0, spec.PITCH: 2.0}


def calm_looks(net) -> None:
    """Set the actor's yaw and pitch biases so small turns are the likely ones (FRESH_LOOK_SCALE_DEG). Its
    weights start near zero (std 0.01), so the biases are the starting odds."""
    bins = {spec.YAW: spec.YAW_BINS_DEG, spec.PITCH: spec.PITCH_BINS_DEG}
    starts = np.concatenate([[0], np.cumsum(spec.ACTION_NVEC)])
    with torch.no_grad():
        for head, scale in FRESH_LOOK_SCALE_DEG.items():
            logits = -np.abs(np.asarray(bins[head])) / scale
            net.actor.bias[starts[head]:starts[head + 1]] = torch.as_tensor(logits - logits.max(),
                                                                            dtype=net.actor.bias.dtype)


def prepare_init(config: RLConfig, run_dir: str | Path) -> RLConfig:
    """`init="fresh"`: no behavioural prior. Build a new pixel+audio policy that also looks at the HUD corner
    (demos/hud_crops.py), save it as an ordinary BC-format
    checkpoint at `run_dir/init.pt` (so the actors, the fleet's workers and every BC tool load it as they load
    any start), and return the config pointing there -- with no KL anchor and no critic warm-up, since there is
    no cloned behaviour to protect. Any other config comes back unchanged."""
    if config.init != "fresh":
        return config
    from zombiesai.demos import bc
    from zombiesai.demos.hearing import AudioFeatureConfig

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bc_config = bc.BCConfig(frame_offsets=FRESH_OFFSETS, use_audio=True, use_hud_view=True, seed=config.seed)
    features = AudioFeatureConfig()
    torch.manual_seed(config.seed)
    path = run_dir / "init.pt"
    net = bc.build_net(bc_config, features)
    calm_looks(net)
    bc.save(path, net, bc_config, 0, {}, {}, features)
    return replace(config, init=str(path), kl_coef=0.0, kl_min=0.0, critic_warmup_updates=0)


@dataclass
class Staged:
    """One update's segments on the device: frames stored once, and indices that stack them."""

    frames: torch.Tensor  # (frames, H, W, 3) uint8, every segment's frames back to back
    stacks: torch.Tensor  # (observations, len(offsets)) int64: each observation's stack, as rows of `frames`
    row_obs: torch.Tensor  # (decisions,) int64: the observation each decision was taken on
    audio: torch.Tensor | None  # (observations, *feature) float32
    audio_mask: torch.Tensor | None  # (observations,) float32
    obs_counts: list[int]  # observations per segment, its steps plus the final one
    hud_view: torch.Tensor | None = None  # (observations, 60, 80, 3) uint8


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
        self.updates = int(previous.get("updates", 0))
        self.kl_coef = float(previous.get("kl_coef", config.kl_coef))
        # A run without an anchor (kl_coef 0 from the start: nothing to stay close to) keeps no reference.
        self.ref = None
        if self.kl_coef > 0:
            if previous:
                self.ref, _, _ = bc.load(self.reference, device)
                self.ref.eval()
            else:
                self.ref = copy.deepcopy(self.net).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.net.critic.parameters(), lr=config.lr * 10, eps=1e-5)
        self.precision = precision(config.amp, device)
        self.amp = self.precision.mode
        self.grad_scaler = self.precision.grad_scaler()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # few shapes (minibatch, eval chunk), each seen every update
        self.offsets = self.bc_config.offsets
        self.uses_audio = self.bc_config.use_audio
        # One step's audio feature, as the actors build it (_actor_loop) and the network takes it; None if deaf.
        self.audio_shape = None
        if self.uses_audio:
            from zombiesai.demos.hearing import AudioFeatureConfig, feature_config

            self.audio_shape = (feature_config(self.meta.get("audio_features")) or AudioFeatureConfig()).shape
        self.uses_hud_view = self.bc_config.use_hud_view
        self.scaler = RunningStd(config.gamma)

    def _forward(self, net, pixels, audio=None, mask=None, hud=None):
        # The frozen prior may predate the HUD corner: a network without that branch ignores `hud`.
        return net(pixels, None, audio, mask, hud)

    def _gather(self, segments: list[Segment], rows: np.ndarray, which: np.ndarray):
        """Stacked pixels (and audio, and the HUD corner) of rows (segment, t) -- t may be n for the final
        observation -- built on the CPU, row by row. The plain statement of what `_stage` and `_minibatch`
        compute in bulk."""
        pixels, audio, masks, views = [], [], [], []
        for s_idx, t in zip(which, rows):
            seg = segments[s_idx]
            idx = seg.context + t - np.asarray(self.offsets)
            pixels.append(seg.frames[idx])
            if self.uses_audio:
                audio.append(seg.audio[t])
                masks.append(seg.audio_mask[t])
            if self.uses_hud_view:
                views.append(seg.hud_view[t] if seg.hud_view is not None else hud_view(None))
        out = [torch.from_numpy(np.stack(pixels)).to(self.device)]
        if self.uses_audio:
            out += [torch.from_numpy(np.stack(audio)).to(self.device), torch.from_numpy(np.asarray(masks)).to(self.device)]
        else:
            out += [None, None]
        out.append(torch.from_numpy(np.stack(views)).to(self.device) if self.uses_hud_view else None)
        return out

    def _stage(self, segments: list[Segment]) -> Staged:
        """Upload every segment's frames and audio in one copy each, and index each observation's stack."""
        n_frames = [len(seg.frames) for seg in segments]
        frame_base = np.concatenate([[0], np.cumsum(n_frames)[:-1]]).astype(np.int64)
        obs_counts = [seg.n + 1 for seg in segments]
        back = np.asarray(self.offsets, dtype=np.int64)
        # Observation t of a segment stacks frames[context + t - back], in that segment's stretch of `frames`.
        stacks = np.concatenate([base + seg.context + np.arange(seg.n + 1)[:, None] - back[None, :]
                                 for base, seg in zip(frame_base, segments)])
        obs_base = np.concatenate([[0], np.cumsum(obs_counts)[:-1]]).astype(np.int64)
        row_obs = np.concatenate([base + np.arange(seg.n) for base, seg in zip(obs_base, segments)])
        pinned = self.device.type == "cuda"
        host = torch.empty((sum(n_frames), *segments[0].frames.shape[1:]), dtype=torch.uint8, pin_memory=pinned)
        np.concatenate([seg.frames for seg in segments], out=host.numpy())
        audio = mask = None
        if self.uses_audio:
            # A segment from a deaf actor hears nothing: masked, exactly as a clip without sound trains.
            audio = torch.empty((sum(obs_counts), *self.audio_shape), dtype=torch.float32, pin_memory=pinned)
            mask = torch.empty(sum(obs_counts), dtype=torch.float32, pin_memory=pinned)
            for base, count, seg in zip(obs_base, obs_counts, segments):
                heard = seg.audio is not None
                audio[base : base + count] = torch.from_numpy(np.asarray(seg.audio, np.float32)) if heard else 0.0
                mask[base : base + count] = torch.from_numpy(np.asarray(seg.audio_mask, np.float32)) if heard else 0.0
            audio, mask = audio.to(self.device, non_blocking=True), mask.to(self.device, non_blocking=True)
        views = None
        if self.uses_hud_view:
            # One view per observation, like the audio; a segment without them (impossible past decode_segment,
            # but an in-process actor's) sees an empty corner.
            views = torch.empty((sum(obs_counts), *HUD_VIEW_SHAPE), dtype=torch.uint8, pin_memory=pinned)
            for base, count, seg in zip(obs_base, obs_counts, segments):
                views[base : base + count] = torch.from_numpy(seg.hud_view) if seg.hud_view is not None else 0
            views = views.to(self.device, non_blocking=True)
        return Staged(frames=host.to(self.device, non_blocking=True), stacks=torch.from_numpy(stacks).to(self.device),
                      row_obs=torch.from_numpy(row_obs).to(self.device), audio=audio, audio_mask=mask,
                      obs_counts=obs_counts, hud_view=views)

    def _minibatch(self, staged: Staged, obs: torch.Tensor):
        """Stacked pixels (and audio, and the HUD corner) of the observations `obs`, gathered on the device."""
        pixels = staged.frames[staged.stacks[obs]]
        views = staged.hud_view[obs] if staged.hud_view is not None else None
        if staged.audio is None:
            return pixels, None, None, views
        return pixels, staged.audio[obs], staged.audio_mask[obs], views

    @torch.no_grad()
    def _evaluate(self, net, staged: Staged, obs: torch.Tensor):
        """(logits, values) of the observations `obs`, fp32, in chunks of EVAL_CHUNK with no gradient."""
        was_training = net.training
        net.eval()
        logits, values = [], []
        for chunk in torch.split(obs, EVAL_CHUNK):
            with self.precision.autocast():
                lg, v, _ = self._forward(net, *self._minibatch(staged, chunk))
            logits.append(lg.float())
            values.append(v.float())
        net.train(was_training)
        return torch.cat(logits), torch.cat(values)

    def update(self, segments: list[Segment]) -> dict:
        c = self.config
        for seg in segments:
            if c.reward_scale:
                self.scaler.update(seg.actor, seg.rewards, seg.terminated)
        scale = self.scaler.std if c.reward_scale else 1.0
        staged = self._stage(segments)
        all_obs = torch.arange(sum(staged.obs_counts), device=self.device)
        values = self._evaluate(self.net, staged, all_obs)[1].cpu().numpy()
        per_segment = np.split(values, np.cumsum(staged.obs_counts)[:-1])
        adv, ret = segment_gae([seg.rewards / scale for seg in segments], [v[:-1] for v in per_segment],
                               [v[-1] for v in per_segment], [seg.terminated for seg in segments],
                               c.gamma, c.gae_lambda)
        old_logp = np.concatenate([seg.logp for seg in segments])
        actions = np.concatenate([seg.actions for seg in segments])
        valid = np.concatenate([~seg.bad for seg in segments])
        pred = np.concatenate([v[:-1] for v in per_segment])
        usable = np.flatnonzero(valid)
        to_device = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)  # noqa: E731
        adv_t, ret_t, old_logp_t, actions_t = map(to_device, (adv, ret, old_logp, actions.astype(np.int64)))
        warmup = self.updates < c.critic_warmup_updates
        # The frozen prior's logits of every step that can be trained on, once: they are the same every epoch.
        ref_logits = None
        if self.ref is not None and not warmup and len(usable):
            ref_logits = torch.zeros((len(valid), sum(spec.ACTION_NVEC)), device=self.device)
            usable_t = to_device(usable)
            ref_logits[usable_t] = self._evaluate(self.ref, staged, staged.row_obs[usable_t])[0]
        stats: dict[str, list[torch.Tensor]] = {k: [] for k in
                                                ("policy_loss", "value_loss", "entropy", "clipfrac", "kl_ref")}
        epoch_kl: list[torch.Tensor] = []
        autocast, scaler = self.precision.autocast, self.grad_scaler
        for _ in range(c.update_epochs):
            epoch_kl = []
            perm = np.random.permutation(usable)
            for start in range(0, len(perm), c.minibatch_size):
                mb = perm[start : start + c.minibatch_size]
                if len(mb) < 2:
                    continue
                rows = to_device(mb)
                px, au, mk, hd = self._minibatch(staged, staged.row_obs[rows])
                target = ret_t[rows]
                if warmup:
                    with torch.no_grad(), autocast():
                        h = self.net.features(px, None, au, mk, hd)
                    with autocast():
                        value = self.net.critic(h).squeeze(-1)
                    value_loss = 0.5 * ((value.float() - target) ** 2).mean()
                    self.critic_opt.zero_grad(set_to_none=True)
                    scaler.scale(value_loss).backward()
                    scaler.step(self.critic_opt)
                    scaler.update()
                    stats["value_loss"].append(value_loss.detach())
                    continue
                with autocast():
                    logits, value, _ = self._forward(self.net, px, au, mk, hd)
                logits, value = logits.float(), value.float()
                dist = FactoredCategorical(logits, spec.ACTION_NVEC)
                log_ratio = dist.log_prob(actions_t[rows]) - old_logp_t[rows]
                ratio = log_ratio.exp()
                a = adv_t[rows]
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - c.clip_coef, 1 + c.clip_coef)).mean()
                value_loss = 0.5 * ((value - target) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy
                if ref_logits is not None:
                    kl_ref = kl_to_reference(logits, ref_logits[rows], spec.ACTION_NVEC).mean()
                    loss = loss + self.kl_coef * kl_ref
                    stats["kl_ref"].append(kl_ref.detach())
                self.opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                scaler.step(self.opt)
                scaler.update()
                with torch.no_grad():
                    epoch_kl.append(((ratio - 1) - log_ratio).mean())
                    stats["clipfrac"].append(((ratio - 1).abs() > c.clip_coef).float().mean())
                stats["policy_loss"].append(policy_loss.detach())
                stats["value_loss"].append(value_loss.detach())
                stats["entropy"].append(entropy.detach())
            # One synchronisation per epoch, not one per number per minibatch.
            if not warmup and c.target_kl is not None and epoch_kl and _mean(epoch_kl) > 1.5 * c.target_kl:
                break
        self.updates += 1
        if not warmup and self.ref is not None:
            self.kl_coef = max(c.kl_min, self.kl_coef * c.kl_decay)
        out = {k: _mean(v) for k, v in stats.items() if v}
        out["approx_kl"] = _mean(epoch_kl) if epoch_kl else 0.0
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


def _mean(values: list[torch.Tensor]) -> float:
    """The mean of per-minibatch scalars, as the float the metrics log wants (np.mean of their floats)."""
    return float(torch.stack(values).double().mean())
