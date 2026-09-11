"""Proximal Policy Optimization from scratch (Schulman et al. 2017; GAE 2016; Huang et al.'s "37 details", 2022)."""

import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from zombiesai import spec
from zombiesai.rl.agent import save_checkpoint
from zombiesai.rl.envs import PRESETS, make_vector_env
from zombiesai.rl.networks import ActorCritic, ObsFlattener, action_nvec


@dataclass
class PPOConfig:
    env: str = "cartpole"
    total_steps: int = 500_000
    num_envs: int = 8
    rollout_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2.5e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    # Off by default: clipping value updates to +/-clip_coef stalls the critic whenever returns span more
    # than a few units (LunarLander plateaued at 115 with it on). It only made sense with reward normalization.
    clip_vloss: bool = False
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    target_kl: float | None = None
    hidden: tuple[int, ...] = (64, 64)
    seed: int = 1
    num_workers: int | None = None  # env worker processes for async presets; default: every CPU thread
    torch_threads: int = 4
    checkpoint_every: int = 25
    sim: dict = field(default_factory=dict)

    @classmethod
    def for_preset(cls, env: str, **overrides) -> "PPOConfig":
        return cls(env=env, **{**PRESETS[env].ppo, **overrides})


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over a (T, N) rollout. dones[t] cuts the trace; a truncated step's reward already has gamma * V(final)."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * live - values[t]
        gae = delta + gamma * lam * live * gae
        advantages[t] = gae
    return advantages, advantages + values


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


def ppo_update(net: ActorCritic, opt: torch.optim.Optimizer, b: Batch, cfg: PPOConfig) -> dict[str, float]:
    n = len(b.obs)
    stats: dict[str, list[float]] = defaultdict(list)
    for _ in range(cfg.update_epochs):
        epoch_kl = []
        for idx in torch.randperm(n).split(n // cfg.num_minibatches):
            dist = net.dist(b.obs[idx])
            log_ratio = dist.log_prob(b.actions[idx]) - b.logprobs[idx]
            ratio = log_ratio.exp()
            adv = b.advantages[idx]
            if cfg.norm_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # Pessimistic bound: the larger of the unclipped and clipped losses, so moving the ratio past
            # 1 +/- clip_coef in the direction the advantage favours earns nothing more.
            policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()

            v = net.value(b.obs[idx])
            v_err = (v - b.returns[idx]) ** 2
            if cfg.clip_vloss:
                v_clipped = b.values[idx] + (v - b.values[idx]).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_err = torch.max(v_err, (v_clipped - b.returns[idx]) ** 2)
            value_loss = 0.5 * v_err.mean()
            entropy = dist.entropy().mean()

            loss = policy_loss - cfg.ent_coef * entropy + cfg.vf_coef * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                # Schulman's low-variance KL estimator: E[(r - 1) - log r] >= 0.
                epoch_kl.append(((ratio - 1) - log_ratio).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
        if cfg.target_kl is not None and np.mean(epoch_kl) > cfg.target_kl:
            break
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    # The last epoch's KL says how far the whole update moved the policy; averaging in the first
    # minibatch (where the ratio is exactly 1) would dilute it.
    out["approx_kl"] = float(np.mean(epoch_kl))
    var = b.returns.var().item()
    out["explained_variance"] = 1.0 - (b.returns - b.values).var().item() / var if var > 0 else None
    return out


def train(cfg: PPOConfig, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "spec_version": spec.SPEC_VERSION}, indent=2))
    preset = PRESETS[cfg.env]
    torch.manual_seed(cfg.seed)

    envs = make_vector_env(cfg.env, cfg.num_envs, sim_kwargs=cfg.sim, num_workers=cfg.num_workers)
    try:
        flat = ObsFlattener(envs.single_observation_space)
        nvec = action_nvec(envs.single_action_space)
        discrete = isinstance(envs.single_action_space, spaces.Discrete)
        net = ActorCritic(flat.dim, nvec, cfg.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

        T, N = cfg.rollout_steps, cfg.num_envs
        num_updates = cfg.total_steps // (T * N)
        obs_buf = torch.zeros(T, N, flat.dim)
        act_buf = torch.zeros(T, N, len(nvec), dtype=torch.long)
        logp_buf, rew_buf, done_buf, val_buf = (torch.zeros(T, N) for _ in range(4))

        obs_np, _ = envs.reset(seed=cfg.seed)
        obs = flat(obs_np)
        ep_return, ep_len = np.zeros(N), np.zeros(N, dtype=np.int64)
        recent: deque[dict] = deque(maxlen=100)
        episodes, global_step, start = 0, 0, time.time()
        log = open(run_dir / "metrics.jsonl", "a", buffering=1)

        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                opt.param_groups[0]["lr"] = cfg.lr * (1.0 - (update - 1) / num_updates)

            # Per-step inference on a handful of rows is all op overhead: extra threads only add sync cost.
            torch.set_num_threads(1)
            for t in range(T):
                with torch.no_grad():
                    dist = net.dist(obs)
                    action = dist.sample()
                    logp_buf[t] = dist.log_prob(action)
                    val_buf[t] = net.value(obs)
                obs_buf[t], act_buf[t] = obs, action
                env_action = action[:, 0].numpy() if discrete else action.numpy()
                obs_np, reward, terminated, truncated, info = envs.step(env_action)

                ep_return += reward
                ep_len += 1
                reward = reward.astype(np.float32)
                # A truncated episode didn't end, we just stopped watching it: bootstrap from its last state.
                if truncated.any():
                    idx = np.flatnonzero(truncated)
                    final = flat(flat.stack([info["final_obs"][i] for i in idx]))
                    with torch.no_grad():
                        reward[idx] += cfg.gamma * net.value(final).numpy()
                done = terminated | truncated
                rew_buf[t] = torch.from_numpy(reward)
                done_buf[t] = torch.from_numpy(done.astype(np.float32))
                for i in np.flatnonzero(done):
                    ep = {"return": float(ep_return[i]), "length": int(ep_len[i])}
                    final_info = info.get("final_info", {})
                    for k in preset.episode_metrics:
                        if k in final_info and final_info["_" + k][i]:
                            ep[k] = float(final_info[k][i])
                    recent.append(ep)
                    episodes += 1
                    ep_return[i], ep_len[i] = 0.0, 0
                obs = flat(obs_np)
                global_step += N

            torch.set_num_threads(cfg.torch_threads)
            with torch.no_grad():
                last_value = net.value(obs)
            advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_value, cfg.gamma, cfg.gae_lambda)
            batch = Batch(
                obs_buf.reshape(T * N, -1),
                act_buf.reshape(T * N, -1),
                logp_buf.reshape(-1),
                advantages.reshape(-1),
                returns.reshape(-1),
                val_buf.reshape(-1),
            )
            stats = ppo_update(net, opt, batch, cfg)

            row = {
                "update": update,
                "step": global_step,
                "sps": int(global_step / (time.time() - start)),
                "lr": opt.param_groups[0]["lr"],
                "episodes": episodes,
                **stats,
            }
            if recent:
                row["return_mean"] = float(np.mean([e["return"] for e in recent]))
                row["length_mean"] = float(np.mean([e["length"] for e in recent]))
                for k in preset.episode_metrics:
                    vals = [e[k] for e in recent if k in e]
                    if vals:
                        row[f"{k}_mean"] = float(np.mean(vals))
            log.write(json.dumps(row) + "\n")
            _print_row(row, num_updates, preset.episode_metrics)
            if update % cfg.checkpoint_every == 0 or update == num_updates:
                save_checkpoint(run_dir / "checkpoint.pt", net, flat, discrete, cfg, global_step)
        log.close()
    finally:
        envs.close()

    if preset.solved_at is not None and recent:
        mean = np.mean([e["return"] for e in recent])
        verdict = "SOLVED" if len(recent) == 100 and mean >= preset.solved_at else "not solved"
        print(f"\nlast {len(recent)} episodes: mean return {mean:.1f} (solved at {preset.solved_at}) -> {verdict}")
    return run_dir / "checkpoint.pt"


def _print_row(row: dict, num_updates: int, metrics: tuple[str, ...]) -> None:
    parts = [f"upd {row['update']:>5}/{num_updates}", f"step {row['step']:>11,}", f"sps {row['sps']:>6,}"]
    if "return_mean" in row:
        parts.append(f"return {row['return_mean']:>8.2f}")
        parts += [f"{k.split('_')[0]} {row[f'{k}_mean']:.2f}" for k in metrics[:1] if f"{k}_mean" in row]
    ev = row["explained_variance"]
    parts.append(
        f"ent {row['entropy']:.2f} kl {row['approx_kl']:.4f} clip {row['clipfrac']:.2f} "
        f"ev {'-' if ev is None else f'{ev:.2f}'}"
    )
    print(" | ".join(parts), flush=True)
