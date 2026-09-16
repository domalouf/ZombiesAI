"""Behavioural cloning: a policy trained purely on what a person did, seen only through the screen.

The network gets a causal stack of decision frames -- nothing the player could not see, in nothing but
pixels -- and predicts the eight action heads. It is the only model in this project that can be trained
before the real-game loop exists, and it is what M7's RL run starts from rather than from noise.

Three things here are the difference between a cloned policy and a policy-shaped object:

* **Class-balanced, focal cross-entropy.** `button` is 'none' ~95% of the time; plain cross-entropy answers
  that by never reloading.
* **A value head fitted to Monte-Carlo returns, and auxiliary heads for "did I score" and "am I hit."**
  They cost nothing at BC time and they hand RL a critic that is already worth something, plus an encoder
  that already represents the two features a critic needs. Both train only on the clips whose source could
  supply the targets, so a run mixing labelled video with sim episodes uses whatever each clip has.
* **Prev-action conditioning is off by default.** It is the sufficient statistic for a delayed MDP and it
  belongs in the real env -- but in BC it is also the strongest single predictor of the label, and a policy
  that learns to copy its last action looks excellent on per-frame accuracy and stands still in the game.
  Turn it on when the copy rate is being watched (see `stats.inertia_report`).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zombiesai import spec
from zombiesai.demos import stats
from zombiesai.demos.clips import DPOINTS_EDGES, Clip
from zombiesai.demos.dataset import ClipDataset, DataConfig, class_weights, split_clips
from zombiesai.demos.idm import resolve_device
from zombiesai.demos.losses import factored_cross_entropy, head_predictions
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.rl.encoders import PixelActorCritic, vector_dim

AUX_HEADS = {"aux_dpoints": len(DPOINTS_EDGES) + 1, "aux_damage": 2}


@dataclass(frozen=True)
class BCConfig:
    frame_stack: int = 4
    use_prev_actions: bool = False
    hidden: int = 512
    width: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 10
    max_batches_per_epoch: int | None = None
    focal_gamma: float = 1.0
    class_balance: float = 0.999
    class_balance_power: float = 0.5
    value_coef: float = 0.5
    aux_coef: float = 0.25
    val_fraction: float = 0.1
    augment: bool = True
    min_confidence: float = 0.0  # raise it to drop the IDM's least certain pseudo-labels
    seed: int = 0
    device: str = "auto"

    @property
    def obs_keys(self) -> tuple[str, ...]:
        return ("pixels", "prev_actions") if self.use_prev_actions else ("pixels",)


def build_net(config: BCConfig) -> PixelActorCritic:
    return PixelActorCritic(
        spec.ACTION_NVEC,
        frame_stack=config.frame_stack,
        vector_dim=vector_dim(config.obs_keys),
        hidden=config.hidden,
        width=config.width,
        aux_heads=dict(AUX_HEADS),
    )


def batch_tensors(batch: dict, config: BCConfig, device: torch.device) -> dict[str, torch.Tensor]:
    out = {
        "pixels": torch.from_numpy(batch["pixels"]).to(device),
        "action": torch.from_numpy(batch["action"]).to(device),
        "weight": torch.from_numpy(batch["weight"]).to(device),
    }
    if config.use_prev_actions:
        out["vector"] = torch.from_numpy(batch["prev_actions"]).to(device)
    for key in ("mc_return", *AUX_HEADS):
        if key in batch:
            out[key] = torch.from_numpy(batch[key]).to(device)
            out[f"{key}_mask"] = torch.from_numpy(batch[f"{key}_mask"]).to(device)
    return out


def compute_loss(net: PixelActorCritic, t: dict[str, torch.Tensor], config: BCConfig, weights) -> tuple:
    logits, value, aux = net(t["pixels"], t.get("vector"))
    loss, per_head = factored_cross_entropy(
        logits, t["action"], class_weights=weights, sample_weights=t["weight"], focal_gamma=config.focal_gamma
    )
    parts = {"policy": float(loss.detach())}
    if config.value_coef and "mc_return" in t and t["mc_return_mask"].any():
        mask = t["mc_return_mask"]
        value_loss = nn.functional.mse_loss(value[mask], t["mc_return"][mask])
        loss = loss + config.value_coef * value_loss
        parts["value"] = float(value_loss.detach())
    if config.aux_coef:
        for name in AUX_HEADS:
            if name not in t or not t[f"{name}_mask"].any():
                continue
            mask = t[f"{name}_mask"]
            aux_loss = nn.functional.cross_entropy(aux[name][mask], t[name][mask])
            loss = loss + config.aux_coef * aux_loss
            parts[name] = float(aux_loss.detach())
    return loss, logits, parts, per_head


@torch.no_grad()
def evaluate(net: PixelActorCritic, data: ClipDataset, config: BCConfig, device: torch.device) -> dict:
    """The plan's first three evaluation levels, on held-out clips.

    Per-frame accuracy is reported against the majority baseline; the behaviour comparison uses *sampled*
    actions, because that is what the policy will actually do in the game -- an argmax policy systematically
    under-fires whenever firing is the minority label.
    """
    was_training = net.training
    net.eval()
    predicted, sampled, target = [], [], []
    rng = np.random.default_rng(0)
    for rows in data.epoch(max(32, config.batch_size), rng, shuffle=False, drop_last=False):
        t = batch_tensors(data.batch(rows), config, device)
        logits, _, _ = net(t["pixels"], t.get("vector"))
        predicted.append(head_predictions(logits).cpu().numpy())
        sampled.append(FactoredCategorical(logits, spec.ACTION_NVEC).sample().cpu().numpy())
        target.append(t["action"].cpu().numpy())
    net.train(was_training)
    if not predicted:
        return {}
    predicted, sampled, target = np.concatenate(predicted), np.concatenate(sampled), np.concatenate(target)
    human = stats.behaviour_stats(target)
    policy = stats.behaviour_stats(sampled)
    return {
        "accuracy": stats.accuracy_report(predicted, target),
        "behaviour": {"human": human, "policy": policy},
        "divergence": stats.divergence_report(policy, human),
        "inertia": stats.inertia_report(sampled, target),
    }


def train(clips: list[Clip], config: BCConfig, run_dir: str | Path) -> Path:
    labelled = [c for c in clips if c.labelled]
    if not labelled:
        raise ValueError("behavioural cloning needs labelled clips: record with input logging or label with the IDM")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    train_clips, val_clips = split_clips(labelled, config.val_fraction, config.seed)
    data_config = DataConfig(min_confidence=config.min_confidence)
    before = config.frame_stack - 1
    train_data = ClipDataset(train_clips, data_config, before=before)
    val_data = ClipDataset(val_clips, data_config, before=before)
    if len(train_data) < config.batch_size:
        raise ValueError(f"only {len(train_data)} usable steps; need at least {config.batch_size}")

    human = stats.behaviour_stats(train_data.actions())
    weights = None
    if config.class_balance:
        weights = [
            torch.from_numpy(w)
            for w in class_weights(train_data.actions(), config.class_balance, config.class_balance_power)
        ]
    net = build_net(config).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "obs_keys": list(config.obs_keys),
                "spec_version": spec.SPEC_VERSION,
                "train_steps": len(train_data),
                "val_steps": len(val_data),
                "train_clips": [str(c.path) for c in train_clips],
                "val_clips": [str(c.path) for c in val_clips],
                "label_sources": sorted({c.label_source for c in labelled}),
                "human_behaviour": human,
            },
            indent=2,
        )
    )
    log = open(run_dir / "metrics.jsonl", "a", buffering=1)
    checkpoint, best, start = run_dir / "bc.pt", -np.inf, time.time()
    report: dict = {}
    for epoch in range(1, config.epochs + 1):
        losses, seen = [], 0
        for batch_index, rows in enumerate(train_data.epoch(config.batch_size, rng)):
            if config.max_batches_per_epoch and batch_index >= config.max_batches_per_epoch:
                break
            t = batch_tensors(train_data.batch(rows, rng, augment=config.augment), config, device)
            loss, _, parts, _ = compute_loss(net, t, config, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(parts)
            seen += len(rows)
        mean = {k: float(np.mean([p[k] for p in losses if k in p])) for k in {k for p in losses for k in p}}
        report = evaluate(net, val_data, config, device) if len(val_data) else {}
        row = {"epoch": epoch, "steps": seen, "loss": mean, "seconds": round(time.time() - start, 1), "val": report}
        log.write(json.dumps(row) + "\n")
        balanced = report.get("accuracy", {}).get("mean_balanced", float("nan"))
        print(
            f"epoch {epoch:>3}/{config.epochs} | policy loss {mean.get('policy', float('nan')):.4f} | "
            f"val balanced {balanced:.3f} | "
            f"stats {'ok' if report.get('divergence', {}).get('passed', False) else 'off'}",
            flush=True,
        )
        score = balanced if np.isfinite(balanced) else -mean.get("policy", 0.0)
        if score >= best:
            best = score
            save(checkpoint, net, config, epoch, report, human)
    log.close()
    if not checkpoint.exists():
        save(checkpoint, net, config, config.epochs, report, human)
    (run_dir / "report.json").write_text(json.dumps({"human": human, "final": report}, indent=2))
    return checkpoint


def save(path: Path, net: PixelActorCritic, config: BCConfig, epoch: int, report: dict, human: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "kind": "bc",
            "model": net.state_dict(),
            "config": asdict(config),
            "obs_keys": list(config.obs_keys),
            "spec_version": spec.SPEC_VERSION,
            "epoch": epoch,
            "val": report,
            "human_behaviour": human,
        },
        tmp,
    )
    tmp.replace(path)


def load(path: str | Path, device: torch.device | str = "cpu") -> tuple[PixelActorCritic, BCConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    spec.require_spec_version(checkpoint["spec_version"], str(path))
    if checkpoint.get("kind") != "bc":
        raise ValueError(f"{path} is a {checkpoint.get('kind')} checkpoint, not a BC policy")
    config = BCConfig(**checkpoint["config"])
    net = build_net(config)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    return net.to(device), config, checkpoint
