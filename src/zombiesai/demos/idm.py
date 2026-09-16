"""Inverse dynamics: what did the player press between these two frames?

This is the bridge that lets a model learn from footage nobody logged input for. The trick (Baker et al.,
"Video PreTraining", 2022) is that guessing an action *after the fact* is far easier than choosing one: the
IDM is allowed to see the frames on both sides of the decision, so a turn to the right is visible as the
scene sliding left, and a reload is visible as the animation that follows it. A model that can only see the
past -- a policy -- has to solve a much harder problem, which is why the IDM's labels are worth training on.

The chain is: a small amount of labelled play (recorded with input logging, or generated in NachtSim) trains
the IDM; the IDM then labels as many hours of unlabelled gameplay video as you can find; behavioural cloning
learns a causal policy from those pseudo-labels. Labelling is cheap, so the expensive resource -- a human
sitting at the game -- buys an hour of IDM training data rather than an hour of demonstrations.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from zombiesai import spec
from zombiesai.demos import stats
from zombiesai.demos.clips import FLAG_LOW_CONFIDENCE, Clip, attach_labels
from zombiesai.demos.dataset import ClipDataset, DataConfig, class_weights, split_clips
from zombiesai.demos.losses import factored_cross_entropy, head_confidence, head_predictions
from zombiesai.rl.encoders import PixelActorCritic


@dataclass(frozen=True)
class IDMConfig:
    before: int = 3  # frames of past context
    after: int = 3  # frames of future context -- the whole point; a policy never gets these
    hidden: int = 512
    width: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 8
    max_batches_per_epoch: int | None = None
    focal_gamma: float = 1.0
    class_balance: float = 0.999  # effective-number beta; 0 turns class weighting off
    class_balance_power: float = 0.5  # how hard to correct; 1.0 is full inverse frequency
    val_fraction: float = 0.15
    augment: bool = True
    min_confidence: float = 0.5
    seed: int = 0
    device: str = "auto"

    @property
    def window(self) -> int:
        return self.before + self.after + 1


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_net(config: IDMConfig) -> PixelActorCritic:
    return PixelActorCritic(
        spec.ACTION_NVEC, frame_stack=config.window, vector_dim=0, hidden=config.hidden, width=config.width
    )


def _tensors(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixels = torch.from_numpy(batch["pixels"]).to(device)
    actions = torch.from_numpy(batch["action"]).to(device)
    weights = torch.from_numpy(batch["weight"]).to(device)
    return pixels, actions, weights


@torch.no_grad()
def evaluate(net: PixelActorCritic, data: ClipDataset, device: torch.device, batch_size: int = 128) -> dict:
    """Balanced per-head accuracy on held-out clips, against the majority baseline."""
    was_training = net.training
    net.eval()
    predicted, target = [], []
    rng = np.random.default_rng(0)
    for rows in data.epoch(batch_size, rng, shuffle=False, drop_last=False):
        batch = data.batch(rows)
        pixels, actions, _ = _tensors(batch, device)
        logits, _, _ = net(pixels)
        predicted.append(head_predictions(logits).cpu().numpy())
        target.append(actions.cpu().numpy())
    net.train(was_training)
    if not predicted:
        return {}
    return stats.accuracy_report(np.concatenate(predicted), np.concatenate(target))


def train(clips: list[Clip], config: IDMConfig, run_dir: str | Path) -> Path:
    """Fit the IDM on labelled clips. Returns the checkpoint path."""
    labelled = [c for c in clips if c.labelled]
    if not labelled:
        raise ValueError("the IDM needs clips with actions: record with input logging, or use --source sim")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    train_clips, val_clips = split_clips(labelled, config.val_fraction, config.seed)
    data_config = DataConfig(min_confidence=config.min_confidence)
    train_data = ClipDataset(train_clips, data_config, before=config.before, after=config.after)
    val_data = ClipDataset(val_clips, data_config, before=config.before, after=config.after)
    if len(train_data) < config.batch_size:
        raise ValueError(f"only {len(train_data)} usable labelled steps; need at least {config.batch_size}")

    weights = None
    if config.class_balance:
        weights = [
            torch.from_numpy(w)
            for w in class_weights(train_data.actions(), config.class_balance, config.class_balance_power)
        ]
    net = build_net(config).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "spec_version": spec.SPEC_VERSION, "train_steps": len(train_data)}, indent=2)
    )
    log = open(run_dir / "metrics.jsonl", "a", buffering=1)
    checkpoint = run_dir / "idm.pt"
    best = -np.inf
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        losses, seen = [], 0
        for batch_index, rows in enumerate(train_data.epoch(config.batch_size, rng)):
            if config.max_batches_per_epoch and batch_index >= config.max_batches_per_epoch:
                break
            batch = train_data.batch(rows, rng, augment=config.augment)
            pixels, actions, sample_weights = _tensors(batch, device)
            logits, _, _ = net(pixels)
            loss, _ = factored_cross_entropy(
                logits, actions, class_weights=weights, sample_weights=sample_weights, focal_gamma=config.focal_gamma
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            seen += len(rows)
        report = evaluate(net, val_data, device) if len(val_data) else {}
        row = {
            "epoch": epoch,
            "steps": seen,
            "loss": float(np.mean(losses)) if losses else None,
            "seconds": round(time.time() - start, 1),
            "val": report,
        }
        log.write(json.dumps(row) + "\n")
        # Model selection on balanced accuracy, not loss: the loss is dominated by the heads that are easy.
        balanced = report.get("mean_balanced", float("nan"))
        score = balanced if np.isfinite(balanced) else -(row["loss"] or 0.0)
        print(
            f"epoch {epoch:>3}/{config.epochs} | loss {row['loss'] or float('nan'):.4f} | "
            f"val balanced {balanced:.3f}",
            flush=True,
        )
        if score >= best:
            best = score
            save(checkpoint, net, config, epoch, report)
    log.close()
    if not checkpoint.exists():  # no validation clips: keep the final weights
        save(checkpoint, net, config, config.epochs, {})
    return checkpoint


def save(path: Path, net: PixelActorCritic, config: IDMConfig, epoch: int, report: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "kind": "idm",
            "model": net.state_dict(),
            "config": asdict(config),
            "spec_version": spec.SPEC_VERSION,
            "epoch": epoch,
            "val": report,
        },
        tmp,
    )
    tmp.replace(path)


def load(path: str | Path, device: torch.device | str = "cpu") -> tuple[PixelActorCritic, IDMConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    spec.require_spec_version(checkpoint["spec_version"], str(path))
    if checkpoint.get("kind") != "idm":
        raise ValueError(f"{path} is a {checkpoint.get('kind')} checkpoint, not an IDM")
    config = IDMConfig(**checkpoint["config"])
    net = build_net(config)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    return net.to(device), config


@torch.no_grad()
def predict(
    net: PixelActorCritic, clip: Clip, config: IDMConfig, *, device: torch.device, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-label every step of a clip. Returns (actions, per-step confidence)."""
    data = ClipDataset(
        [clip], DataConfig(min_confidence=0.0), before=config.before, after=config.after, clamp_edges=True
    )
    actions = np.zeros((clip.n_steps, len(spec.ACTION_NVEC)), dtype=np.uint8)
    confidence = np.zeros(clip.n_steps, dtype=np.float32)
    for start in range(0, len(data), batch_size):
        rows = data.index[start : start + batch_size]
        pixels = torch.from_numpy(data.frames(rows)).to(device)
        logits, _, _ = net(pixels)
        steps = rows[:, 1]
        actions[steps] = head_predictions(logits).cpu().numpy().astype(np.uint8)
        confidence[steps] = head_confidence(logits).mean(-1).cpu().numpy()
    return actions, confidence


def label_clips(
    checkpoint: str | Path,
    clips: list[Clip],
    *,
    device: str = "auto",
    batch_size: int = 128,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Run the IDM over clips and write labels.npz into each. Returns one summary per clip."""
    resolved = resolve_device(device)
    net, config = load(checkpoint, resolved)
    summaries = []
    for clip in clips:
        actions, confidence = predict(net, clip, config, device=resolved, batch_size=batch_size)
        flags = np.where(confidence < min_confidence, FLAG_LOW_CONFIDENCE, 0).astype(np.uint8)
        attach_labels(
            clip.path,
            actions,
            confidence,
            label_source="idm",
            flags=flags,
            detail={
                "checkpoint": str(Path(checkpoint).resolve()),
                "mean_confidence": float(confidence.mean()),
                "min_confidence_kept": min_confidence,
            },
        )
        summaries.append(
            {
                "clip": str(clip.path),
                "steps": int(clip.n_steps),
                "mean_confidence": float(confidence.mean()),
                "kept": int((confidence >= min_confidence).sum()),
                "behaviour": stats.behaviour_stats(actions),
            }
        )
    return summaries
