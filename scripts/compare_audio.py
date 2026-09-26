"""Does hearing help? Vision-only against vision+audio, same data, same split, same budget.

    uv run python scripts/compare_audio.py --out runs/audio_compare \\
        --train data/demos/demo_0000 data/demos/demo_0002:0:0.8 \\
        --val data/demos/demo_0002:0.8:1 data/demos/demo_0001 --epochs 6 --seeds 0 1

A clip is `path[:from:to]`, the fractions of its steps to use -- the held-out tail of a long recording is
`demo:0.8:1`. Three arms, each trained per seed with identical settings:

* `vision`  -- pixels only, the network every checkpoint so far has been;
* `audio`   -- pixels + the stereo log-mel (`demos/hearing.py`);
* `masked`  -- the audio network with every clip's audio masked out: the same parameters and fusion layer,
  nothing to hear. The difference between `audio` and `masked` is what the sound is worth; the difference
  between `masked` and `vision` is what the extra layer costs or buys on its own.

Reported per arm, from the final epoch (no picking the best epoch on the validation set): balanced accuracy
per head, its mean, and the plain cross-entropy per head. Then two diagnostics, on the saved (best-epoch)
checkpoint for the first:
balanced accuracy on *change steps* only (the label differs from the previous step's) -- the moments a
policy has to decide something, where hearing your own gunfire from the step before is no help -- and the
copy rate, because audio carries an echo of the player's own actions just as prev-actions do.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from zombiesai import spec
from zombiesai.demos import bc, stats
from zombiesai.demos.clips import clip_span, load_clip
from zombiesai.demos.dataset import ClipDataset, DataConfig
from zombiesai.demos.hearing import DEFAULT_CACHE, AudioFeatureConfig, clip_features
from zombiesai.demos.losses import head_predictions

ARMS = ("vision", "audio", "masked")


def parse_clip(text: str):
    path, *span = text.split(":")
    clip = load_clip(path, require_labels=True)
    if span:
        lo, hi = (float(x) for x in span)
        clip = clip_span(clip, int(round(lo * clip.n_steps)), int(round(hi * clip.n_steps)))
    return clip


@torch.no_grad()
def change_step_report(net, data: ClipDataset, config: bc.BCConfig, device) -> dict:
    """Balanced accuracy per head on the steps whose label changed from the step before."""
    net.eval()
    predicted, target, changed = [], [], []
    for rows in data.epoch(256, np.random.default_rng(0), shuffle=False, drop_last=False):
        t = bc.batch_tensors(data.batch(rows), config, device)
        logits, _, _ = net(t["pixels"], t.get("vector"), t.get("audio"), t.get("audio_mask"))
        predicted.append(head_predictions(logits).cpu().numpy())
        target.append(t["action"].cpu().numpy())
        prev = np.stack([data.clips[c].actions[max(s - 1, 0)] for c, s in rows]).astype(np.int64)
        first = np.array([s == data._segment_start[c][s] for c, s in rows])
        changed.append((target[-1] != prev) & ~first[:, None])
    predicted, target, changed = np.concatenate(predicted), np.concatenate(target), np.concatenate(changed)
    out = {}
    for head, name in enumerate(spec.ACTION_HEADS):
        m = changed[:, head]
        if m.sum() < 20:
            continue
        out[name] = {"steps": int(m.sum()),
                     "balanced": float(stats.balanced_accuracy(predicted[m], target[m])[head])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=None, help="per epoch")
    parser.add_argument("--audio-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_clips = [parse_clip(c) for c in args.train]
    val_clips = [parse_clip(c) for c in args.val]
    features = AudioFeatureConfig()
    for clip in train_clips + val_clips:  # warm the cache once, outside every run's clock
        clip_features(clip, features, args.audio_cache)
    base = bc.BCConfig(epochs=args.epochs, max_batches_per_epoch=args.max_batches, device=args.device)
    results = {}
    for seed in args.seeds:
        for arm in args.arms:
            config = replace(base, seed=seed, use_audio=arm != "vision")
            run = args.out / f"{arm}_s{seed}"
            print(f"\n== {arm}, seed {seed} -> {run}", flush=True)
            deaf = (lambda clip: None) if arm == "masked" else None
            checkpoint = bc.train(train_clips, config, run, val_clips=val_clips, audio_features=features,
                                  audio_cache=args.audio_cache, audio=deaf)
            rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            final = rows[-1]["val"]
            # The saved checkpoint -- the one that would be played -- is the best epoch on validation.
            net, loaded, _ = bc.load(checkpoint, bc.resolve_device(args.device))
            hearing = {}
            if config.use_audio:
                source = deaf or (lambda clip: clip_features(clip, features, args.audio_cache))
                hearing = {"audio": source, "audio_config": features}
            data = ClipDataset(val_clips, DataConfig(), before=config.frame_stack - 1, **hearing)
            results[f"{arm}_s{seed}"] = {
                "arm": arm, "seed": seed,
                "final_epoch": rows[-1]["epoch"],
                "best_epoch": torch.load(checkpoint, weights_only=True)["epoch"],
                "best_mean_balanced": max(r["val"]["accuracy"]["mean_balanced"] for r in rows),
                "balanced": final["accuracy"]["balanced"],
                "mean_balanced": final["accuracy"]["mean_balanced"],
                "nll": final["nll"],
                "copy_joint": final["inertia"]["policy"]["copy_joint"],
                "human_copy_joint": final["inertia"]["human"]["copy_joint"],
                "change_steps_best_ckpt": change_step_report(net, data, loaded, bc.resolve_device(args.device)),
                "val_steps": len(data),
            }
            (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print_table(results)


def print_table(results: dict) -> None:
    heads = spec.ACTION_HEADS
    print("\nfinal-epoch validation, balanced accuracy per head | mean | mean NLL")
    print(f"{'run':>12} " + " ".join(f"{h[:7]:>7}" for h in heads) + "    mean     nll")
    for name, r in results.items():
        print(f"{name:>12} " + " ".join(f"{r['balanced'][h]:7.3f}" for h in heads)
              + f"  {r['mean_balanced']:6.3f}  {r['nll']['mean']:6.4f}")
    print("\nper-head NLL")
    for name, r in results.items():
        print(f"{name:>12} " + " ".join(f"{r['nll']['per_head'][h]:7.4f}" for h in heads))
    print("\nchange steps only (best-epoch checkpoint), balanced accuracy")
    for name, r in results.items():
        cs = r["change_steps_best_ckpt"]
        print(f"{name:>12} " + " ".join(f"{cs[h]['balanced']:7.3f}" if h in cs else "      -" for h in heads))


if __name__ == "__main__":
    main()
