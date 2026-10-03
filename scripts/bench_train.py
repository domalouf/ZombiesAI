"""Measure the GPU training loops: one PPO update of the learner, and behavioural cloning's samples per second.

    uv run python scripts/bench_train.py learner            # one Learner.update on a realistic 4096-step batch
    uv run python scripts/bench_train.py bc                 # BC (and IDM) throughput on synthetic recordings
    uv run python scripts/bench_train.py learner --amp off  # the same, in fp32

The learner's batch is what four games hand it: 4096 decisions in segments of 16 to 256 steps, random frames
behind each one for the stack to reach into, and an audio feature per observation when the policy hears. It is
timed for the plain 4-stack and the strided (0, 1, 2, 4, 8, 16, 30) history, deaf and hearing, after one
warm-up update (cuDNN's autotuner, the allocator). "assembly" is the time inside the learner's batch-building
methods (stacking frames, uploading them, gathering minibatches), with the GPU synchronised on the way in and
out, so the rest of the update is the network's forward and backward passes.

BC is timed on clips recorded from `synthetic.SyntheticSource` through the real recorder -- memory-mapped
frames and labels exactly as a demo's -- with random per-step audio features standing in for the feature
cache when it hears. Samples per second are read off `bc.train`'s own metrics, after a first warm-up epoch.

Everything here is kept small: the GPU is usually someone's desktop too.
"""

import argparse
import contextlib
import io
import json
import statistics
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from zombiesai import spec
from zombiesai.demos import bc, idm
from zombiesai.demos.hearing import AudioFeatureConfig
from zombiesai.rl.config import RLConfig
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.rl.learner import Learner
from zombiesai.rl.segments import Segment, stack_indices

STACKS = {"4-stack": None, "strided": (0, 1, 2, 4, 8, 16, 30)}
# Methods that build batches, in the learner as it was and as it is: whichever exist are timed as assembly.
ASSEMBLY = ("_gather", "_stage", "_minibatch")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    """Wraps an object's methods to add up the wall time spent inside them, GPU work included."""

    def __init__(self, obj, names, device: torch.device):
        self.total = 0.0
        self.calls = 0
        for name in names:
            method = getattr(obj, name, None)
            if callable(method):
                setattr(obj, name, self._timed(method, device))

    def _timed(self, method, device):
        def run(*args, **kwargs):
            sync(device)
            start = time.perf_counter()
            out = method(*args, **kwargs)
            sync(device)
            self.total += time.perf_counter() - start
            self.calls += 1
            return out

        return run


# ------------------------------------------------------------------------------------------------ learner


def realistic_segments(net, offsets, audio_shape, device, steps: int = 4096, seed: int = 0) -> list[Segment]:
    """`steps` decisions in segments of 16..256, with behaviour log-probs from `net` itself (a lag-0 actor)."""
    # The same segment lengths for every variant: they come from a generator of their own.
    lengths, total = [], 0
    for n in np.random.default_rng(seed).integers(16, 257, steps):
        lengths.append(int(min(n, steps - total)))
        total += lengths[-1]
        if total >= steps:
            break
    rng = np.random.default_rng(seed + 1)
    depth = max(offsets)
    out = []
    for n in lengths:
        frames = rng.integers(0, 256, (depth + n + 1, *spec.PIXELS_SHAPE), dtype=np.uint8)
        actions = np.stack([rng.integers(0, m, n) for m in spec.ACTION_NVEC], axis=1).astype(np.int64)
        audio = mask = None
        if audio_shape is not None:
            audio = rng.normal(0, 1, (n + 1, *audio_shape)).astype(np.float32)
            mask = (rng.random(n + 1) < 0.9).astype(np.float32)
        seg = Segment(actor=len(out) % 4, version=0, context=depth, frames=frames, actions=actions,
                      logp=np.zeros(n, np.float32), rewards=rng.normal(0, 1, n).astype(np.float32),
                      bad=rng.random(n) < 0.05, terminated=bool(rng.random() < 0.1), audio=audio, audio_mask=mask)
        idx = stack_indices(depth, n, offsets)[:-1]
        logp = []
        with torch.no_grad():
            for chunk in np.array_split(np.arange(n), max(1, n // 128)):
                px = torch.from_numpy(frames[idx[chunk]]).to(device)
                au = mk = None
                if audio is not None:
                    au, mk = torch.from_numpy(audio[chunk]).to(device), torch.from_numpy(mask[chunk]).to(device)
                logits = net(px, None, au, mk)[0].float()
                act = torch.from_numpy(actions[chunk]).to(device)
                logp.append(FactoredCategorical(logits, spec.ACTION_NVEC).log_prob(act).cpu().numpy())
        seg.logp = np.concatenate(logp).astype(np.float32)
        out.append(seg)
    return out


def bench_learner(args, device: torch.device, tmp: Path) -> list[dict]:
    rows = []
    has_amp = any(f.name == "amp" for f in fields(RLConfig))
    for stack_name, offsets in STACKS.items():
        for hears in (False, True):
            config = bc.BCConfig(frame_offsets=offsets, use_audio=hears)
            torch.manual_seed(0)
            path = tmp / f"init_{stack_name}_{int(hears)}.pt"
            bc.save(path, bc.build_net(config), config, 0, {}, {})
            rl = RLConfig(init=str(path), device=str(device), critic_warmup_updates=0, target_kl=None,
                          kl_coef=args.kl_coef, seed=0)
            if has_amp:
                rl = replace(rl, amp=args.amp)
            learner = Learner(rl, device)
            audio_shape = AudioFeatureConfig().shape if hears else None
            segments = realistic_segments(learner.net, learner.offsets, audio_shape, device, args.steps)
            timer = Timer(learner, ASSEMBLY, device)
            learner.update(segments)  # warm-up: cuDNN autotuning, the allocator's first sizes
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            times, assembly = [], []
            for _ in range(args.updates):
                timer.total = 0.0
                sync(device)
                start = time.perf_counter()
                stats = learner.update(segments)
                sync(device)
                times.append(time.perf_counter() - start)
                assembly.append(timer.total)
            row = {
                "stack": stack_name, "audio": hears, "segments": len(segments), "steps": sum(s.n for s in segments),
                "update_ms": 1000 * statistics.median(times), "assembly_ms": 1000 * statistics.median(assembly),
                "amp": getattr(learner, "amp", "off") if has_amp else "off (no amp field)",
                "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                "policy_loss": stats.get("policy_loss"),
            }
            row["assembly_share"] = row["assembly_ms"] / row["update_ms"]
            rows.append(row)
            print(f"  {stack_name:>8} audio={str(hears):5} {row['segments']:>2} segments: update "
                  f"{row['update_ms']:7.0f} ms, assembly {row['assembly_ms']:6.0f} ms "
                  f"({100 * row['assembly_share']:.0f}%), peak {row['peak_mem_mb'] or 0:.0f} MB, amp {row['amp']}",
                  flush=True)
            del learner, segments
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------------------------------------ BC


def record_clips(root: Path, n_clips: int, steps: int):
    from zombiesai.demos.clips import load_clip
    from zombiesai.demos.inputs import InputConfig
    from zombiesai.demos.recorder import RecorderConfig, record
    from zombiesai.synthetic import SyntheticSource

    clips = []
    for seed in range(n_clips):
        source = SyntheticSource(seed=seed, max_steps=steps)
        config = RecorderConfig(max_steps=steps, realtime=False, input=InputConfig(counts_per_degree=10.0))
        clips.append(load_clip(record(source, source, root / f"demo_{seed}", config, stop=lambda s=source: s.done,
                                      progress_every=0)))
    return clips


def fake_features(clips, root: Path) -> dict:
    """Per-step audio features for each clip, memory-mapped from disk like the real feature cache."""
    rng = np.random.default_rng(0)
    out = {}
    for i, clip in enumerate(clips):
        path = root / f"features_{i}.npy"
        np.save(path, rng.normal(0, 1, (clip.n_steps, *AudioFeatureConfig().shape)).astype(np.float32))
        out[str(clip.path)] = np.load(path, mmap_mode="r")
    return out


def epoch_rate(run_dir: Path) -> float:
    """Samples per second over every epoch after the first (which pays for cuDNN's autotuning)."""
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return sum(r["steps"] for r in rows[1:]) / max(rows[-1]["seconds"] - rows[0]["seconds"], 1e-9)


def loader_rate(data, config: bc.BCConfig, device: torch.device, batches: int, augment: bool) -> float:
    """Samples per second of batch assembly alone, uploaded to the device: what the trainer can be fed at."""
    rng = np.random.default_rng(0)
    rows = [r for _, r in zip(range(batches), data.epoch(config.batch_size, rng))]
    sync(device)
    start = time.perf_counter()
    for r in rows:
        bc.batch_tensors(data.batch(r, rng, augment=augment), config, device)
    sync(device)
    return len(rows) * config.batch_size / (time.perf_counter() - start)


def bench_bc(args, device: torch.device, tmp: Path) -> list[dict]:
    from zombiesai.demos.dataset import ClipDataset, DataConfig

    start = time.time()
    clips = record_clips(tmp / "clips", args.clips, args.clip_steps)
    features = fake_features(clips, tmp)
    print(f"  recorded {len(clips)} synthetic clips, {sum(c.n_steps for c in clips):,} steps, "
          f"in {time.time() - start:.1f} s", flush=True)
    has_amp = any(f.name == "amp" for f in fields(bc.BCConfig))
    rows = []
    for stack_name, offsets in STACKS.items():
        if stack_name not in args.bc_stacks:
            continue
        for hears in (False, True):
            config = bc.BCConfig(frame_offsets=offsets, use_audio=hears, epochs=args.epochs,
                                 max_batches_per_epoch=args.batches, batch_size=args.batch_size, val_fraction=0.0,
                                 device=str(device), seed=0)
            if has_amp:
                config = replace(config, amp=args.amp)
            hear = (lambda clip: features[str(clip.path)]) if hears else None
            run_dir = tmp / f"bc_{stack_name}_{int(hears)}"
            with contextlib.redirect_stdout(io.StringIO()):
                bc.train(clips, config, run_dir, audio=hear)
            hearing = {"audio": hear, "audio_config": AudioFeatureConfig()} if hears else {}
            data = ClipDataset(clips, DataConfig(), offsets=config.data_offsets, **hearing)
            row = {"model": "bc", "stack": stack_name, "audio": hears, "batch_size": config.batch_size,
                   "train_samples_per_s": epoch_rate(run_dir),
                   "loader_samples_per_s": loader_rate(data, config, device, 20, augment=False),
                   "loader_aug_samples_per_s": loader_rate(data, config, device, 20, augment=True)}
            rows.append(row)
            print(f"  bc {stack_name:>8} audio={str(hears):5} batch {config.batch_size}: train "
                  f"{row['train_samples_per_s']:6.0f} samples/s | CPU batches alone "
                  f"{row['loader_samples_per_s']:6.0f}/s, augmented {row['loader_aug_samples_per_s']:6.0f}/s",
                  flush=True)
    if args.idm:
        config = idm.IDMConfig(epochs=args.epochs, max_batches_per_epoch=args.batches, batch_size=args.batch_size,
                               val_fraction=0.0, min_confidence=0.0, device=str(device), seed=0)
        if any(f.name == "amp" for f in fields(idm.IDMConfig)):
            config = replace(config, amp=args.amp)
        with contextlib.redirect_stdout(io.StringIO()):
            idm.train(clips, config, tmp / "idm")
        row = {"model": "idm", "stack": "3+1+3", "audio": False, "batch_size": config.batch_size,
               "train_samples_per_s": epoch_rate(tmp / "idm")}
        rows.append(row)
        print(f"  idm window 7 batch {config.batch_size}: train {row['train_samples_per_s']:6.0f} samples/s",
              flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("learner", "bc", "all"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", default="auto", help="auto, off, bf16 or fp16 (where the code has the option)")
    parser.add_argument("--steps", type=int, default=4096, help="decisions in the learner's batch")
    parser.add_argument("--updates", type=int, default=3, help="timed learner updates (after one warm-up)")
    parser.add_argument("--kl-coef", type=float, default=0.2, help="0 skips the reference policy, where it can")
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument("--clip-steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches", type=int, default=60, help="BC batches per epoch")
    parser.add_argument("--batch-size", type=int, default=bc.BCConfig.batch_size)
    parser.add_argument("--bc-stacks", nargs="+", default=["4-stack"], choices=list(STACKS))
    parser.add_argument("--idm", action="store_true", help="also time the inverse dynamics model's training")
    parser.add_argument("--json", type=Path, help="append the results here")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}",
          flush=True)
    results: dict = {"device": str(device), "amp": args.amp}
    with tempfile.TemporaryDirectory() as tmp:
        if args.what in ("learner", "all"):
            print(f"learner: one PPO update over {args.steps} decisions (3 epochs of 256-step minibatches)", flush=True)
            results["learner"] = bench_learner(args, device, Path(tmp))
        if args.what in ("bc", "all"):
            print("behavioural cloning on synthetic recordings", flush=True)
            results["bc"] = bench_bc(args, device, Path(tmp))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
