"""Give a trained policy the HUD-corner view (demos/hud_crops.py), keeping everything it has learned.

The converted checkpoint acts exactly as the original until training changes it: the new HUD encoder's
features enter the mixer through weights that start at zero (bc.with_hud_view). Everything else in the
checkpoint -- an RL run's update count, its KL anchor, its episode count -- is carried over, so `train_rl.py`
continues from it as from the original. The anchor itself (the BC prior) stays as it was: it does not need the
view, and the learner feeds it the same batches without it.

    uv run python scripts/add_hud_view.py runs/rl4/checkpoint.pt runs/rl4/checkpoint_hud.pt
    uv run python scripts/train_rl.py runs/rl4/checkpoint_hud.pt --actors auto --out runs/rl5
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.demos.hearing import feature_config
from zombiesai.demos.hud_crops import HUD_VIEW_SHAPE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="a BC or RL checkpoint")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: pick a new path, the original is kept as it is")

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net, config, _ = bc.load(args.checkpoint)
    audio = feature_config(raw.get("audio_features"))
    new, new_config = bc.with_hud_view(net, config, audio)

    # Same actions on the same observation, whatever the HUD corner shows: checked before anything is written.
    rng = np.random.default_rng(0)
    pixels = torch.from_numpy(rng.integers(0, 256, (4, len(config.offsets), *spec.PIXELS_SHAPE), dtype=np.uint8))
    audio_in = mask = None
    if config.use_audio:
        audio_in = torch.from_numpy(rng.normal(0, 1, (4, *(audio or bc.AudioFeatureConfig()).shape)).astype(np.float32))
        mask = torch.ones(4)
    views = torch.from_numpy(rng.integers(0, 256, (4, *HUD_VIEW_SHAPE), dtype=np.uint8))
    new.eval()
    with torch.no_grad():
        before = net(pixels, None, audio_in, mask)
        after = new(pixels, None, audio_in, mask, views)
    gap = max(float((before[0] - after[0]).abs().max()), float((before[1] - after[1]).abs().max()))
    if gap > 1e-5:
        raise SystemExit(f"the converted policy acts differently (by {gap:.2e}); nothing written")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    torch.save({**raw, "model": new.state_dict(), "config": bc.asdict(new_config),
                "obs_keys": list(new_config.obs_keys)}, tmp)
    tmp.replace(args.out)
    added = sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in net.parameters())
    print(f"wrote {args.out}: {added:,} new parameters (the HUD encoder and its mixer inputs), same actions "
          f"(largest logit gap {gap:.1e})")


if __name__ == "__main__":
    main()
