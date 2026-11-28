"""Re-label recorded demos from their raw input logs, under this code's action bins.

    uv run python scripts/requantize.py data/demos            # every demo under it
    uv run python scripts/requantize.py data/demos --dry-run  # say what would change, write nothing

A demo's labels are bin indices, and the bins are the spec's: after a change to them (the half- and one-degree
look bins, 2026-11) every recording is refused until it is re-labelled. The raw log beside each clip
(inputs.jsonl) is what makes that arithmetic instead of another evening of play (demos/recorder.py
`requantize`). Each clip is re-read with the counts per degree, bindings and mark key it was recorded with.

The labels it replaces are kept beside the new ones as labels.<their spec>.npz. A clip already on this spec
is left alone unless --force; a policy's play run is skipped (its corrections are not on the recorder's clock).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos.inputs import InputConfig
from zombiesai.demos.recorder import requantize


def clip_dirs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if (path / "clip.json").exists():
            found.append(path)
        else:
            found += sorted(p.parent for p in path.rglob("clip.json"))
    return found


def recorded_input(manifest: dict) -> InputConfig:
    """The InputConfig a clip was recorded with. A clip from before the mark key existed has none."""
    recorded = manifest.get("config", {}).get("recorder", {}).get("input") or {}
    if "counts_per_degree" not in recorded:
        raise ValueError("its clip.json does not say what counts per degree it was recorded with")
    return InputConfig(counts_per_degree=float(recorded["counts_per_degree"]), bindings=dict(recorded["bindings"]),
                       hold_fraction=float(recorded.get("hold_fraction", InputConfig.hold_fraction)),
                       flag_clamped=bool(recorded.get("flag_clamped", True)), mark_key=recorded.get("mark_key"))


def unchanged_apart_from_bins(old: dict, new: dict) -> bool:
    """Whether re-labelling changed only what new bins should: the same steps and flags, the other heads' labels,
    and the degrees turned. Not to the step: the recorder gave a mouse event to the decision that was open
    when it *read* the event, the log gives it to the one its timestamp falls in, so a count or a key press at
    a boundary moves to the neighbouring step on 2-4% of them (nothing is lost: the turn is the same)."""
    if len(old["action"]) != len(new["action"]) or not np.array_equal(old["flags"], new["flags"]):
        return False
    other = [h for h in range(len(spec.ACTION_HEADS)) if h not in (spec.YAW, spec.PITCH)]
    moved = float((old["action"][:, other] != new["action"][:, other]).any(axis=1).mean())
    turns = np.corrcoef(old["yaw_deg"], new["yaw_deg"])[0, 1] if np.ptp(old["yaw_deg"]) else 1.0
    return moved < 0.001 and turns > 0.995


def look_summary(actions: np.ndarray) -> str:
    still = lambda head: float((actions[:, head] == spec.NEUTRAL_ACTION[head]).mean())  # noqa: E731
    return f"no turn on {still(spec.YAW):.0%} of steps, level on {still(spec.PITCH):.0%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+", help="clip directories, or directories holding clips")
    parser.add_argument("--dry-run", action="store_true", help="say what would be done, write nothing")
    parser.add_argument("--force", action="store_true", help="also re-label clips already on this spec")
    args = parser.parse_args()

    failed = 0
    for clip_dir in clip_dirs(args.paths):
        manifest = json.loads((clip_dir / "clip.json").read_text())
        was = manifest.get("spec_version")
        if manifest.get("label_source") != "input_log" or not (clip_dir / "inputs.jsonl").exists():
            print(f"{clip_dir}: skipped ({manifest.get('label_source')} labels, not a demo's input log)")
            continue
        if was == spec.SPEC_VERSION and not args.force:
            print(f"{clip_dir}: already on spec {was}")
            continue
        try:
            config = recorded_input(manifest)
        except ValueError as error:
            print(f"{clip_dir}: FAILED: {error}", file=sys.stderr)
            failed += 1
            continue
        if args.dry_run:
            print(f"{clip_dir}: would re-label from spec {was} at {config.counts_per_degree:g} counts a degree")
            continue
        old = dict(np.load(clip_dir / "labels.npz"))
        backup = clip_dir / f"labels.{was}.npz"
        if not backup.exists():
            shutil.copyfile(clip_dir / "labels.npz", backup)
        requantize(clip_dir, config)
        new = dict(np.load(clip_dir / "labels.npz"))
        note = "" if unchanged_apart_from_bins(old, new) else (
            "  WARNING: more than the look bins changed -- compare with " + backup.name)
        failed += bool(note)
        print(f"{clip_dir}: spec {was} -> {spec.SPEC_VERSION}, {len(new['action'])} steps; "
              f"{look_summary(new['action'])}{note}")
    if failed:
        raise SystemExit(f"{failed} clips need a look")


if __name__ == "__main__":
    main()
