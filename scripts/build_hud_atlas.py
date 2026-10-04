"""Rebuild the HUD digit atlas (src/zombiesai/hud/glyph_atlas.npz) from recordings' HUD crops.

Two steps, with a human in between -- the glyphs are harvested, the labels are looked at:

    # 1. segment digit-like runs across the clips, cluster them, dump a contact sheet per glyph set
    uv run --with pillow python scripts/build_hud_atlas.py harvest data/demos/demo_0001 data/demos/demo_0002 \\
        runs/play --out /tmp/atlas
    # 2. look at /tmp/atlas/<set>.png: row k is cluster k (its centre, then members). Write one character
    #    per cluster into configs/hud/atlas_labels.json ("0"-"9", "?" = not a digit, a reject template,
    #    "-" = drop the cluster), then:
    uv run python scripts/build_hud_atlas.py build /tmp/atlas configs/hud/atlas_labels.json

Harvest is deterministic for the same clips, stride, k and seed (all recorded in clusters.npz and the
labels file), so the committed labels rebuild the committed atlas. After labelling new recordings, the
existing atlas's guess for every cluster is written to <set>.txt beside the sheet: start from that.
Pillow is only needed for the sheets.

The weapon names (hud/weapons.py) have their own templates, src/zombiesai/hud/weapon_names.npz, and the same
two steps. Sources are clips or sample_hud.py's .npz files:

    uv run --with pillow python scripts/build_hud_atlas.py harvest-weapons data/hud_samples/live.npz --out /tmp/weapons
    # /tmp/weapons/names.png: row k is cluster k (its centre, then members as the screen shows them).
    # Write one weapon key per cluster (hud/weapons.py: WEAPONS) into configs/hud/weapon_labels.json --
    # "?" for a cluster that is not a name (scene showing through), "-" to drop it -- then:
    uv run python scripts/build_hud_atlas.py build-weapons /tmp/weapons configs/hud/weapon_labels.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

from zombiesai.demos.clips import load_clip
from zombiesai.hud.glyphs import ATLAS_PATH, LINES, REJECT, GlyphSet, harvest, load_atlas, save_atlas
from zombiesai.hud.parse import fit_crops
from zombiesai.hud.weapons import (
    NAME_MIN_INK,
    NAMES_PATH,
    WEAPONS,
    load_names,
    name_band,
    name_ink,
    save_names,
    signature,
)

DEFAULT_K = {"points": 40, "ammo": 40}


def clip_dirs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if (path / "clip.json").exists():
            found.append(path)
        else:
            found += sorted(p.parent for p in path.rglob("clip.json"))
    return found


def do_harvest(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_set: dict[str, list[np.ndarray]] = {}
    sources = []
    for clip_dir in clip_dirs(args.paths):
        clip = load_clip(clip_dir)
        for line in LINES.values():
            crops = clip.hud(line.region)
            if crops is None:
                continue
            stack = fit_crops(np.asarray(crops[:: args.stride]), line.region)
            patches, _ = harvest(stack, line)
            by_set.setdefault(line.glyphs, []).append(patches)
            print(f"{clip_dir.name} {line.name}: {len(patches)} glyphs")
        sources.append(str(clip_dir))
    try:
        atlas = load_atlas(ATLAS_PATH)
    except FileNotFoundError:
        atlas = {}
    saved = {}
    for name, parts in by_set.items():
        pats = np.concatenate(parts)
        x = pats.reshape(len(pats), -1)
        k = args.k or DEFAULT_K.get(name, 40)
        centres, assign = kmeans2(x, k, seed=args.seed, minit="++", iter=30)
        centres = centres.reshape(k, *pats.shape[1:])
        counts = np.bincount(assign, minlength=k)
        saved[f"{name}__centres"] = centres
        saved[f"{name}__counts"] = counts
        guess = ""
        if name in atlas:
            labels, _, _ = atlas[name].classify(centres)
            guess = "".join(labels.tolist())
        (out / f"{name}.txt").write_text(f"clusters: {k}\ncounts: {counts.tolist()}\nexisting atlas says: {guess}\n")
        _sheet(out / f"{name}.png", pats, assign, centres)
        print(f"{name}: {len(pats)} glyphs in {k} clusters -> {out / (name + '.png')}; atlas guess: {guess}")
    tally = harvest_tally(args, clip_dirs(args.paths))
    if tally is not None:
        centres, counts = tally
        saved["tally__centres"], saved["tally__counts"] = centres.astype(np.float16), counts
        (out / "tally.txt").write_text(f"clusters: {len(centres)}\ncounts: {counts.tolist()}\n")
        _tally_sheet(out / "tally.png", centres)
        print(f"tally: {len(centres)} clusters -> {out / 'tally.png'}")
    np.savez_compressed(out / "clusters.npz", **saved,
                        meta=np.array(json.dumps({"sources": sources, "stride": args.stride, "seed": args.seed})))


def _sheet(path: Path, pats, assign, centres, per_row: int = 12, scale: int = 4) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("  (no pillow: skipping the contact sheet; run with `uv run --with pillow`)")
        return
    h, w = pats.shape[1:]
    rows = []
    for k in range(len(centres)):
        idx = np.flatnonzero(assign == k)
        rng = np.random.default_rng(k)
        members = rng.choice(idx, min(per_row - 1, len(idx)), replace=False) if len(idx) else []
        row = np.full((h + 2, per_row * (w + 2)), 0.5, np.float32)
        row[1 : h + 1, :w] = centres[k]
        for j, s in enumerate(members, start=1):
            row[1 : h + 1, j * (w + 2) : j * (w + 2) + w] = pats[s]
        rows.append(row)
    sheet = (np.concatenate(rows) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(sheet).resize((sheet.shape[1] * scale, sheet.shape[0] * scale), Image.NEAREST).save(path)


def tally_ink(crops: np.ndarray) -> np.ndarray:
    """The tally strokes' own red, ~(108, 1, 0): nothing in the scene is that saturated."""
    x = crops.astype(np.int16)
    return (x[..., 0] > 60) & (x[..., 1] < 25) & (x[..., 2] < 25)


def harvest_tally(args, dirs: list[Path], k: int = 16):
    masks = []
    for clip_dir in dirs:
        crops = load_clip(clip_dir).hud("round")
        if crops is not None:
            masks.append(tally_ink(fit_crops(np.asarray(crops[:: args.stride]), "round")))
    if not masks:
        return None
    m = np.concatenate(masks)
    x = m[:, ::2, ::2].reshape(len(m), -1).astype(np.float32)  # half resolution is plenty to tell counts apart
    _, assign = kmeans2(x, k, seed=args.seed, minit="++", iter=30)
    centres = np.stack([m[assign == j].mean(0) if (assign == j).any() else np.zeros(m.shape[1:]) for j in range(k)])
    return centres, np.bincount(assign, minlength=k)


def _tally_sheet(path: Path, centres) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    sheet = np.concatenate([np.pad(c, 3, constant_values=0.5) for c in centres], axis=1)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(path)


def tally_strokes(centres: np.ndarray, states: str) -> np.ndarray:
    """Cluster means labelled with how many strokes they show -> (10, h, w) uint8 mask per stroke.

    Stroke n is what state n has that state n-1 lacks. Rounds 1-10 are two groups of four verticals and a
    diagonal, the second group the first one shifted right; the shift is measured by sliding stroke 1 onto
    stroke 6, and strokes the recordings never showed (9, 10) are the first group's shifted and clipped."""
    from scipy import ndimage

    def parts(m, min_px):
        lab, n = ndimage.label(m)
        if n == 0:
            return m
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        if min_px is None:  # the largest part: a vertical stroke is one piece
            return lab == (np.argmax(sizes) + 1)
        return np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))  # the diagonal is cut up by the verticals

    shape = centres.shape[1:]
    state = {0: np.zeros(shape, bool)}
    for n in sorted({int(c) for c in states if c.isdigit()}):
        state[n] = centres[[i for i, c in enumerate(states) if c == str(n)]].mean(0) > 0.5
    strokes = np.zeros((10, *shape), bool)
    for n in range(1, 11):
        if n in state and n - 1 in state:
            strokes[n - 1] = parts(state[n] & ~state[n - 1], 20 if n in (5, 10) else None)
    if not strokes[5].any():
        raise SystemExit("tally: no cluster shows 6 strokes, so the second group's offset cannot be measured")
    shift = max(range(60, shape[1]), key=lambda o: (strokes[0][:, : shape[1] - o] & strokes[5][:, o:]).sum())
    for n in range(6, 11):
        if not strokes[n - 1].any():
            strokes[n - 1][:, shift:] = strokes[n - 6][:, : shape[1] - shift]
    print(f"tally: second group {shift} px right of the first; strokes (px): {strokes.sum((1, 2)).tolist()}")
    return strokes.astype(np.uint8)


def do_build(args) -> None:
    clusters = np.load(Path(args.harvest) / "clusters.npz")
    labels = json.loads(Path(args.labels).read_text())
    sets = {}
    for name, chars in labels["sets"].items():
        centres = clusters[f"{name}__centres"]
        if len(chars) != len(centres):
            sys.exit(f"{name}: {len(chars)} labels for {len(centres)} clusters")
        keep = [i for i, c in enumerate(chars) if c != "-"]
        sets[name] = GlyphSet(centres[keep], np.array([chars[i] for i in keep]))
        n_digit = sum(chars[i] != REJECT for i in keep)
        print(f"{name}: {n_digit} digit templates over {sorted(set(chars) - {'-', REJECT})}, "
              f"{len(keep) - n_digit} reject")
    extra = {}
    if "tally" in labels:
        extra["tally__strokes"] = tally_strokes(clusters["tally__centres"].astype(np.float32), labels["tally"])
    save_atlas(args.out, sets, meta=json.loads(str(clusters["meta"])), extra=extra)
    print(f"wrote {args.out}")


def weapon_crops(paths: list[Path], stride: int):
    """points_ammo crops at the reference size from clips and sample_hud.py files -> (stack, source names)."""
    stacks, sources = [], []
    for path in paths:
        if path.suffix == ".npz":
            with np.load(path) as z:
                stacks.append(fit_crops(z["points_ammo"][::stride], "points_ammo"))
            sources.append(str(path))
            continue
        for clip_dir in clip_dirs([path]):
            crops = load_clip(clip_dir).hud("points_ammo")
            if crops is not None:
                stacks.append(fit_crops(np.asarray(crops[::stride]), "points_ammo"))
                sources.append(str(clip_dir))
    if not stacks:
        sys.exit("no points_ammo crops in those paths")
    return np.concatenate(stacks), sources


def do_harvest_weapons(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    crops, sources = weapon_crops(args.paths, args.stride)
    inked = name_ink(name_band(crops))
    keep = np.flatnonzero(inked.sum((1, 2)) >= NAME_MIN_INK)
    sig = signature(inked[keep])
    k = min(args.k, len(sig))
    centres, assign = kmeans2(sig.reshape(len(sig), -1), k, seed=args.seed, minit="++", iter=30)
    centres = centres.reshape(k, *sig.shape[1:])
    counts = np.bincount(assign, minlength=k)
    guess = []
    try:
        labels, _, _ = load_names(NAMES_PATH).classify(centres)
        guess = labels.tolist()
    except FileNotFoundError:
        pass
    (out / "names.txt").write_text(f"clusters: {k}\ncounts: {counts.tolist()}\nexisting templates say: {guess}\n")
    _name_sheet(out / "names.png", name_band(crops[keep]), assign, centres)
    np.savez_compressed(out / "names.npz", centres=centres.astype(np.float16), counts=counts,
                        meta=np.array(json.dumps({"sources": sources, "stride": args.stride, "seed": args.seed})))
    print(f"{len(crops)} crops, {len(sig)} with a name, in {k} clusters (sizes {counts.tolist()}) "
          f"-> {out / 'names.png'}")
    if guess:
        print(f"existing templates say: {guess}")


def _name_sheet(path: Path, bands, assign, centres, per_row: int = 6, scale: int = 2) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("  (no pillow: skipping the contact sheet; run with `uv run --with pillow`)")
        return
    rows = []
    for k in range(len(centres)):
        idx = np.flatnonzero(assign == k)
        rng = np.random.default_rng(k)
        members = rng.choice(idx, min(per_row - 1, len(idx)), replace=False) if len(idx) else []
        cells = [np.repeat((centres[k] * 255).clip(0, 255).astype(np.uint8)[..., None], 3, -1)]
        cells += [bands[i] for i in members]
        cells += [np.zeros_like(cells[0])] * (per_row - len(cells))
        rows.append(np.concatenate([np.pad(c, ((2, 2), (2, 2), (0, 0)), constant_values=128) for c in cells], 1))
    sheet = np.concatenate(rows)
    Image.fromarray(sheet).resize((sheet.shape[1] * scale, sheet.shape[0] * scale), Image.NEAREST).save(path)


def do_build_weapons(args) -> None:
    clusters = np.load(Path(args.harvest) / "names.npz")
    labels = json.loads(Path(args.labels).read_text())["clusters"]
    centres = clusters["centres"].astype(np.float32)
    if len(labels) != len(centres):
        sys.exit(f"{len(labels)} labels for {len(centres)} clusters")
    keep = [i for i, c in enumerate(labels) if c != "-"]
    save_names(args.out, centres[keep], [labels[i] for i in keep], meta=str(clusters["meta"]))
    order = [w.key for w in WEAPONS]
    known = sorted({labels[i] for i in keep} - {REJECT}, key=order.index)
    print(f"wrote {args.out}: {len(keep)} templates for {known}, {sum(labels[i] == REJECT for i in keep)} reject")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest", help="segment and cluster glyphs, dump contact sheets")
    h.add_argument("paths", type=Path, nargs="+", help="clip directories, or directories holding clips")
    h.add_argument("--out", required=True)
    h.add_argument("--stride", type=int, default=3, help="use every n-th step")
    h.add_argument("--k", type=int, default=0, help="clusters per glyph set (default per set)")
    h.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("build", help="write the atlas from labelled clusters")
    b.add_argument("harvest", help="the --out directory of a harvest")
    b.add_argument("labels", help="JSON: {'sets': {set: one char per cluster}}")
    b.add_argument("--out", default=str(ATLAS_PATH))
    hw = sub.add_parser("harvest-weapons", help="cluster weapon-name bands, dump a contact sheet")
    hw.add_argument("paths", type=Path, nargs="+",
                    help="clip directories (or directories holding clips) and sample_hud.py .npz files")
    hw.add_argument("--out", required=True)
    hw.add_argument("--stride", type=int, default=1, help="use every n-th crop")
    hw.add_argument("--k", type=int, default=12, help="clusters")
    hw.add_argument("--seed", type=int, default=0)
    bw = sub.add_parser("build-weapons", help="write the weapon-name templates from labelled clusters")
    bw.add_argument("harvest", help="the --out directory of a harvest-weapons")
    bw.add_argument("labels", help="JSON: {'clusters': [a weapon key, '?' or '-' per cluster]}")
    bw.add_argument("--out", default=str(NAMES_PATH))
    args = parser.parse_args()
    {"harvest": do_harvest, "build": do_build, "harvest-weapons": do_harvest_weapons,
     "build-weapons": do_build_weapons}[args.cmd](args)


if __name__ == "__main__":
    main()
