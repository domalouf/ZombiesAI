"""Digit glyphs: where each number sits in its HUD crop, how its pixels become a glyph mask, and the atlas.

WaW draws its numbers in a fixed font at fixed positions, so a digit is recognised by comparing it with
glyphs harvested from real frames rather than by generic OCR (PLAN.md, "Perception"). Per text line:

    crop rows -> ink map (a colour rule per line, 0..1) -> columns holding ink -> runs of columns
    (one run per glyph; the font leaves a gap between digits) -> each run cut out as a fixed-size patch,
    left-aligned at its first column, zero outside the run -> nearest template in the atlas.

The ink rules are what make the backgrounds go away. The points sit on an opaque red brush stroke
(~(105, 3, 3)) with white digits: min(G, B) is ~255 on a digit, ~3 on the brush and ~0 under the yellow
"+N" popups, which float over the points but are (255, 255, 0). The grenade count and reserve ammo are
drawn straight on the scene in light grey, or red when low: both have G == B, which brown walls, blue
stone and yellow muzzle flash do not, so the ink is R where |G - B| is small.

The atlas (`glyph_atlas.npz` beside this file) holds, per glyph set, a few templates per character --
cluster centres of harvested patches, so the sub-pixel phases a 0.5x downsample produces are each
represented -- plus reject templates ("?") for things that segment like digits and are not (scene
texture, a half-faded glyph). `scripts/build_hud_atlas.py` rebuilds it; docs/hud.md says how.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

ATLAS_PATH = Path(__file__).with_name("glyph_atlas.npz")
REJECT = "?"
TOPHAT = 5  # px: the opening that separates strokes from the scene behind the ammo counter
PATCH_FLOOR = 0.3  # patches fainter than this are not scaled up: they are noise, not a faded glyph


@dataclass(frozen=True)
class TextLine:
    """One number's line in a HUD crop, in pixels of the reference crop (2560x1440 captured, 0.5 scale)."""

    name: str
    region: str  # which HUD crop
    rows: tuple[int, int]  # [top, bottom) of the glyph band
    cols: tuple[int, int]  # [left, right) searched for glyphs
    glyphs: str  # glyph set in the atlas
    ink: str  # "white_on_red" | "grey_or_red"
    width: int  # patch width: the widest glyph plus a column
    threshold: float = 0.45  # ink level a column needs to count as part of a glyph
    min_height: int = 1  # rows of ink a run needs, so tick marks and dashes are not glyphs


LINES = {
    "points": TextLine("points", "points_ammo", (23, 35), (30, 150), "points", "white_on_red", 10, min_height=8),
    "grenades": TextLine("grenades", "points_ammo", (103, 117), (52, 90), "ammo", "grey_or_red", 11,
                         threshold=0.5, min_height=8),
    "reserve": TextLine("reserve", "points_ammo", (121, 135), (40, 150), "ammo", "grey_or_red", 11,
                        threshold=0.5, min_height=8),
}
# The crops these rows and columns were measured on (demos/hud_crops.py at 2560x1440). Crops of another
# size are area-resized to these first.
REFERENCE_SHAPES = {"points_ammo": (135, 190), "round": (140, 160)}


def ink(band: np.ndarray, kind: str) -> np.ndarray:
    """(..., h, w, 3) uint8 -> (..., h, w) float32 in [0, 1]: how much each pixel looks like text."""
    x = band.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    if kind == "white_on_red":
        v = (np.minimum(g, b) - 20) * (1 / 200)
    elif kind == "grey_or_red":
        v = np.clip((r - 50) * (1 / 150) * (np.abs(g - b) <= 12), 0.0, 1.0).astype(np.float32)
        # Top-hat: keep only what is thinner than the opening. A stroke is 2-3 px wide; a pale wall behind
        # the counter is not, and without this it reads as ink and swallows the digits.
        size = (1,) * (v.ndim - 2) + (TOPHAT, TOPHAT)
        v = v - ndimage.grey_opening(v, size=size)
    else:
        raise ValueError(f"unknown ink {kind!r}")
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def runs(on: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True in a 1-D mask."""
    d = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def glyph_runs(inked: np.ndarray, line: TextLine) -> list[tuple[int, int]]:
    """Column runs of one line's ink map (h, w), in the map's own columns, that could be glyphs."""
    mask = inked > line.threshold
    cols = mask.any(axis=0)
    out = []
    for a, b in runs(cols):
        if line.min_height > 1 and np.count_nonzero(mask[:, a:b].any(axis=1)) < line.min_height:
            continue
        out.append((a, b))
    return out


def patch(inked: np.ndarray, start: int, end: int, width: int) -> np.ndarray:
    """A glyph's patch: `width` columns from `start`, ink outside [start, end) zeroed, scaled so its
    brightest pixel is 1 (a HUD fading out, or text over a pale wall, keeps its shape but not its level)."""
    h, w = inked.shape
    out = np.zeros((h, width), np.float32)
    n = max(0, min(end, w, start + width) - start)
    out[:, :n] = inked[:, start : start + n]
    out *= 1.0 / max(float(out.max()), PATCH_FLOOR)
    return out


@dataclass
class GlyphSet:
    """Templates for one font: (N, h, w) float32 patches and the character each one is."""

    templates: np.ndarray
    labels: np.ndarray  # (N,) '<U1'

    def __post_init__(self):
        self.templates = np.asarray(self.templates, np.float32)
        self.labels = np.asarray(self.labels, dtype="<U1")
        self._flat = self.templates.reshape(len(self.templates), -1)
        self._sq = (self._flat**2).sum(1)
        self._chars = sorted(set(self.labels.tolist()))
        self._char_index = np.array([self._chars.index(c) for c in self.labels])

    @property
    def shape(self) -> tuple[int, int]:
        return self.templates.shape[1:]

    def classify(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(M, h, w) patches -> (label, distance, margin) per patch.

        distance is the mean squared difference to the best template (0 = identical, ~0.25 = unrelated);
        margin is how much further the best template of any *other* character is: a confident read has a
        small distance and a wide margin."""
        m = len(patches)
        if m == 0:
            return np.zeros(0, "<U1"), np.zeros(0, np.float32), np.zeros(0, np.float32)
        x = patches.reshape(m, -1).astype(np.float32)
        d = ((x**2).sum(1)[:, None] - 2 * x @ self._flat.T + self._sq[None]) / x.shape[1]
        per_char = np.full((m, len(self._chars)), np.inf, np.float32)
        np.minimum.at(per_char.T, self._char_index, d.T)
        order = np.argsort(per_char, axis=1)
        best, second = order[:, 0], order[:, 1] if per_char.shape[1] > 1 else order[:, 0]
        rows = np.arange(m)
        dist = np.maximum(per_char[rows, best], 0.0)
        margin = per_char[rows, second] - per_char[rows, best]
        return np.array(self._chars)[best], dist.astype(np.float32), margin.astype(np.float32)


def save_atlas(path: str | Path, sets: dict[str, GlyphSet], meta: dict | None = None,
               extra: dict[str, np.ndarray] | None = None) -> None:
    """Glyph sets as <set>__templates/<set>__labels, plus `extra` arrays (the tally strokes) as they are."""
    arrays = dict(extra or {})
    for name, gs in sets.items():
        arrays[f"{name}__templates"] = gs.templates.astype(np.float16)
        arrays[f"{name}__labels"] = gs.labels
    if meta:
        import json

        arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


@lru_cache(maxsize=4)
def load_atlas(path: str | Path = ATLAS_PATH) -> dict[str, GlyphSet]:
    with np.load(path) as z:
        names = sorted(k.split("__")[0] for k in z.files if k.endswith("__templates"))
        return {n: GlyphSet(z[f"{n}__templates"].astype(np.float32), z[f"{n}__labels"]) for n in names}


def line_ink(crops: np.ndarray, line: TextLine) -> np.ndarray:
    """(..., H, W, 3) crops of the line's region -> (..., h, w) ink over the line's band."""
    (y0, y1), (x0, x1) = line.rows, line.cols
    return ink(crops[..., y0:y1, x0:x1, :], line.ink)


def harvest(crops: np.ndarray, line: TextLine, max_width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Every glyph-like run in a stack of crops, as patches -> ((M, h, width) patches, (M, 3) step/start/end).
    Runs wider than `max_width` (two glyphs touching, or not text at all) are left out."""
    max_width = max_width or line.width - 1
    inked = line_ink(np.asarray(crops), line)
    patches, where = [], []
    for i, m in enumerate(inked):
        for a, b in glyph_runs(m, line):
            if 3 <= b - a <= max_width:
                patches.append(patch(m, a, b, line.width))
                where.append((i, a, b))
    h = line.rows[1] - line.rows[0]
    return (np.array(patches, np.float32).reshape(-1, h, line.width), np.array(where, np.int32).reshape(-1, 3))
