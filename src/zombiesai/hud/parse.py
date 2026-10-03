"""One step's HUD crops -> numbers: points, round, grenades, reserve ammo, magazine, each with a confidence.

This is the per-frame reader. It never looks at other steps; `hud/track.py` is where temporal consistency
(plausible point deltas, the round only ever going up by one) accepts or rejects what it reads.

Every field has a status as well as a value, because "I can't read it" comes in kinds the tracker treats
differently:

    OK          read, with a confidence
    ABSENT      nothing is drawn there: the ammo counter fades out a few seconds after the last shot,
                the round counter is blank between a round's end and the next one's first tally, menus
                and black frames have no HUD at all
    UNREADABLE  something is there and it is not a confident read: a "+N" popup or muzzle flash over
                the digits, a bright wall behind the ammo, a half-faded glyph, round numerals (round 11+
                has none in the recordings yet, so the atlas cannot read them)
    TRANSITION  round only: the tallies flash white at a round change (the count is still readable)

Round is counted in tally strokes, not matched as glyphs: WaW draws rounds 1-10 as chalk tallies (groups
of four verticals and a diagonal) at fixed places. The strokes come from real frames (`glyph_atlas.npz`,
"tally" set) and each is judged present when it is solid red (normal) or clearly brighter than the scene
around it (the round-change flash, white or pink or fading). The ninth stroke would sit just right of the
round crop's edge, so round 9 looks like round 8 here: `round` is then 8 with `ROUND_CAPPED` set, and
the tracker counts round changes to tell them apart.
"""

from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np
from scipy import ndimage

from zombiesai.demos.frames import area_resize
from zombiesai.hud.glyphs import (
    ATLAS_PATH,
    LINES,
    REFERENCE_SHAPES,
    REJECT,
    GlyphSet,
    TextLine,
    glyph_runs,
    ink,
    load_atlas,
    patch,
)

OK, ABSENT, UNREADABLE, TRANSITION = 0, 1, 2, 3
STATUS_NAMES = ("ok", "absent", "unreadable", "transition")

# A glyph whose best template is further than this (mean squared difference of 0..1 ink) is not a digit
# of this font; within it, confidence falls linearly. Measured: harvested digits sit at 0.001-0.01.
GLYPH_MAX_DIST = 0.04
# How much closer the best character must be than the runner-up for full confidence.
GLYPH_FULL_MARGIN = 0.02
MIN_CONFIDENCE = 0.5
MAX_GAP = 7  # columns between two digits of one number: 1-3, but up to 6 either side of a narrow "1"
NEIGHBOUR_GAP = 14  # columns: a blob this close to a number on an unanchored line makes it unreadable
# The points line is left-aligned: its first digit starts in these columns of the crop.
POINTS_LEFT = (41, 48)


def fit_crops(crops: np.ndarray, region: str) -> np.ndarray:
    """Crops of `region` at the reference size the line geometry was measured at (a no-op for 1440p)."""
    h, w = REFERENCE_SHAPES[region]
    if crops.shape[-3:-1] == (h, w):
        return crops
    flat = crops.reshape(-1, *crops.shape[-3:])
    return np.stack([area_resize(c, h, w) for c in flat]).reshape(*crops.shape[:-3], h, w, 3)


@dataclass
class NumberRead:
    value: int = -1
    confidence: float = 0.0
    status: int = ABSENT
    start: int = -1  # ink-map column the first digit starts at (for the ammo lines, where it varies)
    end: int = -1  # and the column after the last digit


def _glyph_confidence(dist: np.ndarray, margin: np.ndarray) -> np.ndarray:
    return np.clip(1 - dist / GLYPH_MAX_DIST, 0, 1) * np.clip(margin / GLYPH_FULL_MARGIN, 0, 1)


def _read_runs(inked: np.ndarray, spans: list[tuple[int, int]], line: TextLine, gs: GlyphSet):
    """Classify each run as one glyph, or as two when it is too wide for one (digits touching)."""
    chars, confs = [], []
    single = [(a, b) for a, b in spans if b - a < line.width]
    wide = [(a, b) for a, b in spans if b - a >= line.width]
    if single:
        labels, dist, margin = gs.classify(np.stack([patch(inked, a, b, line.width) for a, b in single]))
        got = dict(zip(single, zip(labels.tolist(), _glyph_confidence(dist, margin).tolist())))
    else:
        got = {}
    for a, b in wide:
        if b - a > 2 * line.width:
            got[(a, b)] = (REJECT, 0.0)
            continue
        cuts = list(range(a + 3, b - 2))
        pats = [patch(inked, a, c, line.width) for c in cuts] + [patch(inked, c, b, line.width) for c in cuts]
        labels, dist, margin = gs.classify(np.stack(pats))
        n = len(cuts)
        total = dist[:n] + dist[n:]
        k = int(np.argmin(total))
        conf = _glyph_confidence(np.array([dist[k], dist[n + k]]), np.array([margin[k], margin[n + k]]))
        if (a, b) == spans[-1] and labels[n + k] == REJECT and labels[k] != REJECT:
            # The last digit with scene showing through the brush right after it (the brush is ragged
            # there): the digit is drawn over everything, so what follows it is not part of the number.
            got[(a, b)] = (labels[k], float(conf[0]))
            continue
        got[(a, b)] = (labels[k] + labels[n + k], float(conf.min()))
    for span in spans:
        c, p = got[span]
        chars.append(c)
        confs.append(p)
    return chars, confs


def read_number(inked: np.ndarray, line: TextLine, gs: GlyphSet, left: tuple[int, int] | None = None) -> NumberRead:
    """One line's ink map -> the number drawn in it. `left`, if given, is where the number must start (in
    the ink map's columns); otherwise the line must hold exactly one group of digits."""
    spans = glyph_runs(inked, line)
    if not spans:
        return NumberRead(status=ABSENT)
    if left is not None:  # nothing left of where the number starts belongs to it (popups, scene)
        spans = [s for s in spans if s[0] >= left[0]]
        if not spans:
            return NumberRead(status=UNREADABLE)
    groups = [[spans[0]]]
    for a, b in spans[1:]:
        if a - groups[-1][-1][1] <= MAX_GAP:
            groups[-1].append((a, b))
        else:
            groups.append([(a, b)])
    if left is not None:
        groups = [g for g in groups if left[0] <= g[0][0] < left[1]]
        if not groups:  # ink somewhere, but not where the number starts: covered, or not the HUD at all
            return NumberRead(status=UNREADABLE)
    else:
        # Scene edges and tick marks make small stray groups; the number is the group that reads best.
        groups = [g for g in groups if sum(b - a for a, b in g) >= 3]
        if not groups:
            return NumberRead(status=ABSENT)
    best = None
    for g in groups:
        chars, confs = _read_runs(inked, g, line, gs)
        text = "".join(chars)
        conf = min(confs)
        ok = REJECT not in text and 0 < len(text) <= 7 and not (len(text) > 1 and text[0] == "0")
        score = (ok, conf if ok else 0.0)
        if best is None or score > best[0]:
            best = (score, text, conf, g[0][0], g[-1][1])
        elif ok and best[0][0] and conf >= MIN_CONFIDENCE and best[2] >= MIN_CONFIDENCE:
            return NumberRead(status=UNREADABLE)  # two confident numbers on one line: not the HUD
    (ok, _), text, conf, start, end = best
    if not ok:
        return NumberRead(status=UNREADABLE, start=start)
    if left is None and len(groups) > 1:
        # Another blob right beside the number may be one of its digits, split off by the scene: reading
        # "28" as "2" is worse than not reading it.
        chosen = next(g for g in groups if g[0][0] == start)
        for g in groups:
            if g is not chosen and (0 < chosen[0][0] - g[-1][1] <= NEIGHBOUR_GAP or 0 < g[0][0] - chosen[-1][1] <= NEIGHBOUR_GAP):
                return NumberRead(status=UNREADABLE, start=start)
    if conf < MIN_CONFIDENCE:
        return NumberRead(-1, float(conf), UNREADABLE, start, end)
    return NumberRead(int(text), float(conf), OK, start, end)


def band_of(crop: np.ndarray, line: TextLine) -> np.ndarray:
    return crop[line.rows[0] : line.rows[1], line.cols[0] : line.cols[1]]


# Over a wall as pale as the text the top-hat cannot tell a digit from the wall, so a digit there vanishes
# rather than misreads -- and "150" with its "50" gone is still a confident "1". So the columns either
# side of an ammo number, where a lost digit would be, must not be text-coloured.
WASHED_COLS = 9
WASHED_LEVEL = 150  # red channel of grey scene that is as bright as the text (~198)
WASHED_SHARE = 0.25


def _unless_washed_out(read: NumberRead, band: np.ndarray) -> NumberRead:
    if read.status != OK:
        return read
    x = band.astype(np.int16)
    pale = (x[..., 0] >= WASHED_LEVEL) & (np.abs(x[..., 1] - x[..., 2]) <= 12)
    for a, b in ((read.end, read.end + WASHED_COLS), (read.start - WASHED_COLS, read.start)):
        a, b = max(0, a), min(band.shape[1], b)
        if b - a >= 3 and pale[:, a:b].mean() > WASHED_SHARE:
            return NumberRead(status=UNREADABLE, start=read.start, end=read.end)
    return read


# A "+N" popup is yellow, (255, 255, 0), so it has no ink in the points' rule -- which also means a popup
# drifting over the number's last digits would silently shorten it ("940" read as "94"). Popup yellow on
# the number, or where a further digit would be, makes it unreadable instead.
POPUP_COLS = 10
POPUP_PIXELS = 3


def _unless_popup_over(read: NumberRead, band: np.ndarray) -> NumberRead:
    if read.status != OK:
        return read
    x = band[:, max(0, read.start - 1) : read.end + POPUP_COLS].astype(np.int16)
    yellow = (x[..., 0] > 140) & (x[..., 1] > 140) & (x[..., 2] < 90)
    if np.count_nonzero(yellow) >= POPUP_PIXELS:
        return NumberRead(status=UNREADABLE, start=read.start, end=read.end)
    return read


# ------------------------------------------------------------------------------------------------ round

ROUND_CAPPED = 1  # flag: 8 visible strokes, which is also what round 9 looks like in this crop
N_STROKES = 10


@dataclass(frozen=True)
class _Strokes:
    idx: tuple  # per stroke: flat pixel indices inside the stroke
    ring: tuple  # per stroke: flat indices of the scene just around it
    union: np.ndarray  # flat indices of every stroke
    # Every pixel any stroke or ring reads (a fifth of the crop), and each stroke's and ring's indices into
    # that list: the evidence is computed on those pixels alone.
    used: np.ndarray
    used_idx: tuple
    used_ring: tuple


@lru_cache(maxsize=4)
def _strokes(path=ATLAS_PATH) -> _Strokes:
    with np.load(path) as z:
        masks = z["tally__strokes"].astype(bool)
    allm = masks.any(0)
    near = ndimage.binary_dilation(allm, iterations=2)
    idx, ring = [], []
    for m in masks:
        idx.append(np.flatnonzero(m))
        around = ndimage.binary_dilation(m, iterations=6) & ~near
        ring.append(np.flatnonzero(around))
    used = np.unique(np.concatenate(idx + ring))
    return _Strokes(tuple(idx), tuple(ring), np.flatnonzero(allm), used,
                    tuple(np.searchsorted(used, i) for i in idx), tuple(np.searchsorted(used, r) for r in ring))


# Stroke judgement, per stroke. Red ink is the tally's own colour (~(108, 1, 0)); nothing in Nacht's scene
# is that saturated. Contrast is mean brightness inside the stroke minus the scene around it.
RED_PRESENT, RED_ABSENT = 0.5, 0.12
CONTRAST_PRESENT, CONTRAST_ABSENT = 35.0, 14.0


def stroke_evidence(crop: np.ndarray, strokes: _Strokes | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(red fraction, brightness contrast) of each tally stroke in one round crop.

    Only the pixels a stroke or ring reads are converted. The means are `ndarray.mean` without its wrapper:
    the same reduction over the same gathered values, divided by the same count, so the same bits."""
    s = strokes or _strokes()
    x = crop.reshape(-1, 3)[s.used].astype(np.int16)
    red = (x[:, 0] > 60) & (x[:, 1] < 30) & (x[:, 2] < 30)
    lum = x.sum(1) * (1 / 3)
    reds = np.zeros(N_STROKES)
    contrast = np.zeros(N_STROKES)
    add = np.add.reduce
    for k in range(N_STROKES):
        idx, ring = s.used_idx[k], s.used_ring[k]
        if len(idx) == 0:
            reds[k], contrast[k] = np.nan, np.nan  # outside the crop: unknowable
            continue
        reds[k] = np.count_nonzero(red[idx]) / len(idx)
        contrast[k] = add(lum[idx]) / len(idx) - (add(lum[ring]) / len(ring) if len(ring) else 0.0)
    return reds, contrast


@dataclass
class RoundRead:
    value: int = -1
    confidence: float = 0.0
    status: int = ABSENT
    flags: int = 0


def _prefix(on: np.ndarray, known: np.ndarray) -> int:
    """How many strokes from the first are on (strokes the crop cannot show are skipped over, but never
    end the count)."""
    n = 0
    while n < N_STROKES and (not known[n] or on[n]):
        n += 1
    while n > 0 and not known[n - 1]:
        n -= 1
    return n


def read_round(crop: np.ndarray) -> RoundRead:
    reds, contrast = stroke_evidence(crop)
    known = ~np.isnan(reds)
    reds = np.where(known, reds, 0.0)
    contrast = np.where(known, contrast, 0.0)
    red_on, red_off = reds >= RED_PRESENT, reds <= RED_ABSENT

    def rest_after(n: int) -> np.ndarray:
        return known & (np.arange(N_STROKES) >= n)

    sure_red = np.clip((reds - RED_PRESENT) / 0.3 + 0.5, 0, 1)
    sure_not_red = np.clip((RED_ABSENT - reds) / 0.1 + 0.5, 0, 1)

    # Normal play: strokes 1..n solid red and nothing red after them. Red is the tally's own colour, so the
    # scene behind the missing strokes only has to not be red -- its texture does not matter.
    n = _prefix(red_on, known)
    if n and red_off[rest_after(n)].all():
        conf = float(min(sure_red[:n][known[:n]].min(), sure_not_red[rest_after(n)].min(initial=1.0)))
        return RoundRead(n, conf, OK, ROUND_CAPPED if n == 8 else 0)

    # The round-change flash: strokes white, pink or fading (and a newly added one red). Present means red
    # or clearly brighter than the scene around it; absent means neither.
    lit = contrast >= CONTRAST_PRESENT
    present = red_on | lit
    absent = red_off & (contrast <= CONTRAST_ABSENT)
    if (absent | ~known).all():
        return RoundRead(status=ABSENT, confidence=float(sure_not_red[known].min(initial=1.0)))
    n = _prefix(present, known)
    if n == 0 or not absent[rest_after(n)].all():
        return RoundRead(status=UNREADABLE)
    sure_on = np.maximum(sure_red, np.clip((contrast - CONTRAST_PRESENT) / 30 + 0.5, 0, 1))
    sure_off = np.minimum(sure_not_red, np.clip((CONTRAST_ABSENT - contrast) / 10 + 0.5, 0, 1))
    conf = float(min(sure_on[:n][known[:n]].min(), sure_off[rest_after(n)].min(initial=1.0)))
    return RoundRead(n, conf, TRANSITION, ROUND_CAPPED if n == 8 else 0)


# ------------------------------------------------------------------------------------------------ magazine

# The magazine is drawn as a row of marks left of the reserve: thin ticks two columns apart for most guns,
# dashes for the Kar98k and the shotguns. Loaded rounds are bright, spent ones dim, and spending goes from
# the left, so the magazine is the count of bright marks. Their right end is fixed near column 73; a long
# magazine (the Kar98k's fifth dash, a box gun's belt) runs off the crop's left edge, and then the count is
# only a lower bound unless the leftmost visible mark is already spent.
MAG_CORE = (128, 130)  # rows through the marks' bright core
MAG_BG = ((124, 126), (132, 134))  # rows above and below them: the scene behind
MAG_COLS = (0, 76)
MAG_BRIGHT = 42.0  # a loaded mark stands this far above the scene (ticks 50-80, dashes ~140; spent 12-30)
MAG_AMBIGUOUS = (34.0, 42.0)  # neither clearly loaded nor clearly spent
MAG_TICK_PROMINENCE = 30.0  # a loaded tick over both neighbouring columns (measured 45-65; spent ticks 10-20)
MAG_DASH = 6  # a run this many columns wide is a dash, not a tick
MAG_TICK_GAP = 4  # columns between two loaded ticks: they sit two apart, and one faint tick may be missed
MAG_DASH_GAP = 8
MAG_PITCH = 2
MAG_END = (69, 75)  # where the chain's right end lies (exclusive), whatever the gun: measured 71-72


def read_mag(crop: np.ndarray) -> NumberRead:
    """Bright magazine marks in a points_ammo crop (only meaningful while the ammo counter is drawn)."""
    x0, x1 = MAG_COLS
    lum = crop[:, x0:x1, 0].astype(np.float32)  # red: loaded marks are white, or red when ammo is low
    core = lum[MAG_CORE[0] : MAG_CORE[1]].mean(0)
    bg = np.median(np.concatenate([lum[a:b] for a, b in MAG_BG]), axis=0)
    c = core - bg
    bright = c >= MAG_BRIGHT
    spans = list(_runs(bright))
    dashes = [(a, b) for a, b in spans if b - a >= MAG_DASH]
    if dashes:
        marks, max_gap = dashes, MAG_DASH_GAP
    else:
        left = np.concatenate(([np.inf], c[:-1]))
        right = np.concatenate((c[1:], [np.inf]))
        # A loaded tick against the gap columns either side of it: the second difference along the row,
        # which a pale or textured wall behind the ticks mostly cancels (the rows above and below don't).
        prom = c - np.maximum(left, right)
        peak = (prom >= MAG_TICK_PROMINENCE) & (c >= left) & (c > right)
        marks, max_gap = [(int(i), int(i) + 1) for i in np.flatnonzero(peak)], MAG_TICK_GAP
    if not marks:
        return NumberRead(0, 1.0 if not (c[MAG_END[0] - 8 : MAG_END[1]] >= MAG_AMBIGUOUS[0]).any() else 0.4,
                          OK, -1)
    # The loaded marks are one evenly spaced chain ending at the right end; anything bright further left,
    # past a gap, is scene.
    if not MAG_END[0] <= marks[-1][1] <= MAG_END[1]:
        return NumberRead(status=UNREADABLE)
    chain = [marks[-1]]
    for m in reversed(marks[:-1]):
        if chain[-1][0] - m[1] > max_gap:
            break
        chain.append(m)
    lo = chain[-1][0]
    if lo <= 1:  # runs off the crop's edge: more loaded rounds than can be seen
        return NumberRead(status=UNREADABLE)
    # Ticks are counted by the chain's length at their fixed pitch, so a tick the peak finder missed in
    # the middle does not drop the count.
    n = len(chain) if max_gap == MAG_DASH_GAP else (chain[0][1] - 1 - lo) // MAG_PITCH + 1
    seg = slice(lo, MAG_END[1])
    ambiguous = np.count_nonzero((c[seg] >= MAG_AMBIGUOUS[0]) & (c[seg] < MAG_AMBIGUOUS[1]) & ~_near(bright, 1)[seg])
    conf = 1.0 / (1.0 + 0.5 * ambiguous)
    if conf < MIN_CONFIDENCE:
        return NumberRead(-1, conf, UNREADABLE, lo)
    return NumberRead(int(n), conf, OK, lo)


def _runs(on: np.ndarray):
    d = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    return zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist())


def _near(mask: np.ndarray, k: int) -> np.ndarray:
    out = mask.copy()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


@dataclass
class HudReading:
    """Everything read from one step. -1 is "no value"; see the status constants."""

    points: int = -1
    points_conf: float = 0.0
    points_status: int = ABSENT
    round: int = -1
    round_conf: float = 0.0
    round_status: int = ABSENT
    round_flags: int = 0
    grenades: int = -1
    grenades_conf: float = 0.0
    grenades_status: int = ABSENT
    reserve: int = -1
    reserve_conf: float = 0.0
    reserve_status: int = ABSENT
    reserve_low: bool = False  # drawn red: the game thinks ammo is low
    mag: int = -1
    mag_conf: float = 0.0
    mag_status: int = ABSENT

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


class HudParser:
    """Reads HUD crops, one step at a time (`parse`) or a clip's worth (`parse_many`)."""

    def __init__(self, atlas_path=ATLAS_PATH):
        atlas = load_atlas(atlas_path)
        self.points_glyphs = atlas["points"]
        self.ammo_glyphs = atlas["ammo"]
        _strokes(atlas_path)

    def parse(self, crops: dict[str, np.ndarray]) -> HudReading:
        out = HudReading()
        pa = crops.get("points_ammo")
        if pa is not None:
            pa = fit_crops(np.asarray(pa), "points_ammo")
            self._points_ammo(pa, out)
        rd = crops.get("round")
        if rd is not None:
            r = read_round(fit_crops(np.asarray(rd), "round"))
            out.round, out.round_conf, out.round_status, out.round_flags = r.value, r.confidence, r.status, r.flags
        return out

    def _points_ammo(self, crop: np.ndarray, out: HudReading) -> None:
        line = LINES["points"]
        x0 = line.cols[0]
        inked = ink(crop[line.rows[0] : line.rows[1], line.cols[0] : line.cols[1]], line.ink)
        p = _unless_popup_over(read_number(inked, line, self.points_glyphs, (POINTS_LEFT[0] - x0, POINTS_LEFT[1] - x0)),
                               band_of(crop, line))
        out.points, out.points_conf, out.points_status = p.value, p.confidence, p.status

        line = LINES["grenades"]
        inked = ink(band_of(crop, line), line.ink)
        g = _unless_washed_out(read_number(inked, line, self.ammo_glyphs), band_of(crop, line))
        if g.status == OK and g.value > 9:
            g = NumberRead(status=UNREADABLE)
        out.grenades, out.grenades_conf, out.grenades_status = g.value, g.confidence, g.status

        line = LINES["reserve"]
        band = crop[line.rows[0] : line.rows[1], line.cols[0] : line.cols[1]]
        inked = ink(band, line.ink)
        r = _unless_washed_out(read_number(inked, line, self.ammo_glyphs), band)
        out.reserve, out.reserve_conf, out.reserve_status = r.value, r.confidence, r.status
        if r.status == OK:
            on = inked > 0.5
            px = band[on].astype(np.int16)
            out.reserve_low = bool(len(px) and np.median(px[:, 0] - px[:, 1]) > 80)
            m = read_mag(crop)
            out.mag, out.mag_conf, out.mag_status = m.value, m.confidence, m.status
        elif r.status == UNREADABLE:
            out.mag_status = UNREADABLE

    def parse_many(self, crops: dict[str, np.ndarray], batch: int = 450) -> dict[str, np.ndarray]:
        """Parse every step of (T, h, w, 3) crop stacks (arrays, memmaps or HudVideo) into per-field arrays."""
        n = min(len(v) for v in crops.values())
        names = HudReading.field_names()
        cols = {k: [] for k in names}
        for start in range(0, n, batch):
            chunk = {k: np.asarray(v[start : min(n, start + batch)]) for k, v in crops.items()}
            for i in range(len(next(iter(chunk.values())))):
                r = self.parse({k: v[i] for k, v in chunk.items()})
                for k in names:
                    cols[k].append(getattr(r, k))
        return {k: _as_array(k, v) for k, v in cols.items()}


def _as_array(name: str, values: list) -> np.ndarray:
    if name.endswith("_conf"):
        return np.asarray(values, np.float32)
    if name.endswith("_status") or name.endswith("_flags"):
        return np.asarray(values, np.int8)
    if name == "reserve_low":
        return np.asarray(values, bool)
    return np.asarray(values, np.int32)
