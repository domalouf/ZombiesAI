"""The round, read from the 128x72 policy frames alone -- for recordings made before HUD crops existed.

The tallies are the one HUD element that survives the downsample: a stroke is under a pixel wide at 128x72
but pure red, (108, 1, 0), which nothing else in Nacht is. The round crop's box starts exactly on a policy
pixel boundary (rows 58-72, columns 0-16 of the frame; each policy pixel is 10x10 crop pixels), so the
atlas's stroke masks, block-averaged, say how much red each pixel should hold at each count. The count is
the one whose coverage map fits the frame's redness best. Measured against the full-resolution reader on
demo_0001 and demo_0002: the same count on all 16,217 steps both call readable (it calls ~80% readable), and
the same round changes, less the ninth.

No points, no ammo (a smear of a few pixels), and no round-change flash (white tallies are not red), so
round 9 cannot be told from round 8 here and a game's end is only visible as the tallies resetting.
"""

import numpy as np

from zombiesai.hud.glyphs import ATLAS_PATH
from zombiesai.hud.parse import ABSENT, OK, ROUND_CAPPED, UNREADABLE

ROWS, COLS = (50, 72), (0, 24)  # the policy-frame window searched: the round box (58-72, 0-16) and a margin
BOX_TOP = 58  # policy row the round crop's top sits on
BLOCK = 10
RED = 105.0  # R - max(G, B) of a solid stroke
MIN_MARGIN = 0.6  # squared-error gap to the next best count
MAX_RESIDUAL = 5.0
COUNTS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)  # 9 looks exactly like 8: its stroke is off the crop


def coverage_maps(offset: tuple[int, int] = (0, 0), path=ATLAS_PATH) -> np.ndarray:
    """(len(COUNTS), rows, cols) expected red coverage of each policy pixel in the window, with the tallies
    moved by `offset` (dy, dx) crop pixels from where the 2560x1440 recordings have them."""
    with np.load(path) as z:
        strokes = z["tally__strokes"].astype(np.float32)
    h, w = (ROWS[1] - ROWS[0]) * BLOCK, (COLS[1] - COLS[0]) * BLOCK
    maps = []
    for n in COUNTS:
        m = strokes[:n].max(0) if n else np.zeros(strokes.shape[1:], np.float32)
        canvas = np.zeros((h, w), np.float32)
        y0, x0 = (BOX_TOP - ROWS[0]) * BLOCK + offset[0], -COLS[0] * BLOCK + offset[1]
        ys, xs = slice(max(0, y0), min(h, y0 + m.shape[0])), slice(max(0, x0), min(w, x0 + m.shape[1]))
        canvas[ys, xs] = m[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
        maps.append(canvas.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean((1, 3)))
    return np.stack(maps)


def redness(frames: np.ndarray) -> np.ndarray:
    f = np.asarray(frames[:, ROWS[0] : ROWS[1], COLS[0] : COLS[1]]).astype(np.float32)
    return np.clip((f[..., 0] - np.maximum(f[..., 1], f[..., 2])) / RED, 0, 1)


def find_offset(frames: np.ndarray, search: range = range(-20, 44, 2), stride: int = 10) -> tuple[int, int]:
    """The tally offset that best explains a clip's frames. demo_0000, recorded before the HUD boxes were
    fixed, has its tallies 20 crop px lower and 22 further right than every later recording."""
    red = redness(frames[::stride])
    red = red[red.sum((1, 2)) > 0.5]
    if not len(red):
        return (0, 0)
    best = None
    for dy in search:
        for dx in search:
            maps = coverage_maps((dy, dx))
            err = ((red[:, None] - maps[None]) ** 2).sum((2, 3)).min(1).mean()
            if best is None or err < best[0]:
                best = (err, (dy, dx))
    return best[1]


def round_from_frames(frames: np.ndarray, offset: tuple[int, int] = (0, 0), batch: int = 4096) -> dict[str, np.ndarray]:
    """(T, 72, 128, 3) policy frames -> {"round", "round_conf", "round_status", "round_flags"}."""
    maps = coverage_maps(offset)
    n = len(frames)
    value = np.full(n, -1, np.int32)
    conf = np.zeros(n, np.float32)
    status = np.full(n, UNREADABLE, np.int8)
    for a in range(0, n, batch):
        red = redness(frames[a : a + batch])
        d = ((red[:, None] - maps[None]) ** 2).sum((2, 3))
        order = np.argsort(d, axis=1)
        rows = np.arange(len(d))
        best, second = d[rows, order[:, 0]], d[rows, order[:, 1]]
        margin = second - best
        good = (margin >= MIN_MARGIN) & (best <= MAX_RESIDUAL)
        counts = np.array(COUNTS)[order[:, 0]]
        c = np.clip(margin / (2 * MIN_MARGIN), 0, 1) * np.clip(1 - best / MAX_RESIDUAL, 0, 1)
        sl = slice(a, a + len(d))
        value[sl] = np.where(good & (counts > 0), counts, -1)
        status[sl] = np.where(good, np.where(counts > 0, OK, ABSENT), UNREADABLE)
        conf[sl] = c
    flags = np.where(value == 8, ROUND_CAPPED, 0).astype(np.int8)
    return {"round": value, "round_conf": conf, "round_status": status, "round_flags": flags}
