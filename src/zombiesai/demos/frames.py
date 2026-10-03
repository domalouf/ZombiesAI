"""Turning real captured pixels into the spec's policy frames: area-average resize, bar removal, frame deltas.

Every path that ever feeds real footage to a network goes through `to_policy_frame`, so a recording made
today and a screen capture made in M5 are downsampled by exactly the same arithmetic.

**Whole multiples are exact, and cheap.** The game runs at 2560x1440, exactly 20 times the 128x72 policy
frame (1920x1080 is 15 times, 1280x720 10 times), and the HUD crops at 1440p are exactly half size. When the
source is a whole multiple of the target, every output pixel is the mean of a block of whole pixels: it is
summed in integers and rounded half to even, which is the true mean, the same on every CPU and BLAS. Those
sums run straight off the bytes X hands over (`channels=BGRX`), so the live capture never makes a full-frame
RGB copy or a float32 one: a 1440p frame costs ~2 ms where converting and resizing it cost ~21.

Anything else takes the general path: a separable float32 product with the exact overlap weights. It rounds a
float32 sum, so a block whose true mean is exactly half-way between two integers could come out either side
of it. That is the one way the two paths can differ, by one, and only where such ties exist at all: never for
an odd factor (15x15 blocks), for halving never (the float32 sums are exact there), and on random frames at
20x about one value in 2,000, at 10x one in 500 (tests/test_demo_frames.py measures it).
"""

import functools

import numpy as np
from scipy.sparse import csr_matrix

from zombiesai import spec

FRAME_H, FRAME_W = spec.PIXELS_SHAPE[:2]
ASPECT = FRAME_W / FRAME_H
# Below this max luminance a row or column is a letterbox bar, not dark gameplay. WaW's darkest interiors
# still sit well above it; encoder ringing around a hard black bar sits just under.
BAR_LUMA = 18.0
MIN_BAR_PX = 2
# Rec. 601 luma, the weighting every encoder in the chain already assumes.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

FITS = ("crop", "pad", "stretch")

# Where R, G and B sit in a pixel as X hands it over: B, G, R, X bytes (32-bit TrueColor, little-endian). The
# functions here that take `channels` read a frame in that layout as it is, without converting it first.
BGRX = slice(2, None, -1)


@functools.cache
def _axis_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) area-average weights: output pixel i is the mean over its exact input interval.

    Area averaging, not nearest: at ~15x reduction nearest-neighbour aliases thin limbs in and out between
    frames, injecting noise that looks exactly like motion (PLAN.md, "Observation").
    """
    if n_in < 1 or n_out < 1:
        raise ValueError(f"resize needs positive sizes, got {n_in} -> {n_out}")
    edges = np.linspace(0.0, n_in, n_out + 1)
    idx = np.arange(n_in)[None, :]
    overlap = np.clip(np.minimum(edges[1:, None], idx + 1) - np.maximum(edges[:-1, None], idx), 0.0, None)
    weights = overlap / overlap.sum(axis=1, keepdims=True)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    weights.setflags(write=False)
    return weights


@functools.cache
def _row_weights(n_in: int, n_out: int) -> csr_matrix:
    """The row pass as a sparse matrix. Each output row touches about `n_in / n_out` input rows, so the dense
    product spends 99% of its multiplies on zeros -- and at 1080p this pass is on the 66 ms tick budget."""
    return csr_matrix(_axis_weights(n_in, n_out))


def area_resize(image: np.ndarray, out_h: int, out_w: int, *, channels=None) -> np.ndarray:
    """Exact area-average resize to an (out_h, out_w, 3) uint8 RGB image. Identity when the size already matches.

    `image` is HxWx3 RGB, or any HxWxC layout with `channels` picking R, G and B out of it (`BGRX` for what X
    hands over); a strided view, such as a crop of a bigger frame, is read in place. A whole-multiple size is
    the integer path (`block_mean`), anything else the general one (see the module docstring).
    """
    if image.ndim != 3 or image.shape[2] < 3 or (channels is None and image.shape[2] != 3):
        raise ValueError(f"expected an HxWx3 image (or HxWxC with channels), got {image.shape}")
    h, w = image.shape[:2]
    if (h, w) == (out_h, out_w):
        return np.ascontiguousarray(image if channels is None else image[..., channels], dtype=np.uint8)
    if image.dtype == np.uint8 and 0 < out_h <= h and 0 < out_w <= w and h % out_h == 0 and w % out_w == 0:
        return block_mean(image, h // out_h, w // out_w, channels=channels)
    return _area_resize_float(image, out_h, out_w, channels)


def _area_resize_float(image: np.ndarray, out_h: int, out_w: int, channels=None) -> np.ndarray:
    """The general path, for any two sizes: HxWxC -> (out_h, out_w, 3), in float32 with the overlap weights.

    Rows are reduced first with the sparse operator and columns second with a dense one: by then the image
    is `out_h` rows tall and the dense product is small. Measured at about 3.7 ms for 1920x1080 -> 128x72,
    against 6.5 ms for the dense form, bit-for-bit identical. The row pass takes every channel as it lies (a
    BGRX crop converts to float in one contiguous sweep, where picking R, G, B first is a strided gather) and
    R, G, B are picked after it: each column of that product is computed alone, so this changes no value.
    """
    h, w, c = image.shape
    x = image.astype(np.float32).reshape(h, w * c)
    rows = (_row_weights(h, out_h) @ x).reshape(out_h, w, c)
    if channels is not None:
        rows = rows[..., channels]
    rows = rows.transpose(0, 2, 1).reshape(out_h * 3, w)
    if w == 2 * out_w:
        # Halving the width (the console crop at 1440p: 1946 columns to 973): each output column has two
        # weights of exactly 0.5 and ~1,900 exact zeros, so the dense product rounds exactly once, whatever
        # order its kernel adds in -- this is that product, bit for bit, without its multiplications by zero.
        out = (rows[:, 0::2] + rows[:, 1::2]) * np.float32(0.5)
    else:
        out = rows @ _axis_weights(w, out_w).T
    out = out.reshape(out_h, 3, out_w).transpose(0, 2, 1)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def block_mean(image: np.ndarray, fy: int, fx: int, *, channels=None) -> np.ndarray:
    """(H, W, C) uint8 -> (H/fy, W/fx, 3) uint8: the exact mean of each fy x fx block, rounded half to even.

    `channels` picks R, G and B out of C as in `area_resize`; H and W must be multiples of fy and fx. Only the
    block sums are ever materialised -- no copy of the image in any other layout or type.
    """
    h, w, c = image.shape
    if h % fy or w % fx:
        raise ValueError(f"{h}x{w} is not a whole number of {fy}x{fx} blocks")
    if channels is None and c != 3:
        raise ValueError(f"which of {c} channels are R, G and B? Pass channels")
    sums = _block_sums(image, fy, fx)
    if channels is not None:  # gathered into a contiguous array: the rounding is several passes over it
        sums = np.take(sums, np.arange(c)[channels], axis=-1)
    return _round_half_even(sums, fy * fx)


@functools.cache
def _lane_rows(fy: int, fx: int) -> int:
    """How many rows of a block can be summed before its columns are, with no 16-bit sum able to overflow
    (see `_block_sums`): the largest divisor of fy that keeps fx * rows * 255 within 65535, or 0 if none."""
    for rows in range(fy, 0, -1):
        if fy % rows == 0 and fx * rows * 255 <= 0xFFFF:
            return rows
    return 0


def _block_sums(image: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """(H, W, C) uint8 -> (H/fy, W/fx, C) uint16 or uint32 (`_sum_dtype`): each block's sum per channel, exactly.

    Rows are summed first: a block row's fy image rows are added as whole rows, which is one long contiguous
    reduction even when the image is a strided view into a bigger frame. Columns are then fx-wide groups of
    pixels. With four channels a pixel's four 16-bit row sums fill exactly one 64-bit word, so adding words
    adds all four channels at once -- as long as no 16-bit lane can carry into the next, which is why the rows
    are summed in chunks of `_lane_rows` (10 of a 20x20 block: 20 columns x 10 rows x 255 < 65536) and the
    chunks added after. At 2560x1440 -> 128x72 the whole thing is ~2 ms, nearly all of it the one pass over
    the 14.7 MB frame.
    """
    h, w, c = image.shape
    oh, ow = h // fy, w // fx
    rows = _lane_rows(fy, fx) if c == 4 else 0
    if rows:
        k = fy // rows
        partial = image.reshape(oh, k, rows, w, 4).sum(axis=2, dtype=np.uint16)  # (oh, k, w, 4)
        words = partial.view(np.uint64).reshape(oh, k, ow, fx)
        if fx <= 4:  # a few whole-array adds beat a reduction along a short axis
            total = words[..., 0].copy()
            for i in range(1, fx):
                total += words[..., i]
        else:
            total = words.sum(axis=-1)
        lanes = total.view(np.uint16).reshape(oh, k, ow, 4)
        return lanes[:, 0] if k == 1 else lanes.sum(axis=1, dtype=_sum_dtype(fy * fx))
    partial = image.reshape(oh, fy, w, c).sum(axis=1, dtype=_sum_dtype(fy))
    blocks = partial.reshape(oh, ow, fx, c)
    total = blocks[:, :, 0].astype(_sum_dtype(fy * fx))
    for i in range(1, fx):
        total += blocks[:, :, i]
    return total


def _sum_dtype(n: int):
    """The narrowest type a sum of n bytes fits with room to round it (sum + n/2): uint16 up to 256 terms. The
    rounding is several passes over the sums, and on 16-bit lanes they are a fraction of the cost."""
    return np.uint16 if n * 255 + n <= 0xFFFF else np.uint32


def _round_half_even(sums: np.ndarray, n: int) -> np.ndarray:
    """Non-negative integer `sums` / n, rounded to the nearest integer and half-way cases to the even one --
    np.rint's rule, applied to the exact quotient -- as uint8."""
    if n == 1:
        return sums.astype(np.uint8)
    if n & (n - 1) == 0:  # a power of two: (s + n/2 - 1 + [the quotient is odd]) >> log2(n)
        shift = n.bit_length() - 1
        return ((sums + ((n >> 1) - 1) + ((sums >> shift) & 1)) >> shift).astype(np.uint8)
    quotient, rest = np.divmod(sums, n)
    rest *= 2
    quotient += (rest > n) | ((rest == n) & (quotient & 1).astype(bool))
    return quotient.astype(np.uint8)


def luma(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) @ LUMA


def detect_bars(frames) -> tuple[int, int, int, int]:
    """Letterbox/pillarbox box as (top, bottom, left, right) insets, from the brightest value each row and
    column ever reaches. One bright frame is enough to rule a row out, which is what we want: a dark frame
    must never be allowed to widen the crop."""
    stack = np.asarray(frames, dtype=np.uint8)
    if stack.ndim == 3:
        stack = stack[None]
    bright = luma(stack).max(axis=0)
    rows = bright.max(axis=1) < BAR_LUMA
    cols = bright.max(axis=0) < BAR_LUMA
    return (*_edge_run(rows), *_edge_run(cols))


def _edge_run(dark: np.ndarray) -> tuple[int, int]:
    """Length of the dark run at each end of a 1-D mask, ignoring runs shorter than MIN_BAR_PX."""
    n = len(dark)
    lead = int(np.argmin(dark)) if dark.any() and not dark.all() else (n if dark.all() else 0)
    tail = int(np.argmin(dark[::-1])) if dark.any() and not dark.all() else 0
    if lead + tail >= n:  # an all-dark frame is not a crop, it is a black frame
        return 0, 0
    return (lead if lead >= MIN_BAR_PX else 0), (tail if tail >= MIN_BAR_PX else 0)


def crop_box(height: int, width: int, bars: tuple[int, int, int, int], fit: str) -> tuple[int, int, int, int]:
    """Source rectangle as (y, x, h, w): bars removed, then squared up to 16:9 if `fit` is "crop"."""
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    top, bottom, left, right = bars
    y, x = top, left
    h, w = height - top - bottom, width - left - right
    if h < 1 or w < 1:
        raise ValueError(f"bars {bars} leave nothing of a {height}x{width} frame")
    if fit == "crop":
        if w > h * ASPECT:  # wider than 16:9 (ultrawide): trim the sides
            new_w = int(round(h * ASPECT))
            x, w = x + (w - new_w) // 2, new_w
        elif w < h * ASPECT:  # narrower (4:3): trim top and bottom, which costs HUD -- see note in demos.md
            new_h = int(round(w / ASPECT))
            y, h = y + (h - new_h) // 2, new_h
    return y, x, h, w


def to_policy_frame(frame: np.ndarray, box: tuple[int, int, int, int] | None = None, fit: str = "crop", *,
                    channels=None) -> np.ndarray:
    """One captured frame as the agent's (72, 128, 3) uint8 observation.

    `box` is a source rectangle from `crop_box`; pass the one computed once per video rather than
    re-detecting bars per frame, so the geometry can't drift mid-clip. `frame` is RGB, or the BGRX X hands
    over with `channels=BGRX` -- read in place, so a view of the shared capture buffer is never copied.
    """
    frame = np.asarray(frame, dtype=np.uint8)
    if box is None:
        box = crop_box(*frame.shape[:2], detect_bars(frame if channels is None else frame[..., channels]), fit)
    y, x, h, w = box
    view = frame[y : y + h, x : x + w]
    if fit != "pad" or abs(w / h - ASPECT) < 1e-9:
        return area_resize(view, FRAME_H, FRAME_W, channels=channels)
    # Pad: keep the whole source frame (a 4:3 recording's HUD included) inside the 16:9 observation.
    if w < h * ASPECT:
        inner_w, inner_h = max(1, int(round(FRAME_H * w / h))), FRAME_H
    else:
        inner_w, inner_h = FRAME_W, max(1, int(round(FRAME_W * h / w)))
    out = np.zeros(spec.PIXELS_SHAPE, dtype=np.uint8)
    y0, x0 = (FRAME_H - inner_h) // 2, (FRAME_W - inner_w) // 2
    out[y0 : y0 + inner_h, x0 : x0 + inner_w] = area_resize(view, inner_h, inner_w, channels=channels)
    return out


def frame_delta(frames: np.ndarray) -> np.ndarray:
    """Mean absolute luma change between consecutive frames, in 0-255 units; delta[0] is 0 by convention.

    This one number separates the three things a raw recording is full of: gameplay (small), a cut or a
    loading screen (large), and a paused or frozen capture (zero).
    """
    stack = np.asarray(frames)
    if len(stack) < 2:
        return np.zeros(len(stack), dtype=np.float32)
    y = luma(stack)
    d = np.abs(np.diff(y, axis=0)).mean(axis=(1, 2))
    return np.concatenate(([0.0], d)).astype(np.float32)


def estimate_shift(before: np.ndarray, after: np.ndarray, max_shift: int = 24) -> tuple[int, float]:
    """Horizontal image motion from `before` to `after`, in pixels of the frame's own width, by SAD search.

    A positive shift means the scene moved right, i.e. the view turned left. It is the cheap optical-flow
    cross-check the plan asks for on demo labels: yaw that doesn't correlate with this is a timing bug.
    Returns (shift, confidence in [0, 1]) where confidence is how much the best shift beats the median.
    """
    a, b = luma(before), luma(after)
    h, w = a.shape
    margin = min(max_shift, w // 3)
    if margin < 1:
        return 0, 0.0
    band = slice(h // 6, h - h // 6)  # ignore the HUD strip and the ceiling, which are static under yaw
    core = a[band, margin : w - margin]
    shifts = np.arange(-margin, margin + 1)
    costs = np.array([np.abs(core - b[band, margin + s : w - margin + s]).mean() for s in shifts])
    best = int(costs.argmin())
    spread = float(np.median(costs) - costs[best])
    confidence = spread / (float(np.median(costs)) + 1e-6)
    return int(shifts[best]), float(np.clip(confidence, 0.0, 1.0))
