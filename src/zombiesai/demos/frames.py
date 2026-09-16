"""Turning real captured pixels into the spec's policy frames: area-average resize, bar removal, frame deltas.

Every path that ever feeds real footage to a network goes through `to_policy_frame`, so a recording made
today and a screen capture made in M5 are downsampled by exactly the same arithmetic.
"""

import functools

import numpy as np

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


def area_resize(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Exact area-average resize of an HxWx3 uint8 image. Identity when the size already matches."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 image, got {image.shape}")
    h, w = image.shape[:2]
    if (h, w) == (out_h, out_w):
        return np.ascontiguousarray(image, dtype=np.uint8)
    x = image.astype(np.float32)
    x = np.tensordot(_axis_weights(h, out_h), x, axes=(1, 0))  # (out_h, w, 3)
    x = np.tensordot(x, _axis_weights(w, out_w).T, axes=(1, 0))  # (out_h, 3, out_w)
    return np.rint(x.transpose(0, 2, 1)).clip(0, 255).astype(np.uint8)


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


def to_policy_frame(frame: np.ndarray, box: tuple[int, int, int, int] | None = None, fit: str = "crop") -> np.ndarray:
    """One captured frame as the agent's (72, 128, 3) uint8 observation.

    `box` is a source rectangle from `crop_box`; pass the one computed once per video rather than
    re-detecting bars per frame, so the geometry can't drift mid-clip.
    """
    frame = np.asarray(frame, dtype=np.uint8)
    if box is None:
        box = crop_box(*frame.shape[:2], detect_bars(frame), fit)
    y, x, h, w = box
    view = frame[y : y + h, x : x + w]
    if fit != "pad" or abs(w / h - ASPECT) < 1e-9:
        return area_resize(view, FRAME_H, FRAME_W)
    # Pad: keep the whole source frame (a 4:3 recording's HUD included) inside the 16:9 observation.
    if w < h * ASPECT:
        inner_w, inner_h = max(1, int(round(FRAME_H * w / h))), FRAME_H
    else:
        inner_w, inner_h = FRAME_W, max(1, int(round(FRAME_W * h / w)))
    out = np.zeros(spec.PIXELS_SHAPE, dtype=np.uint8)
    y0, x0 = (FRAME_H - inner_h) // 2, (FRAME_W - inner_w) // 2
    out[y0 : y0 + inner_h, x0 : x0 + inner_w] = area_resize(view, inner_h, inner_w)
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
