"""Sampling training batches out of clips: frame stacks, splits, augmentation, class weights.

Two rules here are load-bearing rather than stylistic:

* **Frames are stored once and stacked by index arithmetic at sample time, clamped at clip boundaries.**
  Stacking at write time is 4x the storage for zero information, and an unclamped stack quietly teaches the
  network that the end of one clip causes the start of the next.
* **Held-out data is held out by clip, never by step.** Neighbouring frames of one clip are nearly identical,
  so a step-level split reports a validation accuracy that is really a memorisation score.
"""

from dataclasses import dataclass

import numpy as np

from zombiesai import spec
from zombiesai.demos.clips import Clip

# Per-step targets beyond the action, present only when the clip's source could supply them.
EXTRA_KEYS = {"mc_return": np.float32, "aux_dpoints": np.int64, "aux_damage": np.int64}


@dataclass(frozen=True)
class DataConfig:
    min_confidence: float = 0.0  # drop steps whose label is worth less than this
    shift_px: int = 4  # random translation, the one augmentation that reliably helps pixel control
    brightness: float = 0.1  # +/- fraction of multiplicative brightness jitter
    # No horizontal flips, ever: a flip inverts the yaw label and mirrors the HUD, and Nacht is not
    # mirror-symmetric (PLAN.md, "Demonstrations and BC").


class ClipDataset:
    """An index over one or more clips: every usable step, addressable as (frames window, action)."""

    def __init__(
        self,
        clips: list[Clip],
        config: DataConfig | None = None,
        *,
        before: int = 3,
        after: int = 0,
        clamp_edges: bool = False,
    ):
        self.clips = [c for c in clips if c.n_steps > 0]
        self.config = config or DataConfig()
        self.before, self.after = before, after
        index = []
        for ci, clip in enumerate(self.clips):
            usable = clip.usable(self.config.min_confidence)
            steps = np.flatnonzero(usable)
            # Training drops steps whose future window runs off the end, because a clamped future frame is a
            # lie about what happened next. Labelling keeps them: every step of the video needs an answer.
            if after and not clamp_edges:
                steps = steps[steps < clip.n_steps - after]
            index.append(np.stack([np.full(len(steps), ci), steps], axis=1))
        self.index = np.concatenate(index) if index else np.zeros((0, 2), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def window(self) -> int:
        return self.before + self.after + 1

    def actions(self) -> np.ndarray:
        """Every labelled action in the dataset, in index order -- the input to class weighting and stats."""
        return np.stack([self.clips[c].actions[t] for c, t in self.index]).astype(np.int64)

    def frames(self, rows: np.ndarray) -> np.ndarray:
        """(B, window, 72, 128, 3) uint8. Steps before a clip's start repeat its first frame."""
        out = np.empty((len(rows), self.window, *spec.PIXELS_SHAPE), dtype=np.uint8)
        offsets = np.arange(-self.before, self.after + 1)
        for i, (ci, t) in enumerate(rows):
            clip = self.clips[ci]
            taps = np.clip(t + offsets, 0, clip.n_steps - 1)
            out[i] = clip.frames[taps]
        return out

    def batch(self, rows: np.ndarray, rng: np.random.Generator | None = None, augment: bool = False) -> dict:
        rows = np.atleast_2d(np.asarray(rows, dtype=np.int64))
        frames = self.frames(rows)
        if augment:
            frames = augment_frames(frames, self.config, rng or np.random.default_rng())
        actions = np.stack([self.clips[c].actions[t] for c, t in rows]).astype(np.int64)
        weights = np.array([self.clips[c].confidence[t] for c, t in rows], dtype=np.float32)

        prev = np.stack([self.prev_actions(c, t) for c, t in rows])
        batch = {"pixels": frames, "action": actions, "weight": weights, "prev_actions": prev}
        batch.update(self.extras(rows))
        return batch

    def extras(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Value and auxiliary targets for the steps that have them, each with a mask of which ones do.

        Clips from video carry none of this, clips from sim episodes carry all of it, and a training run
        usually mixes both -- so the targets travel with a mask instead of the loader pretending to know."""
        out: dict[str, np.ndarray] = {}
        for key, dtype in EXTRA_KEYS.items():
            if not any(c.extra(key) is not None for c in self.clips):
                continue
            values = np.zeros(len(rows), dtype=dtype)
            mask = np.zeros(len(rows), dtype=bool)
            for i, (ci, t) in enumerate(rows):
                series = self.clips[ci].extra(key)
                if series is not None:
                    values[i], mask[i] = series[t], True
            out[key] = values
            out[f"{key}_mask"] = mask
        return out

    def prev_actions(self, clip_index: int, step: int) -> np.ndarray:
        """The spec's prev-action encoding, built from the clip's own labels, clamped at the clip start."""
        clip = self.clips[clip_index]
        if step == 0 or not clip.labelled:
            return spec.encode_prev_actions([spec.NEUTRAL_ACTION] * spec.PREV_ACTION_HISTORY)
        history = [tuple(int(v) for v in clip.actions[max(step - 1 - i, 0)]) for i in range(spec.PREV_ACTION_HISTORY)]
        return spec.encode_prev_actions(history)

    def epoch(self, batch_size: int, rng: np.random.Generator, shuffle: bool = True, drop_last: bool = True):
        """Batches of row indices. Training drops the short final batch; evaluation keeps it, because a
        held-out set smaller than one batch would otherwise report nothing at all."""
        order = rng.permutation(len(self)) if shuffle else np.arange(len(self))
        stop = len(order) - batch_size + 1 if drop_last else len(order)
        for start in range(0, stop, batch_size):
            yield self.index[order[start : start + batch_size]]


def augment_frames(frames: np.ndarray, config: DataConfig, rng: np.random.Generator) -> np.ndarray:
    """Random shift and brightness jitter, identical across every frame of one stack.

    Shifting the frames of a stack independently would fabricate camera motion, which is exactly the signal
    the network is being asked to read.
    """
    b, t = frames.shape[:2]
    h, w = spec.PIXELS_SHAPE[:2]
    out = frames
    if config.shift_px:
        p = config.shift_px
        padded = np.pad(frames, ((0, 0), (0, 0), (p, p), (p, p), (0, 0)), mode="edge")
        oy = rng.integers(0, 2 * p + 1, size=b)
        ox = rng.integers(0, 2 * p + 1, size=b)
        rows = oy[:, None] + np.arange(h)
        cols = ox[:, None] + np.arange(w)
        out = padded[
            np.arange(b)[:, None, None, None],
            np.arange(t)[None, :, None, None],
            rows[:, None, :, None],
            cols[:, None, None, :],
        ]
    if config.brightness:
        gain = rng.uniform(1.0 - config.brightness, 1.0 + config.brightness, size=(b, 1, 1, 1, 1))
        out = np.clip(out * gain, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out, dtype=np.uint8)


def split_clips(clips: list[Clip], val_fraction: float = 0.1, seed: int = 0) -> tuple[list[Clip], list[Clip]]:
    """Split whole clips into train and validation, keeping at least one clip on each side when possible."""
    if len(clips) < 2 or val_fraction <= 0:
        return list(clips), []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))
    n_val = max(1, int(round(len(clips) * val_fraction)))
    val = {int(i) for i in order[:n_val]}
    return [c for i, c in enumerate(clips) if i not in val], [c for i, c in enumerate(clips) if i in val]


def class_weights(actions: np.ndarray, beta: float = 0.999, power: float = 0.5) -> list[np.ndarray]:
    """Per-head class weights by effective number of samples (Cui et al. 2019), normalized to mean 1.

    The button head is 'none' about 95% of the time. With plain cross-entropy the predictable result is a
    policy that never reloads -- it is never worth predicting a class that rare. This is the fix the plan
    names, and the reason every head gets weights rather than just the obvious one.

    `power` softens the correction (0.5 is square-root inverse frequency, 1.0 the full version, 0 none).
    Full correction overshoots on the look heads, where "don't turn" is the majority for good reason: a
    policy that turns on nine steps out of ten fails the rollout-statistics check as surely as one that
    never turns at all, just in the other direction.
    """
    actions = np.asarray(actions, dtype=np.int64)
    weights = []
    for head, n_values in enumerate(spec.ACTION_NVEC):
        counts = np.bincount(actions[:, head], minlength=n_values).astype(np.float64)
        effective = (1.0 - beta ** np.maximum(counts, 1)) / (1.0 - beta)
        w = (1.0 / effective) ** power
        present = counts > 0
        w[~present] = 0.0  # a value never demonstrated gets no weight rather than infinite weight
        w[present] /= w[present].mean()
        weights.append(w.astype(np.float32))
    return weights


def majority_baseline(actions: np.ndarray) -> np.ndarray:
    """Per-head accuracy of always predicting that head's most common value -- the number any honest report
    of per-frame accuracy has to beat."""
    actions = np.asarray(actions, dtype=np.int64)
    return np.array(
        [np.bincount(actions[:, h], minlength=n).max() / len(actions) for h, n in enumerate(spec.ACTION_NVEC)]
    )
