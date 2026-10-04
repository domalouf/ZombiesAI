"""Sampling training batches out of clips: frame stacks, splits, augmentation, class weights.

Two rules here are load-bearing rather than stylistic:

* **Frames are stored once and stacked by index arithmetic at sample time, clamped at clip boundaries.**
  Stacking at write time is 4x the storage for zero information, and an unclamped stack quietly teaches the
  network that the end of one clip causes the start of the next. A FLAG_CLIP_START mid-clip (play resuming
  after a stretch the player marked as a menu) is a boundary too: history clamps there as well.
* **Held-out data is held out in long blocks, never by step.** Neighbouring frames of one clip are nearly
  identical, so a step-level split reports a validation accuracy that is really a memorisation score. A long
  recording gives up minutes-long blocks, cut off from training by a gap on each side; a short one goes to
  one side whole.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from zombiesai import spec
from zombiesai.demos.clips import Clip, clip_span, iter_clips
from zombiesai.demos.hud_crops import hud_view

# Per-step targets beyond the action, present only when the clip's source could supply them.
EXTRA_KEYS = {"mc_return": np.float32, "aux_dpoints": np.int64, "aux_damage": np.int64}

# Validation carved out of a long recording: blocks of about two minutes, none shorter than thirty seconds.
VAL_BLOCK_STEPS = 120 * spec.DECISION_HZ
MIN_VAL_BLOCK_STEPS = 30 * spec.DECISION_HZ


@dataclass(frozen=True)
class DataConfig:
    min_confidence: float = 0.0  # drop steps whose label is worth less than this
    shift_px: int = 4  # random translation, the one augmentation that reliably helps pixel control
    brightness: float = 0.1  # +/- fraction of multiplicative brightness jitter
    # Loss weight of a human correction from a policy's play run (clips.Clip.corrections) relative to a demo
    # step. Neutral here; behavioural cloning sets its own (bc.BCConfig.correction_weight).
    correction_weight: float = 1.0
    audio_gain_db: float = 6.0  # +/- random loudness, so the policy does not hinge on where the volume sat
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
        offsets: Sequence[int] | None = None,
        audio=None,
        audio_config=None,
        hud_view: bool = False,
    ):
        """`before`/`after` give the contiguous window [t - before, t + after]. `offsets` replaces it with any
        set of relative steps (negative in the past), e.g. a strided history (-30, -16, -8, -4, -2, -1, 0).

        `audio`, when given, is what each clip sounds like: a callable clip -> per-step features
        (`hearing.clip_features`) or None for a clip without sound; batches then carry `audio` and
        `audio_mask` (0 where there was nothing to hear, padded with the feature of silence).

        `hud_view` adds each step's view of the HUD corner (demos/hud_crops.py), cut from the clip's recorded
        points_ammo crops; zeros for a clip recorded without them."""
        self.clips = [c for c in clips if c.n_steps > 0]
        self.config = config or DataConfig()
        if offsets is None:
            self.offsets = np.arange(-before, after + 1)
        else:
            self.offsets = np.array(sorted({int(o) for o in offsets}), dtype=np.int64)
            if len(self.offsets) != len(offsets) or not len(self.offsets):
                raise ValueError(f"offsets must be distinct relative steps, got {tuple(offsets)}")
            before, after = max(0, -int(self.offsets[0])), max(0, int(self.offsets[-1]))
        self.before, self.after = before, after
        self._audio = None
        if audio is not None:
            from zombiesai.demos.hearing import AudioFeatureConfig, silence

            self.audio_config = audio_config or AudioFeatureConfig()
            self._audio = [audio(clip) for clip in self.clips]
            self._silence = silence(self.audio_config)
        self._hud = [clip.hud("points_ammo") for clip in self.clips] if hud_view else None
        # Where each step's history begins, per clip: computed once, read on every sample.
        self._segment_start = [clip.segment_start for clip in self.clips]
        # A play run's usable steps are all corrections (`Clip.usable`), so its whole clip carries their weight.
        self._clip_weight = [self.config.correction_weight if clip.is_play_run else 1.0 for clip in self.clips]
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
        return len(self.offsets)

    @property
    def n_corrections(self) -> int:
        """How many of the steps are human corrections from play runs."""
        return int(sum(self.clips[c].is_play_run for c, _ in self.index))

    def actions(self) -> np.ndarray:
        """Every labelled action in the dataset, in index order -- the input to class weighting and stats."""
        return np.stack([self.clips[c].actions[t] for c, t in self.index]).astype(np.int64)

    def frames(self, rows: np.ndarray) -> np.ndarray:
        """(B, window, 72, 128, 3) uint8, oldest first. Steps before a clip's (or a resumed segment's) start
        repeat its first frame."""
        out = np.empty((len(rows), self.window, *spec.PIXELS_SHAPE), dtype=np.uint8)
        offsets = self.offsets
        for i, (ci, t) in enumerate(rows):
            clip = self.clips[ci]
            taps = np.clip(t + offsets, self._segment_start[ci][t], clip.n_steps - 1)
            out[i] = clip.frames[taps]
        return out

    def batch(self, rows: np.ndarray, rng: np.random.Generator | None = None, augment: bool = False) -> dict:
        rows = np.atleast_2d(np.asarray(rows, dtype=np.int64))
        frames = self.frames(rows)
        if augment:
            frames = augment_frames(frames, self.config, rng or np.random.default_rng())
        actions = np.stack([self.clips[c].actions[t] for c, t in rows]).astype(np.int64)
        weights = np.array([self.clips[c].confidence[t] * self._clip_weight[c] for c, t in rows], dtype=np.float32)

        prev = np.stack([self.prev_actions(c, t) for c, t in rows])
        batch = {"pixels": frames, "action": actions, "weight": weights, "prev_actions": prev}
        batch.update(self.extras(rows))
        if self._audio is not None:
            batch.update(self.audio(rows, (rng or np.random.default_rng()) if augment else None))
        if self._hud is not None:
            batch["hud_view"] = np.stack([hud_view(None if self._hud[c] is None else np.asarray(self._hud[c][t]))
                                          for c, t in rows])
        return batch

    def audio(self, rows: np.ndarray, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        """(B, 2, frames, mels) features and a (B,) has-audio mask. With `rng`, a random gain per sample --
        a constant in the log domain -- on the ones that have audio."""
        out = np.empty((len(rows), *self._silence.shape), dtype=np.float32)
        mask = np.zeros(len(rows), dtype=np.float32)
        for i, (ci, t) in enumerate(rows):
            features = self._audio[ci]
            if features is None:
                out[i] = self._silence
            else:
                out[i], mask[i] = features[t], 1.0
        if rng is not None and self.config.audio_gain_db:
            gain = rng.uniform(-self.config.audio_gain_db, self.config.audio_gain_db, size=len(rows))
            out += (mask * gain / self.audio_config.scale_db)[:, None, None, None].astype(np.float32)
        return {"audio": out, "audio_mask": mask}

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
        """The spec's prev-action encoding, built from the clip's own labels, clamped at the segment start."""
        clip = self.clips[clip_index]
        first = int(self._segment_start[clip_index][step])
        if step == first or not clip.labelled:
            return spec.encode_prev_actions([spec.NEUTRAL_ACTION] * spec.PREV_ACTION_HISTORY)
        history = [
            tuple(int(v) for v in clip.actions[max(step - 1 - i, first)]) for i in range(spec.PREV_ACTION_HISTORY)
        ]
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


def training_clips(roots, min_confidence: float = 0.0) -> tuple[list[Clip], list[Clip]]:
    """Every labelled clip under `roots` with something to train on, and the labelled ones left out.

    A root may be a clip, or any directory of them -- data/demos, runs/play, one runs/play/play_0003. A play
    run offers only its human corrections (`Clip.usable`); one in which nobody took over offers nothing, and
    is left out rather than risk being drawn as the held-out clip."""
    labelled = [c for root in roots for c in iter_clips(root) if c.labelled]
    keep = [c for c in labelled if c.usable(min_confidence).any()]
    return keep, [c for c in labelled if not c.usable(min_confidence).any()]


def split_clips(
    clips: list[Clip],
    val_fraction: float = 0.1,
    seed: int = 0,
    *,
    gap: int = spec.DECISION_HZ,
    block_steps: int = VAL_BLOCK_STEPS,
    min_block_steps: int = MIN_VAL_BLOCK_STEPS,
) -> tuple[list[Clip], list[Clip]]:
    """Hold out `val_fraction` of every recording long enough to spare it, as evenly spaced blocks of about
    `block_steps`; the rest of it trains, as spans that stop `gap` steps short of each block on both sides.

    Every session is validated on, so a split can't land on the one short recording, or on the one without
    audio -- and nothing here depends on `seed`, so two runs over the same clips compare on the same steps.
    Spans clamp their history at their own start (`clips.clip_span`), so no frame stack reaches across a
    cut; `gap` keeps training steps off the near-duplicates at a block's edges as well -- pass the history
    window plus a second. Play runs always train whole: their corrections are too few to spend on scoring.

    When no clip is long enough to carve from, whole clips are held out instead (`split_whole_clips`)."""
    if val_fraction <= 0:
        return list(clips), []
    train, val = [], []
    for clip in clips:
        blocks = [] if clip.is_play_run else val_blocks(clip.n_steps, val_fraction, block_steps, min_block_steps)
        edge = 0
        for start, stop in blocks:
            if start - gap > edge:
                train.append(clip_span(clip, edge, start - gap))
            val.append(clip_span(clip, start, stop))
            edge = stop + gap
        if not blocks:
            train.append(clip)
        elif edge < clip.n_steps:
            train.append(clip_span(clip, edge, clip.n_steps))
    if val:
        return train, val
    return split_whole_clips(clips, val_fraction, seed)


def val_blocks(n_steps: int, val_fraction: float, block_steps: int = VAL_BLOCK_STEPS,
               min_block_steps: int = MIN_VAL_BLOCK_STEPS) -> list[tuple[int, int]]:
    """[start, stop) of the held-out blocks of an `n_steps` recording: `val_fraction` of it in as many blocks
    as make each about `block_steps`, centred in equal shares of the recording so they sample its early, middle
    and late play alike. None when the fraction would come to less than `min_block_steps`."""
    held = n_steps * val_fraction
    if held < min_block_steps:
        return []
    k = max(1, round(held / block_steps))
    centres = (np.arange(k) + 0.5) * n_steps / k
    return [(int(round(c - held / k / 2)), int(round(c + held / k / 2))) for c in centres]


def split_whole_clips(clips: list[Clip], val_fraction: float = 0.1, seed: int = 0) -> tuple[list[Clip], list[Clip]]:
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
