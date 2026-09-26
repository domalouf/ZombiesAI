"""The clip store: contiguous runs of real gameplay frames at the decision rate, with or without actions.

An episode (`store/episode_store.py`) is something an agent did in an environment that answered back with
rewards. A clip is the other thing this project learns from: a stretch of footage a human produced, where
there is no reward, no HUD parse worth trusting yet, and the actions are either logged from the human's own
input device or guessed later by the inverse dynamics model. Keeping the two formats apart is what lets
ingested video carry honest metadata -- where the pixels came from, how they were cropped, how much the
labels are worth -- instead of being forced into fields an episode needs and a video does not have.

    <root>/<clip_id>/
      frames.u8    (T, 72, 128, 3) uint8, decision rate, area-averaged
      clip.json    provenance: source video, crop box, fit, spec stamps, label source
      labels.npz   actions (T, 8) uint8 + per-step confidence and raw pre-quantization yaw/pitch degrees
      hud_<name>.u8  optional (T, h, w, 3) uint8 full-resolution HUD crops, one file per region, same step
                     index as frames.u8; shapes and boxes in clip.json["hud"] (see demos/hud_crops.py)
      audio.s16 | audio.flac  optional game audio, s16le 48 kHz stereo; raw while recording, FLAC after a
                     clean stop (see demos/audio.py)
      audio_index.bin  (sample_end, t_mono) per captured chunk: maps samples onto the recorder's clock;
                     rate, device and latency in clip.json["audio"]

Frames and labels are versioned separately on purpose: a spec change that touches the HUD layout must not
invalidate a weekend of ingested video, but one that moves the yaw bins must invalidate its labels.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.store.episode_store import git_provenance

FLAG_BAD_STEP = 1  # capture overrun, dropped frame, or a step spanning a cut: excluded from training
FLAG_LOW_CONFIDENCE = 2  # pseudo-label below the keep threshold
FLAG_CLIP_START = 4  # no usable history before this step, so a frame stack must clamp here
FLAG_NOT_PLAYING = 8  # the player marked this stretch as menus/pause/loading (the mark key): not gameplay

LABEL_SOURCES = ("input_log", "agent", "idm", "none")
_FRAME_BYTES = int(np.prod(spec.PIXELS_SHAPE))


def frame_contract() -> dict:
    """What a stored frame means. Narrower than SPEC_VERSION so ingested pixels survive unrelated spec churn."""
    return {"pixels_shape": list(spec.PIXELS_SHAPE), "decision_hz": spec.DECISION_HZ}


class ClipWriter:
    """Appends frames (and optionally their actions) as they are captured, decoded, or simulated."""

    def __init__(self, path: str | Path, *, source: dict, label_source: str = "none", config: dict | None = None):
        if label_source not in LABEL_SOURCES:
            raise ValueError(f"label_source must be one of {LABEL_SOURCES}, got {label_source!r}")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.label_source = label_source
        self.n_steps = 0
        self._frames = open(self.path / "frames.u8", "ab")
        self._hud: dict[str, object] | None = None  # region name -> open file, fixed by the first step
        self._labels: dict[str, list] = {}
        self._manifest = {
            "spec_version": spec.SPEC_VERSION,
            "frame_contract": frame_contract(),
            "action_nvec": list(spec.ACTION_NVEC),
            "git": git_provenance(),
            "source": source,
            "label_source": label_source,
            "config": config or {},
            "created_unix": time.time(),
            "status": "open",
        }
        self._write_manifest()

    def annotate(self, key: str, value) -> None:
        """Set a top-level manifest entry now, not at close, so it survives a crash (audio uses this)."""
        self._manifest[key] = value
        self._write_manifest()

    def _write_manifest(self) -> None:
        tmp = self.path / "clip.json.tmp"
        tmp.write_text(json.dumps(self._manifest, indent=2, default=str))
        tmp.replace(self.path / "clip.json")

    def add(
        self,
        frame: np.ndarray,
        action=None,
        *,
        confidence: float = 1.0,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        flags: int = 0,
        hud: dict[str, np.ndarray] | None = None,
    ) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != spec.PIXELS_SHAPE:
            raise ValueError(f"frame shaped {frame.shape}, expected {spec.PIXELS_SHAPE}")
        hud_rows = self._check_hud(hud)
        if self.label_source == "none":
            if action is not None:
                raise ValueError("this clip was opened with label_source='none' but was given an action")
        elif action is None:
            # Frames and labels are two files indexed by the same step number. One step without an action
            # would shift every later label by one, which nothing downstream could detect.
            raise ValueError(f"this clip's labels come from {self.label_source!r}; every step needs an action")
        self._frames.write(frame.tobytes())
        for name, crop in hud_rows.items():
            self._hud[name].write(crop.tobytes())
        if action is not None:
            row = {
                "action": spec.validate_action(action).astype(np.uint8),
                "confidence": np.float32(confidence),
                # Raw pre-quantization look, so changing the bins later is a re-quantization, not a re-recording.
                "yaw_deg": np.float32(yaw_deg),
                "pitch_deg": np.float32(pitch_deg),
                "flags": np.uint8(flags | (FLAG_CLIP_START if self.n_steps == 0 else 0)),
            }
            for key, value in row.items():
                self._labels.setdefault(key, []).append(value)
        self.n_steps += 1

    def _check_hud(self, hud: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
        """HUD crops follow the same rule as actions: once a clip has them, every step must, or the crops
        would drift out of step with the frames. Region names and shapes are fixed by the first step."""
        if self._hud is None:
            if not hud:
                if self.n_steps == 0:
                    self._hud = {}
                return {}
            if self.n_steps:
                raise ValueError("HUD crops started mid-clip; they must arrive from the first step")
            shapes = {}
            for name, crop in hud.items():
                self._hud_file(name)
                shapes[name] = list(np.shape(crop))
            self._manifest["hud"] = dict(self._manifest.get("hud", {}), shapes=shapes)
            self._write_manifest()
        if not self._hud:
            if hud:
                raise ValueError("this clip started without HUD crops; they cannot start mid-clip")
            return {}
        if not hud or set(hud) != set(self._hud):
            raise ValueError(f"every step needs HUD crops for {sorted(self._hud)}, got {sorted(hud or {})}")
        rows = {}
        for name, crop in hud.items():
            crop = np.ascontiguousarray(crop, dtype=np.uint8)
            want = tuple(self._manifest["hud"]["shapes"][name])
            if crop.shape != want:
                raise ValueError(f"HUD crop {name!r} shaped {crop.shape}, expected {want}")
            rows[name] = crop
        return rows

    def _hud_file(self, name: str):
        if self._hud is None:
            self._hud = {}
        self._hud[name] = open(self.path / f"hud_{name}.u8", "ab")
        return self._hud[name]

    def close(self, summary: dict | None = None) -> Path:
        self._frames.close()
        for f in (self._hud or {}).values():
            f.close()
        if self._labels:
            np.savez(self.path / "labels.npz", **{k: np.stack(v) for k, v in self._labels.items()})
        self._manifest.update(status="closed", n_steps=self.n_steps, summary=summary or {})
        self._write_manifest()
        return self.path


@dataclass
class Clip:
    path: Path
    manifest: dict
    frames: np.ndarray  # memmapped (T, 72, 128, 3)
    labels: dict[str, np.ndarray] | None
    _audio: object = field(default=None, repr=False, compare=False)

    @property
    def n_steps(self) -> int:
        return len(self.frames)

    @property
    def labelled(self) -> bool:
        return self.labels is not None

    @property
    def label_source(self) -> str:
        return self.manifest.get("label_source", "none")

    @property
    def actions(self) -> np.ndarray | None:
        return None if self.labels is None else self.labels["action"]

    @property
    def confidence(self) -> np.ndarray:
        if self.labels is None:
            return np.zeros(self.n_steps, dtype=np.float32)
        return self.labels["confidence"]

    @property
    def flags(self) -> np.ndarray:
        if self.labels is None:
            return np.zeros(self.n_steps, dtype=np.uint8)
        return self.labels["flags"]

    @property
    def segment_start(self) -> np.ndarray:
        """(T,) the step each step's history begins at: the latest FLAG_CLIP_START at or before it (0 when
        the clip has none). Frame stacks and previous-action histories clamp here, not just at step 0."""
        starts = np.where((self.flags & FLAG_CLIP_START) != 0, np.arange(self.n_steps), 0)
        return np.maximum.accumulate(starts) if len(starts) else starts

    def extra(self, key: str) -> np.ndarray | None:
        """Optional per-step target (`mc_return`, `aux_dpoints`, `aux_damage`); None when this clip has none.

        Video a human recorded has pixels and, after labelling, actions -- and nothing else. Sim episodes
        have all three. The trainers ask rather than assume, and train the heads the data can supply."""
        return None if self.labels is None else self.labels.get(key)

    @property
    def hud_regions(self) -> list[str]:
        return sorted(self.manifest.get("hud", {}).get("shapes", {}))

    def hud(self, name: str) -> np.ndarray | None:
        """Full-resolution HUD crops for one region, (T, h, w, 3) memmapped; None if this clip has none."""
        shape = self.manifest.get("hud", {}).get("shapes", {}).get(name)
        if shape is None:
            return None
        crops = np.memmap(self.path / f"hud_{name}.u8", dtype=np.uint8, mode="r")
        per = int(np.prod(shape))
        return crops[: (len(crops) // per) * per].reshape(-1, *shape)[: self.n_steps]

    def audio(self):
        """The clip's game audio as a `demos.audio.ClipAudio`, or None if it was recorded without any."""
        if self._audio is None and self.manifest.get("audio"):
            from zombiesai.demos.audio import load_audio

            self._audio = load_audio(self.path, self.manifest["audio"]) or False
        return self._audio or None

    def step_time(self, k) -> np.ndarray:
        """Monotonic time at which step k's frame was grabbed: `t0 + k / decision_hz`, the recorder's deadline."""
        t0 = self.manifest.get("summary", {}).get("t0_mono")
        if t0 is None:  # a recording that never closed still wrote t0 beside its audio
            t0 = (self.manifest.get("audio") or {}).get("t0_mono")
        if t0 is None:
            raise ValueError(f"{self.path} has no t0_mono, so its steps have no times")
        hz = self.manifest.get("config", {}).get("decision_hz", spec.DECISION_HZ)
        return t0 + np.asarray(k) / hz

    def audio_for_step(self, k: int, window_s: float = 0.2) -> np.ndarray | None:
        """The `window_s` of audio that had played by the time step k's frame was grabbed, (n, channels) int16.

        Causal on purpose: it ends at the frame, so it is what the player had heard when they chose step k's
        action -- the same information a policy consuming the reserved `audio` key would get. The default
        200 ms is the plan's log-mel window. Silence where nothing was captured; None without audio.
        """
        audio = self.audio()
        return None if audio is None else audio.window(float(self.step_time(k)), window_s)

    def usable(self, min_confidence: float = 0.0) -> np.ndarray:
        """Boolean mask of steps fit to train on: not flagged bad, not marked as out of play, and confidently
        enough labelled."""
        ok = (self.flags & (FLAG_BAD_STEP | FLAG_NOT_PLAYING)) == 0
        if self.labelled:
            ok &= self.confidence >= min_confidence
        return ok


def with_play_marks(flags: np.ndarray, not_playing: np.ndarray) -> np.ndarray:
    """Rewrite a clip's play marking from a (T,) not-playing mask, leaving every other flag alone.

    Marked steps get FLAG_NOT_PLAYING, and the first step of play after a marked stretch gets
    FLAG_CLIP_START: the frames behind it are a pause menu or a loading screen, and a frame stack that
    reached back into them would teach the policy that a menu is what comes before a fight. Existing marks
    are cleared first -- the marking is derived from the raw log, so re-deriving it (a different mark key)
    must be able to remove marks as well as add them. Only the FLAG_CLIP_START this marking put there (on a
    step just after a marked one) is cleared: the recorder also sets it on the first good frame after a
    capture outage, and that one is not the mark key's to take away. Step 0 keeps its own.
    """
    flags = np.asarray(flags, dtype=np.uint8)
    was_marked = (flags & FLAG_NOT_PLAYING) != 0
    ours = np.zeros(len(flags), dtype=bool)
    ours[1:] = was_marked[:-1] & ~was_marked[1:]
    flags = flags & np.uint8(~FLAG_NOT_PLAYING & 0xFF)
    flags = np.where(ours, flags & np.uint8(~FLAG_CLIP_START & 0xFF), flags).astype(np.uint8)
    marked = np.asarray(not_playing, dtype=bool)[: len(flags)]
    resumed = np.zeros_like(marked)
    resumed[1:] = marked[:-1] & ~marked[1:]
    flags = flags | np.where(marked, FLAG_NOT_PLAYING, 0).astype(np.uint8)
    flags = flags | np.where(resumed, FLAG_CLIP_START, 0).astype(np.uint8)
    if len(flags):
        flags[0] |= FLAG_CLIP_START
    return flags


def load_clip(path: str | Path, *, require_labels: bool = False) -> Clip:
    """Load a clip, refusing frames written under a different frame contract and labels written under a
    different SPEC_VERSION -- the labels are bin indices, and bins are spec."""
    path = Path(path)
    manifest = json.loads((path / "clip.json").read_text())
    if manifest.get("frame_contract") != frame_contract():
        raise spec.SpecMismatchError(
            f"{path} holds {manifest.get('frame_contract')} frames, this code wants {frame_contract()}"
        )
    size = (path / "frames.u8").stat().st_size
    if size < _FRAME_BYTES:  # an empty clip (stopped before its first step) cannot be memory-mapped
        frames = np.zeros((0, *spec.PIXELS_SHAPE), dtype=np.uint8)
    else:
        frames = np.memmap(
            path / "frames.u8", dtype=np.uint8, mode="r", shape=(size // _FRAME_BYTES, *spec.PIXELS_SHAPE)
        )
    labels = None
    if (path / "labels.npz").exists():
        spec.require_spec_version(manifest["spec_version"], f"{path} labels")
        labels = dict(np.load(path / "labels.npz"))
        if len(labels["action"]) != len(frames):
            n = min(len(labels["action"]), len(frames))  # a crash between the frame write and the label flush
            frames, labels = frames[:n], {k: v[:n] for k, v in labels.items()}
    elif require_labels:
        raise FileNotFoundError(f"{path} has no labels.npz; label it with the IDM or record it with input logging")
    for name, shape in manifest.get("hud", {}).get("shapes", {}).items():
        # A crash between the frame write and a HUD write leaves that region one step short.
        n = (path / f"hud_{name}.u8").stat().st_size // int(np.prod(shape))
        if n < len(frames):
            frames = frames[:n]
            labels = None if labels is None else {k: v[:n] for k, v in labels.items()}
    return Clip(path, manifest, frames, labels)


def iter_clips(root: str | Path, *, require_labels: bool = False):
    """Every clip under `root`, in path order. A clip directory is one holding a clip.json."""
    for manifest_path in sorted(Path(root).rglob("clip.json")):
        yield load_clip(manifest_path.parent, require_labels=require_labels)


def attach_labels(
    path: str | Path,
    actions: np.ndarray,
    confidence: np.ndarray,
    *,
    label_source: str,
    flags: np.ndarray | None = None,
    yaw_deg: np.ndarray | None = None,
    pitch_deg: np.ndarray | None = None,
    detail: dict | None = None,
) -> None:
    """Write labels onto an existing clip -- the IDM pseudo-labelling step. Overwrites any previous labels."""
    path = Path(path)
    manifest = json.loads((path / "clip.json").read_text())
    actions = np.asarray(actions, dtype=np.uint8)
    n = len(actions)
    if actions.shape != (n, len(spec.ACTION_NVEC)):
        raise ValueError(f"actions shaped {actions.shape}, expected (T, {len(spec.ACTION_NVEC)})")
    if (np.asarray(spec.ACTION_NVEC) <= actions).any():
        raise ValueError("labels contain an out-of-range action index")
    zeros = np.zeros(n, dtype=np.float32)
    flags = np.zeros(n, dtype=np.uint8) if flags is None else np.asarray(flags, dtype=np.uint8).copy()
    flags[0] |= FLAG_CLIP_START
    np.savez(
        path / "labels.npz",
        action=actions,
        confidence=np.asarray(confidence, dtype=np.float32),
        yaw_deg=zeros if yaw_deg is None else np.asarray(yaw_deg, dtype=np.float32),
        pitch_deg=zeros if pitch_deg is None else np.asarray(pitch_deg, dtype=np.float32),
        flags=flags,
    )
    manifest.update(
        spec_version=spec.SPEC_VERSION,
        label_source=label_source,
        labels=dict(detail or {}, written_unix=time.time(), n_steps=n),
    )
    (path / "clip.json").write_text(json.dumps(manifest, indent=2, default=str))


def monte_carlo_returns(rewards: np.ndarray, gamma: float, bootstrap: float = 0.0) -> np.ndarray:
    """Discounted return from each step to the end of the episode -- the value head's pretraining target.

    A random critic in a 15 Hz environment with 20-minute episodes takes an enormous number of real steps to
    become useful. Fitting it to returns that are already recorded is the plan's highest return per line.
    """
    out = np.empty(len(rewards), dtype=np.float32)
    running = float(bootstrap)
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


# Delta-points buckets for the auxiliary head: nothing, a hit, a kill, a big round-clearing swing. The
# boundaries are in points, from the game's own scoring table (10 a hit, 60 a body kill, 100 a headshot).
DPOINTS_EDGES = (1.0, 50.0, 130.0)


def clip_from_episode(path: str | Path, gamma: float = 0.995) -> Clip:
    """View a recorded episode (agent or demo) as a labelled clip, so one dataset can span both formats.

    An episode brings more than actions: its rewards give Monte-Carlo returns for the value head, and its
    reward terms give the two auxiliary targets -- "did I just score" and "am I being hit" -- that make the
    encoder represent what a value function will need.
    """
    from zombiesai.reward import REWARD_TERMS
    from zombiesai.store.episode_store import load_episode

    episode = load_episode(path)
    if episode.frames is None:
        raise ValueError(f"{path} has no frames.u8; record it with the render observation profile")
    n = min(len(episode.frames), episode.n_steps)
    flags = (episode.meta["flags"][:n].astype(np.uint8) & FLAG_BAD_STEP).copy()
    flags[0] |= FLAG_CLIP_START
    labels = {
        "action": episode.meta["action"][:n].astype(np.uint8),
        "confidence": np.ones(n, dtype=np.float32),
        "yaw_deg": np.zeros(n, dtype=np.float32),
        "pitch_deg": np.zeros(n, dtype=np.float32),
        "flags": flags,
    }
    if "reward" in episode.meta:
        labels["mc_return"] = monte_carlo_returns(episode.meta["reward"][:n], gamma)
    if "reward_terms" in episode.meta:
        terms = episode.meta["reward_terms"][:n]
        gain = terms[:, REWARD_TERMS.index("gain")] * 100.0  # back to points; reward divides by points_scale
        labels["aux_dpoints"] = np.digitize(gain, DPOINTS_EDGES).astype(np.uint8)
        labels["aux_damage"] = (terms[:, REWARD_TERMS.index("damage")] < 0).astype(np.uint8)
    manifest = {
        "spec_version": episode.manifest["spec_version"],
        "frame_contract": frame_contract(),
        "action_nvec": list(spec.ACTION_NVEC),
        "source": {"kind": "episode", "path": str(path), "is_demo": episode.manifest.get("is_demo", False)},
        "label_source": "input_log" if episode.manifest.get("is_demo") else "agent",
        "status": episode.manifest.get("status", "closed"),
        "n_steps": n,
    }
    return Clip(Path(path), manifest, episode.frames[:n], labels)
