"""One on-disk episode format (runs/<run_id>/ep_<n>/) shared by agent runs and human demos."""

import functools
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.reward import REWARD_TERMS

FLAG_BAD_STEP = 1
FLAG_TERMINATED = 2
FLAG_TRUNCATED = 4
FLAG_ROUND_COMPLETE = 8
FLAG_GAIN_CLIPPED = 16

_FRAME_BYTES = int(np.prod(spec.PIXELS_SHAPE))
_REPO_ROOT = Path(__file__).resolve().parents[3]


@functools.cache
def git_provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}


def episode_dir(run_dir: str | Path, index: int) -> Path:
    return Path(run_dir) / f"ep_{index:05d}"


class EpisodeWriter:
    """Appends steps as they happen; every per-step field comes from the obs and the env's info dict."""

    def __init__(
        self,
        path: str | Path,
        *,
        is_demo: bool = False,
        config: dict | None = None,
        hud_crop_shape: tuple[int, ...] | None = None,
        flush_every: int = 1024,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.hud_crop_shape = tuple(hud_crop_shape) if hud_crop_shape else None
        self.flush_every = flush_every
        self.n_steps = 0
        self._part = 0
        self._rows: dict[str, list] = {}
        self._frames = None
        self._crops = None
        self._events = open(self.path / "events.jsonl", "a", buffering=1)
        self._manifest = {
            "spec_version": spec.SPEC_VERSION,
            "spec": spec.spec_canonical(),
            "git": git_provenance(),
            "is_demo": is_demo,
            "config": config or {},
            "hud_crop_shape": self.hud_crop_shape,
            "reward_terms": REWARD_TERMS,
            "created_unix": time.time(),
            "status": "open",
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        tmp = self.path / "spec.json.tmp"
        tmp.write_text(json.dumps(self._manifest, indent=2, default=str))
        tmp.replace(self.path / "spec.json")

    def add_step(
        self,
        obs: dict[str, np.ndarray],
        action,
        reward: float,
        info: dict,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record the observation acted on, the action taken, and what came back from env.step."""
        action = spec.validate_action(action)
        events = info.get("events", ())
        flags = (
            FLAG_BAD_STEP * bool(info.get("bad_step", False))
            | FLAG_TERMINATED * bool(terminated)
            | FLAG_TRUNCATED * bool(truncated)
            | FLAG_ROUND_COMPLETE * any(e.get("type") == "round_complete" for e in events)
            | FLAG_GAIN_CLIPPED * bool(info.get("gain_clipped", False))
        )
        row = {
            "action": action.astype(np.uint8),
            "reward": np.float32(reward),
            "reward_terms": np.asarray(info["reward_terms"], dtype=np.float32),
            "hud": np.asarray(obs["hud"], dtype=np.float32),
            "t_mono": np.float64(time.monotonic()),
            "dt": np.float32(info["dt"]),
            "flags": np.uint8(flags),
        }
        if "state" in obs:
            row["state"] = np.asarray(obs["state"], dtype=np.float32)
        for key, value in row.items():
            self._rows.setdefault(key, []).append(value)

        if "pixels" in obs:
            if self._frames is None:
                self._frames = open(self.path / "frames.u8", "ab")
            self._frames.write(np.ascontiguousarray(obs["pixels"], dtype=np.uint8).tobytes())
        if "hud_crops" in info:
            crops = np.ascontiguousarray(info["hud_crops"], dtype=np.uint8)
            if crops.shape != self.hud_crop_shape:
                raise ValueError(f"hud crops shaped {crops.shape}, writer expects {self.hud_crop_shape}")
            if self._crops is None:
                self._crops = open(self.path / "hud_crops.u8", "ab")
            self._crops.write(crops.tobytes())

        for event in events:
            self.add_event(event)
        self.n_steps += 1
        if len(self._rows["action"]) >= self.flush_every:
            self._flush_part()

    def add_event(self, event: dict) -> None:
        self._events.write(json.dumps({"step": self.n_steps, **event}, default=str) + "\n")

    def _flush_part(self) -> None:
        # Chunked until close, so a crash mid-episode loses at most flush_every steps.
        if not self._rows.get("action"):
            return
        arrays = {k: np.stack(v) for k, v in self._rows.items()}
        np.savez(self.path / f"meta_part_{self._part:05d}.npz", **arrays)
        self._part += 1
        self._rows = {}
        for f in (self._frames, self._crops):
            if f is not None:
                f.flush()

    def close(self, summary: dict | None = None, final_obs: dict[str, np.ndarray] | None = None) -> None:
        self._flush_part()
        parts = sorted(self.path.glob("meta_part_*.npz"))
        if parts:
            loaded = [dict(np.load(p)) for p in parts]
            np.savez(self.path / "meta.npz", **{k: np.concatenate([d[k] for d in loaded]) for k in loaded[0]})
            for p in parts:
                p.unlink()
        if final_obs is not None:
            np.savez(self.path / "final_obs.npz", **final_obs)
        for f in (self._frames, self._crops, self._events):
            if f is not None:
                f.close()
        self._manifest.update(status="closed", n_steps=self.n_steps, summary=summary or {})
        self._write_manifest()


@dataclass
class Episode:
    path: Path
    manifest: dict
    meta: dict[str, np.ndarray]
    events: list[dict]
    frames: np.ndarray | None
    hud_crops: np.ndarray | None
    final_obs: dict[str, np.ndarray] | None

    @property
    def n_steps(self) -> int:
        return len(self.meta["action"]) if "action" in self.meta else 0

    @property
    def complete(self) -> bool:
        return self.manifest.get("status") == "closed"


def load_episode(path: str | Path) -> Episode:
    """Load an episode, refusing any written under a different SPEC_VERSION. Unclosed (crashed) ones load
    from their flushed chunks, truncated to the last fully recorded step."""
    path = Path(path)
    manifest = json.loads((path / "spec.json").read_text())
    spec.require_spec_version(manifest["spec_version"], str(path))

    if (path / "meta.npz").exists():
        meta = dict(np.load(path / "meta.npz"))
    else:
        parts = [dict(np.load(p)) for p in sorted(path.glob("meta_part_*.npz"))]
        meta = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]} if parts else {}
    n = len(meta["action"]) if meta else 0

    frames = None
    if (path / "frames.u8").exists():
        count = min(n, (path / "frames.u8").stat().st_size // _FRAME_BYTES)
        frames = np.memmap(path / "frames.u8", dtype=np.uint8, mode="r", shape=(count, *spec.PIXELS_SHAPE))
    crops = None
    if (path / "hud_crops.u8").exists():
        shape = tuple(manifest["hud_crop_shape"])
        count = min(n, (path / "hud_crops.u8").stat().st_size // int(np.prod(shape)))
        crops = np.memmap(path / "hud_crops.u8", dtype=np.uint8, mode="r", shape=(count, *shape))

    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines() if line.strip()]
    final = dict(np.load(path / "final_obs.npz")) if (path / "final_obs.npz").exists() else None
    return Episode(path, manifest, meta, events, frames, crops, final)
