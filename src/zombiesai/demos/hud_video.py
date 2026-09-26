"""HUD crops kept as video: ~17x smaller than the raw crops, at an error the eye (and a digit matcher) can't see.

The raw crops are most of a recording -- ~7.4 GB of a 20-minute session's ~7.9 at 1440p -- and lossless
compression only halves them, because every box has the moving scene behind its text. A video codec is made
for exactly that. So once a session has ended, `pack_hud` re-encodes each region as H.264 (yuv444p, so red
tally marks and red damage keep full colour resolution) at CRF 12, decodes every chunk back, compares it
with the raw crops, and deletes the raw file only when every chunk is within `MAX_MEAN_ERR`/`MAX_P99_ERR`.
Measured on demo_0002: 17x smaller, a mean error of 1.4 levels in 255 and 8 on the 99th-percentile text pixel.
Encoding runs at ~1100 steps a second per region, off the recording's clock.

    hud_<name>/00000.mkv, 00001.mkv, ...   CHUNK_STEPS (450 = 30 s) steps each, the last one shorter

Chunks rather than one file so that any step can be decoded exactly without trusting a container's seek:
a read decodes the whole chunk it falls in. clip.json["hud"]["video"][name] records the codec, the step
count, the chunk length and the measured error; `Clip.hud()` returns a `HudVideo`, which indexes like the
memmapped raw array it replaces.
"""

import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

CHUNK_STEPS = 450
CRF = 12
CODEC = "libx264"
PIX_FMT = "yuv444p"
MAX_MEAN_ERR = 3.0  # levels in 255, per chunk; CRF 12 measures ~1.4
MAX_P99_ERR = 16.0  # 99th-percentile absolute error, per chunk; CRF 12 measures ~6
FPS = 15


class PackError(RuntimeError):
    pass


def _ffmpeg() -> str:
    tool = shutil.which("ffmpeg")
    if tool is None:
        raise PackError("ffmpeg is not on PATH; packing HUD crops needs it")
    return tool


def encode_chunk(crops: np.ndarray, out: Path, crf: int = CRF) -> None:
    t, h, w, _ = crops.shape
    cmd = [_ffmpeg(), "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", CODEC, "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", PIX_FMT, str(out)]
    done = subprocess.run(cmd, input=np.ascontiguousarray(crops, dtype=np.uint8).tobytes(), capture_output=True)
    if done.returncode != 0:
        raise PackError(f"ffmpeg failed encoding {out}: {done.stderr.decode(errors='replace')[-300:]}")


def decode_chunk(path: Path, shape) -> np.ndarray:
    h, w, c = shape
    done = subprocess.run([_ffmpeg(), "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-"], capture_output=True)
    if done.returncode != 0:
        raise PackError(f"ffmpeg failed decoding {path}: {done.stderr.decode(errors='replace')[-300:]}")
    per = h * w * c
    if len(done.stdout) % per:
        raise PackError(f"{path} decoded to {len(done.stdout)} bytes, not a whole number of {w}x{h} crops")
    return np.frombuffer(done.stdout, np.uint8).reshape(-1, h, w, c)


class HudVideo:
    """(T, h, w, 3) uint8 HUD crops decoded from video chunks on demand. Indexes like the memmapped raw
    array -- an int, a slice, an index array or a mask for the step, then anything for the rest -- and
    `np.asarray()` decodes all of it. Keeps the last few chunks decoded, so a sequential read decodes each
    chunk once."""

    dtype = np.dtype(np.uint8)
    ndim = 4

    def __init__(self, directory: Path, crop_shape, n_steps: int, chunk_steps: int = CHUNK_STEPS):
        self.directory = Path(directory)
        self.crop_shape = tuple(crop_shape)
        self.n_steps = int(n_steps)
        self.chunk_steps = int(chunk_steps)
        self._chunk = lru_cache(maxsize=4)(self._decode)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.n_steps, *self.crop_shape)

    def __len__(self) -> int:
        return self.n_steps

    def _decode(self, i: int) -> np.ndarray:
        chunk = decode_chunk(self.directory / f"{i:05d}.mkv", self.crop_shape)
        want = min(self.chunk_steps, self.n_steps - i * self.chunk_steps)
        if len(chunk) < want:
            raise PackError(f"{self.directory}/{i:05d}.mkv holds {len(chunk)} steps, expected {want}")
        return chunk

    def __getitem__(self, key):
        rest = ()
        if isinstance(key, tuple):
            key, rest = key[0], key[1:]
        if isinstance(key, (int, np.integer)):
            k = int(key) + (self.n_steps if key < 0 else 0)
            if not 0 <= k < self.n_steps:
                raise IndexError(f"step {key} out of range for {self.n_steps} steps")
            out = self._chunk(k // self.chunk_steps)[k % self.chunk_steps]
            return out[rest] if rest else out
        else:
            steps = np.arange(self.n_steps)[key]
            out = np.empty((len(steps), *self.crop_shape), np.uint8)
            chunks = steps // self.chunk_steps
            for i in np.unique(chunks):
                where = chunks == i
                out[where] = self._chunk(int(i))[steps[where] % self.chunk_steps]
        return out[(slice(None), *rest)] if rest else out

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return out if dtype is None else out.astype(dtype)

    def __iter__(self):
        for k in range(self.n_steps):
            yield self[k]


def _pack_region(clip_dir: Path, name: str, shape, n_steps: int, crf: int, chunk_steps: int,
                 cancel: threading.Event) -> dict:
    raw = np.memmap(clip_dir / f"hud_{name}.u8", dtype=np.uint8, mode="r")
    per = int(np.prod(shape))
    raw = raw[: (len(raw) // per) * per].reshape(-1, *shape)[:n_steps]
    if len(raw) < n_steps:
        raise PackError(f"hud_{name}.u8 holds {len(raw)} steps, the clip {n_steps}")
    final = clip_dir / f"hud_{name}"
    work = clip_dir / f"hud_{name}.packing"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    worst_mean = worst_p99 = 0.0
    try:
        for i, start in enumerate(range(0, n_steps, chunk_steps)):
            if cancel.is_set():
                raise PackError(f"{name}: stopped")
            crops = np.asarray(raw[start : start + chunk_steps])
            path = work / f"{i:05d}.mkv"
            encode_chunk(crops, path, crf)
            back = decode_chunk(path, shape)
            if back.shape != crops.shape:
                raise PackError(f"{name} chunk {i} decoded to {back.shape}, expected {crops.shape}")
            err = np.abs(back.astype(np.int16) - crops.astype(np.int16))
            mean, p99 = float(err.mean()), float(np.percentile(err, 99))
            if mean > MAX_MEAN_ERR or p99 > MAX_P99_ERR:
                raise PackError(f"{name} chunk {i} came back with mean error {mean:.2f}, p99 {p99:.0f}: "
                                f"over the limits ({MAX_MEAN_ERR}, {MAX_P99_ERR}), so the raw crops stay")
            worst_mean, worst_p99 = max(worst_mean, mean), max(worst_p99, p99)
    except BaseException:  # a failure or a Ctrl-C: the raw crops are untouched, drop the half-made video
        shutil.rmtree(work, ignore_errors=True)
        raise
    shutil.rmtree(final, ignore_errors=True)
    work.rename(final)
    return {"dir": final.name, "codec": CODEC, "pix_fmt": PIX_FMT, "crf": crf, "n_steps": n_steps,
            "chunk_steps": chunk_steps, "bytes": sum(p.stat().st_size for p in final.iterdir()),
            "raw_bytes": n_steps * per, "worst_chunk_mean_err": round(worst_mean, 3),
            "worst_chunk_p99_err": worst_p99}


def _write_manifest(clip_dir: Path, manifest: dict) -> None:
    tmp = clip_dir / "clip.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp, clip_dir / "clip.json")


def pack_hud(clip_dir: str | Path, *, crf: int = CRF, chunk_steps: int = CHUNK_STEPS, keep_raw: bool = False,
             say=print) -> dict:
    """Re-encode a closed clip's raw HUD crops as verified video and delete the raw files. Returns
    {region: what was written}; regions already packed are skipped. A region that fails verification keeps
    its raw crops (PackError). Safe to interrupt: the raw file goes only after the manifest says the video
    is there."""
    from zombiesai.demos.clips import load_clip

    clip_dir = Path(clip_dir)
    clip = load_clip(clip_dir)  # the trimmed step count: a crash can leave a region a step long
    manifest = json.loads((clip_dir / "clip.json").read_text())
    hud = manifest.get("hud", {})
    video = dict(hud.get("video", {}))
    todo = {n: s for n, s in hud.get("shapes", {}).items() if n not in video and (clip_dir / f"hud_{n}.u8").exists()}
    if not keep_raw:  # packed last time, but stopped before the raw file went
        for name in video:
            (clip_dir / f"hud_{name}.u8").unlink(missing_ok=True)
    if not todo or not clip.n_steps:
        return {}
    say(f"packing HUD crops of {clip_dir.name} as video: {', '.join(sorted(todo))} ...")
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=len(todo)) as pool:  # each region is its own ffmpeg process
        futures = {n: pool.submit(_pack_region, clip_dir, n, s, clip.n_steps, crf, chunk_steps, cancel)
                   for n, s in todo.items()}
        written, errors = {}, {}
        try:
            for name, future in futures.items():
                try:
                    written[name] = future.result()
                except PackError as e:
                    errors[name] = str(e)
        except BaseException:  # Ctrl-C: every region stops at its next chunk and drops its half-made video
            cancel.set()
            raise
    if written:
        video.update(written)
        manifest["hud"] = dict(hud, video=video)
        _write_manifest(clip_dir, manifest)
        if not keep_raw:
            for name in written:
                (clip_dir / f"hud_{name}.u8").unlink()
    for name, info in sorted(written.items()):
        say(f"  {name}: {info['raw_bytes'] / 1e9:.2f} GB -> {info['bytes'] / 1e9:.3f} GB "
            f"({info['raw_bytes'] / max(1, info['bytes']):.0f}x), worst chunk error mean "
            f"{info['worst_chunk_mean_err']:.2f} / p99 {info['worst_chunk_p99_err']:.0f}")
    if errors:
        raise PackError("; ".join(f"{n}: {e}" for n, e in sorted(errors.items())))
    return written
