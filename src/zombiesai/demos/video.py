"""Video ingest: any recording of real Nacht der Untoten play -> clips of policy frames at the decision rate.

This is the front door for footage that was never recorded with input logging -- an OBS capture, an old
ShadowPlay file, a session someone recorded before this project existed. It answers three questions the rest
of the pipeline cannot: which pixels are the game (bars off, aspect fixed), which frames are one continuous
stretch of play (a hard cut is not a transition an MDP can contain), and which stretches are not gameplay at
all (menus, loading, a paused capture, the desktop).

Decoding, frame-rate conversion and the area-average downsample are ffmpeg's job -- the same `flags=area`
arithmetic `frames.area_resize` does in-process, several hundred times faster. Segmentation is ours, and is
a pure function of the frame deltas so it can be tested without a video file.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos import frames as fr
from zombiesai.demos.clips import ClipWriter

FRAME_BYTES = int(np.prod(spec.PIXELS_SHAPE))


class FFmpegMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestConfig:
    fit: str = "crop"
    fps: float = float(spec.DECISION_HZ)
    # Mean absolute luma change per frame, in 0-255 units. Gameplay at 15 Hz sits in single digits even when
    # spinning; a cut between scenes jumps far above that; a frozen or paused capture sits at zero. A cut has
    # to clear both the floor and a multiple of this video's own median, so footage that is busy throughout
    # is not chopped up by its own fast turns, and calm footage still has its cuts found.
    cut_delta: float = 35.0
    cut_ratio: float = 4.0
    static_delta: float = 0.35
    static_steps: int = 30  # 2 s of no motion ends a clip: a menu, a pause, or a dropped capture
    min_steps: int = 60  # shorter than 4 s is not worth a training sample
    black_luma: float = 6.0
    start_s: float = 0.0
    duration_s: float | None = None


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration_s: float

    @property
    def aspect(self) -> float:
        return self.width / self.height


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegMissing(f"{name} is not on PATH; video ingest needs ffmpeg (apt install ffmpeg)")
    return path


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    if not data.get("streams"):
        raise ValueError(f"{path} has no video stream")
    stream = data["streams"][0]
    num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return VideoInfo(path, int(stream["width"]), int(stream["height"]), fps, duration)


def sample_frames(path: str | Path, info: VideoInfo, n: int = 8) -> np.ndarray:
    """A handful of full-resolution frames spread across the video, for letterbox detection."""
    times = np.linspace(0.05, 0.95, n) * max(info.duration_s, 1.0)
    out = []
    for t in times:
        raw = subprocess.run(
            [_tool("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=False,
        ).stdout
        if len(raw) == info.width * info.height * 3:
            out.append(np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3))
    if not out:
        raise ValueError(f"could not decode any frame of {path}")
    return np.stack(out)


def filter_chain(box: tuple[int, int, int, int], config: IngestConfig) -> str:
    y, x, h, w = box
    height, width = spec.PIXELS_SHAPE[:2]
    chain = [f"fps={config.fps}", f"crop={w}:{h}:{x}:{y}"]
    if config.fit == "pad":
        chain += [
            f"scale={width}:{height}:flags=area:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        ]
    else:  # crop and stretch differ only in the box; by here the box already has the right aspect
        chain.append(f"scale={width}:{height}:flags=area")
    return ",".join(chain)


def decode_to(path: str | Path, dest: Path, box, config: IngestConfig) -> int:
    """Stream the whole video through ffmpeg into a flat uint8 frame file. Returns the frame count."""
    cmd = [_tool("ffmpeg"), "-v", "error"]
    if config.start_s:
        cmd += ["-ss", f"{config.start_s:.3f}"]
    if config.duration_s:
        cmd += ["-t", f"{config.duration_s:.3f}"]
    cmd += ["-i", str(path), "-vf", filter_chain(box, config), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    written = 0
    with open(dest, "wb") as out:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_BYTES * 16)
        assert proc.stdout is not None
        while chunk := proc.stdout.read(FRAME_BYTES * 16):
            out.write(chunk)
            written += len(chunk)
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed decoding {path}")
    return written // FRAME_BYTES


def segment(deltas: np.ndarray, dark: np.ndarray, config: IngestConfig) -> list[tuple[int, int]]:
    """Contiguous gameplay runs as [start, stop) index pairs.

    Three things end a run: a cut (the next frame has nothing to do with this one), a stretch with no motion
    at all (menu, pause, frozen capture), and darkness (a fade or a loading screen). A run that survives all
    three and is long enough becomes a clip.
    """
    n = len(deltas)
    alive = ~dark
    moving = deltas[deltas > config.static_delta]
    threshold = max(config.cut_delta, config.cut_ratio * float(np.median(moving)) if len(moving) else 0.0)
    if n:
        # A static stretch invalidates the frames it spans, including the ones before we noticed.
        still = deltas < config.static_delta
        run = 0
        for i in range(n):
            run = run + 1 if still[i] else 0
            if run >= config.static_steps:
                alive[i - run + 1 : i + 1] = False
    runs: list[tuple[int, int]] = []
    start = None
    for i in range(n):
        cut = i > 0 and deltas[i] > threshold
        if start is not None and (not alive[i] or cut):
            runs.append((start, i))
            start = None
        if start is None and alive[i]:
            start = i
    if start is not None:
        runs.append((start, n))
    return [(a, b) for a, b in runs if b - a >= config.min_steps]


def ingest(
    video: str | Path,
    out_root: str | Path,
    config: IngestConfig | None = None,
    *,
    staging: Path | None = None,
    name: str | None = None,
) -> list[Path]:
    """Decode one video into clips under `out_root`, returning the clip directories written."""
    config = config or IngestConfig()
    video = Path(video)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    name = name or video.stem
    info = probe(video)
    bars = fr.detect_bars(sample_frames(video, info))
    box = fr.crop_box(info.height, info.width, bars, config.fit)

    staging = Path(staging or out_root) / f".staging_{name}"
    staging.mkdir(parents=True, exist_ok=True)
    raw = staging / "frames.u8"
    try:
        n = decode_to(video, raw, box, config)
        if n == 0:
            return []
        stream = np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, *spec.PIXELS_SHAPE))
        deltas = fr.frame_delta(stream)
        dark = fr.luma(stream).mean(axis=(1, 2)) < config.black_luma
        source = {
            "kind": "video",
            "path": str(video.resolve()),
            "fps_in": info.fps,
            "size_in": [info.width, info.height],
            "crop_box": list(box),
            "bars": list(bars),
            "fit": config.fit,
            "offset_s": config.start_s,
        }
        written = []
        for i, (a, b) in enumerate(segment(deltas, dark, config)):
            clip_dir = out_root / f"{name}_{i:04d}"
            writer = ClipWriter(
                clip_dir,
                source={**source, "start_s": config.start_s + a / config.fps, "start_frame": int(a)},
                label_source="none",
                config={"ingest": vars(config)},
            )
            for frame in stream[a:b]:
                writer.add(frame)
            writer.close(summary={"mean_delta": float(deltas[a:b].mean()), "seconds": (b - a) / config.fps})
            written.append(clip_dir)
        return written
    finally:
        shutil.rmtree(staging, ignore_errors=True)
