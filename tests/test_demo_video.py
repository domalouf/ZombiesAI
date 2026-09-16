import shutil
import subprocess

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import video
from zombiesai.demos.clips import load_clip

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def deltas(values):
    return np.array(values, dtype=np.float32)


def test_a_continuous_run_is_one_clip():
    d = deltas([0.0] + [4.0] * 99)
    config = video.IngestConfig(min_steps=10)
    assert video.segment(d, np.zeros(100, bool), config) == [(0, 100)]


def test_a_cut_splits_the_run():
    d = deltas([4.0] * 100)
    d[50] = 90.0
    config = video.IngestConfig(min_steps=10)
    assert video.segment(d, np.zeros(100, bool), config) == [(0, 50), (50, 100)]


def test_busy_footage_is_not_chopped_up_by_its_own_motion():
    """A cut has to stand out from what this video normally does, not just clear an absolute number."""
    rng = np.random.default_rng(0)
    busy = deltas(rng.uniform(25.0, 40.0, size=200))
    config = video.IngestConfig(min_steps=10)
    assert video.segment(busy, np.zeros(200, bool), config) == [(0, 200)]
    busy[120] = 300.0  # a real cut, far above everything around it
    assert video.segment(busy, np.zeros(200, bool), config) == [(0, 120), (120, 200)]


def test_a_frozen_capture_is_dropped_including_the_frames_before_it_was_noticed():
    d = deltas([4.0] * 40 + [0.0] * 40 + [4.0] * 40)
    config = video.IngestConfig(min_steps=5, static_steps=10)
    runs = video.segment(d, np.zeros(120, bool), config)
    assert runs == [(0, 40), (80, 120)]


def test_dark_frames_end_a_run():
    dark = np.zeros(100, bool)
    dark[40:60] = True
    runs = video.segment(deltas([4.0] * 100), dark, video.IngestConfig(min_steps=5))
    assert runs == [(0, 40), (60, 100)]


def test_runs_shorter_than_the_minimum_are_not_worth_keeping():
    dark = np.zeros(100, bool)
    dark[10:90] = True
    assert video.segment(deltas([4.0] * 100), dark, video.IngestConfig(min_steps=20)) == []


def test_the_filter_chain_crops_before_it_scales_with_area_averaging():
    chain = video.filter_chain((140, 0, 800, 1920), video.IngestConfig())
    assert chain == f"fps=15.0,crop=1920:800:0:140,scale={spec.PIXELS_SHAPE[1]}:{spec.PIXELS_SHAPE[0]}:flags=area"
    padded = video.filter_chain((0, 0, 960, 1280), video.IngestConfig(fit="pad"))
    assert "force_original_aspect_ratio=decrease" in padded and padded.endswith(":black")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_ingesting_a_real_file_produces_loadable_clips(tmp_path):
    # Two seconds of moving test pattern, letterboxed, then a hard cut to two seconds of another.
    path = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1280x536:rate=30:duration=2",
         "-f", "lavfi", "-i", "smptebars=size=1280x536:rate=30:duration=2", "-filter_complex",
         "[0:v]pad=1280:720:0:92[a];[1:v]pad=1280:720:0:92[b];[a][b]concat=n=2:v=1[out]",
         "-map", "[out]", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    written = video.ingest(path, tmp_path / "clips", video.IngestConfig(min_steps=5, static_steps=45))
    assert written, "expected at least one clip"
    clip = load_clip(written[0])
    assert clip.frames.shape[1:] == spec.PIXELS_SHAPE and not clip.labelled
    assert clip.manifest["source"]["kind"] == "video"
    assert clip.manifest["source"]["bars"][0] >= 80  # the letterbox was found and removed
    assert not any((tmp_path / "clips").glob(".staging*"))
