import shutil
import subprocess

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.rl.best_episode import write_brain
from zombiesai.viz.overlay import build_ass, film_size, render

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def frames(n=45):
    action = [0] * len(spec.ACTION_HEADS)
    action[spec.FORWARD], action[spec.FIRE], action[spec.YAW] = 2, 1, 7  # forward, firing, turning right
    lines = [{"s": i / spec.DECISION_HZ, "v": i / 10, "sure": 0.5, "fire": 0.9, "a": action, "r": 0.0}
             for i in range(n)]
    lines[20].update(ev="gain", d=60)
    return lines


def test_the_overlay_has_the_clock_the_keys_and_the_points_change():
    header = {"fps": 15, "start_clock": {"train_s": 6.5 * 3600, "games": 812}, "moment": "box"}
    ass = build_ass(header, frames(), 1280, 720)
    assert "Hour 6 · game 812" in ass and "First mystery box" in ass
    assert "+60" in ass and "FIRE" in ass and "→ 14°" in ass
    # a stretch of the film is timed from its own start, and leaves out what is outside it
    part = build_ass(header, frames(), 1280, 720, start=1.0, end=2.0)
    assert "+60" in part and "Dialogue: 2,0:00:00.00" in part
    assert "+60" not in build_ass(header, frames(), 1280, 720, start=2.0, end=3.0)


@needs_ffmpeg
def test_the_overlay_is_burned_into_a_copy_of_the_film(tmp_path):
    film = tmp_path / "f.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c=gray:s=320x180:r={spec.DECISION_HZ}:d=3", "-pix_fmt", "yuv420p", str(film)], check=True)
    write_brain(film.with_suffix(".brain.jsonl"), frames(), {"fps": 15})
    out = render(film, tmp_path / "f.overlay.mp4")
    assert film_size(out) == (320, 180) and not out.with_suffix(".ass").exists()
    grab = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", "1.5", "-i", str(out), "-frames:v", "1",
                           "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    assert np.frombuffer(grab, np.uint8).std() > 5  # not the flat grey it was
