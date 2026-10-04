import shutil
import subprocess

import numpy as np
import pytest

from zombiesai.viz.grid import GAP, layout, mosaic


def test_the_grid_is_near_16_9_and_fits_the_width():
    assert layout(1, 2560, 1440, 1920) == (1, 1, 2)
    assert layout(4, 2560, 1440, 1920) == (2, 2, 3)
    cols, rows, step = layout(6, 2560, 1440, 1920)
    assert (cols, rows) == (3, 2) and cols * (2560 // step) + (cols - 1) * GAP <= 1920
    assert layout(8, 2560, 1440, 1920)[:2] == (3, 3)


def test_tiles_go_in_reading_order_and_a_missing_one_is_dark():
    tiles = [np.full((10, 16, 4), v, np.uint8) for v in (50, 100, 150)]
    out = mosaic([tiles[0], None, tiles[2]], cols=2, rows=2, tile_h=10, tile_w=16)
    assert out.shape == (10 * 2 + GAP, 16 * 2 + GAP, 4)
    assert out[0, 0, 0] == 50 and out[0, 16 + GAP, 0] == 0 and out[10 + GAP, 0, 0] == 150


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_games_are_filmed_together_for_as_long_as_asked(tmp_path, monkeypatch):
    from zombiesai.demos import x11_capture
    from zombiesai.realgame import viewer
    from zombiesai.viz.grid import record

    class FakeGrabber:
        def __init__(self, display):
            self.value = 60 * int(display[1:]) % 256
            self.size = (320, 180)

        def grab_bgrx(self):
            return np.full((180, 320, 4), self.value, np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(viewer, "running_screens", lambda: [viewer.Screen(f":{n}", 320, 180) for n in (1, 2, 3)])
    monkeypatch.setattr(x11_capture, "X11Grabber", FakeGrabber)
    out = record(tmp_path / "grid.mp4", 1.0, fps=10, max_width=400, say=lambda m: None)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames,width,height", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert probe == "324,184,10"  # 2x2 tiles of 160x90 with the gaps; ten frames
