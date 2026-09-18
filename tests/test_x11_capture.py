"""Capture tests against a real X server (Xvfb), which is what XWayland looks like to an X11 client."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from zombiesai import spec

pytestmark = pytest.mark.skipif(
    shutil.which("Xvfb") is None, reason="Xvfb is not installed; capture needs an X server to test against"
)

TITLE = "Call of Duty World at War"
WIDTH, HEIGHT = 640, 480
HELPER = Path(__file__).with_name("_x11_window.py")


@pytest.fixture(scope="module")
def x_display():
    for number in range(90, 100):
        display = f":{number}"
        server = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1280x720x24"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if server.poll() is not None:
                break  # display already taken; try the next one
            if Path(f"/tmp/.X11-unix/X{number}").exists():
                try:
                    yield display
                finally:
                    server.terminate()
                    server.wait(timeout=5)
                return
            time.sleep(0.1)
        server.terminate()
    pytest.skip("could not start Xvfb on any display")


def spawn_window(display: str, title: str = TITLE) -> tuple[subprocess.Popen, int]:
    helper = subprocess.Popen(
        [sys.executable, str(HELPER), title, str(WIDTH), str(HEIGHT)],
        env={**os.environ, "DISPLAY": display},
        stdout=subprocess.PIPE,
    )
    line = helper.stdout.readline()
    assert line, "the helper never reported its window id"
    return helper, int(line.strip())


@pytest.fixture(scope="module")
def window(x_display):
    helper, window_id = spawn_window(x_display)
    try:
        yield window_id
    finally:
        helper.terminate()
        helper.wait(timeout=5)


def grabber(window, **kwargs):
    from zombiesai.demos.x11_capture import X11Grabber

    return X11Grabber(window=window, **kwargs)


def test_a_window_is_found_by_the_name_the_game_gives_it(x_display, window):
    from zombiesai.demos.x11_capture import X11Error, find_window, list_windows

    found = find_window("world at war", display=x_display)
    assert found.id == window and (found.width, found.height) == (WIDTH, HEIGHT)
    assert any(w.id == window for w in list_windows(x_display))
    with pytest.raises(X11Error):
        find_window("no such game", display=x_display)


@pytest.mark.parametrize("shm", [True, False])
def test_the_pixels_come_back_as_rgb_the_right_way_up(x_display, window, shm):
    capture = grabber(window, display=x_display, shm=shm)
    try:
        assert capture.backend == ("xshm" if shm else "xgetimage")
        frame = capture.grab()
        assert frame.shape == (HEIGHT, WIDTH, 3) and frame.dtype == np.uint8
        quarter_x, quarter_y = WIDTH // 4, HEIGHT // 4
        # The helper paints red, green, blue, grey clockwise from the top left. Getting BGRX byte order or
        # the row order wrong shows up here and nowhere else.
        assert tuple(frame[quarter_y, quarter_x]) == (255, 0, 0)
        assert tuple(frame[quarter_y, 3 * quarter_x]) == (0, 255, 0)
        assert tuple(frame[3 * quarter_y, quarter_x]) == (0, 0, 255)
        assert tuple(frame[3 * quarter_y, 3 * quarter_x]) == (128, 128, 128)
    finally:
        capture.close()


def test_both_backends_agree(x_display, window):
    fast, slow = grabber(window, display=x_display), grabber(window, display=x_display, shm=False)
    try:
        np.testing.assert_array_equal(fast.grab(), slow.grab())
    finally:
        fast.close()
        slow.close()


def test_a_region_is_cropped_by_the_server(x_display, window):
    capture = grabber(window, display=x_display, region=(10, 20, 100, 50))
    try:
        frame = capture.grab()
        assert frame.shape == (50, 100, 3) and capture.size == (100, 50)
        assert tuple(frame[0, 0]) == (255, 0, 0)  # still inside the top-left quadrant
    finally:
        capture.close()


def test_repeated_grabs_do_not_hand_back_the_same_buffer(x_display, window):
    """The shared segment is overwritten in place, so a view into it would change under the caller."""
    capture = grabber(window, display=x_display)
    try:
        first = capture.grab()
        second = capture.grab()
        assert first.base is not second.base or not np.shares_memory(first, second)
    finally:
        capture.close()


def test_screen_capture_picks_x11_and_produces_a_policy_frame(x_display, window, monkeypatch):
    from zombiesai.demos.capture import ScreenCapture

    monkeypatch.setenv("DISPLAY", x_display)
    capture = ScreenCapture(window=window, display=x_display)
    try:
        assert capture.backend == "x11"
        frame = capture.read()
        assert frame.shape == spec.PIXELS_SHAPE and frame.dtype == np.uint8
        assert capture.describe()["source"]["window"] == f"0x{window:x}"
    finally:
        capture.close()


def test_a_missing_window_says_so_instead_of_crashing(x_display):
    from zombiesai.demos.x11_capture import X11Error

    with pytest.raises(X11Error):
        grabber(0xDEADBEEF, display=x_display)


def test_a_window_that_disappears_raises_instead_of_killing_the_process(x_display):
    """Xlib's default error handler calls exit(). A game that crashes at 3am has to arrive as an exception
    the watchdog can act on, not as the agent process vanishing with it."""
    from zombiesai.demos.x11_capture import X11Error

    helper, window_id = spawn_window(x_display, "Doomed Window")
    capture = grabber(window_id, display=x_display)
    try:
        assert capture.grab().shape == (HEIGHT, WIDTH, 3)
        helper.terminate()
        helper.wait(timeout=5)
        time.sleep(0.3)  # the server reaps the client's windows asynchronously
        with pytest.raises(X11Error):
            for _ in range(5):
                capture.grab()
    finally:
        capture.close()


def test_a_demo_can_be_recorded_from_a_real_window(x_display, window, tmp_path):
    """Capture to clip on the Linux path: the X server is not World at War, but everything between the
    window and the clip store is the code that will run against it."""
    from zombiesai.demos.capture import ReplayInput, ScreenCapture
    from zombiesai.demos.clips import load_clip
    from zombiesai.demos.inputs import InputConfig, synthesize
    from zombiesai.demos.recorder import RecorderConfig, record

    config = InputConfig(counts_per_degree=10.0)
    dt = 1.0 / spec.DECISION_HZ
    actions = [spec.make_action(forward=1, yaw=6.0), spec.make_action(fire=1), spec.make_action(button="reload")]
    events = [e for k, a in enumerate(actions) for e in synthesize(a, k * dt, dt, config)]

    path = record(
        ScreenCapture(window=window, display=x_display),
        ReplayInput(events),
        tmp_path / "demo",
        RecorderConfig(max_steps=len(actions), realtime=False, input=config),
        progress_every=0,
    )
    clip = load_clip(path)
    assert clip.n_steps == len(actions)
    assert clip.frames.shape[1:] == spec.PIXELS_SHAPE
    np.testing.assert_array_equal(clip.actions, np.array(actions, dtype=np.uint8))
    assert clip.manifest["source"]["backend"] == "x11"
    assert clip.frames[0].any(), "the recorded frames are all black"
