"""Losing the game window mid-recording: a workspace switch costs steps, not the session.

Everything here runs against fakes -- a scripted grabber for ScreenCapture and the recorder, and a fake Xlib for
X11Grabber's own recovery logic -- so none of it needs an X server, let alone the game.
"""

import ctypes

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import x11_capture
from zombiesai.demos.capture import CaptureLost, ReplayInput, ScreenCapture
from zombiesai.demos.clips import FLAG_BAD_STEP, FLAG_CLIP_START, load_clip
from zombiesai.demos.hud_crops import HUD_REGIONS
from zombiesai.demos.inputs import InputConfig
from zombiesai.demos.recorder import RecorderConfig, record
from zombiesai.demos.x11_capture import WindowGone, WindowUnavailable, X11Error, X11Grabber

HEIGHT, WIDTH = 144, 256
LOST, GONE = "lost", "gone"
CONFIG = InputConfig(counts_per_degree=10.0)


def value(k: int) -> int:
    return 60 + 10 * k  # bright enough that no row or column reads as a letterbox bar


class ScriptedGrabber:
    """Stands in for X11Grabber: each grab plays the next entry of a script. An int is a good frame filled
    with `value(int)`, a (height, width, int) tuple is a good frame at another size, LOST is the BadMatch a
    workspace switch produces, and GONE is a destroyed window. Past the end of the script the source is done."""

    script: list = []

    def __init__(self, window=None, display=None, region=None):
        self.steps = iter(type(self).script)

    def grab(self):
        step = next(self.steps)  # StopIteration ends a recording, as it does for ClipPlayback
        if step == LOST:
            raise WindowUnavailable("grabbing window 0x4800006: BadMatch on request 130.4")
        if step == GONE:
            raise WindowGone("window 0x4800006 no longer exists")
        height, width, k = step if isinstance(step, tuple) else (HEIGHT, WIDTH, step)
        return np.full((height, width, 3), value(k), np.uint8)

    def describe(self):
        return {"kind": "x11", "window": "0x4800006"}

    def close(self):
        pass


@pytest.fixture
def capture(monkeypatch):
    def make(script, hud=True):
        monkeypatch.setattr(ScriptedGrabber, "script", script)
        monkeypatch.setattr(x11_capture, "X11Grabber", ScriptedGrabber)
        return ScreenCapture(backend="x11", hud_regions=HUD_REGIONS if hud else None, hud_scale=0.5)

    return make


def run(source, tmp_path, steps=100, **config):
    config = RecorderConfig(max_steps=steps, realtime=False, input=CONFIG, **config)
    return load_clip(record(source, ReplayInput([]), tmp_path / "demo", config, progress_every=0))


def test_a_lost_window_repeats_the_last_good_frame_and_says_it_is_stale(capture):
    source = capture([0, 1, LOST, LOST, 4])
    first, second = source.read(), source.read()
    good_hud = source.last_hud
    assert not source.last_stale and source.stale_reason is None
    for _ in range(2):
        repeat = source.read()
        assert source.last_stale and "BadMatch" in source.stale_reason
        np.testing.assert_array_equal(repeat, second)
        assert source.last_hud is good_hud  # the crops that went with that frame, not new ones
    assert not (first == second).all()
    back = source.read()
    assert not source.last_stale and back[0, 0, 0] == value(4)
    assert source.last_hud["round"][0, 0, 0] == value(4)


def test_with_no_good_frame_to_fall_back_on_the_error_propagates(capture, tmp_path):
    with pytest.raises(WindowUnavailable):
        capture([LOST, 1]).read()
    with pytest.raises(WindowGone):
        capture([GONE]).read()
    with pytest.raises(WindowUnavailable):
        run(capture([LOST, 1, 2]), tmp_path)


def test_a_destroyed_window_is_capture_lost_for_good_not_a_stale_frame(capture):
    source = capture([0, GONE])
    source.read()
    with pytest.raises(CaptureLost, match="no longer exists"):
        source.read()


def test_a_window_back_at_another_size_stays_stale_until_it_is_put_back(capture):
    """The HUD crops' shapes are fixed for the life of a clip, so a resized window cannot simply resume."""
    source = capture([0, (72, 128, 1), 2])
    source.read()
    stale = source.read()
    assert source.last_stale and "128x72" in source.stale_reason and "256x144" in source.stale_reason
    assert stale[0, 0, 0] == value(0) and source.last_hud["round"][0, 0, 0] == value(0)
    assert source.read()[0, 0, 0] == value(2) and not source.last_stale


def test_stale_steps_are_flagged_bad_on_the_step_their_frame_belongs_to(capture, tmp_path, capsys):
    """Step k pairs the frame read at tick k with the input of the interval after it, so a stale read at
    tick k must flag step k -- not k+1, which is where it would land if staleness were read off the source
    at write time instead of travelling with the pending frame."""
    clip = run(capture([0, 1, LOST, LOST, 4, 5, 6]), tmp_path)
    assert clip.n_steps == 6  # reads 0..6; the last one has no interval after it to pair with
    bad = (clip.flags & FLAG_BAD_STEP) != 0
    np.testing.assert_array_equal(bad, [False, False, True, True, False, False])
    # The stale steps hold the last good frame and HUD crops, byte for byte.
    np.testing.assert_array_equal(clip.frames[:, 0, 0, 0], [value(k) for k in (0, 1, 1, 1, 4, 5)])
    np.testing.assert_array_equal(clip.hud("round")[:, 0, 0, 0], [value(k) for k in (0, 1, 1, 1, 4, 5)])
    # The first good frame after the outage has only repeats behind it: a frame stack must clamp there.
    np.testing.assert_array_equal((clip.flags & FLAG_CLIP_START) != 0, [True, False, False, False, True, False])
    summary = clip.manifest["summary"]
    assert summary["stale_steps"] == 2 and summary["capture_outages"] == 1 and summary["capture_lost"] is None
    out = capsys.readouterr().out
    assert "capture lost" in out and "BadMatch" in out and "capture back" in out


def test_every_outage_is_counted_and_the_recording_runs_to_the_end(capture, tmp_path):
    clip = run(capture([0, LOST, 2, LOST, LOST, LOST, 6, 7, (72, 128, 8), 9, 10]), tmp_path)
    assert clip.n_steps == 10
    bad = (clip.flags & FLAG_BAD_STEP) != 0
    np.testing.assert_array_equal(np.flatnonzero(bad), [1, 3, 4, 5, 8])
    summary = clip.manifest["summary"]
    assert summary["stale_steps"] == 5 and summary["capture_outages"] == 3
    assert summary["stale_rate"] == pytest.approx(0.5)


def test_a_sustained_outage_stops_the_recording_cleanly(capture, tmp_path, capsys):
    limit = 1.0  # seconds, i.e. 15 decisions
    clip = run(capture([0, 1] + [LOST] * 200), tmp_path, steps=500, max_outage_seconds=limit)
    outage = int(limit * spec.DECISION_HZ)
    # Two good steps, then stale steps until the outage reached the limit -- and nothing after.
    assert clip.n_steps == 2 + outage - 1
    assert not (clip.flags[:2] & FLAG_BAD_STEP).any() and (clip.flags[2:] & FLAG_BAD_STEP).all()
    summary = clip.manifest["summary"]
    assert clip.manifest["status"] == "closed"
    assert summary["capture_lost"].startswith("no capture for 1s") and summary["stale_steps"] == outage - 1
    assert "stopping the recording" in capsys.readouterr().out


def test_a_destroyed_window_stops_the_recording_at_once_and_keeps_what_was_recorded(capture, tmp_path, capsys):
    clip = run(capture([0, 1, 2, GONE, 4]), tmp_path)
    assert clip.n_steps == 2 and clip.manifest["status"] == "closed"
    assert "no longer exists" in clip.manifest["summary"]["capture_lost"]
    assert "capture lost for good" in capsys.readouterr().out


# X11Grabber's own recovery, against a fake Xlib. The grabber is built without __init__ so that nothing here
# loads libX11 or opens a display.


class FakeXlib:
    """Just enough of Xlib for X11Grabber.grab's recovery path: window attributes and error syncing."""

    def __init__(self, width=WIDTH, height=HEIGHT):
        self.exists, self.viewable, self.width, self.height = True, True, width, height

    def get_attributes(self, display, window, attributes):
        if not self.exists:
            x11_capture._errors.append(f"BadWindow on request 3.0 for resource 0x{window:x}")
            return 0
        attrs = attributes._obj
        attrs.width, attrs.height = self.width, self.height
        attrs.map_state = x11_capture.IS_VIEWABLE if self.viewable else 0
        return 1

    def sync(self, display, discard):
        return 1


def fake_grabber(x, region=None):
    grabber = X11Grabber.__new__(X11Grabber)
    grabber.x, grabber.display, grabber.window, grabber.screen = x, ctypes.c_void_p(1), 0x4800006, 0
    grabber.shm_info = grabber.image = grabber._buffer = None
    grabber.region = region or (0, 0, x.width, x.height)
    grabber._follows_window, grabber._lost, grabber._shm = region is None, False, True
    grabber.shm_rebuilds = 0

    def read():
        # The real read raises BadMatch whenever the region overhangs the window or the window is unviewable.
        left, top, width, height = grabber.region
        if not (x.exists and x.viewable) or left + width > x.width or top + height > x.height:
            raise X11Error(f"grabbing window 0x{grabber.window:x}: BadMatch on request 130.4")
        return np.zeros((height, width, 3), np.uint8)

    def attach():
        grabber.shm_rebuilds += 1
        grabber.image = object()

    grabber._read = read
    grabber._attach_shm = attach
    grabber._release_shm = lambda: setattr(grabber, "image", None)
    return grabber


def test_a_grabber_whose_window_is_hidden_says_so_and_recovers_when_it_is_shown():
    x = FakeXlib()
    grabber = fake_grabber(x)
    assert grabber.grab().shape == (HEIGHT, WIDTH, 3)
    x.viewable = False
    with pytest.raises(WindowUnavailable, match="not viewable"):
        grabber.grab()
    with pytest.raises(WindowUnavailable):
        grabber.grab()
    x.viewable = True
    assert grabber.grab().shape == (HEIGHT, WIDTH, 3)
    assert grabber.shm_rebuilds == 0  # same size: the shared segment is still good


@pytest.mark.parametrize("size", [(128, 72), (512, 288)])
def test_a_grabber_follows_its_window_through_a_resize_and_rebuilds_the_shared_image(size):
    """Smaller would be BadMatch forever; larger would grab only the old top-left corner without complaint."""
    x = FakeXlib()
    grabber = fake_grabber(x)
    grabber.grab()
    x.viewable = False
    with pytest.raises(WindowUnavailable):
        grabber.grab()
    x.viewable, (x.width, x.height) = True, size
    assert grabber.grab().shape == (size[1], size[0], 3)
    assert grabber.size == size and grabber.shm_rebuilds == 1 and grabber.backend == "xshm"


def test_a_grabber_does_not_move_a_region_the_caller_chose():
    x = FakeXlib()
    grabber = fake_grabber(x, region=(0, 0, 200, 100))
    grabber.grab()
    x.width, x.height = 128, 72
    with pytest.raises(WindowUnavailable):
        grabber.grab()
    with pytest.raises(WindowUnavailable):
        grabber.grab()
    assert grabber.region == (0, 0, 200, 100) and grabber.shm_rebuilds == 0


def test_a_grabber_whose_window_was_destroyed_raises_window_gone():
    x = FakeXlib()
    grabber = fake_grabber(x)
    grabber.grab()
    x.exists = False
    with pytest.raises(WindowGone, match="no longer exists"):
        grabber.grab()
    assert not x11_capture._errors  # reported, not left behind to poison the next check


def test_re_deriving_the_play_marks_keeps_the_clip_start_an_outage_set():
    """requantize rewrites the mark key's flags from the log; the fresh start after a capture outage is not
    the mark key's, and a frame stack must still not reach back across the repeats before it."""
    from zombiesai.demos.clips import FLAG_BAD_STEP, FLAG_CLIP_START, FLAG_NOT_PLAYING, with_play_marks

    flags = np.zeros(10, np.uint8)
    flags[0] = FLAG_CLIP_START
    flags[2:4] = FLAG_BAD_STEP  # stale frames during an outage
    flags[4] = FLAG_CLIP_START  # first good frame after it
    flags[6:8] = FLAG_NOT_PLAYING  # an old F8 stretch...
    flags[8] = FLAG_CLIP_START  # ...and the resume it set
    marked = np.zeros(10, bool)  # re-derived under a key that was never pressed
    out = with_play_marks(flags, marked)
    assert out[4] & FLAG_CLIP_START and out[0] & FLAG_CLIP_START
    assert not out[8] & FLAG_CLIP_START and not (out & FLAG_NOT_PLAYING).any()
    assert (out[2:4] & FLAG_BAD_STEP).all()


def test_a_picture_that_stops_changing_is_stale_even_though_every_grab_succeeds(capture):
    """A window on a hidden Hyprland workspace can still be grabbed: XWayland hands back its last frame. The
    first real smoke test recorded 48 s of that as play."""
    source = capture([0, 1, 2, 2, 2, 2, 3, 4])
    stale = []
    for _ in range(8):
        source.read()
        stale.append(source.last_stale)
    # The third identical frame in a row is where it becomes an outage; a fresh picture ends it at once.
    assert stale == [False, False, False, False, True, True, False, False]
    assert "not changed" in source.stale_reason if source.last_stale else True


def test_a_frozen_picture_is_flagged_bad_in_the_recording(capture, tmp_path):
    clip = run(capture([0, 1, 2, 3] + [4] * 10 + [5, 6, 7]), tmp_path)
    bad = (clip.flags & FLAG_BAD_STEP) != 0
    assert bad[6:13].all() and not bad[:5].any() and not bad[-2:].any()
    assert clip.manifest["summary"]["capture_outages"] == 1


def test_waiting_for_the_window_needs_a_picture_that_is_moving(capture, monkeypatch):
    source = capture([0, 0, 0, 1, 2])
    monkeypatch.setattr(ScriptedGrabber, "is_capturable", lambda self: True, raising=False)
    answers = [source.is_capturable() for _ in range(5)]
    # First look has nothing to compare with; then frozen, frozen, and live once it starts changing.
    assert answers == [False, False, False, True, True]
