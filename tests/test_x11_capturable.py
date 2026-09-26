"""Whether the game window can be captured yet, decided without an X server: the geometry is a pure function,
and `X11Grabber.is_capturable` runs against a fake Xlib that answers the three questions it asks.

(The real-server version is in test_x11_capture.py, which needs Xvfb.)"""

import numpy as np
import pytest

from zombiesai.demos.x11_capture import IS_VIEWABLE, X11Error, X11Grabber, region_on_screen

ROOT, GAME = 0x100, 0x2A00005
ROOT_SIZE = (2560, 1440)


def test_a_window_wholly_on_screen_is_capturable_and_one_hanging_off_any_edge_is_not():
    region = (0, 0, 1920, 1080)
    assert region_on_screen((0, 0), region, ROOT_SIZE)
    assert region_on_screen((640, 360), region, ROOT_SIZE)  # flush with the bottom-right corner
    assert not region_on_screen((641, 0), region, ROOT_SIZE)
    assert not region_on_screen((0, 361), region, ROOT_SIZE)
    assert not region_on_screen((-1, 0), region, ROOT_SIZE)
    assert not region_on_screen((0, -2000), region, ROOT_SIZE)  # parked above the desktop, workspace-style


def test_only_the_captured_region_has_to_be_on_screen():
    """With --region, the part of the window outside it is never asked for, so it may hang off the edge."""
    assert region_on_screen((-100, 0), (100, 0, 800, 600), ROOT_SIZE)
    assert not region_on_screen((-100, 0), (99, 0, 800, 600), ROOT_SIZE)


class FakeXlib:
    """Answers XGetWindowAttributes and XTranslateCoordinates for one game window and the root."""

    def __init__(self, *, exists=True, map_state=IS_VIEWABLE, origin=(0, 0), size=(1920, 1080)):
        self.exists, self.map_state, self.origin, self.size = exists, map_state, origin, size

    def get_attributes(self, display, window, attributes):
        attributes = attributes._obj
        if window == ROOT:
            attributes.width, attributes.height, attributes.map_state = *ROOT_SIZE, IS_VIEWABLE
            return 1
        if not self.exists:
            return 0
        attributes.width, attributes.height = self.size
        attributes.map_state = self.map_state
        return 1

    def translate_coordinates(self, display, source, destination, x, y, left, top, child):
        assert (source, destination) == (GAME, ROOT)
        left._obj.value, top._obj.value = self.origin
        return 1

    def sync(self, display, discard):
        return 0


def fake_grabber(x, grab_fails=False):
    grabber = X11Grabber.__new__(X11Grabber)  # skip __init__: it would open a real display
    grabber.x, grabber.display, grabber.root, grabber.window = x, 1, ROOT, GAME
    grabber.region = (0, 0, *x.size)
    grabber.grabs = 0

    def grab():
        grabber.grabs += 1
        if grab_fails:
            raise X11Error("grabbing window: BadMatch on request 130.4")
        return np.zeros((x.size[1], x.size[0], 3), dtype=np.uint8)

    grabber.grab = grab
    grabber.close = lambda: None
    return grabber


def test_a_viewable_window_on_screen_is_capturable_once_a_trial_grab_succeeds():
    grabber = fake_grabber(FakeXlib(origin=(320, 180)))
    assert grabber.is_capturable()
    assert grabber.grabs == 1


def test_a_window_on_another_workspace_is_not_capturable_and_is_not_even_grabbed():
    grabber = fake_grabber(FakeXlib(map_state=0))  # IsUnmapped
    assert not grabber.is_capturable()
    assert grabber.grabs == 0


def test_a_window_moved_partly_off_screen_is_not_capturable():
    assert not fake_grabber(FakeXlib(origin=(1000, 0))).is_capturable()


def test_a_grab_that_fails_anyway_means_not_yet():
    """The attribute checks say why when they can; the trial grab is the ground truth when they can't."""
    assert not fake_grabber(FakeXlib(), grab_fails=True).is_capturable()


def test_a_window_that_no_longer_exists_raises_rather_than_waiting_forever():
    with pytest.raises(X11Error, match="no longer exists"):
        fake_grabber(FakeXlib(exists=False)).is_capturable()


def test_screen_capture_on_a_monitor_backend_is_always_capturable():
    from zombiesai.demos.capture import ScreenCapture

    capture = ScreenCapture.__new__(ScreenCapture)
    capture.backend = "mss"
    assert capture.is_capturable()
