import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos.capture import ReplayInput
from zombiesai.demos.clips import ClipWriter, load_clip
from zombiesai.demos.hud_crops import HUD_REGIONS, crop_regions, crop_shape, region_box
from zombiesai.demos.inputs import InputConfig
from zombiesai.demos.recorder import RecorderConfig, record


def test_regions_are_screen_fractions_so_a_resolution_change_is_a_rescale():
    assert region_box(1440, 2560, HUD_REGIONS["points_ammo"]) == (2180, 1170, 380, 270)
    left, top, w, h = region_box(1080, 1920, HUD_REGIONS["points_ammo"])
    assert (left, top) == (1635, 878) and left + w == 1920 and top + h == 1080


def test_a_region_is_clipped_to_the_frame_and_an_empty_one_is_refused():
    assert region_box(100, 100, (0.9, 0.9, 0.5, 0.5)) == (90, 90, 10, 10)
    with pytest.raises(ValueError):
        region_box(100, 100, (1.0, 0.0, 0.1, 0.1))


def test_crops_come_from_the_right_corner_at_the_right_scale():
    frame = np.zeros((1440, 2560, 3), np.uint8)
    frame[1200:1260, 2300:2400] = (255, 0, 0)  # a red block where the points counter sits
    crops = crop_regions(frame, scale=0.5)
    assert crops["points_ammo"].shape == crop_shape(1440, 2560, HUD_REGIONS["points_ammo"], 0.5) == (135, 190, 3)
    assert crops["points_ammo"][..., 0].max() == 255
    assert crops["round"].max() == 0  # nothing in the other corner


def _hud(value: int) -> dict[str, np.ndarray]:
    return {"points_ammo": np.full((6, 8, 3), value, np.uint8), "round": np.full((5, 4, 3), value, np.uint8)}


def _frame(value: int) -> np.ndarray:
    return np.full(spec.PIXELS_SHAPE, value, np.uint8)


def test_hud_crops_round_trip_through_the_clip_store(tmp_path):
    writer = ClipWriter(tmp_path / "c", source={"kind": "test"})
    for k in range(5):
        writer.add(_frame(k), hud=_hud(k))
    writer.close()
    clip = load_clip(tmp_path / "c")
    assert clip.hud_regions == ["points_ammo", "round"]
    assert clip.hud("points_ammo").shape == (5, 6, 8, 3)
    np.testing.assert_array_equal(clip.hud("round")[:, 0, 0, 0], np.arange(5))
    assert clip.hud("missing") is None


def test_a_clip_without_hud_crops_has_no_regions(tmp_path):
    writer = ClipWriter(tmp_path / "c", source={"kind": "test"})
    writer.add(_frame(0))
    writer.close()
    clip = load_clip(tmp_path / "c")
    assert clip.hud_regions == [] and clip.hud("points_ammo") is None


@pytest.mark.parametrize(
    "steps",
    [
        [_hud(0), None],  # stopped arriving
        [None, _hud(1)],  # started mid-clip
        [_hud(0), {"points_ammo": _hud(1)["points_ammo"]}],  # a region went missing
        [_hud(0), {**_hud(1), "round": np.zeros((5, 5, 3), np.uint8)}],  # a region changed shape
    ],
)
def test_hud_crops_that_would_drift_out_of_step_with_the_frames_are_refused(tmp_path, steps):
    writer = ClipWriter(tmp_path / "c", source={"kind": "test"})
    writer.add(_frame(0), hud=steps[0])
    with pytest.raises(ValueError):
        writer.add(_frame(1), hud=steps[1])


def test_a_crash_that_left_a_region_short_trims_the_clip_to_match(tmp_path):
    writer = ClipWriter(tmp_path / "c", source={"kind": "test"})
    for k in range(4):
        writer.add(_frame(k), hud=_hud(k))
    writer.close()
    path = tmp_path / "c" / "hud_round.u8"
    path.write_bytes(path.read_bytes()[: -5 * 4 * 3])
    clip = load_clip(tmp_path / "c")
    assert clip.n_steps == 3 and len(clip.hud("points_ammo")) == 3


class _Countdown:
    """A frame source whose k-th read returns frame k and HUD crops k, as a screen capture does."""

    def __init__(self, n):
        self.k, self.n, self.last_hud = -1, n, None

    def read(self):
        self.k += 1
        if self.k >= self.n:
            raise StopIteration
        self.last_hud = _hud(self.k)
        return _frame(self.k)

    def close(self):
        pass


def test_the_recorder_saves_each_steps_hud_from_the_same_grab_as_its_frame(tmp_path):
    config = RecorderConfig(max_steps=20, realtime=False, input=InputConfig(counts_per_degree=10.0))
    path = record(_Countdown(8), ReplayInput([]), tmp_path / "demo", config, progress_every=0)
    clip = load_clip(path)
    assert clip.n_steps == 7
    np.testing.assert_array_equal(clip.frames[:, 0, 0, 0], np.arange(7))
    np.testing.assert_array_equal(clip.hud("points_ammo")[:, 0, 0, 0], np.arange(7))
    np.testing.assert_array_equal(clip.hud("round")[:, 0, 0, 0], np.arange(7))
