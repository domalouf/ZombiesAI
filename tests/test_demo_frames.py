import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import frames as fr


def gradient(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), np.full_like(x, 128)], -1).astype(np.uint8)


def test_area_resize_averages_exactly():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    out = fr.area_resize(image, 2, 2)
    expected = image.reshape(2, 2, 2, 2, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(out, np.rint(expected), atol=0)


def test_area_resize_is_identity_at_the_same_size_and_keeps_dtype():
    image = gradient(72, 128)
    out = fr.area_resize(image, 72, 128)
    np.testing.assert_array_equal(out, image)
    assert out.dtype == np.uint8


def test_area_resize_beats_nearest_on_thin_features():
    # A one-pixel bright column, moved one pixel between frames. Nearest sampling makes it flicker in and
    # out; area averaging keeps it present in both, which is the whole reason the plan insists on it.
    downsampled = []
    for x in (600, 601):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, x] = 255
        downsampled.append(fr.area_resize(frame, *spec.PIXELS_SHAPE[:2]).astype(int))
    assert all(d.max() > 0 for d in downsampled)
    assert abs(downsampled[0].sum() - downsampled[1].sum()) < 0.25 * downsampled[0].sum()


def test_detect_bars_finds_letterbox_and_ignores_dark_gameplay():
    frame = gradient(1080, 1920)
    frame[:120] = 0
    frame[-120:] = 0
    assert fr.detect_bars(frame) == (120, 120, 0, 0)
    dim = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    assert fr.detect_bars(dim) == (0, 0, 0, 0)


def test_detect_bars_uses_the_brightest_frame_not_the_darkest():
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    bright = np.full((100, 100, 3), 200, dtype=np.uint8)
    bright[:10] = 0
    assert fr.detect_bars(np.stack([dark, bright])) == (10, 0, 0, 0)


def test_all_black_frames_are_not_a_crop():
    assert fr.detect_bars(np.zeros((64, 64, 3), dtype=np.uint8)) == (0, 0, 0, 0)


@pytest.mark.parametrize("size", [(1080, 1920), (720, 1280), (1080, 1440), (1200, 3440), (480, 640)])
def test_to_policy_frame_always_produces_the_spec_shape(size):
    frame = gradient(*size)
    for fit in fr.FITS:
        out = fr.to_policy_frame(frame, fit=fit)
        assert out.shape == spec.PIXELS_SHAPE and out.dtype == np.uint8


def test_crop_fit_trims_a_4_3_frame_and_pad_keeps_all_of_it():
    frame = np.full((960, 1280, 3), 200, dtype=np.uint8)
    frame[-40:] = 30  # a HUD strip along the bottom
    cropped = fr.to_policy_frame(frame, fit="crop")
    padded = fr.to_policy_frame(frame, fit="pad")
    assert cropped[-1].mean() > 100  # the strip was cut away with the rest of the bottom
    assert padded[:, 0].mean() == 0 and padded[:, -1].mean() == 0  # pillarbox
    assert padded[..., 0].min() < 100  # the HUD strip survived


def test_frame_delta_separates_gameplay_from_cuts_and_freezes():
    base = gradient(72, 128)
    nudged = np.roll(base, 2, axis=1)
    cut = np.full_like(base, 255)
    deltas = fr.frame_delta(np.stack([base, base, nudged, cut]))
    assert deltas[0] == 0.0
    assert deltas[1] == 0.0  # frozen capture
    assert 0 < deltas[2] < deltas[3]


@pytest.mark.parametrize("shift", [-6, -1, 0, 3, 9])
def test_estimate_shift_recovers_horizontal_motion(shift):
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, size=(72, 256, 3), dtype=np.uint8)
    before = scene[:, 64:192]
    after = scene[:, 64 + shift : 192 + shift]
    found, confidence = fr.estimate_shift(before, after)
    assert found == -shift  # sampling further right means the scene moved left
    assert confidence > 0.1
