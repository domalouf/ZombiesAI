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


# ---------------------------------------------------------------------------------------- whole-multiple sizes


def exact_mean(rgb: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """The reference: each block's true mean, rounded half to even, in Python integers' worth of precision."""
    h, w, _ = rgb.shape
    sums = rgb.reshape(h // fy, fy, w // fx, fx, 3).astype(np.int64).sum(axis=(1, 3))
    n = fy * fx
    quotient, rest = np.divmod(sums, n)
    return (quotient + ((2 * rest > n) | ((2 * rest == n) & (quotient % 2 == 1)))).astype(np.uint8)


def bgrx_frame(rng, h: int, w: int, pad: int = 0) -> np.ndarray:
    """A BGRX frame as MIT-SHM hands it over: a view into rows padded to bytes_per_line."""
    rows = rng.integers(0, 256, (h, w * 4 + pad), dtype=np.uint8)
    return rows[:, : w * 4].reshape(h, w, 4)


@pytest.mark.parametrize("h, w, fy, fx", [(1440, 2560, 20, 20), (1080, 1920, 15, 15), (720, 1280, 10, 10),
                                          (270, 380, 2, 2), (280, 320, 2, 2), (64, 64, 16, 16), (40, 64, 4, 2),
                                          (34, 300, 17, 30), (600, 40, 300, 2)])
def test_whole_multiples_are_the_exact_mean_from_rgb_and_straight_from_bgrx(h, w, fy, fx):
    rng = np.random.default_rng(fy * 1000 + fx)
    bgrx = bgrx_frame(rng, h, w, pad=8)
    rgb = np.ascontiguousarray(bgrx[..., 2::-1])
    want = exact_mean(rgb, fy, fx)
    np.testing.assert_array_equal(fr.block_mean(rgb, fy, fx), want)
    np.testing.assert_array_equal(fr.block_mean(bgrx, fy, fx, channels=fr.BGRX), want)
    np.testing.assert_array_equal(fr.area_resize(bgrx, h // fy, w // fx, channels=fr.BGRX), want)
    white = np.full_like(bgrx, 255)
    assert (fr.block_mean(white, fy, fx, channels=fr.BGRX) == 255).all()  # no 16-bit lane overflows


@pytest.mark.parametrize("h, w, f", [(1440, 2560, 20), (1080, 1920, 15), (720, 1280, 10), (270, 380, 2)])
def test_the_float_path_differs_from_the_exact_mean_only_by_one_on_exact_ties(h, w, f):
    """The general path rounds a float32 sum, so a mean exactly half-way between two integers can land either
    side; everywhere else it is the exact mean. Halving and odd factors have no such cases at all."""
    rng = np.random.default_rng(f)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    exact = exact_mean(rgb, f, f).astype(int)
    old = fr._area_resize_float(rgb, h // f, w // f).astype(int)
    sums = rgb.reshape(h // f, f, w // f, f, 3).astype(np.int64).sum(axis=(1, 3))
    tie = 2 * (sums % (f * f)) == f * f
    assert np.abs(old - exact).max() <= 1 and (old == exact)[~tie].all()
    if f in (2, 15):
        np.testing.assert_array_equal(old, exact)
    if f == 20:
        assert 0 < (old != exact).mean() < 0.002  # about one value in 2,000


def test_policy_frames_from_bgrx_equal_policy_frames_from_rgb():
    rng = np.random.default_rng(3)
    bgrx = bgrx_frame(rng, 1440, 2560, pad=64)
    bgrx[:200] = 0  # a letterbox, so the box is not a whole multiple and the float path is taken
    rgb = np.ascontiguousarray(bgrx[..., 2::-1])
    for fit in fr.FITS:
        box = fr.crop_box(1440, 2560, fr.detect_bars(rgb), fit)
        np.testing.assert_array_equal(fr.to_policy_frame(bgrx, box, fit, channels=fr.BGRX),
                                      fr.to_policy_frame(rgb, box, fit))
    np.testing.assert_array_equal(fr.to_policy_frame(bgrx, channels=fr.BGRX), fr.to_policy_frame(rgb))
    whole = (0, 0, 1440, 2560)
    np.testing.assert_array_equal(fr.to_policy_frame(bgrx, whole, channels=fr.BGRX), exact_mean(rgb, 20, 20))


def _dense_float(image, out_h, out_w):
    """The general path as it was before halving was special-cased: the full dense column product."""
    h, w = image.shape[:2]
    x = image.astype(np.float32).reshape(h, w * 3)
    rows = (fr._row_weights(h, out_h) @ x).reshape(out_h, w, 3).transpose(0, 2, 1).reshape(out_h * 3, w)
    out = (rows @ fr._axis_weights(w, out_w).T).reshape(out_h, 3, out_w).transpose(0, 2, 1)
    return np.rint(out).clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize("h, w, out_h, out_w", [(43, 1946, 22, 973), (87, 563, 44, 282), (202, 285, 135, 190),
                                                (32, 1460, 21, 973), (37, 91, 13, 29)])
def test_non_multiples_are_the_float_path_bit_for_bit_from_rgb_or_bgrx(h, w, out_h, out_w):
    """The console crop halves the width (two weights of exactly 0.5, no other terms), which is done as one
    add and an exact halving; and a BGRX crop's row pass runs over all four channels. Neither changes a bit."""
    rng = np.random.default_rng(h + w)
    for k in range(4):
        frame = rng.integers(0, 256, (h + 3, w + 5, 4), dtype=np.uint8)
        if k % 2:
            frame[:] = rng.integers(0, 256)
        bgrx = frame[2 : 2 + h, 3 : 3 + w]
        rgb = np.ascontiguousarray(bgrx[..., 2::-1])
        want = _dense_float(rgb, out_h, out_w)
        np.testing.assert_array_equal(fr.area_resize(rgb, out_h, out_w), want)
        np.testing.assert_array_equal(fr.area_resize(bgrx, out_h, out_w, channels=fr.BGRX), want)
