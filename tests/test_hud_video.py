import json
import shutil
import threading

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import hud_video
from zombiesai.demos.clips import ClipWriter, load_clip
from zombiesai.demos.hud_video import HudVideo, PackError, pack_hud

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

SHAPES = {"points_ammo": (18, 24, 3), "round": (15, 10, 3)}  # odd sizes on purpose: 4:4:4 has no halving


def _crop(shape, k: int) -> np.ndarray:
    """A moving gradient with a red bar and a white 'digit' -- scene behind text, like the real boxes."""
    h, w, _ = shape
    y, x = np.mgrid[0:h, 0:w]
    crop = np.stack([(x * 6 + k * 3) % 200, (y * 8 + k) % 180, np.full((h, w), 40 + k % 30)], -1)
    crop[1:3, :] = (200, 20, 20)
    crop[h // 2 : h // 2 + 4, 2 + k % 4 : 5 + k % 4] = 255
    return crop.astype(np.uint8)


def _clip(tmp_path, steps: int = 25):
    writer = ClipWriter(tmp_path / "c", source={"kind": "test"})
    for k in range(steps):
        writer.add(np.full(spec.PIXELS_SHAPE, k, np.uint8), hud={n: _crop(s, k) for n, s in SHAPES.items()})
    writer.close()
    return tmp_path / "c"


def test_packing_replaces_the_raw_crops_with_video_that_reads_back_the_same_way(tmp_path):
    path = _clip(tmp_path)
    written = pack_hud(path, chunk_steps=10, say=lambda _: None)  # 3 chunks, the last one short
    assert set(written) == set(SHAPES)
    assert not list(path.glob("hud_*.u8")) and not list(path.glob("*.packing"))
    assert sorted(p.name for p in (path / "hud_round").iterdir()) == ["00000.mkv", "00001.mkv", "00002.mkv"]

    clip = load_clip(path)
    assert clip.n_steps == 25 and clip.hud_regions == sorted(SHAPES)
    for name, shape in SHAPES.items():
        crops = clip.hud(name)
        assert isinstance(crops, HudVideo) and crops.shape == (25, *shape) and len(crops) == 25
        want = np.stack([_crop(shape, k) for k in range(25)]).astype(int)
        got = np.asarray(crops).astype(int)
        assert got.shape == want.shape and np.abs(got - want).mean() < hud_video.MAX_MEAN_ERR
    crops = clip.hud("points_ammo")
    everything = np.asarray(crops)
    np.testing.assert_array_equal(crops[12], everything[12])
    np.testing.assert_array_equal(crops[-1], everything[24])
    np.testing.assert_array_equal(crops[8:13], everything[8:13])  # across a chunk boundary
    np.testing.assert_array_equal(crops[[24, 0, 11]], everything[[24, 0, 11]])
    np.testing.assert_array_equal(crops[:, 0, 0, 0], everything[:, 0, 0, 0])
    np.testing.assert_array_equal(crops[np.arange(25) % 2 == 0], everything[::2])
    with pytest.raises(IndexError):
        crops[25]
    info = json.loads((path / "clip.json").read_text())["hud"]["video"]["round"]
    assert info["n_steps"] == 25 and info["chunk_steps"] == 10 and info["codec"] == "libx264"


def test_packing_twice_does_nothing_the_second_time(tmp_path):
    path = _clip(tmp_path, steps=5)
    pack_hud(path, say=lambda _: None)
    assert pack_hud(path, say=lambda _: None) == {}
    assert load_clip(path).hud("round").shape == (5, *SHAPES["round"])


def test_a_region_that_fails_verification_keeps_its_raw_crops(tmp_path, monkeypatch):
    path = _clip(tmp_path, steps=5)
    before = (path / "clip.json").read_text()
    monkeypatch.setattr(hud_video, "MAX_MEAN_ERR", -1.0)
    with pytest.raises(PackError, match="raw crops stay"):
        pack_hud(path, say=lambda _: None)
    assert (path / "hud_round.u8").exists() and (path / "hud_points_ammo.u8").exists()
    assert not (path / "hud_round").exists() and not list(path.glob("*.packing"))
    assert (path / "clip.json").read_text() == before
    assert isinstance(load_clip(path).hud("round"), np.ndarray)


def test_kept_raw_crops_are_ignored_once_packed_and_cleared_by_the_next_pack(tmp_path):
    path = _clip(tmp_path, steps=5)
    pack_hud(path, keep_raw=True, say=lambda _: None)  # what an interruption after the manifest write leaves
    assert (path / "hud_round.u8").exists()
    assert isinstance(load_clip(path).hud("round"), HudVideo)
    pack_hud(path, say=lambda _: None)
    assert not list(path.glob("hud_*.u8"))


def test_a_stopped_pack_drops_its_half_made_video(tmp_path):
    path = _clip(tmp_path, steps=5)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(PackError, match="stopped"):
        hud_video._pack_region(path, "round", SHAPES["round"], 5, hud_video.CRF, 10, cancel)
    assert (path / "hud_round.u8").exists() and not (path / "hud_round.packing").exists()


def test_a_clip_without_hud_crops_has_nothing_to_pack(tmp_path):
    writer = ClipWriter(tmp_path / "c", source={"kind": "test"})
    writer.add(np.zeros(spec.PIXELS_SHAPE, np.uint8))
    writer.close()
    assert pack_hud(tmp_path / "c", say=lambda _: None) == {}
