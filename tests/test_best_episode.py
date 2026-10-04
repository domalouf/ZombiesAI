import json
import shutil
import subprocess

import numpy as np
import pytest

from zombiesai.rl.best_episode import (
    BestEpisodeRecorder,
    VideoRecorder,
    episode_rank,
    keep_if_best,
    read_best,
    video_frame,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def game(round_reached, points=None, kills=None, *, gained=0, hud_kills=0):
    return {"round_reached": round_reached, "end_points": points, "end_kills": kills,
            "points_gained": gained, "kills": hud_kills}


def test_games_rank_by_round_then_kills_then_points():
    ranked = sorted([game(3, 1400, 4), game(5, 600, 2), game(3, 900, 9), game(3, 1600, 4)], key=episode_rank)
    assert [episode_rank(g) for g in ranked] == [(3, 4, 1400), (3, 4, 1600), (3, 9, 900), (5, 2, 600)]


def test_the_scoreboard_counts_win_and_the_hud_stands_in_without_them():
    assert episode_rank(game(2, 1210, 7, gained=300, hud_kills=3)) == (2, 7, 1210)
    # the scoreboard's points count the 500 a game starts with; the HUD's gains do not
    assert episode_rank(game(2, gained=710, hud_kills=3)) == (2, 3, 1210)


def video(path, text=b"film"):
    path.write_bytes(text)
    return path


def test_a_better_game_replaces_the_best_and_a_worse_or_equal_one_is_thrown_away(tmp_path):
    assert keep_if_best(tmp_path, video(tmp_path / "a.mp4", b"a"), game(3, 900, 5), actor=0, episode=1)
    assert (tmp_path / "best.mp4").read_bytes() == b"a" and not (tmp_path / "a.mp4").exists()
    assert read_best(tmp_path)["rank"] == [3, 5, 900] and read_best(tmp_path)["episode"] == 1

    for name, worse in (("b", game(2, 5000, 50)), ("c", game(3, 900, 5))):  # lower round; an exact tie
        assert not keep_if_best(tmp_path, video(tmp_path / f"{name}.mp4"), worse, actor=1, episode=2)
        assert not (tmp_path / f"{name}.mp4").exists()
    assert (tmp_path / "best.mp4").read_bytes() == b"a"

    assert keep_if_best(tmp_path, video(tmp_path / "d.mp4", b"d"), game(3, 900, 6), actor=1, episode=3)
    best = json.loads((tmp_path / "best.json").read_text())
    assert (tmp_path / "best.mp4").read_bytes() == b"d"
    assert (best["round"], best["points"], best["kills"], best["actor"]) == (3, 900, 6, 1)

    # more kills beats more points
    assert not keep_if_best(tmp_path, video(tmp_path / "e.mp4"), game(3, 5000, 5), actor=2, episode=4)
    assert keep_if_best(tmp_path, video(tmp_path / "f.mp4", b"f"), game(3, 700, 7), actor=2, episode=5)
    assert (tmp_path / "best.mp4").read_bytes() == b"f"


def test_a_best_kept_with_the_old_rank_order_is_read_by_its_named_numbers(tmp_path):
    (tmp_path / "best.mp4").write_bytes(b"old")
    (tmp_path / "best.json").write_text(json.dumps(
        {"rank": [1, 510, 0], "round": 1, "points": 510, "kills": 0, "video": "best.mp4"}))
    # by the stale list, (1, 510, 0) would outrank (1, 3, 400); by kills it does not
    assert keep_if_best(tmp_path, video(tmp_path / "a.mp4", b"a"), game(1, 400, 3), actor=0, episode=1)
    assert (tmp_path / "best.mp4").read_bytes() == b"a"


def test_the_film_frame_is_every_other_pixel_of_the_grab_with_even_sides():
    grab = np.random.default_rng(0).integers(0, 256, (1440, 2566, 4), dtype=np.uint8)
    padded = np.zeros((1440, 2600, 4), np.uint8)
    padded[:, :2566] = grab
    for frame in (grab, padded[:, :2566]):  # a shared-memory grab's rows can be padded
        out = video_frame(frame, 720)
        assert out.shape == (720, 1282, 4) and out.flags.c_contiguous
        assert np.array_equal(out, grab[::2, ::2][:, :1282])
    assert video_frame(grab[:1080, :1920], 720).shape == (540, 960, 4)  # never taller than asked
    assert video_frame(grab[:72, :128], 720).shape == (72, 128, 4)


def frames_in(path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["nb_read_frames"]), stream["width"], stream["height"]


@needs_ffmpeg
def test_a_recorder_encodes_every_frame_it_is_given(tmp_path):
    recorder = VideoRecorder(tmp_path / "f.mp4", (72, 128, 3))
    for v in range(30):
        recorder.add(np.full((72, 128, 3), v * 8, np.uint8))
    recorder.add(np.zeros((36, 64, 3), np.uint8))  # another size: not a frame of this film
    assert recorder.close(), recorder.failed
    assert (recorder.frames, recorder.dropped) == (30, 1)
    assert frames_in(tmp_path / "f.mp4") == (30, 128, 72)
    assert not (tmp_path / "f.log").exists()


@needs_ffmpeg
def test_an_actor_keeps_only_its_best_finished_game(tmp_path):
    said = []
    recorder = BestEpisodeRecorder(tmp_path, actor=2, say=said.append)
    for episode, (summary, steps) in enumerate([(game(2, 800, 4), 10), (game(4, 700, 3), 20), (game(3), 15)]):
        recorder.start(episode)
        for _ in range(steps):
            recorder.add(np.full((72, 128, 4), 40 * episode, np.uint8))
        recorder.finish(summary)
    recorder.start(3)
    for _ in range(5):
        recorder.add(np.zeros((72, 128, 4), np.uint8))  # cut short by the actor stopping
    recorder.close()

    best = read_best(tmp_path / "best")
    assert (best["round"], best["episode"], best["actor"], best["frames"]) == (4, 1, 2, 20)
    assert frames_in(tmp_path / "best" / "best.mp4")[0] == 20
    assert sorted(p.name for p in (tmp_path / "best").iterdir()) == [".lock", "best.json", "best.mp4"]
    # games settle on threads of their own, so which of round 2 and 3 led for a moment is a race; round 4 wins
    assert any(m.startswith("new best game: round 4, 700 points, 3 kills") for m in said)
