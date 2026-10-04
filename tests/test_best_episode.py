import json
import shutil
import subprocess
import time

import numpy as np
import pytest

from zombiesai.rl.best_episode import (
    BestEpisodeRecorder,
    VideoRecorder,
    episode_rank,
    final_stats,
    keep_if_best,
    read_best,
    video_frame,
    video_frame_rgb,
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


def test_best_json_keeps_the_games_final_stats_and_says_where_they_came_from(tmp_path):
    ended = {**game(2, 1210, 7), "end_headshots": 2, "seconds": 183.5, "shots": 90, "hits": 31,
             "reason": "down"}
    assert keep_if_best(tmp_path, video(tmp_path / "a.mp4"), ended, actor=0, episode=1)
    assert read_best(tmp_path)["stats"] == {"round": 2, "points": 1210, "kills": 7, "headshots": 2,
                                            "from": "scoreboard", "seconds": 183.5, "shots": 90, "hits": 31,
                                            "ended_by": "down"}
    # no scoreboard read: the HUD's numbers, and no headshots rather than a guess
    unread = {**game(1, gained=390, hud_kills=3), "end_headshots": None, "reason": "death"}
    assert final_stats(unread) == {"round": 1, "points": 890, "kills": 3, "headshots": None, "from": "hud",
                                   "seconds": None, "shots": None, "hits": None, "ended_by": "death"}


def test_an_rgb_grab_makes_the_same_film_frame_as_the_bgrx_grab_of_it():
    rgb = np.random.default_rng(1).integers(0, 256, (1440, 2566, 3), dtype=np.uint8)
    bgrx = np.zeros((1440, 2566, 4), np.uint8)
    bgrx[..., :3] = rgb[..., ::-1]
    assert np.array_equal(video_frame_rgb(rgb, 720), video_frame(bgrx, 720))
    assert video_frame_rgb(rgb[:1080, :1920], 720).shape == (540, 960, 4)


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


@needs_ffmpeg
def test_frames_go_in_the_slot_their_grab_time_falls_in(tmp_path):
    recorder = VideoRecorder(tmp_path / "f.mp4", (72, 128, 3))
    t0 = 1000.0
    # slots 0, 1, (2 missed: repeated), 3, a second grab in 3 (dropped), 4 a little late, 5
    for v, t in enumerate([0, 1, 3, 3.3, 4.2, 5]):
        recorder.add(np.full((72, 128, 3), v * 8, np.uint8), t0 + t / 15)
    assert recorder.close(), recorder.failed
    assert (recorder.frames, recorder.repeated, recorder.dropped, recorder.slots) == (5, 1, 1, 6)
    assert recorder.t0 == t0 and recorder.seconds == pytest.approx(6 / 15)
    assert frames_in(tmp_path / "f.mp4")[0] == 6


RATE, CHUNK = 48_000, 480


class ClickStream:
    """A sink monitor stand-in: stereo samples that really played from `t_start` on, silent but for a burst at
    each of `clicks` (monotonic times), handed over in 10 ms chunks that arrive late by up to 8 ms of jitter --
    every 10th on time, the way the promptest chunks pin the clock in a real capture."""

    def __init__(self, t_start, seconds, clicks, seed=0):
        n = int(seconds * RATE)
        self.samples = np.zeros((n, 2), "<i2")
        for t in clicks:
            i = int(round((t - t_start) * RATE))
            self.samples[i:i + 48] = 20_000
        self.t_start, self.rng, self.k = t_start, np.random.default_rng(seed), 0
        self.rate, self.channels = RATE, 2

    def open(self):
        pass

    def read(self):
        first = self.k * CHUNK
        if first >= len(self.samples):
            return b"", 0.0
        self.k += 1
        jitter = 0.0 if self.k % 10 == 0 else self.rng.uniform(0, 0.008)
        chunk = self.samples[first:first + CHUNK]
        return chunk.tobytes(), self.t_start + (first + len(chunk)) / RATE + jitter

    def describe(self):
        return {"latency_s": 0.0}

    def close(self):
        pass


def onsets(path) -> np.ndarray:
    """Film times (s) at which the decoded sound track goes loud."""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"],
                         capture_output=True, check=True).stdout
    loud = np.abs(np.frombuffer(pcm, "<i2").astype(np.int32)) > 8_000
    rising = np.flatnonzero(loud[1:] & ~loud[:-1]) + 1
    keep = rising[np.concatenate([[True], np.diff(rising) > RATE // 10])] if len(rising) else rising
    return keep / RATE


@needs_ffmpeg
def test_the_best_film_has_its_sound_lined_up_on_the_grab_clock(tmp_path):
    from zombiesai.rl.best_episode import AV_DELAY_S

    said = []
    t0 = 1000.0
    clicks = [1000.5, 1001.25, 1002.6]
    # the capture starts a little before the first frame and the frames come unevenly, one tick slow
    recorder = BestEpisodeRecorder(tmp_path, actor=1, say=said.append,
                                   sound=lambda: ClickStream(t0 - 0.05, 3.5, clicks))
    recorder.start(0)
    grabs = t0 + np.arange(45) / 15 + np.random.default_rng(1).uniform(0, 0.01, 45)
    grabs[0] = t0
    for k, t in enumerate(np.delete(grabs, 20)):  # a frame the encoder never got
        recorder.add(np.full((72, 128, 4), k, np.uint8), float(t))
    time.sleep(0.2)  # let the capture thread drain the stream
    recorder.finish(game(2, 900, 3))
    recorder.close()

    best = read_best(tmp_path / "best")
    assert best["sound"] == "aac", best
    assert best["repeated_frames"] == 1 and best["sound_heard"] > 0.99
    film = tmp_path / "best" / "best.mp4"
    assert frames_in(film)[0] == 45
    expected = np.array(clicks) - t0 + AV_DELAY_S
    assert np.allclose(onsets(film), expected, atol=0.002), (onsets(film), expected)
    assert sorted(p.name for p in (tmp_path / "best").iterdir()) == [".lock", "best.json", "best.mp4"]
    assert any(m.endswith("with sound") for m in said)


@needs_ffmpeg
def test_a_sound_that_cannot_be_captured_leaves_the_films_silent(tmp_path):
    def broken():
        raise RuntimeError("parec not found")

    said = []
    recorder = BestEpisodeRecorder(tmp_path, actor=0, say=said.append, sound=broken)
    for episode in range(2):
        recorder.start(episode)
        for k in range(10):
            recorder.add(np.zeros((72, 128, 4), np.uint8), 1000.0 + k / 15)
        recorder.finish(game(episode + 1))
    recorder.close()
    assert "sound" not in read_best(tmp_path / "best") and read_best(tmp_path / "best")["round"] == 2
    assert sum("films will be silent" in m for m in said) == 1  # said once, not every game


def test_the_game_over_frames_go_to_the_film_once():
    from types import SimpleNamespace

    from zombiesai.rl.actors import _end_frames

    env = SimpleNamespace(end_frames=[("a", 1.0), ("b", 1.1)])
    assert _end_frames(env) == [("a", 1.0), ("b", 1.1)]
    assert _end_frames(env) == [] and env.end_frames == []  # the next game starts with none
    assert _end_frames(SimpleNamespace()) == [] and _end_frames(SimpleNamespace(end_frames=None)) == []
