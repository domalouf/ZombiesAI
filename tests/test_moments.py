import json
import shutil
import subprocess
import time

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.rl.best_episode import BestEpisodeRecorder
from zombiesai.rl.keepsakes import Keepsakes
from zombiesai.rl.moments import MomentBook, MomentSpotter, kind

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def act(fire=False, button="none"):
    a = np.zeros(len(spec.ACTION_NVEC), np.int64)
    a[spec.FIRE] = int(fire)
    a[spec.BUTTON] = spec.BUTTONS.index(button)
    return a


def info(event="", delta=0, *, repair=False, weapon="colt", round_=1, points=500):
    return {"points_event": event, "points_delta": delta, "repair": repair, "weapon": weapon, "round": round_,
            "points": points}


def kinds(moments):
    return [m["kind"] for m in moments]


def test_each_kind_is_spotted_once_a_game():
    s = MomentSpotter()
    assert kinds(s.step(info(), act())) == []
    assert kinds(s.step(info("gain", 10, repair=True), act(button="use"))) == ["repair"]
    assert kinds(s.step(info("gain", 60), act(fire=True))) == ["kill"]
    assert kinds(s.step(info("gain", 60), act(fire=True))) == []  # a second kill is not a first
    assert kinds(s.step(info("gain", 100), act(fire=True))) == ["headshot"]
    assert kinds(s.step(info("gain", 130), act(button="melee"))) == ["knife_kill"]
    assert kinds(s.step(info("spend", -1000), act(button="use"))) == ["door"]
    assert kinds(s.step(info("spend", -950), act(button="use"))) == ["box"]
    assert kinds(s.step(info(weapon="ray_gun"), act())) == ["weapon_ray_gun"]
    assert kinds(s.step(info("spend", -1000, weapon="ray_gun"), act())) == ["doors_2"]
    assert kinds(s.step(info("spend", -600, weapon="ray_gun"), act())) == ["wall_buy"]
    assert kinds(s.step(info(weapon="m1_carbine", round_=2), act())) == ["weapon_m1_carbine", "round_2"]
    s.reset()
    assert kinds(s.step(info("gain", 60), act(fire=True))) == ["kill"]


def test_a_headshot_needs_the_trigger_and_a_knife_kill_the_melee():
    s = MomentSpotter()
    assert kinds(s.step(info("gain", 100), act(button="melee"))) == ["kill"]
    s.reset()
    assert kinds(s.step(info("gain", 130), act(fire=True))) == ["kill"]


def test_the_longest_stretch_without_a_kill_counts_the_one_still_going():
    s = MomentSpotter()
    for _ in range(30):
        s.step(info(), act())
    s.step(info("gain", 60), act(fire=True))  # a kill on step 31
    for _ in range(15):
        s.step(info(), act())
    assert s.longest_without_kill_s() == pytest.approx(31 / spec.DECISION_HZ)
    for _ in range(60):
        s.step(info(), act())
    assert s.longest_without_kill_s() == pytest.approx(75 / spec.DECISION_HZ)


def test_round_milestones_skip_rounds_not_on_the_list():
    s = MomentSpotter()
    seen = [k for r in range(1, 14) for k in kinds(s.step(info(round_=r), act()))]
    assert seen == [f"round_{r}" for r in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)]
    assert kind("round_10").title == "First time reaching round 10"
    assert kind("weapon_ray_gun").title == "First time holding the Ray Gun"


def moment(name, t):
    return {"kind": name, "title": kind(name).title, "t_unix": t}


def test_the_book_keeps_the_earliest_first_of_each_kind(tmp_path):
    book = MomentBook(tmp_path)
    for name, data in (("a", b"a"), ("b", b"b"), ("c", b"c")):
        (tmp_path / f"{name}.tmp").write_bytes(data)
    assert book.offer_clip(moment("door", 20.0), {".mp4": tmp_path / "a.tmp"})
    assert not book.offer_clip(moment("door", 30.0), {".mp4": tmp_path / "b.tmp"})  # later: thrown away
    assert not (tmp_path / "b.tmp").exists()
    assert book.offer_clip(moment("door", 10.0), {".mp4": tmp_path / "c.tmp"})  # another actor saw it first
    assert (tmp_path / "door.mp4").read_bytes() == b"c" and book.read()["door"]["t_unix"] == 10.0
    assert book.wanted([moment("door", 15.0), moment("box", 15.0)], {}) == [moment("box", 15.0)]
    # a guessed headshot only counts in a game whose scoreboard has one
    assert book.wanted([moment("headshot", 1.0)], {"end_headshots": 0}) == []
    assert book.wanted([moment("headshot", 1.0)], {"end_headshots": None}) == []
    assert book.wanted([moment("headshot", 1.0)], {"end_headshots": 2}) == [moment("headshot", 1.0)]


def seconds_in(path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


@needs_ffmpeg
def test_a_finished_games_firsts_are_cut_from_its_film(tmp_path):
    said = []
    keepsakes = Keepsakes(tmp_path / "video", progress_every_h=0, records=False)
    book = keepsakes.firsts
    recorder = BestEpisodeRecorder(tmp_path / "rl1", actor=0, say=said.append, keep_best=False, keepsakes=keepsakes)
    recorder.start(0)
    for step in range(300):  # 20 s at 15 Hz
        t = 100.0 + step / spec.DECISION_HZ
        recorder.add(np.full((72, 128, 4), step % 256, np.uint8), t=t)
        recorder.note({"v": step / 100}, t)
        time.sleep(0.002)  # a game's pace, not faster than the encoder's queue takes it
        if step == 150:
            recorder.mark(moment("door", 5.0), t=t, policy_version=7)
    recorder.finish({"round_reached": 1, "reason": "down"})
    recorder.start(1)
    recorder.add(np.zeros((72, 128, 4), np.uint8))
    recorder.mark(moment("box", 6.0))  # a game cut short keeps no moments
    recorder.close()

    kept = book.read()
    assert list(kept) == ["door"], said
    door = kept["door"]
    assert (door["run"], door["actor"], door["episode"], door["policy_version"]) == ("rl1", 0, 0, 7)
    assert door["at_s"] == pytest.approx(10.0) and door["clip_start_s"] == pytest.approx(2.0)
    window = kind("door")
    assert seconds_in(tmp_path / "video" / "firsts" / "door.mp4") == pytest.approx(
        window.before_s + window.after_s, abs=0.15)
    # the clip's sidecar is the clip's own frames, timed from the clip's start
    lines = (tmp_path / "video" / "firsts" / door["brain"]).read_text().splitlines()
    assert json.loads(lines[0])["header"]["moment"] == "door"
    first = json.loads(lines[1])
    assert first["s"] == pytest.approx(0.0, abs=0.07) and first["v"] == pytest.approx(0.3, abs=0.011)
    # with keep_best off, nothing is kept as the best and no film is left behind
    assert not list((tmp_path / "rl1" / "best").glob("*.mp4"))
