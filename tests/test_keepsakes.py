import json
import shutil
import time

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.rl.best_episode import BestEpisodeRecorder
from zombiesai.rl.clock import RunClock, TrainingClock
from zombiesai.rl.keepsakes import Keepsakes, Progress, Records, record_values

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def hours(h):
    return {"train_s": h * 3600.0}


def test_a_progress_slot_is_the_first_game_started_in_its_hour(tmp_path):
    shelf = Progress(tmp_path, every_h=2.0)
    assert shelf.key(hours(0.1)) == shelf.key(hours(1.9)) == "h0000.00"
    assert shelf.key(hours(5.0)) == "h0004.00"
    assert shelf.key({}) is None
    (tmp_path / "late.mp4").write_bytes(b"late")
    (tmp_path / "early.mp4").write_bytes(b"early")
    assert shelf.offer("h0000.00", {"start_clock": hours(1.5)}, {".mp4": tmp_path / "late.mp4"})
    assert (tmp_path / "late.mp4").exists()  # left for the caller: the same film may go on other shelves
    assert not shelf.wants("h0000.00", {"start_clock": hours(1.7)})
    assert shelf.offer("h0000.00", {"start_clock": hours(0.2)}, {".mp4": tmp_path / "early.mp4"})
    assert (tmp_path / "h0000.00.mp4").read_bytes() == b"early"
    assert shelf.read()["h0000.00"]["clip"] == "h0000.00.mp4"


def test_records_hold_the_extremes_of_finished_games(tmp_path):
    game = {"seconds": 41.0, "reason": "down", "events": {"repair": 3}, "shots": 40, "hits": 6,
            "longest_without_kill_s": 30.5}
    assert record_values(game) == {"shortest_game": 41.0, "most_repairs": 3.0, "worst_aim": 0.15,
                                   "longest_without_kill": 30.5}
    # a game cut short is not short play, and a handful of shots says nothing about aim
    assert record_values({"seconds": 5.0, "reason": "picture_lost", "shots": 4, "hits": 0}) == {}
    shelf = Records(tmp_path)
    (tmp_path / "a.mp4").write_bytes(b"a")
    for name, record in shelf.candidates(game):
        assert shelf.offer(name, record, {".mp4": tmp_path / "a.mp4"})
    better = {**game, "seconds": 30.0, "events": {"repair": 2}, "hits": 2, "longest_without_kill_s": 10.0}
    assert [name for name, _ in shelf.candidates(better)] == ["shortest_game", "worst_aim"]


def test_the_run_clock_carries_on_from_where_the_last_run_stopped():
    clock = RunClock(TrainingClock(train_s=3600.0, steps=1000, games=10, played_s=7200.0, updates=5), now=100.0)
    clock.game(60.0)
    clock.game(None)
    now = clock.at(now=1900.0, steps=500, updates=9)
    assert (now.train_s, now.steps, now.games, now.played_s, now.updates) == (5400.0, 1500, 12, 7260.0, 9)
    assert now.label() == "Hour 1 · game 12"
    assert TrainingClock.from_dict(now.to_dict()) == now


@needs_ffmpeg
def test_one_game_can_be_a_check_in_and_a_record_and_takes_its_sidecar_along(tmp_path):
    said = []
    keepsakes = Keepsakes(tmp_path / "video", firsts=False, progress_every_h=1.0)
    recorder = BestEpisodeRecorder(tmp_path / "rl1", actor=1, say=said.append, keepsakes=keepsakes)
    recorder.start(0)
    recorder.started_at(hours(2.5))
    for step in range(30):
        t = 50.0 + step / spec.DECISION_HZ
        recorder.add(np.full((72, 128, 4), 8 * step, np.uint8), t=t)
        recorder.note({"v": 1.0, "a": [0] * 8}, t)
        time.sleep(0.002)
    recorder.finish({"round_reached": 1, "reason": "down", "seconds": 2.0, "events": {"repair": 1}})
    recorder.close()
    video = tmp_path / "video"
    progress = json.loads((video / "progress" / "progress.json").read_text())
    assert list(progress) == ["h0002.00"] and progress["h0002.00"]["start_clock"] == hours(2.5), said
    assert (video / "progress" / "h0002.00.brain.jsonl").exists()
    records = json.loads((video / "records" / "records.json").read_text())
    assert sorted(records) == ["most_repairs", "shortest_game"]
    assert (tmp_path / "rl1" / "best" / "best.mp4").exists() and (tmp_path / "rl1" / "best" / "best.brain.jsonl").exists()
    assert not list((tmp_path / "rl1" / "best").glob(".a1_*"))  # no temporary film or sidecar left behind


def test_another_pcs_films_merge_in_by_each_shelfs_rule(tmp_path):
    from zombiesai.rl.keepsakes import merge

    def keep(root, shelf_name, key, record, data):
        shelf = Keepsakes(root).shelves()[shelf_name]
        film = tmp_path / f"{data}.mp4"
        film.write_bytes(data.encode())
        assert shelf.offer(key, record, {".mp4": film})

    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    keep(ours, "firsts", "door", {"kind": "door", "t_unix": 50.0}, "our-door")
    keep(theirs, "firsts", "door", {"kind": "door", "t_unix": 20.0, "host": "gamer2"}, "their-door")
    keep(ours, "records", "most_repairs", {"record": "most_repairs", "value": 9.0}, "our-repairs")
    keep(theirs, "records", "most_repairs", {"record": "most_repairs", "value": 4.0}, "their-repairs")
    keep(theirs, "progress", "h0003.00", {"start_clock": {"train_s": 11000.0}}, "their-h3")
    assert merge(ours, theirs) == {"firsts": 1, "progress": 1, "records": 0}
    assert (ours / "firsts" / "door.mp4").read_bytes() == b"their-door"
    assert Keepsakes(ours).firsts.read()["door"]["host"] == "gamer2"
    assert (ours / "records" / "most_repairs.mp4").read_bytes() == b"our-repairs"
    assert (ours / "progress" / "h0003.00.mp4").exists()
    assert merge(ours, theirs) == {"firsts": 0, "progress": 0, "records": 0}  # a second gather takes nothing new
