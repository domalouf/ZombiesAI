import json

from zombiesai.rl.evaluate import find_snapshots, human_baseline, pick, score, snapshot_hours, write_results


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_snapshots_are_found_across_runs_in_training_order_and_picked_per_bucket(tmp_path):
    for run, hours in (("rl1", (0.0, 1.0, 2.01)), ("rl2", (3.0, 4.5, 7.2))):
        for h in hours:
            touch(tmp_path / run / "snapshots" / f"h{h:07.2f}.pt")
    touch(tmp_path / "rl2" / "checkpoint.pt")
    found = find_snapshots([tmp_path])
    assert [snapshot_hours(p) for p in found] == [0.0, 1.0, 2.01, 3.0, 4.5, 7.2]
    assert [snapshot_hours(p) for p in pick(found, 3.0)] == [0.0, 3.0, 7.2]  # the latest always comes too


def test_a_score_is_the_rounds_the_kills_and_the_time_survived():
    games = [{"round_reached": 2, "end_kills": 9, "kills": 3, "seconds": 100.0},
             {"round_reached": 5, "end_kills": None, "kills": 20, "seconds": 300.0},
             {"round_reached": 3, "seconds": 200.0}]
    assert score(games) == {"games": 3, "round_mean": 3.33, "round_median": 3.0, "round_best": 5,
                            "kills_mean": 14.5, "seconds_mean": 200.0}
    assert score([]) == {"games": 0}


def test_the_human_row_counts_only_games_seen_whole(tmp_path):
    games = [{"started_in_clip": True, "game_over": True, "rounds_reached": 8, "seconds": 900.0},
             {"started_in_clip": False, "game_over": True, "rounds_reached": 12, "seconds": 300.0},
             {"started_in_clip": True, "game_over": False, "rounds_reached": 4, "seconds": 200.0}]
    touch(tmp_path / "demo_0001" / "hud_summary.json").write_text(json.dumps({"games": games}))
    human = human_baseline([tmp_path])
    assert human["games"] == 1 and human["round_mean"] == 8.0
    write_results(tmp_path / "evals", [{"snapshot": "s", "hours": 2.0, **score([{"round_reached": 3}])}], human)
    rows = (tmp_path / "evals" / "evals.csv").read_text().splitlines()
    assert rows[0].startswith("hours,games,round_mean") and rows[1].startswith("2.0,1,3.0")
    assert rows[2].startswith("human,1,8.0")
