import json

import pytest

from zombiesai.viz.stream import (
    EpisodeTail, StreamFeed, accuracy, demo_games, pick_run, recent_games, rolling, round_counts, stream_payload,
    summarize, write_stream_site,
)


def game(rnd, seconds=100.0, shots=10, hits=2, **kw):
    return {"round_reached": rnd, "seconds": seconds, "shots": shots, "hits": hits, **kw}


def test_the_overlay_numbers_are_the_runs_best_and_the_recent_games_means():
    games = [game(9, 900.0)] + [game(r, 60.0 * r) for r in (1, 2, 3, 2)]
    s = summarize(games, window=4)
    assert s["best_round"] == 9 and s["games"] == 5 and s["window"] == 4  # best over the run, not the window
    assert s["avg_round"] == 2.0 and s["avg_survival_s"] == 120.0 and s["accuracy"] == 0.2
    assert s["avg_points"] is None  # no game said what it earned
    assert summarize([]) == {"games": 0, "window": 0, "best_round": None, "avg_round": None, "accuracy": None,
                             "avg_survival_s": None, "avg_points": None, "avg_kills": None}


def test_average_points_are_what_the_recent_games_earned():
    games = [game(1, points_gained=p) for p in (5000, 0, 10, 50, 150)]
    assert summarize(games, window=4)["avg_points"] == 52.5


def test_average_kills_skip_games_from_before_kills_were_counted():
    games = [game(1), game(1, kills=3), game(1, kills=0), game(1, kills=2)]
    assert summarize(games)["avg_kills"] == 1.67
    assert summarize([game(1)])["avg_kills"] is None


def test_the_scoreboards_kills_are_never_averaged_with_the_hud_estimate():
    games = [game(1, kills=0, end_kills=4), game(1, kills=2, end_kills=None), game(1, end_kills=3)]
    assert summarize(games)["avg_kills"] == 3.5  # the game without a scoreboard read is left out, not estimated
    assert summarize([game(1, kills=2), game(1, kills=1, end_kills=None)])["avg_kills"] == 1.5  # an older run


def test_accuracy_weighs_games_by_their_shots_and_ignores_games_without_counts():
    assert accuracy([game(1, shots=2, hits=2), game(1, shots=98, hits=0)]) == pytest.approx(0.02)
    assert accuracy([{"round_reached": 1}]) is None  # a run from before shots were counted


def test_rolling_curves_end_at_the_last_game_and_stay_small():
    games = [game(1 + i // 100) for i in range(1000)]
    c = rolling(games, window=100, points=50)
    assert len(c["x"]) == 50 and c["x"][-1] == 1000 and c["round"][-1] == 10.0 and c["accuracy"][-1] == 0.2
    assert rolling([]) == {"x": [], "round": [], "survival": [], "accuracy": []}


def test_round_counts_keep_empty_rounds_and_recent_games_are_newest_first():
    games = [game(1), game(3), game(3, reason="death", shots=0)]
    assert round_counts(games) == [[1, 1], [2, 0], [3, 2]]
    recent = recent_games(games, n=2)
    assert [g["n"] for g in recent] == [3, 2] and recent[0]["accuracy"] is None and recent[1]["accuracy"] == 0.2


def test_the_tail_reads_only_new_whole_lines_and_starts_over_on_a_new_file(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(game(1)) + "\n" + '{"round_reached": 2, "sec')
    tail = EpisodeTail()
    assert [g["round_reached"] for g in tail.read(path)] == [1]  # the half-written line waits
    with open(path, "a") as f:
        f.write('onds": 5}\n' + json.dumps(game(3)) + "\n")
    assert [g["round_reached"] for g in tail.read(path)] == [1, 2, 3]
    path.write_text(json.dumps(game(7)) + "\n")  # truncated and rewritten: a new run under the same name
    assert [g["round_reached"] for g in tail.read(path)] == [7]
    assert tail.read(None) == []


def test_the_stream_follows_the_real_game_being_trained_before_a_rehearsal(tmp_path):
    paths = {}
    for name in ("rl-rehearsal", "rl-real", "rl-old", "bc1"):
        (tmp_path / name).mkdir()
        if name != "bc1":
            (tmp_path / name / "episodes.jsonl").write_text(json.dumps(game(1)) + "\n")
        paths[str(tmp_path / name)] = name
    trainers = [{"run": "rl-rehearsal", "rehearsal": True}, {"run": "rl-real", "rehearsal": False},
                {"run": "bc1", "rehearsal": False}]
    assert pick_run(paths, trainers)[0] == "rl-real"
    assert pick_run(paths, [])[0] in ("rl-rehearsal", "rl-real", "rl-old")  # nothing training: the last games written
    assert pick_run(paths, trainers, prefer="rl-old")[0] == "rl-old"
    assert pick_run(paths, trainers, prefer="bc1") is None  # it plays no games


def test_the_payload_carries_the_runs_progress_and_health(tmp_path):
    (tmp_path / "rl1").mkdir()
    (tmp_path / "rl1" / "episodes.jsonl").write_text("".join(json.dumps(game(r)) + "\n" for r in (1, 2, 4)))
    run = {"name": "rl1", "env": "real-waw", "status": "running", "progress": 0.25, "progress_text": "step 1 of 4",
           "elapsed_s": 60.0, "eta_s": 180.0,
           "last_row": {"step": 1000, "update": 4, "episodes": 3, "sps": 40, "entropy": 3.2, "kl_ref": 0.01,
                        "approx_kl": float("nan")},
           "series": {"return_mean": {"x": [1, 2], "y": [0.5, 1.0], "lo": [0, 0], "hi": [1, 1], "label": "Return",
                                      "hint": "", "last": 1.0, "best": 1.0, "trend": {"verdict": "improving"}}}}
    feed = StreamFeed()
    feed.choose({str(tmp_path / "rl1"): "rl1"}, [{"run": "rl1"}])
    out = feed.payload({"runs": [run]}, [{"run": "rl1"}], now=5.0)
    json.dumps(out, allow_nan=False)
    assert out["run"] == "rl1" and out["live"] and out["stats"]["best_round"] == 4 and out["rounds"][-1] == [4, 1]
    ppo = out["ppo"]
    assert ppo["steps"] == 1000 and ppo["updates"] == 4 and ppo["eta_s"] == 180.0
    assert [h["key"] for h in ppo["health"]] == ["entropy", "kl_ref"]  # a NaN is left out, not published
    assert ppo["series"]["return_mean"]["trend"] == "improving" and "lo" not in ppo["series"]["return_mean"]
    nothing = StreamFeed().payload(None, [], now=5.0)
    assert nothing["run"] is None and nothing["ppo"] is None and nothing["stats"]["games"] == 0


def test_the_site_is_two_pages_that_poll_the_live_file_and_embed_the_channel(tmp_path):
    data = stream_payload("demo", demo_games(50), None, live=False, now=1.0)
    index, overlay = write_stream_site(tmp_path / "live", channel="@zombies_ai", snapshot=data,
                                       links=[("Training Room", "/zombies/training/")], description="d")
    html = index.read_text()
    assert 'data-channel="zombies_ai"' in html and '<a href="/zombies/training/">Training Room</a>' in html
    assert '"run":"demo"' in html and "../training/live/stream.json" in html
    assert "fonts.googleapis" not in html and "url(fonts/" in html and (tmp_path / "live" / "fonts").is_dir()
    o = overlay.read_text()
    assert "../../training/live/stream.json" in o and "background: transparent" in o
    assert (overlay.parent / "fonts" / "red-hat-mono.woff2").exists()
    write_stream_site(tmp_path / "none", channel=None)
    assert 'data-channel=""' in (tmp_path / "none" / "index.html").read_text()
    with pytest.raises(ValueError):
        write_stream_site(tmp_path / "bad", channel='x" onload="alert(1)')
