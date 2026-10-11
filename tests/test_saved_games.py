import json

import pytest

from zombiesai.viz import live_site
from zombiesai.viz.saved_games import read_index, save_game, saved_games

BEST = {"round": 1, "kills": 2, "points": 600, "host": "DomPC-mk3", "recorded_unix": 1791171315.7,
        "stats": {"seconds": 80.9, "from": "scoreboard"}, "summary": {"seconds": 80.9}}


def a_run(runs, name="rl10", film=b"film"):
    best = runs / name / "best"
    best.mkdir(parents=True)
    (best / "best.json").write_text(json.dumps(BEST))
    (best / "best.mp4").write_bytes(film)
    (best / "best.brain.jsonl").write_text("{}\n")
    return runs / name


def test_a_saved_game_outlives_the_runs_next_best(tmp_path):
    run = a_run(tmp_path / "runs")
    saved = tmp_path / "runs" / "saved"
    entry = save_game(run, saved, "rl10-grenade", " Took two with him ", "Its grenade kills two zombies.", now=5.0)
    assert entry == {"name": "rl10-grenade", "title": "Took two with him", "caption": "Its grenade kills two zombies.",
                     "run": "rl10", "round": 1, "points": 600, "kills": 2, "seconds": 80.9,
                     "recorded": 1791171315.7, "saved": 5.0}
    assert read_index(saved) == {"rl10-grenade": entry}
    assert json.loads((saved / "rl10-grenade.json").read_text())["host"] == "DomPC-mk3"  # the record, kept here
    # the run does better: its best.mp4 is replaced, as rl/best_episode.py replaces it, and the saved one stays
    (run / "best" / "next.mp4").write_bytes(b"better")
    (run / "best" / "next.mp4").replace(run / "best" / "best.mp4")
    assert (saved / "rl10-grenade.mp4").read_bytes() == b"film"
    assert (saved / "rl10-grenade.brain.jsonl").exists()


def test_a_name_is_saved_once_and_must_fit_a_file_name(tmp_path):
    run = a_run(tmp_path / "runs")
    saved = tmp_path / "saved"
    save_game(run, saved, "first", "First")
    with pytest.raises(FileExistsError):
        save_game(run, saved, "first", "Again")
    for bad in ("../x", "Grenade", "a.mp4", ""):
        with pytest.raises(ValueError):
            save_game(run, saved, bad, "Title")
    with pytest.raises(ValueError):
        save_game(run, saved, "untitled", "  ")
    assert list(read_index(saved)) == ["first"]


def test_the_live_page_ships_every_checkouts_saved_films_newest_first(tmp_path, monkeypatch):
    main, tree = tmp_path / "main" / "runs", tmp_path / "wt" / "runs"
    save_game(a_run(main), main / "saved", "older", "Older", now=1.0)
    save_game(a_run(tree, "rl4", b"newer"), tree / "saved", "newer", "Newer", now=2.0)
    (main / "saved" / "gone.json").write_text("{}")
    index = read_index(main / "saved")
    index["gone"] = dict(index["older"], name="gone")  # its film is not there: not shown
    (main / "saved" / "saved.json").write_text(json.dumps(index))
    assert [g["name"] for g, _ in saved_games([main, tree])] == ["newer", "older"]

    monkeypatch.setattr(live_site, "build_payload", lambda roots: {"runs": [], "run_paths": {}})
    monkeypatch.setattr(live_site, "run_roots", lambda repo: [("main", main), ("wt", tree)])
    monkeypatch.setattr(live_site, "live_trainers", lambda paths: [])
    site = tmp_path / "site"
    html = live_site.write_live_page(tmp_path, site, "d").read_text()
    assert (site / "saved" / "newer.mp4").read_bytes() == b"newer"
    assert (site / "saved" / "older.mp4").read_bytes() == b"film"
    assert not (site / "saved" / "gone.mp4").exists()
    data = json.loads(html.split("let DATA = ", 1)[1].split(";\n", 1)[0])
    assert [g["name"] for g in data["saved"]] == ["newer", "older"]
    assert data["saved"][0]["video"].startswith("saved/newer.mp4?v=")
    assert "DomPC-mk3" not in html and str(tmp_path) not in html
