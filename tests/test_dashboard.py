import json
import re

import pytest

from zombiesai.viz.dashboard import (
    GATES,
    bucket,
    build_dashboard,
    classify,
    collect_runs,
    read_run,
    trend,
    write_dashboard_html,
)


def write_run(root, name, rows, config=None, checkpoint=None):
    run = root / name
    run.mkdir(parents=True)
    if config is not None:
        (run / "config.json").write_text(json.dumps(config))
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if checkpoint:
        (run / checkpoint).write_bytes(b"stand-in")
    return run


def ppo_rows(n, env_metrics=False, per=2048):
    rows = []
    for u in range(1, n + 1):
        row = {
            "update": u,
            "step": u * per,
            "sps": 4000,
            "lr": 3e-4,
            "episodes": u * 3,
            "clipfrac": 0.15,
            "policy_loss": -0.01,
            "value_loss": 30.0 - u * 0.1,
            "entropy": 2.5 - u * 0.01,
            "approx_kl": 0.01,
            "explained_variance": -0.2 + u * 0.02,
            "return_mean": float(u),
            "length_mean": 100.0 + u,
        }
        if env_metrics:
            row |= {"round_reached_mean": 1.0 + u * 0.05, "max_term_share_mean": 0.4, "repair_share_mean": 0.1}
        rows.append(row)
    return rows


def test_reads_a_ppo_run_and_names_its_progress(tmp_path):
    write_run(
        tmp_path,
        "ppo-nacht-state-s1",
        ppo_rows(40, env_metrics=True),
        {"env": "nacht-state", "total_steps": 20_000_000, "num_envs": 16, "rollout_steps": 128},
        checkpoint="checkpoint.pt",
    )
    run = read_run(tmp_path / "ppo-nacht-state-s1")
    assert run["kind"] == "ppo" and run["env"] == "nacht-state" and run["group"] == "nacht-state"
    assert run["status"] == "running"  # just written, and nowhere near total_steps
    assert run["progress_text"] == "update 40 of 9,765"
    assert run["series"]["return_mean"]["last"] == 40 and run["series"]["return_mean"]["best"] == 40
    assert run["stats"]["episodes"] == 120
    # 40 updates x 2048 steps at 4000 steps/s, with 99.6% of the run still to go.
    assert run["elapsed_s"] == pytest.approx(20.5, abs=0.1)
    assert run["eta_s"] > run["elapsed_s"] * 200


def test_a_half_written_last_line_is_skipped_not_fatal(tmp_path):
    run = write_run(tmp_path, "live", ppo_rows(12), {"env": "cartpole", "total_steps": 500_000})
    with (run / "metrics.jsonl").open("a") as f:
        f.write('{"update": 13, "step": 266')  # what a reader sees mid-write
    parsed = read_run(run)
    assert parsed["rows"] == 12 and parsed["series"]["return_mean"]["last"] == 12


def test_a_run_with_no_config_still_reads(tmp_path):
    write_run(tmp_path, "nameless", ppo_rows(9))
    run = read_run(tmp_path / "nameless")
    assert run["kind"] == "ppo" and run["progress"] is None
    assert run["progress_text"] == "18,432 steps"


def test_directories_without_metrics_are_not_runs(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "readme.txt").write_text("not a run")
    assert read_run(tmp_path / "notes") is None
    assert collect_runs(tmp_path) == []
    assert build_dashboard(tmp_path / "does-not-exist")["runs"] == []


def test_kinds_are_told_apart(tmp_path):
    bc_row = {"epoch": 1, "loss": {"policy": 2.0}, "val": {"accuracy": {"mean_balanced": 0.4}}}
    bc = write_run(tmp_path, "bc1", [bc_row])
    idm = write_run(tmp_path, "idm1", [{"epoch": 1, "loss": 2.0, "val": {"mean_balanced": 0.5}}])
    assert classify(bc, {}, [{"loss": {"policy": 2.0}}]) == "bc"
    assert classify(idm, {}, [{"epoch": 1, "loss": 2.0}]) == "idm"
    assert read_run(bc)["series"]["val.accuracy.mean_balanced"]["last"] == 0.4
    assert read_run(idm)["series"]["val.mean_balanced"]["last"] == 0.5
    # BC and the IDM are their own groups, so their charts never share an axis with an RL run's.
    assert {r["group"] for r in collect_runs(tmp_path)} == {"bc", "idm"}


def test_runs_keep_their_colour_and_are_newest_first(tmp_path):
    for name in ("a", "b", "c"):
        write_run(tmp_path, name, ppo_rows(5))
    import os
    import time

    now = time.time()
    for i, name in enumerate(("a", "b", "c")):
        os.utime(tmp_path / name / "metrics.jsonl", (now - i * 100, now - i * 100))
    runs = collect_runs(tmp_path)
    assert [r["name"] for r in runs] == ["a", "b", "c"]
    assert len({r["color"] for r in runs}) == 3


def test_stale_runs_read_as_stopped_and_finished_ones_as_done(tmp_path):
    import os

    write_run(tmp_path, "cold", ppo_rows(5), {"env": "cartpole", "total_steps": 500_000})
    os.utime(tmp_path / "cold" / "metrics.jsonl", (1_000_000, 1_000_000))
    assert read_run(tmp_path / "cold")["status"] == "stopped"

    # 244 updates x 2048 steps is the whole 500k budget.
    config = {"env": "cartpole", "total_steps": 500_000, "num_envs": 4, "rollout_steps": 512}
    write_run(tmp_path, "finished", ppo_rows(244), config)
    done = read_run(tmp_path / "finished")
    assert done["status"] == "done" and done["progress"] == 1.0


def test_gate_breach_is_reported_as_critical(tmp_path):
    rows = ppo_rows(20, env_metrics=True)
    for row in rows[-5:]:
        row["repair_share_mean"] = GATES["repair_share"]["limit"] + 0.02
    write_run(tmp_path, "farming", rows, {"env": "nacht-state", "total_steps": 20_000_000})
    notes = read_run(tmp_path / "farming")["notes"]
    breach = [n for n in notes if n["level"] == "critical"]
    assert len(breach) == 1 and "repairs below 25%" in breach[0]["text"]

    clean = read_run(write_run(tmp_path, "clean", ppo_rows(20, env_metrics=True), {"env": "nacht-state"}))
    assert not [n for n in clean["notes"] if n["level"] == "critical"]
    assert any("anti-hacking gates hold" in n["text"] for n in clean["notes"])


def test_solved_at_is_checked_against_the_preset(tmp_path):
    rows = ppo_rows(30)
    for row in rows:
        row["return_mean"] = 490.0
    write_run(tmp_path, "cp", rows, {"env": "cartpole", "total_steps": 500_000})
    run = read_run(tmp_path / "cp")
    assert run["stats"]["solved_at"] == 475.0 and run["stats"]["solved"] is True
    assert any("counts as solved" in n["text"] for n in run["notes"])


def test_trend_calls_noise_flat_and_a_climb_climbing():
    xs = list(range(60))
    assert trend(xs, [1.0] * 60)["verdict"] == "flat"
    assert trend(xs, [float(x) for x in xs])["verdict"] == "climbing"
    assert trend(xs, [float(-x) for x in xs])["verdict"] == "falling"
    # A slope smaller than the scatter it was fitted through is not news.
    noisy = [(1.0 if x % 2 else -1.0) + x * 0.001 for x in xs]
    assert trend(xs, noisy)["verdict"] == "flat"
    assert trend([1, 2], [1.0, 2.0]) is None


def test_bucketing_keeps_the_ends_and_the_spread():
    xs = list(range(100))
    ys = [float(x % 10) for x in xs]
    b = bucket(xs, ys, 10)
    assert len(b["x"]) == 10 and b["x"][-1] == 99
    assert all(b["lo"][i] <= b["y"][i] <= b["hi"][i] for i in range(10))
    assert b["lo"][0] == 0 and b["hi"][0] == 9  # the spread inside a bucket survives the averaging
    assert bucket([], [], 10) == {"x": [], "y": [], "lo": [], "hi": []}
    assert len(bucket(xs, ys, 500)["x"]) == 100  # never invents points it does not have


def test_page_is_standalone_and_embeds_the_data(tmp_path):
    config = {"env": "nacht-state", "total_steps": 20_000_000}
    write_run(tmp_path, "ppo-nacht-state-s1", ppo_rows(30, env_metrics=True), config)
    payload = build_dashboard(tmp_path)
    path = write_dashboard_html(payload, tmp_path / "out" / "dashboard.html")
    html = path.read_text()
    assert html.startswith("<!doctype html>") and "<title>Training Room</title>" in html
    assert "http-equiv=\"refresh\"" not in html
    # Self-contained: fonts inlined, and nothing fetched from anywhere.
    assert "fonts.googleapis.com" not in html and "data:font/woff2;base64," in html
    assert not re.search(r"(src|href)=\"(?!#)(https?:)?//", html)
    embedded = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    assert embedded["runs"][0]["name"] == "ppo-nacht-state-s1"
    assert embedded["gates"]["max_term_share"]["limit"] == 0.60


def test_watch_mode_asks_the_page_to_reload(tmp_path):
    write_run(tmp_path, "r", ppo_rows(5))
    html = write_dashboard_html(build_dashboard(tmp_path), tmp_path / "d.html", refresh=30).read_text()
    assert '<meta http-equiv="refresh" content="30">' in html


def test_embedded_data_cannot_close_the_script_tag(tmp_path):
    # Run names and config values are whatever is on disk. Inside a <script> the one sequence that
    # escapes is "</script", so it is escaped; everything else is inert JSON string content.
    hostile = "</script><script>alert(1)</script>"
    write_run(tmp_path, "<img src=x onerror=alert(1)>", ppo_rows(5), {"env": "nacht-state", "note": hostile})
    html = write_dashboard_html(build_dashboard(tmp_path), tmp_path / "d.html").read_text()
    blob = re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1)
    assert "</script>" not in blob and "<\\/script>" in blob
    data = json.loads(blob)  # \/ is a legal JSON escape, so the value survives intact
    assert data["runs"][0]["config"]["note"] == hostile
    assert data["runs"][0]["name"] == "<img src=x onerror=alert(1)>"
    # And the page builds every label as text, so a name is never parsed as markup.
    assert re.search(r"\.innerHTML\s*=", html) is None
