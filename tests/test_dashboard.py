import json
import re

import pytest

from zombiesai.viz.dashboard import (
    GATES,
    bucket,
    build_dashboard,
    classify,
    collect_runs,
    public_payload,
    read_run,
    trend,
    write_dashboard_html,
    write_dashboard_site,
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
        "rl1",
        ppo_rows(40, env_metrics=True),
        {"env": "real-waw", "total_steps": 20_000_000, "n_actors": 4, "batch_steps": 4096},
        checkpoint="checkpoint.pt",
    )
    run = read_run(tmp_path / "rl1")
    assert run["kind"] == "ppo" and run["env"] == "real-waw" and run["group"] == "real-waw"
    assert run["status"] == "running"  # just written, and nowhere near total_steps
    assert run["progress_text"] == "81,920 of 20,000,000 steps"
    assert run["series"]["return_mean"]["last"] == 40 and run["series"]["return_mean"]["best"] == 40
    assert run["stats"]["episodes"] == 120
    # 40 updates x 2048 steps at 4000 steps/s, with 99.6% of the run still to go.
    assert run["elapsed_s"] == pytest.approx(20.5, abs=0.1)
    assert run["eta_s"] > run["elapsed_s"] * 200


def test_a_half_written_last_line_is_skipped_not_fatal(tmp_path):
    run = write_run(tmp_path, "live", ppo_rows(12), {"env": "synthetic", "total_steps": 500_000})
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

    write_run(tmp_path, "cold", ppo_rows(5), {"env": "synthetic", "total_steps": 500_000})
    os.utime(tmp_path / "cold" / "metrics.jsonl", (1_000_000, 1_000_000))
    assert read_run(tmp_path / "cold")["status"] == "stopped"

    # 244 updates x 2048 steps is the whole budget.
    config = {"env": "synthetic", "total_steps": 244 * 2048, "batch_steps": 2048}
    write_run(tmp_path, "finished", ppo_rows(244), config)
    done = read_run(tmp_path / "finished")
    assert done["status"] == "done" and done["progress"] == 1.0


def test_gate_breach_is_reported_as_critical(tmp_path):
    rows = ppo_rows(20, env_metrics=True)
    for row in rows[-5:]:
        row["repair_share_mean"] = GATES["repair_share"]["limit"] + 0.02
    write_run(tmp_path, "farming", rows, {"env": "real-waw", "total_steps": 20_000_000})
    notes = read_run(tmp_path / "farming")["notes"]
    breach = [n for n in notes if n["level"] == "critical"]
    assert len(breach) == 1 and "repairs below 25%" in breach[0]["text"]

    clean = read_run(write_run(tmp_path, "clean", ppo_rows(20, env_metrics=True), {"env": "real-waw"}))
    assert not [n for n in clean["notes"] if n["level"] == "critical"]
    assert any("anti-hacking gates hold" in n["text"] for n in clean["notes"])


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
    config = {"env": "real-waw", "total_steps": 20_000_000}
    write_run(tmp_path, "rl1", ppo_rows(30, env_metrics=True), config)
    payload = build_dashboard(tmp_path)
    path = write_dashboard_html(payload, tmp_path / "out" / "dashboard.html")
    html = path.read_text()
    assert html.startswith("<!doctype html>") and "<title>Training Room</title>" in html
    assert "http-equiv=\"refresh\"" not in html
    # Self-contained: fonts inlined, and nothing fetched from anywhere.
    assert "fonts.googleapis.com" not in html and "data:font/woff2;base64," in html
    assert not re.search(r"(src|href)=\"(?!#)(https?:)?//", html)
    embedded = json.loads(re.search(r"(?:const|let) DATA = (\{.*?\});\n", html, re.S).group(1))
    assert embedded["runs"][0]["name"] == "rl1"
    assert embedded["gates"]["max_term_share"]["limit"] == 0.60


def test_watch_mode_asks_the_page_to_reload(tmp_path):
    write_run(tmp_path, "r", ppo_rows(5))
    html = write_dashboard_html(build_dashboard(tmp_path), tmp_path / "d.html", refresh=30).read_text()
    assert '<meta http-equiv="refresh" content="30">' in html


def test_embedded_data_cannot_close_the_script_tag(tmp_path):
    # Run names and config values are whatever is on disk. Inside a <script> the one sequence that
    # escapes is "</script", so it is escaped; everything else is inert JSON string content.
    hostile = "</script><script>alert(1)</script>"
    write_run(tmp_path, "<img src=x onerror=alert(1)>", ppo_rows(5), {"env": "real-waw", "note": hostile})
    html = write_dashboard_html(build_dashboard(tmp_path), tmp_path / "d.html").read_text()
    blob = re.search(r"(?:const|let) DATA = (\{.*?\});\n", html, re.S).group(1)
    assert "</script>" not in blob and "<\\/script>" in blob
    data = json.loads(blob)  # \/ is a legal JSON escape, so the value survives intact
    assert data["runs"][0]["config"]["note"] == hostile
    assert data["runs"][0]["name"] == "<img src=x onerror=alert(1)>"
    # And the page builds every label as text, so a name is never parsed as markup.
    assert re.search(r"\.innerHTML\s*=", html) is None


def test_public_payload_drops_anything_path_shaped(tmp_path):
    config = {
        "env": "real-waw",
        "device": "cuda",
        "obs_keys": ["pixels", "vector"],
        "hidden": [128, 128],
        "lr": 0.0003,
        "train_clips": ["/home/dom/Videos/waw/clip_004.mp4"],
        "out_dir": "runs/bc1",
    }
    write_run(tmp_path, "bc1", ppo_rows(5), config)
    public = public_payload(build_dashboard(tmp_path))
    kept = public["runs"][0]["config"]
    assert kept["env"] == "real-waw" and kept["hidden"] == [128, 128] and kept["obs_keys"] == ["pixels", "vector"]
    assert kept["device"] == "cuda" and kept["lr"] == 0.0003
    assert "train_clips" not in kept and "out_dir" not in kept  # the user's filesystem stays theirs
    assert public["root"] == "runs/"  # never the absolute path the build happened to run from


@pytest.mark.parametrize("listen", ["192.168.1.20:47860", "gamingpc.lan:47860"])
def test_public_payload_drops_the_address_a_fleet_learner_listened_on(tmp_path, listen):
    config = {"env": "real-waw", "device": "cuda:0", "n_actors": 4, "listen": listen, "lr": 0.0003,
              "note": "learner at 10.0.0.5 for now"}
    rows = ppo_rows(5)
    rows[-1]["peer"] = "[fe80::1]:47860"  # should a trainer ever log one, the last row is published too
    write_run(tmp_path / "runs", "rl5", rows, config)
    payload = build_dashboard(tmp_path / "runs")
    assert payload["runs"][0]["config"]["listen"] == listen  # the local page keeps it: it is the user's own
    public = public_payload(payload)
    kept, text = public["runs"][0]["config"], json.dumps(public)
    assert "listen" not in kept and "note" not in kept and "peer" not in public["runs"][0]["last_row"]
    assert kept["device"] == "cuda:0" and kept["n_actors"] == 4 and kept["env"] == "real-waw"
    host = listen.split(":")[0]
    assert host not in text and "47860" not in text and "10.0.0.5" not in text and "fe80" not in text
    index = write_dashboard_site(payload, tmp_path / "site", "intro", [], "d").read_text()
    assert host not in index and "47860" not in index and "10.0.0.5" not in index


def test_site_build_is_a_static_directory_with_no_external_request(tmp_path):
    write_run(tmp_path / "runs", "rl1", ppo_rows(30, env_metrics=True), {"env": "real-waw"})
    out = tmp_path / "site"
    index = write_dashboard_site(
        build_dashboard(tmp_path / "runs"),
        out,
        intro="How training is going.",
        links=[("← domalouf.com", "/"), ("Code", "https://github.com/domalouf/ZombiesAI")],
        description="Learning curves for an RL agent.",
    )
    html = index.read_text()
    assert index == out / "index.html"
    # Fonts ship as files, not data: URIs -- a default-src 'self' CSP refuses the latter.
    for face in ("big-shoulders-stencil-display", "red-hat-mono", "sofia-sans-condensed"):
        assert (out / "fonts" / f"{face}.woff2").exists() and f"url(fonts/{face}.woff2)" in html
    assert "data:font" not in html
    assert not re.search(r'(src|href)="(?!#)(https?:)?//', html)  # nothing fetched from anywhere
    assert '<meta name="description" content="Learning curves for an RL agent.">' in html
    assert 'http-equiv="refresh"' not in html  # a published page is a snapshot, not a poller
    data = json.loads(re.search(r"(?:const|let) DATA = (\{.*?\});\n", html, re.S).group(1))
    assert data["intro"] == "How training is going." and data["links"][0] == ["← domalouf.com", "/"]


def fleet_rows(n, per=4096):
    rows = []
    for u in range(1, n + 1):
        machines = {
            "0": {"segments": 6, "dropped_segments": 0, "steps": 1500, "bad_step_frac": 0.03, "round_reached_mean": 4.0},
            "1": {"segments": 4, "dropped_segments": 1 if u % 2 else 0, "steps": 500, "bad_step_frac": 0.2,
                  "round_reached_mean": None},
        }
        rows.append({"update": u, "step": u * per, "return_mean": 1.0, "machines": 2, "per_machine": machines})
    return rows


def test_a_run_on_several_pcs_charts_each_machine_on_its_own(tmp_path):
    run_dir = write_run(tmp_path, "rl5", fleet_rows(20), {"total_steps": 100_000, "env": "real-waw"})
    (run_dir / "fleet.json").write_text(json.dumps({"workers": [{"name": "rig2", "machine": 1}]}))
    machines = read_run(run_dir)["machines"]
    assert machines["ids"] == ["0", "1"] and machines["names"] == {"1": "rig2"}
    assert machines["colors"]["0"] != machines["colors"]["1"] and machines["late_warn_pct"] == 10.0
    series = machines["series"]
    assert series["bad_step_pct"]["by"]["1"]["y"][-1] == pytest.approx(20.0)
    assert series["share_pct"]["by"]["0"]["y"][-1] == pytest.approx(75.0)  # 1500 of the update's 2000 steps
    # pooled over the last 10 updates: 5 stale of 45 segments, not a 20% spike every other update
    assert series["stale_pct"]["by"]["1"]["y"][-1] == pytest.approx(100 * 5 / 45, abs=0.01)
    assert series["round_reached_mean"]["by"].keys() == {"0"}  # machine 1 has finished no game: no line, not 0


def test_a_run_on_one_pc_has_no_machine_charts(tmp_path):
    assert read_run(write_run(tmp_path, "solo", ppo_rows(5), {"total_steps": 100_000}))["machines"] is None


def test_the_site_shows_machine_numbers_never_their_names(tmp_path):
    run_dir = write_run(tmp_path / "runs", "rl5", fleet_rows(5), {"total_steps": 100_000})
    (run_dir / "fleet.json").write_text(json.dumps({"workers": [{"name": "DomPC-mk3", "machine": 1}]}))
    payload = build_dashboard(tmp_path / "runs")
    assert payload["runs"][0]["machines"]["names"] == {"1": "DomPC-mk3"}  # the local page may say who it is
    public = public_payload(payload)
    assert public["runs"][0]["machines"]["names"] == {} and "DomPC" not in json.dumps(public)
    index = write_dashboard_site(payload, tmp_path / "site", "intro", [], "d")
    assert "DomPC" not in index.read_text() and "machinesSection" in index.read_text()
