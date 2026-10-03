import json
import socket
import threading
import time

import numpy as np
import pytest
import torch

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.rl import fleet as fleet_mod
from zombiesai.rl.fleet import (ACTOR_BLOCK, FleetClient, FleetError, FleetServer, FleetWorker, NameInUse,
                                WorkerOptions, decode_segment, encode_segment, provenance, settings_differences)
from zombiesai.rl.parallel_ppo import RLConfig, Segment, episode_line, publish, train

TOKEN = "test-token"


@pytest.fixture
def bc_checkpoint(tmp_path):
    config = bc.BCConfig(frame_offsets=(0, 1, 3), hidden=32)
    torch.manual_seed(0)
    path = tmp_path / "bc.pt"
    bc.save(path, bc.build_net(config), config, 0, {}, {"fire_duty": 0.1})
    return path


def segment(steps: int = 6, context: int = 3, audio: bool = False, seed: int = 0) -> Segment:
    rng = np.random.default_rng(seed)
    return Segment(
        actor=2, version=4, context=context,
        frames=rng.integers(0, 255, (context + steps + 1, *spec.PIXELS_SHAPE), dtype=np.uint8),
        actions=np.stack([rng.integers(0, m, steps) for m in spec.ACTION_NVEC], axis=1).astype(np.int64),
        logp=rng.normal(size=steps).astype(np.float32), rewards=rng.normal(size=steps).astype(np.float32),
        bad=rng.random(steps) < 0.3, terminated=True,
        audio=rng.normal(size=(steps + 1, 5)).astype(np.float32) if audio else None,
        audio_mask=np.ones(steps + 1, np.float32) if audio else None,
    )


@pytest.mark.parametrize("audio", [False, True])
def test_a_segment_crosses_the_wire_unchanged_but_for_its_fleet_wide_actor(audio):
    seg = segment(audio=audio)
    back = decode_segment(encode_segment(seg), actor=105, context=3, audio_shape=(5,) if audio else None)
    assert back.actor == 105 and back.version == 4 and back.context == 3 and back.terminated
    for name in ("frames", "actions", "logp", "rewards", "bad"):
        np.testing.assert_array_equal(getattr(back, name), getattr(seg, name))
    if audio:
        np.testing.assert_array_equal(back.audio, seg.audio)
        np.testing.assert_array_equal(back.audio_mask, seg.audio_mask)
    else:
        assert back.audio is None


def test_a_malformed_segment_never_reaches_the_learner(monkeypatch):
    short = segment()
    short.frames = short.frames[:-1]  # one frame missing: the stacks would read the wrong frames
    with pytest.raises(ValueError, match="frames"):
        decode_segment(encode_segment(short), actor=100, context=3, audio_shape=None)
    wild = segment()
    wild.actions[0, 0] = spec.ACTION_NVEC[0]  # one past the head's last action
    with pytest.raises(ValueError, match="range"):
        decode_segment(encode_segment(wild), actor=100, context=3, audio_shape=None)
    data = encode_segment(segment())
    monkeypatch.setattr(fleet_mod.spec, "SPEC_VERSION", "another")
    with pytest.raises(ValueError, match="spec"):
        decode_segment(data, actor=100, context=3, audio_shape=None)


def test_a_segment_that_does_not_fit_the_learners_policy_never_trains():
    """A segment without the frames the policy reaches back into would index wrapped-around frames; audio that is
    not float32 of the feature's shape would crash the update."""
    with pytest.raises(ValueError, match="context"):
        decode_segment(encode_segment(segment(context=0)), actor=100, context=3, audio_shape=None)
    wide = segment(audio=True)
    wide.audio = wide.audio.astype(np.float64)
    with pytest.raises(ValueError, match="float64"):
        decode_segment(encode_segment(wide), actor=100, context=3, audio_shape=(5,))
    for shape in ((4,), (5, 1)):
        with pytest.raises(ValueError, match="audio"):
            decode_segment(encode_segment(segment(audio=True)), actor=100, context=3, audio_shape=shape)
    short_mask = segment(audio=True)
    short_mask.audio_mask = short_mask.audio_mask[:-1]
    with pytest.raises(ValueError, match="mask"):
        decode_segment(encode_segment(short_mask), actor=100, context=3, audio_shape=(5,))
    with pytest.raises(ValueError, match="no audio"):
        decode_segment(encode_segment(segment()), actor=100, context=3, audio_shape=(5,))
    with pytest.raises(ValueError, match="does not hear"):
        decode_segment(encode_segment(segment(audio=True)), actor=100, context=3, audio_shape=None)
    narrow = segment(audio=True)
    narrow.audio = narrow.audio.astype(np.float16)  # widens to float32 without loss: taken, as float32
    assert decode_segment(encode_segment(narrow), actor=100, context=3, audio_shape=(5,)).audio.dtype == np.float32


def test_an_episode_line_never_raises_on_an_odd_summary():
    assert episode_line(3, {"episode": 2, "round_reached": 4, "return": 1.5, "length": 90, "reason": "downed"}) \
        == "  actor 3 episode 2: round 4, return 1.5, 90 steps (downed)"
    for odd in ({}, {"return": None, "length": None, "reason": None}, {"return": "x", "episode": [1]},
                {"return": float("nan"), "round_reached": float("inf"), "reason": 7}, {"return": True}):
        line = episode_line(3, odd)
        assert "return ?" in line and "None" not in line
    assert "return 2.5, 7 steps" in episode_line(1, {"return": np.float32(2.5), "length": np.int64(7)})


def test_settings_that_change_what_an_action_does_are_named():
    ours = {"dvars": {"sensitivity": "2", "cg_fov": "80"}, "binds": {"mouse1": "+attack", "f": "+activate"}}
    assert settings_differences(ours, json.loads(json.dumps(ours))) == []
    theirs = {"dvars": {"sensitivity": "3", "cg_fov": "80"}, "binds": {"mouse1": "+attack", "e": "+activate"}}
    diffs = settings_differences(ours, theirs)
    assert any(d.startswith("sensitivity") for d in diffs)
    assert any(d.startswith("bind e") for d in diffs) and any(d.startswith("bind f") for d in diffs)
    assert "Plutonium" in settings_differences(ours, None)[0]


@pytest.fixture
def server(bc_checkpoint, tmp_path):
    run_dir = tmp_path / "rl_test"
    run_dir.mkdir()
    config = RLConfig(init=str(bc_checkpoint), env="real")
    settings = {"dvars": {"sensitivity": "2"}, "binds": {"mouse1": "+attack"}}
    srv = FleetServer("127.0.0.1:0", TOKEN, config=config, run_dir=run_dir, settings=settings,
                      context=3, audio_shape=None, say=lambda m: None).start()  # the checkpoint's offsets (0, 1, 3)
    yield srv
    srv.close()


def client(srv, name="rig2", token=TOKEN):
    host, port = srv.address
    return FleetClient(f"{host}:{port}", token, name, timeout_s=10)


def test_the_learner_turns_away_a_machine_that_would_train_on_other_rules(server):
    with pytest.raises(FleetError, match="token"):
        client(server, token="wrong").hello(2, server.settings)
    with pytest.raises(FleetError, match="settings differ.*sensitivity"):
        client(server).hello(2, {"dvars": {"sensitivity": "9"}, "binds": {"mouse1": "+attack"}})
    c = client(server)
    for field, value, match in (("spec_version", "old", "spec"), ("sha", "0" * 40, "commit")):
        code, body = c._post_json("/hello", {"name": "rig2", "actors": 2, "settings": server.settings,
                                             **{**provenance(), field: value}})
        assert code == 409 and match in body["error"]
    assert server.workers == {}  # nothing refused was let in


def test_each_machine_gets_its_own_block_of_actors(server, bc_checkpoint):
    a, b = client(server, "rig2"), client(server, "rig3")
    hello_a = a.hello(2, server.settings)
    hello_b = b.hello(3, server.settings)
    assert (hello_a["first_actor"], hello_b["first_actor"]) == (ACTOR_BLOCK, 2 * ACTOR_BLOCK)
    assert a.hello(2, server.settings)["first_actor"] == ACTOR_BLOCK  # a restarted worker keeps its block
    assert hello_a["run"] == "rl_test" and RLConfig(**hello_a["config"]).init == str(bc_checkpoint)
    assert a.init() == bc_checkpoint.read_bytes()

    assert b.segment(1, encode_segment(segment())) == 200
    kind, actor, seg = server.inbox.get_nowait()
    assert (kind, actor, seg.actor) == ("segment", 2 * ACTOR_BLOCK + 1, 2 * ACTOR_BLOCK + 1)
    assert b.segment(3, encode_segment(segment())) == 400  # rig3 has actors 0..2
    assert a.episode({"actor": 1, "episode": 7, "round_reached": 3, "terms": {"kill": 1.0}}) == 200
    kind, actor, payload = server.inbox.get_nowait()
    assert actor == payload["actor"] == ACTOR_BLOCK + 1 and payload["round_reached"] == 3
    assert "terms" not in payload  # numbers only
    assert a.heartbeat(2, 2) == 200 and server.remote_actors_alive() == 2
    assert {w["name"] for w in server.snapshot()["workers"]} == {"rig2", "rig3"}


def test_a_machine_must_say_hello_before_it_sends_anything(server):
    c = client(server, "stranger")
    assert c.segment(0, encode_segment(segment())) == 409
    assert c.weights(-1)[0] == 409
    assert server.inbox.empty()


def test_weights_go_out_only_when_they_changed(server, bc_checkpoint, tmp_path):
    c = client(server)
    c.hello(1, server.settings)
    assert c.weights(-1)[0] == 204  # nothing published yet
    net, _, _ = bc.load(bc_checkpoint)
    path = tmp_path / "weights.pt"
    publish(path, net, 3)
    server.set_weights(3, path)
    code, version, blob = c.weights(-1)
    assert (code, version, blob) == (200, 3, path.read_bytes())
    assert c.weights(3)[0] == 304


def test_two_machines_with_one_name_are_not_merged(server):
    first, second = client(server, "bazzite"), client(server, "bazzite")  # two processes, two instance ids
    assert first.hello(2, server.settings)["first_actor"] == ACTOR_BLOCK
    with pytest.raises(FleetError, match="another machine is already playing as bazzite.*--name"):
        second.hello(2, server.settings)
    assert second.segment(0, encode_segment(segment())) == 409  # nor can it send under the first one's name
    assert server.inbox.empty() and server.workers["bazzite"].instance == first.instance

    server.workers["bazzite"].last_seen -= fleet_mod.STALE_S + 1  # the first went quiet: it was restarted
    assert second.hello(3, server.settings)["first_actor"] == ACTOR_BLOCK  # and its PC keeps the block
    assert len(server.workers) == 1 and server.workers["bazzite"].actors == 3
    assert first.segment(0, encode_segment(segment())) == 409  # the old process is now the stranger
    assert second.segment(0, encode_segment(segment())) == 200


def test_a_restarted_worker_waits_for_its_old_self_instead_of_exiting(server, tmp_path):
    """`fleet.py prep` restarts a PC's worker after an update; the new process says hello before the learner has
    stopped hearing the old one. It must wait and ask again, not exit as for a refusal a person has to fix."""
    old = client(server, "bazzite")
    old.hello(1, server.settings)
    server.workers["bazzite"].last_seen -= fleet_mod.STALE_S - 0.5  # quiet for nearly long enough
    new = client(server, "bazzite")
    with pytest.raises(NameInUse) as refused:
        new.hello(1, server.settings)
    assert 0 < refused.value.retry_after <= 1.0

    said = []
    worker = FleetWorker(new, WorkerOptions(actors=1, out_root=str(tmp_path), retry_s=0.1),
                         settings=server.settings, say=said.append)
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    thread.start()  # no exception escapes: it waits, then asks again
    deadline = time.time() + 10
    while not any("asking again" in m for m in said) and time.time() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive() and any("already playing as bazzite" in m for m in said)
    assert json.loads(worker.status_path.read_text())["state"] == "stopped"


def test_a_machine_plays_only_the_actors_it_said_hello_with(server):
    c = client(server)
    for actors in (0, ACTOR_BLOCK + 1, "x", None, True, 2.0):
        code, body = c._post_json("/hello", {"name": "rig2", "instance": c.instance, "actors": actors,
                                             "settings": server.settings, **provenance()})
        assert code == 400 and "actors" in body["error"], actors
    assert server.workers == {}  # refused before it was registered

    c.hello(2, server.settings)
    for actor in (2, -1, ACTOR_BLOCK, "1", None, 1.5):
        assert c.episode({"actor": actor, "episode": 1, "return": 1.0}) == 400, actor
    code, _ = c._post_json("/episode", {"episode": 1})  # no actor at all
    assert code == 400 and server.inbox.empty()

    assert c.heartbeat(50, 50) == 200  # a heartbeat cannot add actors to the block
    assert server.workers["rig2"].actors == 2 and server.workers["rig2"].alive == 2
    assert c.segment(5, encode_segment(segment())) == 400 and c.segment(2, encode_segment(segment())) == 400
    assert c._post_json("/heartbeat", {"actors": 2, "alive": "x"})[0] == 400
    assert c._request("POST", "/segment", encode_segment(segment()), {"X-Actor": "one"})[0] == 400
    assert c._request("POST", "/segment", b"not a segment at all", {"X-Actor": "0"})[0] == 400
    assert server.inbox.empty()


def test_a_bad_episode_summary_is_cleaned_before_the_learner_sees_it(server):
    c = client(server)
    c.hello(2, server.settings)
    code, _, _ = c._request("POST", "/episode", b'{"actor": 1, "episode": 3, "return": null, "length": "x", '
                                             b'"round_reached": NaN, "seconds": 12.5, "reason": "downed", '
                                             b'"shots": true, "terms": {"kill": 1}, "extra": 9}')
    assert code == 200
    kind, actor, payload = server.inbox.get_nowait()
    assert actor == ACTOR_BLOCK + 1
    assert payload == {"actor": ACTOR_BLOCK + 1, "episode": 3, "seconds": 12.5, "reason": "downed",
                       "machine": "rig2"}
    assert "return ?" in episode_line(actor, payload)  # and the learner's log line copes with what is missing
    assert c.episode({"actor": 0, "reason": ["not", "words"]}) == 200
    assert "reason" not in server.inbox.get_nowait()[2]
    assert c._request("POST", "/episode", b"[1, 2]")[0] == 400  # not a JSON object


def test_a_segment_that_does_not_fit_is_refused_and_counted(server):
    c = client(server)
    c.hello(1, server.settings)
    assert c.segment(0, encode_segment(segment(context=0))) == 400
    assert c.segment(0, encode_segment(segment(audio=True))) == 400  # the server's policy does not hear
    assert server.workers["rig2"].refused == 2 and server.inbox.empty()
    assert c.segment(0, encode_segment(segment())) == 200


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_run_trains_on_its_own_games_and_another_machines(bc_checkpoint, tmp_path, monkeypatch):
    """The learner with one rehearsal actor of its own, and a worker with one more, in one PPO run: the worker's games
    arrive under its own actor numbers, and when the run ends the worker stops its games and goes."""
    monkeypatch.chdir(tmp_path)
    port = free_port()
    config = RLConfig(init=str(bc_checkpoint), env="synthetic", n_actors=1, total_steps=1200, segment_steps=48,
                      batch_steps=384, critic_warmup_updates=1, minibatch_size=96, device="cpu",
                      synthetic={"max_steps": 120}, listen=f"127.0.0.1:{port}")
    said: list[str] = []
    worker = FleetWorker(FleetClient(f"127.0.0.1:{port}", TOKEN, "rig2", timeout_s=10),
                         WorkerOptions(actors=1, out_root=str(tmp_path / "fleet"), lost_s=3.0, poll_s=0.2,
                                       retry_s=0.2),
                         say=said.append)
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,), kwargs={"once": True}, daemon=True)
    thread.start()
    try:
        train(config, tmp_path / "run", say=lambda m: None, fleet_token=TOKEN)
    finally:
        thread.join(timeout=60)
        stop.set()
    assert not thread.is_alive()
    games = [json.loads(line) for line in (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()]
    actors = {g["actor"] for g in games}
    assert 0 in actors and ACTOR_BLOCK in actors, actors
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["step"] >= 1200 and max(r["machines"] for r in rows) == 2
    snapshot = json.loads((tmp_path / "run" / "fleet.json").read_text())
    assert snapshot["workers"][0]["name"] == "rig2" and snapshot["workers"][0]["segments"] > 0
    # fleet.json holds the last update's window only, and either machine may have sent nothing into that batch;
    # across the run's rows, both must have.
    assert "stats" in snapshot["workers"][0] and "stats" in snapshot["learner"]
    per_machine = [r["per_machine"] for r in rows]
    assert all(set(p) <= {"0", "1"} for p in per_machine)
    for m in ("0", "1"):
        assert sum((p.get(m) or {}).get("steps", 0) for p in per_machine) > 0, m
    assert "rig2" not in json.dumps(rows)  # metrics.jsonl's last row is published: numbers, not names
    joined = json.loads((tmp_path / "fleet" / "run" / "config.json").read_text())
    assert joined["first_actor"] == ACTOR_BLOCK and joined["seed"] == config.seed + ACTOR_BLOCK
    assert any(m.startswith("left run run") for m in said), said
    assert json.loads((tmp_path / "fleet" / "status.json").read_text())["state"] == "stopped"


def test_a_worker_with_no_learner_says_it_is_waiting(tmp_path):
    """What the Training Room shows for a gaming PC between runs (publish_live.py --worker reads this file)."""
    worker = FleetWorker(FleetClient(f"127.0.0.1:{free_port()}", TOKEN, "rig2", timeout_s=2),
                         WorkerOptions(actors=2, out_root=str(tmp_path), retry_s=0.1), say=lambda m: None)
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not worker.status_path.exists() and time.time() < deadline:
        time.sleep(0.05)
    status = json.loads(worker.status_path.read_text())
    assert status["state"] == "waiting" and status["actors"] == 2 and status["at"] > time.time() - 10
    stop.set()
    thread.join(timeout=10)
    assert json.loads(worker.status_path.read_text())["state"] == "stopped"
