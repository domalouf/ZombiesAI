import json
import threading
import time
from dataclasses import replace

import pytest
import torch

from zombiesai.demos import bc
from zombiesai.rl import discovery
from zombiesai.rl.discovery import (MAGIC, Announcer, Beacon, BeaconError, BeaconListener, choose, encode_beacon,
                                    verify_beacon)
from zombiesai.rl.fleet import FleetClient, FleetServer, FleetWorker, WorkerOptions, provenance
from zombiesai.rl.parallel_ppo import RLConfig

TOKEN = "test-token"
NOW = 1_800_000_000.0


def beacon(**kw) -> Beacon:
    return Beacon(**{"host": "10.0.0.5", "port": 47860, "run": "rl5", "sha": "a" * 40, "spec_version": "s1",
                     "started": NOW - 100, "t": NOW, "name": "gamingpc", **kw})


def test_a_beacon_signed_with_the_token_is_taken_whole():
    b = beacon()
    data = encode_beacon(b, TOKEN)
    assert data.startswith(MAGIC) and len(data) < discovery.MAX_BEACON
    assert verify_beacon(data, TOKEN, now=NOW + 3) == b and b.address == "10.0.0.5:47860"


def test_a_forged_or_foreign_beacon_is_ignored():
    data = encode_beacon(beacon(), TOKEN)
    with pytest.raises(BeaconError) as error:
        verify_beacon(data, "another fleet's token", now=NOW)
    assert error.value.kind == "token"
    # The address is inside what is signed: pointing a genuine beacon elsewhere breaks it.
    forged = data.replace(b'"host":"10.0.0.5"', b'"host":"10.0.0.66"')
    assert forged != data
    with pytest.raises(BeaconError) as error:
        verify_beacon(forged, TOKEN, now=NOW)
    assert error.value.kind == "token"
    for other in (b"", b"hello", b"M-SEARCH * HTTP/1.1\r\n", MAGIC + b" short " + b"{}", b"x" * 5000):
        with pytest.raises(BeaconError) as error:
            verify_beacon(other, TOKEN, now=NOW)
        assert error.value.kind == "foreign"


def test_an_old_beacon_or_a_wrong_clock_is_named():
    data = encode_beacon(beacon(t=NOW - 3600), TOKEN)
    with pytest.raises(BeaconError, match="clock") as error:
        verify_beacon(data, TOKEN, now=NOW)
    assert error.value.kind == "stale"


def test_a_signed_beacon_that_is_not_a_learners_is_refused():
    def signed(fields: dict) -> bytes:
        body = json.dumps(fields).encode()
        return MAGIC + b" " + discovery._mac(TOKEN, body) + b" " + body

    good = json.loads(json.dumps(beacon().__dict__))
    for bad in ({**good, "port": "47860"}, {**good, "port": 0}, {**good, "port": True}, {**good, "host": ""},
                {**good, "t": "now"}, {k: v for k, v in good.items() if k != "run"}, [1, 2]):
        with pytest.raises(BeaconError) as error:
            verify_beacon(signed(bad), TOKEN, now=NOW)
        assert error.value.kind == "malformed", bad


def test_the_worker_picks_its_run_then_a_learner_that_will_take_it_then_the_newest():
    old = beacon(host="a", run="rl4", started=NOW - 5000)
    new = beacon(host="b", run="rl6", started=NOW - 10)
    other_commit = beacon(host="c", run="rl7", started=NOW, sha="b" * 40)
    assert choose([old, new], sha="a" * 40, spec_version="s1") == new
    assert choose([old, new, other_commit], sha="a" * 40, spec_version="s1") == new  # c would refuse us
    assert choose([other_commit], sha="a" * 40, spec_version="s1") == other_commit  # so its hello says why
    assert choose([old, new], run="rl4") == old
    assert choose([old, new], run="rl9") is None and choose([]) is None


def test_a_learner_on_loopback_announces_only_to_this_machine():
    a = Announcer(lambda h: beacon(host=h), TOKEN, bound_host="127.0.0.1")
    assert a.targets() == [("127.0.0.1", "127.0.0.1")]
    fixed = Announcer(lambda h: beacon(host=h), TOKEN, bound_host="0.0.0.0", targets=["10.0.0.255"])
    assert fixed.targets() == [("10.0.0.255", None)]  # the address is the one the kernel sends from
    with pytest.raises(ValueError, match="token"):
        Announcer(lambda h: beacon(host=h), "", bound_host="127.0.0.1")
    for addr, brd in discovery.broadcast_addresses():  # whatever this machine has: never loopback
        assert not addr.startswith("127.") and brd != "0.0.0.0"


@pytest.fixture
def bc_checkpoint(tmp_path):
    config = bc.BCConfig(frame_offsets=(0, 1, 3), hidden=32)
    torch.manual_seed(0)
    path = tmp_path / "bc.pt"
    bc.save(path, bc.build_net(config), config, 0, {}, {"fire_duty": 0.1})
    return path


def learner(bc_checkpoint, tmp_path, name: str, beacon_port: int) -> FleetServer:
    run_dir = tmp_path / name
    run_dir.mkdir()
    return FleetServer("127.0.0.1:0", TOKEN, config=RLConfig(init=str(bc_checkpoint), env="synthetic"),
                       run_dir=run_dir, settings=None, context=3, audio_shape=None, say=lambda m: None,
                       beacon_port=beacon_port, beacon_interval_s=0.1)


def test_a_worker_finds_the_learner_by_its_beacon_and_again_when_it_moves(bc_checkpoint, tmp_path):
    """Loopback only: the learner's beacon to a worker's listener, the worker's hello to the address it named;
    then that learner goes, a run continued elsewhere starts announcing, and the same worker finds it."""
    listener = BeaconListener(TOKEN, port=0, bind="127.0.0.1")
    said: list[str] = []
    client = FleetClient(None, TOKEN, "rig2", timeout_s=5)
    worker = FleetWorker(client, WorkerOptions(actors=1, out_root=str(tmp_path / "fleet"), retry_s=5.0),
                         say=said.append, listener=listener)
    assert worker.discovering
    stop = threading.Event()
    first = learner(bc_checkpoint, tmp_path, "rl5", listener.port).start()
    try:
        assert worker._locate(stop)
        assert client.base == f"http://127.0.0.1:{first.address[1]}"
        assert client.hello(1, None)["run"] == "rl5"
        assert any("found the learner of run rl5" in m for m in said)
    finally:
        first.close()
    assert first.announcer.sent > 0 and not first.announcer._thread.is_alive()

    # The old learner's last beacons may still be waiting in the socket: the newer run wins over them.
    second = learner(bc_checkpoint, tmp_path, "rl5b", listener.port).start()
    try:
        assert worker._locate(stop)
        assert client.base == f"http://127.0.0.1:{second.address[1]}"
        hello = client.hello(1, None)
        assert hello["run"] == "rl5b" and client.instance == worker.client.instance
    finally:
        second.close()
        listener.close()


def test_a_beacon_with_another_token_is_reported_not_followed(tmp_path):
    listener = BeaconListener(TOKEN, port=0, bind="127.0.0.1")
    stranger = Announcer(lambda h: beacon(host=h, t=time.time()), "not our token", bound_host="127.0.0.1",
                         port=listener.port)
    said: list[str] = []
    worker = FleetWorker(FleetClient(None, TOKEN, "rig2"), WorkerOptions(out_root=str(tmp_path), retry_s=0.5),
                         say=said.append, listener=listener)
    try:
        stranger.send_once(stranger.targets())
        assert not worker._locate(threading.Event()) and worker.client.base is None
        assert any("another fleet token" in m and "ZOMBIES_FLEET_TOKEN" in m for m in said), said
        assert json.loads(worker.status_path.read_text())["state"] == "waiting"
    finally:
        listener.close()


def test_the_worker_waits_for_the_run_it_was_told_to_join(tmp_path):
    listener = BeaconListener(TOKEN, port=0, bind="127.0.0.1")
    ours = provenance()
    other = Announcer(lambda h: replace(beacon(host=h, run="rl9", t=time.time()), sha=ours["sha"]), TOKEN,
                      bound_host="127.0.0.1", port=listener.port)
    said: list[str] = []
    worker = FleetWorker(FleetClient(None, TOKEN, "rig2"),
                         WorkerOptions(out_root=str(tmp_path), retry_s=0.5, run="rl5"), say=said.append,
                         listener=listener)
    try:
        other.send_once(other.targets())
        assert not worker._locate(threading.Event())
        assert any("heard learners for rl9; waiting for run rl5" in m for m in said), said
    finally:
        listener.close()
