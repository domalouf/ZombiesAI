"""How fast the agent looks: a fresh policy's starting odds, the smoothed mouse, and degrees that mean the same
on every PC (counts per degree from the game's own sensitivity)."""

import threading
import time

import numpy as np
import torch

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig, FakeSink
from zombiesai.rl.actors import FALLBACK_COUNTS_PER_DEGREE, counts_per_degree
from zombiesai.rl.config import RLConfig
from zombiesai.rl.learner import prepare_init


def head_probs(net, head: int) -> np.ndarray:
    starts = np.concatenate([[0], np.cumsum(spec.ACTION_NVEC)])
    logits = net.actor.bias.detach()[starts[head]:starts[head + 1]]
    return torch.softmax(logits, 0).numpy()


def test_a_fresh_policy_starts_with_small_turns_not_uniform_flicks(tmp_path):
    config = prepare_init(RLConfig(init="fresh", device="cpu"), tmp_path / "run")
    net, _, _ = bc.load(config.init)
    for head, bins in ((spec.YAW, spec.YAW_BINS_DEG), (spec.PITCH, spec.PITCH_BINS_DEG)):
        p, bins = head_probs(net, head), np.abs(np.asarray(bins))
        assert np.argmax(p) == list(bins).index(0.0)  # no turn is the likeliest
        assert np.all(np.diff(p[bins.argsort()]) <= 1e-6)  # and each bigger turn is less likely
        assert p.min() > 0  # but every bin can still be explored
    yaw = head_probs(net, spec.YAW) @ np.abs(spec.YAW_BINS_DEG)
    assert yaw < 3.0  # degrees a decision; uniform odds would be ~11.6 (175 deg/s of jitter at 15 Hz)


def install(root, text: str):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.cfg").write_text(text)  # what `fleet.py prep` installs on every PC
    return str(root)


def test_counts_per_degree_come_from_the_games_sensitivity(tmp_path, monkeypatch):
    monkeypatch.setattr("zombiesai.demos.game_settings.candidate_configs", lambda: [])
    said = []
    demos = install(tmp_path / "a", 'seta sensitivity "5"\nseta m_yaw "0.022"\n')
    assert abs(counts_per_degree(RLConfig(fleet_root=demos), say=said.append) - 1 / 0.11) < 1e-9
    # Half the sensitivity takes twice the counts for the same degrees: the bin means the same turn.
    slow = install(tmp_path / "b", 'seta sensitivity "2.5"\nseta m_yaw "0.022"\n')
    assert abs(counts_per_degree(RLConfig(fleet_root=slow), say=said.append) - 2 / 0.11) < 1e-9
    assert said == []


def test_a_calibrated_number_wins_but_a_big_gap_is_said(tmp_path, monkeypatch):
    monkeypatch.setattr("zombiesai.demos.game_settings.candidate_configs", lambda: [])
    root = install(tmp_path / "a", 'seta sensitivity "5"\nseta m_yaw "0.022"\n')
    said = []
    assert counts_per_degree(RLConfig(fleet_root=root, counts_per_degree=9.3), say=said.append) == 9.3
    assert said == []  # a calibration a few percent off the formula is expected
    assert counts_per_degree(RLConfig(fleet_root=root, counts_per_degree=18.0), say=said.append) == 18.0
    assert len(said) == 1 and "1.98x" in said[0]


def test_no_config_falls_back_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr("zombiesai.demos.game_settings.candidate_configs", lambda: [])
    said = []
    assert counts_per_degree(RLConfig(fleet_root=str(tmp_path / "none")), say=said.append) == FALLBACK_COUNTS_PER_DEGREE
    assert len(said) == 1


def test_the_rl_motor_has_no_dead_zone_to_shave_small_turns():
    dispatcher = ActionDispatcher(FakeSink(), DispatchConfig(counts_per_degree=10.0), motor=True,
                                  motor_dead_zone_deg_s=(0.0, 0.0))
    try:
        dispatcher.motor.set_rate(30.0, -30.0)  # a 2-degree bin at 15 Hz
        assert np.allclose(dispatcher.motor.target, (30.0, -30.0))
    finally:
        dispatcher.close()


class ExclusiveLib:
    """Stands in for libX11/libXtst and fails if two threads are ever inside it at once."""

    def __init__(self):
        self.inside, self.overlaps, self.calls = 0, 0, 0
        self.guard = threading.Lock()

    def _call(self, *args):
        with self.guard:
            self.inside += 1
            self.overlaps += self.inside > 1
            self.calls += 1
        time.sleep(0.0002)
        with self.guard:
            self.inside -= 1
        return 1

    def __getattr__(self, name):
        if name == "get_input_focus":
            return lambda dpy, window, revert: self._call()
        return self._call


def test_the_xtest_sink_never_lets_two_threads_into_xlib_at_once():
    from zombiesai.realgame.xtest import XTestSink

    lib = ExclusiveLib()
    sink = XTestSink(":99", lib=lib)

    def motor():
        for _ in range(200):
            sink.move(3, -1)
            sink.sync()

    thread = threading.Thread(target=motor)
    thread.start()
    for _ in range(200):
        sink.focus(1234)
        sink.key("w", True)
    thread.join()
    assert lib.calls > 800 and lib.overlaps == 0
