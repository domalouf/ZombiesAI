import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import stats


def constant(action, n=100):
    return np.tile(np.array(spec.action_tuple(action)), (n, 1))


def test_behaviour_stats_read_off_what_the_hands_did():
    actions = constant(spec.make_action(forward=1, fire=1, yaw=6.0, button="reload"))
    out = stats.behaviour_stats(actions)
    assert out["fire_duty"] == 1.0 and out["moving_duty"] == 1.0
    assert out["abs_yaw_deg_per_s"] == pytest.approx(6.0 * spec.DECISION_HZ)
    assert out["reload_per_min"] == pytest.approx(60.0 * spec.DECISION_HZ)
    assert out["copy_joint"] == 1.0


def test_a_policy_that_never_fires_fails_the_rollout_check():
    human = stats.behaviour_stats(constant(spec.make_action(fire=1)))
    silent = stats.behaviour_stats(constant(spec.make_action()))
    report = stats.divergence_report(silent, human)
    assert not report["passed"] and "fire_duty" in report["failed"]
    assert stats.divergence_report(human, human)["passed"]


def test_a_statistic_neither_side_ever_does_is_not_a_failure():
    neither = stats.behaviour_stats(constant(spec.make_action()))
    assert stats.divergence_report(neither, neither)["passed"]


def test_balanced_accuracy_does_not_reward_always_saying_none():
    target = np.zeros((100, len(spec.ACTION_NVEC)), dtype=np.int64)
    target[:5, spec.BUTTON] = spec.BUTTONS.index("reload")
    lazy = np.zeros_like(target)
    report = stats.accuracy_report(lazy, target)
    assert report["per_head"]["button"] == pytest.approx(0.95)
    assert report["balanced"]["button"] == pytest.approx(0.5)  # one class perfect, one never found
    assert not report["beats_baseline"]["button"]


def test_action_inertia_is_measured_against_the_human_not_an_absolute():
    rng = np.random.default_rng(0)
    human = rng.integers(0, spec.ACTION_NVEC, size=(500, len(spec.ACTION_NVEC)))
    frozen = constant(spec.make_action(), 500)
    report = stats.inertia_report(frozen, human)
    assert not report["passed"] and "copy_joint" in report["inert_heads"]
    assert stats.inertia_report(human, human)["passed"]
