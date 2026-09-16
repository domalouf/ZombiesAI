"""Behaviour statistics: the evaluation that catches what per-frame accuracy hides.

A cloned policy can be 70% accurate per frame and never pull the trigger, because 'don't fire' is the
majority label and firing is where the points are. So every check here is about *what the policy does over
time* rather than how often it agrees with a single human frame:

* balanced per-head accuracy, against the majority-class baseline it has to beat;
* rollout statistics -- fire duty cycle, mean |yaw|/s, reload rate -- which the plan asks to be within 2x of
  the human's;
* the copy rate, which catches action inertia: a policy conditioned on its own previous action can score
  well by repeating it, and the tell is a copy rate far above the human's own.
"""

import numpy as np

from zombiesai import spec

_YAW_DEG = np.array(spec.YAW_BINS_DEG)
_PITCH_DEG = np.array(spec.PITCH_BINS_DEG)


def behaviour_stats(actions: np.ndarray, dt: float = 1.0 / spec.DECISION_HZ) -> dict[str, float]:
    """Summarise a stretch of play by what the hands did, not by what was on screen."""
    a = np.asarray(actions, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != len(spec.ACTION_NVEC):
        raise ValueError(f"expected (T, {len(spec.ACTION_NVEC)}) actions, got {a.shape}")
    if not len(a):
        raise ValueError("no actions to summarise")
    n = len(a)
    per_minute = 60.0 / (n * dt) if n else 0.0
    yaw = _YAW_DEG[a[:, spec.YAW]]
    pitch = _PITCH_DEG[a[:, spec.PITCH]]
    stats = {
        "steps": float(n),
        "fire_duty": float(a[:, spec.FIRE].mean()),
        "ads_duty": float(a[:, spec.ADS].mean()),
        "sprint_duty": float(a[:, spec.SPRINT].mean()),
        "moving_duty": float(((a[:, spec.FORWARD] != 1) | (a[:, spec.STRAFE] != 1)).mean()),
        "backpedal_duty": float((a[:, spec.FORWARD] == 0).mean()),
        "abs_yaw_deg_per_s": float(np.abs(yaw).mean() / dt),
        "abs_pitch_deg_per_s": float(np.abs(pitch).mean() / dt),
        "yaw_still_frac": float((a[:, spec.YAW] == spec.YAW_BINS_DEG.index(0.0)).mean()),
        "net_yaw_deg_per_s": float(yaw.mean() / dt),
    }
    for i, name in enumerate(spec.BUTTONS):
        if name != "none":
            stats[f"{name}_per_min"] = float((a[:, spec.BUTTON] == i).sum() * per_minute)
    stats["button_any_frac"] = float((a[:, spec.BUTTON] != spec.BUTTONS.index("none")).mean())
    stats.update(copy_rates(a))
    return stats


def copy_rates(actions: np.ndarray) -> dict[str, float]:
    """How often each head repeats its previous value, plus the joint repeat rate."""
    a = np.asarray(actions, dtype=np.int64)
    if len(a) < 2:
        return {f"copy_{h}": float("nan") for h in spec.ACTION_HEADS} | {"copy_joint": float("nan")}
    same = a[1:] == a[:-1]
    out = {f"copy_{head}": float(same[:, i].mean()) for i, head in enumerate(spec.ACTION_HEADS)}
    out["copy_joint"] = float(same.all(axis=1).mean())
    return out


# The statistics the plan asks to be within a factor of the human's; ratios on anything rarer than these
# are noise, not evidence.
ROLLOUT_STATS = ("fire_duty", "ads_duty", "moving_duty", "abs_yaw_deg_per_s", "reload_per_min", "button_any_frac")


def stat_ratios(policy: dict[str, float], human: dict[str, float], keys=ROLLOUT_STATS) -> dict[str, float]:
    """policy/human for each statistic, as a factor >= 1 in whichever direction it is wrong.

    A tiny floor keeps a human statistic of exactly zero from producing an infinite ratio; it makes 'the
    human never did this and neither does the policy' a pass, and 'the policy does it constantly' a failure.
    """
    out = {}
    for key in keys:
        p, h = policy.get(key), human.get(key)
        if p is None or h is None:
            continue
        floor = 1e-3
        ratio = max(abs(p), floor) / max(abs(h), floor)
        out[key] = float(ratio if ratio >= 1.0 else 1.0 / ratio)
    return out


def divergence_report(policy: dict, human: dict, factor: float = 2.0, keys=ROLLOUT_STATS) -> dict:
    ratios = stat_ratios(policy, human, keys)
    failed = {k: v for k, v in ratios.items() if v > factor}
    return {"ratios": ratios, "factor": factor, "failed": failed, "passed": not failed}


def per_head_accuracy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    p, t = np.asarray(predicted, dtype=np.int64), np.asarray(target, dtype=np.int64)
    return (p == t).mean(axis=0)


def balanced_accuracy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean per-class recall for each head: the metric a 95%-'none' button head cannot cheat."""
    p, t = np.asarray(predicted, dtype=np.int64), np.asarray(target, dtype=np.int64)
    out = np.zeros(len(spec.ACTION_NVEC))
    for head, n_values in enumerate(spec.ACTION_NVEC):
        recalls = [float((p[t[:, head] == v, head] == v).mean()) for v in range(n_values) if (t[:, head] == v).any()]
        out[head] = float(np.mean(recalls)) if recalls else float("nan")
    return out


def accuracy_report(predicted: np.ndarray, target: np.ndarray) -> dict:
    from zombiesai.demos.dataset import majority_baseline

    raw = per_head_accuracy(predicted, target)
    balanced = balanced_accuracy(predicted, target)
    baseline = majority_baseline(target)
    return {
        "per_head": {h: float(raw[i]) for i, h in enumerate(spec.ACTION_HEADS)},
        "balanced": {h: float(balanced[i]) for i, h in enumerate(spec.ACTION_HEADS)},
        "majority_baseline": {h: float(baseline[i]) for i, h in enumerate(spec.ACTION_HEADS)},
        "beats_baseline": {h: bool(raw[i] > baseline[i]) for i, h in enumerate(spec.ACTION_HEADS)},
        "mean_balanced": float(np.nanmean(balanced)),
    }


def inertia_report(policy_actions: np.ndarray, human_actions: np.ndarray, factor: float = 1.25) -> dict:
    """Action inertia check: the policy's copy rate against the human's own action autocorrelation."""
    policy, human = copy_rates(policy_actions), copy_rates(human_actions)
    keys = [f"copy_{h}" for h in spec.ACTION_HEADS] + ["copy_joint"]
    excess = {k: float(policy[k] - human[k]) for k in keys}
    over = {k: v for k, v in excess.items() if policy[k] > min(1.0, human[k] * factor) and human[k] < 0.98}
    return {"policy": policy, "human": human, "excess": excess, "inert_heads": sorted(over), "passed": not over}
