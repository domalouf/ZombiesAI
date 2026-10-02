from dataclasses import replace

import numpy as np
import pytest
import torch

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.demos.agent import BCAgent
from zombiesai.rl.parallel_ppo import (
    FrameHistory,
    Learner,
    RLConfig,
    RunningStd,
    Segment,
    WeightFollower,
    compute_gae,
    kl_to_reference,
    publish,
    stack_indices,
    train,
)

OFFSETS = (0, 1, 3)


@pytest.fixture
def bc_checkpoint(tmp_path):
    config = bc.BCConfig(frame_offsets=OFFSETS, hidden=32)
    torch.manual_seed(0)
    path = tmp_path / "bc.pt"
    bc.save(path, bc.build_net(config), config, 0, {}, {"fire_duty": 0.1})
    return path


def frame(v: int) -> np.ndarray:
    return np.full(spec.PIXELS_SHAPE, v % 256, dtype=np.uint8)


def test_the_learner_rebuilds_exactly_the_stacks_the_actor_acted_on():
    """The actor's FrameHistory and the learner's index arithmetic over a segment must agree step for step --
    right after a reset (where the stack clamps) and in a segment that starts mid-episode."""
    history = FrameHistory(bc.BCConfig(frame_offsets=OFFSETS).offsets)
    offsets = history.offsets
    history.reset(frame(1))
    value = 1
    for n_steps in (5, 7):  # the first segment starts at a reset, the second continues the episode
        frames = history.context() + [history.frames[-1]]
        acted_on = []
        for _ in range(n_steps):
            acted_on.append(history.stack())
            value += 1
            history.push(frame(value))
            frames.append(history.frames[-1])
        frames = np.stack(frames)
        idx = stack_indices(history.depth, n_steps, offsets)
        for t in range(n_steps):
            np.testing.assert_array_equal(frames[idx[t]], acted_on[t])
        np.testing.assert_array_equal(frames[idx[n_steps]], history.stack())  # the bootstrap observation


def test_gae_bootstraps_a_cut_segment_but_not_a_finished_episode():
    rewards, values = np.array([1.0, 0.0, 2.0], np.float32), np.array([0.5, 0.5, 0.5], np.float32)
    adv_end, ret_end = compute_gae(rewards, values, last_value=10.0, terminated=True, gamma=0.9, lam=1.0)
    adv_cut, _ = compute_gae(rewards, values, last_value=10.0, terminated=False, gamma=0.9, lam=1.0)
    # lam=1: the return is the discounted sum, plus the bootstrap only for the cut segment
    assert ret_end[0] == pytest.approx(1.0 + 0.9 * 0.0 + 0.81 * 2.0)
    assert adv_cut[0] - adv_end[0] == pytest.approx(0.9**3 * 10.0)


def test_kl_to_the_bc_policy_is_zero_at_the_start_and_grows_when_the_policy_moves():
    logits = torch.randn(4, sum(spec.ACTION_NVEC))
    assert kl_to_reference(logits, logits, spec.ACTION_NVEC).abs().max() < 1e-5
    moved = logits.clone()
    moved[:, 0] += 3.0
    assert (kl_to_reference(moved, logits, spec.ACTION_NVEC) > 0).all()


def test_running_std_tracks_the_return_scale():
    s = RunningStd(gamma=0.0)  # gamma 0: the "return" is the reward itself
    s.update(0, np.random.default_rng(0).normal(0, 5, 5000).astype(np.float32), terminated=False)
    assert s.std == pytest.approx(5.0, rel=0.1)


def synthetic_segments(learner: Learner, n: int = 3, steps: int = 40, seed: int = 0) -> list[Segment]:
    """Segments whose behaviour log-probs come from the learner's own (initial) policy, as a lag-0 actor's would."""
    rng = np.random.default_rng(seed)
    depth = max(learner.offsets)
    out = []
    for k in range(n):
        frames = rng.integers(0, 255, (depth + steps + 1, *spec.PIXELS_SHAPE), dtype=np.uint8)
        actions = np.stack([rng.integers(0, m, steps) for m in spec.ACTION_NVEC], axis=1)
        seg = Segment(actor=k, version=0, context=depth, frames=frames, actions=actions,
                      logp=np.zeros(steps, np.float32), rewards=rng.normal(0, 1, steps).astype(np.float32),
                      bad=rng.random(steps) < 0.1, terminated=k == 0)
        with torch.no_grad():
            px, _, _ = learner._gather([seg], np.arange(steps), np.zeros(steps, int))
            dist = learner.net.dist(px)
            seg.logp = dist.log_prob(torch.from_numpy(actions)).numpy().astype(np.float32)
        out.append(seg)
    return out


def test_the_critic_warms_up_before_the_policy_moves(bc_checkpoint, tmp_path):
    config = RLConfig(init=str(bc_checkpoint), critic_warmup_updates=1, minibatch_size=32, update_epochs=2,
                      device="cpu", target_kl=None)
    learner = Learner(config, torch.device("cpu"))
    actor_before = learner.net.actor.weight.detach().clone()
    encoder_before = next(learner.net.encoder.parameters()).detach().clone()
    critic_before = learner.net.critic.weight.detach().clone()
    stats = learner.update(synthetic_segments(learner))
    assert stats["warmup"]
    assert torch.equal(learner.net.actor.weight, actor_before)
    assert torch.equal(next(learner.net.encoder.parameters()), encoder_before)
    assert not torch.equal(learner.net.critic.weight, critic_before)

    stats = learner.update(synthetic_segments(learner, seed=1))
    assert not stats["warmup"] and not torch.equal(learner.net.actor.weight, actor_before)
    assert stats["kl_ref"] >= 0 and 0 < stats["bad_step_frac"] < 0.3 and stats["kl_coef"] < config.kl_coef

    path = tmp_path / "checkpoint.pt"
    learner.checkpoint(path, step=240)
    net, bc_config, meta = bc.load(path)
    assert meta["rl"]["updates"] == 2 and meta["human_behaviour"] == {"fire_duty": 0.1}
    assert bc_config.offsets == learner.offsets
    agent = BCAgent(path)
    assert agent.act({"pixels": frame(3)}).shape == (len(spec.ACTION_NVEC),)


def test_actors_follow_published_weights(bc_checkpoint, tmp_path):
    net, _, _ = bc.load(bc_checkpoint)
    follower = WeightFollower(tmp_path / "weights.pt")
    assert follower.poll(net) == -1  # nothing published yet
    other, _, _ = bc.load(bc_checkpoint)
    with torch.no_grad():
        other.actor.weight.add_(1.0)
    publish(tmp_path / "weights.pt", other, 3)
    assert follower.poll(net) == 3 and torch.equal(net.actor.weight, other.actor.weight)


def test_parallel_training_runs_end_to_end_on_the_sim(bc_checkpoint, tmp_path):
    """Two actor processes, the learner, warm-up then PPO, and a checkpoint every BC tool can play."""
    config = RLConfig(init=str(bc_checkpoint), env="sim", n_actors=2, total_steps=1200, segment_steps=48,
                      batch_steps=384, critic_warmup_updates=1, minibatch_size=96, device="cpu",
                      sim={"max_steps": 200})
    checkpoint = train(config, tmp_path / "run", say=lambda m: None)
    import json

    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) >= 3 and rows[0]["warmup"] and not rows[-1]["warmup"]
    assert rows[-1]["step"] >= 1200 and "return_mean" in rows[-1] and "round_reached_mean" in rows[-1]
    assert json.loads((tmp_path / "run" / "config.json").read_text())["env"] == "nacht-render"
    games = [json.loads(line) for line in (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()]
    assert games and {"round_reached", "seconds", "shots", "hits", "step", "actor"} <= set(games[0])
    assert all(0 <= g["hits"] <= g["shots"] and g["seconds"] > 0 for g in games)
    assert BCAgent(checkpoint).act({"pixels": frame(9)}).shape == (len(spec.ACTION_NVEC),)


def test_a_continued_run_keeps_its_anchor_on_the_original_prior(bc_checkpoint, tmp_path):
    config = RLConfig(init=str(bc_checkpoint), critic_warmup_updates=1, minibatch_size=32, update_epochs=2,
                      device="cpu", target_kl=None)
    first = Learner(config, torch.device("cpu"))
    for seed in range(3):
        first.update(synthetic_segments(first, seed=seed))
    one = tmp_path / "one.pt"
    first.checkpoint(one, step=100)
    # a checkpoint from before the anchor was recorded: only its parent is known
    blob = torch.load(one, weights_only=False)
    del blob["rl"]["reference"]
    torch.save(blob, one)

    second = Learner(replace(config, init=str(one)), torch.device("cpu"))
    assert second.reference == str(bc_checkpoint) and second.updates == 3
    assert second.kl_coef == first.kl_coef < config.kl_coef
    stats = second.update(synthetic_segments(second, seed=9))
    assert not stats["warmup"] and stats["kl_ref"] > 0  # anchored to BC, which the policy has moved from
    two = tmp_path / "two.pt"
    second.checkpoint(two, step=200)
    assert Learner(replace(config, init=str(two)), torch.device("cpu")).reference == str(bc_checkpoint)
