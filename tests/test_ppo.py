import json

import numpy as np
import pytest
import torch

from zombiesai import spec
from zombiesai.rl.agent import PolicyAgent, save_checkpoint
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.rl.networks import ActorCritic, ObsFlattener, action_nvec
from zombiesai.rl.ppo import PPOConfig, compute_gae, train
from zombiesai.sim.nacht_sim import NachtSim, SimConfig


def col(*xs):
    return torch.tensor(xs, dtype=torch.float32)[:, None]


def test_gae_by_hand():
    # gamma 0.9, lambda 0.8, episode ends at the last step so last_value must be ignored.
    adv, ret = compute_gae(col(1, 1, 1), col(0.5, 0.5, 0.5), col(0, 0, 1), torch.tensor([10.0]), 0.9, 0.8)
    np.testing.assert_allclose(adv[:, 0].numpy(), [1.8932, 1.31, 0.5], rtol=1e-5)
    np.testing.assert_allclose(ret.numpy(), adv.numpy() + 0.5)


def test_gae_with_lambda_one_is_the_monte_carlo_return():
    _, ret = compute_gae(col(1, 1, 1), col(3, -2, 7), col(0, 0, 1), torch.tensor([99.0]), 0.9, 1.0)
    np.testing.assert_allclose(ret[:, 0].numpy(), [1 + 0.9 + 0.81, 1 + 0.9, 1], rtol=1e-6)


def test_gae_bootstraps_an_unfinished_rollout():
    _, ret = compute_gae(col(0, 0), col(0, 0), col(0, 0), torch.tensor([10.0]), 0.5, 1.0)
    np.testing.assert_allclose(ret[:, 0].numpy(), [2.5, 5.0])


def test_folding_the_truncation_bootstrap_into_the_reward_is_exact():
    """Truncating at t=1 with r += gamma * V(final) must match a rollout that simply continued from V(final)."""
    gamma, lam, v_final = 0.9, 0.95, 4.0
    values = col(1.0, 2.0)
    cut, _ = compute_gae(col(1.0, 1.0 + gamma * v_final), values, col(0, 1), torch.tensor([123.0]), gamma, lam)
    cont, _ = compute_gae(col(1.0, 1.0), values, col(0, 0), torch.tensor([v_final]), gamma, lam)
    np.testing.assert_allclose(cut.numpy(), cont.numpy(), rtol=1e-6)


def test_factored_categorical_matches_independent_heads():
    torch.manual_seed(0)
    nvec = spec.ACTION_NVEC
    logits = torch.randn(5, sum(nvec))
    dist = FactoredCategorical(logits, nvec)
    heads = [torch.distributions.Categorical(logits=chunk) for chunk in logits.split(list(nvec), dim=-1)]
    actions = dist.sample()
    assert actions.shape == (5, len(nvec))
    assert torch.all(actions < torch.tensor(nvec))
    expected = sum(h.log_prob(actions[:, i]) for i, h in enumerate(heads))
    torch.testing.assert_close(dist.log_prob(actions), expected)
    torch.testing.assert_close(dist.entropy(), sum(h.entropy() for h in heads))


def test_gradients_are_finite_with_padded_heads():
    logits = torch.randn(4, sum(spec.ACTION_NVEC), requires_grad=True)
    dist = FactoredCategorical(logits, spec.ACTION_NVEC)
    (dist.log_prob(dist.sample()).sum() - dist.entropy().sum()).backward()
    assert torch.isfinite(logits.grad).all()


def test_uniform_logits_have_maximum_entropy():
    nvec = spec.ACTION_NVEC
    dist = FactoredCategorical(torch.zeros(1, sum(nvec)), nvec)
    assert dist.entropy().item() == pytest.approx(sum(np.log(nvec)))


def test_sampling_follows_the_probabilities():
    torch.manual_seed(0)
    dist = FactoredCategorical(torch.log(torch.tensor([[0.7, 0.2, 0.1]])).expand(20_000, 3), (3,))
    freq = torch.bincount(dist.sample()[:, 0], minlength=3).float() / 20_000
    torch.testing.assert_close(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.015, rtol=0)


def test_checkpoint_round_trip_drives_nachtsim(tmp_path):
    env = NachtSim(SimConfig(max_steps=200))
    flat = ObsFlattener(env.observation_space)
    net = ActorCritic(flat.dim, action_nvec(env.action_space), (32, 32))
    cfg = PPOConfig.for_preset("nacht-state", hidden=(32, 32))
    save_checkpoint(tmp_path / "ck.pt", net, flat, False, cfg, step=0)
    agent = PolicyAgent(tmp_path / "ck.pt")
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        action = agent.act(obs)
        spec.action_tuple(action)
        obs, *_ = env.step(action)


@pytest.mark.slow
def test_ppo_learns_cartpole(tmp_path):
    ck = train(PPOConfig.for_preset("cartpole", total_steps=60_000, torch_threads=1), tmp_path / "run")
    final = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[-1])
    assert final["return_mean"] > 150, final
    assert PolicyAgent(ck).act(np.zeros(4, dtype=np.float32)) in (0, 1)
