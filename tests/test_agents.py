import numpy as np
import pytest

from zombiesai import spec
from zombiesai.agents.random_agent import RandomAgent
from zombiesai.agents.scripted import ScriptedAgent
from zombiesai.rollout import run_episode
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.wrappers import CompactActionWrapper


@pytest.mark.parametrize("agent", [RandomAgent(0), ScriptedAgent()], ids=["random", "scripted"])
def test_agents_emit_valid_actions(agent):
    env = NachtSim(SimConfig(max_steps=400))
    obs, _ = env.reset(seed=0)
    agent.reset()
    for _ in range(400):
        action = agent.act(obs)
        spec.action_tuple(action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break


def test_compact_wrapper():
    env = CompactActionWrapper(NachtSim(SimConfig(max_steps=100)))
    assert env.action_space.n == spec.N_COMPACT
    env.reset(seed=0)
    for i in range(spec.N_COMPACT):
        env.step(i)
    assert env.unwrapped._applied == spec.COMPACT_ACTIONS[spec.N_COMPACT - 1]


@pytest.mark.slow
def test_scripted_outlasts_random():
    env = NachtSim(SimConfig(max_steps=200_000))
    rounds = {
        name: np.mean([run_episode(env, agent, seed=s)["round_reached"] for s in range(10)])
        for name, agent in (("random", RandomAgent(0)), ("scripted", ScriptedAgent()))
    }
    assert rounds["scripted"] >= rounds["random"] + 1.0
