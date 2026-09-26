"""A policy with memory: strided frame histories, identical in the loader and in the live agent.

Train/serve skew is the failure this file exists for. A stack the agent builds one way and the loader another
does not crash anything -- it just plays worse than its validation accuracy says, with nothing pointing at
why -- so the tests compare the two frame by frame, including across the resets and segment starts where
they are most likely to disagree.
"""

import numpy as np
import pytest
import torch

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.demos.agent import BCAgent
from zombiesai.demos.clips import FLAG_BAD_STEP, FLAG_CLIP_START, ClipWriter, load_clip
from zombiesai.demos.dataset import ClipDataset, DataConfig

STRIDED = (0, 1, 2, 4, 8)


def frame(value: int) -> np.ndarray:
    return np.full(spec.PIXELS_SHAPE, value % 256, np.uint8)


def values(stack: np.ndarray) -> list[int]:
    return [int(f[0, 0, 0]) for f in stack]


def make_clip(path, n: int, starts=(), bad=()):
    """Frame k is filled with k; FLAG_CLIP_START at `starts` (and step 0), FLAG_BAD_STEP at `bad`."""
    writer = ClipWriter(path, source={"kind": "test"}, label_source="input_log")
    rng = np.random.default_rng(0)
    for k in range(n):
        flags = (FLAG_CLIP_START if k in starts else 0) | (FLAG_BAD_STEP if k in bad else 0)
        writer.add(frame(k), rng.integers(0, spec.ACTION_NVEC), flags=flags)
    writer.close()
    return load_clip(path)


def make_checkpoint(path, **config) -> str:
    config = bc.BCConfig(hidden=32, device="cpu", **config)
    torch.manual_seed(0)
    bc.save(path, bc.build_net(config), config, 0, {}, {})
    return path


def test_frame_offsets_generalise_the_frame_stack():
    assert bc.BCConfig(frame_stack=4).offsets == (3, 2, 1, 0)
    assert bc.BCConfig(frame_stack=4).data_offsets == (-3, -2, -1, 0)
    strided = bc.BCConfig(frame_offsets=[8, 0, 2, 1, 4])  # any order, and a list as JSON hands it back
    assert strided.frame_offsets == (0, 1, 2, 4, 8)
    assert strided.offsets == (8, 4, 2, 1, 0)  # oldest first, the order frames are stacked in
    assert strided.frame_stack == 5 and bc.build_net(strided).frame_stack == 5
    for bad in [(1, 2), (0, 0, 1), (0, -1)]:
        with pytest.raises(ValueError):
            bc.BCConfig(frame_offsets=bad)


def test_strided_offsets_clamp_at_the_clip_start_and_at_every_segment_start(tmp_path):
    clip = make_clip(tmp_path / "clip", 40, starts={20})
    data = ClipDataset([clip], offsets=(-8, -4, -2, -1, 0))
    assert data.window == 5 and data.before == 8 and data.after == 0
    assert len(data) == 40  # a history never costs a step: early steps clamp instead of being dropped
    stacks = data.frames(np.array([[0, 0], [0, 3], [0, 12], [0, 19], [0, 20], [0, 23], [0, 30]]))
    assert values(stacks[0]) == [0, 0, 0, 0, 0]
    assert values(stacks[1]) == [0, 0, 1, 2, 3]
    assert values(stacks[2]) == [4, 8, 10, 11, 12]
    assert values(stacks[3]) == [11, 15, 17, 18, 19]  # the segment ends here; it does not see past its end
    assert values(stacks[4]) == [20, 20, 20, 20, 20]  # nothing from before the resume leaks in
    assert values(stacks[5]) == [20, 20, 21, 22, 23]
    assert values(stacks[6]) == [22, 26, 28, 29, 30]


def test_a_contiguous_window_is_unchanged_by_the_offsets_option(tmp_path):
    clip = make_clip(tmp_path / "clip", 12)
    rows = np.array([[0, t] for t in range(12)])
    np.testing.assert_array_equal(
        ClipDataset([clip], before=3).frames(rows), ClipDataset([clip], offsets=(-3, -2, -1, 0)).frames(rows)
    )
    assert len(ClipDataset([clip], offsets=(-2, -1, 0, 1, 2))) == len(ClipDataset([clip], before=2, after=2))


@pytest.mark.parametrize("config", [dict(frame_stack=4), dict(frame_offsets=STRIDED), dict(frame_offsets=(0, 3, 30))])
def test_the_live_agent_builds_exactly_the_loaders_input(tmp_path, config):
    """Feed the agent a clip's frames in order, resetting where the clip starts a segment (which is where the
    live loop resets it), and every stack must equal the loader's for that step."""
    clip = make_clip(tmp_path / "clip", 70, starts={25, 26, 50}, bad={40, 41})
    agent = BCAgent(make_checkpoint(tmp_path / "bc.pt", **config))
    data = ClipDataset([clip], DataConfig(), offsets=agent.config.data_offsets)
    assert {int(t) for _, t in data.index} == set(range(70)) - {40, 41}
    for t in range(clip.n_steps):
        if clip.flags[t] & FLAG_CLIP_START:
            agent.reset()
        live = agent.stack(np.array(clip.frames[t]))
        np.testing.assert_array_equal(live, data.frames(np.array([[0, t]]))[0], err_msg=f"step {t}")


def test_reset_forgets_the_whole_history(tmp_path):
    agent = BCAgent(make_checkpoint(tmp_path / "bc.pt", frame_offsets=STRIDED))
    for k in range(20):
        agent.stack(frame(k))
    assert values(agent.stack(frame(20))) == [12, 16, 18, 19, 20]
    agent.reset()
    assert values(agent.stack(frame(99))) == [99] * 5
    assert values(agent.stack(frame(100))) == [99, 99, 99, 99, 100]


def test_the_agent_keeps_only_as_many_frames_as_its_oldest_offset_needs(tmp_path):
    agent = BCAgent(make_checkpoint(tmp_path / "bc.pt", frame_offsets=(0, 1, 30)))
    for k in range(100):
        stack = agent.stack(frame(k))
    assert values(stack) == [69, 98, 99]
    assert len(agent._frames) == 31


def test_a_strided_policy_acts_and_round_trips_through_its_checkpoint(tmp_path):
    path = make_checkpoint(tmp_path / "bc.pt", frame_offsets=STRIDED)
    net, config, _ = bc.load(path)
    assert config.frame_offsets == STRIDED and config.offsets == (8, 4, 2, 1, 0)
    agent = BCAgent(path)
    for k in range(12):
        action = agent.act({"pixels": frame(k)})
        assert spec.action_tuple(action)
    # The agent's action comes from the loader's input: same stack, same logits.
    data_input = torch.from_numpy(agent.stack(frame(12))[None])
    assert data_input.shape == (1, 5, *spec.PIXELS_SHAPE)
    logits, _, _ = net(data_input)
    assert logits.shape == (1, sum(spec.ACTION_NVEC))


def test_a_checkpoint_from_before_frame_offsets_loads_and_stacks_as_it_did(tmp_path):
    """Checkpoints written before frame_offsets existed have no such key; they mean the last frame_stack
    frames, first frame repeated -- exactly what BCAgent built then."""
    path = make_checkpoint(tmp_path / "bc.pt", frame_stack=4)
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["config"]["frame_offsets"]
    torch.save(checkpoint, path)
    agent = BCAgent(path)
    assert agent.config.frame_offsets is None and agent.config.offsets == (3, 2, 1, 0)
    assert agent.net.frame_stack == 4
    assert values(agent.stack(frame(7))) == [7, 7, 7, 7]
    assert values(agent.stack(frame(9))) == [7, 7, 7, 9]
    for k in range(10, 14):
        stack = agent.stack(frame(k))
    assert values(stack) == [10, 11, 12, 13]


class RecordingAgent(BCAgent):
    """A real BCAgent that remembers every input it built."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = []

    def stack(self, frame):
        out = super().stack(frame)
        self.inputs.append(out)
        return out


def test_what_the_live_loop_recorded_trains_on_exactly_what_the_policy_saw(tmp_path):
    """The whole path: the live loop feeds a strided policy, pauses it (focus lost, a frozen picture, standby
    and back), and records the run; the loader then rebuilds every acted step's input from the recording.
    They must match frame for frame -- this is what the corrections data will be trained on."""
    from test_play import TOGGLE, Clock, Focus, Hands, Screen, TimedDispatcher

    from zombiesai.realgame.play import HumanWatch, PlayConfig, play

    agent = RecordingAgent(make_checkpoint(tmp_path / "bc.pt", frame_offsets=STRIDED))
    clock = Clock()
    config = PlayConfig(max_seconds=60.0)
    kill = {"t": 0, "type": "key", "code": "f9", "down": True}
    hands = Hands({30: [TOGGLE], 33: [TOGGLE], 60: [kill]})  # standby for a few ticks, then back
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="agent")
    play(Screen(stale={20}), agent, TimedDispatcher(clock), focus=Focus(unfocused={9, 10, 11}),
         human=HumanWatch(hands, config), config=config, writer=writer, clock=clock, say=lambda _: None)

    clip = load_clip(tmp_path / "run")
    data = ClipDataset([clip], DataConfig(), offsets=agent.config.data_offsets)
    acted = [int(t) for _, t in data.index]
    assert len(acted) == len(agent.inputs) > 40
    assert (clip.flags & FLAG_CLIP_START).sum() >= 4  # start, after the focus loss, the freeze, and standby
    for t, live in zip(acted, agent.inputs):
        np.testing.assert_array_equal(live, data.frames(np.array([[0, t]]))[0], err_msg=f"step {t}")
