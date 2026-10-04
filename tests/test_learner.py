"""The learner's fast path -- everything staged on the device once, values and the prior's logits in bulk, GAE and
reward scaling vectorised -- against the plain statement of the same PPO update, row by row, on the same data.

PPO math must not move when the bookkeeping does, so the reference here is the update as it was written before
any of that (per-row stacking, per-segment GAE), in fp32 on the CPU -- running the network at the fast path's batch
shapes, for the reason reference_update gives."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from zombiesai import spec
from zombiesai.demos import bc
from zombiesai.demos.agent import BCAgent
from zombiesai.demos.hearing import AudioFeatureConfig
from zombiesai.demos.hud_crops import HUD_VIEW_SHAPE
from zombiesai.rl.config import RLConfig
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.rl.encoders import precision
from zombiesai.rl.learner import (
    FRESH_OFFSETS,
    Learner,
    RunningStd,
    compute_gae,
    kl_to_reference,
    prepare_init,
    segment_gae,
)
from zombiesai.rl.segments import Segment

AUDIO = AudioFeatureConfig().shape


def loop_gae(rewards, values, last_value, terminated, gamma, lam):
    """GAE as it was first written: one step at a time, backwards -- in float64, as segment_gae accumulates."""
    n = len(rewards)
    adv = np.zeros(n)
    next_value = 0.0 if terminated else float(last_value)
    gae = 0.0
    for t in reversed(range(n)):
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_value = float(values[t])
    adv = adv.astype(np.float32)
    return adv, adv + np.asarray(values, dtype=np.float32)


class LoopStd:
    """Welford's update, one discounted return at a time, as RunningStd was first written."""

    def __init__(self, gamma):
        self.gamma, self.count, self.mean, self.m2, self.ret = gamma, 1e-4, 0.0, 1.0, {}

    def update(self, actor, rewards, terminated):
        ret = self.ret.get(actor, 0.0)
        for r in rewards:
            ret = ret * self.gamma + float(r)
            self.count += 1
            delta = ret - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (ret - self.mean)
        self.ret[actor] = 0.0 if terminated else ret

    @property
    def std(self):
        return float(np.sqrt(max(self.m2 / self.count, 1e-8)))


def reference_update(learner: Learner, segments: list[Segment], scaler: LoopStd) -> dict:
    """The PPO update written the plain way: stacks built per row on the CPU, GAE per segment, the frozen prior's
    logits indexed per minibatch. Same random permutations as the fast path (np.random).

    The network runs at the fast path's batch shapes -- every observation's value in one batch, the prior over every
    usable step in one -- because a CPU convolution's rounding depends on the batch it is in, and a one-ulp
    difference is not harmless here: it can tip a ReLU sitting on zero the other way, which changes that sample's
    gradient outright, and from there the two trajectories part by far more than any tolerance. Which batch sizes
    round alike is down to the kernel torch picks for the CPU, so a test built on that passes on one runner and
    fails on the next (AVX-512 ones, mostly)."""
    c = learner.config
    for seg in segments:
        scaler.update(seg.actor, seg.rewards, seg.terminated)
    scale = scaler.std
    every_which = np.concatenate([np.full(seg.n + 1, s_idx) for s_idx, seg in enumerate(segments)])
    every_row = np.concatenate([np.arange(seg.n + 1) for seg in segments])
    learner.net.eval()
    with torch.no_grad():
        px, au, mk, hd = learner._gather(segments, every_row, every_which)
        values = np.split(learner.net(px, None, au, mk, hd)[1].numpy(), np.cumsum([s.n + 1 for s in segments])[:-1])
    learner.net.train()
    adv, ret, which, rows, valid = [], [], [], [], []
    for s_idx, (seg, v) in enumerate(zip(segments, values)):
        a, r = loop_gae(seg.rewards / scale, v[:-1], v[-1], seg.terminated, c.gamma, c.gae_lambda)
        adv.append(a)
        ret.append(r)
        which.append(np.full(seg.n, s_idx))
        rows.append(np.arange(seg.n))
        valid.append(~seg.bad)
    adv, ret, which, rows = map(np.concatenate, (adv, ret, which, rows))
    old_logp = np.concatenate([s.logp for s in segments])
    actions = np.concatenate([s.actions for s in segments])
    usable = np.flatnonzero(np.concatenate(valid))
    warmup = learner.updates < c.critic_warmup_updates
    prior = np.zeros((len(rows), sum(spec.ACTION_NVEC)), np.float32)
    if not warmup and len(usable):
        px, au, mk, hd = learner._gather(segments, rows[usable], which[usable])
        with torch.no_grad():
            prior[usable] = learner.ref(px, None, au, mk, hd)[0].numpy()
    losses = []
    for _ in range(c.update_epochs):
        perm = np.random.permutation(usable)
        for start in range(0, len(perm), c.minibatch_size):
            mb = perm[start : start + c.minibatch_size]
            if len(mb) < 2:
                continue
            px, au, mk, hd = learner._gather(segments, rows[mb], which[mb])
            target = torch.from_numpy(ret[mb])
            if warmup:
                with torch.no_grad():
                    h = learner.net.features(px, None, au, mk, hd)
                value_loss = 0.5 * ((learner.net.critic(h).squeeze(-1) - target) ** 2).mean()
                learner.critic_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                learner.critic_opt.step()
                losses.append(float(value_loss.detach()))
                continue
            logits, value, _ = learner.net(px, None, au, mk, hd)
            ref_logits = torch.from_numpy(prior[mb])
            dist = FactoredCategorical(logits, spec.ACTION_NVEC)
            log_ratio = dist.log_prob(torch.from_numpy(actions[mb])) - torch.from_numpy(old_logp[mb])
            ratio = log_ratio.exp()
            a = torch.from_numpy(adv[mb])
            a = (a - a.mean()) / (a.std() + 1e-8)
            policy_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - c.clip_coef, 1 + c.clip_coef)).mean()
            value_loss = 0.5 * ((value - target) ** 2).mean()
            entropy = dist.entropy().mean()
            kl_ref = kl_to_reference(logits, ref_logits, spec.ACTION_NVEC).mean()
            loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy + learner.kl_coef * kl_ref
            learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(learner.net.parameters(), c.max_grad_norm)
            learner.opt.step()
            losses.append(float(policy_loss.detach()))
    learner.updates += 1
    if not warmup:
        learner.kl_coef = max(c.kl_min, learner.kl_coef * c.kl_decay)
    return {"loss_mean": float(np.mean(losses)), "ret": ret, "adv": adv, "scale": scale}


def make_segments(net, offsets, audio: bool, lengths=(17, 40, 1, 33), seed=0, hud=False) -> list[Segment]:
    """Segments of mixed length (a one-step one included) with log-probs from `net`, as a lag-0 actor's."""
    rng = np.random.default_rng(seed)
    depth = max(offsets)
    out = []
    for k, n in enumerate(lengths):
        frames = rng.integers(0, 255, (depth + n + 1, *spec.PIXELS_SHAPE), dtype=np.uint8)
        actions = np.stack([rng.integers(0, m, n) for m in spec.ACTION_NVEC], axis=1)
        seg = Segment(actor=k % 2, version=0, context=depth, frames=frames, actions=actions,
                      logp=np.zeros(n, np.float32), rewards=rng.normal(0, 1, n).astype(np.float32),
                      bad=rng.random(n) < 0.15, terminated=k == 1,
                      audio=rng.normal(0, 1, (n + 1, *AUDIO)).astype(np.float32) if audio else None,
                      audio_mask=(rng.random(n + 1) < 0.7).astype(np.float32) if audio else None,
                      hud_view=rng.integers(0, 255, (n + 1, *HUD_VIEW_SHAPE), dtype=np.uint8) if hud else None)
        idx = depth + np.arange(n)[:, None] - np.asarray(offsets)[None, :]
        with torch.no_grad():
            au = torch.from_numpy(seg.audio[:n]) if audio else None
            mk = torch.from_numpy(seg.audio_mask[:n]) if audio else None
            hd = torch.from_numpy(seg.hud_view[:n]) if hud else None
            logits = net(torch.from_numpy(frames[idx]), None, au, mk, hd)[0]
            seg.logp = FactoredCategorical(logits, spec.ACTION_NVEC).log_prob(torch.from_numpy(actions)).numpy()
        out.append(seg)
    return out


def checkpoint(tmp_path, offsets=(0, 1, 3), audio=False, hud=False) -> str:
    config = bc.BCConfig(frame_offsets=offsets, hidden=32, audio_dim=16, use_audio=audio, use_hud_view=hud, hud_dim=16)
    torch.manual_seed(0)
    path = tmp_path / "bc.pt"
    bc.save(path, bc.build_net(config, AudioFeatureConfig() if audio else None), config, 0, {}, {},
            AudioFeatureConfig() if audio else None)
    return str(path)


@pytest.mark.parametrize("audio,hud", [(False, False), (True, False), (True, True), (False, True)])
def test_the_fast_update_is_the_plain_update(tmp_path, audio, hud):
    """Warm-up then two PPO updates, from the same weights on the same segments: the same parameters after."""
    config = RLConfig(init=checkpoint(tmp_path, (0, 1, 2, 4), audio, hud), critic_warmup_updates=1,
                      minibatch_size=16, update_epochs=2, device="cpu", target_kl=None, amp="auto")
    fast, plain = Learner(config, torch.device("cpu")), Learner(config, torch.device("cpu"))
    assert fast.amp == "off"  # the CPU always trains in fp32
    scaler = LoopStd(config.gamma)
    for seed in range(3):
        segments = make_segments(fast.net, fast.offsets, audio, seed=seed, hud=hud)
        np.random.seed(seed)
        stats = fast.update(segments)
        np.random.seed(seed)
        reference = reference_update(plain, segments, scaler)
        assert stats["reward_scale"] == pytest.approx(reference["scale"], rel=1e-9)
        assert stats["warmup"] == (seed == 0) and fast.kl_coef == pytest.approx(plain.kl_coef)
        for (name, p), q in zip(fast.net.named_parameters(), plain.net.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6, msg=f"{name} after update {seed}")
    assert stats["policy_loss"] == pytest.approx(reference["loss_mean"], rel=1e-3, abs=1e-6)


def test_staged_minibatches_are_the_stacks_the_actor_acted_on(tmp_path):
    learner = Learner(RLConfig(init=checkpoint(tmp_path, (0, 2, 5), audio=True, hud=True), device="cpu"),
                      torch.device("cpu"))
    segments = make_segments(learner.net, learner.offsets, audio=True, hud=True)
    staged = learner._stage(segments)
    which = np.concatenate([np.full(s.n + 1, i) for i, s in enumerate(segments)])
    rows = np.concatenate([np.arange(s.n + 1) for s in segments])
    got = learner._minibatch(staged, torch.arange(len(rows)))
    expected = learner._gather(segments, rows, which)
    assert len(got) == len(expected) == 4
    for g, want in zip(got, expected):
        assert (g is None and want is None) or torch.equal(g, want)
    # A decision's observation is its own step, never the segment's final (bootstrap) one.
    finals = np.cumsum([s.n + 1 for s in segments]) - 1
    assert not np.isin(staged.row_obs.numpy(), finals).any() and len(staged.row_obs) == sum(s.n for s in segments)


def test_vectorised_gae_and_return_scale_match_the_step_by_step_versions():
    rng = np.random.default_rng(3)
    lengths = [5, 1, 0, 30, 12]
    rewards = [rng.normal(0, 2, n).astype(np.float32) for n in lengths]
    values = [rng.normal(0, 1, n).astype(np.float32) for n in lengths]
    last, terminated = rng.normal(0, 1, len(lengths)), [False, True, False, True, False]
    adv, ret = segment_gae(rewards, values, last, terminated, 0.99, 0.9)
    want = [loop_gae(r, v, lv, t, 0.99, 0.9) for r, v, lv, t in zip(rewards, values, last, terminated)]
    np.testing.assert_allclose(adv, np.concatenate([w[0] for w in want]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, np.concatenate([w[1] for w in want]), rtol=1e-5, atol=1e-5)
    one = compute_gae(rewards[3], values[3], last[3], terminated[3], 0.99, 0.9)
    np.testing.assert_allclose(one[0], want[3][0], rtol=1e-5, atol=1e-5)

    fast, slow = RunningStd(0.995), LoopStd(0.995)
    for k, r in enumerate(rewards * 3):
        fast.update(k % 2, r, terminated=k % 4 == 3)
        slow.update(k % 2, r, terminated=k % 4 == 3)
        assert fast.std == pytest.approx(slow.std, rel=1e-9) and fast.ret == pytest.approx(slow.ret)


def test_without_an_anchor_there_is_no_reference_to_run(tmp_path):
    config = RLConfig(init=checkpoint(tmp_path), kl_coef=0.0, critic_warmup_updates=0, minibatch_size=16,
                      device="cpu")
    learner = Learner(config, torch.device("cpu"))
    assert learner.ref is None
    stats = learner.update(make_segments(learner.net, learner.offsets, audio=False))
    assert "kl_ref" not in stats and learner.kl_coef == 0.0 and np.isfinite(stats["policy_loss"])


def test_a_fresh_start_hears_trains_and_plays(tmp_path):
    """No BC at all: a new pixel+audio policy, saved as BC checkpoints are, trained from the first update."""
    run = tmp_path / "run"
    assert prepare_init(RLConfig(init="runs/bc/bc.pt"), run) == RLConfig(init="runs/bc/bc.pt")
    config = prepare_init(RLConfig(init="fresh", minibatch_size=32, update_epochs=1, device="cpu"), run)
    assert config.init == str(run / "init.pt") and config.kl_coef == 0.0 and config.critic_warmup_updates == 0
    net, bc_config, meta = bc.load(config.init)
    assert bc_config.use_audio and bc_config.frame_offsets == FRESH_OFFSETS and net.audio_encoder is not None
    assert bc_config.use_hud_view and net.hud_encoder is not None  # and it looks at the HUD corner
    learner = Learner(config, torch.device("cpu"))
    assert learner.ref is None and learner.audio_shape == AUDIO and learner.uses_hud_view
    for seed in range(2):
        stats = learner.update(make_segments(learner.net, learner.offsets, audio=True, lengths=(40, 25), seed=seed,
                                             hud=True))
        assert not stats["warmup"] and np.isfinite(stats["policy_loss"])
    learner.checkpoint(run / "checkpoint.pt", step=130)
    net, bc_config, meta = bc.load(run / "checkpoint.pt")
    assert meta["rl"]["kl_coef"] == 0.0 and meta["rl"]["updates"] == 2
    agent = BCAgent(run / "checkpoint.pt")
    frame = np.zeros(spec.PIXELS_SHAPE, np.uint8)
    view = np.zeros(HUD_VIEW_SHAPE, np.uint8)
    assert agent.act({"pixels": frame, "audio": np.zeros(AUDIO, np.float32), "audio_mask": 1.0,
                      "hud_view": view}).shape == (8,)
    assert agent.act({"pixels": frame}).shape == (8,)  # and deaf and HUD-blind, if the streams die
    again = Learner(replace(config, init=str(run / "checkpoint.pt")), torch.device("cpu"))
    assert again.ref is None and again.updates == 2  # continuing it stays unanchored


def test_precision_follows_the_device():
    assert precision("auto", "cpu").mode == "off" and precision("fp16", "cpu").mode == "off"
    with pytest.raises(ValueError):
        precision("half", "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_mixed_precision_updates_stay_finite_and_close_to_fp32(tmp_path, amp):
    config = RLConfig(init=checkpoint(tmp_path, audio=True), critic_warmup_updates=1, minibatch_size=32,
                      update_epochs=1, device="cuda", target_kl=None, amp=amp)
    cuda = torch.device("cuda")
    mixed, full = Learner(config, cuda), Learner(replace(config, amp="off"), cuda)
    assert mixed.amp == amp and full.amp == "off"
    segments = make_segments(bc.load(config.init)[0], mixed.offsets, audio=True, lengths=(60, 50))
    for _ in range(2):
        np.random.seed(0)
        a = mixed.update(segments)
        np.random.seed(0)
        b = full.update(segments)
    assert all(np.isfinite(a[k]) for k in ("policy_loss", "value_loss", "entropy", "kl_ref"))
    assert a["entropy"] == pytest.approx(b["entropy"], rel=0.02)
    assert a["value_loss"] == pytest.approx(b["value_loss"], rel=0.1)
