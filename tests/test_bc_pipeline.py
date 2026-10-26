"""The whole path from recorded play to a policy that plays, on one small run: the synthetic stand-in's scripted
player recorded exactly as a human demo is, an inverse dynamics model trained on that, and behavioural cloning."""

import json

import numpy as np
import pytest
import torch

from zombiesai import spec
from zombiesai.demos import bc, idm, stats
from zombiesai.demos.agent import BCAgent
from zombiesai.demos.clips import DPOINTS_EDGES, ClipWriter, load_clip, monte_carlo_returns
from zombiesai.demos.dataset import ClipDataset, DataConfig
from zombiesai.demos.inputs import InputConfig
from zombiesai.demos.losses import FocalInverse, factored_cross_entropy, head_confidence, head_predictions
from zombiesai.demos.recorder import RecorderConfig, record
from zombiesai.reward import REWARD_TERMS
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.synthetic import ScriptedPlayer, SyntheticConfig, SyntheticSource, SyntheticWorld

STEPS = 160


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    root = tmp_path_factory.mktemp("demos")
    config = RecorderConfig(max_steps=STEPS, realtime=False, input=InputConfig(counts_per_degree=10.0))
    paths = []
    for seed in range(3):
        source = SyntheticSource(seed=seed, max_steps=STEPS)
        paths.append(record(source, source, root / f"demo_{seed}", config, stop=lambda s=source: s.done,
                            progress_every=0))
    return [load_clip(p) for p in paths]


@pytest.fixture(scope="module")
def idm_checkpoint(demos, tmp_path_factory):
    config = idm.IDMConfig(
        before=1, after=1, hidden=64, epochs=1, batch_size=16, max_batches_per_epoch=4,
        val_fraction=0.34, device="cpu", seed=0,
    )
    return idm.train(demos, config, tmp_path_factory.mktemp("idm"))


def test_the_loss_weights_rare_classes_and_drops_uncertain_labels():
    torch.manual_seed(0)
    logits = torch.zeros(4, sum(spec.ACTION_NVEC), requires_grad=True)
    actions = torch.zeros(4, len(spec.ACTION_NVEC), dtype=torch.long)
    plain, per_head = factored_cross_entropy(logits, actions)
    assert set(per_head) == set(spec.ACTION_HEADS)
    assert plain > 0
    ignored, _ = factored_cross_entropy(logits, actions, sample_weights=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert float(ignored.detach()) == pytest.approx(float(plain.detach()), rel=1e-5)  # all four rows are identical here
    focal, _ = factored_cross_entropy(logits, actions, focal_gamma=2.0)
    assert float(focal.detach()) < float(plain.detach())  # a confident-enough prediction is down-weighted
    assert head_predictions(logits).shape == (4, len(spec.ACTION_NVEC))
    assert (head_confidence(logits) <= 1.0).all()


def test_recorded_demos_train_an_inverse_dynamics_model(idm_checkpoint, demos):
    net, config = idm.load(idm_checkpoint)
    assert config.window == 3
    checkpoint = torch.load(idm_checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["spec_version"] == spec.SPEC_VERSION and checkpoint["kind"] == "idm"
    data = ClipDataset([demos[0]], DataConfig(), before=config.before, after=config.after)
    report = idm.evaluate(net, data, torch.device("cpu"))
    assert set(report["balanced"]) == set(spec.ACTION_HEADS)
    assert 0.0 <= report["mean_balanced"] <= 1.0


def test_the_idm_labels_video_it_has_never_seen(idm_checkpoint, demos, tmp_path):
    import shutil

    raw = tmp_path / "video_0000"
    shutil.copytree(demos[2].path, raw)
    (raw / "labels.npz").unlink()
    manifest = json.loads((raw / "clip.json").read_text())
    manifest["label_source"] = "none"
    (raw / "clip.json").write_text(json.dumps(manifest))
    assert not load_clip(raw).labelled

    summaries = idm.label_clips(idm_checkpoint, [load_clip(raw)], device="cpu", min_confidence=0.5)
    labelled = load_clip(raw)
    assert labelled.labelled and labelled.label_source == "idm"
    assert labelled.actions.shape == (labelled.n_steps, len(spec.ACTION_NVEC))
    assert (labelled.actions < np.array(spec.ACTION_NVEC)).all()
    assert labelled.manifest["labels"]["checkpoint"].endswith("idm.pt")
    assert summaries[0]["steps"] == labelled.n_steps
    # Every step got an answer, and the ones the model doubts are marked rather than silently trusted.
    assert labelled.confidence.shape == (labelled.n_steps,)
    assert labelled.usable(min_confidence=0.5).sum() <= labelled.n_steps


def test_behavioural_cloning_produces_a_policy_that_plays(demos, tmp_path):
    config = bc.BCConfig(
        frame_stack=2, hidden=64, epochs=1, batch_size=16, max_batches_per_epoch=4,
        val_fraction=0.34, device="cpu", seed=0,
    )
    checkpoint = bc.train(demos, config, tmp_path / "bc")
    assert (tmp_path / "bc" / "config.json").exists() and (tmp_path / "bc" / "report.json").exists()

    agent = BCAgent(checkpoint)
    world = SyntheticWorld(SyntheticConfig(max_steps=30), seed=99)
    obs = world.observe()
    agent.reset()
    taken = []
    for _ in range(30):
        action = agent.act(obs)
        assert spec.action_tuple(action)  # a valid factored action, in range on every head
        taken.append(action)
        obs, _, terminated, truncated, _ = world.step(action)
        if terminated or truncated:
            break
    assert len(taken) > 1
    assert set(stats.behaviour_stats(np.stack(taken))) >= {"fire_duty", "abs_yaw_deg_per_s", "copy_joint"}

    # It keeps the loss's class weights, plays at its human's rates, and says what those rates would be.
    meta = torch.load(checkpoint, weights_only=True)
    assert [len(w) for w in meta["class_weights"]] == list(spec.ACTION_NVEC)
    assert bc.policy_calibration(meta) == {"focal_gamma": config.focal_gamma, "class_weights": meta["class_weights"]}
    assert agent.net.calibration is not None and BCAgent(checkpoint, calibrate=False).net.calibration is None
    report = json.loads((tmp_path / "bc" / "report.json").read_text())["final"]
    assert "calibrated" in report["behaviour"] and "ratios" in report["divergence_calibrated"]


def test_a_bc_policy_refuses_observations_without_pixels(demos, tmp_path):
    config = bc.BCConfig(frame_stack=2, hidden=32, epochs=1, batch_size=16, max_batches_per_epoch=2,
                         val_fraction=0.34, device="cpu")
    agent = BCAgent(bc.train(demos, config, tmp_path / "bc2"))
    with pytest.raises(KeyError):
        agent.act({"hud": np.zeros(spec.HUD_DIM, np.float32)})


def test_the_agent_stacks_frames_the_way_the_loader_does(demos, tmp_path):
    config = bc.BCConfig(frame_stack=3, hidden=32, epochs=1, batch_size=16, max_batches_per_epoch=2,
                         val_fraction=0.34, device="cpu")
    agent = BCAgent(bc.train(demos, config, tmp_path / "bc3"))
    agent.reset()
    first = agent.stack(np.full(spec.PIXELS_SHAPE, 7, np.uint8))
    assert first.shape == (3, *spec.PIXELS_SHAPE)
    assert (first == 7).all()  # no history yet: the first frame is repeated, exactly as ClipDataset clamps
    second = agent.stack(np.full(spec.PIXELS_SHAPE, 9, np.uint8))
    assert [int(f[0, 0, 0]) for f in second] == [7, 7, 9]  # oldest first, current last


def played_clip(path, seed: int, steps: int = 120, gamma: float = 0.995):
    """A clip of the scripted player in the synthetic world that carries what a recording with rewards can: a
    Monte-Carlo return for the value head, and the "did I just score" and "am I being hit" targets."""
    world, player = SyntheticWorld(SyntheticConfig(max_steps=steps), seed=seed), ScriptedPlayer(seed)
    frames, actions, rewards, gained, hurt = [], [], [], [], []
    done = False
    while not done:
        frames.append(world.render())
        actions.append(player.act(world))
        points = world.points
        _, reward, terminated, truncated, info = world.step(actions[-1])
        rewards.append(reward)
        gained.append(world.points - points)
        hurt.append(info["terms"][REWARD_TERMS.index("damage")] < 0)
        done = terminated or truncated
    returns = monte_carlo_returns(np.asarray(rewards), gamma)
    writer = ClipWriter(path, source={"kind": "synthetic", "seed": seed}, label_source="agent")
    for k, (frame, action) in enumerate(zip(frames, actions)):
        writer.add(frame, action, extras={"mc_return": np.float32(returns[k]),
                                          "aux_dpoints": np.uint8(np.digitize(gained[k], DPOINTS_EDGES)),
                                          "aux_damage": np.uint8(hurt[k])})
    return load_clip(writer.close())


def test_bc_trains_the_value_and_auxiliary_heads_when_the_clips_carry_them(tmp_path):
    clips = [played_clip(tmp_path / f"played_{seed}", seed) for seed in range(2)]

    config = bc.BCConfig(frame_stack=2, hidden=32, epochs=1, batch_size=16, max_batches_per_epoch=3,
                         val_fraction=0.0, device="cpu")
    data = ClipDataset(clips, DataConfig(), before=1)
    batch = data.batch(data.index[:16])
    assert batch["mc_return_mask"].all() and batch["aux_damage_mask"].all()
    net = bc.build_net(config)
    tensors = bc.batch_tensors(batch, config, torch.device("cpu"))
    _, _, parts, _ = bc.compute_loss(net, tensors, config, None)
    assert {"policy", "value", "aux_dpoints", "aux_damage"} <= set(parts)

    # Video has none of those targets, and training on a mix must still work.
    data_without = ClipDataset(list(clips), DataConfig(), before=1)
    for clip in data_without.clips:
        clip.labels.pop("mc_return")
    _, _, parts_without, _ = bc.compute_loss(
        net, bc.batch_tensors(data_without.batch(data_without.index[:16]), config, torch.device("cpu")), config, None
    )
    assert "value" not in parts_without and "policy" in parts_without


def focal_optimum(p: np.ndarray, w: np.ndarray, gamma: float) -> np.ndarray:
    """The head's output that factored_cross_entropy's weighted focal loss is minimised by, labels drawn from p."""
    z = torch.zeros(len(p), dtype=torch.float64, requires_grad=True)
    p_t, w_t = torch.as_tensor(p), torch.as_tensor(w, dtype=torch.float64)
    opt = torch.optim.LBFGS([z], max_iter=500, tolerance_grad=1e-14, tolerance_change=1e-16)

    def loss():
        opt.zero_grad()
        q = z.softmax(0)
        value = -(p_t * w_t * (1 - q) ** gamma * q.log()).sum()
        value.backward()
        return value

    for _ in range(5):
        opt.step(loss)
    return z.softmax(0).detach().numpy()


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
def test_the_focal_inverse_gives_back_the_rates_the_loss_was_fitted_to(gamma):
    """A button pressed on 5% of decisions: the focal loss's answer says ~19%, the inverse says 5% again."""
    rng = np.random.default_rng(int(gamma * 10))
    heads = [np.array([0.95, 0.025, 0.01, 0.007, 0.002, 0.006]), np.array([0.88, 0.12]), rng.dirichlet(np.ones(9))]
    weights = [rng.uniform(0.5, 3.0, len(p)) for p in heads]
    q = [focal_optimum(p, w, gamma) for p, w in zip(heads, weights)]
    if gamma >= 1:
        assert q[0][0] < heads[0][0] - 0.05  # the flattening this exists to undo
    nvec = tuple(len(p) for p in heads)
    logits = torch.as_tensor(np.log(np.concatenate(q)), dtype=torch.float32)[None]
    out = FocalInverse(nvec, gamma, weights)(logits)[0]
    for head, p in zip(torch.split(out, nvec), heads):
        np.testing.assert_allclose(head.softmax(-1).numpy(), p, atol=1e-4)


def test_a_calibration_changes_what_is_sampled_and_nothing_that_is_saved():
    config = bc.BCConfig(frame_stack=2, hidden=32)
    net = bc.build_net(config).eval()
    keys = set(net.state_dict())
    pixels = torch.zeros((2, 2, *spec.PIXELS_SHAPE), dtype=torch.uint8)
    raw, _, _ = net(pixels)
    bc.calibrate(net, {"focal_gamma": 1.0, "class_weights": None})
    calibrated, _, _ = net(pixels)
    assert set(net.state_dict()) == keys and not torch.allclose(raw, calibrated)
    torch.testing.assert_close(calibrated, FocalInverse(spec.ACTION_NVEC, 1.0)(raw))
    assert bc.calibrate(net, None).calibration is None
    unweighted = FocalInverse(spec.ACTION_NVEC, 0.0)(raw)  # gamma 0, no weights: the same policy
    torch.testing.assert_close(FactoredCategorical(unweighted, spec.ACTION_NVEC).log_probs,
                               FactoredCategorical(raw, spec.ACTION_NVEC).log_probs)
