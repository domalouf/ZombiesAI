import re

import pytest
import torch

from zombiesai import spec
from zombiesai.realgame.viewer import AMBER, CHALK_DIM, SANGUINE, choices_marks
from zombiesai.rl.actors import LOOK_BINS, brain_feed, brain_line, watched
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.viz.choices import choices_panel, valid_probs
from zombiesai.viz.live_bus import Subscriber, brain_bus


def distribution():
    """A known distribution: each head sure of one option, the yaw head split 0.6 / 0.4 over 0 and +2."""
    probs = [[0.0] * n for n in spec.ACTION_NVEC]
    for head, row in enumerate(probs):
        row[0] = 1.0
    probs[spec.YAW] = [0.0] * len(spec.YAW_BINS_DEG)
    probs[spec.YAW][spec.YAW_BINS_DEG.index(0.0)] = 0.6
    probs[spec.YAW][spec.YAW_BINS_DEG.index(2.0)] = 0.4
    probs[spec.FIRE] = [0.25, 0.75]
    return probs


def drawing(events, colour):
    """The rectangles (x, y, w, h) of the bars' drawing in `colour` (alpha 00) among `events`: the last one, as a
    fully sure move cell is drawn in the same colour before the bars."""
    found = [e for e in events if f"\\1c{colour}\\1a&H00&}}m " in e]
    assert found
    rects = []
    for m in re.finditer(r"m ([\d.]+) ([\d.]+) l ([\d.]+) [\d.]+ [\d.]+ ([\d.]+)", found[-1]):
        x0, y0, x1, y1 = map(float, m.groups())
        rects.append((x0, y0, x1 - x0, y1 - y0))
    return rects


def test_the_chosen_option_is_lit_and_bars_are_drawn_to_probability():
    probs = distribution()
    action = list(spec.NEUTRAL_ACTION)
    action[spec.YAW] = spec.YAW_BINS_DEG.index(2.0)  # sampled the less likely turn
    action[spec.FIRE] = 1
    events = choices_panel(probs, action, 100, 50, 1.0)
    text = "\n".join(events)
    assert "CHOICES" in text and "TURN" in text and "BUTTON" in text and "reload" in text
    assert "→ 2°  40%" in text  # the turn it took, and how much it wanted it
    lit = drawing(events, AMBER)  # the chosen yaw bar, plus the chosen button and pitch (both at 100%)
    yaw = [r for r in lit if abs(r[3] - 48 * 0.4) < 0.2]
    assert len(yaw) == 1  # 40% of the 48-unit track
    dim = drawing(events, CHALK_DIM)
    assert any(abs(r[3] - 48 * 0.6) < 0.2 for r in dim)  # the turn it did not take, at 60%
    fire = drawing(events, SANGUINE)
    assert len(fire) == 1 and abs(fire[0][2] - 74 * 0.75) < 0.2  # the trigger, held at 75%


def test_it_scales_with_the_frame():
    small = drawing(choices_panel(distribution(), None, 0, 0, 1.0), SANGUINE)
    big = drawing(choices_panel(distribution(), None, 0, 0, 2.0), SANGUINE)
    assert big[0][2] == pytest.approx(2 * small[0][2], abs=0.2)


def test_malformed_or_missing_distributions_draw_nothing():
    assert choices_panel(None, None, 0, 0, 1.0) == []
    assert choices_panel([[1.0]] * 8, None, 0, 0, 1.0) == []
    assert not valid_probs([[0.5, 0.5]])
    # a distribution without an action is drawn with each head's likeliest option picked
    assert choices_panel(distribution(), None, 0, 0, 1.0)


def test_brain_line_keeps_every_heads_whole_distribution():
    logits = torch.randn(1, sum(spec.ACTION_NVEC))
    dist = FactoredCategorical(logits, spec.ACTION_NVEC)
    line = brain_line(dist, torch.tensor([0.5]), list(spec.NEUTRAL_ACTION), 0.0, {})
    assert [len(p) for p in line["p"]] == list(spec.ACTION_NVEC)
    for row in line["p"]:
        assert sum(row) == pytest.approx(1.0, abs=0.01)
    assert line["fire"] == line["p"][spec.FIRE][1]
    assert valid_probs(line["p"])


def test_a_brain_message_goes_from_the_actor_to_the_viewer(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    class Env:
        display = ":60"

    feed = brain_feed(Env())
    assert feed is not None and not watched(feed)  # nobody listening: the actor builds nothing
    assert brain_feed(object()) is None  # the synthetic stand-in has no display
    sub = Subscriber(brain_bus(":60"))
    assert watched(feed)
    dist = FactoredCategorical(torch.zeros(1, sum(spec.ACTION_NVEC)), spec.ACTION_NVEC)
    line = brain_line(dist, torch.tensor([0.1]), list(spec.NEUTRAL_ACTION), 0.0, {"points": 500})
    assert feed.send({"kind": "brain", **line, "t": 0.0, "actor": 0, "version": 7, "episode": 3,
                      "look_bins": LOOK_BINS})
    sub.poll()
    message = sub.get("brain")
    assert message["version"] == 7 and message["pts"] == 500
    drawn = "\n".join(choices_marks(message, 2560, 1440, 440))
    assert "policy v7 · game 3 · actor 0" in drawn and "CHOICES" in drawn
    assert "policy not acting" in "\n".join(choices_marks(None, 2560, 1440, 440))
    sub.close()
    feed.close()


def test_the_panel_sits_inside_the_frame_clear_of_the_inset_and_the_points():
    from zombiesai.demos.hud_crops import HUD_REGIONS, region_box
    from zombiesai.realgame.viewer import AgentInset
    from zombiesai.viz.choices import panel_size

    for width, height in ((2560, 1440), (1280, 720)):
        inset = AgentInset(height, width)
        top = inset.bottom + 40 * height / 1080
        u = height / 720
        cw, ch = panel_size(u)
        x, y = width - AgentInset.MARGIN - cw, top + 26 * u
        assert x >= 0 and y > inset.bottom
        points_top = region_box(height, width, HUD_REGIONS["points_ammo"])[1]
        assert y + ch < points_top - 40 * u  # room for the reward's label above the points box
        drawn = choices_marks({"p": distribution(), "a": None}, width, height, top)
        assert drawn
