import numpy as np

from zombiesai import spec
from zombiesai.hud.track import Tracked
from zombiesai.realgame.hud_reward import HudSignals
from zombiesai.reward import REWARD_TERMS, RewardShaper

IDLE = spec.make_action()
USE = spec.make_action(button="use")
FIRE = spec.make_action(fire=1)


def tracked(event="", delta=0, round_=1) -> Tracked:
    return Tracked(points=500, points_event=event, points_delta=delta, round=round_)


def test_settled_gains_pay_and_implausible_changes_do_not():
    s = HudSignals()
    assert s.step(tracked("gain", 60), FIRE).delta_points == 60
    assert s.step(tracked("implausible", 4000), FIRE).delta_points == 0
    assert s.events["gain"] == 1 and s.events["implausible"] == 1 and s.points_gained == 60


def test_a_gain_of_a_kills_size_counts_a_kill_and_a_hit_or_a_repair_does_not():
    s = HudSignals()
    for points in (10, 60, 100, 130, 20):
        s.step(tracked("gain", points), FIRE)
    assert s.events["kill"] == 3 and s.events["gain"] == 5
    s = HudSignals()
    s.step(tracked(), USE)
    s.step(tracked("gain", 50), IDLE)  # rebuilding, trigger untouched: a repair, however it adds up
    assert s.events["repair"] == 1 and s.events["kill"] == 0


def test_a_small_gain_after_pressing_use_without_firing_is_a_repair():
    s = HudSignals()
    s.step(tracked(), USE)
    for _ in range(4):
        s.step(tracked(), IDLE)
    out = s.step(tracked("gain", 10), IDLE)
    assert out.delta_points == 10 and out.repair_points == 10

    s.reset()
    s.step(tracked(), USE)
    s.step(tracked(), FIRE)  # shooting in the window: the +10 is a hit
    assert s.step(tracked("gain", 10), IDLE).repair_points == 0

    s.reset()
    s.step(tracked(), USE)
    assert s.step(tracked("gain", 100), IDLE).repair_points == 0  # a kill is never a plank


def test_use_long_ago_does_not_make_a_repair():
    s = HudSignals()
    s.step(tracked(), USE)
    for _ in range(s.config.repair_window_steps):
        s.step(tracked(), IDLE)
    assert s.step(tracked("gain", 10), IDLE).repair_points == 0


def test_purchases_are_classified_by_price():
    s = HudSignals()
    doors = [s.step(tracked("spend", -1000), USE).doors_opened for _ in range(4)]
    assert doors == [("door0",), ("door1",), ("door2",), ()]  # two doors and the debris, then nothing
    assert s.step(tracked("spend", -600), USE).wall_weapons_bought == ("wall600",)
    out = s.step(tracked("spend", -950), USE)  # the box: a gamble, not progress
    assert out.doors_opened == () and out.wall_weapons_bought == ()


def test_death_is_reported_once_and_rounds_only_count_upwards():
    s = HudSignals()
    assert s.step(tracked(round_=1), IDLE).rounds_completed == 0
    assert s.step(tracked(round_=2), IDLE).rounds_completed == 1
    assert s.step(tracked(round_=2), IDLE).rounds_completed == 0
    assert s.step(tracked("downed", -30, round_=2), IDLE).death
    assert not s.step(tracked("game_over", 900, round_=2), IDLE).death
    assert s.step(tracked(round_=1), IDLE).rounds_completed == 0  # a new game is not a round


def test_through_the_shaper_repairs_are_down_weighted_and_capped():
    s, shaper = HudSignals(), RewardShaper()
    total = np.zeros(len(REWARD_TERMS))
    for _ in range(40):  # a board-farming loop: press use, get a plank, forever
        s.step(tracked(), USE)
        total += shaper(s.step(tracked("gain", 10), IDLE)).terms
    repair = total[REWARD_TERMS.index("repair")]
    assert total[REWARD_TERMS.index("gain")] == 0
    assert np.isclose(repair, shaper.config.repair_points_cap_per_round * shaper.config.repair_weight / 100)
    assert shaper.stats.repair_share() == 1.0  # and the alarm sees it
