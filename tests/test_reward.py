import pytest

from zombiesai.reward import REWARD_TERMS, RewardConfig, RewardShaper, StepSignals

T = {name: i for i, name in enumerate(REWARD_TERMS)}


@pytest.fixture
def shaper():
    return RewardShaper()


def test_nothing_happening_is_worth_nothing(shaper):
    # No per-step survival bonus: "stand in a corner" must not be a local optimum.
    r = shaper(StepSignals())
    assert r.reward == 0.0 and not r.terms.any()


def test_kill_points_flow_through(shaper):
    assert shaper(StepSignals(delta_points=60)).reward == pytest.approx(0.6)


def test_spending_is_reward_neutral(shaper):
    assert shaper(StepSignals(delta_points=-1000)).reward == 0.0


def test_gain_cap_makes_misparses_unconvertible(shaper):
    r = shaper(StepSignals(delta_points=45_000))
    assert r.gain_clipped
    assert r.terms[T["gain"]] == pytest.approx(4.0)
    assert shaper.stats.gain_clips == 1


def test_repairs_are_down_weighted_and_capped_per_round(shaper):
    r = shaper(StepSignals(delta_points=10, repair_points=10))
    assert r.terms[T["gain"]] == 0.0
    assert r.terms[T["repair"]] == pytest.approx(0.02)
    for _ in range(25):
        shaper(StepSignals(delta_points=10, repair_points=10))
    assert shaper(StepSignals(delta_points=10, repair_points=10)).reward == 0.0
    shaper(StepSignals(rounds_completed=1))
    assert shaper(StepSignals(delta_points=10, repair_points=10)).reward == pytest.approx(0.02)
    assert shaper.stats.repair_share() == pytest.approx(1.0)


def test_progression_bonuses_are_novelty_gated(shaper):
    first = shaper(StepSignals(delta_points=-1000, doors_opened=("help_door",)))
    again = shaper(StepSignals(doors_opened=("help_door",)))
    assert first.reward == pytest.approx(3.0) and again.reward == 0.0
    assert shaper(StepSignals(delta_points=-600, wall_weapons_bought=("m1a1_carbine",))).reward == pytest.approx(2.0)
    assert shaper(StepSignals(wall_weapons_bought=("m1a1_carbine",))).reward == 0.0


def test_ammo_rebuys_pay_only_when_needed(shaper):
    assert shaper(StepSignals(ammo_rebuy_reserve_fractions=(0.9,))).reward == 0.0
    assert shaper(StepSignals(ammo_rebuy_reserve_fractions=(0.2,))).reward == pytest.approx(0.5)


def test_round_damage_and_death():
    s = RewardShaper()
    assert s(StepSignals(rounds_completed=1)).reward == pytest.approx(5.0)
    assert s(StepSignals(damage_event=True)).reward == pytest.approx(-1.0)
    assert s(StepSignals(death=True, damage_event=True)).reward == pytest.approx(-16.0)


def test_final_clip():
    s = RewardShaper(RewardConfig(death=-5000.0))
    r = s(StepSignals(death=True))
    assert r.reward == -20.0 and r.reward_clipped
    assert s(StepSignals(delta_points=300, rounds_completed=1)).reward == 8.0


def test_reset_clears_novelty_and_stats(shaper):
    shaper(StepSignals(doors_opened=("help_door",)))
    shaper.reset()
    assert shaper(StepSignals(doors_opened=("help_door",))).reward == pytest.approx(3.0)
    assert shaper.stats.term_sums[T["door"]] == pytest.approx(3.0)
