"""Backend-agnostic reward shaping in points-equivalent units, with every term reported separately."""

from dataclasses import dataclass, field

import numpy as np

REWARD_TERMS = ("gain", "repair", "round", "door", "wall_weapon", "ammo", "damage", "death")


@dataclass(frozen=True)
class RewardConfig:
    points_scale: float = 100.0
    # A safety property, not a tuning knob: a misparse (4500 read as 45000) must never become 45,000 reward.
    gain_cap: float = 400.0
    repair_weight: float = 0.2
    repair_points_cap_per_round: float = 200.0
    round_complete: float = 500.0
    door_opened: float = 300.0
    wall_weapon: float = 200.0
    ammo_rebuy: float = 50.0
    # Rebuys only pay when the reserve was actually low; otherwise spending points on ammo is a free reward loop.
    ammo_need_fraction: float = 0.5
    damage_event: float = -100.0
    death: float = -1500.0
    clip_low: float = -20.0
    clip_high: float = 8.0


@dataclass(frozen=True, slots=True)
class StepSignals:
    """What a backend observed during one decision step, in HUD-level terms both sim and real can produce."""

    delta_points: float = 0.0
    repair_points: float = 0.0
    rounds_completed: int = 0
    doors_opened: tuple[str, ...] = ()
    wall_weapons_bought: tuple[str, ...] = ()
    # reserve_before / reserve_after for each ammo rebuy
    ammo_rebuy_reserve_fractions: tuple[float, ...] = ()
    damage_event: bool = False
    death: bool = False


@dataclass(frozen=True)
class RewardResult:
    reward: float
    terms: np.ndarray
    gain_clipped: bool
    reward_clipped: bool


@dataclass
class EpisodeRewardStats:
    term_sums: np.ndarray = field(default_factory=lambda: np.zeros(len(REWARD_TERMS)))
    gain_clips: int = 0
    reward_clips: int = 0
    points_gained: float = 0.0
    repair_points: float = 0.0

    def repair_share(self) -> float:
        """Board-farming alarm input: fraction of all points gained that came from barrier repairs."""
        return self.repair_points / self.points_gained if self.points_gained > 0 else 0.0

    def max_term_share(self) -> float:
        """Anti-hacking gate input: the largest single term's share of total absolute return."""
        total = np.abs(self.term_sums).sum()
        return float(np.abs(self.term_sums).max() / total) if total > 0 else 0.0


class RewardShaper:
    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.reset()

    def reset(self) -> None:
        self._doors_seen: set[str] = set()
        self._weapons_seen: set[str] = set()
        self._repair_points_this_round = 0.0
        self.stats = EpisodeRewardStats()

    def __call__(self, s: StepSignals) -> RewardResult:
        c = self.config
        repair_points = max(0.0, s.repair_points)
        gain_raw = max(0.0, s.delta_points - repair_points)
        gain = min(gain_raw, c.gain_cap)

        repair = min(repair_points, max(0.0, c.repair_points_cap_per_round - self._repair_points_this_round))
        self._repair_points_this_round += repair_points
        if s.rounds_completed:
            self._repair_points_this_round = 0.0

        n_doors = n_weapons = n_rebuys = 0
        if s.doors_opened:
            novel = set(s.doors_opened) - self._doors_seen
            self._doors_seen |= novel
            n_doors = len(novel)
        if s.wall_weapons_bought:
            novel = set(s.wall_weapons_bought) - self._weapons_seen
            self._weapons_seen |= novel
            n_weapons = len(novel)
        if s.ammo_rebuy_reserve_fractions:
            n_rebuys = sum(1 for f in s.ammo_rebuy_reserve_fractions if f < c.ammo_need_fraction)

        inv = 1.0 / c.points_scale
        values = (
            gain * inv,
            repair * c.repair_weight * inv,
            s.rounds_completed * c.round_complete * inv,
            n_doors * c.door_opened * inv,
            n_weapons * c.wall_weapon * inv,
            n_rebuys * c.ammo_rebuy * inv,
            c.damage_event * inv if s.damage_event else 0.0,
            c.death * inv if s.death else 0.0,
        )
        total = sum(values)
        reward = min(max(total, c.clip_low), c.clip_high)
        terms = np.array(values)

        st = self.stats
        st.term_sums += terms
        st.gain_clips += gain_raw > c.gain_cap
        st.reward_clips += reward != total
        st.points_gained += gain_raw + repair_points
        st.repair_points += repair_points
        return RewardResult(reward, terms, gain_raw > c.gain_cap, reward != total)
