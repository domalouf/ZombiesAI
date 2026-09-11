"""Guessed (non-ground-truth) sim parameters, each with a relative uncertainty for domain randomization."""

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Weapon:
    name: str
    damage: float
    head_mult: float
    rpm: float  # cyclic rate for automatics; fastest trigger/bolt rate otherwise
    automatic: bool
    mag_size: int
    max_reserve: int
    reload_s: float
    hip_spread_deg: float
    ads_spread_deg: float
    range_m: float
    falloff_mult: float = 0.6  # damage multiplier beyond range_m
    pellets: int = 1
    pellet_spread_deg: float = 0.0


# Approximate WaW values; the weapon files were not consulted, so the sim randomizes damage and spread.
WEAPONS = {
    w.name: w
    for w in (
        Weapon("m1911", 30, 3.0, 400, False, 8, 80, 1.6, 2.2, 0.8, 20),
        Weapon("kar98k", 100, 3.0, 50, False, 5, 50, 2.8, 3.0, 0.3, 60),
        Weapon("m1a1_carbine", 40, 2.5, 400, False, 15, 150, 2.2, 2.2, 0.6, 40),
        Weapon("double_barrel", 45, 1.5, 200, False, 2, 60, 3.0, 2.5, 2.0, 8, 0.2, 8, 4.0),
        Weapon("thompson", 30, 2.5, 720, True, 20, 200, 2.5, 2.8, 1.2, 30),
        Weapon("mp40", 35, 2.5, 550, True, 32, 192, 2.2, 2.8, 1.2, 30),
        Weapon("stg44", 40, 2.5, 600, True, 30, 180, 2.4, 2.8, 1.0, 40),
        Weapon("bar", 60, 2.5, 450, True, 20, 140, 2.5, 3.0, 0.9, 50),
        Weapon("trench_gun", 50, 1.5, 80, False, 6, 60, 3.5, 2.5, 2.0, 8, 0.2, 6, 4.0),
        Weapon("m1_garand", 70, 2.5, 400, False, 8, 128, 2.4, 2.2, 0.4, 60),
        Weapon("gewehr43", 60, 2.5, 400, False, 10, 170, 2.4, 2.2, 0.4, 60),
        Weapon("ptrs41", 400, 2.0, 60, False, 5, 60, 3.2, 4.0, 0.2, 80),
        Weapon("ray_gun", 1000, 1.0, 180, False, 20, 160, 2.8, 1.5, 0.8, 60),
    )
}
STARTING_WEAPON = "m1911"
STARTING_RESERVE = 32
# Box weights are invented; Nacht's real box table and odds are unmodelled.
BOX_WEIGHTS = {
    "mp40": 3,
    "stg44": 3,
    "bar": 3,
    "trench_gun": 3,
    "m1_garand": 3,
    "gewehr43": 3,
    "ptrs41": 2,
    "ray_gun": 1,
    "thompson": 2,
    "double_barrel": 2,
    "kar98k": 2,
    "m1a1_carbine": 2,
}


@dataclass(frozen=True)
class SimParams:
    player_max_hp: float = 100.0
    player_regen_delay_s: float = 2.5
    player_regen_rate: float = 40.0
    player_run_speed: float = 4.6
    player_strafe_speed: float = 4.0
    player_back_speed: float = 3.2
    player_radius_m: float = 0.35
    sprint_mult: float = 1.5
    sprint_duration_s: float = 4.0
    sprint_recovery_s: float = 3.0
    ads_move_mult: float = 0.5
    eye_height_m: float = 1.52

    zombie_damage: float = 50.0
    zombie_attack_range_m: float = 1.0
    zombie_attack_windup_s: float = 0.45
    zombie_attack_cooldown_s: float = 1.3
    zombie_reach_through_window_m: float = 1.4
    zombie_walk_speed: float = 1.2
    zombie_run_speed: float = 3.0
    zombie_sprint_speed: float = 4.2
    # Walk/run/sprint tiers from the later _zombiemode_spawner: randomintrange(speed - 15, speed + 15).
    zombie_speed_jitter: int = 15
    zombie_walk_max: int = 35
    zombie_run_max: int = 70
    zombie_tear_interval_s: float = 1.2
    zombie_climb_s: float = 1.5
    zombie_radius_m: float = 0.3
    zombie_head_z_m: float = 1.62
    zombie_head_r_m: float = 0.12
    zombie_torso_z_m: float = 1.15
    zombie_torso_half_width_m: float = 0.24
    zombie_torso_half_height_m: float = 0.35
    first_spawn_delay_s: float = 1.0

    hit_points: int = 10
    ammo_cost_fraction: float = 0.5
    repair_interval_s: float = 0.9
    repair_radius_m: float = 1.6
    interact_radius_m: float = 1.3
    round_intermission_s: float = 10.0

    melee_damage: float = 150.0
    melee_range_m: float = 1.6
    melee_cone_deg: float = 40.0
    melee_cooldown_s: float = 0.9
    melee_lockout_s: float = 0.5
    grenade_damage: float = 300.0
    grenade_radius_m: float = 5.0
    grenade_self_damage_mult: float = 0.75
    grenade_fuse_s: float = 2.0
    grenade_throw_m: float = 8.0
    grenades_start: int = 2
    grenades_per_round: int = 2
    grenades_max: int = 4
    weapon_swap_s: float = 0.8
    weapon_damage_mult: float = 1.0
    aim_sigma_mult: float = 1.0
    move_spread_mult: float = 1.5

    # The red overlay tracks missing health; there is no health bar, so this is the agent's only health signal.
    flash_gain: float = 1.25
    flash_decay_tau_s: float = 0.8
    flash_detect_jump: float = 0.15
    # Detector and parser failure rates at hardness 1; they scale linearly with sim_hardness.
    flash_noise_std: float = 0.02
    flash_false_positive_hz: float = 0.03
    hud_misread_prob: float = 0.002


# Relative half-width of each parameter's randomization range at hardness 1.
UNCERTAINTY = {
    "player_regen_delay_s": 0.3,
    "player_regen_rate": 0.4,
    "player_run_speed": 0.1,
    "player_strafe_speed": 0.1,
    "player_back_speed": 0.1,
    "zombie_damage": 0.2,
    "zombie_attack_windup_s": 0.3,
    "zombie_attack_cooldown_s": 0.3,
    "zombie_walk_speed": 0.3,
    "zombie_run_speed": 0.2,
    "zombie_sprint_speed": 0.15,
    "zombie_tear_interval_s": 0.4,
    "zombie_climb_s": 0.4,
    "repair_interval_s": 0.3,
    "round_intermission_s": 0.3,
    "melee_damage": 0.2,
    "grenade_damage": 0.3,
    "weapon_damage_mult": 0.2,
    "aim_sigma_mult": 0.4,
    "flash_gain": 0.3,
    "flash_decay_tau_s": 0.5,
}


@dataclass(frozen=True)
class TimingParams:
    latency_steps: int
    frames_per_step: int
    action_dropout: float


def sample_params(rng: np.random.Generator, hardness: float, base: SimParams) -> SimParams:
    changes = {
        name: getattr(base, name) * rng.uniform(1.0 - hardness * u, 1.0 + hardness * u)
        for name, u in UNCERTAINTY.items()
    }
    if hardness > 0:
        changes["hit_points"] = int(rng.choice((5, 10)))
    return dataclasses.replace(base, **changes)


def sample_timing(
    rng: np.random.Generator,
    hardness: float,
    latency_steps: int | None,
    frames_per_step: int | None,
    action_dropout: float | None,
) -> TimingParams:
    if latency_steps is None:
        latency_steps = int(rng.integers(0, 1 + round(3 * hardness)))
    if frames_per_step is None:
        frames_per_step = int(rng.choice((3, 5))) if rng.random() < hardness else 4
    if action_dropout is None:
        action_dropout = 0.05 * hardness
    return TimingParams(latency_steps, frames_per_step, action_dropout)
