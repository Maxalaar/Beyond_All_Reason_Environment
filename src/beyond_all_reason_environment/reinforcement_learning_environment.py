"""Reinforcement learning environment: a fight between alliances of teams, one agent per unit of every team played
by Agent().

Several agent teams make self-play; allies may be scripted teams. Everything an agent receives is seen from its own
alliance: allies, enemies, alliance numbers (0 for its own), visibility and reward.

The game is divided into three dimensions: resource gathering, production, and combat. This environment
covers combat only for now — placed units, no commanders, no factories, no resource collection.

The interface follows PettingZoo's parallel API (reset, step, close, one dictionary entry per agent) without
depending on it, so any reinforcement learning library can wrap it.

Observation of an agent (dictionary of float32 arrays):
    Hierarchical structure: three levels of attention.
    Level 1 — per entity: chassis (static) + state (dynamic) + weapons → entity token.
    Level 2 — per category: entity tokens → category token (5 categories).
    Level 3 — global: all category tokens + context + terrain → action.

    Self (category 0):
        self_chassis           chassis features of the agent's own unit (max health, armor type
                               one-hot, normalized speed/range/sight, can_move)
        self_state             dynamic state features (position, current health, velocity, command, target,
                               allyteam_id_normalized)
        self_weapons           (maximum_weapons_per_unit, n_weapon) — per-weapon features; padded with zeros
        self_weapon_mask       1 for real slots, 0 for padding; shape (maximum_weapons_per_unit,)

    Ally units (category 1):
        ally_units_chassis     (maximum_ally_units, n_chassis)
        ally_units_state       (maximum_ally_units, n_state)
        ally_unit_mask         1 for real rows, 0 for padding
        ally_units_weapons     (maximum_ally_units, maximum_weapons_per_unit, n_weapon)
        ally_units_weapon_mask (maximum_ally_units, maximum_weapons_per_unit)

    Enemy units (category 2) — all enemy alliances combined, nearest first; their current orders are hidden:
        enemy_units_chassis    (maximum_enemy_units, n_chassis)
        enemy_units_state      (maximum_enemy_units, n_state)
        enemy_unit_mask        1 for real rows, 0 for padding
        enemy_units_weapons    (maximum_enemy_units, maximum_weapons_per_unit, n_weapon)
        enemy_units_weapon_mask (maximum_enemy_units, maximum_weapons_per_unit)

    Ally buildings (category 3):
        ally_buildings_chassis     (maximum_ally_buildings, n_chassis)
        ally_buildings_state       (maximum_ally_buildings, n_state)
        ally_building_mask         1 for real rows, 0 for padding
        ally_buildings_weapons     (maximum_ally_buildings, maximum_weapons_per_unit, n_weapon)
        ally_buildings_weapon_mask (maximum_ally_buildings, maximum_weapons_per_unit)

    Enemy buildings (category 4) — all enemy alliances combined, nearest first:
        enemy_buildings_chassis     (maximum_enemy_buildings, n_chassis)
        enemy_buildings_state       (maximum_enemy_buildings, n_state)
        enemy_building_mask         1 for real rows, 0 for padding
        enemy_buildings_weapons     (maximum_enemy_buildings, maximum_weapons_per_unit, n_weapon)
        enemy_buildings_weapon_mask (maximum_enemy_buildings, maximum_weapons_per_unit)

    Level 3 inputs:
        context     global context features sent directly to level 3 (map size, episode time,
                    decision interval ratio); does not depend on which agent is observing
        terrain     (terrain_height, terrain_width, 4) — the full map resampled to a fixed grid;
                    channels: normalised height (raw_elmos/1000), slope squashed to [0,1), water mask,
                    and the visibility of the agent's alliance (1.0 where one of its units sees: within
                    sight range, without terrain occlusion); the first three channels are constant within
                    an episode, the visibility channel is updated every step

    command_mask   1 for commands the unit type may give (ActionConfiguration.commands)

    Centralized critic keys (per alliance, relative to the agent's, for CTDE training only):
        allyteam0_units_chassis    (1 + maximum_ally_units, n_chassis)  — self prepended
        allyteam0_units_state      (1 + maximum_ally_units, n_state)
        allyteam0_unit_mask        (1 + maximum_ally_units,)
        allyteam0_units_weapons    (1 + maximum_ally_units, maximum_weapons_per_unit, n_weapon)
        allyteam0_units_weapon_mask (1 + maximum_ally_units, maximum_weapons_per_unit)
        allyteam0_buildings_chassis (maximum_ally_buildings, n_chassis)
        … (same pattern)
        allyteamN_units_chassis    (maximum_units_per_enemy_allyteam, n_chassis)  — enemy alliance t = 1..N-1
        … (same pattern)

Chassis features (static within an episode for a given unit type on a given map):
    maximum_health_thousands, armor_type one-hot (NUMBER_OF_ARMOR_TYPES),
    speed_x/z, weapon_range_x/z, sight_x/z, can_move.
    No unit type one-hot: a unit is known by what it is (chassis and weapons), so the observation has the same size
    in every scenario, the weights pass from one scenario to another, and an unknown unit is still described.

State features (dynamic per step):
    x, z, relative_x/z, distance, health_fraction, velocity_x/z,
    command one-hot (COMMAND_STATES), has_target, target_x/z, allyteam_id_normalized.

allyteam_id_normalized: relative alliance number / max(n_allyteams - 1, 1).
    0.0 for the agent's own alliance (self and allies), >0 for enemies, up to 1.0 for the last enemy alliance.
    Lets the actor distinguish enemy alliances when there are more than two.

Weapon features per slot: range_x/z, reload_decisions, aoe_fraction, is_paralyzed,
    damage_armor_0 … damage_armor_{NUMBER_OF_ARMOR_TYPES-1} (all in thousands).

Positions are fractions of the map in [0, 1]. All other lengths (relative positions, distances,
speeds, ranges, sight, AoE) are divided by DISTANCE_SCALE (a fixed constant in elmos) so the network
sees a stable physical reference frame regardless of map size. Context features give the actual map
dimensions (map_size_x/z_thousands) so the network can relate map fractions to that reference.

Action of an agent: {"command": index in COMMANDS, "target": [x, z] in [0, 1]^2}.
"attack" snaps to the nearest enemy within attack_snap_radius; otherwise it is ignored.

Timing: an agent decides every decision_interval_frames. BridgeAI runs inside the simulation thread and
blocks on Python; the whole simulation pauses while Python computes actions, giving exact frame timing.
"""

import contextlib
import json
import os
import shutil
import socket
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .controllers import DoNothing
from .game import controller_mod_options, no_commander_mod_options, unit_states
from .game_extension import extension_game_name
from .installation import build_bridge_ai
from .match import RUNS_DIRECTORY, start_engine
from .reinforcement_learning_environment_configuration import COMMANDS, ReinforcementLearningEnvironmentConfiguration
from .scenario import ScenarioConfiguration, spawn_positions
from .start_script import FRAMES_PER_SECOND, MatchConfiguration
from .units import COMMANDERS

CONNECTION_TIMEOUT_SECONDS = 300

COMMAND_STATES = ("none", "move", "fight", "attack", "stop", "other")

# Fixed reference length for normalising all spatial quantities (speeds, ranges, relative distances).
# Using a stable physical constant instead of the variable map size keeps per-unit features consistent
# across maps of different sizes.
DISTANCE_SCALE = 512.0

# Armor types of the pinned game version (installation.GAME_VERSION). The observation space depends on it and is
# built before the engine starts; _start checks it against the count the engine reports.
NUMBER_OF_ARMOR_TYPES = 15

CONTEXT_FEATURES = [
    "map_size_x_thousands",
    "map_size_z_thousands",
    "map_aspect_ratio",
    "episode_time_fraction",
    "decision_interval_ratio",
]


def chassis_feature_names(configuration: ReinforcementLearningEnvironmentConfiguration) -> list[str]:
    """Static features of any entity: health capacity, armor, mobility and scale."""
    return [
        "maximum_health_thousands",
        *(f"armor_type_{i}" for i in range(NUMBER_OF_ARMOR_TYPES)),
        "speed_x",
        "speed_z",
        "weapon_range_x",
        "weapon_range_z",
        "sight_x",
        "sight_z",
        "can_move",
    ]


def state_feature_names() -> list[str]:
    """Dynamic state features of any entity, relative to the observing unit."""
    return [
        "x",
        "z",
        "relative_x",
        "relative_z",
        "distance",
        "health_fraction",
        "velocity_x",
        "velocity_z",
        *(f"command_{name}" for name in COMMAND_STATES),
        "has_target",
        "target_x",
        "target_z",
        "allyteam_id_normalized",
    ]


def weapon_feature_names() -> list[str]:
    """Feature names for a single weapon slot."""
    return [
        "range_x",
        "range_z",
        "reload_decisions",
        "aoe_fraction",
        "is_paralyzed",
        *(f"damage_armor_{i}" for i in range(NUMBER_OF_ARMOR_TYPES)),
    ]


def _chassis_features(
    unit: dict,
    unit_type: dict,
    configuration: ReinforcementLearningEnvironmentConfiguration,
) -> list[float]:
    armor_t = int(unit_type.get("armor_type", 0))
    speed = unit_type.get("speed", 0.0)
    weapon_range = unit_type.get("weapon_range", 0.0)
    sight = unit_type.get("sight", 0.0)
    return [
        unit_type.get("maximum_health", 1.0) / 1000.0,
        *(float(armor_t == i) for i in range(NUMBER_OF_ARMOR_TYPES)),
        speed / DISTANCE_SCALE,
        speed / DISTANCE_SCALE,
        weapon_range / DISTANCE_SCALE,
        weapon_range / DISTANCE_SCALE,
        sight / DISTANCE_SCALE,
        sight / DISTANCE_SCALE,
        float(unit_type.get("can_move", 1)),
    ]


def _state_features(
    unit: dict,
    observer: dict,
    configuration: ReinforcementLearningEnvironmentConfiguration,
    map_size: tuple[float, float],
    allyteam_id_normalized: float = 0.0,
) -> list[float]:
    size_x, size_z = map_size
    rel_x_elmos = (unit["x"] - observer["x"]) * size_x
    rel_z_elmos = (unit["z"] - observer["z"]) * size_z
    interval = configuration.action.decision_interval_frames
    target_x = unit.get("target_x")
    target_z = unit.get("target_z")
    has_target = target_x is not None and target_z is not None
    return [
        unit["x"],
        unit["z"],
        rel_x_elmos / DISTANCE_SCALE,
        rel_z_elmos / DISTANCE_SCALE,
        float(np.hypot(rel_x_elmos, rel_z_elmos) / DISTANCE_SCALE),
        unit["health"] / unit["maximum_health"] if unit["maximum_health"] > 0 else 0.0,
        unit["velocity_x"] * interval / DISTANCE_SCALE,
        unit["velocity_z"] * interval / DISTANCE_SCALE,
        *(float(unit["command"] == name) for name in COMMAND_STATES),
        float(has_target),
        float(target_x) if target_x is not None else 0.0,
        float(target_z) if target_z is not None else 0.0,
        allyteam_id_normalized,
    ]


def _weapon_features(weapon: dict, decision_interval: int) -> list[float]:
    damage_per_type: list[float] = weapon.get("damage_per_type", [])
    return [
        weapon["range"] / DISTANCE_SCALE,
        weapon["range"] / DISTANCE_SCALE,
        weapon["reload"] * FRAMES_PER_SECOND / decision_interval,
        weapon["aoe"] / DISTANCE_SCALE,
        float(weapon.get("paralyzed", 0)),
        *(
            (float(damage_per_type[i]) / 1000.0 if i < len(damage_per_type) else 0.0)
            for i in range(NUMBER_OF_ARMOR_TYPES)
        ),
    ]


def _unit_weapons(
    unit_type: dict, configuration: ReinforcementLearningEnvironmentConfiguration
) -> tuple[np.ndarray, np.ndarray]:
    n_weapon = len(weapon_feature_names())
    max_weapons = configuration.observation.maximum_weapons_per_unit
    matrix = np.zeros((max_weapons, n_weapon), dtype=np.float32)
    mask = np.zeros(max_weapons, dtype=np.float32)
    for slot, weapon in enumerate(unit_type.get("weapons", [])[:max_weapons]):
        matrix[slot] = _weapon_features(weapon, configuration.action.decision_interval_frames)
        mask[slot] = 1.0
    return matrix, mask


def _context_features(
    episode_frame: int,
    frames_since_previous_decision: int,
    configuration: ReinforcementLearningEnvironmentConfiguration,
    map_size: tuple[float, float],
) -> list[float]:
    size_x, size_z = map_size
    return [
        size_x / 1000,
        size_z / 1000,
        size_x / size_z,
        episode_frame / configuration.game.scenario.maximum_game_frames,
        frames_since_previous_decision / configuration.action.decision_interval_frames,
    ]


def relative_allyteams(allyteams: list[int], observer_allyteam: int) -> dict[int, int]:
    """Alliance -> its number seen from ``observer_allyteam``: 0 for its own, 1, 2, ... for the enemy alliances in
    order. Every agent sees itself as alliance 0, so a policy plays any side the same way (self-play)."""
    enemies = sorted(set(allyteams) - {observer_allyteam})
    return {observer_allyteam: 0, **{allyteam: rank + 1 for rank, allyteam in enumerate(enemies)}}


def _hidden_orders(unit: dict) -> dict:
    """An enemy unit as an agent sees it: its current order is not known."""
    return {**unit, "command": "none", "target_x": None, "target_z": None}


def encode_observation(
    raw: dict,
    unit_id: int,
    unit_name: str,
    configuration: ReinforcementLearningEnvironmentConfiguration,
    map_size: tuple[float, float],
    allyteams: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Observation of the agent whose unit is unit_id; zeros (and empty masks) when the unit is dead.

    ``allyteams`` gives the alliance of each team (default: the scenario's). Allies are the units of the agent's
    alliance, enemies those of the other alliances, whose current orders are hidden.
    """
    if allyteams is None:
        allyteams = configuration.game.scenario.allyteams
    n_chassis = len(chassis_feature_names(configuration))
    n_state = len(state_feature_names())
    max_ally = configuration.observation.maximum_ally_units
    max_enemy = configuration.observation.maximum_enemy_units
    max_ally_bld = configuration.observation.maximum_ally_buildings
    max_enemy_bld = configuration.observation.maximum_enemy_buildings
    max_weapons = configuration.observation.maximum_weapons_per_unit
    n_weapon = len(weapon_feature_names())
    n_allyteams = len(set(allyteams))
    n_enemy_allyteams = max(n_allyteams - 1, 1)
    max_units_per_enemy_at = configuration.observation.maximum_units_per_enemy_allyteam
    max_bld_per_enemy_at = configuration.observation.maximum_buildings_per_enemy_allyteam
    max_at0_units = 1 + max_ally  # self prepended before ally units

    observation: dict[str, np.ndarray] = {
        "self_chassis": np.zeros(n_chassis, dtype=np.float32),
        "self_state": np.zeros(n_state, dtype=np.float32),
        "self_weapons": np.zeros((max_weapons, n_weapon), dtype=np.float32),
        "self_weapon_mask": np.zeros(max_weapons, dtype=np.float32),
        "ally_units_chassis": np.zeros((max_ally, n_chassis), dtype=np.float32),
        "ally_units_state": np.zeros((max_ally, n_state), dtype=np.float32),
        "ally_unit_mask": np.zeros(max_ally, dtype=np.float32),
        "ally_units_weapons": np.zeros((max_ally, max_weapons, n_weapon), dtype=np.float32),
        "ally_units_weapon_mask": np.zeros((max_ally, max_weapons), dtype=np.float32),
        "ally_buildings_chassis": np.zeros((max_ally_bld, n_chassis), dtype=np.float32),
        "ally_buildings_state": np.zeros((max_ally_bld, n_state), dtype=np.float32),
        "ally_building_mask": np.zeros(max_ally_bld, dtype=np.float32),
        "ally_buildings_weapons": np.zeros((max_ally_bld, max_weapons, n_weapon), dtype=np.float32),
        "ally_buildings_weapon_mask": np.zeros((max_ally_bld, max_weapons), dtype=np.float32),
        "enemy_units_chassis": np.zeros((max_enemy, n_chassis), dtype=np.float32),
        "enemy_units_state": np.zeros((max_enemy, n_state), dtype=np.float32),
        "enemy_unit_mask": np.zeros(max_enemy, dtype=np.float32),
        "enemy_units_weapons": np.zeros((max_enemy, max_weapons, n_weapon), dtype=np.float32),
        "enemy_units_weapon_mask": np.zeros((max_enemy, max_weapons), dtype=np.float32),
        "enemy_buildings_chassis": np.zeros((max_enemy_bld, n_chassis), dtype=np.float32),
        "enemy_buildings_state": np.zeros((max_enemy_bld, n_state), dtype=np.float32),
        "enemy_building_mask": np.zeros(max_enemy_bld, dtype=np.float32),
        "enemy_buildings_weapons": np.zeros((max_enemy_bld, max_weapons, n_weapon), dtype=np.float32),
        "enemy_buildings_weapon_mask": np.zeros((max_enemy_bld, max_weapons), dtype=np.float32),
        "context": np.zeros(len(CONTEXT_FEATURES), dtype=np.float32),
        "command_mask": np.array(
            [float(command in configuration.action.allowed_commands(unit_name)) for command in COMMANDS],
            dtype=np.float32,
        ),
        # Critic keys — allyteam 0 (the agent's own alliance): self (row 0) prepended before ally units (rows 1..)
        "allyteam0_units_chassis": np.zeros((max_at0_units, n_chassis), dtype=np.float32),
        "allyteam0_units_state": np.zeros((max_at0_units, n_state), dtype=np.float32),
        "allyteam0_unit_mask": np.zeros(max_at0_units, dtype=np.float32),
        "allyteam0_units_weapons": np.zeros((max_at0_units, max_weapons, n_weapon), dtype=np.float32),
        "allyteam0_units_weapon_mask": np.zeros((max_at0_units, max_weapons), dtype=np.float32),
        "allyteam0_buildings_chassis": np.zeros((max_ally_bld, n_chassis), dtype=np.float32),
        "allyteam0_buildings_state": np.zeros((max_ally_bld, n_state), dtype=np.float32),
        "allyteam0_building_mask": np.zeros(max_ally_bld, dtype=np.float32),
        "allyteam0_buildings_weapons": np.zeros((max_ally_bld, max_weapons, n_weapon), dtype=np.float32),
        "allyteam0_buildings_weapon_mask": np.zeros((max_ally_bld, max_weapons), dtype=np.float32),
        **{
            key: np.zeros(shape, dtype=np.float32)
            for t in range(1, n_allyteams)
            for key, shape in [
                (f"allyteam{t}_units_chassis", (max_units_per_enemy_at, n_chassis)),
                (f"allyteam{t}_units_state", (max_units_per_enemy_at, n_state)),
                (f"allyteam{t}_unit_mask", (max_units_per_enemy_at,)),
                (f"allyteam{t}_units_weapons", (max_units_per_enemy_at, max_weapons, n_weapon)),
                (f"allyteam{t}_units_weapon_mask", (max_units_per_enemy_at, max_weapons)),
                (f"allyteam{t}_buildings_chassis", (max_bld_per_enemy_at, n_chassis)),
                (f"allyteam{t}_buildings_state", (max_bld_per_enemy_at, n_state)),
                (f"allyteam{t}_building_mask", (max_bld_per_enemy_at,)),
                (f"allyteam{t}_buildings_weapons", (max_bld_per_enemy_at, max_weapons, n_weapon)),
                (f"allyteam{t}_buildings_weapon_mask", (max_bld_per_enemy_at, max_weapons)),
            ]
        },
    }

    units = {u["id"]: u for u in raw["units"]}
    unit_types_data: dict = raw.get("unit_types") or {}
    observer = units.get(unit_id)
    if observer is None:
        return observation

    # Alliances relative to the observer: 0 for its own, 1.. for the enemy alliances.
    relative = relative_allyteams(allyteams, allyteams[observer["team"]])

    def relative_allyteam(u: dict) -> int:
        return relative[allyteams[u["team"]]]

    def seen(u: dict) -> dict:
        return u if relative_allyteam(u) == 0 else _hidden_orders(u)

    observer_type = unit_types_data.get(unit_name, {})
    observation["self_chassis"][:] = _chassis_features(observer, observer_type, configuration)
    observation["self_state"][:] = _state_features(observer, observer, configuration, map_size, 0.0)
    observation["self_weapons"], observation["self_weapon_mask"] = _unit_weapons(observer_type, configuration)
    observation["context"][:] = _context_features(
        raw["episode_frame"],
        raw.get("frames_since_previous_decision", 0),
        configuration,
        map_size,
    )

    size_x, size_z = map_size

    def dist_sq(u: dict) -> float:
        return ((u["x"] - observer["x"]) * size_x) ** 2 + ((u["z"] - observer["z"]) * size_z) ** 2

    def is_mobile(u: dict) -> bool:
        return bool(unit_types_data.get(u["unit"], {}).get("can_move", 1))

    # ── Actor categories ──────────────────────────────────────────────────────────────────────────
    all_allies = [u for u in raw["units"] if u["id"] != unit_id and relative_allyteam(u) == 0]
    all_enemies = [u for u in raw["units"] if relative_allyteam(u) != 0]
    ally_units = sorted([u for u in all_allies if is_mobile(u)], key=dist_sq)
    ally_buildings = sorted([u for u in all_allies if not is_mobile(u)], key=dist_sq)
    enemy_units = sorted([u for u in all_enemies if is_mobile(u)], key=dist_sq)
    enemy_buildings = sorted([u for u in all_enemies if not is_mobile(u)], key=dist_sq)

    def fill_rows(entities: list[dict], prefix: str, mask_key: str, max_rows: int, row_offset: int = 0) -> None:
        """Rows ``row_offset``.. of the ``prefix``_chassis/_state/_weapons/_weapon_mask arrays and of mask_key."""
        for row, entity in enumerate(entities[:max_rows], start=row_offset):
            entity_type = unit_types_data.get(entity["unit"], {})
            at_norm = relative_allyteam(entity) / n_enemy_allyteams
            observation[f"{prefix}_chassis"][row] = _chassis_features(entity, entity_type, configuration)
            observation[f"{prefix}_state"][row] = _state_features(
                seen(entity), observer, configuration, map_size, at_norm
            )
            observation[mask_key][row] = 1.0
            weapons, weapon_mask = _unit_weapons(entity_type, configuration)
            observation[f"{prefix}_weapons"][row] = weapons
            observation[f"{prefix}_weapon_mask"][row] = weapon_mask

    fill_rows(ally_units, "ally_units", "ally_unit_mask", max_ally)
    fill_rows(ally_buildings, "ally_buildings", "ally_building_mask", max_ally_bld)
    fill_rows(enemy_units, "enemy_units", "enemy_unit_mask", max_enemy)
    fill_rows(enemy_buildings, "enemy_buildings", "enemy_building_mask", max_enemy_bld)

    # ── Critic keys: allyteam 0 (self prepended at row 0, ally units at rows 1..) ─────────────────
    fill_rows([observer], "allyteam0_units", "allyteam0_unit_mask", 1)
    fill_rows(ally_units, "allyteam0_units", "allyteam0_unit_mask", max_ally, row_offset=1)
    fill_rows(ally_buildings, "allyteam0_buildings", "allyteam0_building_mask", max_ally_bld)

    # ── Critic keys: enemy alliances 1..N-1, one group of arrays per alliance ─────────────────────
    for t in range(1, n_allyteams):
        enemy_t_all = [u for u in all_enemies if relative_allyteam(u) == t]
        enemy_t_units = sorted([u for u in enemy_t_all if is_mobile(u)], key=dist_sq)
        enemy_t_blds = sorted([u for u in enemy_t_all if not is_mobile(u)], key=dist_sq)
        fill_rows(enemy_t_units, f"allyteam{t}_units", f"allyteam{t}_unit_mask", max_units_per_enemy_at)
        fill_rows(enemy_t_blds, f"allyteam{t}_buildings", f"allyteam{t}_building_mask", max_bld_per_enemy_at)

    return observation


def side_defeated(units: list[dict], teams: set[int], scenario: ScenarioConfiguration) -> bool:
    """True when the units of ``teams`` lost: none left, or, with scenario.commander_death_ends_game, no commander."""
    side = [u for u in units if u["team"] in teams]
    if scenario.commander_death_ends_game:
        return not any(u["unit"] in COMMANDERS for u in side)
    return not side


def team_reward(
    previous: dict,
    current: dict,
    initial_enemy_health: float,
    initial_ally_health: float,
    initial_enemies: int,
    configuration: ReinforcementLearningEnvironmentConfiguration,
    ally_teams: set[int],
) -> float:
    """Reward of one step for the alliance made of ``ally_teams``, from the unit states before and after it."""
    before = {u["id"]: u for u in previous["units"]}
    after = {u["id"]: u for u in current["units"]}

    def health_lost(allies: bool) -> float:
        return sum(
            max(0.0, unit["health"] - (after[identifier]["health"] if identifier in after else 0.0))
            for identifier, unit in before.items()
            if (unit["team"] in ally_teams) == allies
        )

    weights = configuration.rewards
    scenario = configuration.game.scenario
    enemy_teams = {u["team"] for u in [*before.values(), *after.values()]} - ally_teams
    enemies_before = sum(u["team"] not in ally_teams for u in before.values())
    enemies_after = sum(u["team"] not in ally_teams for u in after.values())
    reward = weights.damage_dealt * health_lost(False) / max(initial_enemy_health, 1e-9)
    reward -= weights.damage_received * health_lost(True) / max(initial_ally_health, 1e-9)
    reward += weights.kill * (enemies_before - enemies_after) / max(initial_enemies, 1)
    enemies_defeated_before = side_defeated(list(before.values()), enemy_teams, scenario)
    enemies_defeated = side_defeated(list(after.values()), enemy_teams, scenario)
    allies_defeated = side_defeated(list(after.values()), ally_teams, scenario)
    # Both sides defeated in the same step is a draw: neither victory nor defeat.
    if enemies_defeated and not enemies_defeated_before and not allies_defeated:
        reward += weights.victory
    elif allies_defeated and not enemies_defeated:
        reward += weights.defeat
    return float(reward)


def merge_commands(raw: dict) -> None:
    """Add "command", "target_x" and "target_z" to each unit of ``raw`` from its "commands" text, sent by the gadget
    as "<unit>,<command>,<target x>,<target z>;..." (an empty target: none)."""
    orders = {}
    for entry in filter(None, raw.pop("commands", "").split(";")):
        identifier, command, *target = entry.split(",")
        x, z = (target + ["", ""])[:2]
        orders[int(identifier)] = (command, float(x) if x else None, float(z) if z else None)
    for unit in raw["units"]:
        command, x, z = orders.get(unit["id"], ("none", None, None))
        unit.update(command=command, target_x=x, target_z=z)


def visibility(
    raw: dict, ally_teams: set[int], map_size: tuple[float, float], resolution: tuple[int, int]
) -> np.ndarray:
    """(height, width) grid: 1 where a unit of ``ally_teams`` sees the cell centre (within its sight range; terrain
    does not block the view), 0 elsewhere."""
    width, height = resolution
    size_x, size_z = map_size
    cell_x = (np.arange(width) + 0.5) / width * size_x
    cell_z = (np.arange(height) + 0.5) / height * size_z
    grid = np.zeros((height, width), dtype=np.float32)
    unit_types = raw.get("unit_types") or {}
    for unit in raw["units"]:
        if unit["team"] in ally_teams:
            sight = unit_types.get(unit["unit"], {}).get("sight", 0.0)
            dx = (cell_x - unit["x"] * size_x) ** 2
            dz = (cell_z - unit["z"] * size_z) ** 2
            grid[dz[:, None] + dx[None, :] <= sight**2] = 1.0
    return grid


class ReinforcementLearningEnvironment:
    """One engine process running a game; episodes reset inside the match without reloading the game.

    The PettingZoo parallel API contract: reset(), step(), close(), observation_space(agent), action_space(agent),
    possible_agents and agents. The engine talks to Python over a TCP connection from the BridgeAI.
    """

    metadata = {"name": "beyond_all_reason"}

    def __init__(self, configuration: ReinforcementLearningEnvironmentConfiguration | None = None):
        self.configuration = configuration or ReinforcementLearningEnvironmentConfiguration()
        self.agents: list[str] = []
        self.run_directory: Path | None = None
        self._process: subprocess.Popen | None = None
        self._server: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._reader = None
        game = self.configuration.game
        scenario = game.scenario
        # One agent per unit of every team the agents play: "team<t>_unit_<k>".
        self.possible_agents: list[str] = []
        self.agent_unit_names: list[str] = []
        self.agent_teams: list[int] = []
        for team_index in game.agent_teams:
            names = [name for name, count in (scenario.teams[team_index].units or {}).items() for _ in range(count)]
            self.possible_agents += [f"team{team_index}_unit_{rank}" for rank in range(len(names))]
            self.agent_unit_names += names
            self.agent_teams += [team_index] * len(names)
        self._allyteams = scenario.allyteams

        n_chassis = len(chassis_feature_names(self.configuration))
        n_state = len(state_feature_names())
        max_ally = self.configuration.observation.maximum_ally_units
        max_enemy = self.configuration.observation.maximum_enemy_units
        max_ally_bld = self.configuration.observation.maximum_ally_buildings
        max_enemy_bld = self.configuration.observation.maximum_enemy_buildings
        max_weapons = self.configuration.observation.maximum_weapons_per_unit
        n_weapon = len(weapon_feature_names())
        terrain_w, terrain_h = self.configuration.observation.terrain_resolution
        n_allyteams = scenario.number_of_allyteams
        max_at0_units = 1 + max_ally
        max_units_per_enemy_at = self.configuration.observation.maximum_units_per_enemy_allyteam
        max_bld_per_enemy_at = self.configuration.observation.maximum_buildings_per_enemy_allyteam

        critic_spaces: dict = {
            "allyteam0_units_chassis": spaces.Box(-np.inf, np.inf, (max_at0_units, n_chassis), np.float32),
            "allyteam0_units_state": spaces.Box(-np.inf, np.inf, (max_at0_units, n_state), np.float32),
            "allyteam0_unit_mask": spaces.Box(0, 1, (max_at0_units,), np.float32),
            "allyteam0_units_weapons": spaces.Box(-np.inf, np.inf, (max_at0_units, max_weapons, n_weapon), np.float32),
            "allyteam0_units_weapon_mask": spaces.Box(0, 1, (max_at0_units, max_weapons), np.float32),
            "allyteam0_buildings_chassis": spaces.Box(-np.inf, np.inf, (max_ally_bld, n_chassis), np.float32),
            "allyteam0_buildings_state": spaces.Box(-np.inf, np.inf, (max_ally_bld, n_state), np.float32),
            "allyteam0_building_mask": spaces.Box(0, 1, (max_ally_bld,), np.float32),
            "allyteam0_buildings_weapons": spaces.Box(
                -np.inf, np.inf, (max_ally_bld, max_weapons, n_weapon), np.float32
            ),
            "allyteam0_buildings_weapon_mask": spaces.Box(0, 1, (max_ally_bld, max_weapons), np.float32),
        }
        for t in range(1, n_allyteams):
            critic_spaces.update(
                {
                    f"allyteam{t}_units_chassis": spaces.Box(
                        -np.inf, np.inf, (max_units_per_enemy_at, n_chassis), np.float32
                    ),
                    f"allyteam{t}_units_state": spaces.Box(
                        -np.inf, np.inf, (max_units_per_enemy_at, n_state), np.float32
                    ),
                    f"allyteam{t}_unit_mask": spaces.Box(0, 1, (max_units_per_enemy_at,), np.float32),
                    f"allyteam{t}_units_weapons": spaces.Box(
                        -np.inf, np.inf, (max_units_per_enemy_at, max_weapons, n_weapon), np.float32
                    ),
                    f"allyteam{t}_units_weapon_mask": spaces.Box(
                        0, 1, (max_units_per_enemy_at, max_weapons), np.float32
                    ),
                    f"allyteam{t}_buildings_chassis": spaces.Box(
                        -np.inf, np.inf, (max_bld_per_enemy_at, n_chassis), np.float32
                    ),
                    f"allyteam{t}_buildings_state": spaces.Box(
                        -np.inf, np.inf, (max_bld_per_enemy_at, n_state), np.float32
                    ),
                    f"allyteam{t}_building_mask": spaces.Box(0, 1, (max_bld_per_enemy_at,), np.float32),
                    f"allyteam{t}_buildings_weapons": spaces.Box(
                        -np.inf, np.inf, (max_bld_per_enemy_at, max_weapons, n_weapon), np.float32
                    ),
                    f"allyteam{t}_buildings_weapon_mask": spaces.Box(
                        0, 1, (max_bld_per_enemy_at, max_weapons), np.float32
                    ),
                }
            )

        self._observation_space = spaces.Dict(
            {
                "self_chassis": spaces.Box(-np.inf, np.inf, (n_chassis,), np.float32),
                "self_state": spaces.Box(-np.inf, np.inf, (n_state,), np.float32),
                "self_weapons": spaces.Box(-np.inf, np.inf, (max_weapons, n_weapon), np.float32),
                "self_weapon_mask": spaces.Box(0, 1, (max_weapons,), np.float32),
                "ally_units_chassis": spaces.Box(-np.inf, np.inf, (max_ally, n_chassis), np.float32),
                "ally_units_state": spaces.Box(-np.inf, np.inf, (max_ally, n_state), np.float32),
                "ally_unit_mask": spaces.Box(0, 1, (max_ally,), np.float32),
                "ally_units_weapons": spaces.Box(-np.inf, np.inf, (max_ally, max_weapons, n_weapon), np.float32),
                "ally_units_weapon_mask": spaces.Box(0, 1, (max_ally, max_weapons), np.float32),
                "ally_buildings_chassis": spaces.Box(-np.inf, np.inf, (max_ally_bld, n_chassis), np.float32),
                "ally_buildings_state": spaces.Box(-np.inf, np.inf, (max_ally_bld, n_state), np.float32),
                "ally_building_mask": spaces.Box(0, 1, (max_ally_bld,), np.float32),
                "ally_buildings_weapons": spaces.Box(
                    -np.inf, np.inf, (max_ally_bld, max_weapons, n_weapon), np.float32
                ),
                "ally_buildings_weapon_mask": spaces.Box(0, 1, (max_ally_bld, max_weapons), np.float32),
                "enemy_units_chassis": spaces.Box(-np.inf, np.inf, (max_enemy, n_chassis), np.float32),
                "enemy_units_state": spaces.Box(-np.inf, np.inf, (max_enemy, n_state), np.float32),
                "enemy_unit_mask": spaces.Box(0, 1, (max_enemy,), np.float32),
                "enemy_units_weapons": spaces.Box(-np.inf, np.inf, (max_enemy, max_weapons, n_weapon), np.float32),
                "enemy_units_weapon_mask": spaces.Box(0, 1, (max_enemy, max_weapons), np.float32),
                "enemy_buildings_chassis": spaces.Box(-np.inf, np.inf, (max_enemy_bld, n_chassis), np.float32),
                "enemy_buildings_state": spaces.Box(-np.inf, np.inf, (max_enemy_bld, n_state), np.float32),
                "enemy_building_mask": spaces.Box(0, 1, (max_enemy_bld,), np.float32),
                "enemy_buildings_weapons": spaces.Box(
                    -np.inf, np.inf, (max_enemy_bld, max_weapons, n_weapon), np.float32
                ),
                "enemy_buildings_weapon_mask": spaces.Box(0, 1, (max_enemy_bld, max_weapons), np.float32),
                "context": spaces.Box(-np.inf, np.inf, (len(CONTEXT_FEATURES),), np.float32),
                "command_mask": spaces.Box(0, 1, (len(COMMANDS),), np.float32),
                "terrain": spaces.Box(-np.inf, np.inf, (terrain_h, terrain_w, 4), np.float32),
                **critic_spaces,
            }
        )
        self._action_space = spaces.Dict(
            {"command": spaces.Discrete(len(COMMANDS)), "target": spaces.Box(0, 1, (2,), np.float32)}
        )

        self._map_size = (1.0, 1.0)
        # Height, slope and water; the fourth channel, visibility, depends on the alliance.
        self._terrain = np.zeros((terrain_h, terrain_w, 3), dtype=np.float32)
        self._random = np.random.default_rng()
        self._episode = -1
        self._agent_unit_ids: dict[str, int] = {}
        self._previous: dict | None = None
        # Alliance -> (initial enemy health, initial ally health, initial number of enemies).
        self._initial: dict[int, tuple[float, float, int]] = {}

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Dict:
        return self._action_space

    def _teams_of(self, allyteam: int) -> set[int]:
        return {team for team, other in enumerate(self._allyteams) if other == allyteam}

    def _allyteam_of(self, agent: str) -> int:
        return self._allyteams[self.agent_teams[self.possible_agents.index(agent)]]

    # --- Engine connection --------------------------------------------------------------------------------------

    def _send(self, message: dict) -> None:
        assert self._connection is not None
        self._connection.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())

    def _receive(self) -> dict:
        assert self._reader is not None
        line = self._reader.readline()
        if not line:
            log = self.run_directory / "engine_output.log" if self.run_directory else "the engine log"
            raise RuntimeError(f"The engine closed the connection, see {log}")
        return json.loads(line)

    def close(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(OSError):
                self._send({"type": "quit"})
        if self._process is not None:
            try:
                self._process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        for resource in (self._reader, self._connection, self._server):
            if resource is not None:
                resource.close()
        self._process = self._connection = self._server = self._reader = None
        self.agents = []
        if self.run_directory is not None and not self.configuration.engine.keep_run_directory:
            shutil.rmtree(self.run_directory, ignore_errors=True)

    def __enter__(self) -> "ReinforcementLearningEnvironment":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def _players(self, port: int) -> list:
        """The BridgeAI on the first agent team (it talks to Python for every agent team); NullAI on the other agent
        teams, whose orders the gadget gives; each other team's own controller."""
        configuration = self.configuration
        game = configuration.game
        players = []
        for index, (team, player) in enumerate(zip(game.scenario.teams, game.players(), strict=True)):
            if index == game.agent_teams[0]:
                player = replace(
                    player,
                    options={
                        "port": str(port),
                        "decision_interval": str(configuration.action.decision_interval_frames),
                        "attack_snap_radius": f"{configuration.action.attack_snap_radius:g}",
                        "terrain_w": str(configuration.observation.terrain_resolution[0]),
                        "terrain_h": str(configuration.observation.terrain_resolution[1]),
                    },
                )
            elif index in game.agent_teams:
                player = replace(DoNothing().player(team.side), label=player.label)
            players.append(player)
        return players

    def _start(self) -> None:
        configuration = self.configuration
        build_bridge_ai()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_directory = (
            RUNS_DIRECTORY / "reinforcement_learning_environments" / f"{timestamp}_{os.getpid()}_{id(self) % 100000}"
        )
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(CONNECTION_TIMEOUT_SECONDS)
        port = self._server.getsockname()[1]
        match_configuration = MatchConfiguration(
            players=self._players(port),
            allyteams=self._allyteams,
            map_name=configuration.game.scenario.map_name,  # pyright: ignore[reportArgumentType]  (checked: one map)
            flat_map_squares=configuration.game.scenario.flat_map_squares,
            game_version=extension_game_name(),
            speed=configuration.engine.effective_speed,
            maximum_game_minutes=1_000_000,
            extra_mod_options={
                "deathmode": "neverend",
                "reinforcement_learning_full_vision": "1" if configuration.game.scenario.full_vision else "0",
                **controller_mod_options(configuration.game),
                **no_commander_mod_options(),
            },
        )
        self._process = start_engine(match_configuration, self.run_directory, graphics=configuration.engine.graphics)
        try:
            self._connection, _ = self._server.accept()
        except TimeoutError as error:
            raise RuntimeError(f"The engine did not connect, see {self.run_directory / 'engine_output.log'}") from error
        self._reader = self._connection.makefile("r", encoding="utf-8")
        ready = self._receive()
        self._map_size = (float(ready["map"]["size_x"]), float(ready["map"]["size_z"]))
        engine_armor_types = int(ready.get("num_armor_types", NUMBER_OF_ARMOR_TYPES))
        if engine_armor_types != NUMBER_OF_ARMOR_TYPES:
            raise RuntimeError(
                f"The engine reports {engine_armor_types} armor types, NUMBER_OF_ARMOR_TYPES is "
                f"{NUMBER_OF_ARMOR_TYPES}: update it in reinforcement_learning_environment.py for this game version."
            )
        terrain_w, terrain_h = self.configuration.observation.terrain_resolution
        terrain_msg = self._receive()
        self._terrain[:] = np.array(terrain_msg["data"], dtype=np.float32).reshape(terrain_h, terrain_w, 3)

    # --- Episodes -----------------------------------------------------------------------------------------------

    def _receive_observation(self) -> dict:
        raw = self._receive()
        merge_commands(raw)
        return raw

    def _observations(self, raw: dict) -> dict[str, dict[str, np.ndarray]]:
        resolution = self.configuration.observation.terrain_resolution
        # Terrain with the visibility of each alliance that has agents, shared by its agents.
        terrains = {
            allyteam: np.concatenate(
                [self._terrain, visibility(raw, self._teams_of(allyteam), self._map_size, resolution)[..., None]],
                axis=2,
            )
            for allyteam in {self._allyteams[team] for team in self.agent_teams}
        }
        observations = {}
        for index, agent in enumerate(self.possible_agents):
            if agent in self.agents:
                observation = encode_observation(
                    raw,
                    self._agent_unit_ids[agent],
                    self.agent_unit_names[index],
                    self.configuration,
                    self._map_size,
                    self._allyteams,
                )
                observation["terrain"] = terrains[self._allyteams[self.agent_teams[index]]]
                observations[agent] = observation
        return observations

    def _information(self, raw: dict, allyteam: int) -> dict:
        """Information for the agents of ``allyteam``: allies and enemies are counted from its side."""
        allies = self._teams_of(allyteam)
        return {
            "episode": raw["episode"],
            "episode_frame": raw["episode_frame"],
            "frames_since_previous_decision": raw.get("frames_since_previous_decision", 0),
            "allies_alive": sum(u["team"] in allies for u in raw["units"]),
            "enemies_alive": sum(u["team"] not in allies for u in raw["units"]),
            # Who lost (scenario.commander_death_ends_game: a side can lose with units left); both: a draw.
            "allies_defeated": side_defeated(raw["units"], allies, self.configuration.game.scenario),
            "enemies_defeated": side_defeated(
                raw["units"], set(range(len(self._allyteams))) - allies, self.configuration.game.scenario
            ),
            "late_actions": raw["late_actions"],
            "maximum_lead_frames": raw.get("maximum_lead", 0),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._random = np.random.default_rng(seed)
        if self._process is None:
            self._start()
        self._episode += 1
        configuration = self.configuration
        states = unit_states(
            configuration.game, (configuration.action.agent_fire_state, configuration.action.agent_move_state)
        )
        spawns = spawn_positions(configuration.game.scenario, self._random)
        self._send({"type": "reset", "episode": self._episode, "spawns": spawns, "states": states})
        raw = self._receive_observation()
        spawned = [int(identifier) for identifier in str(raw["spawned_units"]).split(",")]
        # Units removed by the reset (the commanders, the previous episode) are still listed for one frame.
        raw["units"] = [u for u in raw["units"] if u["id"] in set(spawned)]
        # Spawns come in team order, units in the order of their team: the same order as the agents.
        unit_ids_by_team: dict[int, list[int]] = {}
        for spawn, unit_id in zip(spawns, spawned, strict=True):
            unit_ids_by_team.setdefault(spawn["team"], []).append(unit_id)
        self._agent_unit_ids = {}
        for agent, team in zip(self.possible_agents, self.agent_teams, strict=True):
            self._agent_unit_ids[agent] = unit_ids_by_team[team][int(agent.rsplit("_", 1)[1])]
        self._initial = {}
        for allyteam in {self._allyteams[team] for team in self.agent_teams}:
            allies = self._teams_of(allyteam)
            enemies = [u for u in raw["units"] if u["team"] not in allies]
            friends = [u for u in raw["units"] if u["team"] in allies]
            self._initial[allyteam] = (
                sum(u["maximum_health"] for u in enemies),
                sum(u["maximum_health"] for u in friends),
                len(enemies),
            )
        self._previous = raw
        self.agents = list(self.possible_agents)
        infos = {agent: self._information(raw, self._allyteam_of(agent)) for agent in self.agents}
        return self._observations(raw), infos

    def step(self, actions: dict[str, dict]):
        if not self.agents:
            raise RuntimeError("The episode is over: call reset().")
        orders = []
        for agent in self.agents:
            action = actions.get(agent)
            if action is None:
                continue
            command = COMMANDS[int(action["command"])]
            if command == "continue" or command not in self.configuration.action.allowed_commands(
                self.agent_unit_names[self.possible_agents.index(agent)]
            ):
                continue
            x, z = np.clip(np.asarray(action["target"], dtype=float), 0.0, 1.0)
            orders.append(f"{self._agent_unit_ids[agent]},{command},{x:.6f},{z:.6f}")
        self._send({"type": "actions", "orders": ";".join(orders)})
        raw = self._receive_observation()

        assert self._previous is not None
        rewards_by_allyteam = {
            allyteam: team_reward(
                self._previous, raw, *initial, self.configuration, ally_teams=self._teams_of(allyteam)
            )
            for allyteam, initial in self._initial.items()
        }
        self._previous = raw
        alive_units = {u["id"] for u in raw["units"]}
        # The episode ends when one alliance at most is not defeated (no unit left, or no commander).
        scenario = self.configuration.game.scenario
        alliances = set(self._allyteams)
        episode_over = sum(not side_defeated(raw["units"], self._teams_of(a), scenario) for a in alliances) <= 1
        time_over = raw["episode_frame"] >= self.configuration.game.scenario.maximum_game_frames

        observations = self._observations(raw)
        rewards = {agent: rewards_by_allyteam[self._allyteam_of(agent)] for agent in self.agents}
        terminations = {agent: self._agent_unit_ids[agent] not in alive_units or episode_over for agent in self.agents}
        truncations = {agent: time_over and not terminations[agent] for agent in self.agents}
        infos = {agent: self._information(raw, self._allyteam_of(agent)) for agent in self.agents}
        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]
        return observations, rewards, terminations, truncations, infos
