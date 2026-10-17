"""Combat environment: a fight between two groups of units, one agent per unit of team 0.

The game is divided into three dimensions: resource gathering, production, and combat. This environment
covers combat only — no commanders, no factories, no resource collection.

The interface follows PettingZoo's parallel API (reset, step, close, one dictionary entry per agent) without
depending on it, so any reinforcement learning library can wrap it.

Observation of an agent (dictionary of float32 arrays):
    Hierarchical structure: three levels of attention.
    Level 1 — per entity: chassis (static) + state (dynamic) + weapons → entity token.
    Level 2 — per category: entity tokens → category token (5 categories).
    Level 3 — global: all category tokens + context + terrain → action.

    Self (category 0):
        self_chassis           chassis features of the agent's own unit (type one-hot, max health,
                               armor type one-hot, normalized speed/range/sight, can_move)
        self_state             dynamic state features (position, current health, velocity, command, target)
        self_weapons           (maximum_weapons_per_unit, n_weapon) — per-weapon features; padded with zeros
        self_weapon_mask       1 for real slots, 0 for padding; shape (maximum_weapons_per_unit,)

    Ally units (category 1):
        ally_units_chassis     (maximum_ally_units, n_chassis)
        ally_units_state       (maximum_ally_units, n_state)
        ally_unit_mask         1 for real rows, 0 for padding
        ally_units_weapons     (maximum_ally_units, maximum_weapons_per_unit, n_weapon)
        ally_units_weapon_mask (maximum_ally_units, maximum_weapons_per_unit)

    Enemy units (category 2):
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

    Enemy buildings (category 4):
        enemy_buildings_chassis     (maximum_enemy_buildings, n_chassis)
        enemy_buildings_state       (maximum_enemy_buildings, n_state)
        enemy_building_mask         1 for real rows, 0 for padding
        enemy_buildings_weapons     (maximum_enemy_buildings, maximum_weapons_per_unit, n_weapon)
        enemy_buildings_weapon_mask (maximum_enemy_buildings, maximum_weapons_per_unit)

    Level 3 inputs:
        context     global context features sent directly to level 3 (map size, episode time,
                    decision interval ratio); does not depend on which agent is observing
        terrain     (terrain_height, terrain_width, 3) — the full map resampled to a fixed grid;
                    channels: normalised height (raw_elmos/1000), slope squashed to [0,1), water mask;
                    shared across all agents, constant within an episode

    command_mask   1 for commands the unit type may give (CombatConfiguration.commands)

Chassis features (static within an episode for a given unit type on a given map):
    type one-hot, maximum_health_thousands, armor_type one-hot (num_armor_types),
    speed_x/z, weapon_range_x/z, sight_x/z, can_move.

State features (dynamic per step):
    x, z, relative_x/z, distance, health_fraction, velocity_x/z,
    command one-hot (COMMAND_STATES), has_target, target_x/z.

Weapon features per slot: range_x/z, reload_decisions, aoe_fraction, is_paralyzed,
    damage_armor_0 … damage_armor_{num_armor_types-1} (all in thousands).

Positions are fractions of the map in [0, 1]. All other lengths (relative positions, distances,
speeds, ranges, sight, AoE) are divided by DISTANCE_SCALE (a fixed constant in elmos) so the network
sees a stable physical reference frame regardless of map size. Context features give the actual map
dimensions (map_size_x/z_thousands) so the network can relate map fractions to that reference.

Action of an agent: {"command": index in COMMANDS, "target": [x, z] in [0, 1]^2}.
"attack" snaps to the nearest enemy within attack_snap_radius; otherwise it is ignored.

Timing: an agent decides every decision_interval_frames. BridgeAI runs inside the simulation thread and
blocks on Python; the whole simulation pauses while Python computes actions, giving exact frame timing.
"""

import os
import socket
from datetime import datetime

import numpy as np
from gymnasium import spaces

from .combat_configuration import COMMANDS, OPPONENT_STATES, CombatConfiguration
from .environment import CONNECTION_TIMEOUT_SECONDS, Environment
from .environment_configuration import FRAMES_PER_SECOND
from .game_extension import write_game_extension
from .installation import build_bridge_ai
from .match import RUNS_DIRECTORY, start_engine
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration

AGENT_TEAM, OPPONENT_TEAM = 0, 1
COMMAND_STATES = ("none", "move", "fight", "attack", "stop", "other")

# Fixed reference length for normalising all spatial quantities (speeds, ranges, relative distances).
# Using a stable physical constant instead of the variable map size keeps per-unit features consistent
# across maps of different sizes.
DISTANCE_SCALE = 512.0

CONTEXT_FEATURES = [
    "map_size_x_thousands",
    "map_size_z_thousands",
    "map_aspect_ratio",
    "episode_time_fraction",
    "decision_interval_ratio",
]


def chassis_feature_names(configuration: CombatConfiguration) -> list[str]:
    """Static features of any entity: type, health capacity, armor, mobility and scale."""
    return [
        *(f"type_{name}" for name in configuration.unit_types),
        "maximum_health_thousands",
        *(f"armor_type_{i}" for i in range(configuration.num_armor_types)),
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
    ]


def weapon_feature_names(num_armor_types: int) -> list[str]:
    """Feature names for a single weapon slot."""
    return [
        "range_x",
        "range_z",
        "reload_decisions",
        "aoe_fraction",
        "is_paralyzed",
        *(f"damage_armor_{i}" for i in range(num_armor_types)),
    ]


def _chassis_features(
    unit: dict,
    unit_type: dict,
    configuration: CombatConfiguration,
) -> list[float]:
    armor_t = int(unit_type.get("armor_type", 0))
    speed = unit_type.get("speed", 0.0)
    weapon_range = unit_type.get("weapon_range", 0.0)
    sight = unit_type.get("sight", 0.0)
    return [
        *(float(unit["unit"] == name) for name in configuration.unit_types),
        unit_type.get("maximum_health", 1.0) / 1000.0,
        *(float(armor_t == i) for i in range(configuration.num_armor_types)),
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
    configuration: CombatConfiguration,
    map_size: tuple[float, float],
) -> list[float]:
    size_x, size_z = map_size
    rel_x_elmos = (unit["x"] - observer["x"]) * size_x
    rel_z_elmos = (unit["z"] - observer["z"]) * size_z
    interval = configuration.decision_interval_frames
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
    ]


def _weapon_features(weapon: dict, decision_interval: int, num_armor_types: int) -> list[float]:
    damage_per_type: list[float] = weapon.get("damage_per_type", [])
    return [
        weapon["range"] / DISTANCE_SCALE,
        weapon["range"] / DISTANCE_SCALE,
        weapon["reload"] * FRAMES_PER_SECOND / decision_interval,
        weapon["aoe"] / DISTANCE_SCALE,
        float(weapon.get("paralyzed", 0)),
        *((float(damage_per_type[i]) / 1000.0 if i < len(damage_per_type) else 0.0) for i in range(num_armor_types)),
    ]


def _unit_weapons(unit_type: dict, configuration: CombatConfiguration) -> tuple[np.ndarray, np.ndarray]:
    num_at = configuration.num_armor_types
    n_weapon = len(weapon_feature_names(num_at))
    max_weapons = configuration.maximum_weapons_per_unit
    matrix = np.zeros((max_weapons, n_weapon), dtype=np.float32)
    mask = np.zeros(max_weapons, dtype=np.float32)
    for slot, weapon in enumerate(unit_type.get("weapons", [])[:max_weapons]):
        matrix[slot] = _weapon_features(weapon, configuration.decision_interval_frames, num_at)
        mask[slot] = 1.0
    return matrix, mask


def _context_features(
    episode_frame: int,
    frames_since_previous_decision: int,
    configuration: CombatConfiguration,
    map_size: tuple[float, float],
) -> list[float]:
    size_x, size_z = map_size
    return [
        size_x / 1000,
        size_z / 1000,
        size_x / size_z,
        episode_frame / configuration.maximum_episode_frames,
        frames_since_previous_decision / configuration.decision_interval_frames,
    ]


def encode_observation(
    raw: dict,
    unit_id: int,
    unit_name: str,
    configuration: CombatConfiguration,
    map_size: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Observation of the agent whose unit is unit_id; zeros (and empty masks) when the unit is dead."""
    n_chassis = len(chassis_feature_names(configuration))
    n_state = len(state_feature_names())
    max_ally = configuration.maximum_ally_units
    max_enemy = configuration.maximum_enemy_units
    max_ally_bld = configuration.maximum_ally_buildings
    max_enemy_bld = configuration.maximum_enemy_buildings
    max_weapons = configuration.maximum_weapons_per_unit
    num_at = configuration.num_armor_types
    n_weapon = len(weapon_feature_names(num_at))

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
            [float(command in configuration.allowed_commands(unit_name)) for command in COMMANDS], dtype=np.float32
        ),
    }

    units = {u["id"]: u for u in raw["units"]}
    unit_types_data: dict = raw.get("unit_types") or {}
    observer = units.get(unit_id)
    if observer is None:
        return observation

    observer_type = unit_types_data.get(unit_name, {})
    observation["self_chassis"][:] = _chassis_features(observer, observer_type, configuration)
    observation["self_state"][:] = _state_features(observer, observer, configuration, map_size)
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

    all_allies = [u for u in raw["units"] if u["id"] != unit_id and u["team"] == observer["team"]]
    all_enemies = [u for u in raw["units"] if u["team"] != observer["team"]]
    ally_units = sorted([u for u in all_allies if is_mobile(u)], key=dist_sq)
    ally_buildings = sorted([u for u in all_allies if not is_mobile(u)], key=dist_sq)
    enemy_units = sorted([u for u in all_enemies if is_mobile(u)], key=dist_sq)
    enemy_buildings = sorted([u for u in all_enemies if not is_mobile(u)], key=dist_sq)

    def fill_rows(
        entities: list[dict],
        chassis_key: str,
        state_key: str,
        mask_key: str,
        weapons_key: str,
        weapon_mask_key: str,
        max_rows: int,
    ) -> None:
        for row, entity in enumerate(entities[:max_rows]):
            entity_type = unit_types_data.get(entity["unit"], {})
            observation[chassis_key][row] = _chassis_features(entity, entity_type, configuration)
            observation[state_key][row] = _state_features(entity, observer, configuration, map_size)
            observation[mask_key][row] = 1.0
            observation[weapons_key][row], observation[weapon_mask_key][row] = _unit_weapons(entity_type, configuration)

    fill_rows(
        ally_units,
        "ally_units_chassis",
        "ally_units_state",
        "ally_unit_mask",
        "ally_units_weapons",
        "ally_units_weapon_mask",
        max_ally,
    )
    fill_rows(
        ally_buildings,
        "ally_buildings_chassis",
        "ally_buildings_state",
        "ally_building_mask",
        "ally_buildings_weapons",
        "ally_buildings_weapon_mask",
        max_ally_bld,
    )
    fill_rows(
        enemy_units,
        "enemy_units_chassis",
        "enemy_units_state",
        "enemy_unit_mask",
        "enemy_units_weapons",
        "enemy_units_weapon_mask",
        max_enemy,
    )
    fill_rows(
        enemy_buildings,
        "enemy_buildings_chassis",
        "enemy_buildings_state",
        "enemy_building_mask",
        "enemy_buildings_weapons",
        "enemy_buildings_weapon_mask",
        max_enemy_bld,
    )

    return observation


def team_reward(
    previous: dict,
    current: dict,
    initial_enemy_health: float,
    initial_ally_health: float,
    initial_enemies: int,
    configuration: CombatConfiguration,
) -> float:
    """Reward of one step for team 0, from the unit states before and after it."""
    before = {u["id"]: u for u in previous["units"]}
    after = {u["id"]: u for u in current["units"]}

    def health_lost(team: int) -> float:
        return sum(
            max(0.0, unit["health"] - (after[identifier]["health"] if identifier in after else 0.0))
            for identifier, unit in before.items()
            if unit["team"] == team
        )

    weights = configuration.rewards
    enemies_before = sum(u["team"] == OPPONENT_TEAM for u in before.values())
    enemies_after = sum(u["team"] == OPPONENT_TEAM for u in after.values())
    allies_after = sum(u["team"] == AGENT_TEAM for u in after.values())
    reward = weights.damage_dealt * health_lost(OPPONENT_TEAM) / max(initial_enemy_health, 1e-9)
    reward -= weights.damage_received * health_lost(AGENT_TEAM) / max(initial_ally_health, 1e-9)
    reward += weights.kill * (enemies_before - enemies_after) / max(initial_enemies, 1)
    if enemies_after == 0 and enemies_before > 0:
        reward += weights.victory
    elif allies_after == 0:
        reward += weights.defeat
    return float(reward)


def spawn_positions(configuration: CombatConfiguration, random: np.random.Generator) -> list[dict]:
    """Agents first, then opponents: two columns facing each other around the map centre."""
    spawns = []
    for team, units, centre_x in [
        (AGENT_TEAM, configuration.agent_units, 0.5 - configuration.group_separation / 2),
        (OPPONENT_TEAM, configuration.opponent_units, 0.5 + configuration.group_separation / 2),
    ]:
        names = [name for name, count in units.items() for _ in range(count)]
        for index, name in enumerate(names):
            offset = (index / (len(names) - 1) - 0.5) if len(names) > 1 else 0.0
            jitter_x, jitter_z = random.uniform(-1, 1, 2) * configuration.spawn_jitter
            spawns.append(
                {
                    "team": team,
                    "unit": name,
                    "x": float(np.clip(centre_x + jitter_x, 0, 1)),
                    "z": float(np.clip(0.5 + offset * configuration.group_spread + jitter_z, 0, 1)),
                }
            )
    return spawns


class CombatEnvironment(Environment):
    """One engine process running a combat scenario; episodes reset inside the match without reloading the game."""

    metadata = {"name": "beyond_all_reason_combat"}

    def __init__(self, configuration: CombatConfiguration | None = None):
        super().__init__(configuration or CombatConfiguration())
        self.configuration: CombatConfiguration
        self.agent_unit_names = [name for name, count in self.configuration.agent_units.items() for _ in range(count)]
        self.possible_agents = [f"unit_{index}" for index in range(len(self.agent_unit_names))]

        n_chassis = len(chassis_feature_names(self.configuration))
        n_state = len(state_feature_names())
        max_ally = self.configuration.maximum_ally_units
        max_enemy = self.configuration.maximum_enemy_units
        max_ally_bld = self.configuration.maximum_ally_buildings
        max_enemy_bld = self.configuration.maximum_enemy_buildings
        max_weapons = self.configuration.maximum_weapons_per_unit
        n_weapon = len(weapon_feature_names(self.configuration.num_armor_types))
        terrain_w, terrain_h = self.configuration.terrain_resolution

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
                "terrain": spaces.Box(-np.inf, np.inf, (terrain_h, terrain_w, 3), np.float32),
            }
        )
        self._action_space = spaces.Dict(
            {"command": spaces.Discrete(len(COMMANDS)), "target": spaces.Box(0, 1, (2,), np.float32)}
        )

        self._map_size = (1.0, 1.0)
        self._terrain = np.zeros((terrain_h, terrain_w, 3), dtype=np.float32)
        self._random = np.random.default_rng()
        self._episode = -1
        self._agent_unit_ids: dict[str, int] = {}
        self._previous: dict | None = None
        self._initial = (1.0, 1.0, 1)

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Dict:
        return self._action_space

    # --- Engine connection --------------------------------------------------------------------------------------

    def _start(self) -> None:
        configuration = self.configuration
        build_bridge_ai()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_directory = RUNS_DIRECTORY / "combat_environments" / f"{timestamp}_{os.getpid()}_{id(self) % 100000}"
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(CONNECTION_TIMEOUT_SECONDS)
        port = self._server.getsockname()[1]
        match_configuration = MatchConfiguration(
            players=[
                ArtificialIntelligencePlayer(
                    short_name="BridgeAI",
                    version="0.1",
                    side="Armada",
                    options={
                        "port": str(port),
                        "decision_interval": str(configuration.decision_interval_frames),
                        "attack_snap_radius": f"{configuration.attack_snap_radius:g}",
                        "terrain_w": str(configuration.terrain_resolution[0]),
                        "terrain_h": str(configuration.terrain_resolution[1]),
                    },
                ),
                ArtificialIntelligencePlayer(short_name="NullAI", version="0.1", side="Cortex"),
            ],
            map_name=configuration.map_name,
            game_version=write_game_extension(self.run_directory),
            speed=configuration.speed,
            maximum_game_minutes=1_000_000,
            extra_mod_options={
                "deathmode": "neverend",
                "reinforcement_learning_full_vision": "1" if configuration.full_vision else "0",
            },
        )
        self._process = start_engine(match_configuration, self.run_directory, graphical=configuration.graphical)
        try:
            self._connection, _ = self._server.accept()
        except TimeoutError as error:
            raise RuntimeError(f"The engine did not connect, see {self.run_directory / 'engine_output.log'}") from error
        self._reader = self._connection.makefile("r", encoding="utf-8")
        ready = self._receive()
        self._map_size = (float(ready["map"]["size_x"]), float(ready["map"]["size_z"]))
        engine_armor_types = int(ready.get("num_armor_types", self.configuration.num_armor_types))
        if engine_armor_types != self.configuration.num_armor_types:
            raise RuntimeError(
                f"num_armor_types mismatch: engine reports {engine_armor_types}, "
                f"configuration has {self.configuration.num_armor_types}. "
                f"Update CombatConfiguration.num_armor_types to match."
            )
        terrain_w, terrain_h = self.configuration.terrain_resolution
        terrain_msg = self._receive()
        self._terrain = np.array(terrain_msg["data"], dtype=np.float32).reshape(terrain_h, terrain_w, 3)

    # --- Episodes -----------------------------------------------------------------------------------------------

    def _observations(self, raw: dict) -> dict[str, dict[str, np.ndarray]]:
        observations = {
            agent: encode_observation(
                raw, self._agent_unit_ids[agent], self.agent_unit_names[index], self.configuration, self._map_size
            )
            for index, agent in enumerate(self.possible_agents)
            if agent in self.agents
        }
        for obs in observations.values():
            obs["terrain"] = self._terrain
        return observations

    def _information(self, raw: dict) -> dict:
        return {
            "episode": raw["episode"],
            "episode_frame": raw["episode_frame"],
            "frames_since_previous_decision": raw.get("frames_since_previous_decision", 0),
            "agents_alive": sum(u["team"] == AGENT_TEAM for u in raw["units"]),
            "enemies_alive": sum(u["team"] == OPPONENT_TEAM for u in raw["units"]),
            "late_actions": raw["late_actions"],
            "maximum_lead_frames": raw.get("maximum_lead", 0),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._random = np.random.default_rng(seed)
        if self._process is None:
            self._start()
        self._episode += 1
        agent_fire, agent_move = self.configuration.agent_fire_state, self.configuration.agent_move_state
        opponent_fire, opponent_move = OPPONENT_STATES[self.configuration.opponent_behaviour]
        self._send(
            {
                "type": "reset",
                "episode": self._episode,
                "spawns": spawn_positions(self.configuration, self._random),
                "states": [agent_fire, agent_move, opponent_fire, opponent_move],
            }
        )
        raw = self._receive()
        spawned = [int(identifier) for identifier in str(raw["spawned_units"]).split(",")]
        self._agent_unit_ids = dict(zip(self.possible_agents, spawned[: len(self.possible_agents)], strict=True))
        enemies = [u for u in raw["units"] if u["team"] == OPPONENT_TEAM]
        allies = [u for u in raw["units"] if u["team"] == AGENT_TEAM]
        self._initial = (
            sum(u["maximum_health"] for u in enemies),
            sum(u["maximum_health"] for u in allies),
            len(enemies),
        )
        self._previous = raw
        self.agents = list(self.possible_agents)
        information = self._information(raw)
        return self._observations(raw), {agent: dict(information) for agent in self.agents}

    def step(self, actions: dict[str, dict]):
        if not self.agents:
            raise RuntimeError("The episode is over: call reset().")
        wire_actions = []
        for agent in self.agents:
            action = actions.get(agent)
            if action is None:
                continue
            command = COMMANDS[int(action["command"])]
            if command not in self.configuration.allowed_commands(
                self.agent_unit_names[self.possible_agents.index(agent)]
            ):
                command = "continue"
            x, z = np.clip(np.asarray(action["target"], dtype=float), 0.0, 1.0)
            wire_actions.append({"unit": self._agent_unit_ids[agent], "command": command, "x": float(x), "z": float(z)})
        self._send({"type": "actions", "actions": wire_actions})
        raw = self._receive()

        assert self._previous is not None
        reward = team_reward(self._previous, raw, *self._initial, self.configuration)
        self._previous = raw
        alive = {u["id"] for u in raw["units"] if u["team"] == AGENT_TEAM}
        enemies_left = any(u["team"] == OPPONENT_TEAM for u in raw["units"])
        time_over = raw["episode_frame"] >= self.configuration.maximum_episode_frames

        observations = self._observations(raw)
        rewards = {agent: reward for agent in self.agents}
        terminations = {
            agent: (self._agent_unit_ids[agent] not in alive) or not enemies_left or not alive for agent in self.agents
        }
        truncations = {agent: time_over and not terminations[agent] for agent in self.agents}
        information = self._information(raw)
        infos = {agent: dict(information) for agent in self.agents}
        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]
        return observations, rewards, terminations, truncations, infos
