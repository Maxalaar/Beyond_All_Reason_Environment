"""Micro environment: a small fight between two groups of units, one agent per unit of team 0.

The interface follows PettingZoo's parallel API (reset, step, close, one dictionary entry per agent) without
depending on it, so any reinforcement learning library can wrap it.

Observation of an agent (dictionary of float32 arrays):
    self           its own unit, the features of ENTITY_FEATURES then the scale features of SCALE_FEATURES
    entities       the maximum_entities_seen nearest other units (allies and enemies), one row each
    entity_mask    1 for the rows that hold a unit, 0 for padding
    command_mask   1 for the commands its unit type may give (MicroScenarioConfiguration.commands)
Positions are fractions of the map in [0, 1]; the scale features tell what such a fraction means for the unit
(distance covered per decision, weapon and sight ranges, map size and aspect ratio).

Action of an agent: {"command": index in COMMANDS, "target": [x, z] in [0, 1]^2}, a point of the map
((0, 0) one corner, (1, 1) the opposite one). "attack" targets the enemy nearest to the point.

Timing: an agent decides every decision_interval_frames and its action reaches its unit action_delay_frames later,
at the next decision when both are equal. Orders from Lua travel through the engine server, which may simulate
ahead of the client while Python thinks; when an action arrives after its frame it is given at once, and the next
observation waits for it. So each observation shows, as the unit's current command, the action decided the step
before, and the self feature "decision_interval_ratio" gives the time actually elapsed (1 when on time).
"""

import contextlib
import json
import os
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .game_extension import write_game_extension
from .match import RUNS_DIRECTORY, start_engine
from .micro_configuration import COMMANDS, OPPONENT_STATES, MicroScenarioConfiguration
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration

AGENT_TEAM, OPPONENT_TEAM = 0, 1
COMMAND_STATES = ("none", "move", "fight", "attack", "stop", "other")
# Unit states the engine reports per frame; observations give them per decision.
CONNECTION_TIMEOUT_SECONDS = 300


def entity_feature_names(configuration: MicroScenarioConfiguration) -> list[str]:
    return [
        "is_ally",
        "is_enemy",
        *(f"type_{name}" for name in configuration.unit_types),
        "x",
        "z",
        "relative_x",
        "relative_z",
        "distance",
        "health_fraction",
        "maximum_health_thousands",
        "velocity_x",
        "velocity_z",
        *(f"command_{name}" for name in COMMAND_STATES),
        "has_target",
        "target_x",
        "target_z",
    ]


SCALE_FEATURES = [
    "map_size_x_thousands",
    "map_size_z_thousands",
    "map_aspect_ratio",
    "move_per_decision_x",
    "move_per_decision_z",
    "weapon_range_x",
    "weapon_range_z",
    "sight_x",
    "sight_z",
    "episode_time_fraction",
    "decision_interval_ratio",
]


def _entity_features(
    unit: dict, observer: dict, configuration: MicroScenarioConfiguration, map_size: tuple[float, float]
) -> list[float]:
    size_x, size_z = map_size
    relative_x, relative_z = unit["x"] - observer["x"], unit["z"] - observer["z"]
    interval = configuration.decision_interval_frames
    target_x, target_z = unit.get("target_x"), unit.get("target_z")
    has_target = target_x is not None and target_z is not None
    return [
        float(unit["team"] == observer["team"]),
        float(unit["team"] != observer["team"]),
        *(float(unit["unit"] == name) for name in configuration.unit_types),
        unit["x"],
        unit["z"],
        relative_x,
        relative_z,
        float(np.hypot(relative_x * size_x, relative_z * size_z) / max(size_x, size_z)),
        unit["health"] / unit["maximum_health"] if unit["maximum_health"] > 0 else 0.0,
        unit["maximum_health"] / 1000,
        unit["velocity_x"] * interval / size_x,
        unit["velocity_z"] * interval / size_z,
        *(float(unit["command"] == name) for name in COMMAND_STATES),
        float(has_target),
        float(target_x) if target_x is not None else 0.0,
        float(target_z) if target_z is not None else 0.0,
    ]


def _scale_features(
    unit: dict,
    unit_types: dict,
    episode_frame: int,
    frames_since_previous_decision: int,
    configuration: MicroScenarioConfiguration,
    map_size,
) -> list[float]:
    size_x, size_z = map_size
    unit_type = unit_types.get(unit["unit"], {})
    move = unit_type.get("speed", 0.0) * configuration.decision_interval_frames / 30
    weapon_range, sight = unit_type.get("weapon_range", 0.0), unit_type.get("sight", 0.0)
    return [
        size_x / 1000,
        size_z / 1000,
        size_x / size_z,
        move / size_x,
        move / size_z,
        weapon_range / size_x,
        weapon_range / size_z,
        sight / size_x,
        sight / size_z,
        episode_frame / configuration.maximum_episode_frames,
        frames_since_previous_decision / configuration.decision_interval_frames,
    ]


def encode_observation(
    raw: dict,
    unit_id: int,
    unit_name: str,
    configuration: MicroScenarioConfiguration,
    map_size: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Observation of the agent whose unit is ``unit_id``; zeros (and empty masks) when the unit is dead."""
    number_of_entity_features = len(entity_feature_names(configuration))
    maximum = configuration.maximum_entities_seen
    observation = {
        "self": np.zeros(number_of_entity_features + len(SCALE_FEATURES), dtype=np.float32),
        "entities": np.zeros((maximum, number_of_entity_features), dtype=np.float32),
        "entity_mask": np.zeros(maximum, dtype=np.float32),
        "command_mask": np.array(
            [float(command in configuration.allowed_commands(unit_name)) for command in COMMANDS], dtype=np.float32
        ),
    }
    units = {unit["id"]: unit for unit in raw["units"]}
    observer = units.get(unit_id)
    if observer is None:
        return observation
    observation["self"][:] = _entity_features(observer, observer, configuration, map_size) + _scale_features(
        observer,
        raw.get("unit_types") or {},
        raw["episode_frame"],
        raw.get("frames_since_previous_decision", 0),
        configuration,
        map_size,
    )
    others = [unit for unit in raw["units"] if unit["id"] != unit_id]
    size_x, size_z = map_size
    others.sort(
        key=lambda unit: ((unit["x"] - observer["x"]) * size_x) ** 2 + ((unit["z"] - observer["z"]) * size_z) ** 2
    )
    for row, unit in enumerate(others[:maximum]):
        observation["entities"][row] = _entity_features(unit, observer, configuration, map_size)
        observation["entity_mask"][row] = 1.0
    return observation


def team_reward(
    previous: dict,
    current: dict,
    initial_enemy_health: float,
    initial_ally_health: float,
    initial_enemies: int,
    configuration: MicroScenarioConfiguration,
) -> float:
    """Reward of one step for team 0, from the unit states before and after it."""
    before = {unit["id"]: unit for unit in previous["units"]}
    after = {unit["id"]: unit for unit in current["units"]}

    def health_lost(team: int) -> float:
        return sum(
            max(0.0, unit["health"] - (after[identifier]["health"] if identifier in after else 0.0))
            for identifier, unit in before.items()
            if unit["team"] == team
        )

    weights = configuration.rewards
    enemies_before = sum(unit["team"] == OPPONENT_TEAM for unit in before.values())
    enemies_after = sum(unit["team"] == OPPONENT_TEAM for unit in after.values())
    allies_after = sum(unit["team"] == AGENT_TEAM for unit in after.values())
    reward = weights.damage_dealt * health_lost(OPPONENT_TEAM) / max(initial_enemy_health, 1e-9)
    reward -= weights.damage_received * health_lost(AGENT_TEAM) / max(initial_ally_health, 1e-9)
    reward += weights.kill * (enemies_before - enemies_after) / max(initial_enemies, 1)
    if enemies_after == 0 and enemies_before > 0:
        reward += weights.victory
    elif allies_after == 0:
        reward += weights.defeat
    return float(reward)


def spawn_positions(configuration: MicroScenarioConfiguration, random: np.random.Generator) -> list[dict]:
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


class MicroEnvironment:
    """One engine process; episodes are reset inside the running match, without reloading the game."""

    metadata = {"name": "beyond_all_reason_micro"}

    def __init__(self, configuration: MicroScenarioConfiguration | None = None):
        self.configuration = configuration or MicroScenarioConfiguration()
        self.agent_unit_names = [name for name, count in self.configuration.agent_units.items() for _ in range(count)]
        self.possible_agents = [f"unit_{index}" for index in range(len(self.agent_unit_names))]
        self.agents: list[str] = []
        self.run_directory: Path | None = None

        entity_features = len(entity_feature_names(self.configuration))
        maximum = self.configuration.maximum_entities_seen
        self._observation_space = spaces.Dict(
            {
                "self": spaces.Box(-np.inf, np.inf, (entity_features + len(SCALE_FEATURES),), np.float32),
                "entities": spaces.Box(-np.inf, np.inf, (maximum, entity_features), np.float32),
                "entity_mask": spaces.Box(0, 1, (maximum,), np.float32),
                "command_mask": spaces.Box(0, 1, (len(COMMANDS),), np.float32),
            }
        )
        self._action_space = spaces.Dict(
            {"command": spaces.Discrete(len(COMMANDS)), "target": spaces.Box(0, 1, (2,), np.float32)}
        )

        self._process: subprocess.Popen | None = None
        self._server: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._reader = None
        self._map_size = (1.0, 1.0)
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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_directory = RUNS_DIRECTORY / "micro_environments" / f"{timestamp}_{os.getpid()}_{id(self) % 100000}"
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(CONNECTION_TIMEOUT_SECONDS)
        port = self._server.getsockname()[1]
        match_configuration = MatchConfiguration(
            # Both teams belong to NullAI, which does nothing: the scenario gadget gives every order.
            players=[
                ArtificialIntelligencePlayer(short_name="NullAI", version="0.1", side="Armada"),
                ArtificialIntelligencePlayer(short_name="NullAI", version="0.1", side="Cortex"),
            ],
            map_name=configuration.map_name,
            game_version=write_game_extension(self.run_directory),
            speed=configuration.speed,
            maximum_game_minutes=1_000_000,  # episodes end inside the match; close() ends the engine
            extra_mod_options={
                "deathmode": "neverend",
                "reinforcement_learning_port": str(port),
                "reinforcement_learning_decision_interval": str(configuration.decision_interval_frames),
                "reinforcement_learning_action_delay": str(configuration.action_delay_frames),
                "reinforcement_learning_attack_snap_radius": f"{configuration.attack_snap_radius:g}",
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

    def _send(self, message: dict) -> None:
        assert self._connection is not None
        self._connection.sendall((json.dumps(message) + "\n").encode())

    def _receive(self) -> dict:
        assert self._reader is not None
        line = self._reader.readline()
        if not line:
            log = self.run_directory / "engine_output.log" if self.run_directory else "the engine log"
            raise RuntimeError(f"The engine closed the connection, see {log}")
        return json.loads(line)

    # --- Episodes -----------------------------------------------------------------------------------------------

    def _observations(self, raw: dict) -> dict[str, dict[str, np.ndarray]]:
        return {
            agent: encode_observation(
                raw, self._agent_unit_ids[agent], self.agent_unit_names[index], self.configuration, self._map_size
            )
            for index, agent in enumerate(self.possible_agents)
            if agent in self.agents
        }

    def _information(self, raw: dict) -> dict:
        return {
            "episode": raw["episode"],
            "episode_frame": raw["episode_frame"],
            "frames_since_previous_decision": raw.get("frames_since_previous_decision", 0),
            "agents_alive": sum(unit["team"] == AGENT_TEAM for unit in raw["units"]),
            "enemies_alive": sum(unit["team"] == OPPONENT_TEAM for unit in raw["units"]),
            # Decisions whose actions reached the simulation after their apply frame (they were given late, and the
            # next observation waited for them), and the largest delay between a decision and the arrival of its
            # actions, in frames.
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
        enemies = [unit for unit in raw["units"] if unit["team"] == OPPONENT_TEAM]
        allies = [unit for unit in raw["units"] if unit["team"] == AGENT_TEAM]
        self._initial = (
            sum(unit["maximum_health"] for unit in enemies),
            sum(unit["maximum_health"] for unit in allies),
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
                command = "continue"  # a masked command does nothing
            x, z = np.clip(np.asarray(action["target"], dtype=float), 0.0, 1.0)
            wire_actions.append({"unit": self._agent_unit_ids[agent], "command": command, "x": float(x), "z": float(z)})
        self._send({"type": "actions", "actions": wire_actions})
        raw = self._receive()

        assert self._previous is not None
        reward = team_reward(self._previous, raw, *self._initial, self.configuration)
        self._previous = raw
        alive = {unit["id"] for unit in raw["units"] if unit["team"] == AGENT_TEAM}
        enemies_left = any(unit["team"] == OPPONENT_TEAM for unit in raw["units"])
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

    def close(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(OSError):  # the engine may already be gone
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
        if self.run_directory is not None and not self.configuration.keep_run_directory:
            shutil.rmtree(self.run_directory, ignore_errors=True)

    def __enter__(self) -> "MicroEnvironment":
        return self

    def __exit__(self, *exception) -> None:
        self.close()
