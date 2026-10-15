"""Settings of the micro environment: a small fight between two groups of units (see micro_environment.py)."""

from dataclasses import dataclass, field
from typing import Literal

from .installation import DEFAULT_MAP

# Every command an agent may give, in the order of the action index. A unit type only gets the ones listed for it
# in MicroScenarioConfiguration.commands; the others are masked.
COMMANDS = ("continue", "move", "fight", "attack", "stop")

# The engine server may simulate up to 30 frames ahead of the local client, and orders from Lua travel through it,
# so an order sent at frame F reaches the simulation by F + 30 at the latest: delaying every action by more than
# that makes its timing exact (see reinforcement_learning_scenario.lua).
MAXIMUM_SERVER_LEAD_FRAMES = 30

FRAMES_PER_SECOND = 30

# Fire states: 0 hold fire, 1 return fire, 2 fire at will. Move states: 0 hold position, 1 manoeuvre, 2 roam.
FIRE_AT_WILL, HOLD_FIRE = 2, 0
HOLD_POSITION, MANOEUVRE = 0, 1
OPPONENT_STATES = {
    "passive": (HOLD_FIRE, HOLD_POSITION),  # does not move, does not shoot
    "hold_and_fire": (FIRE_AT_WILL, HOLD_POSITION),  # does not move, shoots what comes in range
}


@dataclass
class RewardWeights:
    """Team reward of each step, shared by all agents. Damages are fractions of the total health of the side."""

    damage_dealt: float = 1.0
    damage_received: float = 1.0
    # Per enemy destroyed, divided by the number of enemies.
    kill: float = 0.5
    victory: float = 1.0
    defeat: float = -1.0


@dataclass
class MicroScenarioConfiguration:
    map_name: str = DEFAULT_MAP
    # Unit name -> number of units. Agents are team 0, the opponent team 1.
    agent_units: dict[str, int] = field(default_factory=lambda: {"armpw": 5})
    opponent_units: dict[str, int] = field(default_factory=lambda: {"armpw": 5})
    opponent_behaviour: Literal["passive", "hold_and_fire"] = "passive"

    # Spawn: both groups centred on the map centre, apart along x, spread along z (fractions of the map).
    group_separation: float = 0.15
    group_spread: float = 0.06
    spawn_jitter: float = 0.01

    maximum_episode_seconds: float = 60.0
    # An agent decides every decision_interval_frames; its action reaches the units action_delay_frames later.
    # With both equal, each action starts exactly at the next decision, which then observes it as the unit's order.
    decision_interval_frames: int = 32
    action_delay_frames: int = 32

    # Observation: the nearest entities each agent sees.
    maximum_entities_seen: int = 16

    # Both teams see the whole map. Observations show every unit, and the engine only lets a unit attack what its
    # team sees: without it, "attack" on a unit beyond sight range is dropped.
    full_vision: bool = True

    # Commands allowed per unit name; "default" for the other units.
    commands: dict[str, list[str]] = field(default_factory=lambda: {"default": list(COMMANDS)})
    # "attack" targets the enemy nearest to the clicked point within this radius (game units), otherwise it is ignored.
    attack_snap_radius: float = 60.0

    rewards: RewardWeights = field(default_factory=RewardWeights)

    # States of the agents' units, as a player would set them. Manoeuvre (the game's default) lets a unit move to
    # reach the target of an "attack" order; on hold position the engine drops an attack out of range.
    agent_fire_state: int = FIRE_AT_WILL
    agent_move_state: int = MANOEUVRE

    # Engine: as fast as possible without graphics; with graphical=True a window shows the episodes at speed.
    graphical: bool = False
    speed: float = 1000.0
    # Keep runs/micro_environments/<run>/ (start script, engine log) after close(), for debugging.
    keep_run_directory: bool = False

    def __post_init__(self) -> None:
        if self.action_delay_frames <= MAXIMUM_SERVER_LEAD_FRAMES:
            raise ValueError(f"action_delay_frames must be above {MAXIMUM_SERVER_LEAD_FRAMES} for exact timing")
        if self.decision_interval_frames < 1:
            raise ValueError("decision_interval_frames must be at least 1")
        for unit, allowed in self.commands.items():
            unknown = set(allowed) - set(COMMANDS)
            if unknown:
                raise ValueError(f"unknown commands for {unit}: {sorted(unknown)}, known: {COMMANDS}")
        if self.opponent_behaviour not in OPPONENT_STATES:
            raise ValueError(f"opponent_behaviour must be one of {sorted(OPPONENT_STATES)}")

    @property
    def unit_types(self) -> list[str]:
        """Every unit name of the scenario, sorted: the order of the unit type one-hot in observations."""
        return sorted(set(self.agent_units) | set(self.opponent_units))

    @property
    def number_of_agents(self) -> int:
        return sum(self.agent_units.values())

    def allowed_commands(self, unit_name: str) -> list[str]:
        return self.commands.get(unit_name, self.commands.get("default", list(COMMANDS)))

    @property
    def maximum_episode_frames(self) -> int:
        return int(self.maximum_episode_seconds * FRAMES_PER_SECOND)
