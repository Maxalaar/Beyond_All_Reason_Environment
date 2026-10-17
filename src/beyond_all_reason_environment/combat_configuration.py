"""Configuration for combat environments: unit fights with no production or resource gathering."""

from dataclasses import dataclass, field
from typing import Literal

from .environment_configuration import EnvironmentConfiguration

# Every command an agent may give, in the order of the action index. A unit type only gets the ones listed for it
# in CombatConfiguration.commands; the others are masked.
COMMANDS = ("continue", "move", "fight", "attack", "stop")

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
class CombatConfiguration(EnvironmentConfiguration):
    """Configuration for a combat-only environment: two groups of units fight until one side is eliminated."""

    # Unit name -> number of units. Agents are team 0, the opponent team 1.
    agent_units: dict[str, int] = field(default_factory=lambda: {"armpw": 5})
    opponent_units: dict[str, int] = field(default_factory=lambda: {"armpw": 5})
    # passive: does not move, does not shoot. hold_and_fire: does not move, shoots what comes in range.
    opponent_behaviour: Literal["passive", "hold_and_fire"] = "passive"

    # Spawn: both groups centred on the map centre, apart along x, spread along z (fractions of the map).
    group_separation: float = 0.15
    group_spread: float = 0.06
    spawn_jitter: float = 0.01

    # Commands allowed per unit name; "default" for the other units.
    commands: dict[str, list[str]] = field(default_factory=lambda: {"default": list(COMMANDS)})
    # "attack" targets the enemy nearest to the clicked point within this radius (game units), otherwise ignored.
    attack_snap_radius: float = 60.0

    # Observation: separate entity lists, each taking the nearest units up to the given count.
    maximum_ally_units: int = 4
    maximum_enemy_units: int = 8
    maximum_ally_buildings: int = 0
    maximum_enemy_buildings: int = 0
    # Weapon slots per entity (up to this many weapon tokens, padded).
    maximum_weapons_per_unit: int = 4
    # Number of BAR armor types (15 in the current game version).
    num_armor_types: int = 15

    rewards: RewardWeights = field(default_factory=RewardWeights)

    # States of the agents' units, as a player would set them. Manoeuvre (the game's default) lets a unit move to
    # reach the target of an "attack" order; on hold position the engine drops an attack out of range.
    agent_fire_state: int = FIRE_AT_WILL
    agent_move_state: int = MANOEUVRE

    def __post_init__(self) -> None:
        super().__post_init__()
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
