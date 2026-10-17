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
class AllyteamConfiguration:
    """Units and (for non-agent allyteams) behaviour of one allyteam.

    Allyteam 0 is always controlled by the RL agents; its behaviour field is ignored.
    Allyteams 1+ are NullAI-driven with the specified fire/move state preset:
        passive:      does not move, does not shoot.
        hold_and_fire: does not move, shoots what comes into range.
    """

    units: dict[str, int] = field(default_factory=lambda: {"armpw": 5})
    behaviour: Literal["passive", "hold_and_fire"] = "hold_and_fire"


@dataclass
class CombatConfiguration(EnvironmentConfiguration):
    """Configuration for a combat-only environment: N allyteams fight until one side is eliminated.

    allyteams[0]  — RL agents (one agent per unit, shared policy).
    allyteams[1+] — NullAI opponents, with per-allyteam fire/move state presets.
    """

    # Each entry defines one allyteam; allyteams[0] = agents, rest = opponents.
    allyteams: list[AllyteamConfiguration] = field(
        default_factory=lambda: [
            AllyteamConfiguration(units={"armpw": 5}, behaviour="hold_and_fire"),
            AllyteamConfiguration(units={"armpw": 5}, behaviour="hold_and_fire"),
        ]
    )

    # Spawn: allyteams spread along x, units spread along z (fractions of the map).
    group_separation: float = 0.15
    group_spread: float = 0.06
    spawn_jitter: float = 0.01
    # Spawn zone range in [0, 1]: controls how wide each allyteam's spawn zone is along x.
    # At 0.0 each allyteam spawns at a single deterministic x (evenly spread across [0, 1]).
    # At 1.0 each allyteam occupies its full band of the map width.
    # When None, the classic group_separation / group_spread / spawn_jitter logic is used instead.
    spawn_x_range: float | None = None

    # Commands allowed per unit name; "default" for the other units.
    commands: dict[str, list[str]] = field(default_factory=lambda: {"default": list(COMMANDS)})
    # "attack" targets the enemy nearest to the clicked point within this radius (game units), otherwise ignored.
    attack_snap_radius: float = 60.0

    # Actor observation: separate entity lists, each taking the nearest units up to the given count.
    maximum_ally_units: int = 4
    maximum_enemy_units: int = 8
    maximum_ally_buildings: int = 0
    maximum_enemy_buildings: int = 0
    # Weapon slots per entity (up to this many weapon tokens, padded).
    maximum_weapons_per_unit: int = 4
    # Number of BAR armor types (15 in the current game version).
    num_armor_types: int = 15

    # Critic observation: per-enemy-allyteam budgets (distinct from the combined actor budgets above).
    maximum_units_per_enemy_allyteam: int = 8
    maximum_buildings_per_enemy_allyteam: int = 0

    rewards: RewardWeights = field(default_factory=RewardWeights)

    # Fire and move states applied to agent units (allyteam 0) at episode start.
    agent_fire_state: int = FIRE_AT_WILL
    agent_move_state: int = MANOEUVRE

    def __post_init__(self) -> None:
        super().__post_init__()
        for unit, allowed in self.commands.items():
            unknown = set(allowed) - set(COMMANDS)
            if unknown:
                raise ValueError(f"unknown commands for {unit}: {sorted(unknown)}, known: {COMMANDS}")
        if len(self.allyteams) < 2:
            raise ValueError("CombatConfiguration requires at least 2 allyteams.")
        for t, at in enumerate(self.allyteams[1:], start=1):
            if at.behaviour not in OPPONENT_STATES:
                raise ValueError(
                    f"allyteams[{t}].behaviour must be one of {sorted(OPPONENT_STATES)}, got {at.behaviour!r}"
                )

    @property
    def unit_types(self) -> list[str]:
        """Every unit name across all allyteams, sorted: the order of the unit type one-hot in observations."""
        return sorted(set(name for at in self.allyteams for name in at.units))

    @property
    def number_of_agents(self) -> int:
        return sum(self.allyteams[0].units.values())

    def allowed_commands(self, unit_name: str) -> list[str]:
        return self.commands.get(unit_name, self.commands.get("default", list(COMMANDS)))
