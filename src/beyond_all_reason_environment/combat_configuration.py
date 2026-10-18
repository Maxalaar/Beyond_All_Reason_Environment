"""Configuration for combat environments: unit fights with no production or resource gathering.

A CombatConfiguration is made of separate parts, each answering one question:

    scenario     ScenarioConfiguration    what happens in the game: map, armies, start positions, episode length
    observation  ObservationConfiguration what each agent sees
    action       ActionConfiguration      what each agent can do, and how often it decides
    rewards      RewardWeights            how the agents are rewarded
    engine       EngineConfiguration      how the engine runs (window, speed), without effect on the results

Scenarios (beyond_all_reason_environment.scenarios) only define the first part; an experiment (a training
configuration) chooses the scenario and the other parts.
"""

from dataclasses import dataclass, field, replace
from typing import Literal

from .engine_configuration import EngineConfiguration
from .installation import DEFAULT_MAP
from .start_script import FRAMES_PER_SECOND

# Every command an agent may give, in the order of the action index. A unit type only gets the ones listed for it
# in ActionConfiguration.commands; the others are masked.
COMMANDS = ("continue", "move", "fight", "attack", "stop")

# Fire states: 0 hold fire, 1 return fire, 2 fire at will. Move states: 0 hold position, 1 manoeuvre, 2 roam.
FIRE_AT_WILL, HOLD_FIRE = 2, 0
HOLD_POSITION, MANOEUVRE = 0, 1
OPPONENT_STATES = {
    "passive": (HOLD_FIRE, HOLD_POSITION),  # does not move, does not shoot
    "hold_and_fire": (FIRE_AT_WILL, HOLD_POSITION),  # does not move, shoots what comes in range
}


@dataclass
class AllyteamConfiguration:
    """Units and (for non-agent allyteams) behaviour of one allyteam.

    Allyteam 0 is always controlled by the agents; its behaviour field is ignored.
    Allyteams 1+ are NullAI-driven with the specified fire/move state preset:
        passive:       does not move, does not shoot.
        hold_and_fire: does not move, shoots what comes into range.
    """

    units: dict[str, int] = field(default_factory=lambda: {"armpw": 5})
    behaviour: Literal["passive", "hold_and_fire"] = "hold_and_fire"


@dataclass
class ScenarioConfiguration:
    """What happens in the game, independently of the agents: N allyteams fight until one side is eliminated.

    allyteams[0]  — the agents' units (one agent per unit).
    allyteams[1+] — NullAI opponents, with per-allyteam fire/move state presets.
    """

    map_name: str = DEFAULT_MAP
    # (width, height) in map squares (multiples of 128): map_name is then a flat map generated with this size
    # the first time the environment starts, if game_data/maps/ does not have it yet.
    flat_map_squares: tuple[int, int] | None = None
    allyteams: list[AllyteamConfiguration] = field(
        default_factory=lambda: [AllyteamConfiguration(units={"armpw": 5}), AllyteamConfiguration(units={"armpw": 5})]
    )
    maximum_episode_minutes: float = 1.0

    # Spawn: allyteams spread along x, units spread along z (fractions of the map).
    group_separation: float = 0.15
    group_spread: float = 0.06
    spawn_jitter: float = 0.01
    # Spawn zone range in [0, 1]: controls how wide each allyteam's spawn zone is along x.
    # At 0.0 each allyteam spawns at a single deterministic x (evenly spread across [0, 1]).
    # At 1.0 each allyteam occupies its full band of the map width.
    # When None, the classic group_separation / group_spread / spawn_jitter logic is used instead.
    spawn_x_range: float | None = None

    # Every allyteam sees the whole map; required so the engine accepts attack orders on distant units.
    full_vision: bool = True

    def __post_init__(self) -> None:
        if len(self.allyteams) < 2:
            raise ValueError("a scenario requires at least 2 allyteams")
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

    @property
    def maximum_episode_frames(self) -> int:
        return int(self.maximum_episode_minutes * 60 * FRAMES_PER_SECOND)


@dataclass
class ObservationConfiguration:
    """What each agent sees. Entity lists take the nearest entities, up to these counts (padded and masked).

    A count left to None takes every unit the scenario can have in that list (resolved by CombatConfiguration).
    """

    # Actor observation: allies and enemies of all opponent allyteams combined.
    maximum_ally_units: int | None = None
    maximum_enemy_units: int | None = None
    maximum_ally_buildings: int = 0
    maximum_enemy_buildings: int = 0
    # Weapon slots per entity (up to this many weapon tokens, padded).
    maximum_weapons_per_unit: int = 4
    # Critic observation: per-enemy-allyteam budgets (distinct from the combined actor budgets above).
    maximum_units_per_enemy_allyteam: int | None = None
    maximum_buildings_per_enemy_allyteam: int = 0
    # Terrain observation: the map resampled to this fixed (width, height) grid.
    # Channels: normalised height, slope in [0,1), water mask, LoS visibility (1=visible, 0=fog).
    # The first three are constant within an episode; visibility is updated every decision step.
    terrain_resolution: tuple[int, int] = (64, 64)

    def __post_init__(self) -> None:
        if self.terrain_resolution[0] < 1 or self.terrain_resolution[1] < 1:
            raise ValueError("terrain_resolution dimensions must be at least 1")

    def resolved(self, scenario: ScenarioConfiguration) -> "ObservationConfiguration":
        """A copy where every None count is replaced by the number of units the scenario can have there."""
        unit_counts = [sum(at.units.values()) for at in scenario.allyteams]

        def chosen(value: int | None, default: int) -> int:
            return default if value is None else value

        return replace(
            self,
            maximum_ally_units=chosen(self.maximum_ally_units, unit_counts[0] - 1),
            maximum_enemy_units=chosen(self.maximum_enemy_units, sum(unit_counts[1:])),
            maximum_units_per_enemy_allyteam=chosen(self.maximum_units_per_enemy_allyteam, max(unit_counts[1:])),
        )


@dataclass
class ActionConfiguration:
    """What each agent can do, and how often it decides."""

    # Commands allowed per unit name; "default" for the other units.
    commands: dict[str, list[str]] = field(default_factory=lambda: {"default": list(COMMANDS)})
    # "attack" targets the enemy nearest to the clicked point within this radius (game units), otherwise ignored.
    attack_snap_radius: float = 60.0
    # Frames between two decisions (30 frames = 1 game second); the last order goes on in between.
    decision_interval_frames: int = 32
    # Fire and move states of the agents' units, as a player would set them. Manoeuvre (the game's default) lets a
    # unit move to reach the target of an "attack" order; on hold position the engine drops an attack out of range.
    agent_fire_state: int = FIRE_AT_WILL
    agent_move_state: int = MANOEUVRE

    def __post_init__(self) -> None:
        if self.decision_interval_frames < 1:
            raise ValueError("decision_interval_frames must be at least 1")
        for unit, allowed in self.commands.items():
            unknown = set(allowed) - set(COMMANDS)
            if unknown:
                raise ValueError(f"unknown commands for {unit}: {sorted(unknown)}, known: {COMMANDS}")

    def allowed_commands(self, unit_name: str) -> list[str]:
        return self.commands.get(unit_name, self.commands.get("default", list(COMMANDS)))


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
class CombatConfiguration:
    """Everything a CombatEnvironment needs: a scenario, and how the agents see, act and are rewarded in it."""

    scenario: ScenarioConfiguration = field(default_factory=ScenarioConfiguration)
    observation: ObservationConfiguration = field(default_factory=ObservationConfiguration)
    action: ActionConfiguration = field(default_factory=ActionConfiguration)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    engine: EngineConfiguration = field(default_factory=EngineConfiguration)

    def __post_init__(self) -> None:
        # replace() makes a copy: an ObservationConfiguration shared by several configurations stays untouched.
        self.observation = self.observation.resolved(self.scenario)
