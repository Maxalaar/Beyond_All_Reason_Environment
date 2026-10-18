"""Configuration for combat environments: unit fights with no production or resource gathering.

A CombatConfiguration is made of separate parts, each answering one question:

    scenario     ScenarioConfiguration    what happens in the game: map, allyteams and who plays them, placed units,
                                          start positions, episode length (scenario.py, shared with the matches)
    observation  ObservationConfiguration what each agent sees
    action       ActionConfiguration      what each agent can do, and how often it decides
    rewards      RewardWeights            how the agents are rewarded
    engine       EngineConfiguration      how the engine runs (window, speed), without effect on the results

Scenarios (beyond_all_reason_environment.scenarios) only define the first part; an experiment (a training
configuration) chooses the scenario and the other parts.
"""

from dataclasses import dataclass, field, replace

from .engine_configuration import EngineConfiguration
from .scenario import FIRE_AT_WILL, MANOEUVRE, ScenarioConfiguration, is_agent
from .scenarios.five_versus_five import five_versus_five

# Every command an agent may give, in the order of the action index. A unit type only gets the ones listed for it
# in ActionConfiguration.commands; the others are masked.
COMMANDS = ("continue", "move", "fight", "attack", "stop")


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
        unit_counts = [sum((at.units or {}).values()) for at in scenario.allyteams]

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

    scenario: ScenarioConfiguration = field(default_factory=lambda: five_versus_five)
    observation: ObservationConfiguration = field(default_factory=ObservationConfiguration)
    action: ActionConfiguration = field(default_factory=ActionConfiguration)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    engine: EngineConfiguration = field(default_factory=EngineConfiguration)

    def __post_init__(self) -> None:
        scenario = self.scenario
        if not is_agent(scenario.allyteams[0].player) or any(is_agent(p) for p in scenario.players[1:]):
            raise ValueError("the agents (scenario.AGENT) play allyteam 0, and only it")
        if not scenario.places_units:
            raise ValueError("the combat environment needs placed units (allyteam units), not a normal start")
        if not isinstance(scenario.map_name, str):
            raise ValueError("the combat environment plays a single map, not a list")
        if not scenario.full_vision:
            raise ValueError(
                "the combat environment needs scenario.full_vision: the engine drops attack orders on units the "
                "team does not see"
            )
        for player in scenario.players:
            if not isinstance(player.side, str):
                raise ValueError(f"the combat environment needs one faction per player, not {player.side!r}")
        # replace() makes a copy: an ObservationConfiguration shared by several configurations stays untouched.
        self.observation = self.observation.resolved(self.scenario)
