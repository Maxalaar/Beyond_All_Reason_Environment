"""Configuration of the reinforcement learning environment (ReinforcementLearningEnvironment: unit fights, no
production or resource gathering for now).

A ReinforcementLearningEnvironmentConfiguration is made of separate parts, each answering one question:

    game         GameConfiguration        the game: a scenario (map, teams, alliances, placed units, start
                                          positions, episode length) and who plays each team, Agent() for the
                                          agents (game.py, shared with the runs of games)
    observation  ObservationConfiguration what each agent sees
    action       ActionConfiguration      what each agent can do, and how often it decides
    rewards      RewardWeights            how the agents are rewarded
    engine       EngineConfiguration      how the engine runs (window, speed), without effect on the results

Scenarios (beyond_all_reason_environment.scenarios) only define the situation; an experiment (a training
configuration) chooses the scenario, the controllers and the other parts.
"""

from dataclasses import dataclass, field, replace

from .controllers import FIRE_AT_WILL, MANOEUVRE, Agent, HoldAndFire
from .engine_configuration import EngineConfiguration
from .game import GameConfiguration
from .scenarios.five_versus_five import five_versus_five

# Every command an agent may give, in the order of the action index. A unit type only gets the ones listed for it
# in ActionConfiguration.commands; the others are masked.
COMMANDS = ("continue", "move", "fight", "attack", "stop")


@dataclass
class ObservationConfiguration:
    """What each agent sees. Entity lists take the nearest entities, up to these counts (padded and masked).

    A count left to None takes every unit the game can have in that list (resolved by
    ReinforcementLearningEnvironmentConfiguration).
    """

    # Actor observation: units of the agent's alliance (other than its own), and of all enemy alliances combined.
    maximum_ally_units: int | None = None
    maximum_enemy_units: int | None = None
    maximum_ally_buildings: int = 0
    maximum_enemy_buildings: int = 0
    # Weapon slots per entity (up to this many weapon tokens, padded).
    maximum_weapons_per_unit: int = 4
    # Critic observation: budgets per enemy alliance (distinct from the combined actor budgets above).
    maximum_units_per_enemy_allyteam: int | None = None
    maximum_buildings_per_enemy_allyteam: int = 0
    # Terrain observation: the map resampled to this fixed (width, height) grid.
    # Channels: normalised height, slope in [0,1), water mask, and the visibility of the agent's alliance (1 where
    # one of its units sees: within sight range, without terrain occlusion). The first three are constant within an
    # episode; visibility is updated every decision step.
    terrain_resolution: tuple[int, int] = (64, 64)

    def __post_init__(self) -> None:
        if self.terrain_resolution[0] < 1 or self.terrain_resolution[1] < 1:
            raise ValueError("terrain_resolution dimensions must be at least 1")

    def resolved(self, game: GameConfiguration) -> "ObservationConfiguration":
        """A copy where every None count is replaced by the number of units the game can have there."""
        counts = game.scenario.units_per_allyteam()
        # The largest list any agent can see: over the alliances that have agents.
        agent_allyteams = {game.scenario.allyteams[team] for team in game.agent_teams}

        def chosen(value: int | None, default: int) -> int:
            return default if value is None else value

        return replace(
            self,
            maximum_ally_units=chosen(self.maximum_ally_units, max(counts[a] for a in agent_allyteams) - 1),
            maximum_enemy_units=chosen(self.maximum_enemy_units, max(sum(counts) - counts[a] for a in agent_allyteams)),
            maximum_units_per_enemy_allyteam=chosen(self.maximum_units_per_enemy_allyteam, max(counts)),
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
    """Reward of each step, shared by the agents of an alliance. Damages are fractions of the total health of the
    side (the alliance, or all its enemies)."""

    damage_dealt: float = 1.0
    damage_received: float = 1.0
    # Per enemy destroyed, divided by the number of enemies.
    kill: float = 0.5
    victory: float = 1.0
    defeat: float = -1.0


@dataclass
class ReinforcementLearningEnvironmentConfiguration:
    """Everything a ReinforcementLearningEnvironment needs: a game, and how the agents see, act and are rewarded in
    it."""

    game: GameConfiguration = field(
        default_factory=lambda: GameConfiguration(five_versus_five, controllers=[Agent(), HoldAndFire()])
    )
    observation: ObservationConfiguration = field(default_factory=ObservationConfiguration)
    action: ActionConfiguration = field(default_factory=ActionConfiguration)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    engine: EngineConfiguration = field(default_factory=EngineConfiguration)

    def __post_init__(self) -> None:
        scenario = self.game.scenario
        if not self.game.agent_teams:
            raise ValueError("the environment needs at least one team played by the agents (controllers.Agent)")
        if not scenario.places_units:
            raise ValueError("the environment needs placed units (team units), not a normal start")
        if not isinstance(scenario.map_name, str):
            raise ValueError("the environment plays a single map, not a list")
        if not scenario.full_vision:
            raise ValueError(
                "the environment needs scenario.full_vision: the engine drops attack orders on units the team does "
                "not see"
            )
        for team in scenario.teams:
            if not isinstance(team.side, str):
                raise ValueError(f"the environment needs one faction per team, not {team.side!r}")
        for team_index in self.game.agent_teams:
            if scenario.teams[team_index].has_drawn_units:
                raise ValueError(
                    f"team {team_index} is played by the agents: its numbers of units must be fixed (the agents are "
                    "its units); draw the numbers of the other teams"
                )
        # replace() makes a copy: an ObservationConfiguration shared by several configurations stays untouched.
        self.observation = self.observation.resolved(self.game)
