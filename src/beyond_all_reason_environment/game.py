"""A game: a scenario and who plays each of its teams (one controller per team, controllers.py).

The same GameConfiguration runs as games without Python (RunConfiguration, run.py) or as the reinforcement learning
environment (ReinforcementLearningEnvironmentConfiguration), where the teams with an Agent() controller are played
from Python.
"""

import base64
import json
from dataclasses import dataclass, replace

import numpy as np

from .controllers import DEFAULT_STATE, Agent, Controller
from .scenario import ScenarioConfiguration, spawn_positions
from .start_script import ArtificialIntelligencePlayer, Side


@dataclass
class GameConfiguration:
    """A scenario, and the controller of each of its teams, in the order of scenario.teams."""

    scenario: ScenarioConfiguration
    controllers: list[Controller]

    def __post_init__(self) -> None:
        if len(self.controllers) != len(self.scenario.teams):
            raise ValueError(
                f"one controller per team: {len(self.controllers)} controllers for {len(self.scenario.teams)} teams"
            )

    def players(self, sides: list[Side] | None = None) -> list[ArtificialIntelligencePlayer]:
        """The artificial intelligence the engine runs for each team, with ``sides`` (default: the teams' own)."""
        if sides is None:
            sides = [team.side for team in self.scenario.teams]  # pyright: ignore[reportAssignmentType]
        return [controller.player(side) for controller, side in zip(self.controllers, sides, strict=True)]

    @property
    def agent_teams(self) -> list[int]:
        """Indices of the teams the agents play (Agent controller)."""
        return [index for index, controller in enumerate(self.controllers) if isinstance(controller, Agent)]

    @property
    def has_agents(self) -> bool:
        return bool(self.agent_teams)

    @property
    def uses_game_extension(self) -> bool:
        """Placed units, full vision and scripted controllers come from the gadget of the game extension."""
        scenario = self.scenario
        return scenario.places_units or scenario.full_vision or any(c.gadget_script() for c in self.controllers)

    def with_controller(self, index: int, controller: Controller) -> "GameConfiguration":
        """A copy where ``controller`` plays team ``index``."""
        controllers = list(self.controllers)
        controllers[index] = controller
        return replace(self, controllers=controllers)


def unit_states(game: GameConfiguration, agent_states: tuple[int, int]) -> list[int]:
    """Fire and move state of each team's placed units, two values per team (-1: unit default)."""
    states = []
    for controller in game.controllers:
        states.extend(agent_states if isinstance(controller, Agent) else controller.unit_states())
    return states


def controller_mod_options(game: GameConfiguration) -> dict[str, str]:
    """The mod option that makes the gadget run the scripted controllers: "<team>,<script>,<parameter>.../..."."""
    scripts = [
        f"{index},{script}"
        for index, controller in enumerate(game.controllers)
        if (script := controller.gadget_script()) is not None
    ]
    return {"reinforcement_learning_controllers": "/".join(scripts)} if scripts else {}


def team_color_mod_options(scenario: ScenarioConfiguration) -> dict[str, str]:
    """The mod option that makes the gadget colour the teams that have a colour: "<team>,<red>,<green>,<blue>/..."."""
    colors = [
        f"{index},{','.join(map(str, team.color))}"
        for index, team in enumerate(scenario.teams)
        if team.color is not None
    ]
    return {"reinforcement_learning_team_colors": "/".join(colors)} if colors else {}


def team_name_mod_options(game: GameConfiguration) -> dict[str, str]:
    """The mod option that makes the gadget name the teams after their controllers: "<team>,<name>/...". A label
    that several teams share (Agent in self-play) is followed by the team's number, from 1."""
    labels = [controller.label for controller in game.controllers]
    names = [f"{label} {index + 1}" if labels.count(label) > 1 else label for index, label in enumerate(labels)]
    return {"reinforcement_learning_team_names": "/".join(f"{index},{name}" for index, name in enumerate(names))}


# Names no unit: BAR's loadout gadget (unit_scenario_loadout.lua) skips it, so the loadout creates nothing.
LOADOUT_PLACEHOLDER = "reinforcement_learning_placeholder"


def no_commander_mod_options() -> dict[str, str]:
    """BAR's own scenario mod option, so that no commander is created: game_initial_spawn.lua spawns a unit loadout
    instead of the commanders, without their spawn effects, and the game extension's gadget places the scenario's
    units. The loadout only holds a placeholder: BAR wants positions in game units, the map size is known only in the
    game. Encoded as BAR expects (modoption_payload.lua: base64url of JSON; padding is optional there, and dropped
    because "=" is risky in a start-script value). The faction picker is turned off, there is no commander to pick.
    """
    payload = {
        "unitloadout": [{"name": LOADOUT_PLACEHOLDER, "x": 0, "z": 0, "rot": 0, "team": 0}],
        "disablefactionpicker": True,
    }
    return {"scenariooptions": base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")}


def game_mod_options(game: GameConfiguration, seed: int) -> dict[str, str]:
    """Mod options that make the game extension's gadget set the game up without Python.

    Placed units: spawn positions are drawn from ``seed``, so a game is reproducible, and the game ends when every
    unit of a side is destroyed (deathmode killall), or its commanders with scenario.commander_death_ends_game
    (deathmode com); a normal start keeps the game's own end condition. Scripted
    controllers, team colours and names, and full vision apply in both cases.
    """
    scenario = game.scenario
    options = {
        "reinforcement_learning_full_vision": "1" if scenario.full_vision else "0",
        **controller_mod_options(game),
        **team_color_mod_options(scenario),
        **team_name_mod_options(game),
    }
    if not scenario.places_units:
        return options
    spawns = spawn_positions(scenario, np.random.default_rng(seed))
    states = unit_states(game, (DEFAULT_STATE, DEFAULT_STATE))
    spawns_text = "/".join(f"{spawn['team']},{spawn['unit']},{spawn['x']:.6f},{spawn['z']:.6f}" for spawn in spawns)
    options["reinforcement_learning_scenario"] = f"{spawns_text}|{','.join(map(str, states))}"
    options["deathmode"] = "com" if scenario.commander_death_ends_game else "killall"
    return options | no_commander_mod_options()
