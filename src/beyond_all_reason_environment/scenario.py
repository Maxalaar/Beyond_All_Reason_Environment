"""A scenario: what happens in a game (map, allyteams, who plays them, how they start), whatever runs it.

The same ScenarioConfiguration serves a series of matches (batch.py, run_match.py, run_matches.py) and the combat
environment (combat_configuration.py). Each allyteam has a controller (controllers.py: DoNothing, Passive,
HoldAndFire, AttackNearest, BuiltInArtificialIntelligence such as BARb, or Agent, played from Python) and a faction.
An allyteam either starts normally (a commander and its economy) or, when ``units`` is given, with exactly these
units placed on the map; the game extension's gadget places them at the start of the match, or at each reset of the
environment.
"""

from dataclasses import dataclass, replace

import numpy as np

from .controllers import DEFAULT_STATE, Agent, Controller
from .installation import DEFAULT_MAP
from .start_script import FRAMES_PER_SECOND, ArtificialIntelligencePlayer, Side


@dataclass
class AllyteamConfiguration:
    """One allyteam: who plays it, its faction and how it starts."""

    controller: Controller
    # One faction, or a list: each match of a series then draws one (reproducibly, from the match seed).
    side: Side | list[Side] = "Armada"
    # Unit name -> number: the allyteam starts with exactly these units. None: a normal start (commander).
    units: dict[str, int] | None = None

    def player(self, side: Side | None = None) -> ArtificialIntelligencePlayer:
        """The artificial intelligence the engine runs for this allyteam, with ``side`` (default: its own)."""
        return self.controller.player(self.side if side is None else side)


@dataclass
class ScenarioConfiguration:
    """What happens in the game: N allyteams fight until one side is eliminated, or until the time limit."""

    allyteams: list[AllyteamConfiguration]
    # One map, or a list: the matches of a series then take the maps in turn (match i plays map i % len), so each
    # map gets the same number of matches. Maps must be downloaded first
    # (python scripts/download_game_data.py --map "<name>"); the combat environment needs a single map.
    map_name: str | list[str] = DEFAULT_MAP
    # (width, height) in map squares (multiples of 128): map_name is then a flat map generated with this size
    # the first time it is needed, if game_data/maps/ does not have it yet.
    flat_map_squares: tuple[int, int] | None = None
    # Game time limit: of a match, or of an episode of the combat environment.
    maximum_game_minutes: float = 60.0
    # Every allyteam sees the whole map. The combat environment needs it: the engine drops attack orders on units
    # the team does not see.
    full_vision: bool = False

    # Placed units: allyteams spread along x, units spread along z (fractions of the map).
    group_separation: float = 0.15
    group_spread: float = 0.06
    spawn_jitter: float = 0.01
    # Spawn zone range in [0, 1]: controls how wide each allyteam's spawn zone is along x.
    # At 0.0 each allyteam spawns at a single deterministic x (evenly spread across [0, 1]).
    # At 1.0 each allyteam occupies its full band of the map width.
    # When None, the classic group_separation / group_spread / spawn_jitter logic is used instead.
    spawn_x_range: float | None = None

    def __post_init__(self) -> None:
        if len(self.allyteams) < 2:
            raise ValueError("a scenario requires at least 2 allyteams")
        if not self.map_name:
            raise ValueError("map_name must be a map or a non-empty list of maps")
        if len({allyteam.units is None for allyteam in self.allyteams}) > 1:
            raise ValueError("either every allyteam has placed units, or none (a normal start)")

    @property
    def places_units(self) -> bool:
        """True when the allyteams start with placed units instead of a commander."""
        return self.allyteams[0].units is not None

    @property
    def players(self) -> list[ArtificialIntelligencePlayer]:
        return [allyteam.player() for allyteam in self.allyteams]

    @property
    def has_agents(self) -> bool:
        return any(isinstance(allyteam.controller, Agent) for allyteam in self.allyteams)

    @property
    def uses_game_extension(self) -> bool:
        """Placed units, full vision and scripted controllers come from the gadget of the game extension."""
        return self.places_units or self.full_vision or any(a.controller.gadget_script() for a in self.allyteams)

    @property
    def unit_types(self) -> list[str]:
        """Every placed unit name across all allyteams, sorted: the order of the unit type one-hot in observations."""
        return sorted({name for allyteam in self.allyteams for name in allyteam.units or {}})

    @property
    def maximum_game_frames(self) -> int:
        return int(self.maximum_game_minutes * 60 * FRAMES_PER_SECOND)

    def with_controller(self, index: int, controller: Controller) -> "ScenarioConfiguration":
        """A copy where ``controller`` plays allyteam ``index``: for example the agents' allyteam played by
        AttackNearest() or BARb, to watch or measure the scenario without Python."""
        allyteams = list(self.allyteams)
        allyteams[index] = replace(allyteams[index], controller=controller)
        return replace(self, allyteams=allyteams)


def unit_states(scenario: ScenarioConfiguration, agent_states: tuple[int, int]) -> list[int]:
    """Fire and move state of each allyteam's placed units, two values per allyteam (-1: unit default)."""
    states = []
    for allyteam in scenario.allyteams:
        is_agent = isinstance(allyteam.controller, Agent)
        states.extend(agent_states if is_agent else allyteam.controller.unit_states())
    return states


def spawn_positions(scenario: ScenarioConfiguration, random: np.random.Generator) -> list[dict]:
    """One spawn entry per placed unit across all allyteams, allyteam 0 first.

    Allyteams are spread evenly along x; units within each allyteam are spread along z.

    With spawn_x_range set, allyteam t is centred at t / (N-1) along x, and the zone width
    is spawn_x_range / (N-1) on each side (clipped to [0,1]).
    spawn_x_range=0 gives deterministic positions at 0, 1/(N-1), …, 1.
    spawn_x_range=1 gives each allyteam its full 1/N slice of the map.

    With spawn_x_range=None, the classic group_separation/group_spread/spawn_jitter logic is used,
    with allyteams placed at equal intervals of group_separation around the map centre.
    """
    n_allyteams = len(scenario.allyteams)
    spawns = []
    x_range = scenario.spawn_x_range

    if x_range is not None:
        spread = x_range / (2 * max(n_allyteams - 1, 1))
        for t, at in enumerate(scenario.allyteams):
            base_x = t / max(n_allyteams - 1, 1)
            lo = max(0.0, base_x - spread)
            hi = min(1.0, base_x + spread)
            names = [name for name, count in (at.units or {}).items() for _ in range(count)]
            for index, name in enumerate(names):
                x = float(lo if lo == hi else random.uniform(lo, hi))
                offset = (index / (len(names) - 1) - 0.5) if len(names) > 1 else 0.0
                jitter_z = float(random.uniform(-1, 1) * scenario.spawn_jitter)
                z = float(np.clip(0.5 + offset * scenario.group_spread + jitter_z, 0, 1))
                spawns.append({"team": t, "unit": name, "x": x, "z": z})
    else:
        for t, at in enumerate(scenario.allyteams):
            centre_x = 0.5 + (t - (n_allyteams - 1) / 2) * scenario.group_separation
            names = [name for name, count in (at.units or {}).items() for _ in range(count)]
            for index, name in enumerate(names):
                offset = (index / (len(names) - 1) - 0.5) if len(names) > 1 else 0.0
                jitter_x, jitter_z = random.uniform(-1, 1, 2) * scenario.spawn_jitter
                spawns.append(
                    {
                        "team": t,
                        "unit": name,
                        "x": float(np.clip(centre_x + jitter_x, 0, 1)),
                        "z": float(np.clip(0.5 + offset * scenario.group_spread + jitter_z, 0, 1)),
                    }
                )
    return spawns


def controller_mod_options(scenario: ScenarioConfiguration) -> dict[str, str]:
    """The mod option that makes the gadget run the scripted controllers: "<allyteam>,<script>,<parameter>.../..."."""
    scripts = [
        f"{index},{script}"
        for index, allyteam in enumerate(scenario.allyteams)
        if (script := allyteam.controller.gadget_script()) is not None
    ]
    return {"reinforcement_learning_controllers": "/".join(scripts)} if scripts else {}


def scenario_mod_options(scenario: ScenarioConfiguration, seed: int) -> dict[str, str]:
    """Mod options that make the game extension's gadget set the scenario up in a match without Python.

    Placed units: spawn positions are drawn from ``seed``, so a match is reproducible, and the game ends when every
    unit of a side is destroyed (deathmode killall); a normal start keeps the game's own end condition. Scripted
    controllers and full vision apply in both cases.
    """
    options = {
        "reinforcement_learning_full_vision": "1" if scenario.full_vision else "0",
        **controller_mod_options(scenario),
    }
    if not scenario.places_units:
        return options
    spawns = spawn_positions(scenario, np.random.default_rng(seed))
    states = unit_states(scenario, (DEFAULT_STATE, DEFAULT_STATE))
    spawns_text = "/".join(f"{spawn['team']},{spawn['unit']},{spawn['x']:.6f},{spawn['z']:.6f}" for spawn in spawns)
    options["reinforcement_learning_scenario"] = f"{spawns_text}|{','.join(map(str, states))}"
    options["deathmode"] = "killall"
    return options
