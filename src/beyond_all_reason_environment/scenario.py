"""A scenario: what happens in a game (map, teams and alliances, who plays them, how they start), whatever runs it.

The same ScenarioConfiguration serves a series of matches (batch.py, run_match.py, run_matches.py) and the combat
environment (combat_configuration.py). Each team has a controller (controllers.py: DoNothing, Passive,
HoldAndFire, AttackNearest, BuiltInArtificialIntelligence such as BARb, or Agent, played from Python), a faction and
an alliance (allyteam: allied teams share victory and vision). A team either starts normally (a commander and its
economy) or, when ``units`` is given, with exactly these units placed on the map; the game extension's gadget places
them at the start of the match, or at each reset of the environment.
"""

from dataclasses import dataclass, replace

import numpy as np

from .controllers import DEFAULT_STATE, Agent, Controller
from .installation import DEFAULT_MAP
from .start_script import FRAMES_PER_SECOND, ArtificialIntelligencePlayer, Side


@dataclass
class TeamConfiguration:
    """One team: who plays it, its faction, how it starts, and the alliance it belongs to."""

    controller: Controller
    # One faction, or a list: each match of a series then draws one (reproducibly, from the match seed).
    side: Side | list[Side] = "Armada"
    # Unit name -> number: the team starts with exactly these units. None: a normal start (commander).
    units: dict[str, int] | None = None
    # Teams with the same allyteam number are allies: they share victory and vision, and never fight each other.
    # None: an alliance of its own. Numbers only group teams; their values do not matter.
    allyteam: int | None = None

    def player(self, side: Side | None = None) -> ArtificialIntelligencePlayer:
        """The artificial intelligence the engine runs for this team, with ``side`` (default: its own)."""
        return self.controller.player(self.side if side is None else side)


@dataclass
class ScenarioConfiguration:
    """What happens in the game: teams grouped in alliances fight until one alliance is left, or until the time
    limit."""

    teams: list[TeamConfiguration]
    # One map, or a list: the matches of a series then take the maps in turn (match i plays map i % len), so each
    # map gets the same number of matches. Maps must be downloaded first
    # (python scripts/download_game_data.py --map "<name>"); the combat environment needs a single map.
    map_name: str | list[str] = DEFAULT_MAP
    # (width, height) in map squares (multiples of 128): map_name is then a flat map generated with this size
    # the first time it is needed, if game_data/maps/ does not have it yet.
    flat_map_squares: tuple[int, int] | None = None
    # Game time limit: of a match, or of an episode of the combat environment.
    maximum_game_minutes: float = 60.0
    # Every alliance sees the whole map. The combat environment needs it: the engine drops attack orders on units
    # the team does not see.
    full_vision: bool = False

    # Placed units: alliances spread along x, the units of an alliance spread along z (fractions of the map).
    group_separation: float = 0.15
    group_spread: float = 0.06
    spawn_jitter: float = 0.01
    # Spawn zone range in [0, 1]: controls how wide each alliance's spawn zone is along x.
    # At 0.0 each alliance spawns at a single deterministic x (evenly spread across [0, 1]).
    # At 1.0 each alliance occupies its full band of the map width.
    # When None, the classic group_separation / group_spread / spawn_jitter logic is used instead.
    spawn_x_range: float | None = None

    def __post_init__(self) -> None:
        if len(set(self.allyteams)) < 2:
            raise ValueError("a scenario requires at least 2 alliances")
        if not self.map_name:
            raise ValueError("map_name must be a map or a non-empty list of maps")
        if len({team.units is None for team in self.teams}) > 1:
            raise ValueError("either every team has placed units, or none (a normal start)")

    @property
    def allyteams(self) -> list[int]:
        """Alliance of each team, numbered 0, 1, ... in order of first appearance."""
        numbers: dict[object, int] = {}
        for index, team in enumerate(self.teams):
            key = ("own", index) if team.allyteam is None else ("shared", team.allyteam)
            numbers.setdefault(key, len(numbers))
        return [numbers[("own", i) if t.allyteam is None else ("shared", t.allyteam)] for i, t in enumerate(self.teams)]

    @property
    def number_of_allyteams(self) -> int:
        return len(set(self.allyteams))

    @property
    def places_units(self) -> bool:
        """True when the teams start with placed units instead of a commander."""
        return self.teams[0].units is not None

    @property
    def players(self) -> list[ArtificialIntelligencePlayer]:
        return [team.player() for team in self.teams]

    @property
    def agent_teams(self) -> list[int]:
        """Indices of the teams the agents play (Agent controller)."""
        return [index for index, team in enumerate(self.teams) if isinstance(team.controller, Agent)]

    @property
    def has_agents(self) -> bool:
        return bool(self.agent_teams)

    @property
    def uses_game_extension(self) -> bool:
        """Placed units, full vision and scripted controllers come from the gadget of the game extension."""
        return self.places_units or self.full_vision or any(t.controller.gadget_script() for t in self.teams)

    @property
    def unit_types(self) -> list[str]:
        """Every placed unit name across all teams, sorted: the order of the unit type one-hot in observations."""
        return sorted({name for team in self.teams for name in team.units or {}})

    def units_per_allyteam(self) -> list[int]:
        """Number of placed units of each alliance."""
        counts = [0] * self.number_of_allyteams
        for team, allyteam in zip(self.teams, self.allyteams, strict=True):
            counts[allyteam] += sum((team.units or {}).values())
        return counts

    @property
    def maximum_game_frames(self) -> int:
        return int(self.maximum_game_minutes * 60 * FRAMES_PER_SECOND)

    def with_controller(self, index: int, controller: Controller) -> "ScenarioConfiguration":
        """A copy where ``controller`` plays team ``index``: for example the agents' team played by AttackNearest()
        or BARb, to watch or measure the scenario without Python, or Agent() on both sides for self-play."""
        teams = list(self.teams)
        teams[index] = replace(teams[index], controller=controller)
        return replace(self, teams=teams)


def unit_states(scenario: ScenarioConfiguration, agent_states: tuple[int, int]) -> list[int]:
    """Fire and move state of each team's placed units, two values per team (-1: unit default)."""
    states = []
    for team in scenario.teams:
        states.extend(agent_states if isinstance(team.controller, Agent) else team.controller.unit_states())
    return states


def spawn_positions(scenario: ScenarioConfiguration, random: np.random.Generator) -> list[dict]:
    """One spawn entry per placed unit, in team order: {"team", "unit", "x", "z"}.

    Alliances are spread evenly along x; the units of an alliance (all its teams together) are spread along z.

    With spawn_x_range set, alliance a is centred at a / (N-1) along x, and the zone width
    is spawn_x_range / (N-1) on each side (clipped to [0,1]).
    spawn_x_range=0 gives deterministic positions at 0, 1/(N-1), …, 1.
    spawn_x_range=1 gives each alliance its full 1/N slice of the map.

    With spawn_x_range=None, the classic group_separation/group_spread/spawn_jitter logic is used,
    with alliances placed at equal intervals of group_separation around the map centre.
    """
    allyteams = scenario.allyteams
    n_allyteams = scenario.number_of_allyteams
    spread = None if scenario.spawn_x_range is None else scenario.spawn_x_range / (2 * max(n_allyteams - 1, 1))
    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for allyteam in range(n_allyteams):
        # (team, rank of the unit in its team) of every unit of the alliance, in team order.
        members = [
            (team_index, rank)
            for team_index, team in enumerate(scenario.teams)
            if allyteams[team_index] == allyteam
            for rank in range(sum((team.units or {}).values()))
        ]
        for index, member in enumerate(members):
            offset = (index / (len(members) - 1) - 0.5) if len(members) > 1 else 0.0
            if spread is not None:
                base_x = allyteam / max(n_allyteams - 1, 1)
                lo, hi = max(0.0, base_x - spread), min(1.0, base_x + spread)
                x = float(lo if lo == hi else random.uniform(lo, hi))
                jitter_z = float(random.uniform(-1, 1) * scenario.spawn_jitter)
            else:
                centre_x = 0.5 + (allyteam - (n_allyteams - 1) / 2) * scenario.group_separation
                jitter_x, jitter_z = random.uniform(-1, 1, 2) * scenario.spawn_jitter
                x = float(np.clip(centre_x + jitter_x, 0, 1))
            z = float(np.clip(0.5 + offset * scenario.group_spread + jitter_z, 0, 1))
            positions[member] = (x, z)
    return [
        {"team": team_index, "unit": name, "x": positions[(team_index, rank)][0], "z": positions[(team_index, rank)][1]}
        for team_index, team in enumerate(scenario.teams)
        for rank, name in enumerate(name for name, count in (team.units or {}).items() for _ in range(count))
    ]


def controller_mod_options(scenario: ScenarioConfiguration) -> dict[str, str]:
    """The mod option that makes the gadget run the scripted controllers: "<team>,<script>,<parameter>.../..."."""
    scripts = [
        f"{index},{script}"
        for index, team in enumerate(scenario.teams)
        if (script := team.controller.gadget_script()) is not None
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
