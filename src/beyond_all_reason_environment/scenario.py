"""A scenario: the situation of a game (map, teams and alliances, how they start), whoever plays it.

A scenario says nothing of who plays each team: GameConfiguration (game.py) adds one controller per team. Each team
has a faction and an alliance (allyteam: allied teams share victory and vision). A team either starts normally (a
commander and its economy) or, when ``units`` is given, with exactly these units placed on the map; the game
extension's gadget places them at the start of the game, or at each reset of the reinforcement learning environment.
"""

from dataclasses import dataclass

import numpy as np

from .installation import DEFAULT_MAP
from .start_script import FRAMES_PER_SECOND, Side

# Distance kept from the map border by the default spawn zones (fraction of the map), so no unit starts on the edge.
SPAWN_MARGIN = 0.02


@dataclass
class SpawnZone:
    """A rectangle of the map where a team's units start, each drawn uniformly at random inside it.

    Fractions of the map: x from the west edge (0) to the east edge (1), z from the north edge (0) to the south edge
    (1). A minimum equal to its maximum makes a line, or a point: deterministic positions.
    """

    x_minimum: float = 0.0
    x_maximum: float = 1.0
    z_minimum: float = 0.0
    z_maximum: float = 1.0

    def __post_init__(self) -> None:
        if not (0 <= self.x_minimum <= self.x_maximum <= 1 and 0 <= self.z_minimum <= self.z_maximum <= 1):
            raise ValueError(f"a spawn zone needs 0 <= minimum <= maximum <= 1 on x and z: {self}")


@dataclass
class TeamConfiguration:
    """One team: its faction, how it starts (and where), and the alliance it belongs to."""

    # One faction, or a list: each game of a run then draws one (reproducibly, from the game's seed).
    side: Side | list[Side] = "Armada"
    # Unit name -> number: the team starts with exactly these units, or (minimum, maximum): a number drawn uniformly
    # for each game (each episode of the environment), with the spawn positions. None: a normal start (commander).
    units: dict[str, int | tuple[int, int]] | None = None
    # Teams with the same allyteam number are allies: they share victory and vision, and never fight each other.
    # None: an alliance of its own. Numbers only group teams; their values do not matter.
    allyteam: int | None = None
    # Where the placed units start, in [0, 1]: the share of the team's alliance band taken from the map edge. The
    # alliances split the map width into equal bands, in order (two alliances: the west half and the east half);
    # 1 fills the band, 0.2 only the fifth of it nearest the edge (a middle band is taken around its centre). The zone
    # covers the whole map height, SPAWN_MARGIN apart from the border.
    spawn_depth: float = 1.0
    # An explicit zone instead, for full control (spawn_depth is then unused).
    spawn_zone: SpawnZone | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.spawn_depth <= 1:
            raise ValueError(f"spawn_depth must be in [0, 1], not {self.spawn_depth}")
        for name, count in (self.units or {}).items():
            minimum, maximum = count if isinstance(count, tuple) else (count, count)
            if not 1 <= minimum <= maximum:
                raise ValueError(f"{name}: a number of units, or (minimum, maximum) with 1 <= minimum <= maximum")

    @property
    def has_drawn_units(self) -> bool:
        """True when the number of some of its units is drawn for each game ((minimum, maximum))."""
        return any(isinstance(count, tuple) for count in (self.units or {}).values())


@dataclass
class ScenarioConfiguration:
    """What happens in the game: teams grouped in alliances fight until one alliance is left, or until the time
    limit."""

    teams: list[TeamConfiguration]
    # One map, or a list: the games of a run then take the maps in turn (match i plays map i % len), so each
    # map gets the same number of matches. Maps must be downloaded first
    # (python scripts/download_game_data.py --map "<name>"); the reinforcement learning environment needs a single map.
    map_name: str | list[str] = DEFAULT_MAP
    # (width, height) in map squares (multiples of 128): map_name is then a flat map generated with this size
    # the first time it is needed, if game_data/maps/ does not have it yet.
    flat_map_squares: tuple[int, int] | None = None
    # Game time limit: of a game, or of an episode of the reinforcement learning environment.
    maximum_game_minutes: float = 60.0
    # Every alliance sees the whole map. The reinforcement learning environment needs it: the engine drops attack
    # orders on units the team does not see.
    full_vision: bool = False

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
    def unit_types(self) -> list[str]:
        """Every placed unit name across all teams, sorted: the order of the unit type one-hot in observations."""
        return sorted({name for team in self.teams for name in team.units or {}})

    def units_per_allyteam(self) -> list[int]:
        """Number of placed units of each alliance; the maximum when it is drawn for each game."""
        counts = [0] * self.number_of_allyteams
        for team, allyteam in zip(self.teams, self.allyteams, strict=True):
            counts[allyteam] += sum(maximum_count(count) for count in (team.units or {}).values())
        return counts

    def spawn_zone(self, team_index: int) -> SpawnZone:
        """Where the placed units of team ``team_index`` start: its spawn_zone, or the zone of its spawn_depth."""
        team = self.teams[team_index]
        if team.spawn_zone is not None:
            return team.spawn_zone
        allyteam, number = self.allyteams[team_index], self.number_of_allyteams
        band_minimum, band_maximum = allyteam / number, (allyteam + 1) / number
        width = (band_maximum - band_minimum - SPAWN_MARGIN) * team.spawn_depth
        if allyteam == 0:  # anchored on the west edge
            x_minimum = band_minimum + SPAWN_MARGIN
            x_maximum = x_minimum + width
        elif allyteam == number - 1:  # anchored on the east edge
            x_maximum = band_maximum - SPAWN_MARGIN
            x_minimum = x_maximum - width
        else:  # a middle band: around its centre
            centre = (band_minimum + band_maximum) / 2
            x_minimum, x_maximum = centre - width / 2, centre + width / 2
        return SpawnZone(x_minimum, x_maximum, SPAWN_MARGIN, 1 - SPAWN_MARGIN)

    @property
    def maximum_game_frames(self) -> int:
        return int(self.maximum_game_minutes * 60 * FRAMES_PER_SECOND)


def maximum_count(count: int | tuple[int, int]) -> int:
    """The largest number of units a count of TeamConfiguration.units can give."""
    return count[1] if isinstance(count, tuple) else count


def spawn_positions(scenario: ScenarioConfiguration, random: np.random.Generator) -> list[dict]:
    """One spawn entry per placed unit, in team order: {"team", "unit", "x", "z"}.

    Each unit is drawn uniformly at random in its team's spawn zone (ScenarioConfiguration.spawn_zone), and a number
    of units given as (minimum, maximum) uniformly in it, from ``random``: the same seed gives the same game.
    """
    spawns = []
    for team_index, team in enumerate(scenario.teams):
        zone = scenario.spawn_zone(team_index)
        for name, count in (team.units or {}).items():
            if isinstance(count, tuple):
                count = int(random.integers(count[0], count[1] + 1))
            for _ in range(count):
                x = float(random.uniform(zone.x_minimum, zone.x_maximum))
                z = float(random.uniform(zone.z_minimum, zone.z_maximum))
                spawns.append({"team": team_index, "unit": name, "x": x, "z": z})
    return spawns
