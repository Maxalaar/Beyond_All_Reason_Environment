"""Run several matches in parallel, described by a Python configuration file, and summarise them."""

import argparse
import importlib.util
import json
import random
import re
import shutil
import statistics
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import get_type_hints

from .game_extension import extension_game_name
from .installation import GAME_VERSION, REPOSITORY_ROOT
from .match import MatchResult, read_match_result, run_match
from .scenario import ScenarioConfiguration, scenario_mod_options
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration, Side

# Final engine statistics averaged per player in the summary.
SUMMARISED_STATISTICS = [
    "metalProduced",
    "energyProduced",
    "damageDealt",
    "damageReceived",
    "unitsProduced",
    "unitsKilled",
    "unitsDied",
    "unitsAlive",
]


@dataclass
class BatchConfiguration:
    """A series of matches of one scenario. Files in configurations/matches/ define one, named ``configuration``."""

    # Also the name of the results folder: runs/<name>/version_<n>/, a new version at each run.
    name: str
    # What happens in each match: map(s), players, how they start, time limit (see scenarios/).
    scenario: ScenarioConfiguration
    matches: int = 5
    # Matches at the same time, about 5 GB of memory each.
    parallel: int = 4
    # Match i uses seed first_seed + i.
    first_seed: int = 1
    # Save a replay of each match (a few hundred kB each), to watch with scripts/watch_replay.py.
    record_replays: bool = False
    # Keep everything the engine writes in each match folder (about 5 MB), for debugging.
    keep_engine_files: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError(f"name {self.name!r} may only contain letters, digits, _ and -")
        if self.matches < 1 or self.parallel < 1:
            raise ValueError("matches and parallel must be at least 1")
        if self.scenario.number_of_allyteams != len(self.scenario.teams):
            raise ValueError(
                "match statistics count one team per alliance: give each team an alliance of its own (allyteam=None)"
            )
        if self.scenario.has_agents:
            raise ValueError(
                "nobody plays the agents in a match without Python: give their team another controller, "
                "scenario.with_controller(0, AttackNearest()) for example"
            )

    def match_configurations(self) -> list[MatchConfiguration]:
        """One configuration per match; players with several factions get one drawn from their list."""
        return [self.match_configuration(index) for index in range(self.matches)]

    def match_configuration(self, index: int) -> MatchConfiguration:
        """Match ``index`` of the series: seed first_seed + index, map index % number of maps.

        ``index`` may go beyond ``matches``, to run one more match of the same series.
        """
        seed = self.first_seed + index
        scenario = self.scenario
        maps = scenario.map_name
        uses_game_extension = scenario.uses_game_extension
        return MatchConfiguration(
            players=[team.player(_draw_side(team.side, seed, number)) for number, team in enumerate(scenario.teams)],
            map_name=maps if isinstance(maps, str) else maps[index % len(maps)],
            flat_map_squares=scenario.flat_map_squares,
            game_version=extension_game_name() if uses_game_extension else GAME_VERSION,
            seed=seed,
            maximum_game_minutes=scenario.maximum_game_minutes,
            record_replay=self.record_replays,
            keep_engine_files=self.keep_engine_files,
            extra_mod_options=scenario_mod_options(scenario, seed) if uses_game_extension else {},
        )


def _draw_side(side: Side | list[Side], seed: int, number: int) -> Side:
    """The faction of team ``number`` in the match with this seed; the same seed always draws the same one."""
    if isinstance(side, str):
        return side
    return random.Random(f"{seed}:{number}").choice(side)


VERSION_PREFIX = "version_"


def new_version_directory(name_directory: Path) -> Path:
    """runs/<name>/version_<n>/ with n one more than the highest existing version, created empty.

    Running a configuration again never touches earlier results: each run is a new version.
    """
    numbers = [
        int(path.name.removeprefix(VERSION_PREFIX))
        for path in name_directory.glob(f"{VERSION_PREFIX}*")
        if path.name.removeprefix(VERSION_PREFIX).isdigit()
    ]
    directory = name_directory / f"{VERSION_PREFIX}{max(numbers, default=0) + 1}"
    directory.mkdir(parents=True)
    return directory


def load_batch_configuration(path: Path) -> BatchConfiguration:
    """Execute a configuration file and return its ``configuration`` variable."""
    specification = importlib.util.spec_from_file_location(f"configuration_{Path(path).stem}", path)
    if specification is None or specification.loader is None:
        raise ValueError(f"{path} is not a Python file")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    configuration = getattr(module, "configuration", None)
    if not isinstance(configuration, BatchConfiguration):
        raise ValueError(f"{path} must define `configuration = BatchConfiguration(...)`")
    return configuration


# Used by run_match.py and run_matches.py when no configuration file is given.
DEFAULT_CONFIGURATION_FILE = REPOSITORY_ROOT / "configurations" / "matches" / "barb_versus_barb.py"
# Fields that a command line option cannot replace: they stay in the configuration file.
NOT_OVERRIDABLE_FIELDS = {"scenario", "teams", "flat_map_squares"}


def _overridable_fields() -> dict[str, tuple[str, object]]:
    """Field name -> (owner, type) for every field with a command line option; owner is "batch" or "scenario".

    The scenario's fields are offered next to the series' own (--map_name, --maximum_game_minutes...).
    """
    overridable = {}
    for owner, cls in (("batch", BatchConfiguration), ("scenario", ScenarioConfiguration)):
        hints = get_type_hints(cls)
        for field in fields(cls):
            if field.name not in NOT_OVERRIDABLE_FIELDS:
                overridable[field.name] = (owner, hints[field.name])
    return overridable


def add_configuration_arguments(parser: argparse.ArgumentParser) -> None:
    """The configuration file, optional, and one --<field> option per field to replace, of the series or of its
    scenario.

    Options take the exact field names (--map_name). Booleans also have a --no-<field> form; map_name takes
    one map or several.
    """
    parser.add_argument(
        "configuration",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIGURATION_FILE,
        help=f"Python configuration file, see configurations/matches/ (default: {DEFAULT_CONFIGURATION_FILE.name})",
    )
    group = parser.add_argument_group("replace a field of the configuration file or of its scenario")
    for name, (_, annotation) in _overridable_fields().items():
        options: dict = {"dest": f"override_{name}", "default": None}
        if annotation is bool:
            options["action"] = argparse.BooleanOptionalAction
        elif annotation in (int, float, str, float | None):
            value_type = float if annotation == float | None else annotation
            options["type"] = value_type
            options["metavar"] = value_type.__name__  # pyright: ignore[reportAttributeAccessIssue]
        elif annotation == str | list[str]:
            options["nargs"] = "+"
            options["metavar"] = "str"
        else:
            raise TypeError(f"no command line option for the type of {name}: {annotation}")
        group.add_argument(f"--{name}", **options)


def configuration_from_arguments(arguments: argparse.Namespace) -> BatchConfiguration:
    """The configuration file of ``arguments``, with the fields given on the command line replaced."""
    configuration = load_batch_configuration(arguments.configuration)
    overrides: dict[str, dict] = {"batch": {}, "scenario": {}}
    for name, (owner, _) in _overridable_fields().items():
        value = getattr(arguments, f"override_{name}")
        if value is None:
            continue
        # nargs="+" always gives a list; one map stays a plain name.
        overrides[owner][name] = value[0] if isinstance(value, list) and len(value) == 1 else value
    # replace() runs __post_init__ again, so replaced values are checked like the ones of the file.
    scenario = replace(configuration.scenario, **overrides["scenario"])
    return replace(configuration, scenario=scenario, **overrides["batch"])


def save_configuration(batch_directory: Path, configuration: BatchConfiguration, arguments: argparse.Namespace) -> None:
    """configuration.py (copy of the file) and configuration.json (the values used, after the command line)."""
    shutil.copy(arguments.configuration, batch_directory / "configuration.py")
    content = {
        "configuration_file": str(arguments.configuration),
        "command_line": sys.argv[1:],
        "configuration": asdict(configuration),
    }
    (batch_directory / "configuration.json").write_text(json.dumps(content, indent=2))


def run_matches(
    configurations: list[MatchConfiguration],
    batch_directory: Path,
    parallel: int,
    on_result: Callable[[int, MatchResult], None] | None = None,
) -> list[MatchResult]:
    """Run each configuration in its own engine process, at most ``parallel`` at a time.

    Results come back in the order of ``configurations``; ``on_result(index, result)`` is called
    as soon as each match finishes, for progress reporting.
    """
    results: list[MatchResult | None] = [None] * len(configurations)
    # Threads are enough: each match is a separate engine process, Python only waits for it.
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                run_match,
                configuration,
                batch_directory / "matches" / f"match_{index:03d}_seed{configuration.seed}",
            ): index
            for index, configuration in enumerate(configurations)
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results[index] = result
            if on_result is not None:
                on_result(index, result)
    return [result for result in results if result is not None]  # all filled once the pool is done


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _wins_by_side(player: int, configurations: list[MatchConfiguration], results: list[MatchResult]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for configuration, result in zip(configurations, results, strict=True):
        side = counts.setdefault(str(configuration.players[player].side), {"wins": 0, "matches": 0})
        side["matches"] += 1
        side["wins"] += result.winner == player
    return dict(sorted(counts.items()))


def summarise(configurations: list[MatchConfiguration], results: list[MatchResult]) -> dict:
    """Win counts, game durations and averaged team statistics, per player position."""
    finished = [result for result in results if result.game_minutes is not None]
    number_of_players = len(configurations[0].players)
    players = []
    for player in range(number_of_players):
        played = [result.team_statistics[player] for result in finished if result.team_statistics]
        players.append(
            {
                "name": configurations[0].players[player].display_name,
                "sides": sorted({str(configuration.players[player].side) for configuration in configurations}),
                "wins": sum(result.winner == player for result in results),
                # {side: {"wins": ..., "matches": ...}}, for players that play several factions
                "wins_by_side": _wins_by_side(player, configurations, results),
                **{key: _mean([values[key] for values in played if key in values]) for key in SUMMARISED_STATISTICS},
            }
        )
    durations = [result.game_minutes for result in finished if result.game_minutes is not None]
    return {
        "matches": len(results),
        "outcomes": {
            outcome: sum(result.outcome == outcome for result in results)
            for outcome in ["victory", "draw", "timeout", "crash"]
        },
        "players": players,
        "game_minutes": {
            "mean": _mean(durations),
            "minimum": min(durations, default=None),
            "maximum": max(durations, default=None),
        },
        "mean_speed_factor": _mean([result.speed_factor for result in finished if result.speed_factor is not None]),
    }


def format_summary(summary: dict) -> str:
    """Human-readable table of a summary, printed at the end and saved as summary.txt."""
    outcomes = summary["outcomes"]
    lines = [
        f"{summary['matches']} matches: {outcomes['victory']} victories, {outcomes['draw']} draws, "
        f"{outcomes['timeout']} timeouts, {outcomes['crash']} crashes"
    ]
    durations = summary["game_minutes"]
    if durations["mean"] is not None:
        lines.append(
            f"Game duration: mean {durations['mean']:.1f} min "
            f"(from {durations['minimum']:.1f} to {durations['maximum']:.1f}), "
            f"mean speed x{summary['mean_speed_factor']:.0f}"
        )

    players = summary["players"]
    rows = [
        ("", [f"player {index}: {player['name']}" for index, player in enumerate(players)]),
        ("sides", [" / ".join(player["sides"]) for player in players]),
        ("wins", [f"{player['wins']} ({100 * player['wins'] / summary['matches']:.0f} %)" for player in players]),
    ]
    # One more row per faction when a player played several of them.
    if any(len(player["wins_by_side"]) > 1 for player in players):
        for side in sorted({side for player in players for side in player["wins_by_side"]}):
            counts = [player["wins_by_side"].get(side) for player in players]
            rows.append((f"wins as {side}", [f"{c['wins']} / {c['matches']}" if c else "-" for c in counts]))
    for key in SUMMARISED_STATISTICS:
        rows.append((key, ["-" if player[key] is None else f"{player[key]:,.0f}" for player in players]))
    lines += ["", "Mean per match:"]
    lines += [f"  {label:<16}" + "".join(f"{value:>24}" for value in values) for label, values in rows]
    return "\n".join(lines) + "\n"


def save_batch(
    batch_directory: Path,
    configurations: list[MatchConfiguration],
    results: list[MatchResult],
    summary: dict,
) -> None:
    """Write summary.txt (the table) and summary.json (the summary and every match, for later analysis)."""
    (batch_directory / "summary.txt").write_text(format_summary(summary))
    # The full statistics history stays in each match folder (match_result.json) and in the Parquet tables.
    matches = [
        {
            "configuration": asdict(configuration),
            **{key: value for key, value in asdict(result).items() if key != "team_history"},
            "run_directory": str(result.run_directory),
        }
        for configuration, result in zip(configurations, results, strict=True)
    ]
    # default=str writes paths (replay_path) as text.
    (batch_directory / "summary.json").write_text(
        json.dumps({"summary": summary, "matches": matches}, indent=2, default=str)
    )


def load_batch(batch_directory: Path) -> tuple[list[MatchConfiguration], list[MatchResult]]:
    """Read back the matches of a finished batch (summary.json and each match_result.json), to redraw charts."""
    content = json.loads((batch_directory / "summary.json").read_text())
    configurations, results = [], []
    for match in content["matches"]:
        configuration = dict(match["configuration"])
        configuration["players"] = [ArtificialIntelligencePlayer(**player) for player in configuration["players"]]
        configurations.append(MatchConfiguration(**configuration))
        # Found by name, so the batch folder can be moved or copied.
        run_directory = batch_directory / "matches" / Path(match["run_directory"]).name
        results.append(read_match_result(run_directory, len(configuration["players"]), match["wall_clock_seconds"]))
    return configurations, results
