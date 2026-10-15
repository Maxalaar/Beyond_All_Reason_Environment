"""Run several matches in parallel, described by a YAML configuration file, and summarise them."""

import json
import re
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from .installation import DEFAULT_MAP
from .match import MatchResult, run_match
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration

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
    """Content of a configuration file in configurations/ (see barb_versus_barb.yaml)."""

    # Also the name of the results folder: runs/<name>/.
    name: str
    players: list[ArtificialIntelligencePlayer]
    matches: int = 5
    parallel: int = 4
    first_seed: int = 1
    alternate_sides: bool = False
    maximum_game_minutes: float = 60.0
    map_name: str = DEFAULT_MAP

    def match_configurations(self) -> list[MatchConfiguration]:
        configurations = []
        for index in range(self.matches):
            sides = [player.side for player in self.players]
            if self.alternate_sides and index % 2:
                sides.reverse()
            configurations.append(MatchConfiguration(
                players=[
                    ArtificialIntelligencePlayer(player.short_name, player.version, side)
                    for player, side in zip(self.players, sides)
                ],
                map_name=self.map_name,
                seed=self.first_seed + index,
                maximum_game_minutes=self.maximum_game_minutes,
            ))
        return configurations


def load_batch_configuration(path: Path) -> BatchConfiguration:
    """Read a YAML configuration file; unknown keys are errors, to catch typos."""
    content = yaml.safe_load(Path(path).read_text())
    allowed = {item.name for item in fields(BatchConfiguration)}
    unknown = set(content) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}, allowed: {sorted(allowed)}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(content.get("name", ""))):
        raise ValueError(f"{path}: 'name' is required and may only contain letters, digits, _ and -")
    content["players"] = [ArtificialIntelligencePlayer(**player) for player in content.get("players", [])]
    if len(content["players"]) < 2:
        raise ValueError(f"{path}: at least two players are needed")
    return BatchConfiguration(**content)


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
                batch_directory / f"match_{index:03d}_seed{configuration.seed}",
            ): index
            for index, configuration in enumerate(configurations)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            if on_result is not None:
                on_result(index, results[index])
    return results


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarise(configurations: list[MatchConfiguration], results: list[MatchResult]) -> dict:
    """Win counts, game durations and averaged team statistics, per player position."""
    finished = [result for result in results if result.game_minutes is not None]
    number_of_players = len(configurations[0].players)
    players = []
    for player in range(number_of_players):
        played = [result.team_statistics[player] for result in finished if result.team_statistics]
        players.append({
            "name": configurations[0].players[player].short_name,
            "sides": sorted({configuration.players[player].side for configuration in configurations}),
            "wins": sum(result.winner == player for result in results),
            **{
                key: _mean([values[key] for values in played if key in values])
                for key in SUMMARISED_STATISTICS
            },
        })
    durations = [result.game_minutes for result in finished]
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
        "mean_speed_factor": _mean([result.speed_factor for result in finished]),
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
    matches = [
        {"configuration": asdict(configuration), **asdict(result), "run_directory": str(result.run_directory)}
        for configuration, result in zip(configurations, results)
    ]
    (batch_directory / "summary.json").write_text(json.dumps({"summary": summary, "matches": matches}, indent=2))
