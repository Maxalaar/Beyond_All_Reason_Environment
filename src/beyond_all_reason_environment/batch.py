"""Run several matches in parallel and summarise them."""

import json
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .match import MatchResult, run_match
from .start_script import MatchConfiguration

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


def save_batch(
    batch_directory: Path,
    configurations: list[MatchConfiguration],
    results: list[MatchResult],
    summary: dict,
) -> Path:
    """Write summary.json with the summary and every match, for later analysis."""
    path = batch_directory / "summary.json"
    matches = [
        {"configuration": asdict(configuration), **asdict(result), "run_directory": str(result.run_directory)}
        for configuration, result in zip(configurations, results)
    ]
    path.write_text(json.dumps({"summary": summary, "matches": matches}, indent=2))
    return path
