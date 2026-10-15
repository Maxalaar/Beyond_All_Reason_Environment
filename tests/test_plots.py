"""Charts are drawn from synthetic statistics histories, without the game engine."""

import json
from pathlib import Path

from beyond_all_reason_environment import MatchConfiguration
from beyond_all_reason_environment.batch import SUMMARISED_STATISTICS, load_batch, save_batch, summarise
from beyond_all_reason_environment.match import MatchResult
from beyond_all_reason_environment.plots import METRICS, plot_batch


def synthetic_history(minutes: int, growth: float) -> list[dict]:
    return [
        {
            "time": second,
            "metalProduced": growth * second**1.5,
            "energyProduced": 20 * growth * second**1.5,
            "damageDealt": growth * second**2,
            "unitsProduced": second // 10,
            "unitsDied": second // 30,
            "unitsKilled": second // 25,
            "damageReceived": growth * second**1.9,
            "metalUsed": growth * second**1.4,
        }
        for second in range(0, minutes * 60 + 1, 15)
    ]


def synthetic_batch(directory: Path) -> tuple[list[MatchConfiguration], list[MatchResult]]:
    configurations, results = [], []
    for index, (minutes, winner) in enumerate([(20, 0), (30, 1), (25, 0)]):
        run_directory = directory / "matches" / f"match_{index}"
        run_directory.mkdir(parents=True)
        histories = [synthetic_history(minutes, 1.0), synthetic_history(minutes, 1.2)]
        teams = {str(player): {"history": history, "unitsAlive": 10} for player, history in enumerate(histories)}
        (run_directory / "match_result.json").write_text(json.dumps({"teams": teams}))
        statistics = [{**history[-1], "unitsAlive": 10} for history in histories]
        configurations.append(MatchConfiguration(seed=index))
        results.append(MatchResult("victory", winner, minutes * 60 * 30, 60.0, run_directory, statistics, histories))

    return configurations, results


def test_batch_charts_are_written(tmp_path: Path):
    configurations, results = synthetic_batch(tmp_path)

    paths = plot_batch(configurations, results, tmp_path)

    per_match = 1 + len(METRICS)  # overview + one chart per metric
    # outcomes, end of match, over time in minutes and relative, matches running, overview
    aggregated = 2 + (len(SUMMARISED_STATISTICS) + 1) + 2 * len(METRICS) + 1 + 1
    assert len(paths) == len(results) * per_match + aggregated
    assert all(path.exists() and path.stat().st_size > 10_000 for path in paths)
    assert (tmp_path / "matches" / "match_0" / "charts" / "military" / "units_killed.png").exists()
    assert (tmp_path / "charts" / "over_time" / "economy" / "metal_income.png").exists()
    assert (tmp_path / "overview.png").exists()
    assert (tmp_path / "matches" / "match_0" / "overview.png").exists()


def test_saved_batch_loads_back(tmp_path: Path):
    configurations, results = synthetic_batch(tmp_path)
    save_batch(tmp_path, configurations, results, summarise(configurations, results))

    loaded_configurations, loaded_results = load_batch(tmp_path)

    assert loaded_configurations == configurations
    assert [result.team_history for result in loaded_results] == [result.team_history for result in results]
    assert [result.winner for result in loaded_results] == [result.winner for result in results]
