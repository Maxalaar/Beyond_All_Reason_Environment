"""Charts are drawn from synthetic statistics histories, without the game engine."""

from pathlib import Path

from beyond_all_reason_environment import MatchConfiguration
from beyond_all_reason_environment.match import MatchResult
from beyond_all_reason_environment.plots import plot_batch


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
        }
        for second in range(0, minutes * 60 + 1, 15)
    ]


def test_batch_charts_are_written(tmp_path: Path):
    configurations, results = [], []
    for index, (minutes, winner) in enumerate([(20, 0), (30, 1), (25, 0)]):
        run_directory = tmp_path / f"match_{index}"
        run_directory.mkdir()
        histories = [synthetic_history(minutes, 1.0), synthetic_history(minutes, 1.2)]
        statistics = [{**history[-1], "unitsAlive": 10} for history in histories]
        configurations.append(MatchConfiguration(seed=index))
        results.append(MatchResult("victory", winner, minutes * 60 * 30, 60.0, run_directory, statistics, histories))

    paths = plot_batch(configurations, results, tmp_path / "plots")

    assert len(paths) == len(results) + 3
    assert all(path.exists() and path.stat().st_size > 10_000 for path in paths)
