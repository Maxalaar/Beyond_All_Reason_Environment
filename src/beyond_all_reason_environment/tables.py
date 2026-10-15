"""Results as Parquet tables, for analysis across matches, versions and experiments.

Each batch writes runs/<experiment>/version_<n>/tables/:

    matches.parquet     one row per match: seed, map, outcome, winner, duration...
    players.parquet     one row per player in a match: artificial intelligence, side, start position, won,
                        final statistics
    timeseries.parquet  one row per player and statistics entry (every 15 s of game time): raw cumulative
                        engine statistics and the metrics of metrics.py (metal_income, units_alive...)

Every row carries ``match_id`` ("<experiment>/version_<n>/<match>"), ``experiment`` and ``version``, so the
tables of all batches can be concatenated (``read_tables``) or queried together, for example with DuckDB:
``SELECT side, avg(won) FROM 'runs/*/version_*/tables/players.parquet' GROUP BY side``.
"""

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .batch import VERSION_PREFIX
from .installation import ENGINE_VERSION
from .match import RUNS_DIRECTORY, MatchResult
from .metrics import METRICS, minutes_of
from .start_script import MatchConfiguration

TABLES_FOLDER = "tables"
TABLE_NAMES = ("matches", "players", "timeseries")


def snake_case(name: str) -> str:
    """metalProduced -> metal_produced, the column naming of the tables."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def experiment_and_version(batch_directory: Path) -> tuple[str, int]:
    """runs/<experiment>/version_<n> -> ("<experiment>", n)."""
    return batch_directory.parent.name, int(batch_directory.name.removeprefix(VERSION_PREFIX))


def build_tables(
    batch_directory: Path, configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult]
) -> dict[str, pd.DataFrame]:
    experiment, version = experiment_and_version(batch_directory)
    matches, players, timeseries = [], [], []
    for index, (configuration, result) in enumerate(zip(configurations, results, strict=True)):
        match_id = f"{experiment}/{VERSION_PREFIX}{version}/{index:03d}"
        keys = {"match_id": match_id, "experiment": experiment, "version": version, "match": index}
        map_size = result.map_size or {}
        matches.append(
            {
                **keys,
                "seed": configuration.seed,
                "map_name": configuration.map_name,
                "map_size_x": map_size.get("sizeX"),
                "map_size_z": map_size.get("sizeZ"),
                "game_version": configuration.game_version,
                "engine_version": ENGINE_VERSION,
                "outcome": result.outcome,
                "winner": result.winner,
                "game_minutes": result.game_minutes,
                "wall_clock_seconds": result.wall_clock_seconds,
                "maximum_game_minutes": configuration.maximum_game_minutes,
                "run_directory": str(result.run_directory),
                "has_replay": result.replay_path is not None,
            }
        )
        for number, player in enumerate(configuration.players):
            start = result.start_positions[number] if number < len(result.start_positions) else None
            final = result.team_statistics[number] if number < len(result.team_statistics) else {}
            identity = {
                **keys,
                "player": number,
                "artificial_intelligence": player.short_name,
                "artificial_intelligence_version": player.version,
                "side": str(player.side),
            }
            players.append(
                {
                    **identity,
                    "seed": configuration.seed,
                    "map_name": configuration.map_name,
                    "outcome": result.outcome,
                    "won": result.winner == number,
                    "start_x": start["x"] if start else None,
                    "start_z": start["z"] if start else None,
                    **{snake_case(key): value for key, value in final.items()},
                }
            )
            history = result.team_history[number] if number < len(result.team_history) else []
            if not history:
                continue
            minutes = minutes_of(history)
            columns = {
                "time_seconds": minutes * 60,
                "game_minutes": minutes,
                # Share of the match duration, 0 at the start and 1 at the end, to compare matches of any length.
                "relative_time": minutes / max(float(minutes[-1]), 1e-9),
            }
            for key in history[0]:
                if key not in ("time", "frame"):
                    columns[snake_case(key)] = np.array([entry.get(key, np.nan) for entry in history], dtype=float)
            columns["frame"] = np.array([entry["frame"] for entry in history], dtype=np.int64)
            for metric in METRICS:
                columns.setdefault(metric.name, metric.compute(result.team_history, number))
            frame = pd.DataFrame(columns)
            for position, (key, value) in enumerate(identity.items()):
                frame.insert(position, key, value)
            timeseries.append(frame)

    return {
        "matches": pd.DataFrame(matches).astype({"winner": "Int64"}),
        "players": pd.DataFrame(players),
        "timeseries": pd.concat(timeseries, ignore_index=True) if timeseries else pd.DataFrame(),
    }


def write_tables(
    batch_directory: Path, configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult]
) -> dict[str, Path]:
    directory = batch_directory / TABLES_FOLDER
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in build_tables(batch_directory, configurations, results).items():
        paths[name] = directory / f"{name}.parquet"
        frame.to_parquet(paths[name], index=False)
    return paths


def list_versions(runs_directory: Path = RUNS_DIRECTORY) -> list[Path]:
    """Every runs/<experiment>/version_<n> folder that has tables, sorted by experiment then version."""
    versions = [path.parent for path in runs_directory.glob(f"*/{VERSION_PREFIX}*/{TABLES_FOLDER}")]
    return sorted(versions, key=experiment_and_version)


def read_tables(versions: Sequence[Path]) -> dict[str, pd.DataFrame]:
    """The tables of several versions, concatenated."""
    tables = {}
    for name in TABLE_NAMES:
        frames = [
            pd.read_parquet(path)
            for version in versions
            if (path := version / TABLES_FOLDER / f"{name}.parquet").exists()
        ]
        tables[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return tables
