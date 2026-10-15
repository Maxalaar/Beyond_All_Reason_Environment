"""Statistics on the Parquet tables (tables.py): win rates with confidence intervals, curves over time.

Groups are whatever column the caller chooses in the players or timeseries table: "side" to compare
factions, "artificial_intelligence" to compare artificial intelligences, or a player label.
"""

import math
import warnings

import numpy as np
import pandas as pd

from .metrics import METRICS_BY_NAME

# Below this many matches still running, an aggregated value over game minutes says little.
MINIMUM_MATCHES_RUNNING = 3
MINUTES_STEP = 0.25
RELATIVE_STEP = 0.01


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval of a proportion; reliable even for few trials or extreme proportions."""
    if trials == 0:
        return (math.nan, math.nan)
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def win_rates(players: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Per group: matches, wins, win rate over all matches and over decisive ones, with 95 % intervals.

    A decisive match has a winner; time limits and draws count as not won in the first rate and are left
    out of the second.
    """
    rows = []
    for group, frame in players.groupby(group_column, sort=True):
        decisive = frame[frame["outcome"] == "victory"]
        wins = int(frame["won"].to_numpy().sum())
        low, high = wilson_interval(wins, len(frame))
        decisive_low, decisive_high = wilson_interval(wins, len(decisive))
        rows.append(
            {
                group_column: group,
                "matches": len(frame),
                "wins": wins,
                "win_rate": wins / len(frame),
                "win_rate_low": low,
                "win_rate_high": high,
                "decisive_matches": len(decisive),
                "decisive_win_rate": wins / len(decisive) if len(decisive) else math.nan,
                "decisive_win_rate_low": decisive_low,
                "decisive_win_rate_high": decisive_high,
            }
        )
    return pd.DataFrame(rows)


def resample_over_time(
    timeseries: pd.DataFrame, metric: str, relative: bool, group_column: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Each (match, player) series of ``metric`` on a common grid, grouped.

    The x-axis is game minutes, or the share of each match's duration (``relative``). In minutes, a series
    is NaN after its match ended, except for running totals, which keep their final value so that the
    median of the remaining matches does not drop. Returns the grid and, per group, an array
    (series, grid points).
    """
    x_column = "relative_time" if relative else "game_minutes"
    hold_after_end = METRICS_BY_NAME[metric].is_total if metric in METRICS_BY_NAME else False
    end = 1.0 if relative else float(timeseries[x_column].to_numpy().max())
    grid = np.arange(0, end + 1e-9, RELATIVE_STEP if relative else MINUTES_STEP)
    samples: dict[str, list[np.ndarray]] = {}
    for _, series in timeseries.groupby([group_column, "match_id", "player"], sort=True):
        group = str(series[group_column].iloc[0])
        x, values = series[x_column].to_numpy(), series[metric].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        resampled = np.interp(grid, x[finite], values[finite])
        resampled[grid < x[finite][0]] = np.nan
        if not hold_after_end:
            resampled[grid > x[-1] + 1e-9] = np.nan
        samples.setdefault(group, []).append(resampled)
    return grid, {group: np.array(values) for group, values in samples.items()}


def aggregate_over_time(timeseries: pd.DataFrame, metric: str, relative: bool, group_column: str) -> pd.DataFrame:
    """Per group and grid point: median, 25th and 75th percentiles, and matches still running.

    In game minutes, points where fewer than MINIMUM_MATCHES_RUNNING matches of the group are still
    running are left out (late values would only describe the longest matches).
    """
    grid, samples = resample_over_time(timeseries, metric, relative, group_column)
    # End of each (match, player) series, with its group.
    ends = timeseries.groupby(["match_id", "player"]).agg(group=(group_column, "first"), end=("game_minutes", "max"))
    ends_by_group = {str(group): np.asarray(frame["end"], dtype=float) for group, frame in ends.groupby("group")}
    rows = []
    for group, values in samples.items():
        if relative:
            running = np.full(grid.size, values.shape[0])
        else:
            running = (grid[:, None] <= ends_by_group[group][None, :]).sum(axis=1)
        # Grid points where no series has a value give "all-NaN slice" warnings; they are left out below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
        shown = running >= min(MINIMUM_MATCHES_RUNNING, values.shape[0])
        rows.append(
            pd.DataFrame(
                {
                    group_column: group,
                    "x": grid[shown],
                    "median": median[shown],
                    "low": low[shown],
                    "high": high[shown],
                    "running": running[shown],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
