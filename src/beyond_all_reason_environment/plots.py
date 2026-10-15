"""Charts of one match (statistics over time) and of a batch of matches (aggregated).

Players keep the same color in every chart: player 0 blue, player 1 orange (validated colorblind-safe).
"""

import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window; safe outside the main thread

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from .batch import SUMMARISED_STATISTICS  # noqa: E402
from .match import MatchResult  # noqa: E402
from .start_script import MatchConfiguration  # noqa: E402

# Categorical slots in fixed order, one per player position.
PLAYER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL_COLOR = "#8a8984"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"

# Time step of the aggregated curves, in game minutes (the engine records statistics every 15 s).
AGGREGATION_STEP_MINUTES = 0.25

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "font.size": 9,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    }
)


# --- Statistics over time --------------------------------------------------------------------------


def _cumulative(key: str):
    return lambda history: np.array([entry.get(key, 0.0) for entry in history], dtype=float)


def _rate(key: str):
    """Per-second rate of a cumulative statistic, between consecutive entries."""

    def compute(history):
        values = _cumulative(key)(history)
        times = np.array([entry["time"] for entry in history], dtype=float)
        rates = np.diff(values) / np.maximum(np.diff(times), 1e-9)
        return np.concatenate([[0.0], rates])

    return compute


def _units_alive(history):
    count = lambda key: _cumulative(key)(history)  # noqa: E731
    gained = count("unitsProduced") + count("unitsReceived") + count("unitsCaptured")
    lost = count("unitsDied") + count("unitsSent") + count("unitsOutCaptured")
    return gained - lost


# (title, function of a statistics history returning one value per entry)
TIME_SERIES = [
    ("Metal income (per second)", _rate("metalProduced")),
    ("Energy income (per second)", _rate("energyProduced")),
    ("Units alive", _units_alive),
    ("Damage dealt (total)", _cumulative("damageDealt")),
    ("Units killed (total)", _cumulative("unitsKilled")),
    ("Metal produced (total)", _cumulative("metalProduced")),
]


STATISTIC_TITLES = {
    "metalProduced": "Metal produced",
    "energyProduced": "Energy produced",
    "damageDealt": "Damage dealt",
    "damageReceived": "Damage received",
    "unitsProduced": "Units produced",
    "unitsKilled": "Units killed",
    "unitsDied": "Units lost",
    "unitsAlive": "Units alive at the end",
}


def _minutes(history) -> np.ndarray:
    return np.array([entry["time"] for entry in history], dtype=float) / 60


def _player_label(configuration: MatchConfiguration, index: int, with_side: bool = True) -> str:
    player = configuration.players[index]
    return f"player {index}: {player.short_name}" + (f" ({player.side})" if with_side else "")


def _compact_number(value: float, _position=None) -> str:
    """600000 -> 600k, 12500000 -> 12.5M: readable axes without scientific notation."""
    for threshold, suffix in [(1e9, "G"), (1e6, "M"), (1e3, "k")]:
        if abs(value) >= threshold:
            return f"{value / threshold:.3g}{suffix}"
    return f"{value:.3g}"


def _style_value_axis(axes: Axes) -> None:
    axes.yaxis.set_major_formatter(FuncFormatter(_compact_number))
    axes.set_ylim(bottom=0)


def _direct_label(axes: Axes, x: float, y: float, text: str, color: str) -> None:
    """Small label at the end of a line, in text ink next to a colored dot (identity not by color alone)."""
    axes.plot([x], [y], "o", markersize=4, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
    axes.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", color=TEXT_SECONDARY)


def _time_series_grid(number_of_panels: int):
    columns = 3
    rows = -(-number_of_panels // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), layout="constrained")
    axes = axes.ravel()
    for unused in axes[number_of_panels:]:
        unused.set_visible(False)
    return figure, axes


def plot_match(result: MatchResult, configuration: MatchConfiguration, path: Path) -> Path | None:
    """Statistics of each player over the game time of one match."""
    if not any(result.team_history):
        return None
    figure, axes = _time_series_grid(len(TIME_SERIES))
    for (title, compute), panel in zip(TIME_SERIES, axes, strict=False):
        for index, history in enumerate(result.team_history):
            if not history:
                continue
            minutes, values = _minutes(history), compute(history)
            color = PLAYER_COLORS[index]
            panel.plot(minutes, values, color=color, label=_player_label(configuration, index))
            _direct_label(panel, minutes[-1], values[-1], f"player {index}", color)
        panel.set_title(title)
        panel.set_xlabel("game minutes")
        _style_value_axis(panel)
        panel.margins(x=0.08)

    if result.outcome == "victory" and result.winner is not None:
        outcome = f"{_player_label(configuration, result.winner)} wins"
    else:
        outcome = {"draw": "draw", "timeout": "no winner (time limit)"}.get(result.outcome, result.outcome)
    figure.suptitle(
        f"Seed {configuration.seed}, {configuration.map_name}: {outcome} after {result.game_minutes:.1f} game minutes",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower left", ncols=len(configuration.players))
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


# --- Aggregation over a batch --------------------------------------------------------------------------


def _resample(history, compute, grid: np.ndarray) -> np.ndarray:
    """Values of one match on a common time grid; NaN after the match ended."""
    minutes = _minutes(history)
    values = np.interp(grid, minutes, compute(history))
    values[grid > minutes[-1]] = np.nan
    return values


def plot_batch_time_series(
    configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult], path: Path
) -> Path | None:
    """Median and interquartile band of each statistic over time, per player, across the matches."""
    played = [result for result in results if any(result.team_history)]
    if not played:
        return None
    end = max(_minutes(history)[-1] for result in played for history in result.team_history if history)
    grid = np.arange(0, end + AGGREGATION_STEP_MINUTES, AGGREGATION_STEP_MINUTES)
    match_ends = [max(_minutes(history)[-1] for history in result.team_history if history) for result in played]
    running = np.array([(grid <= match_end) for match_end in match_ends]).sum(axis=0)
    # Below this many matches still running, the median says little: stop the curves there.
    minimum_running = min(3, len(played))
    shown = running >= minimum_running

    figure, axes = _time_series_grid(len(TIME_SERIES) + 1)
    number_of_players = len(configurations[0].players)
    for (title, compute), panel in zip(TIME_SERIES, axes, strict=False):
        for index in range(number_of_players):
            histories = [result.team_history[index] for result in played if result.team_history[index]]
            samples = np.array([_resample(history, compute, grid) for history in histories])
            # Time steps where no match is running give "all-NaN slice" warnings; they are hidden anyway.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low, median, high = np.nanpercentile(samples, [25, 50, 75], axis=0)
            color = PLAYER_COLORS[index]
            panel.fill_between(grid[shown], low[shown], high[shown], color=color, alpha=0.15, linewidth=0)
            panel.plot(grid[shown], median[shown], color=color, label=_player_label(configurations[0], index, False))
            last = np.flatnonzero(shown)[-1]
            _direct_label(panel, float(grid[last]), float(median[last]), f"player {index}", color)
        panel.set_title(title)
        panel.set_xlabel("game minutes")
        _style_value_axis(panel)
        panel.margins(x=0.08)

    panel = axes[len(TIME_SERIES)]
    panel.step(grid, running, where="post", color=NEUTRAL_COLOR)
    panel.axhline(minimum_running, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
    panel.set_title("Matches still running")
    panel.set_xlabel("game minutes")
    panel.set_ylim(0, len(played) + 0.5)
    panel.yaxis.set_major_locator(MaxNLocator(integer=True))

    figure.suptitle(
        f"{len(played)} matches: median per player, band from 25th to 75th percentile "
        f"(curves stop when fewer than {minimum_running} matches are still running)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower left", ncols=number_of_players)
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def plot_batch_outcomes(
    configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult], path: Path
) -> Path:
    """Who won how often, and how long each match lasted."""
    number_of_players = len(configurations[0].players)
    figure, (counts_axes, durations_axes) = plt.subplots(
        1, 2, figsize=(14, 4.5), layout="constrained", width_ratios=[1, 1.4]
    )

    labels = [_player_label(configurations[0], index, False) + " wins" for index in range(number_of_players)]
    values = [sum(result.winner == index for result in results) for index in range(number_of_players)]
    colors = PLAYER_COLORS[:number_of_players]
    for outcome, label in [("draw", "draws"), ("timeout", "time limit"), ("crash", "crashes")]:
        labels.append(label)
        values.append(sum(result.outcome == outcome for result in results))
        colors.append(NEUTRAL_COLOR)
    positions = np.arange(len(labels))[::-1]
    counts_axes.barh(positions, values, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    for position, value in zip(positions, values, strict=True):
        share = f"{value}  ({100 * value / len(results):.0f} %)"
        counts_axes.annotate(share, (value, position), xytext=(4, 0), textcoords="offset points", va="center")
    counts_axes.set_yticks(positions, labels)
    counts_axes.set_title(f"Outcomes of {len(results)} matches")
    counts_axes.set_xlabel("matches")
    counts_axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    counts_axes.grid(axis="y", visible=False)
    counts_axes.margins(x=0.25)

    for match_index, result in enumerate(results):
        if result.game_minutes is None:
            continue
        winner = result.winner if result.outcome == "victory" else None
        color = NEUTRAL_COLOR if winner is None else PLAYER_COLORS[winner]
        durations_axes.plot(
            [match_index], [result.game_minutes], "o", markersize=8, color=color, markeredgecolor=SURFACE
        )
    durations_axes.set_title("Duration of each match (color: winner, gray: no winner)")
    durations_axes.set_xlabel("match")
    durations_axes.set_ylabel("game minutes")
    durations_axes.set_ylim(bottom=0)
    durations_axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def plot_batch_final_statistics(
    configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult], path: Path
) -> Path:
    """End-of-match statistics: one dot per match, a line at the mean, per player."""
    number_of_players = len(configurations[0].players)
    columns = 4
    rows = -(-len(SUMMARISED_STATISTICS) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.6 * rows), layout="constrained")
    random = np.random.default_rng(0)  # fixed jitter, so charts do not change between runs
    for key, panel in zip(SUMMARISED_STATISTICS, axes.ravel(), strict=False):
        for index in range(number_of_players):
            values = np.array(
                [result.team_statistics[index][key] for result in results if key in result.team_statistics[index]]
                if all(result.team_statistics for result in results)
                else [],
                dtype=float,
            )
            if values.size == 0:
                continue
            color = PLAYER_COLORS[index]
            jitter = random.uniform(-0.12, 0.12, values.size)
            panel.plot(index + jitter, values, "o", markersize=8, color=color, markeredgecolor=SURFACE, alpha=0.85)
            panel.plot([index - 0.25, index + 0.25], [values.mean()] * 2, color=TEXT_PRIMARY, linewidth=2)
        panel.set_title(STATISTIC_TITLES.get(key, key))
        panel.set_xticks(range(number_of_players), [f"player {index}" for index in range(number_of_players)])
        panel.set_xlim(-0.6, number_of_players - 0.4)
        _style_value_axis(panel)
        panel.grid(axis="x", visible=False)
    figure.suptitle(
        "Statistics at the end of each match (dot: one match, black line: mean)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def plot_batch(
    configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult], directory: Path
) -> list[Path]:
    """All charts of a batch: one per match in its folder, and the aggregated ones in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_match(result, configuration, result.run_directory / "match.png")
        for configuration, result in zip(configurations, results, strict=True)
    ]
    paths += [
        plot_batch_time_series(configurations, results, directory / "time_series.png"),
        plot_batch_outcomes(configurations, results, directory / "outcomes.png"),
        plot_batch_final_statistics(configurations, results, directory / "final_statistics.png"),
    ]
    return [path for path in paths if path is not None]
