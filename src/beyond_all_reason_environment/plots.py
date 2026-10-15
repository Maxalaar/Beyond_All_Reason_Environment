"""Charts of one match (statistics over time) and of a batch of matches (aggregated).

Every metric gets its own chart file, sorted by category, and an overview image gathers the six most
telling ones:

    <match folder>/charts/overview.png
    <match folder>/charts/{economy,military,units}/<metric>.png
    <batch folder>/charts/overview.png
    <batch folder>/charts/outcomes/{wins,durations}.png
    <batch folder>/charts/over_time/{economy,military,units}/<metric>.png, over_time/matches_running.png
    <batch folder>/charts/end_of_match/<statistic>.png, end_of_match/all.png

Players keep the same color in every chart: player 0 blue, player 1 orange (validated colorblind-safe).
"""

import textwrap
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window; safe outside the main thread

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter  # noqa: E402

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

# The engine records statistics every 15 s of game time; rates are averaged over this many entries (1 minute).
RATE_SMOOTHING_ENTRIES = 4
# Time step of the aggregated curves, in game minutes.
AGGREGATION_STEP_MINUTES = 0.25

SINGLE_CHART_SIZE = (8, 4.5)
OVERVIEW_SIZE = (15, 8.5)
CHART_RESOLUTION = 110

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

Histories = list[list[dict]]


# --- Metrics computed from the statistics history -----------------------------------------------------


@dataclass(frozen=True)
class Metric:
    name: str  # file name
    title: str
    category: str  # economy, military or units: sub-folder of the charts
    # (statistics history of every player, player index) -> one value per history entry
    compute: Callable[[Histories, int], np.ndarray]
    # "share" draws a 0-100 % axis
    kind: str = "value"
    # Dashed horizontal line, for example 1 for a ratio
    reference_line: float | None = None
    # Running total: when aggregating, a finished match keeps its final value instead of dropping out,
    # otherwise the median of the remaining matches could go down.
    is_total: bool = False


def _times(history: list[dict]) -> np.ndarray:
    return np.array([entry["time"] for entry in history], dtype=float)


def _minutes(history: list[dict]) -> np.ndarray:
    return _times(history) / 60


def _values(history: list[dict], key: str) -> np.ndarray:
    return np.array([entry.get(key, 0.0) for entry in history], dtype=float)


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    positions = np.arange(1, values.size + 1)
    sums = np.cumsum(np.concatenate([[0.0], values]))
    return (sums[1:] - sums[np.maximum(positions - window, 0)]) / np.minimum(positions, window)


def cumulative(key: str) -> Callable[[Histories, int], np.ndarray]:
    return lambda histories, index: _values(histories[index], key)


def rate(key: str, scale: float = 1.0) -> Callable[[Histories, int], np.ndarray]:
    """Per-second rate of a cumulative statistic (times ``scale``), averaged over one minute."""

    def compute(histories: Histories, index: int) -> np.ndarray:
        history = histories[index]
        rates = np.diff(_values(history, key)) / np.maximum(np.diff(_times(history)), 1e-9)
        return _trailing_mean(np.concatenate([[0.0], rates]), RATE_SMOOTHING_ENTRIES) * scale

    return compute


def share(key: str) -> Callable[[Histories, int], np.ndarray]:
    """Fraction of the total of all players, entry by entry (entries are recorded at the same frames)."""

    def compute(histories: Histories, index: int) -> np.ndarray:
        length = min(len(history) for history in histories if history)
        totals = np.sum([_values(history[:length], key) for history in histories if history], axis=0)
        own = _values(histories[index][:length], key)
        result = np.full(len(histories[index]), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            result[:length] = np.where(totals > 0, own / totals, np.nan)
        return result

    return compute


def ratio(numerator: str, denominator: str, minimum_denominator: float) -> Callable[[Histories, int], np.ndarray]:
    """Cumulative ratio, hidden while the denominator is too small to be meaningful."""

    def compute(histories: Histories, index: int) -> np.ndarray:
        top, bottom = _values(histories[index], numerator), _values(histories[index], denominator)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(bottom >= minimum_denominator, top / bottom, np.nan)

    return compute


def units_alive(histories: Histories, index: int) -> np.ndarray:
    history = histories[index]
    gained = _values(history, "unitsProduced") + _values(history, "unitsReceived") + _values(history, "unitsCaptured")
    lost = _values(history, "unitsDied") + _values(history, "unitsSent") + _values(history, "unitsOutCaptured")
    return gained - lost


METRICS = [
    # Economy
    Metric("metal_income", "Metal income (per second)", "economy", rate("metalProduced")),
    Metric("metal_spending", "Metal spending (per second)", "economy", rate("metalUsed")),
    Metric("metal_produced", "Metal produced (total)", "economy", cumulative("metalProduced"), is_total=True),
    Metric("metal_wasted", "Metal wasted, storage full (total)", "economy", cumulative("metalExcess"), is_total=True),
    Metric("metal_produced_share", "Share of all metal produced", "economy", share("metalProduced"), "share", 0.5),
    Metric("energy_income", "Energy income (per second)", "economy", rate("energyProduced")),
    Metric("energy_spending", "Energy spending (per second)", "economy", rate("energyUsed")),
    Metric("energy_produced", "Energy produced (total)", "economy", cumulative("energyProduced"), is_total=True),
    Metric(
        "energy_wasted", "Energy wasted, storage full (total)", "economy", cumulative("energyExcess"), is_total=True
    ),
    # Military
    Metric("damage_per_second", "Damage dealt (per second)", "military", rate("damageDealt")),
    Metric("damage_dealt", "Damage dealt (total)", "military", cumulative("damageDealt"), is_total=True),
    Metric("damage_received", "Damage received (total)", "military", cumulative("damageReceived"), is_total=True),
    Metric(
        "damage_efficiency",
        "Damage dealt / damage received",
        "military",
        ratio("damageDealt", "damageReceived", 2000),
        reference_line=1.0,
    ),
    Metric("units_killed", "Units killed (total)", "military", cumulative("unitsKilled"), is_total=True),
    Metric("units_lost", "Units lost (total)", "military", cumulative("unitsDied"), is_total=True),
    # Units
    Metric("units_alive", "Units alive", "units", units_alive),
    Metric("units_produced", "Units produced (total)", "units", cumulative("unitsProduced"), is_total=True),
    Metric("production_rate", "Units produced (per minute)", "units", rate("unitsProduced", 60)),
]
METRICS_BY_NAME = {metric.name: metric for metric in METRICS}

# The most important metrics: the top row of every overview, in this order.
KEY_METRICS = ["metal_income", "energy_income", "units_alive"]
# Match overview: the key metrics, then three more.
MATCH_OVERVIEW = [*KEY_METRICS, "metal_produced_share", "damage_dealt", "units_killed"]
# Batch overview: the key metrics, then the outcomes, durations and this one.
BATCH_OVERVIEW = [*KEY_METRICS, "metal_produced_share"]

# Final statistic -> (file name, title)
STATISTIC_NAMES = {
    "metalProduced": ("metal_produced", "Metal produced"),
    "energyProduced": ("energy_produced", "Energy produced"),
    "damageDealt": ("damage_dealt", "Damage dealt"),
    "damageReceived": ("damage_received", "Damage received"),
    "unitsProduced": ("units_produced", "Units produced"),
    "unitsKilled": ("units_killed", "Units killed"),
    "unitsDied": ("units_lost", "Units lost"),
    "unitsAlive": ("units_alive", "Units alive at the end"),
}


# --- Drawing helpers --------------------------------------------------------------------------------------


def _compact_number(value: float, _position=None) -> str:
    """600000 -> 600k, 12500000 -> 12.5M: readable axes without scientific notation."""
    for threshold, suffix in [(1e9, "G"), (1e6, "M"), (1e3, "k")]:
        if abs(value) >= threshold:
            return f"{value / threshold:.3g}{suffix}"
    return f"{value:.3g}"


def _style_value_axis(axes: Axes, kind: str = "value") -> None:
    if kind == "share":
        axes.yaxis.set_major_formatter(PercentFormatter(1.0))
        axes.set_ylim(0, 1)
    else:
        axes.yaxis.set_major_formatter(FuncFormatter(_compact_number))
        axes.set_ylim(bottom=0)


def _direct_label(axes: Axes, x: np.ndarray, y: np.ndarray, text: str, color: str) -> None:
    """Small label at the last visible point of a line, in text ink next to a colored dot."""
    finite = np.flatnonzero(np.isfinite(y))
    if finite.size == 0:
        return
    last = finite[-1]
    point = (float(x[last]), float(y[last]))
    axes.plot(*point, "o", markersize=4, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
    axes.annotate(text, point, xytext=(5, 0), textcoords="offset points", va="center", color=TEXT_SECONDARY)


def _player_label(configuration: MatchConfiguration, index: int, with_side: bool = True) -> str:
    player = configuration.players[index]
    return f"player {index}: {player.short_name}" + (f" ({player.side})" if with_side else "")


def _finish_time_axes(axes: Axes, metric: Metric) -> None:
    if metric.reference_line is not None:
        axes.axhline(metric.reference_line, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
    axes.set_title(metric.title)
    axes.set_xlabel("game minutes")
    _style_value_axis(axes, metric.kind)
    axes.margins(x=0.08)


def _save(figure, path: Path, suptitle: str | None = None, legend_axes: Axes | None = None) -> Path:
    if suptitle:
        # About 10 characters per inch at this size: wrap instead of letting the title run off the image.
        width = int(figure.get_figwidth() * 10)
        figure.suptitle(textwrap.fill(suptitle, width), x=0.01, ha="left", fontsize=11, fontweight="bold")
    if legend_axes is not None:
        handles, labels = legend_axes.get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="outside lower left", ncols=len(handles))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=CHART_RESOLUTION)
    plt.close(figure)
    return path


Draw = Callable[[Axes], None]


def _save_single(path: Path, draw: Draw, suptitle: str | None = None) -> Path:
    figure, axes = plt.subplots(figsize=SINGLE_CHART_SIZE, layout="constrained")
    draw(axes)
    return _save(figure, path, suptitle, axes)


def _save_overview(path: Path, draws: list[Draw], suptitle: str, legend_index: int = 0) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=OVERVIEW_SIZE, layout="constrained")
    for draw, panel in zip(draws, axes.ravel(), strict=True):
        draw(panel)
    return _save(figure, path, suptitle, axes.ravel()[legend_index])


# --- One match --------------------------------------------------------------------------------------------


def _draw_match_metric(metric: Metric, result: MatchResult, configuration: MatchConfiguration) -> Draw:
    def draw(axes: Axes) -> None:
        for index, history in enumerate(result.team_history):
            if not history:
                continue
            minutes, values = _minutes(history), metric.compute(result.team_history, index)
            color = PLAYER_COLORS[index]
            axes.plot(minutes, values, color=color, label=_player_label(configuration, index))
            _direct_label(axes, minutes, values, f"player {index}", color)
        _finish_time_axes(axes, metric)

    return draw


def _match_description(result: MatchResult, configuration: MatchConfiguration) -> str:
    if result.outcome == "victory" and result.winner is not None:
        outcome = f"{_player_label(configuration, result.winner)} wins"
    else:
        outcome = {"draw": "draw", "timeout": "no winner (time limit)"}.get(result.outcome, result.outcome)
    minutes = result.game_minutes or 0.0
    return f"Seed {configuration.seed}, {configuration.map_name}: {outcome} after {minutes:.1f} game minutes"


def plot_match(result: MatchResult, configuration: MatchConfiguration, directory: Path) -> list[Path]:
    """Overview and one chart per metric of one match, in ``directory`` (usually <match folder>/charts)."""
    if not any(result.team_history):
        return []
    description = _match_description(result, configuration)
    paths = [
        _save_overview(
            directory / "overview.png",
            [_draw_match_metric(METRICS_BY_NAME[name], result, configuration) for name in MATCH_OVERVIEW],
            description,
        )
    ]
    for metric in METRICS:
        paths.append(
            _save_single(
                directory / metric.category / f"{metric.name}.png",
                _draw_match_metric(metric, result, configuration),
                description,
            )
        )
    return paths


# --- Aggregation over a batch -----------------------------------------------------------------------------


@dataclass
class _Timeline:
    """Common time grid of the matches of a batch, and which time steps have enough matches to aggregate."""

    played: list[MatchResult]
    grid: np.ndarray
    running: np.ndarray
    minimum_running: int

    @property
    def shown(self) -> np.ndarray:
        return self.running >= self.minimum_running

    @classmethod
    def of(cls, results: Sequence[MatchResult]) -> "_Timeline | None":
        played = [result for result in results if any(result.team_history)]
        if not played:
            return None
        ends = [max(_minutes(history)[-1] for history in result.team_history if history) for result in played]
        grid = np.arange(0, max(ends) + AGGREGATION_STEP_MINUTES, AGGREGATION_STEP_MINUTES)
        running = np.array([grid <= end for end in ends]).sum(axis=0)
        # Below this many matches still running, the median says little: curves stop there.
        return cls(played, grid, running, min(3, len(played)))


def _resample(minutes: np.ndarray, values: np.ndarray, grid: np.ndarray, hold_after_end: bool) -> np.ndarray:
    """Values of one match on the common time grid; NaN where undefined and, unless ``hold_after_end``,
    after the match ended (np.interp holds the last value by default)."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(grid.size, np.nan)
    resampled = np.interp(grid, minutes[finite], values[finite])
    resampled[grid < minutes[finite][0]] = np.nan
    if not hold_after_end:
        resampled[grid > minutes[-1]] = np.nan
    return resampled


def _draw_batch_metric(metric: Metric, timeline: _Timeline, configuration: MatchConfiguration) -> Draw:
    def draw(axes: Axes) -> None:
        shown, grid = timeline.shown, timeline.grid
        for index in range(len(configuration.players)):
            samples = np.array(
                [
                    _resample(
                        _minutes(result.team_history[index]),
                        metric.compute(result.team_history, index),
                        grid,
                        hold_after_end=metric.is_total,
                    )
                    for result in timeline.played
                    if result.team_history[index]
                ]
            )
            # Time steps where no match has a value give "all-NaN slice" warnings; they are hidden anyway.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low, median, high = np.nanpercentile(samples, [25, 50, 75], axis=0)
            color = PLAYER_COLORS[index]
            axes.fill_between(grid[shown], low[shown], high[shown], color=color, alpha=0.15, linewidth=0)
            axes.plot(grid[shown], median[shown], color=color, label=_player_label(configuration, index, False))
            _direct_label(axes, grid[shown], median[shown], f"player {index}", color)
        _finish_time_axes(axes, metric)

    return draw


def _draw_matches_running(timeline: _Timeline) -> Draw:
    def draw(axes: Axes) -> None:
        axes.step(timeline.grid, timeline.running, where="post", color=NEUTRAL_COLOR)
        axes.axhline(timeline.minimum_running, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
        axes.set_title("Matches still running (below the dashed line, aggregated curves stop)")
        axes.set_xlabel("game minutes")
        axes.set_ylim(0, len(timeline.played) + 0.5)
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))

    return draw


def _draw_wins(configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult]) -> Draw:
    def draw(axes: Axes) -> None:
        number_of_players = len(configurations[0].players)
        labels = [_player_label(configurations[0], index, False) + " wins" for index in range(number_of_players)]
        values = [sum(result.winner == index for result in results) for index in range(number_of_players)]
        colors = PLAYER_COLORS[:number_of_players]
        for outcome, label in [("draw", "draws"), ("timeout", "time limit"), ("crash", "crashes")]:
            labels.append(label)
            values.append(sum(result.outcome == outcome for result in results))
            colors.append(NEUTRAL_COLOR)
        positions = np.arange(len(labels))[::-1]
        axes.barh(positions, values, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        for position, value in zip(positions, values, strict=True):
            text = f"{value}  ({100 * value / len(results):.0f} %)"
            axes.annotate(text, (value, position), xytext=(4, 0), textcoords="offset points", va="center")
        axes.set_yticks(positions, labels)
        axes.set_title(f"Outcomes of {len(results)} matches")
        axes.set_xlabel("matches")
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.grid(axis="y", visible=False)
        axes.margins(x=0.3)

    return draw


def _draw_durations(results: Sequence[MatchResult]) -> Draw:
    def draw(axes: Axes) -> None:
        for match_index, result in enumerate(results):
            if result.game_minutes is None:
                continue
            winner = result.winner if result.outcome == "victory" else None
            color = NEUTRAL_COLOR if winner is None else PLAYER_COLORS[winner]
            axes.plot([match_index], [result.game_minutes], "o", markersize=8, color=color, markeredgecolor=SURFACE)
        axes.set_title("Duration of each match (color: winner, gray: none)")
        axes.set_xlabel("match")
        axes.set_ylabel("game minutes")
        axes.set_ylim(bottom=0)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    return draw


def _draw_final_statistic(
    key: str, configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult]
) -> Draw:
    def draw(axes: Axes) -> None:
        number_of_players = len(configurations[0].players)
        random = np.random.default_rng(0)  # fixed jitter, so charts do not change between runs
        for index in range(number_of_players):
            values = np.array(
                [
                    result.team_statistics[index][key]
                    for result in results
                    if len(result.team_statistics) > index and key in result.team_statistics[index]
                ],
                dtype=float,
            )
            if values.size == 0:
                continue
            color = PLAYER_COLORS[index]
            jitter = random.uniform(-0.12, 0.12, values.size)
            axes.plot(index + jitter, values, "o", markersize=8, color=color, markeredgecolor=SURFACE, alpha=0.85)
            axes.plot([index - 0.25, index + 0.25], [values.mean()] * 2, color=TEXT_PRIMARY, linewidth=2)
        axes.set_title(STATISTIC_NAMES[key][1])
        axes.set_xticks(range(number_of_players), [f"player {index}" for index in range(number_of_players)])
        axes.set_xlim(-0.6, number_of_players - 0.4)
        axes.grid(axis="x", visible=False)
        _style_value_axis(axes)

    return draw


def plot_batch(
    configurations: Sequence[MatchConfiguration], results: Sequence[MatchResult], directory: Path
) -> list[Path]:
    """Charts of every match (in <match folder>/charts) and charts aggregated over the batch in ``directory``."""
    paths: list[Path] = []
    for configuration, result in zip(configurations, results, strict=True):
        paths += plot_match(result, configuration, result.run_directory / "charts")

    paths.append(_save_single(directory / "outcomes" / "wins.png", _draw_wins(configurations, results)))
    paths.append(_save_single(directory / "outcomes" / "durations.png", _draw_durations(results)))

    end_of_match = directory / "end_of_match"
    for key in SUMMARISED_STATISTICS:
        path = end_of_match / f"{STATISTIC_NAMES[key][0]}.png"
        paths.append(_save_single(path, _draw_final_statistic(key, configurations, results), "One dot per match"))
    columns = 4
    rows = -(-len(SUMMARISED_STATISTICS) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.75 * rows), layout="constrained")
    for key, panel in zip(SUMMARISED_STATISTICS, axes.ravel(), strict=False):
        _draw_final_statistic(key, configurations, results)(panel)
    title = "Statistics at the end of each match (dot: one match, black line: mean)"
    paths.append(_save(figure, end_of_match / "all.png", title))

    timeline = _Timeline.of(results)
    if timeline is None:
        return paths
    reference = configurations[0]
    band = (
        f"{len(timeline.played)} matches: median per player, band from 25th to 75th percentile; curves stop when "
        f"fewer than {timeline.minimum_running} matches are still running; totals of finished matches stay at their "
        "final value"
    )
    for metric in METRICS:
        paths.append(
            _save_single(
                directory / "over_time" / metric.category / f"{metric.name}.png",
                _draw_batch_metric(metric, timeline, reference),
                band,
            )
        )
    paths.append(_save_single(directory / "over_time" / "matches_running.png", _draw_matches_running(timeline)))

    metric_draws = [_draw_batch_metric(METRICS_BY_NAME[name], timeline, reference) for name in BATCH_OVERVIEW]
    overview = [*metric_draws[: len(KEY_METRICS)], _draw_wins(configurations, results), _draw_durations(results)]
    overview += metric_draws[len(KEY_METRICS) :]
    paths.append(_save_overview(directory / "overview.png", overview, band))
    return paths
