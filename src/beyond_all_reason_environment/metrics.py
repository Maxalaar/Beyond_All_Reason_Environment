"""Metrics computed from the engine statistics history of a match, shared by the charts and the tables.

The engine records cumulative statistics (metalProduced, damageDealt, unitsKilled...) every 15 s of game
time; each metric turns the histories of all players into one value per entry for one player.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Rates are averaged over this many entries (1 minute).
RATE_SMOOTHING_ENTRIES = 4

Histories = list[list[dict]]


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


def minutes_of(history: list[dict]) -> np.ndarray:
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
# Run overview: the key metrics, then the outcomes, durations and this one.
RUN_OVERVIEW = [*KEY_METRICS, "metal_produced_share"]
