"""Parquet tables and the statistics of the web interface, from a synthetic batch (no game engine)."""

import math
from pathlib import Path

import pytest

from beyond_all_reason_environment.analysis import aggregate_over_time, wilson_interval, win_rates
from beyond_all_reason_environment.tables import list_versions, read_tables, write_tables

from .test_plots import synthetic_batch


@pytest.fixture
def tables(tmp_path: Path):
    batch_directory = tmp_path / "experiment" / "version_1"
    configurations, results = synthetic_batch(batch_directory)
    write_tables(batch_directory, configurations, results)
    assert list_versions(tmp_path) == [batch_directory]
    return read_tables([batch_directory])


def test_tables_have_one_row_per_match_player_and_entry(tables):
    assert len(tables["matches"]) == 3
    assert len(tables["players"]) == 6
    assert tables["players"]["match_id"].iloc[0] == "experiment/version_1/000"
    assert {"metal_income", "units_alive", "relative_time"} <= set(tables["timeseries"].columns)
    assert tables["timeseries"]["relative_time"].max() == pytest.approx(1.0)
    assert tables["players"]["start_x"].notna().all()


def test_win_rates_count_each_group(tables):
    rates = win_rates(tables["players"], "player").set_index("player")
    assert rates.loc[0, "wins"] == 2 and rates.loc[1, "wins"] == 1
    assert rates.loc[0, "win_rate_low"] < rates.loc[0, "win_rate"] < rates.loc[0, "win_rate_high"]


def test_wilson_interval():
    low, high = wilson_interval(9, 12)
    assert 0.46 < low < 0.48 and 0.91 < high < 0.93
    assert all(math.isnan(value) for value in wilson_interval(0, 0))


def test_aggregated_totals_never_decrease(tables):
    aggregated = aggregate_over_time(tables["timeseries"], "metal_produced", relative=False, group_column="side")
    for _, frame in aggregated.groupby("side"):
        assert (frame["median"].diff().dropna() >= -1e-9).all()
