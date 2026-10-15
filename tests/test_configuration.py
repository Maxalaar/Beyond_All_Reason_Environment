"""Fast tests that do not start the game engine."""

from pathlib import Path

import pytest

from beyond_all_reason_environment import ArtificialIntelligencePlayer, BatchConfiguration, MatchConfiguration
from beyond_all_reason_environment.batch import load_batch_configuration, new_version_directory
from beyond_all_reason_environment.start_script import build_start_script

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TWO_PLAYERS = [ArtificialIntelligencePlayer(side="Armada"), ArtificialIntelligencePlayer(side="Cortex")]


def test_start_script_lifts_speed_limit_and_stops_at_time_limit():
    script = build_start_script(MatchConfiguration(seed=7, speed=500, maximum_game_minutes=2))
    assert "FixedRNGSeed=7;" in script
    assert "MaxSpeed=1000;" in script
    assert "debugcommands=1:setspeed 500|3600:quitforce;" in script
    assert script.count("ShortName=BARb;") == 2


def test_each_player_keeps_its_side_and_seeds_follow():
    batch = BatchConfiguration(name="test", players=TWO_PLAYERS, matches=3)
    matches = batch.match_configurations()
    assert [[player.side for player in match.players] for match in matches] == [["Armada", "Cortex"]] * 3
    assert [match.seed for match in matches] == [1, 2, 3]


def test_list_of_sides_is_drawn_reproducibly_per_match():
    players = [ArtificialIntelligencePlayer(side=["Armada", "Cortex", "Legion"]), ArtificialIntelligencePlayer()]
    batch = BatchConfiguration(name="test", players=players, matches=30)
    drawn = [match.players[0].side for match in batch.match_configurations()]
    assert drawn == [match.players[0].side for match in batch.match_configurations()]
    assert set(drawn) == {"Armada", "Cortex", "Legion"}


def test_legion_enables_the_experimental_faction():
    players = [ArtificialIntelligencePlayer(side="Legion"), ArtificialIntelligencePlayer(side="Cortex")]
    assert "experimentallegionfaction=1;" in build_start_script(MatchConfiguration(players=players))
    assert "experimentallegionfaction" not in build_start_script(MatchConfiguration())


def test_unknown_side_is_rejected():
    with pytest.raises(ValueError):
        ArtificialIntelligencePlayer(side="Arm")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "arguments",
    [
        {"name": "has space", "players": TWO_PLAYERS},
        {"name": "test", "players": TWO_PLAYERS[:1]},
        {"name": "test", "players": TWO_PLAYERS, "parallel": 0},
    ],
)
def test_invalid_batch_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        BatchConfiguration(**arguments)


def test_configuration_files_load():
    for path in (REPOSITORY_ROOT / "configurations").glob("*.py"):
        assert isinstance(load_batch_configuration(path), BatchConfiguration)


def test_file_without_configuration_variable_is_rejected(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("x = 1\n")
    with pytest.raises(ValueError, match="configuration"):
        load_batch_configuration(path)


def test_each_run_of_a_name_gets_a_new_version(tmp_path):
    first, second = new_version_directory(tmp_path / "name"), new_version_directory(tmp_path / "name")
    assert (first.name, second.name) == ("version_1", "version_2")
    assert first.is_dir() and second.is_dir()


def test_list_of_maps_is_shared_equally():
    batch = BatchConfiguration(name="test", players=TWO_PLAYERS, matches=6, map_name=["Small", "Medium", "Large"])
    maps = [match.map_name for match in batch.match_configurations()]
    assert maps == ["Small", "Medium", "Large"] * 2
