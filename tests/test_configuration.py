"""Fast tests that do not start the game engine."""

from pathlib import Path

import pytest

from beyond_all_reason_environment import ArtificialIntelligencePlayer, BatchConfiguration, MatchConfiguration
from beyond_all_reason_environment.batch import load_batch_configuration
from beyond_all_reason_environment.start_script import build_start_script

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TWO_PLAYERS = [ArtificialIntelligencePlayer(side="Armada"), ArtificialIntelligencePlayer(side="Cortex")]


def test_start_script_lifts_speed_limit_and_stops_at_time_limit():
    script = build_start_script(MatchConfiguration(seed=7, speed=500, maximum_game_minutes=2))
    assert "FixedRNGSeed=7;" in script
    assert "MaxSpeed=1000;" in script
    assert "debugcommands=1:setspeed 500|3600:quitforce;" in script
    assert script.count("ShortName=BARb;") == 2


def test_alternate_sides_swaps_every_other_match():
    batch = BatchConfiguration(name="test", players=TWO_PLAYERS, matches=3, alternate_sides=True)
    sides = [[player.side for player in match.players] for match in batch.match_configurations()]
    assert sides == [["Armada", "Cortex"], ["Cortex", "Armada"], ["Armada", "Cortex"]]
    assert [match.seed for match in batch.match_configurations()] == [1, 2, 3]


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
