"""Fast tests that do not start the game engine."""

import argparse
import json
from pathlib import Path

import pytest

from beyond_all_reason_environment import (
    AllyteamConfiguration,
    ArtificialIntelligencePlayer,
    AttackNearest,
    BatchConfiguration,
    BuiltInArtificialIntelligence,
    MatchConfiguration,
    ScenarioConfiguration,
)
from beyond_all_reason_environment.batch import (
    DEFAULT_CONFIGURATION_FILE,
    add_configuration_arguments,
    configuration_from_arguments,
    load_batch_configuration,
    new_version_directory,
    save_configuration,
)
from beyond_all_reason_environment.game_extension import extension_game_name
from beyond_all_reason_environment.installation import GAME_VERSION
from beyond_all_reason_environment.scenarios import one_versus_one
from beyond_all_reason_environment.start_script import build_start_script

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TWO_SIDES = ["Armada", "Cortex"]


def scenario(sides: list = TWO_SIDES, **fields) -> ScenarioConfiguration:
    """BARb with a normal start (commander) for each side."""
    return ScenarioConfiguration(
        allyteams=[AllyteamConfiguration(BuiltInArtificialIntelligence("BARb"), side=side) for side in sides], **fields
    )


def test_start_script_lifts_speed_limit_and_stops_at_time_limit():
    script = build_start_script(MatchConfiguration(seed=7, speed=500, maximum_game_minutes=2))
    assert "FixedRNGSeed=7;" in script
    assert "MaxSpeed=1000;" in script
    assert "debugcommands=1:setspeed 500|3600:quitforce;" in script
    assert script.count("ShortName=BARb;") == 2


def test_each_player_keeps_its_side_and_seeds_follow():
    batch = BatchConfiguration(name="test", scenario=scenario(), matches=3)
    matches = batch.match_configurations()
    assert [[player.side for player in match.players] for match in matches] == [["Armada", "Cortex"]] * 3
    assert [match.seed for match in matches] == [1, 2, 3]


def test_list_of_sides_is_drawn_reproducibly_per_match():
    batch = BatchConfiguration(name="test", scenario=scenario([["Armada", "Cortex", "Legion"], "Armada"]), matches=30)
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
    "build",
    [
        lambda: BatchConfiguration(name="has space", scenario=scenario()),
        lambda: BatchConfiguration(name="test", scenario=scenario(TWO_SIDES[:1])),
        lambda: BatchConfiguration(name="test", scenario=scenario(), parallel=0),
        # Nobody would play the agent without Python.
        lambda: BatchConfiguration(name="test", scenario=one_versus_one),
    ],
)
def test_invalid_batch_configuration_is_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_placed_units_and_commanders_do_not_mix():
    with pytest.raises(ValueError, match="placed units"):
        ScenarioConfiguration(
            allyteams=[
                AllyteamConfiguration(AttackNearest(), units={"armpw": 1}),
                AllyteamConfiguration(AttackNearest()),
            ],
        )


def test_a_scenario_with_placed_units_is_set_up_by_the_game_extension():
    batch = BatchConfiguration(name="test", scenario=one_versus_one.with_controller(0, AttackNearest()))
    match = batch.match_configuration(0)
    assert [player.display_name for player in match.players] == ["AttackNearest", "Passive"]
    assert match.game_version == extension_game_name()
    assert match.extra_mod_options["deathmode"] == "killall"
    assert match.extra_mod_options["reinforcement_learning_controllers"] == "0,attack_nearest,30"
    spawns, states = match.extra_mod_options["reinforcement_learning_scenario"].split("|")
    # AttackNearest fires at will and manoeuvres; the Passive Pawn holds fire and position.
    assert spawns.count("/") == 1 and states == "2,1,0,0"
    assert ";" not in match.extra_mod_options["reinforcement_learning_scenario"]
    assert batch.match_configuration(0) == match


def test_a_normal_start_uses_the_plain_game():
    match = BatchConfiguration(name="test", scenario=scenario()).match_configuration(0)
    assert (match.game_version, match.extra_mod_options) == (GAME_VERSION, {})


def test_configuration_files_load():
    for path in (REPOSITORY_ROOT / "configurations" / "matches").glob("*.py"):
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
    batch = BatchConfiguration(name="test", scenario=scenario(map_name=["Small", "Medium", "Large"]), matches=6)
    maps = [match.map_name for match in batch.match_configurations()]
    assert maps == ["Small", "Medium", "Large"] * 2


def parse(command_line: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_configuration_arguments(parser)
    return parser.parse_args(command_line)


def test_without_file_the_default_configuration_is_used():
    assert configuration_from_arguments(parse([])) == load_batch_configuration(DEFAULT_CONFIGURATION_FILE)


def test_command_line_replaces_only_the_fields_given():
    path = REPOSITORY_ROOT / "configurations" / "matches" / "barb_versus_barb.py"
    file = load_batch_configuration(path)
    configuration = configuration_from_arguments(
        parse([str(path), "--matches", "3", "--maximum_game_minutes", "2.5", "--no-record_replays"])
    )
    assert (configuration.matches, configuration.record_replays) == (3, False)
    assert configuration.scenario.maximum_game_minutes == 2.5
    assert (configuration.name, configuration.parallel, configuration.scenario.players) == (
        file.name,
        file.parallel,
        file.scenario.players,
    )


def test_one_map_stays_a_name_and_several_make_a_list():
    assert configuration_from_arguments(parse(["--map_name", "Small"])).scenario.map_name == "Small"
    several = configuration_from_arguments(parse(["--map_name", "Small", "Large"]))
    assert several.scenario.map_name == ["Small", "Large"]


def test_replaced_values_are_checked_like_the_file():
    with pytest.raises(ValueError):
        configuration_from_arguments(parse(["--parallel", "0"]))


def test_one_match_of_a_series_matches_the_batch_and_may_go_beyond_it():
    batch = BatchConfiguration(name="test", scenario=scenario(map_name=["Small", "Large"]), matches=2, first_seed=5)
    assert batch.match_configuration(1) == batch.match_configurations()[1]
    beyond = batch.match_configuration(2)
    assert (beyond.seed, beyond.map_name) == (7, "Small")


def test_saved_configuration_has_the_values_after_the_command_line(tmp_path):
    arguments = parse(["--matches", "3"])
    save_configuration(tmp_path, configuration_from_arguments(arguments), arguments)
    assert json.loads((tmp_path / "configuration.json").read_text())["configuration"]["matches"] == 3
    assert (tmp_path / "configuration.py").read_text() == DEFAULT_CONFIGURATION_FILE.read_text()
