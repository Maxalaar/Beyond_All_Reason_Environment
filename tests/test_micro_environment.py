"""Micro environment: encoding, rewards and spawns without the engine; one end-to-end check with it (marker
"engine", run with `pytest -m engine`)."""

from pathlib import Path

import numpy as np
import pytest

from beyond_all_reason_environment.game_extension import extension_game_name, write_game_extension
from beyond_all_reason_environment.installation import GAME_VERSION
from beyond_all_reason_environment.micro_configuration import COMMANDS, MicroScenarioConfiguration
from beyond_all_reason_environment.micro_environment import (
    SCALE_FEATURES,
    MicroEnvironment,
    encode_observation,
    entity_feature_names,
    spawn_positions,
    team_reward,
)

MAP_SIZE = (6144.0, 4096.0)


def unit(identifier: int, team: int, x: float, z: float, health: float = 370.0, command: str = "none") -> dict:
    return {
        "id": identifier,
        "team": team,
        "unit": "armpw",
        "x": x,
        "z": z,
        "velocity_x": 0.0,
        "velocity_z": 0.0,
        "health": health,
        "maximum_health": 370.0,
        "command": command,
        "target_x": None,
        "target_z": None,
    }


def raw_observation(units: list[dict]) -> dict:
    return {
        "episode": 0,
        "episode_frame": 64,
        "frames_since_previous_decision": 32,
        "units": units,
        "unit_types": {"armpw": {"speed": 87.0, "weapon_range": 180.0, "sight": 429.0, "maximum_health": 370.0}},
        "late_actions": 0,
    }


def test_observation_lists_the_nearest_entities_first():
    configuration = MicroScenarioConfiguration(maximum_entities_seen=3)
    names = entity_feature_names(configuration)
    raw = raw_observation(
        [unit(1, 0, 0.40, 0.50), unit(2, 0, 0.41, 0.50), unit(3, 1, 0.60, 0.50), unit(4, 1, 0.45, 0.50, 185.0)]
    )

    observation = encode_observation(raw, 1, "armpw", configuration, MAP_SIZE)

    assert observation["self"].shape == (len(names) + len(SCALE_FEATURES),)
    assert observation["entities"].shape == (3, len(names))
    assert observation["entity_mask"].tolist() == [1, 1, 1]
    order = [observation["entities"][row, names.index("x")] for row in range(3)]
    assert order == pytest.approx([0.41, 0.45, 0.60])
    assert observation["entities"][1, names.index("is_enemy")] == 1
    assert observation["entities"][1, names.index("health_fraction")] == pytest.approx(0.5)
    assert observation["entities"][1, names.index("relative_x")] == pytest.approx(0.05)
    # Scale: the weapon range as a fraction of the map width.
    scale = dict(zip(SCALE_FEATURES, observation["self"][len(names) :], strict=True))
    assert scale["weapon_range_x"] == pytest.approx(180 / 6144)
    assert scale["decision_interval_ratio"] == pytest.approx(1.0)


def test_dead_agent_gets_an_empty_observation_and_masks_follow_the_configuration():
    configuration = MicroScenarioConfiguration(commands={"default": ["continue", "move"]})
    observation = encode_observation(raw_observation([unit(2, 1, 0.5, 0.5)]), 1, "armpw", configuration, MAP_SIZE)

    assert not observation["self"].any() and not observation["entity_mask"].any()
    assert observation["command_mask"].tolist() == [float(command in ("continue", "move")) for command in COMMANDS]


def test_team_reward_counts_damage_kills_and_victory():
    configuration = MicroScenarioConfiguration()
    before = raw_observation([unit(1, 0, 0.4, 0.5), unit(2, 1, 0.6, 0.5), unit(3, 1, 0.6, 0.6)])
    damaged = raw_observation([unit(1, 0, 0.4, 0.5, 300.0), unit(2, 1, 0.6, 0.5, 185.0), unit(3, 1, 0.6, 0.6)])
    won = raw_observation([unit(1, 0, 0.4, 0.5, 300.0)])

    reward = team_reward(before, damaged, 740.0, 370.0, 2, configuration)
    assert reward == pytest.approx(185 / 740 - 70 / 370)

    final = team_reward(damaged, won, 740.0, 370.0, 2, configuration)
    assert final == pytest.approx((185 + 370) / 740 + 0.5 * 2 / 2 + 1.0)


def test_spawns_put_agents_first_facing_the_opponent():
    configuration = MicroScenarioConfiguration(agent_units={"armpw": 3}, opponent_units={"corak": 2})
    spawns = spawn_positions(configuration, np.random.default_rng(0))

    assert [spawn["team"] for spawn in spawns] == [0, 0, 0, 1, 1]
    assert all(spawn["x"] < 0.5 for spawn in spawns[:3]) and all(spawn["x"] > 0.5 for spawn in spawns[3:])
    assert spawns == spawn_positions(configuration, np.random.default_rng(0))


def test_configuration_rejects_unsafe_delays_and_unknown_commands():
    with pytest.raises(ValueError):
        MicroScenarioConfiguration(action_delay_frames=20)
    with pytest.raises(ValueError):
        MicroScenarioConfiguration(commands={"default": ["dance"]})


def test_game_extension_depends_on_the_pinned_game(tmp_path: Path):
    assert write_game_extension(tmp_path) == extension_game_name()
    archive = next((tmp_path / "games").iterdir())
    assert GAME_VERSION in (archive / "modinfo.lua").read_text()
    assert (archive / "luarules" / "gadgets" / "reinforcement_learning_scenario.lua").exists()
    assert (archive / "luaui" / "widgets" / "reinforcement_learning_bridge.lua").exists()


@pytest.mark.engine
def test_attacking_the_nearest_enemy_beats_a_passive_opponent():
    configuration = MicroScenarioConfiguration(maximum_episode_seconds=45)
    names = entity_feature_names(configuration)
    is_enemy, x, z = (names.index(name) for name in ("is_enemy", "x", "z"))
    with MicroEnvironment(configuration) as environment:
        observations, _ = environment.reset(seed=0)
        assert set(observations) == set(environment.possible_agents)
        assert all(environment.observation_space(agent).contains(observations[agent]) for agent in observations)
        informations = {}
        while environment.agents:
            actions = {}
            for agent in environment.agents:
                entities, mask = observations[agent]["entities"], observations[agent]["entity_mask"]
                enemy = next(row for row in range(len(mask)) if mask[row] and entities[row, is_enemy])
                target = entities[enemy, [x, z]].astype(np.float32)
                actions[agent] = {"command": COMMANDS.index("attack"), "target": target}
            observations, _, _, _, informations = environment.step(actions)
        assert next(iter(informations.values()))["enemies_alive"] == 0
