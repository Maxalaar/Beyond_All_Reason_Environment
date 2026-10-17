"""Combat environment: encoding, rewards and spawns without the engine; one end-to-end check with it (marker
"engine", run with `pytest -m engine`)."""

from pathlib import Path

import numpy as np
import pytest

from beyond_all_reason_environment.combat_configuration import COMMANDS, AllyteamConfiguration, CombatConfiguration
from beyond_all_reason_environment.combat_environment import (
    CONTEXT_FEATURES,
    CombatEnvironment,
    chassis_feature_names,
    encode_observation,
    spawn_positions,
    state_feature_names,
    team_reward,
    weapon_feature_names,
)
from beyond_all_reason_environment.game_extension import extension_game_name, write_game_extension
from beyond_all_reason_environment.installation import GAME_VERSION

MAP_SIZE = (6144.0, 4096.0)
NUM_ARMOR_TYPES = 3  # small value for tests; armpw's armor_type=2 (index into the one-hot)


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
        "unit_types": {
            "armpw": {
                "speed": 87.0,
                "weapon_range": 180.0,
                "sight": 429.0,
                "maximum_health": 370.0,
                "armor_type": 2,
                "can_move": 1,
                "weapons": [
                    {
                        "range": 180.0,
                        "reload": 2.0,
                        "aoe": 12.0,
                        "paralyzed": 0,
                        "damage_per_type": [10.0, 20.0, 50.0],
                    }
                ],
            }
        },
        "late_actions": 0,
    }


def _config(num_armor_types: int = NUM_ARMOR_TYPES, **kwargs) -> CombatConfiguration:
    return CombatConfiguration(num_armor_types=num_armor_types, **kwargs)


def test_observation_separates_allies_and_enemies_nearest_first():
    configuration = _config(maximum_ally_units=3, maximum_enemy_units=3)
    chassis_names = chassis_feature_names(configuration)
    state_names = state_feature_names()
    # unit 1 = observer (team 0, x=0.40)
    # unit 2 = ally (team 0, x=0.41, nearest ally)
    # unit 3 = enemy (team 1, x=0.60, farther enemy)
    # unit 4 = enemy (team 1, x=0.45, 185 HP, nearest enemy)
    raw = raw_observation(
        [unit(1, 0, 0.40, 0.50), unit(2, 0, 0.41, 0.50), unit(3, 1, 0.60, 0.50), unit(4, 1, 0.45, 0.50, 185.0)]
    )

    observation = encode_observation(raw, 1, "armpw", configuration, MAP_SIZE)

    n_chassis = len(chassis_names)
    n_state = len(state_names)
    n_weapon = len(weapon_feature_names(NUM_ARMOR_TYPES))
    max_ally = configuration.maximum_ally_units
    max_enemy = configuration.maximum_enemy_units
    max_weapons = configuration.maximum_weapons_per_unit

    assert observation["self_chassis"].shape == (n_chassis,)
    assert observation["self_state"].shape == (n_state,)
    assert observation["context"].shape == (len(CONTEXT_FEATURES),)
    assert observation["ally_units_chassis"].shape == (max_ally, n_chassis)
    assert observation["ally_units_state"].shape == (max_ally, n_state)
    assert observation["enemy_units_chassis"].shape == (max_enemy, n_chassis)
    assert observation["enemy_units_state"].shape == (max_enemy, n_state)
    assert observation["self_weapons"].shape == (max_weapons, n_weapon)

    # Only one ally, two enemies.
    assert observation["ally_unit_mask"].tolist() == [1, 0, 0]
    assert observation["enemy_unit_mask"].tolist() == [1, 1, 0]

    x_idx = state_names.index("x")
    assert observation["ally_units_state"][0, x_idx] == pytest.approx(0.41)
    assert [observation["enemy_units_state"][row, x_idx] for row in range(2)] == pytest.approx([0.45, 0.60])

    # Nearest enemy (unit 4) checks.
    health_idx = state_names.index("health_fraction")
    rel_x_idx = state_names.index("relative_x")
    assert observation["enemy_units_state"][0, health_idx] == pytest.approx(0.5)
    # relative_x = (0.45 - 0.40) * MAP_SIZE[0] / DISTANCE_SCALE = 0.05 * 6144 / 512 = 0.6
    assert observation["enemy_units_state"][0, rel_x_idx] == pytest.approx(0.6)

    # Weapon present in self_weapons slot 0, absent in slot 1.
    assert observation["self_weapon_mask"][0] == 1.0
    assert observation["self_weapon_mask"][1] == 0.0

    # Context: weapon range as a fraction of the map width.
    context = dict(zip(CONTEXT_FEATURES, observation["context"], strict=True))
    assert context["decision_interval_ratio"] == pytest.approx(1.0)

    # Chassis: armor type one-hot for armpw (armor_type=2 → index 2 is 1.0).
    armor_idx = chassis_names.index("armor_type_2")
    assert observation["self_chassis"][armor_idx] == pytest.approx(1.0)
    assert observation["self_chassis"][chassis_names.index("armor_type_0")] == pytest.approx(0.0)

    # Weapon damage per armor type (damage_per_type=[10, 20, 50], normalized /1000).
    damage_idx = weapon_feature_names(NUM_ARMOR_TYPES).index("damage_armor_2")
    assert observation["self_weapons"][0, damage_idx] == pytest.approx(50.0 / 1000.0)


def test_dead_agent_gets_an_empty_observation_and_masks_follow_the_configuration():
    configuration = _config(commands={"default": ["continue", "move"]})
    observation = encode_observation(raw_observation([unit(2, 1, 0.5, 0.5)]), 1, "armpw", configuration, MAP_SIZE)

    assert not observation["self_chassis"].any()
    assert not observation["self_state"].any()
    assert not observation["ally_unit_mask"].any()
    assert not observation["enemy_unit_mask"].any()
    assert observation["command_mask"].tolist() == [float(command in ("continue", "move")) for command in COMMANDS]


def test_team_reward_counts_damage_kills_and_victory():
    configuration = CombatConfiguration()
    before = raw_observation([unit(1, 0, 0.4, 0.5), unit(2, 1, 0.6, 0.5), unit(3, 1, 0.6, 0.6)])
    damaged = raw_observation([unit(1, 0, 0.4, 0.5, 300.0), unit(2, 1, 0.6, 0.5, 185.0), unit(3, 1, 0.6, 0.6)])
    won = raw_observation([unit(1, 0, 0.4, 0.5, 300.0)])

    reward = team_reward(before, damaged, 740.0, 370.0, 2, configuration)
    assert reward == pytest.approx(185 / 740 - 70 / 370)

    final = team_reward(damaged, won, 740.0, 370.0, 2, configuration)
    assert final == pytest.approx((185 + 370) / 740 + 0.5 * 2 / 2 + 1.0)


def test_spawns_put_agents_first_facing_the_opponent():
    configuration = CombatConfiguration(
        allyteams=[
            AllyteamConfiguration(units={"armpw": 3}),
            AllyteamConfiguration(units={"corak": 2}),
        ]
    )
    spawns = spawn_positions(configuration, np.random.default_rng(0))

    assert [spawn["team"] for spawn in spawns] == [0, 0, 0, 1, 1]
    assert all(spawn["x"] < 0.5 for spawn in spawns[:3]) and all(spawn["x"] > 0.5 for spawn in spawns[3:])
    assert spawns == spawn_positions(configuration, np.random.default_rng(0))


def test_configuration_rejects_unknown_commands():
    with pytest.raises(ValueError):
        CombatConfiguration(commands={"default": ["dance"]})


def test_mobile_and_building_units_are_separated():
    configuration = _config(maximum_ally_units=2, maximum_enemy_units=2, maximum_ally_buildings=2)
    raw = {
        "episode": 0,
        "episode_frame": 0,
        "frames_since_previous_decision": 32,
        "units": [
            {**unit(1, 0, 0.5, 0.5), "unit": "armpw"},  # observer (mobile)
            {**unit(2, 0, 0.4, 0.5), "unit": "factory"},  # ally building
            {**unit(3, 0, 0.45, 0.5), "unit": "armpw"},  # ally mobile unit
        ],
        "unit_types": {
            "armpw": {
                "speed": 87.0,
                "weapon_range": 180.0,
                "sight": 429.0,
                "maximum_health": 370.0,
                "armor_type": 0,
                "can_move": 1,
                "weapons": [
                    {"range": 180.0, "reload": 2.0, "aoe": 12.0, "paralyzed": 0, "damage_per_type": [50.0, 50.0, 50.0]}
                ],
            },
            "factory": {
                "speed": 0.0,
                "weapon_range": 0.0,
                "sight": 300.0,
                "maximum_health": 5000.0,
                "armor_type": 0,
                "can_move": 0,
                "weapons": [],
            },
        },
        "late_actions": 0,
    }

    observation = encode_observation(raw, 1, "armpw", configuration, MAP_SIZE)

    # One ally mobile unit (unit 3), one ally building (unit 2), no enemies.
    assert observation["ally_unit_mask"].tolist()[:2] == [1.0, 0.0]
    assert observation["ally_building_mask"].tolist()[:2] == [1.0, 0.0]
    assert not observation["enemy_unit_mask"].any()


def test_game_extension_depends_on_the_pinned_game(tmp_path: Path):
    assert write_game_extension(tmp_path) == extension_game_name()
    archive = next((tmp_path / "games").iterdir())
    assert GAME_VERSION in (archive / "modinfo.lua").read_text()
    assert (archive / "luarules" / "gadgets" / "reinforcement_learning_scenario.lua").exists()


@pytest.mark.engine
def test_attacking_the_nearest_enemy_beats_a_passive_opponent():
    configuration = CombatConfiguration(maximum_episode_minutes=0.75)
    state_names = state_feature_names()
    x_idx, z_idx = state_names.index("x"), state_names.index("z")
    with CombatEnvironment(configuration) as environment:
        observations, _ = environment.reset(seed=0)
        assert set(observations) == set(environment.possible_agents)
        assert all(environment.observation_space(agent).contains(observations[agent]) for agent in observations)
        informations = {}
        while environment.agents:
            actions = {}
            for agent in environment.agents:
                enemy_state = observations[agent]["enemy_units_state"]
                enemy_mask = observations[agent]["enemy_unit_mask"]
                enemy_idx = next((i for i in range(len(enemy_mask)) if enemy_mask[i] > 0), None)
                if enemy_idx is not None:
                    target = enemy_state[enemy_idx, [x_idx, z_idx]].astype(np.float32)
                    actions[agent] = {"command": COMMANDS.index("attack"), "target": target}
                else:
                    actions[agent] = {"command": COMMANDS.index("continue"), "target": np.zeros(2, np.float32)}
            observations, _, _, _, informations = environment.step(actions)
        assert next(iter(informations.values()))["enemies_alive"] == 0
