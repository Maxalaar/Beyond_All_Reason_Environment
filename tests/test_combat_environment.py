"""Combat environment: encoding, rewards and spawns without the engine; one end-to-end check with it (marker
"engine", run with `pytest -m engine`)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from beyond_all_reason_environment.combat_configuration import (
    COMMANDS,
    ActionConfiguration,
    CombatConfiguration,
    ObservationConfiguration,
)
from beyond_all_reason_environment.combat_environment import (
    CONTEXT_FEATURES,
    NUMBER_OF_ARMOR_TYPES,
    CombatEnvironment,
    chassis_feature_names,
    encode_observation,
    merge_commands,
    relative_allyteams,
    state_feature_names,
    team_reward,
    visibility,
    weapon_feature_names,
)
from beyond_all_reason_environment.controllers import (
    Agent,
    AttackNearest,
    BuiltInArtificialIntelligence,
    DoNothing,
    Passive,
)
from beyond_all_reason_environment.game_extension import extension_game_name, write_game_extension
from beyond_all_reason_environment.installation import GAME_VERSION
from beyond_all_reason_environment.scenario import (
    ScenarioConfiguration,
    TeamConfiguration,
    controller_mod_options,
    spawn_positions,
    unit_states,
)
from beyond_all_reason_environment.scenarios import five_versus_five, one_versus_one

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


def test_observation_separates_allies_and_enemies_nearest_first():
    configuration = CombatConfiguration(
        observation=ObservationConfiguration(maximum_ally_units=3, maximum_enemy_units=3)
    )
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
    n_weapon = len(weapon_feature_names())
    max_ally = configuration.observation.maximum_ally_units
    max_enemy = configuration.observation.maximum_enemy_units
    max_weapons = configuration.observation.maximum_weapons_per_unit

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
    damage_idx = weapon_feature_names().index("damage_armor_2")
    assert observation["self_weapons"][0, damage_idx] == pytest.approx(50.0 / 1000.0)


def test_dead_agent_gets_an_empty_observation_and_masks_follow_the_configuration():
    configuration = CombatConfiguration(action=ActionConfiguration(commands={"default": ["continue", "move"]}))
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

    reward = team_reward(before, damaged, 740.0, 370.0, 2, configuration, ally_teams={0})
    assert reward == pytest.approx(185 / 740 - 70 / 370)

    final = team_reward(damaged, won, 740.0, 370.0, 2, configuration, ally_teams={0})
    assert final == pytest.approx((185 + 370) / 740 + 0.5 * 2 / 2 + 1.0)


def test_spawns_put_agents_first_facing_the_opponent():
    scenario = ScenarioConfiguration(
        teams=[
            TeamConfiguration(controller=Agent(), units={"armpw": 3}),
            TeamConfiguration(controller=DoNothing(), units={"corak": 2}),
        ]
    )
    spawns = spawn_positions(scenario, np.random.default_rng(0))

    assert [spawn["team"] for spawn in spawns] == [0, 0, 0, 1, 1]
    assert all(spawn["x"] < 0.5 for spawn in spawns[:3]) and all(spawn["x"] > 0.5 for spawn in spawns[3:])
    assert spawns == spawn_positions(scenario, np.random.default_rng(0))


def test_configuration_rejects_unknown_commands():
    with pytest.raises(ValueError):
        ActionConfiguration(commands={"default": ["dance"]})


def test_controllers_give_states_scripts_and_players():
    scenario = one_versus_one.with_controller(1, AttackNearest(interval_seconds=0.5))
    # Agent states come from ActionConfiguration; AttackNearest fires at will and manoeuvres.
    assert unit_states(scenario, (2, 1)) == [2, 1, 2, 1]
    assert controller_mod_options(scenario) == {"reinforcement_learning_controllers": "1,attack_nearest,15"}
    assert controller_mod_options(one_versus_one) == {}
    players = scenario.players
    assert [(player.short_name, player.display_name) for player in players] == [
        ("BridgeAI", "Agent"),
        ("NullAI", "AttackNearest"),
    ]
    assert BuiltInArtificialIntelligence("BARb").player("Cortex").display_name == "BARb"


def test_the_environment_needs_a_team_of_agents():
    with pytest.raises(ValueError, match="played by the agents"):
        CombatConfiguration(scenario=one_versus_one.with_controller(0, AttackNearest()))
    # The agents may play any team.
    CombatConfiguration(scenario=one_versus_one.with_controller(0, Passive()).with_controller(1, Agent()))


def test_every_agent_sees_its_own_alliance_as_zero_and_enemy_orders_hidden():
    configuration = CombatConfiguration(scenario=one_versus_one.with_controller(1, Agent()))
    state_names = state_feature_names()
    raw = raw_observation([unit(1, 0, 0.4, 0.5, command="move"), unit(2, 1, 0.6, 0.5, command="attack")])
    for observer, enemy_command in ((1, "attack"), (2, "move")):
        observation = encode_observation(raw, observer, "armpw", configuration, MAP_SIZE)
        enemy = observation["enemy_units_state"][0]
        assert observation["enemy_unit_mask"][0] == 1.0
        assert enemy[state_names.index("allyteam_id_normalized")] == pytest.approx(1.0)
        assert enemy[state_names.index("command_none")] == 1.0
        assert enemy[state_names.index(f"command_{enemy_command}")] == 0.0
        assert observation["self_state"][state_names.index("allyteam_id_normalized")] == 0.0
    assert relative_allyteams([0, 1, 1, 2], 1) == {1: 0, 0: 1, 2: 2}


def test_allied_teams_share_an_alliance_and_its_reward():
    scenario = ScenarioConfiguration(
        teams=[
            TeamConfiguration(Agent(), units={"armpw": 1}, allyteam=0),
            TeamConfiguration(AttackNearest(), units={"armpw": 1}, allyteam=0),
            TeamConfiguration(DoNothing(), units={"armpw": 2}),
        ],
        full_vision=True,
    )
    assert (scenario.allyteams, scenario.number_of_allyteams, scenario.units_per_allyteam()) == ([0, 0, 1], 2, [2, 2])
    observation = CombatConfiguration(scenario=scenario).observation
    assert (observation.maximum_ally_units, observation.maximum_enemy_units) == (1, 2)
    spawns = spawn_positions(scenario, np.random.default_rng(0))
    assert [spawn["team"] for spawn in spawns] == [0, 1, 2, 2]
    # Both teams of alliance 0 spawn on the same side, apart along z.
    assert spawns[0]["x"] == pytest.approx(spawns[1]["x"], abs=0.05) and spawns[0]["z"] != spawns[1]["z"]
    before = raw_observation([unit(1, 0, 0.4, 0.5), unit(2, 1, 0.4, 0.6), unit(3, 2, 0.6, 0.5)])
    after = raw_observation([unit(1, 0, 0.4, 0.5), unit(2, 1, 0.4, 0.6, 185.0)])
    configuration = CombatConfiguration(scenario=scenario)
    reward = team_reward(before, after, 370.0, 740.0, 1, configuration, ally_teams={0, 1})
    assert reward == pytest.approx(370 / 370 - 185 / 740 + 0.5 + 1.0)


def test_self_play_gives_one_agent_per_unit_of_each_agent_team():
    environment = CombatEnvironment(CombatConfiguration(scenario=five_versus_five.with_controller(1, Agent())))
    assert environment.possible_agents[:2] == ["team0_unit_0", "team0_unit_1"]
    assert environment.possible_agents[5:7] == ["team1_unit_0", "team1_unit_1"]
    players = environment._players(port=1234)
    # The BridgeAI talks to Python for both agent teams; the second one runs NullAI, the gadget gives its orders.
    assert [(player.short_name, player.display_name) for player in players] == [
        ("BridgeAI", "Agent"),
        ("NullAI", "Agent"),
    ]


def test_orders_and_visibility_come_from_the_raw_observation():
    raw = {"units": [unit(1, 0, 0.5, 0.5), unit(2, 1, 0.9, 0.5)], "commands": "1,move,0.250000,0.750000;2,none,,"}
    merge_commands(raw)
    assert (raw["units"][0]["command"], raw["units"][0]["target_x"]) == ("move", 0.25)
    assert (raw["units"][1]["command"], raw["units"][1]["target_x"]) == ("none", None)
    raw["unit_types"] = {"armpw": {"sight": 1000.0}}
    grid = visibility(raw, {0}, (4000.0, 4000.0), (4, 4))
    # Sight 1000 around (2000, 2000): the four central cells, not the corners.
    assert grid[1:3, 1:3].all() and not grid[0, 0] and not grid[3, 3]


def test_observation_counts_left_unset_follow_the_scenario():
    assert NUMBER_OF_ARMOR_TYPES == 15
    one = CombatConfiguration(scenario=one_versus_one).observation
    assert (one.maximum_ally_units, one.maximum_enemy_units, one.maximum_units_per_enemy_allyteam) == (0, 1, 1)
    five = CombatConfiguration(scenario=five_versus_five).observation
    assert (five.maximum_ally_units, five.maximum_enemy_units, five.maximum_units_per_enemy_allyteam) == (4, 5, 5)


def test_one_observation_configuration_can_serve_several_scenarios():
    shared = ObservationConfiguration(maximum_weapons_per_unit=1)
    CombatConfiguration(scenario=one_versus_one, observation=shared)
    assert shared.maximum_enemy_units is None
    five = CombatConfiguration(scenario=five_versus_five, observation=shared).observation
    assert (five.maximum_enemy_units, five.maximum_weapons_per_unit) == (5, 1)


def test_mobile_and_building_units_are_separated():
    configuration = CombatConfiguration(
        observation=ObservationConfiguration(maximum_ally_units=2, maximum_enemy_units=2, maximum_ally_buildings=2)
    )
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
    configuration = CombatConfiguration(scenario=replace(five_versus_five, maximum_game_minutes=0.75))
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
