"""Play a few episodes of the combat environment with a simple policy, to check it and measure its speed.

Policies:
    random    each agent picks an allowed command and a random point of the map
    attack    each agent clicks "attack" on the nearest enemy (should win against a passive opponent)

Examples:
    python scripts/try_combat_environment.py --policy attack --episodes 5
    python scripts/try_combat_environment.py --policy random --opponent hold_and_fire
    python scripts/try_combat_environment.py --policy attack --graphical --speed 2   # watch it
"""

import argparse
import time

import numpy as np

from beyond_all_reason_environment.combat_configuration import (
    COMMANDS,
    AllyteamConfiguration,
    CombatConfiguration,
    ScenarioConfiguration,
)
from beyond_all_reason_environment.combat_environment import CombatEnvironment, state_feature_names
from beyond_all_reason_environment.engine_configuration import EngineConfiguration

ATTACK = COMMANDS.index("attack")


def random_action(observation: dict, random: np.random.Generator) -> dict:
    allowed = np.flatnonzero(observation["command_mask"])
    return {"command": int(random.choice(allowed)), "target": random.uniform(0, 1, 2).astype(np.float32)}


def attack_nearest_action(observation: dict, x_idx: int, z_idx: int) -> dict:
    """Click "attack" on the nearest visible enemy (enemy_units sorted by distance)."""
    enemy_mask = observation["enemy_unit_mask"]
    enemy_state = observation["enemy_units_state"]
    for row in range(len(enemy_mask)):
        if enemy_mask[row] > 0:
            target = enemy_state[row, [x_idx, z_idx]].astype(np.float32)
            return {"command": ATTACK, "target": target}
    return {"command": COMMANDS.index("continue"), "target": np.zeros(2, np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["random", "attack"], default="attack")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--opponent", choices=["passive", "hold_and_fire"], default="passive")
    parser.add_argument("--graphical", action="store_true")
    parser.add_argument("--speed", type=float, default=None, help="with --graphical, multiple of real time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-run-directory", action="store_true", help="keep the engine log after the run")
    arguments = parser.parse_args()

    configuration = CombatConfiguration(
        scenario=ScenarioConfiguration(
            allyteams=[
                AllyteamConfiguration(units={"armpw": 5}),
                AllyteamConfiguration(units={"armpw": 5}, behaviour=arguments.opponent),
            ],
        ),
        engine=EngineConfiguration(
            graphical=arguments.graphical,
            speed=arguments.speed or (1.0 if arguments.graphical else 1000.0),
            keep_run_directory=arguments.keep_run_directory,
        ),
    )
    state_names = state_feature_names()
    x_idx, z_idx = state_names.index("x"), state_names.index("z")
    random = np.random.default_rng(arguments.seed)
    start = time.monotonic()
    with CombatEnvironment(configuration) as environment:
        steps = 0
        for episode in range(arguments.episodes):
            observations, informations = environment.reset(seed=arguments.seed + episode)
            if episode == 0:
                start, steps = time.monotonic(), 0  # leave the engine start-up out of the speed
            total_reward, length = 0.0, 0
            while environment.agents:
                actions = {
                    agent: random_action(observations[agent], random)
                    if arguments.policy == "random"
                    else attack_nearest_action(observations[agent], x_idx, z_idx)
                    for agent in environment.agents
                }
                observations, rewards, terminations, truncations, informations = environment.step(actions)
                total_reward += next(iter(rewards.values()))
                length += 1
                steps += 1
            information = next(iter(informations.values()))
            print(
                f"episode {episode}: {length} steps, {information['episode_frame'] / 30:.1f} game seconds, "
                f"team reward {total_reward:+.2f}, agents alive {information['agents_alive']}, "
                f"enemies alive {information['enemies_alive']}, late actions {information['late_actions']}, "
                f"maximum lead {information['maximum_lead_frames']} frames"
            )
        elapsed = time.monotonic() - start
        print(f"{steps} steps in {elapsed:.1f} s: {steps / elapsed:.0f} steps per second (engine start-up excluded)")
        print(f"run directory: {environment.run_directory}")


if __name__ == "__main__":
    main()
