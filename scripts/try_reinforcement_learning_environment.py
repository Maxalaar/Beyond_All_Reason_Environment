"""Play a few episodes of the reinforcement learning environment with a simple policy, to check it and measure its
speed.

Policies:
    random    each agent picks an allowed command and a random point of the map
    attack    each agent clicks "attack" on the nearest enemy (should win against a passive opponent)

Examples:
    python scripts/try_reinforcement_learning_environment.py --policy attack --episodes 5
    python scripts/try_reinforcement_learning_environment.py --policy random --opponent HoldAndFire
    python scripts/try_reinforcement_learning_environment.py --scenario one_versus_one --opponent AttackNearest
    python scripts/try_reinforcement_learning_environment.py --opponent Agent   # self-play: the policy plays both sides
    python scripts/try_reinforcement_learning_environment.py --policy attack --graphical --speed 2   # watch it
    python scripts/try_reinforcement_learning_environment.py --scenario one_versus_one --graphical --speed 2
"""

import argparse
import time

import numpy as np

from beyond_all_reason_environment import (
    Agent,
    AttackNearest,
    DoNothing,
    EngineConfiguration,
    GameConfiguration,
    GraphicsConfiguration,
    HoldAndFire,
    Passive,
    ReinforcementLearningEnvironmentConfiguration,
    scenarios,
)
from beyond_all_reason_environment.reinforcement_learning_environment import (
    ReinforcementLearningEnvironment,
    state_feature_names,
)
from beyond_all_reason_environment.reinforcement_learning_environment_configuration import COMMANDS

ATTACK = COMMANDS.index("attack")
# Agent: the opponents are agents too, playing the same policy (self-play).
OPPONENTS = {
    controller.label: controller for controller in (DoNothing(), Passive(), HoldAndFire(), AttackNearest(), Agent())
}


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
    parser.add_argument("--scenario", choices=["five_versus_five", "one_versus_one"], default="five_versus_five")
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="Passive", help="controller of the opponents")
    parser.add_argument("--graphical", action="store_true")
    parser.add_argument("--speed", type=float, default=None, help="with --graphical, multiple of real time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-run-directory", action="store_true", help="keep the engine log after the run")
    arguments = parser.parse_args()

    scenario = getattr(scenarios, arguments.scenario)
    # The agents play team 0, the opponent controller every other team.
    controllers = [Agent(), *[OPPONENTS[arguments.opponent]] * (len(scenario.teams) - 1)]
    configuration = ReinforcementLearningEnvironmentConfiguration(
        game=GameConfiguration(scenario, controllers),
        engine=EngineConfiguration(
            speed=arguments.speed,
            graphics=GraphicsConfiguration() if arguments.graphical else None,
            keep_run_directory=arguments.keep_run_directory,
        ),
    )
    state_names = state_feature_names()
    x_idx, z_idx = state_names.index("x"), state_names.index("z")
    random = np.random.default_rng(arguments.seed)
    start = time.monotonic()
    with ReinforcementLearningEnvironment(configuration) as environment:
        # Reward and information of the first agent team (with self-play, the other side's are the opposite).
        followed = environment.possible_agents[0].split("_unit_")[0] + "_unit_"
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
                total_reward += next((r for agent, r in rewards.items() if agent.startswith(followed)), 0.0)
                length += 1
                steps += 1
            information = next(i for agent, i in informations.items() if agent.startswith(followed))
            print(
                f"episode {episode}: {length} steps, {information['episode_frame'] / 30:.1f} game seconds, "
                f"team reward {total_reward:+.2f}, allies alive {information['allies_alive']}, "
                f"enemies alive {information['enemies_alive']}, late actions {information['late_actions']}, "
                f"maximum lead {information['maximum_lead_frames']} frames"
            )
        elapsed = time.monotonic() - start
        print(f"{steps} steps in {elapsed:.1f} s: {steps / elapsed:.0f} steps per second (engine start-up excluded)")
        print(f"run directory: {environment.run_directory}")


if __name__ == "__main__":
    main()
