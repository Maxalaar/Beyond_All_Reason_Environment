"""Run a few episodes through the RLlib wrapper to verify the interface.

Uses the 1v1 curriculum scenario (passive opponent) with the default observation, actions and rewards.
Each agent picks its allowed commands randomly and targets a random map point.

Examples:
    python scripts/try_combat_multi_agent_environment.py
    python scripts/try_combat_multi_agent_environment.py --episodes 5
"""

import argparse
import time

import numpy as np

from beyond_all_reason_environment import (
    Agent,
    GameConfiguration,
    Passive,
    ReinforcementLearningEnvironmentConfiguration,
)
from beyond_all_reason_environment.combat_multi_agent_environment import CombatMultiAgentEnvironment
from beyond_all_reason_environment.reinforcement_learning_environment_configuration import COMMANDS
from beyond_all_reason_environment.scenarios import one_versus_one


def random_action(observation: dict, random: np.random.Generator) -> dict:
    allowed = np.flatnonzero(observation["command_mask"])
    return {"command": int(random.choice(allowed)), "target": random.uniform(0, 1, 2).astype(np.float32)}


def attack_nearest_action(observation: dict) -> dict:
    """Attack the closest visible enemy; fall back to continue if none visible."""
    mask = observation["enemy_unit_mask"]
    state = observation["enemy_units_state"]
    for row in range(len(mask)):
        if mask[row]:
            x, z = float(state[row, 0]), float(state[row, 1])
            return {"command": COMMANDS.index("attack"), "target": np.array([x, z], np.float32)}
    return {"command": COMMANDS.index("continue"), "target": np.zeros(2, np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--policy", choices=["random", "attack"], default="attack")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    random = np.random.default_rng(arguments.seed)

    game = GameConfiguration(one_versus_one, controllers=[Agent(), Passive()])
    configuration = ReinforcementLearningEnvironmentConfiguration(game=game)
    env = CombatMultiAgentEnvironment(env_config={"configuration": configuration})
    print(f"agents: {env.possible_agents}")
    first_agent = env.possible_agents[0]
    print(f"observation_space keys: {sorted(env.observation_spaces[first_agent].spaces)}")
    print(f"action_space: {env.action_spaces[first_agent]}")

    start = time.monotonic()
    steps = 0
    for episode in range(arguments.episodes):
        observations, _ = env.reset(seed=arguments.seed + episode)
        if episode == 0:
            start, steps = time.monotonic(), 0
        total_reward, length = 0.0, 0
        while env._env.agents:
            if arguments.policy == "attack":
                actions = {agent: attack_nearest_action(observations[agent]) for agent in env._env.agents}
            else:
                actions = {agent: random_action(observations[agent], random) for agent in env._env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += next(iter(rewards.values()), 0.0)
            length += 1
            steps += 1
        info = next(iter(infos.values()), {})
        outcome = (
            "victory"
            if info.get("enemies_alive", 1) == 0
            else "defeat"
            if info.get("allies_alive", 1) == 0
            else "timeout"
        )
        print(
            f"episode {episode}: {length} steps, {info.get('episode_frame', 0) / 30:.1f} s, "
            f"reward {total_reward:+.2f}, outcome={outcome}"
        )

    elapsed = time.monotonic() - start
    print(f"\n{steps} steps in {elapsed:.1f} s → {steps / elapsed:.0f} steps/s (engine start excluded)")
    env.close()


if __name__ == "__main__":
    main()
