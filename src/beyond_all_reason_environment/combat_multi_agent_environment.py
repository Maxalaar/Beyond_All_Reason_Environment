"""RLlib MultiAgentEnv wrapper for CombatEnvironment.

Thin adapter: no business logic, pure delegation. Configuration is passed via
``env_config["configuration"]`` (a ReinforcementLearningEnvironmentConfiguration instance);
falls back to defaults if absent.

All agents share the same observation and action spaces (homogeneous, shared policy).
``__all__`` in terminations is True when a game-over condition is met (one side wiped
out); ``__all__`` in truncations is True when the episode ends by time limit only.
"""

from ray.rllib import MultiAgentEnv

from beyond_all_reason_environment.combat_environment import CombatEnvironment
from beyond_all_reason_environment.reinforcement_learning_environment_configuration import (
    ReinforcementLearningEnvironmentConfiguration,
)


class CombatMultiAgentEnvironment(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        configuration: ReinforcementLearningEnvironmentConfiguration = (env_config or {}).get(
            "configuration"
        ) or ReinforcementLearningEnvironmentConfiguration()
        self._env = CombatEnvironment(configuration)
        self.possible_agents = list(self._env.possible_agents)
        per_agent_obs = self._env.observation_space(self._env.possible_agents[0])
        per_agent_act = self._env.action_space(self._env.possible_agents[0])
        self.observation_spaces = {aid: per_agent_obs for aid in self.possible_agents}
        self.action_spaces = {aid: per_agent_act for aid in self.possible_agents}

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = self._env.step(action_dict)
        terminations["__all__"] = any(terminations.values())
        truncations["__all__"] = any(truncations.values()) and not terminations["__all__"]
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self._env.close()
