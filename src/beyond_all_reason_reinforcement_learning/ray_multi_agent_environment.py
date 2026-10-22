"""RLlib MultiAgentEnv wrapper for ReinforcementLearningEnvironment.

Thin adapter: no business logic, pure delegation. Configuration is passed via
``env_config["configuration"]`` (a ReinforcementLearningEnvironmentConfiguration instance);
falls back to defaults if absent. A ``render_mode`` in env_config (any value, as RLlib evaluation passes it)
opens the game window at the graphical speed, like the Godot environments.

All agents share the same observation and action spaces (homogeneous, shared policy).
An agent whose unit dies is done on its own; the episode goes on for the others. ``__all__`` is
set once no agent is left: in truncations when the time limit ended the episode, otherwise in
terminations (one side wiped out).
"""

from dataclasses import replace

from ray.rllib import MultiAgentEnv

from beyond_all_reason_environment.reinforcement_learning_environment import ReinforcementLearningEnvironment
from beyond_all_reason_environment.reinforcement_learning_environment_configuration import (
    ReinforcementLearningEnvironmentConfiguration,
)


class RayMultiAgentEnvironment(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        configuration: ReinforcementLearningEnvironmentConfiguration = (env_config or {}).get(
            "configuration"
        ) or ReinforcementLearningEnvironmentConfiguration()
        if (env_config or {}).get("render_mode") is not None:
            # Asked to render (e.g. to watch a trained policy): the game window, at the graphical speed.
            engine = configuration.engine
            configuration = replace(
                configuration, engine=replace(engine, speed=None, graphics=replace(engine.graphics, show=True))
            )
        self._env = ReinforcementLearningEnvironment(configuration)
        self.possible_agents = list(self._env.possible_agents)
        per_agent_obs = self._env.observation_space(self._env.possible_agents[0])
        per_agent_act = self._env.action_space(self._env.possible_agents[0])
        self.observation_spaces = {aid: per_agent_obs for aid in self.possible_agents}
        self.action_spaces = {aid: per_agent_act for aid in self.possible_agents}

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = self._env.step(action_dict)
        # One agent done (its unit died) does not end the episode: only when the environment has no agent left.
        over = not self._env.agents
        truncations["__all__"] = over and any(truncations.values())
        terminations["__all__"] = over and not truncations["__all__"]
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self._env.close()
