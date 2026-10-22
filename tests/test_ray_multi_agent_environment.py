"""RLlib adapter: when the episode ends for every agent (__all__), without the engine."""

import pytest

pytest.importorskip("ray")

from beyond_all_reason_environment.engine_configuration import GRAPHICAL_SPEED, HEADLESS_SPEED  # noqa: E402
from beyond_all_reason_environment.ray_multi_agent_environment import RayMultiAgentEnvironment  # noqa: E402


class ScriptedEnvironment:
    """Stands for ReinforcementLearningEnvironment: each step plays the next (terminations, truncations, agents left)."""

    def __init__(self, steps):
        self.agents = ["a", "b"]
        self._steps = iter(steps)

    def step(self, actions):
        terminations, truncations, self.agents = next(self._steps)
        done = {**terminations, **truncations}
        return {agent: None for agent in done}, {agent: 0.0 for agent in done}, terminations, truncations, {}


def adapter(steps) -> RayMultiAgentEnvironment:
    environment = RayMultiAgentEnvironment.__new__(RayMultiAgentEnvironment)
    environment._env = ScriptedEnvironment(steps)
    return environment


def test_the_death_of_one_agent_does_not_end_the_episode():
    environment = adapter([
        ({"a": True, "b": False}, {"a": False, "b": False}, ["b"]),  # a's unit dies
        ({"b": True}, {"b": False}, []),  # b wins or dies: the episode is over
    ])
    _, _, terminations, truncations, _ = environment.step({"a": {}, "b": {}})
    assert terminations == {"a": True, "b": False, "__all__": False}
    assert truncations["__all__"] is False
    _, _, terminations, truncations, _ = environment.step({"b": {}})
    assert terminations["__all__"] is True and truncations["__all__"] is False


def test_the_time_limit_truncates_the_episode():
    environment = adapter([({"a": False, "b": False}, {"a": True, "b": True}, [])])
    _, _, terminations, truncations, _ = environment.step({"a": {}, "b": {}})
    assert truncations["__all__"] is True and terminations["__all__"] is False


def test_a_render_mode_opens_the_game_window_at_the_graphical_speed():
    # As RLlib's evaluation passes it (display_episodes.py of Maxalaar_Reinforcement_Learning); no engine starts here.
    watched = RayMultiAgentEnvironment({"render_mode": "rgb_array"})._env.configuration.engine
    assert watched.graphics.show and watched.effective_speed == GRAPHICAL_SPEED
    trained = RayMultiAgentEnvironment({})._env.configuration.engine
    assert not trained.graphics.show and trained.effective_speed == HEADLESS_SPEED
