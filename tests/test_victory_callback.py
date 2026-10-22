"""Victory and defeat rates of VictoryCallback, on episodes written by hand (no engine)."""

import pytest

pytest.importorskip("ray")

from ray.rllib.env.multi_agent_episode import MultiAgentEpisode  # noqa: E402
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger  # noqa: E402

from beyond_all_reason_reinforcement_learning.victory_callback import WINDOW, VictoryCallback  # noqa: E402


def information(frame: int, allies_alive: int, enemies_alive: int) -> dict:
    return {"episode_frame": frame, "allies_alive": allies_alive, "enemies_alive": enemies_alive}


def two_agent_episode(final_allies_alive: int, final_enemies_alive: int) -> MultiAgentEpisode:
    """Agent a dies at frame 32; b lives on to frame 64, where the episode ends with these counts."""
    episode = MultiAgentEpisode()
    episode.add_env_reset(observations={"a": 0, "b": 0}, infos={"a": information(0, 2, 2), "b": information(0, 2, 2)})
    episode.add_env_step(
        observations={"a": 1, "b": 1},
        actions={"a": 0, "b": 0},
        rewards={"a": 0, "b": 0},
        infos={"a": information(32, 1, 2), "b": information(32, 1, 2)},
        terminateds={"a": True},
        truncateds={},
    )
    episode.add_env_step(
        observations={"b": 2},
        actions={"b": 0},
        rewards={"b": 0},
        infos={"b": information(64, final_allies_alive, final_enemies_alive)},
        terminateds={"b": True, "__all__": True},
        truncateds={},
    )
    return episode


def rates(callback: VictoryCallback, episodes: list) -> tuple[float, float]:
    logger = MetricsLogger()
    for episode in episodes:
        callback.on_episode_end(episode=episode, metrics_logger=logger, env_index=0)
    return logger.peek("victory_rate"), logger.peek("defeat_rate")


def test_the_latest_information_decides_victory_defeat_or_draw():
    assert rates(VictoryCallback(), [two_agent_episode(1, 0)]) == (1.0, 0.0)  # a died, b won
    assert rates(VictoryCallback(), [two_agent_episode(0, 1)]) == (0.0, 1.0)
    assert rates(VictoryCallback(), [two_agent_episode(1, 1)]) == (0.0, 0.0)  # time limit


def test_the_rates_cover_the_last_episodes_across_iterations():
    callback = VictoryCallback()
    rates(callback, [two_agent_episode(1, 0)] * 30)  # an earlier iteration: 30 victories
    # The next iteration (a new logger, as RLlib's per iteration) still counts them.
    assert rates(callback, [two_agent_episode(1, 1)] * 10) == (30 / 40, 0.0)
    # Only the last WINDOW episodes count.
    assert rates(callback, [two_agent_episode(1, 1)] * WINDOW) == (0.0, 0.0)
