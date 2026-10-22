from collections import deque

from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Episodes the rates are averaged over, on each environment runner.
WINDOW = 100


class VictoryCallback(DefaultCallbacks):
    """Victory and defeat rates of the agents' alliance over the last WINDOW episodes of each environment runner
    (env_runners/victory_rate, env_runners/defeat_rate; RLlib then averages the runners).

    The episode return does not give them once agents can die: a dead agent misses the victory reward. The latest
    information of the episode (beyond_all_reason_environment: allies_alive, enemies_alive) tells who is left; with
    both alive the episode reached its time limit, a draw.

    The window is kept here: RLlib's own window (log_value(window=...)) restarts at every training iteration, so a
    stopping criterion on it would judge ~40 episodes only.
    """

    def __init__(self):
        super().__init__()
        # (victory, defeat) of the last episodes of this environment runner: the callback lives as long as it.
        self._outcomes: deque[tuple[float, float]] = deque(maxlen=WINDOW)

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        informations = [
            agent_episode.get_infos(-1)
            for agent_episode in episode.agent_episodes.values()
            if len(agent_episode.infos) > 0
        ]
        if not informations or metrics_logger is None:
            return
        last = max(informations, key=lambda information: information.get("episode_frame", 0))
        allies_alive, enemies_alive = last.get("allies_alive", 0), last.get("enemies_alive", 0)
        self._outcomes.append(
            (float(allies_alive > 0 and enemies_alive == 0), float(allies_alive == 0 and enemies_alive > 0))
        )

        # window=1: the iteration reports the rates after its last episode.
        metrics_logger.log_value(
            "victory_rate", sum(v for v, _ in self._outcomes) / len(self._outcomes), reduce="mean", window=1
        )
        metrics_logger.log_value(
            "defeat_rate", sum(d for _, d in self._outcomes) / len(self._outcomes), reduce="mean", window=1
        )
        metrics_logger.log_value("victory_rate_episodes", len(self._outcomes), reduce="mean", window=1)
