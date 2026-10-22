"""BeyondAllReasonModule's action distribution, without the engine."""

import pytest

pytest.importorskip("maxalaar_reinforcement_learning")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ray.rllib.core.models.catalog import Catalog  # noqa: E402

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import (  # noqa: E402
    _with_per_sample_deterministic_logp,
)

# As beyond_all_reason_environment's action space: a command and a target point.
ACTION_SPACE = gym.spaces.Dict({"command": gym.spaces.Discrete(5), "target": gym.spaces.Box(0, 1, (2,), np.float32)})


def test_deterministic_actions_have_one_log_probability_per_sample():
    # What RLlib's GetActions does without exploration (evaluation, display_episodes.py): it crashed with
    # "The size of tensor a (5) must match the size of tensor b (2)".
    distribution_class = _with_per_sample_deterministic_logp(
        Catalog._get_dist_cls_from_action_space(action_space=ACTION_SPACE, framework="torch")
    )
    logits = torch.randn(5, distribution_class.required_input_dim(ACTION_SPACE))
    distribution = distribution_class.from_logits(logits)

    deterministic = distribution.to_deterministic()
    actions = deterministic.sample()
    assert deterministic.logp(actions).shape == (5,)
    assert torch.equal(actions["command"], logits[:, :5].argmax(dim=-1))
    assert distribution.logp(distribution.sample()).shape == (5,)  # sampling (training) is unchanged
