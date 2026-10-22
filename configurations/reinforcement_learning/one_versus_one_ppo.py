"""PPO training configuration — 1v1 minimal combat (curriculum step 1).

Scenario (beyond_all_reason_environment.scenarios.one_versus_one): flat 1024×1024-elmo arena, 1 armpw (Pawn) on
each side, deterministic spawns at opposite edges.
Chosen here, for this experiment: the agent against a Passive Pawn (does not move, does not shoot), one weapon slot
per unit, every command allowed, and a terminal reward (+1 victory, -1 defeat, no intermediate shaping). Entity
counts follow the scenario (no ally, one enemy). Stops at 90 % victories.

Expected episode length: ~112 decisions (2 game-minutes at 32 frames/decision).
Measured throughput: ~300 decisions/s for 4 env runners together (complete_episodes, engine speed 1000).
No separate evaluation: its env runner keeps one more engine (~4.5 GB), which left the 31 GB machine at 88 % RAM;
checkpoints are scored on the training victories instead.

Usage (from this repository root, with Maxalaar_Reinforcement_Learning and this package installed:
pip install -e ".[reinforcement_learning]"):
    python ../Maxalaar_Reinforcement_Learning/scripts/reinforcement_learning_training.py \
        --experimentation_configuration_file configurations/reinforcement_learning/one_versus_one_ppo.py
"""

import torch.nn as nn
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

from beyond_all_reason_environment import (
    ActionConfiguration,
    Agent,
    GameConfiguration,
    ObservationConfiguration,
    Passive,
    ReinforcementLearningEnvironmentConfiguration,
    RewardWeights,
)
from beyond_all_reason_environment.scenarios import one_versus_one
from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import BeyondAllReasonModule
from beyond_all_reason_reinforcement_learning.victory_callback import VictoryCallback

one_versus_one_ppo = ExperimentationConfiguration(
    experimentation_name="one_versus_one_ppo",
    environment_name="beyond_all_reason_reinforcement_learning.ray_multi_agent_environment.RayMultiAgentEnvironment",
    environment_configuration={
        "configuration": ReinforcementLearningEnvironmentConfiguration(
            game=GameConfiguration(scenario=one_versus_one, controllers=[Agent(), Passive()]),
            observation=ObservationConfiguration(maximum_weapons_per_unit=1),
            action=ActionConfiguration(),
            rewards=RewardWeights(damage_dealt=0.0, damage_received=0.0, kill=0.0, victory=1.0, defeat=-1.0),
        )
    },
)

# ── Architecture ──────────────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.architecture = BeyondAllReasonModule
one_versus_one_ppo.reinforcement_learning_configuration.architecture_configuration = {
    # Actor
    "d_level1": 64,
    "d_level2": 64,
    "d_level3": 64,
    "entity_projector_hidden_sizes": [64],
    "weapon_projector_hidden_sizes": [32],
    "number_entity_attention_heads": 1,
    "number_category_attention_heads": 1,
    "number_global_attention_heads": 1,
    "number_global_attention_layers": 1,
    "global_feedforward_size": 128,
    "context_projector_hidden_sizes": [],
    "terrain_cnn_channels": [16, 32, 32],
    "action_hidden_sizes": [64, 32],
    "activation_function_class": nn.ReLU,
    # Centralized critic
    "critic_d": 64,
    "critic_entity_projector_hidden_sizes": [64],
    "critic_weapon_projector_hidden_sizes": [32],
    "critic_number_entity_attention_heads": 1,
    "critic_number_category_attention_heads": 1,
    "critic_number_global_attention_heads": 1,
    "critic_number_global_attention_layers": 1,
    "critic_global_feedforward_size": 128,
    "critic_context_projector_hidden_sizes": [],
    "critic_terrain_cnn_channels": [16, 32, 32],
}

# ── Multi-agent: 1 agent per episode, shared policy ───────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.is_multi_agents = True
one_versus_one_ppo.reinforcement_learning_configuration.policies = {"shared_policy"}
one_versus_one_ppo.reinforcement_learning_configuration.policy_mapping_function = lambda agent_id, episode, **kwargs: (
    "shared_policy"
)

# ── Environment ───────────────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.disable_environment_checking = True
one_versus_one_ppo.reinforcement_learning_configuration.normalize_actions = False
one_versus_one_ppo.reinforcement_learning_configuration.flatten_observations = False
one_versus_one_ppo.reinforcement_learning_configuration.compress_observations = False

# ── PPO hyperparameters ───────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.training_name = "base"
one_versus_one_ppo.reinforcement_learning_configuration.gamma = 0.995
one_versus_one_ppo.reinforcement_learning_configuration.lambda_gae = 0.95
one_versus_one_ppo.reinforcement_learning_configuration.learning_rate = 3e-4
one_versus_one_ppo.reinforcement_learning_configuration.gradient_clip = 0.5
one_versus_one_ppo.reinforcement_learning_configuration.clip_policy_parameter = 0.2
one_versus_one_ppo.reinforcement_learning_configuration.entropy_coefficient = 0.005
one_versus_one_ppo.reinforcement_learning_configuration.value_function_loss_coefficient = 1.0
one_versus_one_ppo.reinforcement_learning_configuration.number_epochs = 10
one_versus_one_ppo.reinforcement_learning_configuration.train_batch_size = 4000
one_versus_one_ppo.reinforcement_learning_configuration.minibatch_size = 512
one_versus_one_ppo.reinforcement_learning_configuration.batch_mode = "complete_episodes"
one_versus_one_ppo.reinforcement_learning_configuration.sample_timeout_s = 180

# ── Resources ─────────────────────────────────────────────────────────────────
# Each engine process uses ~2.3 CPU cores and ~4.7 GB RAM.
# 3 runners: ~14 GB RAM, ~7 cores for engines. 4 runners (~19 GB) froze the 31 GB machine (no swap) on 2026-10-18.
one_versus_one_ppo.reinforcement_learning_configuration.number_environment_runners = 3
one_versus_one_ppo.reinforcement_learning_configuration.number_cpus_per_environment_runners = 3
one_versus_one_ppo.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
one_versus_one_ppo.reinforcement_learning_configuration.number_learners = 1
one_versus_one_ppo.reinforcement_learning_configuration.number_cpus_per_learner = 1
one_versus_one_ppo.reinforcement_learning_configuration.number_gpus_per_learner = 1

# ── Evaluation ────────────────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.evaluation_interval = None

# ── Stopping criterion: 90 % victories ────────────────────────────────────────
# VictoryCallback averages the victories over the last 100 episodes of each env runner. Tune saves a
# last checkpoint when it stops: the next stage of a curriculum (pawn_curriculum.py) starts from it.
one_versus_one_ppo.reinforcement_learning_configuration.callback = VictoryCallback
one_versus_one_ppo.reinforcement_learning_configuration.stopping_criterion = {"env_runners/victory_rate": 0.9}

# ── Checkpoints ──────────────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.checkpoint_frequency = 20
one_versus_one_ppo.reinforcement_learning_configuration.number_checkpoint_to_keep = 10
one_versus_one_ppo.reinforcement_learning_configuration.checkpoint_score_attribute = "env_runners/victory_rate"
one_versus_one_ppo.reinforcement_learning_configuration.checkpoint_score_order = "max"
