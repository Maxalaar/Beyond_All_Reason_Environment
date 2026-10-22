"""PPO training configuration — 5v5 Pawn combat against Passive Pawns (curriculum step 2).

Scenario (beyond_all_reason_environment.scenarios.five_versus_five_arena): the 1v1 arena (flat 1024×1024 elmos,
2 game-minutes) with 5 armpw (Pawn) on each side, each team in a small band at the middle of its edge.
Chosen here, for this experiment: the 5 agents (one shared policy) against Passive Pawns (do not move, do not shoot),
one weapon slot per unit, every command allowed, and a terminal reward (+1 victory, -1 defeat, no intermediate
shaping). Entity counts follow the scenario (4 allies, 5 enemies).
Stops at 90 % victories. In pawn_curriculum.py it starts from the 1v1 policy (one_versus_one_ppo.py), and
five_versus_five_hold_and_fire_ppo.py (opponents that shoot back) starts from it.

Expected episode length: at most ~113 decisions (2 game-minutes at 32 frames/decision).
No separate evaluation: its env runner keeps one more engine (~4.5 GB), which left the 31 GB machine at 88 % RAM in
the 1v1 run; checkpoints are scored on the training victories instead.

Usage (from this repository root, with Maxalaar_Reinforcement_Learning and this package installed:
pip install -e ".[reinforcement_learning]"):
    python ../Maxalaar_Reinforcement_Learning/scripts/reinforcement_learning_training.py \
        --experimentation_configuration_file configurations/reinforcement_learning/five_versus_five_passive_ppo.py
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
from beyond_all_reason_environment.scenarios import five_versus_five_arena
from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import BeyondAllReasonModule
from beyond_all_reason_reinforcement_learning.victory_callback import VictoryCallback

five_versus_five_passive_ppo = ExperimentationConfiguration(
    experimentation_name="five_versus_five_passive_ppo",
    environment_name="beyond_all_reason_reinforcement_learning.ray_multi_agent_environment.RayMultiAgentEnvironment",
    environment_configuration={
        "configuration": ReinforcementLearningEnvironmentConfiguration(
            game=GameConfiguration(scenario=five_versus_five_arena, controllers=[Agent(), Passive()]),
            observation=ObservationConfiguration(maximum_weapons_per_unit=1),
            action=ActionConfiguration(),
            rewards=RewardWeights(damage_dealt=0.0, damage_received=0.0, kill=0.0, victory=1.0, defeat=-1.0),
        )
    },
)

# ── Architecture ──────────────────────────────────────────────────────────────
five_versus_five_passive_ppo.reinforcement_learning_configuration.architecture = BeyondAllReasonModule
five_versus_five_passive_ppo.reinforcement_learning_configuration.architecture_configuration = {
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

# ── Multi-agent: 5 agents per episode, shared policy ───────────────────────────
five_versus_five_passive_ppo.reinforcement_learning_configuration.is_multi_agents = True
five_versus_five_passive_ppo.reinforcement_learning_configuration.policies = {"shared_policy"}
five_versus_five_passive_ppo.reinforcement_learning_configuration.policy_mapping_function = (
    lambda agent_id, episode, **kwargs: "shared_policy"
)

# ── Environment ───────────────────────────────────────────────────────────────
five_versus_five_passive_ppo.reinforcement_learning_configuration.disable_environment_checking = True
five_versus_five_passive_ppo.reinforcement_learning_configuration.normalize_actions = False
five_versus_five_passive_ppo.reinforcement_learning_configuration.flatten_observations = False
five_versus_five_passive_ppo.reinforcement_learning_configuration.compress_observations = False

# ── PPO hyperparameters ───────────────────────────────────────────────────────
five_versus_five_passive_ppo.reinforcement_learning_configuration.training_name = "base"
five_versus_five_passive_ppo.reinforcement_learning_configuration.gamma = 0.995
five_versus_five_passive_ppo.reinforcement_learning_configuration.lambda_gae = 0.95
five_versus_five_passive_ppo.reinforcement_learning_configuration.learning_rate = 3e-4
five_versus_five_passive_ppo.reinforcement_learning_configuration.gradient_clip = 0.5
five_versus_five_passive_ppo.reinforcement_learning_configuration.clip_policy_parameter = 0.2
five_versus_five_passive_ppo.reinforcement_learning_configuration.entropy_coefficient = 0.005
five_versus_five_passive_ppo.reinforcement_learning_configuration.value_function_loss_coefficient = 1.0
five_versus_five_passive_ppo.reinforcement_learning_configuration.number_epochs = 10
# 2000 env steps = ~10000 agent steps (5 agents): 4000 kept the 31 GB machine at 89-90 % RAM (2026-10-18).
five_versus_five_passive_ppo.reinforcement_learning_configuration.train_batch_size = 2000
five_versus_five_passive_ppo.reinforcement_learning_configuration.minibatch_size = 512
five_versus_five_passive_ppo.reinforcement_learning_configuration.batch_mode = "complete_episodes"
five_versus_five_passive_ppo.reinforcement_learning_configuration.sample_timeout_s = 180

# ── Resources ─────────────────────────────────────────────────────────────────
# Each engine process uses ~2.3 CPU cores and ~4.7 GB RAM.
# 3 runners: ~14 GB RAM, ~7 cores for engines. 4 runners (~19 GB) froze the 31 GB machine (no swap) on 2026-10-18.
five_versus_five_passive_ppo.reinforcement_learning_configuration.number_environment_runners = 3
five_versus_five_passive_ppo.reinforcement_learning_configuration.number_cpus_per_environment_runners = 3
five_versus_five_passive_ppo.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
five_versus_five_passive_ppo.reinforcement_learning_configuration.number_learners = 1
five_versus_five_passive_ppo.reinforcement_learning_configuration.number_cpus_per_learner = 1
five_versus_five_passive_ppo.reinforcement_learning_configuration.number_gpus_per_learner = 1

# ── Evaluation ────────────────────────────────────────────────────────────────
five_versus_five_passive_ppo.reinforcement_learning_configuration.evaluation_interval = None

# ── Stopping criterion: 90 % victories ────────────────────────────────────────
# VictoryCallback averages the victories over the last 100 episodes of each env runner. Tune saves a
# last checkpoint when it stops: the next stage of a curriculum (pawn_curriculum.py) starts from it.
five_versus_five_passive_ppo.reinforcement_learning_configuration.callback = VictoryCallback
five_versus_five_passive_ppo.reinforcement_learning_configuration.stopping_criterion = {"env_runners/victory_rate": 0.9}

# ── Checkpoints ──────────────────────────────────────────────────────────────
five_versus_five_passive_ppo.reinforcement_learning_configuration.checkpoint_frequency = 20
five_versus_five_passive_ppo.reinforcement_learning_configuration.number_checkpoint_to_keep = 10
five_versus_five_passive_ppo.reinforcement_learning_configuration.checkpoint_score_attribute = (
    "env_runners/victory_rate"
)
five_versus_five_passive_ppo.reinforcement_learning_configuration.checkpoint_score_order = "max"
