"""Hierarchical attention RLModule for Beyond All Reason combat scenarios.

Architecture — three levels of attention (actor):

    Level 1  EntityEncoder (shared weights for all categories):
        chassis ∥ state → MLP → entity_query                      (d_level1)
        weapon[i]       → MLP → weapon_token   [per real slot]    (d_level1)
        cross-attention: entity_query over weapon_tokens
                                             → weapon_context      (d_level1)
        Linear(entity_query ∥ weapon_context)→ entity_token        (d_level1)

    Level 2  CategoryAggregation (one per category — self skips this level):
        learned_query × cross-attention over entity_tokens         (d_level2)
        → category_token  (zero when the category has no real entities)

    Level 3  GlobalAttention:
        7 tokens projected to d_level3:
            0  t_self          ← entity_token of the observing unit (Level 1 direct)
            1  t_ally_units    ← CategoryAggregation output
            2  t_enemy_units   ← CategoryAggregation output
            3  t_ally_buildings← CategoryAggregation output
            4  t_enemy_buildings← CategoryAggregation output
            5  t_context       ← MLP(context features)
            6  t_terrain       ← CNN + Linear(terrain)
        self-attention over all 7 tokens → attended token at position 0 (t_self)
        → embedding (d_level3) → action heads

    Action heads:
        command  Linear(d_level3, n_commands) → logits, masked by command_mask → Categorical
        target   Linear(d_level3, 4)          → (μ_x, μ_z, log_σ_x, log_σ_z) → DiagGaussian

Centralized critic (CentralizedCritic, fully separate weights):
    CTDE — uses the full game state (all allyteams visible via Spring.SetGlobalLos).
    For each allyteam t = 0 (agents) to N-1 (opponents):
        EntityEncoder (own weights, shared across categories)
        CategoryAggregation[t] for units   → units_token[t]
        CategoryAggregation[t] for buildings → buildings_token[t]
    Level 3: 3*N + 2 tokens for N allyteams —
        [at0_cls, …, at{N-1}_cls, at0_units, at0_bld, …, at{N-1}_units, at{N-1}_bld, context, terrain]
    Self-attention → attend CLS positions → Linear → one scalar per allyteam.
    compute_values returns the allyteam 0 (agents) scalar for all agents.
    n_allyteams is inferred from the observation space at setup time.

Actor model configuration keys (all optional — defaults match a small 1v1 network):
    d_level1                       int   entity token size         (default 64)
    d_level2                       int   category token size       (default = d_level1)
    d_level3                       int   global token size         (default = d_level1)
    entity_projector_hidden_sizes  list  hidden layers of chassis+state MLP   ([64])
    weapon_projector_hidden_sizes  list  hidden layers of weapon MLP           ([32])
    number_entity_attention_heads  int   attention heads at Level 1            (1)
    number_category_attention_heads int  attention heads at Level 2            (1)
    number_global_attention_heads  int   attention heads at Level 3            (1)
    number_global_attention_layers int   transformer layers at Level 3         (1)
    global_feedforward_size        int   feedforward dim inside Level 3 layer  (128)
    context_projector_hidden_sizes list  hidden layers of context MLP          ([])
    terrain_cnn_channels           list  out-channels per conv layer           ([16, 32, 32])
    action_hidden_sizes            list  hidden layers of action MLP head      ([64, 32])
    activation_function_class      type  PyTorch activation class              (nn.ReLU)

Centralized critic configuration keys (all prefixed with critic_):
    critic_d                              int   critic embedding size           (default = d_level1)
    critic_entity_projector_hidden_sizes  list                                  ([64])
    critic_weapon_projector_hidden_sizes  list                                  ([32])
    critic_number_entity_attention_heads  int                                   (1)
    critic_number_category_attention_heads int                                  (1)
    critic_number_global_attention_heads  int                                   (1)
    critic_number_global_attention_layers int                                   (1)
    critic_global_feedforward_size        int                                   (128)
    critic_context_projector_hidden_sizes list                                  ([])
    critic_terrain_cnn_channels           list                                  ([16, 32, 32])
"""

from typing import Any

import torch
import torch.nn as nn
import tree
from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture
from maxalaar_reinforcement_learning.training.is_env_runner_process import is_env_runner_process
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchDeterministic, TorchMultiDistribution
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchDistribution
from ray.rllib.utils.annotations import override

from beyond_all_reason_reinforcement_learning.architecture.category_aggregation import CategoryAggregation
from beyond_all_reason_reinforcement_learning.architecture.centralized_critic import CentralizedCritic
from beyond_all_reason_reinforcement_learning.architecture.entity_encoder import EntityEncoder
from beyond_all_reason_reinforcement_learning.architecture.terrain_encoder import TerrainEncoder

# Observation space key tuples for the four group categories.
# (chassis_key, state_key, weapons_key, weapon_mask_key, entity_mask_key)
_CATEGORY_KEYS: dict[str, tuple[str, str, str, str, str]] = {
    "ally_units": (
        "ally_units_chassis",
        "ally_units_state",
        "ally_units_weapons",
        "ally_units_weapon_mask",
        "ally_unit_mask",
    ),
    "enemy_units": (
        "enemy_units_chassis",
        "enemy_units_state",
        "enemy_units_weapons",
        "enemy_units_weapon_mask",
        "enemy_unit_mask",
    ),
    "ally_buildings": (
        "ally_buildings_chassis",
        "ally_buildings_state",
        "ally_buildings_weapons",
        "ally_buildings_weapon_mask",
        "ally_building_mask",
    ),
    "enemy_buildings": (
        "enemy_buildings_chassis",
        "enemy_buildings_state",
        "enemy_buildings_weapons",
        "enemy_buildings_weapon_mask",
        "enemy_building_mask",
    ),
}

# Insertion order fixes the stacking order of category tokens at Level 3 (positions 1–4).
_CATEGORY_ORDER = ("ally_units", "enemy_units", "ally_buildings", "enemy_buildings")


class _PerSampleDeterministic(TorchDeterministic):
    """A deterministic action has log-probability 0, one value per sample. RLlib's TorchDeterministic keeps the
    action dimensions ((B, 2) for the target), which TorchMultiDistribution.logp cannot add to the command's (B,)."""

    def logp(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros(self.loc.shape[:1], device=self.loc.device)


def _with_per_sample_deterministic_logp(
    distribution_class: type[TorchMultiDistribution],
) -> type[TorchMultiDistribution]:
    """The multi-distribution class with a to_deterministic usable without exploration (evaluation, display)."""

    class Distribution(distribution_class):
        def to_deterministic(self) -> TorchMultiDistribution:
            children = [
                _PerSampleDeterministic(loc=child.loc) for child in super().to_deterministic()._flat_child_distributions
            ]
            return TorchMultiDistribution(tree.unflatten_as(self._original_struct, children))

    return Distribution


class BeyondAllReasonModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> type[TorchDistribution]:
        return _with_per_sample_deterministic_logp(
            Catalog._get_dist_cls_from_action_space(
                action_space=self.action_space,
                framework="torch",
            )
        )

    @override(TorchRLModule)
    def setup(self) -> None:
        config = self.model_config
        activation_class = config.get("activation_function_class", nn.ReLU)

        # ── Token sizes ───────────────────────────────────────────────────────────
        d_level1: int = config.get("d_level1", 64)
        d_level2: int = config.get("d_level2", d_level1)
        d_level3: int = config.get("d_level3", d_level1)
        self._d_level1 = d_level1
        self._d_level2 = d_level2
        self._d_level3 = d_level3

        # ── Dimensions from the observation space ─────────────────────────────────
        spaces = self.observation_space.spaces
        n_chassis: int = spaces["self_chassis"].shape[0]
        n_state: int = spaces["self_state"].shape[0]
        n_weapon: int = spaces["self_weapons"].shape[1]  # (W, n_weapon) → shape[1]
        n_context: int = spaces["context"].shape[0]
        terrain_height: int = spaces["terrain"].shape[0]
        terrain_width: int = spaces["terrain"].shape[1]

        # ── Action space ──────────────────────────────────────────────────────────
        self._n_commands: int = self.action_space["command"].n
        action_dist_cls = self.get_inference_action_dist_cls()
        action_output_dim: int = action_dist_cls.required_input_dim(self.action_space)

        # ── Shared kwargs for sub-module constructors ──────────────────────────────
        n_global_heads: int = config.get("number_global_attention_heads", 1)
        assert d_level3 % n_global_heads == 0, (
            f"d_level3 ({d_level3}) must be divisible by number_global_attention_heads ({n_global_heads})"
        )

        entity_encoder_kwargs = dict(
            n_chassis=n_chassis,
            n_state=n_state,
            n_weapon=n_weapon,
            d_level1=d_level1,
            entity_projector_hidden_sizes=config.get("entity_projector_hidden_sizes", [64]),
            weapon_projector_hidden_sizes=config.get("weapon_projector_hidden_sizes", [32]),
            number_entity_attention_heads=config.get("number_entity_attention_heads", 1),
            activation_function_class=activation_class,
        )
        category_aggregation_kwargs = dict(
            d_level1=d_level1,
            d_level2=d_level2,
            number_category_attention_heads=config.get("number_category_attention_heads", 1),
        )
        terrain_encoder_kwargs = dict(
            terrain_height=terrain_height,
            terrain_width=terrain_width,
            d_level3=d_level3,
            cnn_channels=config.get("terrain_cnn_channels", [16, 32, 32]),
            activation_function_class=activation_class,
        )
        global_layer_kwargs = dict(
            d_model=d_level3,
            nhead=n_global_heads,
            dim_feedforward=config.get("global_feedforward_size", 128),
            dropout=0.0,
            batch_first=True,
        )

        # ── Actor stack ───────────────────────────────────────────────────────────
        # Level 1
        self.actor_entity_encoder = EntityEncoder(**entity_encoder_kwargs)

        # Level 2
        for category in _CATEGORY_ORDER:
            setattr(
                self,
                f"actor_category_{category}",
                CategoryAggregation(**category_aggregation_kwargs),
            )

        # Level 3 — projections into d_level3
        self.actor_self_projection = nn.Linear(d_level1, d_level3)
        self.actor_category_projection = nn.Linear(d_level2, d_level3)
        self.actor_context_projector = create_dense_architecture(
            input_dimension=n_context,
            configuration_hidden_layers=config.get("context_projector_hidden_sizes", []),
            output_dimension=d_level3,
            activation_function_class=activation_class,
        )
        self.actor_terrain_encoder = TerrainEncoder(**terrain_encoder_kwargs)
        self.actor_global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**global_layer_kwargs),
            num_layers=config.get("number_global_attention_layers", 1),
        )

        # ── Actor action head ─────────────────────────────────────────────────────
        self.action_head = create_dense_architecture(
            input_dimension=d_level3,
            configuration_hidden_layers=config.get("action_hidden_sizes", [64, 32]),
            output_dimension=action_output_dim,
            activation_function_class=activation_class,
        )

        # ── Centralized critic (fully separate weights) ────────────────────────────
        # Infer n_allyteams by counting allyteamT_units_chassis keys in the observation space.
        n_allyteams: int = sum(1 for k in spaces if k.startswith("allyteam") and k.endswith("_units_chassis"))
        d_critic: int = config.get("critic_d", d_level1)
        self.centralized_critic = CentralizedCritic(
            n_allyteams=n_allyteams,
            n_chassis=n_chassis,
            n_state=n_state,
            n_weapon=n_weapon,
            n_context=n_context,
            terrain_height=terrain_height,
            terrain_width=terrain_width,
            d_critic=d_critic,
            entity_projector_hidden_sizes=config.get("critic_entity_projector_hidden_sizes", [64]),
            weapon_projector_hidden_sizes=config.get("critic_weapon_projector_hidden_sizes", [32]),
            number_entity_attention_heads=config.get("critic_number_entity_attention_heads", 1),
            number_category_attention_heads=config.get("critic_number_category_attention_heads", 1),
            number_global_attention_heads=config.get("critic_number_global_attention_heads", 1),
            number_global_attention_layers=config.get("critic_number_global_attention_layers", 1),
            global_feedforward_size=config.get("critic_global_feedforward_size", 128),
            context_projector_hidden_sizes=config.get("critic_context_projector_hidden_sizes", []),
            terrain_cnn_channels=config.get("critic_terrain_cnn_channels", [16, 32, 32]),
            activation_function_class=activation_class,
        )

        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if not is_env_runner_process():
            print(self)

    # ── Encoding ──────────────────────────────────────────────────────────────────

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run actor three-level encoding and return the attended self-token (B, d_level3)."""
        device = next(self.parameters()).device

        def to(key: str) -> torch.Tensor:
            return obs[key].to(device).float()

        B: int = obs["self_chassis"].shape[0]

        # ── Level 1: self entity (direct path to Level 3, skips Level 2) ─────────
        self_token = self.actor_entity_encoder(
            to("self_chassis"),  # (B, n_chassis)
            to("self_state"),  # (B, n_state)
            to("self_weapons"),  # (B, W, n_weapon)
            to("self_weapon_mask"),  # (B, W)
        )  # (B, d_level1)

        # ── Level 1 → Level 2: group categories ──────────────────────────────────
        category_tokens: dict[str, torch.Tensor] = {}
        for category, (chassis_key, state_key, weapons_key, weapon_mask_key, entity_mask_key) in _CATEGORY_KEYS.items():
            chassis = to(chassis_key)  # (B, N, n_chassis)
            state = to(state_key)  # (B, N, n_state)
            weapons = to(weapons_key)  # (B, N, W, n_weapon)
            weapon_mask = to(weapon_mask_key)  # (B, N, W)
            entity_mask = to(entity_mask_key)  # (B, N)

            N: int = chassis.shape[1]

            # Reshape to a flat entity batch so EntityEncoder processes all rows uniformly.
            # Explicit last-dimension sizes are required: reshape(0, -1) is ambiguous when N = 0.
            entity_tokens = self.actor_entity_encoder(
                chassis.reshape(B * N, chassis.shape[-1]),
                state.reshape(B * N, state.shape[-1]),
                weapons.reshape(B * N, weapons.shape[-2], weapons.shape[-1]),
                weapon_mask.reshape(B * N, weapon_mask.shape[-1]),
            ).reshape(B, N, self._d_level1)  # (B, N, d_level1)

            aggregator: CategoryAggregation = getattr(self, f"actor_category_{category}")
            category_tokens[category] = aggregator(entity_tokens, entity_mask)  # (B, d_level2)

        # ── Level 3: project to d_level3, self-attend, read self token ───────────
        # Stack all 7 tokens; t_self is placed at position 0 so it is read after attention.
        tokens = torch.stack(
            [self.actor_self_projection(self_token)]
            + [self.actor_category_projection(category_tokens[cat]) for cat in _CATEGORY_ORDER]
            + [self.actor_context_projector(to("context")), self.actor_terrain_encoder(to("terrain"))],
            dim=1,
        )  # (B, 7, d_level3)

        attended = self.actor_global_transformer(tokens)  # (B, 7, d_level3)
        return attended[:, 0]  # (B, d_level3) — attended self token

    # ── RLlib interface ───────────────────────────────────────────────────────────

    @override(TorchRLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        embedding = self._encode(batch[Columns.OBS])  # (B, d_level3)
        raw_inputs = self.action_head(embedding)  # (B, action_output_dim)

        # The Dict action space {"command": Discrete(n), "target": Box(2,)} is serialised
        # alphabetically by RLlib → command logits first, then target parameters.
        command_logits = raw_inputs[:, : self._n_commands]  # (B, n_commands)
        target_params = raw_inputs[:, self._n_commands :]  # (B, 4)

        # Set forbidden commands to −∞ so the Categorical assigns them zero probability.
        command_mask = batch[Columns.OBS]["command_mask"].to(embedding.device).float()
        masked_logits = command_logits.masked_fill(command_mask == 0.0, float("-inf"))

        return {Columns.ACTION_DIST_INPUTS: torch.cat([masked_logits, target_params], dim=-1)}

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: Any | None = None,
    ) -> torch.Tensor:
        # Centralized critic: (B, n_allyteams) → take allyteam 0 value for all agents.
        return self.centralized_critic(batch[Columns.OBS])[:, 0]  # (B,)
