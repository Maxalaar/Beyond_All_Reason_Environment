"""Centralized critic for N-allyteam CTDE (Centralized Training, Decentralized Execution).

Architecture — one value per allyteam, using full-game state:

    For each allyteam t (0 = agents, 1..N-1 = opponents):
        EntityEncoder (shared weights across all categories)
          over allyteam t's unit entity tokens     → units_aggregator[t]  → units_token[t]
          over allyteam t's building entity tokens → buildings_aggregator[t] → buildings_token[t]

    Level 3 — 3 * n_allyteams + 2 tokens:
        positions 0 .. n_allyteams-1          : learned CLS tokens (one per allyteam)
        positions n_allyteams .. 3*n_allyteams-1 : units_token[t], buildings_token[t] for t = 0..N-1
        position  3*n_allyteams               : context_token
        position  3*n_allyteams + 1           : terrain_token

        self-attention over all tokens
        → attended CLS positions 0..n_allyteams-1
        → one Linear(d_critic, 1) per allyteam → value scalar

    For n_allyteams = 2: 8 tokens total.

Observation key layout (produced by encode_observation in reinforcement_learning_environment.py):
    allyteam0_units_chassis   (1 + max_ally_units, n_chassis)  — self prepended at row 0
    allyteam0_unit_mask       (1 + max_ally_units,)
    allyteam0_units_state     (1 + max_ally_units, n_state)
    allyteam0_units_weapons   (1 + max_ally_units, W, n_weapon)
    allyteam0_units_weapon_mask (1 + max_ally_units, W)
    allyteam0_buildings_*     (max_ally_buildings, …)
    allyteamT_units_chassis   (max_units_per_enemy_allyteam, n_chassis)  for t = 1..N-1
    allyteamT_unit_mask       (max_units_per_enemy_allyteam,)
    … (same pattern)

This design requires no ConnectorV2: the agent observation already contains full-game
state (Spring.SetGlobalLos gives all allyteams full vision).

Model configuration keys (all optional):
    critic_d                              int   critic embedding size             (default = d_level1)
    critic_entity_projector_hidden_sizes  list  hidden layers of entity MLP       ([64])
    critic_weapon_projector_hidden_sizes  list  hidden layers of weapon MLP       ([32])
    critic_number_entity_attention_heads  int   heads at entity level             (1)
    critic_number_category_attention_heads int  heads at category level           (1)
    critic_number_global_attention_heads  int   heads at global level             (1)
    critic_number_global_attention_layers int   transformer layers at global level (1)
    critic_global_feedforward_size        int   feedforward dim inside layer      (128)
    critic_context_projector_hidden_sizes list  hidden layers of context MLP      ([])
    critic_terrain_cnn_channels           list  out-channels per CNN conv layer   ([16, 32, 32])
    activation_function_class             type  PyTorch activation class          (nn.ReLU)
"""

import torch
import torch.nn as nn
from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture

from beyond_all_reason_reinforcement_learning.architecture.category_aggregation import CategoryAggregation
from beyond_all_reason_reinforcement_learning.architecture.entity_encoder import EntityEncoder
from beyond_all_reason_reinforcement_learning.architecture.terrain_encoder import TerrainEncoder


class CentralizedCritic(nn.Module):
    """Centralized critic: takes a single agent's observation and returns one value per allyteam.

    Since BAR combat uses full vision (Spring.SetGlobalLos), any agent's observation
    encodes the complete game state — no cross-agent aggregation is needed.
    """

    def __init__(
        self,
        n_allyteams: int,
        n_chassis: int,
        n_state: int,
        n_weapon: int,
        n_context: int,
        terrain_height: int,
        terrain_width: int,
        d_critic: int = 64,
        entity_projector_hidden_sizes: list[int] = (64,),
        weapon_projector_hidden_sizes: list[int] = (32,),
        number_entity_attention_heads: int = 1,
        number_category_attention_heads: int = 1,
        number_global_attention_heads: int = 1,
        number_global_attention_layers: int = 1,
        global_feedforward_size: int = 128,
        context_projector_hidden_sizes: list[int] = (),
        terrain_cnn_channels: list[int] = (16, 32, 32),
        activation_function_class: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        assert d_critic % number_entity_attention_heads == 0, (
            f"d_critic ({d_critic}) must be divisible by "
            f"number_entity_attention_heads ({number_entity_attention_heads})"
        )
        assert d_critic % number_category_attention_heads == 0, (
            f"d_critic ({d_critic}) must be divisible by "
            f"number_category_attention_heads ({number_category_attention_heads})"
        )
        assert d_critic % number_global_attention_heads == 0, (
            f"d_critic ({d_critic}) must be divisible by "
            f"number_global_attention_heads ({number_global_attention_heads})"
        )
        self.n_allyteams = n_allyteams
        self.d_critic = d_critic

        entity_encoder_kwargs = dict(
            n_chassis=n_chassis,
            n_state=n_state,
            n_weapon=n_weapon,
            d_level1=d_critic,
            entity_projector_hidden_sizes=list(entity_projector_hidden_sizes),
            weapon_projector_hidden_sizes=list(weapon_projector_hidden_sizes),
            number_entity_attention_heads=number_entity_attention_heads,
            activation_function_class=activation_function_class,
        )
        category_aggregation_kwargs = dict(
            d_level1=d_critic,
            d_level2=d_critic,
            number_category_attention_heads=number_category_attention_heads,
        )

        # Shared entity encoder — same weights for all allyteam/category combinations.
        self.entity_encoder = EntityEncoder(**entity_encoder_kwargs)

        # One CategoryAggregation per (allyteam, category) pair.
        self.units_aggregators = nn.ModuleList(
            [CategoryAggregation(**category_aggregation_kwargs) for _ in range(n_allyteams)]
        )
        self.buildings_aggregators = nn.ModuleList(
            [CategoryAggregation(**category_aggregation_kwargs) for _ in range(n_allyteams)]
        )

        # One learned CLS token per allyteam (prepended at Level 3 positions 0..n_allyteams-1).
        self.allyteam_queries = nn.Parameter(torch.randn(n_allyteams, d_critic))

        # Global context and terrain encoders.
        self.context_projector = create_dense_architecture(
            input_dimension=n_context,
            configuration_hidden_layers=list(context_projector_hidden_sizes),
            output_dimension=d_critic,
            activation_function_class=activation_function_class,
        )
        self.terrain_encoder = TerrainEncoder(
            terrain_height=terrain_height,
            terrain_width=terrain_width,
            d_level3=d_critic,
            cnn_channels=list(terrain_cnn_channels),
            activation_function_class=activation_function_class,
        )

        # Global self-attention.
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_critic,
                nhead=number_global_attention_heads,
                dim_feedforward=global_feedforward_size,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=number_global_attention_layers,
        )

        # One value head per allyteam — reads the attended CLS token at position t.
        self.value_heads = nn.ModuleList([nn.Linear(d_critic, 1) for _ in range(n_allyteams)])

    # ── Helpers ───────────────────────────────────────────────────────────────────

    def _encode_entities(
        self,
        chassis: torch.Tensor,  # (B, N, n_chassis)
        state: torch.Tensor,  # (B, N, n_state)
        weapons: torch.Tensor,  # (B, N, W, n_weapon)
        weapon_mask: torch.Tensor,  # (B, N, W)
    ) -> torch.Tensor:  # (B, N, d_critic)
        B, N, _ = chassis.shape
        if N == 0:
            return torch.zeros(B, 0, self.d_critic, device=chassis.device, dtype=chassis.dtype)
        return self.entity_encoder(
            chassis.reshape(B * N, chassis.shape[-1]),
            state.reshape(B * N, state.shape[-1]),
            weapons.reshape(B * N, weapons.shape[-2], weapons.shape[-1]),
            weapon_mask.reshape(B * N, weapon_mask.shape[-1]),
        ).reshape(B, N, self.d_critic)

    # ── Forward ───────────────────────────────────────────────────────────────────

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:  # (B, n_allyteams)
        device = self.allyteam_queries.device
        B: int = obs["context"].shape[0]

        def to(key: str) -> torch.Tensor:
            return obs[key].to(device).float()

        # ── Level 1 + 2: entity tokens → category tokens per allyteam ────────────
        units_tokens = []
        buildings_tokens = []
        for t in range(self.n_allyteams):
            prefix = f"allyteam{t}"
            units_tokens.append(
                self.units_aggregators[t](
                    self._encode_entities(
                        to(f"{prefix}_units_chassis"),
                        to(f"{prefix}_units_state"),
                        to(f"{prefix}_units_weapons"),
                        to(f"{prefix}_units_weapon_mask"),
                    ),
                    to(f"{prefix}_unit_mask"),
                )
            )
            buildings_tokens.append(
                self.buildings_aggregators[t](
                    self._encode_entities(
                        to(f"{prefix}_buildings_chassis"),
                        to(f"{prefix}_buildings_state"),
                        to(f"{prefix}_buildings_weapons"),
                        to(f"{prefix}_buildings_weapon_mask"),
                    ),
                    to(f"{prefix}_building_mask"),
                )
            )

        # ── Level 3: stack all tokens, self-attend, read CLS positions ────────────
        # Layout: [at0_cls, …, at{N-1}_cls, at0_units, at0_bld, at1_units, at1_bld, …, context, terrain]
        cls_tokens = self.allyteam_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_allyteams, d_critic)

        category_tokens = torch.stack(
            [tok for t in range(self.n_allyteams) for tok in [units_tokens[t], buildings_tokens[t]]],
            dim=1,
        )  # (B, 2*n_allyteams, d_critic)

        context_token = self.context_projector(to("context")).unsqueeze(1)  # (B, 1, d_critic)
        terrain_token = self.terrain_encoder(to("terrain")).unsqueeze(1)  # (B, 1, d_critic)

        tokens = torch.cat([cls_tokens, category_tokens, context_token, terrain_token], dim=1)
        # (B, 3*n_allyteams + 2, d_critic)

        attended = self.global_transformer(tokens)  # (B, 3*n_allyteams + 2, d_critic)

        # Read the attended CLS token of each allyteam → scalar value.
        values = torch.cat(
            [self.value_heads[t](attended[:, t]) for t in range(self.n_allyteams)],
            dim=-1,
        )  # (B, n_allyteams)

        return values
