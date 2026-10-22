"""Level 1 of the hierarchical attention architecture: encode one entity into a token.

One entity = chassis features (static) + state features (dynamic) + weapon slots.

Processing:
    1. MLP(chassis ∥ state)                → entity query  (d_level1)
    2. MLP(weapon[i])  ∀ real weapon slot  → weapon tokens (W × d_level1)
    3. cross-attention: entity_query attends to weapon_tokens → weapon context (d_level1)
    4. Linear(entity_query ∥ weapon_context) → entity token  (d_level1)

The same EntityEncoder is used for all entity categories (self, ally units, enemy units,
ally buildings, enemy buildings) — weights are shared because features share the same
physical meaning across categories.

When encoding a group of N entities from a batch (B × N × features), the caller reshapes
to (B·N × features) before calling forward, then reshapes the output back to (B × N × d_level1).
"""

import torch
import torch.nn as nn
from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture


class EntityEncoder(nn.Module):
    def __init__(
        self,
        n_chassis: int,
        n_state: int,
        n_weapon: int,
        d_level1: int,
        entity_projector_hidden_sizes: list[int],
        weapon_projector_hidden_sizes: list[int],
        number_entity_attention_heads: int,
        activation_function_class=nn.ReLU,
    ):
        super().__init__()
        assert d_level1 % number_entity_attention_heads == 0, (
            f"d_level1 ({d_level1}) must be divisible by "
            f"number_entity_attention_heads ({number_entity_attention_heads})"
        )
        # chassis + state → entity query
        self.entity_projector = create_dense_architecture(
            input_dimension=n_chassis + n_state,
            configuration_hidden_layers=entity_projector_hidden_sizes,
            output_dimension=d_level1,
            activation_function_class=activation_function_class,
        )
        # weapon slot features → key / value tokens
        self.weapon_projector = create_dense_architecture(
            input_dimension=n_weapon,
            configuration_hidden_layers=weapon_projector_hidden_sizes,
            output_dimension=d_level1,
            activation_function_class=activation_function_class,
        )
        # entity_query (Q) attends to weapon_tokens (K, V)
        self.weapon_attention = nn.MultiheadAttention(
            embed_dim=d_level1,
            num_heads=number_entity_attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        # combine query and weapon context into the final entity token
        self.output_projector = nn.Linear(2 * d_level1, d_level1)
        self.d_level1 = d_level1

    def forward(
        self,
        chassis: torch.Tensor,  # (B, n_chassis)
        state: torch.Tensor,  # (B, n_state)
        weapons: torch.Tensor,  # (B, W, n_weapon)
        weapon_mask: torch.Tensor,  # (B, W) — 1.0 = real slot, 0.0 = padding
    ) -> torch.Tensor:  # (B, d_level1)
        entity_query = self.entity_projector(torch.cat([chassis, state], dim=-1))  # (B, d_level1)

        W = weapons.shape[1]
        weapon_tokens = self.weapon_projector(weapons)  # (B, W, d_level1)

        # True entries are ignored by nn.MultiheadAttention
        key_padding_mask = weapon_mask == 0.0  # (B, W)

        if W == 0:
            weapon_context = torch.zeros_like(entity_query)
        else:
            # Rows where every weapon slot is padding produce NaN in softmax(all −∞).
            # Temporarily unmask the first slot for those rows; the result is zeroed out afterwards.
            all_padding = key_padding_mask.all(dim=-1)  # (B,)
            safe_mask = key_padding_mask.clone()
            if all_padding.any():
                safe_mask[all_padding, 0] = False

            weapon_context_raw, _ = self.weapon_attention(
                entity_query.unsqueeze(1),  # (B, 1, d_level1)
                weapon_tokens,  # (B, W, d_level1)
                weapon_tokens,
                key_padding_mask=safe_mask,
            )
            weapon_context = weapon_context_raw.squeeze(1)  # (B, d_level1)

            if all_padding.any():
                weapon_context = weapon_context.masked_fill(all_padding.unsqueeze(-1), 0.0)

        return self.output_projector(torch.cat([entity_query, weapon_context], dim=-1))  # (B, d_level1)
