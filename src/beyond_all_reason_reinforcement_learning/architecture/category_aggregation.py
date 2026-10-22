"""Level 2 of the hierarchical attention architecture: aggregate N entity tokens into one.

One learned query vector per category instance attends over the entity tokens produced by
Level 1, compressing a variable-length (and masked) set of entities into a single fixed-size
category token.

When N = 0 (the maximum for that category is 0 in the configuration) or all entities in a
batch element are padding, the output is a zero tensor so that the Level 3 attention is not
influenced by absent categories.
"""

import torch
import torch.nn as nn


class CategoryAggregation(nn.Module):
    def __init__(
        self,
        d_level1: int,
        d_level2: int,
        number_category_attention_heads: int,
    ):
        super().__init__()
        assert d_level2 % number_category_attention_heads == 0, (
            f"d_level2 ({d_level2}) must be divisible by "
            f"number_category_attention_heads ({number_category_attention_heads})"
        )
        # project entity tokens (d_level1) → key / value space (d_level2)
        self.key_value_projector = nn.Linear(d_level1, d_level2)
        # one learned query vector per category instance
        self.learned_query = nn.Parameter(torch.randn(d_level2))
        self.attention = nn.MultiheadAttention(
            embed_dim=d_level2,
            num_heads=number_category_attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.d_level2 = d_level2

    def forward(
        self,
        entity_tokens: torch.Tensor,  # (B, N, d_level1)
        entity_mask: torch.Tensor,  # (B, N) — 1.0 = real entity, 0.0 = padding
    ) -> torch.Tensor:  # (B, d_level2)
        B, N, _ = entity_tokens.shape

        if N == 0:
            return torch.zeros(B, self.d_level2, device=entity_tokens.device, dtype=entity_tokens.dtype)

        kv = self.key_value_projector(entity_tokens)  # (B, N, d_level2)
        query = self.learned_query.expand(B, 1, -1)  # (B, 1, d_level2)

        key_padding_mask = entity_mask == 0.0  # (B, N) — True = ignore

        # Rows with no real entities produce NaN in softmax(all −∞).
        # Temporarily unmask slot 0; the result is zeroed out afterwards.
        all_padding = key_padding_mask.all(dim=-1)  # (B,)
        safe_mask = key_padding_mask.clone()
        if all_padding.any():
            safe_mask[all_padding, 0] = False

        category_token, _ = self.attention(query, kv, kv, key_padding_mask=safe_mask)
        category_token = category_token.squeeze(1)  # (B, d_level2)

        if all_padding.any():
            category_token = category_token.masked_fill(all_padding.unsqueeze(-1), 0.0)

        return category_token
