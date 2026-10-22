"""Terrain encoder: CNN + linear projection → terrain token (d_level3).

Input: terrain observation (H × W × 4) — channels are normalised height, slope, water mask,
and LoS visibility. Each convolutional layer uses kernel 3×3 with stride 2 (halving H and W),
followed by the activation function. The flattened CNN output is then projected linearly to
d_level3.
"""

import torch
import torch.nn as nn


class TerrainEncoder(nn.Module):
    def __init__(
        self,
        terrain_height: int,
        terrain_width: int,
        d_level3: int,
        cnn_channels: list[int] = (16, 32, 32),
        activation_function_class=nn.ReLU,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 4  # height, slope, water mask, LoS visibility
        h, w = terrain_height, terrain_width
        for out_channels in cnn_channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                activation_function_class(),
            ]
            in_channels = out_channels
            h = (h + 1) // 2
            w = (w + 1) // 2
        self.cnn = nn.Sequential(*layers)
        self.projector = nn.Linear(in_channels * h * w, d_level3)

    def forward(self, terrain: torch.Tensor) -> torch.Tensor:
        # terrain arrives as (B, H, W, 4) — channels-last from the environment
        x = terrain.permute(0, 3, 1, 2).float()  # (B, 4, H, W)
        return self.projector(self.cnn(x).flatten(1))  # (B, d_level3)
