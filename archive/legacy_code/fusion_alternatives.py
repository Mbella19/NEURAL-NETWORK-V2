"""
ARCHIVED: Alternative fusion implementations.

These classes were never used in production (AttentionFusion is the active implementation).
Archived for potential future reference.

Original location: src/models/fusion.py
Archive date: 2024-12-11
"""

import torch
import torch.nn as nn


class SimpleFusion(nn.Module):
    """
    Simpler fusion alternative using learned weighted sum.

    Use this if AttentionFusion causes memory issues.

    NOTE: This class was never instantiated in production code.
    """

    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()

        # Learnable weights for each timeframe
        self.weights = nn.Parameter(torch.ones(3) / 3)

        # Projection and normalization
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(
        self,
        enc_15m: torch.Tensor,
        enc_1h: torch.Tensor,
        enc_4h: torch.Tensor
    ) -> torch.Tensor:
        """Weighted sum fusion."""
        weights = torch.softmax(self.weights, dim=0)

        fused = (
            weights[0] * enc_15m +
            weights[1] * enc_1h +
            weights[2] * enc_4h
        )

        return self.projection(fused)


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion.

    Concatenates all timeframe representations and projects to context dimension.

    NOTE: This class was never instantiated in production code.
    """

    def __init__(
        self,
        d_model: int = 64,
        context_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # 3 timeframes concatenated
        self.projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, context_dim),
            nn.LayerNorm(context_dim)
        )

    def forward(
        self,
        enc_15m: torch.Tensor,
        enc_1h: torch.Tensor,
        enc_4h: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate and project."""
        concat = torch.cat([enc_15m, enc_1h, enc_4h], dim=1)
        return self.projection(concat)
