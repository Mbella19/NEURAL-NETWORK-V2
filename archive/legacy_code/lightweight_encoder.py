"""
ARCHIVED: Lightweight encoder for memory-constrained scenarios.

This class was never used in production (TransformerEncoder is the active implementation).
Archived for potential future reference if OOM issues arise on Apple M2.

Original location: src/models/encoders.py
Archive date: 2024-12-11
"""

import torch
import torch.nn as nn


class LightweightEncoder(nn.Module):
    """
    Even lighter encoder using Conv1D + attention for memory-constrained scenarios.

    Use this if TransformerEncoder causes OOM on M2.

    NOTE: This class was never instantiated in production code.
    The TransformerEncoder with optimized parameters (d_model=64, nhead=4, num_layers=2)
    has proven sufficient for M2 8GB RAM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.conv_layers.append(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(nn.BatchNorm1d(hidden_dim))

        # Additional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(nn.BatchNorm1d(hidden_dim))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Self-attention for global context
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, hidden_dim]
        """
        # Conv1D expects [batch, channels, seq_len]
        x = x.transpose(1, 2)

        for conv, norm in zip(self.conv_layers, self.norm_layers):
            residual = x if x.shape[1] == conv.out_channels else None
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            if residual is not None:
                x = x + residual

        # Back to [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)

        # Self-attention
        x, _ = self.attention(x, x, x)

        # Pool: use last token for temporal forecasting
        return x[:, -1, :]
