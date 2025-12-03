"""
TCN (Temporal Convolutional Network) Market Analyst Model.

A simpler, more stable alternative to the Transformer-based Analyst.
TCNs are better suited for time series classification due to:
- Built-in temporal causality (dilated convolutions)
- Stable gradient flow (no attention collapse)
- Parameter efficient (shared conv kernels)
- Proven performance on financial time series

This module provides a drop-in replacement for MarketAnalyst with
the same interface (forward, get_context, get_probabilities, freeze/unfreeze).
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Dict, Tuple, Optional
import gc


class TCNResidualBlock(nn.Module):
    """
    Residual block with dilated causal convolutions.

    Architecture per block:
        Conv1D (dilated) → WeightNorm → GELU → Dropout
        Conv1D (dilated) → WeightNorm → GELU → Dropout
        + Residual connection (with 1x1 conv if channels differ)

    Dilated convolutions allow exponentially growing receptive field:
    - dilation=1: sees 3 timesteps
    - dilation=2: sees 5 timesteps
    - dilation=4: sees 9 timesteps

    WeightNorm is used instead of BatchNorm for stability with small batches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        # Causal padding: pad only the left side to maintain causality
        # For kernel_size=3 and dilation=d, we need (k-1)*d padding on left
        padding = (kernel_size - 1) * dilation

        # First conv layer with weight normalization
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp1 = padding  # Amount to trim from right side

        # Second conv layer with weight normalization
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp2 = padding

        # Activation and dropout
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection - downsample if channels change
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stability."""
        for module in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, channels, seq_len]
        """
        # First conv + activation + dropout
        out = self.conv1(x)
        # Chomp: remove the future padding (causal)
        out = out[:, :, :-self.chomp1] if self.chomp1 > 0 else out
        out = self.activation(out)
        out = self.dropout(out)

        # Second conv + activation + dropout
        out = self.conv2(out)
        out = out[:, :, :-self.chomp2] if self.chomp2 > 0 else out
        out = self.activation(out)
        out = self.dropout(out)

        # Residual connection
        residual = self.residual(x)
        return out + residual


class TCNEncoder(nn.Module):
    """
    TCN encoder for a single timeframe.

    Architecture:
        Input projection → Stacked residual blocks with increasing dilation → Global pooling

    The dilation pattern [1, 2, 4, 8, ...] creates exponentially growing receptive field,
    allowing the model to capture both short-term and long-term patterns efficiently.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            hidden_dim: Hidden dimension (output dimension)
            num_blocks: Number of residual blocks (each doubles dilation)
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection to hidden dimension
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        # Stack of residual blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            self.blocks.append(TCNResidualBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))

        # Layer norm for output stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, hidden_dim] - single vector summarizing the sequence
        """
        # Ensure float32
        x = x.float()

        # Conv1D expects [batch, channels, seq_len]
        x = x.transpose(1, 2)

        # Project to hidden dimension
        x = self.input_proj(x)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Back to [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)

        # Global average pooling: aggregate sequence to single vector
        # This is more stable than using last token (less sensitive to sequence length)
        x = x.mean(dim=1)  # [batch, hidden_dim]

        # Layer norm for stability
        x = self.layer_norm(x)

        return x


class TCNAnalyst(nn.Module):
    """
    Multi-timeframe TCN Market Analyst.

    Drop-in replacement for MarketAnalyst (Transformer-based) with the same interface.

    Architecture:
        - 3 TCN encoders (15m, 1H, 4H)
        - Simple concatenation fusion (no attention)
        - Direction prediction head

    Why TCN over Transformer:
        - No attention collapse (the persistent issue in v10-v12)
        - Stable gradient flow through convolutions
        - 4x fewer parameters (93K vs 370K)
        - Faster training
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        d_model: int = 64,
        nhead: int = 4,  # Not used, kept for interface compatibility
        num_layers: int = 2,  # Not used, kept for interface compatibility
        dim_feedforward: int = 128,  # Not used
        context_dim: int = 64,
        dropout: float = 0.3,
        use_lightweight: bool = False,  # Not used
        num_classes: int = 2,
        num_blocks: int = 3,
        kernel_size: int = 3
    ):
        """
        Args:
            feature_dims: Dict mapping timeframe to input feature dimension
                         e.g., {'15m': 12, '1h': 12, '4h': 12}
            d_model: Hidden dimension for TCN encoders
            context_dim: Output context vector dimension
            dropout: Dropout rate
            num_classes: 2 for binary classification
            num_blocks: Number of TCN residual blocks per encoder
            kernel_size: Convolution kernel size
        """
        super().__init__()

        self.d_model = d_model
        self.context_dim = context_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.tcn_num_blocks = num_blocks
        self.tcn_kernel_size = kernel_size

        # TCN encoder for each timeframe
        self.encoder_15m = TCNEncoder(
            input_dim=feature_dims.get('15m', 12),
            hidden_dim=d_model,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.encoder_1h = TCNEncoder(
            input_dim=feature_dims.get('1h', 12),
            hidden_dim=d_model,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.encoder_4h = TCNEncoder(
            input_dim=feature_dims.get('4h', 12),
            hidden_dim=d_model,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Simple concatenation fusion (no attention = no collapse)
        # 3 timeframes * d_model features = 3*d_model input
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        # Context projection (if d_model != context_dim)
        if d_model != context_dim:
            self.context_proj = nn.Sequential(
                nn.Linear(d_model, context_dim),
                nn.LayerNorm(context_dim)
            )
        else:
            self.context_proj = nn.Identity()

        # Direction prediction head (binary: single logit)
        if num_classes == 2:
            self.direction_head = nn.Sequential(
                nn.Linear(context_dim, context_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim // 2, 1)
            )
        else:
            self.direction_head = nn.Sequential(
                nn.Linear(context_dim, context_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim // 2, num_classes)
            )

        # Auxiliary heads (kept for interface compatibility, can be disabled)
        self.volatility_head = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim // 2, 1)
        )

        self.regime_head = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim // 2, 1)
        )

        # Multi-horizon heads (binary mode only)
        if num_classes == 2:
            self.horizon_1h_head = nn.Sequential(
                nn.Linear(context_dim, context_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim // 2, 1)
            )
            self.horizon_2h_head = nn.Sequential(
                nn.Linear(context_dim, context_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim // 2, 1)
            )
        else:
            self.horizon_1h_head = None
            self.horizon_2h_head = None

        # Legacy alias
        self.trend_head = self.direction_head

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Xavier uniform."""
        for module in [self.fusion, self.context_proj, self.direction_head,
                       self.volatility_head, self.regime_head]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            # Small positive bias to break symmetry
                            nn.init.constant_(layer.bias, 0.1)

        # Multi-horizon heads
        for head in [self.horizon_1h_head, self.horizon_2h_head]:
            if head is not None:
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0.1)

    def _encode_and_fuse(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode all timeframes and fuse to context vector.

        Returns:
            context: [batch, context_dim]
        """
        # Encode each timeframe
        enc_15m = self.encoder_15m(x_15m)  # [batch, d_model]
        enc_1h = self.encoder_1h(x_1h)
        enc_4h = self.encoder_4h(x_4h)

        # Concatenate and fuse (simple, no attention collapse possible)
        combined = torch.cat([enc_15m, enc_1h, enc_4h], dim=-1)  # [batch, 3*d_model]
        fused = self.fusion(combined)  # [batch, d_model]

        # Project to context dimension
        context = self.context_proj(fused)  # [batch, context_dim]

        return context

    def forward(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        return_aux: bool = False,
        return_multi_horizon: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Full forward pass for training.

        Same interface as MarketAnalyst.forward().

        Args:
            x_15m: 15-minute features [batch, seq_len, features]
            x_1h: 1-hour features [batch, seq_len, features]
            x_4h: 4-hour features [batch, seq_len, features]
            return_aux: If True, return auxiliary predictions
            return_multi_horizon: If True, return multi-horizon predictions

        Returns:
            Default: (context, direction_logits)
            With return_aux: (context, direction, volatility, regime)
            With return_multi_horizon: (context, direction, horizon_1h, horizon_2h)
            With both: (context, direction, volatility, regime, horizon_1h, horizon_2h)
        """
        # Encode and fuse
        context = self._encode_and_fuse(x_15m, x_1h, x_4h)

        # Direction prediction
        direction = self.direction_head(context)

        if not return_aux and not return_multi_horizon:
            return context, direction

        result = [context, direction]

        if return_aux:
            volatility = self.volatility_head(context).squeeze(-1)
            regime = self.regime_head(context).squeeze(-1)
            result.extend([volatility, regime])

        if return_multi_horizon and self.horizon_1h_head is not None:
            horizon_1h = self.horizon_1h_head(context)
            horizon_2h = self.horizon_2h_head(context)
            result.extend([horizon_1h, horizon_2h])

        return tuple(result)

    @torch.no_grad()
    def get_context(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> torch.Tensor:
        """
        Get context vector only (for RL agent inference).

        Args:
            x_15m, x_1h, x_4h: Timeframe features

        Returns:
            Context vector [batch, context_dim]
        """
        context, _ = self.forward(x_15m, x_1h, x_4h)
        return context

    @torch.no_grad()
    def get_probabilities(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get context vector AND probabilities (for RL agent).

        Returns:
            (context, probs) where probs is [batch, 2] as [p_down, p_up]
        """
        context, logits = self.forward(x_15m, x_1h, x_4h)

        if self.num_classes == 2:
            p_up = torch.sigmoid(logits)  # [batch, 1]
            p_down = 1 - p_up
            probs = torch.cat([p_down, p_up], dim=-1)  # [batch, 2]
        else:
            probs = torch.softmax(logits, dim=-1)

        return context, probs

    def freeze(self):
        """Freeze all parameters for use with RL agent."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze parameters for fine-tuning."""
        self.train()
        for param in self.parameters():
            param.requires_grad = True


def create_tcn_analyst(
    feature_dims: Dict[str, int],
    config: Optional[object] = None,
    device: Optional[torch.device] = None
) -> TCNAnalyst:
    """
    Factory function to create a TCN Analyst with config.

    Args:
        feature_dims: Feature dimensions per timeframe
        config: AnalystConfig object (optional)
        device: Target device

    Returns:
        TCNAnalyst model
    """
    if config is None:
        model = TCNAnalyst(
            feature_dims=feature_dims,
            d_model=64,
            context_dim=64,
            dropout=0.3,
            num_classes=2,
            num_blocks=3,
            kernel_size=3
        )
    else:
        model = TCNAnalyst(
            feature_dims=feature_dims,
            d_model=config.d_model,
            context_dim=config.context_dim,
            dropout=config.dropout,
            num_classes=getattr(config, 'num_classes', 2),
            num_blocks=getattr(config, 'tcn_num_blocks', 3),
            kernel_size=getattr(config, 'tcn_kernel_size', 3)
        )

    if device is not None:
        model = model.to(device)

    return model


def load_tcn_analyst(
    path: str,
    feature_dims: Dict[str, int],
    device: Optional[torch.device] = None,
    freeze: bool = True
) -> TCNAnalyst:
    """
    Load a trained TCN Analyst from checkpoint.

    Args:
        path: Path to checkpoint file
        feature_dims: Feature dimensions (must match training)
        device: Target device
        freeze: Whether to freeze the model

    Returns:
        Loaded TCNAnalyst
    """
    checkpoint = torch.load(path, map_location=device or 'cpu')

    # Create model from saved config
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        model = TCNAnalyst(
            feature_dims=feature_dims,
            d_model=saved_config.get('d_model', 64),
            context_dim=saved_config.get('context_dim', 64),
            dropout=saved_config.get('dropout', 0.3),
            num_classes=saved_config.get('num_classes', 2),
            num_blocks=saved_config.get('tcn_num_blocks', 3),
            kernel_size=saved_config.get('tcn_kernel_size', 3)
        )
    else:
        model = create_tcn_analyst(feature_dims)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    if device is not None:
        model = model.to(device)

    if freeze:
        model.freeze()

    # Clear memory
    del checkpoint
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return model
