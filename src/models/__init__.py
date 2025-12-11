"""
Neural network models for the EURUSD trading system.

This module provides the core ML architectures:
- Encoders: TransformerEncoder for temporal feature extraction
- Fusion: AttentionFusion for multi-timeframe representation learning
- Analysts: MarketAnalyst (Transformer-based) and TCNAnalyst (TCN-based)

The TCNAnalyst is the recommended architecture for production (more stable).
"""

from .encoders import TransformerEncoder, PositionalEncoding
from .fusion import AttentionFusion
from .analyst import MarketAnalyst, create_analyst, load_analyst
from .tcn_analyst import TCNAnalyst, create_tcn_analyst, load_tcn_analyst

__all__ = [
    # Encoders
    'TransformerEncoder',
    'PositionalEncoding',
    # Fusion
    'AttentionFusion',
    # Transformer-based Analyst
    'MarketAnalyst',
    'create_analyst',
    'load_analyst',
    # TCN-based Analyst (Recommended)
    'TCNAnalyst',
    'create_tcn_analyst',
    'load_tcn_analyst',
]
