"""
Real-time training visualization backend.

This package provides WebSocket streaming of training metrics
from the Market Analyst and PPO Agent to a React frontend dashboard.
"""

from .data_emitter import DataEmitter, get_emitter, TrainingSnapshot
from .config import VisualizationConfig

__all__ = [
    'DataEmitter',
    'get_emitter',
    'TrainingSnapshot',
    'VisualizationConfig'
]
