"""
Pydantic models for visualization data structures.

These define the shape of data streamed via WebSocket to the frontend.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class OHLCBar(BaseModel):
    """Single OHLC price bar."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class TradeMarker(BaseModel):
    """Trade entry/exit marker for chart."""
    timestamp: float
    price: float
    direction: int  # 1=Long, -1=Short
    size: float
    pnl: Optional[float] = None
    is_entry: bool = True
    close_reason: Optional[str] = None  # 'exit', 'stop_loss', 'take_profit'


class AnalystState(BaseModel):
    """Market Analyst model state."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_acc: float = 0.0
    val_acc: float = 0.0
    direction_acc: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0

    # Attention and predictions
    attention_weights: List[float] = [0.5, 0.5]  # [1h_weight, 4h_weight]
    p_down: float = 0.5
    p_up: float = 0.5
    p_neutral: Optional[float] = None  # For multi-class
    confidence: float = 0.5
    edge: float = 0.0
    uncertainty: float = 0.5

    # Encoder outputs (for visualization)
    encoder_15m_norm: float = 0.0  # L2 norm of encoder output
    encoder_1h_norm: float = 0.0
    encoder_4h_norm: float = 0.0
    context_vector_sample: List[float] = []  # First 8 dims for sparkline


class AgentState(BaseModel):
    """PPO Agent state."""
    timestep: int = 0
    episode: int = 0
    episode_reward: float = 0.0
    episode_pnl: float = 0.0
    episode_trades: int = 0
    win_rate: float = 0.0

    # Action probabilities from policy
    action_probs: List[float] = [0.33, 0.33, 0.34]  # [flat, long, short]
    size_probs: List[float] = [0.25, 0.25, 0.25, 0.25]  # [0.25x, 0.5x, 0.75x, 1.0x]

    # Value function
    value_estimate: float = 0.0
    advantage: float = 0.0

    # Exploration
    entropy: float = 0.0

    # Action taken
    last_action_direction: int = 0  # 0=Flat, 1=Long, 2=Short
    last_action_size: int = 0  # 0-3


class MarketState(BaseModel):
    """Current market and position state."""
    # Price
    current_price: float = 0.0
    ohlc: Optional[OHLCBar] = None

    # Position
    position: int = 0  # -1=Short, 0=Flat, 1=Long
    position_size: float = 0.0
    entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    n_trades: int = 0

    # Risk levels
    sl_level: Optional[float] = None
    tp_level: Optional[float] = None

    # Market features (raw)
    atr: float = 0.0
    chop: float = 50.0
    adx: float = 25.0
    regime: int = 1  # 0=Bullish, 1=Ranging, 2=Bearish
    sma_distance: float = 0.0


class RewardComponents(BaseModel):
    """Breakdown of reward signal."""
    pnl_delta: float = 0.0
    transaction_cost: float = 0.0
    direction_bonus: float = 0.0
    confidence_bonus: float = 0.0
    fomo_penalty: float = 0.0
    chop_penalty: float = 0.0
    total: float = 0.0


class SystemStatus(BaseModel):
    """System health metrics."""
    phase: str = "idle"  # 'idle', 'analyst_training', 'agent_training', 'backtest'
    memory_used_mb: float = 0.0
    memory_total_mb: float = 8192.0
    steps_per_second: float = 0.0
    episodes_per_hour: float = 0.0
    elapsed_seconds: float = 0.0
    device: str = "mps"


class TrainingSnapshot(BaseModel):
    """
    Complete training state snapshot.

    This is the main message format sent via WebSocket.
    All fields are optional to allow incremental updates.
    """
    timestamp: float
    message_type: str = "snapshot"  # 'snapshot', 'trade', 'episode_end'

    # Component states
    analyst: Optional[AnalystState] = None
    agent: Optional[AgentState] = None
    market: Optional[MarketState] = None
    reward: Optional[RewardComponents] = None
    system: Optional[SystemStatus] = None

    # Trade event (when a trade closes)
    trade: Optional[TradeMarker] = None

    # History for charts (sent less frequently)
    price_history: Optional[List[OHLCBar]] = None
    trade_history: Optional[List[TradeMarker]] = None
    loss_history: Optional[List[Dict[str, float]]] = None
    reward_history: Optional[List[float]] = None
    
    # Chart control
    clear_chart: bool = False  # When True, frontend should clear price history

    class Config:
        # Allow extra fields for extensibility
        extra = "allow"


class HistoryResponse(BaseModel):
    """Response for history API endpoint."""
    price_bars: List[OHLCBar]
    trades: List[TradeMarker]
    loss_history: List[Dict[str, float]]
    reward_history: List[float]
    pnl_history: List[float]


class ModelInfoResponse(BaseModel):
    """Response for model info API endpoint."""
    analyst: Dict[str, Any]
    agent: Dict[str, Any]
    config: Dict[str, Any]
