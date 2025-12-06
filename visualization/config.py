"""
Visualization server configuration.
"""

from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for the visualization server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Update rates
    update_hz: int = 10  # WebSocket updates per second

    # Buffer sizes
    max_snapshots: int = 10000  # Max training snapshots to keep
    max_price_bars: int = 500   # Max price bars for chart
    max_trades: int = 100       # Max trade markers to keep

    # Frontend
    frontend_url: str = "http://localhost:3000"

    # Performance
    batch_updates: bool = True  # Batch multiple updates together
    use_orjson: bool = True     # Use faster JSON serialization
    
    # Mode
    server_mode: bool = False   # If True, do not push to remote server (prevents recursion)
