"""
WebSocket connection manager.

Handles multiple client connections and broadcasts training data.
"""

import asyncio
from typing import List, Set
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time data streaming.

    Features:
    - Multiple simultaneous connections
    - Graceful disconnect handling
    - Broadcast to all connected clients
    - Connection health monitoring
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        if not self.active_connections:
            return

        # Copy to avoid modification during iteration
        connections = list(self.active_connections)
        disconnected = []

        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    self.active_connections.discard(conn)

    async def send_personal(self, websocket: WebSocket, message: str):
        """Send message to a specific client."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def has_connections(self) -> bool:
        """Check if there are any active connections."""
        return len(self.active_connections) > 0
