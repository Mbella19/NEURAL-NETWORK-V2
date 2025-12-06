"""
FastAPI server for real-time training visualization.

Endpoints:
- WS /ws/training: Real-time training metrics stream
- GET /api/health: Server health check
- GET /api/history: Historical training data
- GET /api/state: Current training state
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    import orjson

    def json_dumps(obj):
        return orjson.dumps(obj).decode()

except ImportError:
    import json

    def json_dumps(obj):
        return json.dumps(obj, default=str)


from .websocket_manager import ConnectionManager
from .data_emitter import get_emitter
from .config import VisualizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
manager = ConnectionManager()
config = VisualizationConfig(server_mode=True)
broadcast_task: Optional[asyncio.Task] = None


async def broadcast_loop():
    """
    Continuously broadcast new data to all connected clients.

    Runs at configured update rate (default 10Hz).
    """
    emitter = get_emitter()
    interval = 1.0 / config.update_hz

    logger.info(f"Starting broadcast loop at {config.update_hz}Hz")

    while True:
        try:
            if manager.has_connections():
                # Get all pending snapshots
                snapshots = emitter.get_all_pending()

                if snapshots:
                    # Send the most recent one
                    latest = snapshots[-1]
                    message = json_dumps(latest.model_dump())
                    await manager.broadcast(message)

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("Broadcast loop cancelled")
            break
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global broadcast_task

    # Startup
    logger.info("Starting visualization server...")
    
    # Configure emitter with server mode
    emitter = get_emitter()
    emitter.config = config
    
    broadcast_task = asyncio.create_task(broadcast_loop())

    yield

    # Shutdown
    logger.info("Shutting down visualization server...")
    if broadcast_task:
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass


# Create FastAPI app
app = FastAPI(
    title="Trading Bot Visualization",
    description="Real-time training visualization for EURUSD hybrid trading system",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        config.frontend_url,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# WebSocket Endpoints
# =========================================================================


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training data.

    Clients receive:
    - Initial full state on connection
    - Incremental updates during training
    - Trade events
    - Episode summaries
    """
    await manager.connect(websocket)

    # Send initial state
    emitter = get_emitter()
    try:
        initial_state = emitter.get_current_state()
        await websocket.send_text(json_dumps(initial_state.model_dump()))
    except Exception as e:
        logger.error(f"Error sending initial state: {e}")

    try:
        while True:
            # Handle incoming messages from client
            data = await websocket.receive_text()

            # Process control messages
            if data == "ping":
                await websocket.send_text("pong")
            elif data == "get_state":
                state = emitter.get_current_state()
                await websocket.send_text(json_dumps(state.model_dump()))
            elif data == "get_history":
                history = emitter.get_history()
                await websocket.send_text(json_dumps(history))

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


# =========================================================================
# REST API Endpoints
# =========================================================================


@app.get("/api/health")
async def health_check():
    """Server health check."""
    emitter = get_emitter()
    return {
        "status": "healthy",
        "connections": manager.connection_count,
        "emitter_enabled": emitter.is_enabled(),
    }


@app.get("/api/state")
async def get_current_state():
    """Get current training state."""
    emitter = get_emitter()
    state = emitter.get_current_state()
    return JSONResponse(content=state.model_dump())


@app.get("/api/history")
async def get_history():
    """Get historical training data for charts."""
    emitter = get_emitter()
    history = emitter.get_history()
    return JSONResponse(content=history)


@app.post("/api/reset")
async def reset_emitter():
    """Reset emitter state (for new training run)."""
    emitter = get_emitter()
    emitter.reset()
    return {"status": "reset"}


@app.post("/api/enable")
async def enable_emitter():
    """Enable data emission."""
    emitter = get_emitter()
    emitter.enable()
    return {"status": "enabled"}


@app.post("/api/disable")
async def disable_emitter():
    """Disable data emission."""
    emitter = get_emitter()
    emitter.disable()
    return {"status": "disabled"}


@app.post("/api/ingest")
async def ingest_snapshot(snapshot: dict):
    """
    Ingest a training snapshot from an external process.
    
    This allows the training script (running in a separate process)
    to push data to the visualization server.
    """
    try:
        # Import here to avoid circular imports
        from .models import TrainingSnapshot
        
        # Parse and validate
        snap_obj = TrainingSnapshot(**snapshot)
        
        # Push to local emitter (which feeds WebSocket)
        emitter = get_emitter()
        emitter._push_snapshot(snap_obj)
        
        # Also update local history buffers for API access
        # (This duplicates some logic from push_ methods but is necessary
        # since we're receiving the final object)
        with emitter._lock:
            emitter._snapshot_history.append(snap_obj)
            
            if snap_obj.market and snap_obj.market.ohlc:
                emitter._price_history.append(snap_obj.market.ohlc)
                
            if snap_obj.trade:
                emitter._trade_history.append(snap_obj.trade)
                
            if snap_obj.reward and snap_obj.reward.total is not None:
                emitter._reward_history.append(snap_obj.reward.total)
                
            if snap_obj.market and snap_obj.market.total_pnl is not None:
                emitter._pnl_history.append(snap_obj.market.total_pnl)
                
            # Update latest state cache
            if snap_obj.analyst: emitter._latest_analyst = snap_obj.analyst
            if snap_obj.agent: emitter._latest_agent = snap_obj.agent
            if snap_obj.market: emitter._latest_market = snap_obj.market
            if snap_obj.reward: emitter._latest_reward = snap_obj.reward
            if snap_obj.system: emitter._latest_system = snap_obj.system

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})


# =========================================================================
# Main Entry Point
# =========================================================================


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the visualization server."""
    import uvicorn

    logger.info(f"Starting visualization server on {host}:{port}")
    uvicorn.run(
        "visualization.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
