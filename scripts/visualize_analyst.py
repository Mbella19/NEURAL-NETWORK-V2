#!/usr/bin/env python3
"""
Standalone Analyst Visualization Tool.

This script runs a lightweight Flask server to visualize the Analyst model's
performance on the last month of data. It is completely separate from the
main training dashboard.

Usage:
    python scripts/visualize_analyst.py

Dependencies:
    pip install flask flask-socketio
"""

import sys
import os
import time
import threading
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, get_device
from src.models.analyst import load_analyst
from src.data.features import add_market_sessions, detect_fractals, detect_structure_breaks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing Flask dependencies
try:
    from flask import Flask, render_template, send_from_directory
    from flask_socketio import SocketIO
except ImportError:
    logger.error("Flask or Flask-SocketIO not found.")
    logger.info("Please run: pip install flask flask-socketio eventlet")
    sys.exit(1)

# Initialize Flask
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'analyst_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class SimulationState:
    def __init__(self):
        self.running = False
        self.paused = False
        self.speed = 1.0  # Seconds per step (lower is faster)
        self.current_idx = 0
        self.data = None
        self.model = None
        self.device = None
        self.lookbacks = {}
        self.feature_cols = []

state = SimulationState()

def prepare_data():
    """Load model and data for the last 30 days."""
    config = Config()
    device = get_device()
    state.device = device

    # 1. Load Data
    logger.info("Loading data...")
    try:
        # Load normalized data for model
        df_15m = pd.read_parquet(config.paths.data_processed / 'features_15m_normalized.parquet')
        df_1h = pd.read_parquet(config.paths.data_processed / 'features_1h_normalized.parquet')
        df_4h = pd.read_parquet(config.paths.data_processed / 'features_4h_normalized.parquet')
        
        # Load raw data for visualization (prices)
        df_15m_raw = pd.read_parquet(config.paths.data_processed / 'features_15m.parquet')
    except FileNotFoundError:
        logger.error("Processed data not found. Run pipeline first.")
        sys.exit(1)

    # 2. Filter last 30 days (approx 4*24*30 = 2880 bars)
    days_to_visualize = 30
    bars_15m = 4 * 24 * days_to_visualize
    
    # Ensure we have enough data
    if len(df_15m) < bars_15m:
        logger.warning(f"Not enough data for {days_to_visualize} days. Using all available.")
        start_idx = 0
    else:
        start_idx = len(df_15m) - bars_15m

    # Align all dataframes
    df_15m = df_15m.iloc[start_idx:].reset_index(drop=True)
    df_15m_raw = df_15m_raw.iloc[start_idx:].reset_index(drop=True) # Align raw data
    df_1h = df_1h.iloc[-len(df_15m):].reset_index(drop=True)
    df_4h = df_4h.iloc[-len(df_15m):].reset_index(drop=True)

    # 3. Load Model
    analyst_path = config.paths.models_analyst / 'best.pt'
    if not analyst_path.exists():
        logger.error(f"Model not found at {analyst_path}")
        sys.exit(1)

    # Define features (must match training)
    feature_cols = [
        'returns', 'volatility', 'pinbar', 'engulfing', 'doji',
        'ema_trend', 'ema_crossover', 'regime', 'sma_distance',
        'dist_to_resistance', 'dist_to_support',
        'session_asian', 'session_london', 'session_ny',
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    ]
    feature_cols = [c for c in feature_cols if c in df_15m.columns]
    state.feature_cols = feature_cols

    feature_dims = {
        '15m': len(feature_cols),
        '1h': len(feature_cols),
        '4h': len(feature_cols)
    }
    
    logger.info(f"Loading model from {analyst_path}")
    state.model = load_analyst(str(analyst_path), feature_dims, device, freeze=True)
    state.model.eval()

    # 4. Prepare Tensor Data
    state.lookbacks = {
        '15m': config.analyst.lookback_15m,
        '1h': config.analyst.lookback_1h,
        '4h': config.analyst.lookback_4h
    }
    
    # Store dataframes for access
    state.data = {
        '15m': df_15m,
        '1h': df_1h,
        '4h': df_4h,
        'raw_close': df_15m_raw['close'].values # Use RAW close prices
    }
    
    logger.info(f"Ready to visualize {len(df_15m)} steps.")

def simulation_loop():
    """Background thread to run the simulation."""
    logger.info("Simulation loop started.")
    
    # Warmup period
    max_lookback = max(state.lookbacks.values())
    state.current_idx = max_lookback
    
    while True:
        if not state.running:
            time.sleep(0.1)
            continue
            
        if state.paused:
            time.sleep(0.1)
            continue
            
        if state.current_idx >= len(state.data['15m']):
            logger.info("Simulation finished. Restarting.")
            state.current_idx = max_lookback
            
        # 1. Prepare Input
        idx = state.current_idx
        
        # Extract windows
        # Note: This is a simplified extraction. In real pipeline we handle subsampling carefully.
        # Here we assume 1h/4h are aligned row-by-row (which they are in processed data usually)
        # But wait, processed data might be subsampled? 
        # The pipeline aligns them to 15m index. So we can just index them directly.
        
        def get_window(df, idx, lookback):
            if idx < lookback: return np.zeros((lookback, len(state.feature_cols)))
            return df[state.feature_cols].iloc[idx-lookback:idx].values.astype(np.float32)

        x_15m = get_window(state.data['15m'], idx, state.lookbacks['15m'])
        x_1h = get_window(state.data['1h'], idx, state.lookbacks['1h'])
        x_4h = get_window(state.data['4h'], idx, state.lookbacks['4h'])
        
        # To Tensor
        t_15m = torch.tensor(x_15m, device=state.device).unsqueeze(0) # [1, L, F]
        t_1h = torch.tensor(x_1h, device=state.device).unsqueeze(0)
        t_4h = torch.tensor(x_4h, device=state.device).unsqueeze(0)
        
        # 2. Model Inference
        with torch.no_grad():
            # Get activations and probs
            if hasattr(state.model, 'get_activations'):
                context, activations = state.model.get_activations(t_15m, t_1h, t_4h)
                _, probs = state.model.get_probabilities(t_15m, t_1h, t_4h)
                
                # Process activations for JSON
                act_data = {k: v[0].cpu().numpy().tolist() for k, v in activations.items()}
            else:
                # Fallback
                probs = torch.tensor([[0.5, 0.5]], device=state.device)
                act_data = {}

        # 3. Emit Data
        current_price = float(state.data['raw_close'][idx-1]) # Previous close is current price effectively
        
        p_down = float(probs[0, 0].item())
        p_up = float(probs[0, 1].item())
        
        data = {
            'step': idx,
            'price': current_price,
            'p_up': p_up,
            'p_down': p_down,
            'activations': act_data,
            'timestamp': datetime.now().isoformat() # Simulated time
        }
        
        socketio.emit('new_step', data)
        
        # Advance
        state.current_idx += 1
        
        # Sleep
        time.sleep(state.speed)

# Routes
@app.route('/')
def index():
    return render_template('analyst_dashboard.html')

@app.route('/start')
def start():
    state.running = True
    state.paused = False
    return {'status': 'started'}

@app.route('/pause')
def pause():
    state.paused = True
    return {'status': 'paused'}

@app.route('/reset')
def reset():
    state.current_idx = max(state.lookbacks.values())
    return {'status': 'reset'}

@app.route('/speed/<float:speed>')
def set_speed(speed):
    state.speed = max(0.01, min(2.0, speed))
    return {'status': 'speed_set', 'speed': state.speed}

if __name__ == '__main__':
    prepare_data()
    
    # Start simulation thread
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    
    logger.info("Starting Flask server at http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
