# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hybrid EURUSD Trading System using a two-stage architecture:
1. **Market Analyst**: Transformer/TCN model trained via supervised learning to predict price direction
2. **Sniper Agent**: PPO-based RL agent that uses frozen Analyst context vectors to make trading decisions

Optimized for Apple M2 Silicon with 8GB RAM constraints. All computations use float32.

## Common Commands

### Full Training Pipeline
```bash
python scripts/run_pipeline.py
```

### Train Components Separately
```bash
# Train Market Analyst only
python -m src.training.train_analyst

# Train PPO Agent only (requires trained Analyst)
python -m src.training.train_agent

# Pre-compute Analyst outputs for faster agent training
python -m src.training.precompute_analyst
```

### Backtesting
```bash
python scripts/backtest_period.py
```

### Real-time Dashboard
```bash
# Start backend
python scripts/start_dashboard.py

# Start frontend (separate terminal)
cd frontend && npm run dev
```

### Plot Training Progress
```bash
python scripts/plot_training_progress.py
python scripts/plot_logs.py
```

## Architecture

### Two-Stage Pipeline

```
Raw 1M Data → Resample → 15m/1H/4H DataFrames → Feature Engineering
                                                        ↓
                              ┌─────────────────────────┴─────────────────────────┐
                              ↓                                                   ↓
                    Market Analyst (Supervised)                          PPO Agent (RL)
                    - TransformerEncoder per TF                    - Uses frozen Analyst context
                    - AttentionFusion combines TFs                 - MultiDiscrete action: [Dir, Size]
                    - Predicts direction (binary)                  - Gymnasium TradingEnv
                              ↓                                                   ↓
                    Freeze weights                                    Train with SB3 PPO
                              ↓                                                   ↓
                    Context vectors → TradingEnv observation space → Agent learns trading
```

### Key Directory Structure

- `src/models/analyst.py` - MarketAnalyst (Transformer-based)
- `src/models/tcn_analyst.py` - TCNAnalyst (more stable, recommended)
- `src/models/encoders.py` - TransformerEncoder, TCN blocks
- `src/models/fusion.py` - AttentionFusion for multi-timeframe
- `src/agents/sniper_agent.py` - SniperAgent wrapper for SB3 PPO
- `src/environments/trading_env.py` - Gymnasium trading environment
- `src/training/train_analyst.py` - Supervised training for Analyst
- `src/training/train_agent.py` - RL training for Agent
- `src/data/features.py` - Technical indicators and pattern detection
- `config/settings.py` - All hyperparameters and configuration

### Observation Space (TradingEnv)

The agent receives a ~49-dimensional observation:
1. **Context Vector** (32 dims): Frozen Analyst embeddings
2. **Position State** (3 dims): position, entry_price_norm, unrealized_pnl_norm
3. **Market Features** (5-7 dims): ATR, CHOP, ADX, regime, sma_distance, S/R distances
4. **Analyst Metrics** (5 dims): p_down, p_up, edge, confidence, uncertainty
5. **SL/TP Distance** (2 dims): ATR-normalized distance to stop-loss/take-profit

### Action Space

MultiDiscrete([3, 4]):
- Direction: 0=Flat/Exit, 1=Long, 2=Short
- Size: 0=0.25x, 1=0.5x, 2=0.75x, 3=1.0x

### Reward Function

Continuous PnL delta model (not just on exit):
- `reward = (current_unrealized_pnl - prev_unrealized_pnl) * scaling`
- Transaction cost on entry: `-(spread + slippage) * size * scaling`
- Trade entry bonus to encourage exploration
- FOMO penalty when flat during high momentum
- Chop penalty when holding in extreme ranging conditions

## Critical Implementation Details

### Look-Ahead Bias Prevention
- All features use only past data
- Fractals detected with delay (confirmed after n bars)
- Normalization uses training data statistics only
- Market feature stats computed from training split only

### Multi-Timeframe Alignment
- Higher timeframes (1H, 4H) are forward-filled to 15m index
- Subsampling: 1H uses every 4th bar, 4H uses every 16th bar
- All three DataFrames must have identical indices after alignment

### Regime-Balanced Sampling
- Training data often has directional bias (e.g., 61% Ranging)
- `use_regime_sampling=True` ensures 33% Bullish, 33% Ranging, 33% Bearish
- Prevents agent from learning "always stay flat"

### Memory Management (M2 8GB)
- Batch processing with MPS cache clearing
- Pre-computed Analyst context vectors
- Gradient accumulation to simulate larger batches
- `clear_memory()` utility for explicit cleanup

## Configuration

All hyperparameters in `config/settings.py`:
- `AnalystConfig`: d_model=32, num_layers=2, context_dim=32, architecture="tcn"
- `AgentConfig`: lr=1e-4, n_steps=2048, batch_size=256, ent_coef=0.01
- `TradingConfig`: spread=0.2, slippage=0.5, sl_atr=1.0, tp_atr=3.0, reward_scaling=0.1

## Feature Columns

Standard 18 model input features:
```
returns, volatility, pinbar, engulfing, doji, ema_trend, ema_crossover,
regime, sma_distance, dist_to_resistance, dist_to_support,
session_asian, session_london, session_ny,
bos_bullish, bos_bearish, choch_bullish, choch_bearish
```

## Known Constraints

- **DO NOT** enable `enforce_analyst_alignment` - breaks PPO gradients
- **DO NOT** use direction bonuses on exit - causes reward-PnL divergence
- TCN architecture is more stable than Transformer for binary classification
- Checkpoints saved every 100k steps to `models/agent/checkpoints/`
