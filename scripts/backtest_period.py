#!/usr/bin/env python3
"""
Backtest on a specific date range.

Usage:
    python scripts/backtest_period.py --start-date 2025-07-01 --end-date 2025-07-31 --model-path models/checkpoints/sniper_model_4400000_steps.zip --min-confidence 0.60
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from src.agents.sniper_agent import SniperAgent
from src.models.analyst import load_analyst
from src.training.train_agent import create_trading_env, prepare_env_data
from src.evaluation.backtest import run_backtest, compare_with_baseline
from src.evaluation.backtest import print_metrics_report, print_comparison_report, save_backtest_results


def main():
    parser = argparse.ArgumentParser(description='Backtest on specific date range')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-path', required=True, help='Path to agent model')
    parser.add_argument('--min-confidence', type=float, default=0.60, help='Min confidence threshold')
    args = parser.parse_args()

    config = Config()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    df_15m = pd.read_parquet('data/processed/features_15m_normalized.parquet')
    df_1h = pd.read_parquet('data/processed/features_1h_normalized.parquet')
    df_4h = pd.read_parquet('data/processed/features_4h_normalized.parquet')

    # Filter to date range
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    
    df_15m_period = df_15m.loc[start_date:end_date].copy()
    df_1h_period = df_1h.loc[start_date:end_date].copy()
    df_4h_period = df_4h.loc[start_date:end_date].copy()

    print(f"\n{'='*60}")
    print(f"PERIOD BACKTEST: {args.start_date} to {args.end_date}")
    print(f"{'='*60}")
    print(f"15m bars: {len(df_15m_period)}")
    
    # Calculate period return
    if len(df_15m_period) > 0:
        start_price = df_15m_period['close'].iloc[0]
        end_price = df_15m_period['close'].iloc[-1]
        period_return = (end_price - start_price) / start_price * 100
        print(f"Period price change: {period_return:+.2f}%")
        if period_return < 0:
            print("📉 BEARISH PERIOD")
        else:
            print("📈 BULLISH PERIOD")
    print()

    # Get feature columns - MUST match what the analyst was trained on (18 features)
    feature_cols = [
        'returns', 'volatility',           # Price dynamics (normalized)
        'pinbar', 'engulfing', 'doji',     # Price action patterns
        'ema_trend', 'ema_crossover',      # Trend indicators
        'regime', 'sma_distance',          # Regime/trend filters
        'dist_to_resistance', 'dist_to_support', # S/R distance
        'bos_bullish', 'bos_bearish',      # Market Structure (Break of Structure)
        'choch_bullish', 'choch_bearish',  # Market Structure (Change of Character)
        'atr', 'chop', 'adx'               # Volatility & Strength (Normalized)
    ]
    # Filter to only columns that exist in the data
    feature_cols = [c for c in feature_cols if c in df_15m_period.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")


    # Load analyst
    analyst_path = config.paths.models_analyst / 'best.pt'
    feature_dims = {'15m': len(feature_cols), '1h': len(feature_cols), '4h': len(feature_cols)}
    analyst = load_analyst(str(analyst_path), feature_dims, device, freeze=True)

    # Prepare environment data
    lookback_15m = 48
    lookback_1h = 24
    lookback_4h = 12

    data_15m, data_1h, data_4h, close_prices, market_features = prepare_env_data(
        df_15m_period, df_1h_period, df_4h_period, feature_cols,
        lookback_15m, lookback_1h, lookback_4h
    )

    if len(close_prices) < 100:
        print(f"ERROR: Not enough data after lookback alignment. Got {len(close_prices)} bars, need at least 100.")
        sys.exit(1)

    print(f"Backtest bars: {len(close_prices)}")

    # Use full dataset stats for normalization (to be consistent with training)
    df_15m_full = pd.read_parquet('data/processed/features_15m_normalized.parquet')
    df_1h_full = pd.read_parquet('data/processed/features_1h_normalized.parquet')
    df_4h_full = pd.read_parquet('data/processed/features_4h_normalized.parquet')
    
    _, _, _, _, full_market_features = prepare_env_data(
        df_15m_full, df_1h_full, df_4h_full, feature_cols,
        lookback_15m, lookback_1h, lookback_4h
    )
    
    # Use first 85% for normalization stats
    train_end = int(0.85 * len(full_market_features))
    train_market_features = full_market_features[:train_end]
    market_feat_mean = train_market_features.mean(axis=0).astype(np.float32)
    market_feat_std = train_market_features.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)

    # Create test environment
    test_config = config.trading
    test_config.noise_level = 0.0
    
    test_env = create_trading_env(
        data_15m, data_1h, data_4h, close_prices, market_features,
        analyst_model=analyst,
        config=test_config,
        device=device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std
    )

    # Load agent
    from stable_baselines3.common.monitor import Monitor
    test_env = Monitor(test_env)
    agent = SniperAgent.load(args.model_path, test_env, device='cpu')

    print("Running backtest...")
    
    # Run backtest
    results = run_backtest(
        agent=agent,
        env=test_env.unwrapped,
        min_action_confidence=args.min_confidence,
        spread_pips=config.trading.spread_pips + config.trading.slippage_pips
    )

    # Compare with buy-and-hold
    comparison = compare_with_baseline(
        results,
        close_prices,
        initial_balance=10000.0
    )

    # Print reports
    print_metrics_report(results.metrics, f"Agent Performance ({args.start_date} to {args.end_date})")
    print_comparison_report(comparison)

    # Trade direction breakdown
    if hasattr(results, 'trades') and results.trades:
        longs = sum(1 for t in results.trades if getattr(t, 'direction', None) == 'Long')
        shorts = sum(1 for t in results.trades if getattr(t, 'direction', None) == 'Short')
        
        print("\n" + "="*60)
        print("TRADE DIRECTION BREAKDOWN")
        print("="*60)
        print(f"Long trades:  {longs}")
        print(f"Short trades: {shorts}")
        
        # PnL by direction
        long_pnl = sum(getattr(t, 'pnl_pips', 0) for t in results.trades if getattr(t, 'direction', None) == 'Long')
        short_pnl = sum(getattr(t, 'pnl_pips', 0) for t in results.trades if getattr(t, 'direction', None) == 'Short')
        print(f"Long PnL:     {long_pnl:.1f} pips")
        print(f"Short PnL:    {short_pnl:.1f} pips")


    # Save results
    results_path = config.paths.base_dir / 'results' / f"period_{args.start_date}_{args.end_date}"
    save_backtest_results(results, str(results_path), comparison)

    test_env.close()
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
