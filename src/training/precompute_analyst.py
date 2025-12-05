#!/usr/bin/env python3
"""
Pre-compute Analyst outputs for sequential context.

This script runs the Analyst model through the ENTIRE dataset sequentially,
ensuring it sees the full historical context at each timestep. The outputs
(context vectors and probabilities) are cached to disk for use during PPO training.

This fixes the "train-test mismatch" issue where the Analyst only saw
small random windows during PPO training, leading to unreliable predictions.

Usage:
    python src/training/precompute_analyst.py --analyst-path models/analyst/best.pt

The cached outputs are saved to: data/processed/analyst_cache.npz
"""

import sys
from pathlib import Path
import argparse
import logging
import numpy as np
import torch
import pandas as pd
import gc
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, get_device, clear_memory
from src.models.analyst import load_analyst
from src.data.features import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_analyst_outputs(
    analyst_path: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    feature_cols: list,
    lookback_15m: int = 48,
    lookback_1h: int = 24,
    lookback_4h: int = 12,
    device: torch.device = None,
    batch_size: int = 64,
    save_path: str = None,
) -> dict:
    """
    Pre-compute Analyst outputs by running SEQUENTIALLY through all data.
    
    This ensures the Analyst sees continuous historical context, not random fragments.
    
    Args:
        analyst_path: Path to trained Analyst model
        df_15m, df_1h, df_4h: Multi-timeframe DataFrames
        feature_cols: Feature columns to use
        lookback_*: Lookback windows per timeframe
        device: Torch device
        batch_size: Batch size for inference
        save_path: Where to save the cached outputs
        
    Returns:
        Dict with 'contexts' and 'probs' arrays
    """
    if device is None:
        device = get_device()
    
    # Load Analyst model
    feature_dims = {
        '15m': len(feature_cols),
        '1h': len(feature_cols),
        '4h': len(feature_cols)
    }
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)
    analyst.eval()
    
    # Extract features as numpy arrays
    features_15m = df_15m[feature_cols].values.astype(np.float32)
    features_1h = df_1h[feature_cols].values.astype(np.float32)
    features_4h = df_4h[feature_cols].values.astype(np.float32)
    
    # Get close prices for reference
    close_prices = df_15m['close'].values.astype(np.float32)
    
    # Calculate valid start index (need full lookback)
    start_idx = max(lookback_15m, lookback_1h, lookback_4h)
    n_samples = len(features_15m) - start_idx
    
    logger.info(f"Pre-computing Analyst outputs for {n_samples:,} timesteps...")
    logger.info(f"Start index: {start_idx} (after lookback warmup)")
    
    # Prepare windowed data (pre-window for each index)
    # This creates [n_samples, lookback, features] arrays
    logger.info("Creating windowed data arrays...")
    
    data_15m = np.zeros((n_samples, lookback_15m, len(feature_cols)), dtype=np.float32)
    data_1h = np.zeros((n_samples, lookback_1h, len(feature_cols)), dtype=np.float32)
    data_4h = np.zeros((n_samples, lookback_4h, len(feature_cols)), dtype=np.float32)
    
    for i in range(n_samples):
        idx = start_idx + i
        data_15m[i] = features_15m[idx - lookback_15m:idx]
        data_1h[i] = features_1h[idx - lookback_1h:idx]
        data_4h[i] = features_4h[idx - lookback_4h:idx]
    
    logger.info(f"Windowed data shapes: 15m={data_15m.shape}, 1h={data_1h.shape}, 4h={data_4h.shape}")
    
    # Run Analyst in batches (SEQUENTIALLY - order matters for cumulative context)
    all_contexts = []
    all_probs = []
    all_activations = {'15m': [], '1h': [], '4h': []}
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_i = batch_idx * batch_size
            end_i = min(start_i + batch_size, n_samples)
            
            # Get batch data
            batch_15m = torch.tensor(data_15m[start_i:end_i], device=device)
            batch_1h = torch.tensor(data_1h[start_i:end_i], device=device)
            batch_4h = torch.tensor(data_4h[start_i:end_i], device=device)
            
            # Get Analyst outputs (including activations for visualization)
            if hasattr(analyst, 'get_activations'):
                context, activations = analyst.get_activations(batch_15m, batch_1h, batch_4h)
                
                # Get probs separately (or modify get_activations to return them too, but this is cleaner for now)
                # For efficiency, we can just call forward if we modify it, but let's stick to the plan
                # Actually, let's just get probs from forward to avoid double computation if possible
                # But TCN is fast. Let's just call get_probabilities separately or rely on forward.
                # Wait, get_activations calls _encode_and_fuse. get_probabilities calls forward which calls _encode_and_fuse.
                # To avoid double compute, let's just use forward if we can, but forward doesn't return activations easily without flag.
                # Let's use the new get_activations and then get probs.
                
                _, probs = analyst.get_probabilities(batch_15m, batch_1h, batch_4h)
                
                for k in all_activations:
                    all_activations[k].append(activations[k].cpu().numpy())
            elif hasattr(analyst, 'get_probabilities'):
                result = analyst.get_probabilities(batch_15m, batch_1h, batch_4h)
                if len(result) == 3:
                    context, probs, _ = result
                else:
                    context, probs = result
            else:
                context = analyst.get_context(batch_15m, batch_1h, batch_4h)
                # Dummy probs
                probs = torch.ones(len(context), 3, device=device) / 3
            
            all_contexts.append(context.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Memory cleanup
            del batch_15m, batch_1h, batch_4h, context, probs
            if batch_idx % 50 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
    
    # Combine all batches
    contexts = np.vstack(all_contexts).astype(np.float32)
    probs = np.vstack(all_probs).astype(np.float32)
    
    # Combine activations
    activations_15m = np.vstack(all_activations['15m']).astype(np.float32) if all_activations['15m'] else None
    activations_1h = np.vstack(all_activations['1h']).astype(np.float32) if all_activations['1h'] else None
    activations_4h = np.vstack(all_activations['4h']).astype(np.float32) if all_activations['4h'] else None
    
    logger.info(f"Final shapes: contexts={contexts.shape}, probs={probs.shape}")
    
    # Create output dict
    output = {
        'contexts': contexts,
        'probs': probs,
        'close_prices': close_prices[start_idx:],
        'start_idx': start_idx,
        'lookback_15m': lookback_15m,
        'lookback_1h': lookback_1h,
        'lookback_4h': lookback_4h,
        'n_samples': n_samples,
        'feature_cols': feature_cols,
    }
    
    # Save to disk
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            contexts=contexts,
            probs=probs,
            activations_15m=activations_15m,
            activations_1h=activations_1h,
            activations_4h=activations_4h,
            close_prices=close_prices[start_idx:],
            start_idx=start_idx,
            lookback_15m=lookback_15m,
            lookback_1h=lookback_1h,
            lookback_4h=lookback_4h,
        )
        logger.info(f"Saved cached Analyst outputs to: {save_path}")
    
    return output


def load_cached_analyst_outputs(cache_path: str) -> dict:
    """
    Load pre-computed Analyst outputs from disk.
    
    Args:
        cache_path: Path to the .npz file
        
    Returns:
        Dict with 'contexts', 'probs', and metadata
    """
    data = np.load(cache_path, allow_pickle=True)
    return {
        'contexts': data['contexts'],
        'probs': data['probs'],
        'activations_15m': data.get('activations_15m'),
        'activations_1h': data.get('activations_1h'),
        'activations_4h': data.get('activations_4h'),
        'close_prices': data['close_prices'],
        'start_idx': int(data['start_idx']),
        'lookback_15m': int(data['lookback_15m']),
        'lookback_1h': int(data['lookback_1h']),
        'lookback_4h': int(data['lookback_4h']),
    }


def main():
    """Main entry point for pre-computation script."""
    parser = argparse.ArgumentParser(description='Pre-compute Analyst outputs for PPO training')
    parser.add_argument('--analyst-path', type=str, default=None,
                       help='Path to trained Analyst model (default: models/analyst/best.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for cached data (default: data/processed/analyst_cache.npz)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference')
    args = parser.parse_args()
    
    config = Config()
    device = get_device()
    
    # Set default paths
    analyst_path = args.analyst_path or str(config.paths.models_analyst / 'best.pt')
    output_path = args.output or str(config.paths.data_processed / 'analyst_cache.npz')
    
    # Check if analyst model exists
    if not Path(analyst_path).exists():
        logger.error(f"Analyst model not found: {analyst_path}")
        logger.info("Please train the Analyst first, or specify --analyst-path")
        sys.exit(1)
    
    # Load normalized data
    logger.info("Loading normalized data...")
    try:
        df_15m = pd.read_parquet(config.paths.data_processed / 'features_15m_normalized.parquet')
        df_1h = pd.read_parquet(config.paths.data_processed / 'features_1h_normalized.parquet')
        df_4h = pd.read_parquet(config.paths.data_processed / 'features_4h_normalized.parquet')
    except FileNotFoundError:
        logger.error("Normalized data not found. Run the pipeline first to generate it.")
        sys.exit(1)
    
    # Feature columns - MUST match what Analyst was trained with (18 features)
    # Base 11 features + 3 market sessions + 4 structure breaks = 18
    feature_cols = [
        # Base features (11)
        'returns', 'volatility',
        'pinbar', 'engulfing', 'doji',
        'ema_trend', 'ema_crossover',
        'regime', 'sma_distance',
        'dist_to_resistance', 'dist_to_support',
        # Market sessions (3)
        'session_asian', 'session_london', 'session_ny',
        # Structure breaks (4)
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    ]
    
    # Check which features are available and add missing ones
    from src.data.features import add_market_sessions, detect_fractals, detect_structure_breaks
    
    # Add market sessions if missing
    if 'session_asian' not in df_15m.columns:
        logger.info("Adding market session features...")
        df_15m = add_market_sessions(df_15m)
        df_1h = add_market_sessions(df_1h)
        df_4h = add_market_sessions(df_4h)
    
    # Add structure breaks if missing
    if 'bos_bullish' not in df_15m.columns:
        logger.info("Adding structure break features...")
        df_15m = detect_fractals(df_15m)
        df_1h = detect_fractals(df_1h)
        df_4h = detect_fractals(df_4h)
        df_15m = detect_structure_breaks(df_15m)
        df_1h = detect_structure_breaks(df_1h)
        df_4h = detect_structure_breaks(df_4h)
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df_15m.columns]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Pre-compute
    output = precompute_analyst_outputs(
        analyst_path=analyst_path,
        df_15m=df_15m,
        df_1h=df_1h,
        df_4h=df_4h,
        feature_cols=feature_cols,
        device=device,
        batch_size=args.batch_size,
        save_path=output_path,
    )
    
    logger.info("=" * 60)
    logger.info("PRE-COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Cached {output['n_samples']:,} timesteps")
    logger.info(f"Context shape: {output['contexts'].shape}")
    logger.info(f"Probs shape: {output['probs'].shape}")
    logger.info(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
