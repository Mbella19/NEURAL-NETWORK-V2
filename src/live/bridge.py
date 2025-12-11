
import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

from config.settings import config, Config
from src.agents.sniper_agent import SniperAgent
from src.models.analyst import load_analyst
from src.data.features import engineer_all_features, add_market_sessions
from src.data.normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)

class LiveSession:
    """
    Manages the live trading session state and inference pipeline.
    
    Acts as the 'Brain' counterpart to the MT5 'Body'.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.logger = logger
        
        # Models
        self.analyst = None
        self.agent = None
        
        # Normalizers
        self.normalizers: Dict[str, FeatureNormalizer] = {}
        
        # Buffer for latest history (to avoid re-sending all history)
        # Note: For MVP, we might accept full history windows from MT5 to ensure sync
        
        self.feature_cols = [] 
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Load models and normalizers."""
        self.logger.info("Initializing Live Session Pipeline...")
        
        # 1. Load Analyst
        try:
            analyst_path = self.config.paths.models_analyst / 'best.pt'
            # We need to know feature dims to load analyst. Assumed from config or inspection.
            # Ideally load metadata. For now, we assume standard features.
            # We'll set feature_dims dynamically if possible, or use standard defaults.
            # Using standard defaults from TrainingEnv/Pipeline logic
            # base features = 18. normalized input features ~ 15-18.
            # We'll load the keys from a saved normalizer to know the feature columns.
            
            # Load 15m normalizer first to get feature columns
            norm_path_15m = self.config.paths.models_analyst / 'normalizer_15m.pkl'
            if norm_path_15m.exists():
                norm_15m = FeatureNormalizer.load(norm_path_15m)
                self.normalizers['15m'] = norm_15m
                self.feature_cols = norm_15m.feature_cols
                
                # FIX: TrainAnalyst adds session columns automatically, so we must add them here too
                # otherwise input dim is 15 but model expects 18 (15 + 3 session features)
                session_cols = ['session_asian', 'session_london', 'session_ny']
                for col in session_cols:
                    if col not in self.feature_cols:
                        self.feature_cols.append(col)
                        
                self.logger.info(f"Loaded feature columns from 15m normalizer: {len(self.feature_cols)} (including sessions)")
            else:
                raise FileNotFoundError(f"Normalizer not found at {norm_path_15m}")
                
            # Load other normalizers
            for tf in ['1h', '4h']:
                p = self.config.paths.models_analyst / f'normalizer_{tf}.pkl'
                if p.exists():
                    self.normalizers[tf] = FeatureNormalizer.load(p)
                else:
                    self.logger.warning(f"Normalizer for {tf} not found, using 15m fallback (NOT RECOMMENDED)")
                    self.normalizers[tf] = self.normalizers['15m']

            # Load Analyst
            feature_dims = {
                '15m': len(self.feature_cols),
                '1h': len(self.feature_cols),
                '4h': len(self.feature_cols)
            }
            self.analyst = load_analyst(str(analyst_path), feature_dims, self.device, freeze=True)
            self.analyst.eval()
            self.logger.info("Analyst model loaded and frozen.")

            # 2. Load Agent
            # Priority: User specified checkpoint
            agent_path = self.config.paths.base_dir / "models/checkpoints/sniper_model_4400000_steps.zip"
            
            if not agent_path.exists():
                self.logger.warning(f"Requested checkpoint not found: {agent_path}. Falling back to automatic search.")
                agent_path = self.config.paths.models_agent / 'final_model.zip'
                if not agent_path.exists():
                     agent_path = next(self.config.paths.models_agent.glob("*.zip"), None)
                
                # Fallback 2: Latest .zip in models/checkpoints
                if not agent_path or not agent_path.exists():
                    checkpoints_dir = self.config.paths.base_dir / "models" / "checkpoints"
                    if checkpoints_dir.exists():
                        checkpoints = list(checkpoints_dir.glob("*.zip"))
                        if checkpoints:
                            agent_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                            self.logger.info(f"Using latest checkpoint: {agent_path.name}")

            if agent_path and agent_path.exists():
                # We need a dummy env to load the agent? 
                # SB3 PPO.load requires data about env mainly for action space validation.
                # However, for inference `predict`, we just need the object.
                # Passing `None` or a custom dummy might work, or rely on .load defaults
                # Warning: loading without env might warn, but works for predict(obs)
                self.agent = SniperAgent.load(agent_path, env=None, device='cpu') 
                self.logger.info(f"Agent loaded from {agent_path}")
            else:
                raise FileNotFoundError(f"Agent model not found at {self.config.paths.models_agent}")

        except Exception as e:
            self.logger.critical(f"Failed to initialize pipeline: {e}")
            raise e

    def on_tick(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a tick update from MT5.
        
        Payload structure:
        {
            "rates": {
                "15m": [[time, open, high, low, close, vol], ...],
                "1h": [...],
                "4h": [...]
            },
            "position": {
                "type": int (0=Long, 1=Short, -1=None),
                "volume": float,
                "price": float,
                "sl": float,
                "tp": float,
                "pnl": float
            },
            "account": {
                "balance": float,
                "equity": float
            }
        }
        """
        try:
            # 1. Parse Data
            dfs = self._parse_rates(payload['rates'])
            
            # 2. Engineer Features
            dfs_feat = self._engineer_features(dfs)
            
            # Safety: Handle NaNs if any remain (though logic should prevent them with enough history)
            for tf, df_f in dfs_feat.items():
                if df_f.isnull().values.any():
                    self.logger.warning(f"{tf}: Found NaNs after feature engineering. Filling...")
                    dfs_feat[tf] = df_f.ffill().bfill().fillna(0.0)
            
            # 3. Normalize
            dfs_norm = self._normalize_features(dfs_feat)
            
            # 4. Analyst Prediction
            context, metrics = self._get_analyst_prediction(dfs_norm)
            
            # 5. Construct Observation
            obs = self._construct_observation(
                context, 
                metrics, 
                dfs_feat['15m'], # Need unnormalized data for some things? No, using stored stats
                dfs_norm['15m'],
                payload['position']
            )
            
            # 6. Agent Prediction
            action, _ = self.agent.predict(
                obs, 
                deterministic=True, 
                min_action_confidence=self.config.trading.min_action_confidence
            )
            
            # 7. Decode Action
            # Action is [direction, size]
            # Direction: 0=Flat, 1=Long, 2=Short
            # Size: 0=0.25, 1=0.5, 2=0.75, 3=1.0
            direction = int(action[0])
            size_idx = int(action[1])
            size_map = [0.25, 0.5, 0.75, 1.0]
            size_pct = size_map[size_idx] if size_idx < 4 else 0.25
            
            # Calculate SL/TP based on ATR
            current_price = float(dfs['15m']['close'].iloc[-1])
            atr = float(dfs_feat['15m']['atr'].iloc[-1])
            
            sl_price = 0.0
            tp_price = 0.0
            sl_pips = 0.0
            
            if direction == 1: # Long
                sl_dist = atr * self.config.trading.sl_atr_multiplier
                tp_dist = atr * self.config.trading.tp_atr_multiplier
                sl_price = current_price - sl_dist
                tp_price = current_price + tp_dist
                sl_pips = sl_dist / 0.0001  # Convert to pips
            elif direction == 2: # Short
                sl_dist = atr * self.config.trading.sl_atr_multiplier
                tp_dist = atr * self.config.trading.tp_atr_multiplier
                sl_price = current_price + sl_dist
                tp_price = current_price - tp_dist
                sl_pips = sl_dist / 0.0001  # Convert to pips
            
            # --- Dynamic Risk-Based Position Sizing ---
            # Calculate lot size based on account equity and risk per trade
            account_info = payload.get('account', {})
            account_equity = float(account_info.get('equity', 10000.0))  # Default to 10k if missing
            if account_equity <= 0:
                account_equity = 10000.0  # Safety fallback
            
            risk_amount = account_equity * self.config.live.risk_percent  # e.g., 1% of equity
            pip_value = self.config.live.pip_value_per_lot  # e.g., $10 per pip per lot
            
            # Base position size = Risk Amount / (SL Pips * Pip Value)
            if sl_pips > 0 and pip_value > 0:
                base_lot_size = risk_amount / (sl_pips * pip_value)
            else:
                base_lot_size = 0.01  # Minimum if SL is unknown
            
            # Apply Agent's confidence scaling (0.25 - 1.0)
            final_size = base_lot_size * size_pct
            
            # Clamp to broker limits
            final_size = max(self.config.live.min_lot_size, min(final_size, self.config.live.max_lot_size))
            final_size = round(final_size, 2)  # Round to 2 decimal places

            self.logger.info(f"Signal: Dir={direction} Size={final_size:.2f} lots (Equity=${account_equity:.0f}, Risk={self.config.live.risk_percent*100:.1f}%, SL={sl_pips:.1f} pips, Conf={metrics['confidence']:.2f}) SL={sl_price:.5f} TP={tp_price:.5f}")
            
            return {
                "action": direction, # 0=Flat, 1=Long, 2=Short
                "size": final_size,
                "sl": sl_price,
                "tp": tp_price,
                "confidence": float(metrics['confidence']),
                "agent_probs": metrics['probs'].tolist() if isinstance(metrics['probs'], np.ndarray) else metrics['probs']
            }


        except Exception as e:
            self.logger.error(f"Error in on_tick: {e}", exc_info=True)
            return {"action": 0, "size": 0.0, "error": str(e)}

    def _parse_rates(self, rates_data: Dict[str, List]) -> Dict[str, pd.DataFrame]:
        dfs = {}
        cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        
        for tf, data in rates_data.items():
            if not data:
                raise ValueError(f"Empty data for {tf}")
            
            df = pd.DataFrame(data, columns=cols)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Ensure types
            for c in ['open', 'high', 'low', 'close', 'tick_volume']:
                df[c] = df[c].astype(np.float32)
                
            dfs[tf] = df
            
        return dfs

    def _engineer_features(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # Use existing engineer_all_features
        # We need to map config parameters. 
        # Using default config inside engineer_all_features if None, but better to pass matching training config
        
        # Mapping FeatureConfig to dict format for engineer_all_features
        fc = self.config.features
        feat_conf = {
            'pinbar_wick_ratio': fc.pinbar_wick_ratio,
            'doji_body_ratio': fc.doji_body_ratio,
            'fractal_window': fc.fractal_window,
            'sr_lookback': fc.sr_lookback,
            'sma_period': fc.sma_period,
            'ema_fast': fc.ema_fast,
            'ema_slow': fc.ema_slow,
            'chop_period': fc.chop_period,
            'adx_period': fc.adx_period,
            'atr_period': fc.atr_period
        }

        dfs_eng = {}
        for tf, df in dfs.items():
            # engineer_all_features adds columns in-place-ish (returns copy)
            # IMPORTANT: We need enough history for indicators (sma_200 etc).
            # MT5 must send at least 250 bars.
            df_eng = engineer_all_features(df, feat_conf)
            
            # Add Market Sessions (Critical: missing in normalizer but present in model)
            df_eng = add_market_sessions(df_eng)
            
            dfs_eng[tf] = df_eng
            
        return dfs_eng

    def _normalize_features(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        dfs_norm = {}
        for tf, df in dfs.items():
            norm = self.normalizers.get(tf)
            if norm:
                # Transform ONLY the columns in feature_cols
                # FeatureNormalizer.transform(df) does this automatically
                dfs_norm[tf] = norm.transform(df)
            else:
                 # Should not happen given init logic
                 pass
        return dfs_norm

    def _get_analyst_prediction(self, dfs_norm: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Prepare inputs
        # We take the LAST row (current candle).
        # TCN needs shape (1, channels, seq_len) -> converted inside model usually?
        # create_tcn_analyst model expects inputs: (batch, dim, seq) or similar.
        # Wait, train_analyst.py dataset returns: (x_15m, x_1h, x_4h)
        # TCNAnalyst.forward(x_15m, x_1h, x_4h)
        
        # We need to reshape single sample to batch of 1
        # Also need to convert DataFrame row to Tensor
        
        # Check input dims expected by model
        # src/models/tcn_analyst.py: forward(x_15m, x_1h, x_4h)
        # x_15m shape: (batch_size, input_dim, seq_len) (if transposed) or (batch, seq, dim)
        # TCN usually expects (N, C, L). Dataset returns (L, C) or (C, L)?
        # Looking at MultiTimeframeDataset.__getitem__: returns (x_15m, ...), tensor shape (lookback, features)
        # So (Seq, Dim).
        # Model forward usually permutes if it needs (N,C,L).
        
        # Let's verify TCNAnalyst input expectation.
        # Assuming standard (Batch, Seq, Dim) input from Dataset, 
        # TCN often permutes to (Batch, Dim, Seq).
        
        # We will provide (1, Seq_Len, Features).
        # Sequence Length is lookback window.
        # Configuration says lookback_15m=48.
        
        def get_seq(df, lookback):
            if len(df) < lookback:
                # Pad with ffill or 0
                self.logger.warning(f"Data length {len(df)} < lookback {lookback}. Padding.")
                # Simple padding for safety
                pad_len = lookback - len(df)
                padding = pd.DataFrame(np.zeros((pad_len, len(df.columns))), columns=df.columns)
                df = pd.concat([padding, df], ignore_index=True)
                
            return df.iloc[-lookback:].values.astype(np.float32)

        lb_15m = self.config.data.lookback_windows['15m']
        lb_1h = self.config.data.lookback_windows['1h']
        lb_4h = self.config.data.lookback_windows['4h']
        
        # Extract sequences (only feature columns)
        fcols = self.feature_cols
        
        x_15m = torch.tensor(get_seq(dfs_norm['15m'][fcols], lb_15m)).unsqueeze(0).to(self.device)
        x_1h = torch.tensor(get_seq(dfs_norm['1h'][fcols], lb_1h)).unsqueeze(0).to(self.device)
        x_4h = torch.tensor(get_seq(dfs_norm['4h'][fcols], lb_4h)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.analyst, 'get_probabilities'):
                # New interface
                context, probs = self.analyst.get_probabilities(x_15m, x_1h, x_4h)
                # context: (1, 32), probs: (1, 3)
            else:
                context = self.analyst.get_context(x_15m, x_1h, x_4h)
                probs = torch.tensor([[0.5, 0.5]]).to(self.device) # Fallback

        context_np = context.cpu().numpy().flatten()
        probs_np = probs.cpu().numpy().flatten()
        
        metrics = {
            'probs': probs_np,
            'confidence': float(np.max(probs_np)),
        }
        
        return context_np, metrics

    def _construct_observation(self, context, metrics, df_raw_15m, df_norm_15m, pos_info):
        # Replicate TradingEnv._get_observation logic
        
        # 1. Context (32)
        
        # 2. Position State (3)
        # position: -1, 0, 1
        mt5_type = pos_info.get('type') # 0=Buy, 1=Sell, -1=None in Python mapping?
        # MT5: 0=Buy, 1=Sell.
        # Agent: 1=Long (Buy), 2=Short (Sell)? No, Env says:
        # self.position: -1: Short, 0: Flat, 1: Long
        
        position = 0
        if mt5_type == 0: position = 1
        elif mt5_type == 1: position = -1
        
        entry_price = float(pos_info.get('price', 0.0))
        current_price = float(df_raw_15m['close'].iloc[-1])
        
        # Need ATR for normalization (from NORMALIZED features or RAW? Env uses NORMALIZED for features, but RAW for some calcs)
        # Env: atr = self.market_features[self.current_idx, 0].
        # market_features in Env are [atr, chop, adx, regime, sma_dist] (Z-normed).
        # Wait, Env uses raw ATR for pips calculation if available?
        # Env lines 538: atr_safe = max(atr, 1e-6). 
        # If market_features is Z-normalized, ATR can be negative!
        # Checking Env again: 
        #   market_features = market_features.astype(np.float32) (passed in __init__)
        #   step_6_backtest passes *normalized* market_features? 
        #   No, step_6_backtest passes `market_features` which are computed in `prepare_env_data`.
        #   And `step_3b_normalize_features` normalizes `all_feature_cols` including ATR?
        #   Wait, `step_3b` excludes `RAW_COLUMNS` from normalization!
        #   `RAW_COLUMNS` = ['atr', 'chop', 'adx', ...]
        #   So ATR in `market_features` passed to Env is RAW (Unnormalized).
        #   BUT Env._get_observation normalizes market features internally using `market_feat_mean`/`std`.
        
        # Therefore, for `entry_price_norm`, we need RAW ATR.
        
        raw_atr = float(df_raw_15m['atr'].iloc[-1])
        atr_safe = max(raw_atr, 1e-6)
        
        entry_price_norm = 0.0
        unrealized_pnl_norm = 0.0
        
        if position != 0:
            if position == 1:
                entry_price_norm = (current_price - entry_price) / (atr_safe * 100)
            else:
                 entry_price_norm = (entry_price - current_price) / (atr_safe * 100)
            
            entry_price_norm = np.clip(entry_price_norm, -10.0, 10.0)
            
            # Unrealized PnL (pips / 100)
            pnl_pips = float(pos_info.get('profit', 0.0)) # MT5 profit is in currency, not pips usually.
            # We should calc approx pips from price diff.
            # pos_info['pnl'] might be currency.
            # Let's calc from price.
            if position == 1:
                pnl_raw = (current_price - entry_price) / 0.0001
            else:
                pnl_raw = (entry_price - current_price) / 0.0001
            
            # Multiply by size? Env uses: pnl_pips * self.position_size.
            # MT5 volume is e.g. 0.1. Agent size is 0.25-1.0 (relative).
            # Live Agent logic: we want the Agent's *view* of success.
            # Pipelined PnL is (PriceDiff / Pip) * SizeMultiplier.
            # We don't know the exact size multiplier active in MT5 easily (0.01 lot?).
            # Let's assume size=1.0 for observation purpose or use volume relative to max.
            
            unrealized_pnl_norm = (pnl_raw * 1.0) / 100.0 # simplified

        position_state = np.array([float(position), entry_price_norm, unrealized_pnl_norm], dtype=np.float32)
        
        # 3. Market Features (Normalized)
        # [atr, chop, adx, regime, sma_distance]
        # We need to pull these from df_norm_15m.
        # But wait, 'atr', 'chop', 'adx' were NOT normalized in `df_norm_15m` in pipeline (they are RAW_COLUMNS).
        # BUT the Env expects them to be Z-normalized in the observation vector.
        # The Env does this Z-normalization itself! using `self.market_feat_mean`.
        # So I need to replicate that Z-normalization using the saved training stats.
        # Where are `market_feat_mean` saved?
        # Run Pipeline doesn't explicit save them separately?
        # It relies on `market_features` which are just columns.
        # `step_6_backtest` computes them on the fly from training data.
        
        # This is a missing piece in the artifact saving.
        # Workaround: Use the normalizers I have. 
        # Does `normalizer_15m` contain stats for 'atr'?
        # In `step_3b`, `normalize_cols` excluded RAW_COLUMNS.
        # So `normalizer_15m` does NOT have stats for ATR.
        
        # Hack: Compute rough 'mean/std' for EURUSD 15m ATR/CHOP/ADX or use defaults.
        # Or, just use the raw values if the Agent is robust (unlikely).
        # Better: Calculate rolling mean/std on the fly from the history buffer?
        # Yes, standardizing based on recent history (last 500-1000 bars) is a reasonable approximation for Z-score.
        # Or hardcode typical values for EURUSD 15m:
        # ATR ~ 0.0010 (10 pips)
        # CHOP ~ 50
        # ADX ~ 25
        
        # I'll implement dynamic Z-score based on the history buffer I just received.
        hist_15m = df_raw_15m
        
        # CRITICAL FIX: Use GLOBAL training stats for normalization, NOT dynamic stats from short history.
        # Computed from full training dataset (Nov 2020 - Feb 2025)
        MARKET_STATS = {
            'atr': {'mean': 0.000716, 'std': 0.000411},
            'chop': {'mean': 49.811939, 'std': 9.742417},
            'adx': {'mean': 24.800243, 'std': 10.306635},
            'regime': {'mean': 0.396563, 'std': 0.590921},
            'dist_to_support': {'mean': 1.061543, 'std': 1.339726},
            'dist_to_resistance': {'mean': 1.035922, 'std': 1.284660}
        }
        
        def z_score(col, val):
            # Use saved global stats if available, else fallback (though we have them for all key feats)
            stats = MARKET_STATS.get(col)
            if stats:
                return (val - stats['mean']) / stats['std']
            else:
                # Fallback to dynamic if somehow missing (should not happen for core feats)
                mean = hist_15m[col].mean()
                std = hist_15m[col].std() + 1e-6
                return (val - mean) / std

        mf_raw = [
            float(df_raw_15m['atr'].iloc[-1]),
            float(df_raw_15m['chop'].iloc[-1]),
            float(df_raw_15m['adx'].iloc[-1]),
            float(df_raw_15m['regime'].iloc[-1]),
            float(df_norm_15m['sma_distance'].iloc[-1]), # Already normalized by Analyst normalizer
            float(df_raw_15m.get('dist_to_support', pd.Series([0.0])).iloc[-1]),
            float(df_raw_15m.get('dist_to_resistance', pd.Series([0.0])).iloc[-1])
        ]
        
        # Check if sma_distance is in normalizer
        # normalize_cols = [..., 'sma_distance', ...] is in MODEL_INPUT_COLUMNS.
        # So it is already normalized in df_norm_15m.
        
        # Normalize agent-specific features using GLOBAL stats
        atr_n = z_score('atr', mf_raw[0])
        chop_n = z_score('chop', mf_raw[1])
        adx_n = z_score('adx', mf_raw[2])
        regime_n = z_score('regime', mf_raw[3])
        sma_n = mf_raw[4] # Keeping as-is (from analyst normalizer)
        
        # S/R Distances
        sr_sup_n = z_score('dist_to_support', mf_raw[5])
        sr_res_n = z_score('dist_to_resistance', mf_raw[6])

        market_feat = np.array([atr_n, chop_n, adx_n, regime_n, sma_n, sr_sup_n, sr_res_n], dtype=np.float32)
        
        # 4. Analyst Metrics
        probs = metrics['probs']
        # Binary: [p_d, p_u]
        if len(probs) == 2:
            p_down, p_up = probs
            conf = max(p_down, p_up)
            edge = p_up - p_down
            uncert = 1.0 - conf
            analyst_metrics = np.array([p_down, p_up, edge, conf, uncert], dtype=np.float32)
        else:
             # Handle 3-class or 5-class if needed
             analyst_metrics = np.zeros(5, dtype=np.float32)

        # 5. SL/TP Distances
        # Need current price and open SL/TP prices from MT5
        current_sl = float(pos_info.get('sl', 0.0))
        current_tp = float(pos_info.get('tp', 0.0))
        dist_sl_norm = 0.0
        dist_tp_norm = 0.0
        
        if position != 0:
            # If MT5 has actual SL/TP set
            if current_sl > 0:
                dist_sl_norm = (current_price - current_sl) / (atr_safe) if position == 1 else (current_sl - current_price) / (atr_safe)
            else:
                # Use expected SL from config
                dist_sl_norm = self.config.trading.sl_atr_multiplier
            
            if current_tp > 0:
                dist_tp_norm = (current_tp - current_price) / (atr_safe) if position == 1 else (current_price - current_tp) / (atr_safe)
            else:
                dist_tp_norm = self.config.trading.tp_atr_multiplier
                
        sl_tp_feat = np.array([dist_sl_norm, dist_tp_norm], dtype=np.float32)

        # Concatenate
        return np.concatenate([
            context,
            position_state,
            market_feat,
            analyst_metrics,
            sl_tp_feat
        ])
