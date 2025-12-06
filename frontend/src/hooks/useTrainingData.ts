'use client';

import { create } from 'zustand';
import type {
  TrainingStore,
  TrainingSnapshot,
  AnalystState,
  AgentState,
  MarketState,
  RewardComponents,
  SystemStatus,
  OHLCBar,
  TradeMarker,
} from '@/types/training';

// Default states
const defaultAnalyst: AnalystState = {
  epoch: 0,
  train_loss: 0,
  val_loss: 0,
  train_acc: 0,
  val_acc: 0,
  direction_acc: 0,
  grad_norm: 0,
  learning_rate: 0,
  attention_weights: [0.5, 0.5],
  p_down: 0.5,
  p_up: 0.5,
  confidence: 0.5,
  edge: 0,
  uncertainty: 0.5,
  encoder_15m_norm: 0,
  encoder_1h_norm: 0,
  encoder_4h_norm: 0,
  context_vector_sample: [],
};

const defaultAgent: AgentState = {
  timestep: 0,
  episode: 0,
  episode_reward: 0,
  episode_pnl: 0,
  episode_trades: 0,
  win_rate: 0,
  action_probs: [0.33, 0.33, 0.34],
  size_probs: [0.25, 0.25, 0.25, 0.25],
  value_estimate: 0,
  advantage: 0,
  entropy: 0,
  last_action_direction: 0,
  last_action_size: 0,
};

const defaultMarket: MarketState = {
  price: 0,
  current_price: 0,
  position: 0,
  position_size: 0,
  entry_price: 0,
  unrealized_pnl: 0,
  total_pnl: 0,
  n_trades: 0,
  atr: 0,
  chop: 50,
  adx: 25,
  regime: 1,
  sma_distance: 0,
  sl_level: 0,
  tp_level: 0,
};

const defaultReward: RewardComponents = {
  pnl_delta: 0,
  transaction_cost: 0,
  direction_bonus: 0,
  confidence_bonus: 0,
  fomo_penalty: 0,
  chop_penalty: 0,
  total: 0,
};

const defaultSystem: SystemStatus = {
  phase: 'idle',
  memory_used_mb: 0,
  memory_total_mb: 8192,
  steps_per_second: 0,
  episodes_per_hour: 0,
  elapsed_seconds: 0,
  device: 'mps',
};

// Create the store
export const useTrainingData = create<TrainingStore>((set, get) => ({
  // Connection state
  connected: false,
  lastUpdate: 0,

  // Current state
  analyst: null,
  agent: null,
  market: null,
  reward: null,
  system: null,

  // History (limited size)
  priceHistory: [],
  tradeHistory: [],
  lossHistory: [],
  rewardHistory: [],
  pnlHistory: [],

  // Actions
  setConnected: (connected: boolean) => set({ connected }),

  updateFromSnapshot: (snapshot: TrainingSnapshot) => {
    const state = get();
    const updates: Partial<TrainingStore> = {
      lastUpdate: snapshot.timestamp,
    };

    // Update component states
    if (snapshot.analyst) {
      updates.analyst = { ...defaultAnalyst, ...state.analyst, ...snapshot.analyst };
    }
    if (snapshot.agent) {
      updates.agent = { ...defaultAgent, ...state.agent, ...snapshot.agent };
    }
    if (snapshot.market) {
      updates.market = { ...defaultMarket, ...state.market, ...snapshot.market };
    }
    if (snapshot.reward) {
      updates.reward = { ...defaultReward, ...state.reward, ...snapshot.reward };
    }
    if (snapshot.system) {
      updates.system = { ...defaultSystem, ...state.system, ...snapshot.system };
    }

    // Handle history updates (for full state messages)
    // Check if we should clear the chart (new episode)
    if (snapshot.clear_chart) {
      // Start fresh but still allow new data from this snapshot
      updates.priceHistory = [];
      updates.tradeHistory = [];
    }

    // Process new price data (always, even after clear)
    if (snapshot.price_history) {
      const base = snapshot.clear_chart ? [] : state.priceHistory;
      updates.priceHistory = [...base, ...snapshot.price_history].slice(-500);
    } else if (snapshot.market?.ohlc) {
      // Append new bar
      const base = updates.priceHistory ?? state.priceHistory;
      updates.priceHistory = [...base, snapshot.market.ohlc].slice(-500);
    }

    if (snapshot.trade_history) {
      updates.tradeHistory = snapshot.trade_history.slice(-100);
    } else if (snapshot.trade) {
      // Append new trade
      const newHistory = [...state.tradeHistory, snapshot.trade].slice(-100);
      updates.tradeHistory = newHistory;
    }

    if (snapshot.loss_history) {
      updates.lossHistory = snapshot.loss_history.slice(-1000);
    }

    if (snapshot.reward_history) {
      updates.rewardHistory = snapshot.reward_history.slice(-10000);
    }

    // Append to reward history if we have a reward
    if (snapshot.reward?.total !== undefined && !snapshot.reward_history) {
      const newHistory = [...state.rewardHistory, snapshot.reward.total].slice(-10000);
      updates.rewardHistory = newHistory;
    }

    // Append to PnL history
    if (snapshot.market?.total_pnl !== undefined) {
      const newHistory = [...state.pnlHistory, snapshot.market.total_pnl].slice(-10000);
      updates.pnlHistory = newHistory;
    }

    set(updates);
  },

  reset: () =>
    set({
      connected: false,
      lastUpdate: 0,
      analyst: null,
      agent: null,
      market: null,
      reward: null,
      system: null,
      priceHistory: [],
      tradeHistory: [],
      lossHistory: [],
      rewardHistory: [],
      pnlHistory: [],
    }),
}));

// Selector hooks for performance
export const useConnected = () => useTrainingData((s) => s.connected);
export const useAnalyst = () => useTrainingData((s) => s.analyst);
export const useAgent = () => useTrainingData((s) => s.agent);
export const useMarket = () => useTrainingData((s) => s.market);
export const useReward = () => useTrainingData((s) => s.reward);
export const useSystem = () => useTrainingData((s) => s.system);
export const usePriceHistory = () => useTrainingData((s) => s.priceHistory);
export const useTradeHistory = () => useTrainingData((s) => s.tradeHistory);
export const useLossHistory = () => useTrainingData((s) => s.lossHistory);
export const useRewardHistory = () => useTrainingData((s) => s.rewardHistory);
export const usePnlHistory = () => useTrainingData((s) => s.pnlHistory);

// Alias exports for component compatibility
export const useTrades = useTradeHistory;
export const useEquityCurve = () => {
  const pnlHistory = useTrainingData((s) => s.pnlHistory);
  // Convert PnL history to equity curve format
  return pnlHistory.map((pnl, idx) => ({
    step: idx,
    equity: 100 + pnl, // Start with 100, add PnL
    drawdown: 0, // Would need to calculate properly
  }));
};
