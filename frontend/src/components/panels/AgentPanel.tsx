'use client';

import { Card } from '@/components/ui/Card';
import { PositionBadge } from '@/components/ui/Badge';
import { useAgent, useMarket } from '@/hooks/useTrainingData';
import { motion } from 'framer-motion';
import { formatPips, formatPercent, formatLargeNumber } from '@/lib/utils';

export function AgentPanel() {
  const agent = useAgent();
  const market = useMarket();

  const position = market?.position ?? 0;
  const positionSize = market?.position_size ?? 0;
  const entryPrice = market?.entry_price;
  const currentPrice = market?.current_price ?? 0;
  const unrealizedPnl = market?.unrealized_pnl ?? 0;
  const totalPnl = market?.total_pnl ?? 0;
  const valueEstimate = agent?.value_estimate ?? 0;
  const winRate = agent?.win_rate ?? 0;

  return (
    <Card title="Agent Decision">
      <div className="space-y-4">
        {/* Current Position */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Position</span>
          <PositionBadge position={position as -1 | 0 | 1} size={positionSize} />
        </div>

        {/* Entry Price (if in position) */}
        {position !== 0 && entryPrice && (
          <div className="flex items-center justify-between">
            <span className="text-slate-400 text-sm">Entry</span>
            <span className="font-mono text-white">{entryPrice.toFixed(5)}</span>
          </div>
        )}

        {/* Current Price */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Current</span>
          <motion.span
            key={currentPrice}
            initial={{ scale: 1.05 }}
            animate={{ scale: 1 }}
            className="font-mono text-white"
          >
            {currentPrice.toFixed(5)}
          </motion.span>
        </div>

        {/* Unrealized PnL (if in position) */}
        {position !== 0 && (
          <div className="flex items-center justify-between">
            <span className="text-slate-400 text-sm">Unrealized</span>
            <motion.span
              key={unrealizedPnl}
              initial={{ scale: 1.1 }}
              animate={{ scale: 1 }}
              className={`font-mono font-bold ${
                unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'
              }`}
            >
              {formatPips(unrealizedPnl)} pips
            </motion.span>
          </div>
        )}

        {/* Divider */}
        <div className="border-t border-slate-700/50" />

        {/* Value Estimate */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Value Est.</span>
          <span
            className={`font-mono ${
              valueEstimate >= 0 ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {valueEstimate >= 0 ? '+' : ''}{valueEstimate.toFixed(3)}
          </span>
        </div>

        {/* Win Rate */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Win Rate</span>
          <span className="font-mono text-blue-400">{formatPercent(winRate)}</span>
        </div>

        {/* Training Progress */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Episode</span>
          <span className="font-mono text-white">{agent?.episode ?? 0}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Timestep</span>
          <span className="font-mono text-white">
            {formatLargeNumber(agent?.timestep ?? 0)}
          </span>
        </div>

        {/* Divider */}
        <div className="border-t border-slate-700/50" />

        {/* Total PnL */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Session PnL</span>
          <motion.span
            key={totalPnl}
            initial={{ scale: 1.1 }}
            animate={{ scale: 1 }}
            className={`text-xl font-mono font-bold ${
              totalPnl >= 0 ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {formatPips(totalPnl)}
          </motion.span>
        </div>
      </div>
    </Card>
  );
}
