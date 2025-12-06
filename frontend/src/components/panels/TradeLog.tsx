'use client';

import { Card } from '@/components/ui/Card';
import { useTradeHistory } from '@/hooks/useTrainingData';
import { motion, AnimatePresence } from 'framer-motion';
import { formatPips, getActionLabel } from '@/lib/utils';

export function TradeLog() {
  const trades = useTradeHistory();

  // Show most recent trades first
  const recentTrades = [...trades].reverse().slice(0, 10);

  // Calculate stats
  const wins = trades.filter((t) => !t.is_entry && (t.pnl ?? 0) > 0).length;
  const losses = trades.filter((t) => !t.is_entry && (t.pnl ?? 0) < 0).length;
  const totalTrades = wins + losses;
  const winRate = totalTrades > 0 ? (wins / totalTrades) * 100 : 0;

  const avgWin = trades
    .filter((t) => !t.is_entry && (t.pnl ?? 0) > 0)
    .reduce((sum, t) => sum + (t.pnl ?? 0), 0) / (wins || 1);

  const avgLoss = trades
    .filter((t) => !t.is_entry && (t.pnl ?? 0) < 0)
    .reduce((sum, t) => sum + (t.pnl ?? 0), 0) / (losses || 1);

  return (
    <Card title="Trade Log">
      <div className="space-y-3">
        {/* Stats row */}
        <div className="grid grid-cols-4 gap-2 pb-3 border-b border-slate-700/50 text-center">
          <div>
            <div className="text-xs text-slate-400">Trades</div>
            <div className="text-sm font-mono text-white">{totalTrades}</div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Win Rate</div>
            <div className="text-sm font-mono text-blue-400">{winRate.toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Avg Win</div>
            <div className="text-sm font-mono text-green-400">
              {formatPips(avgWin)}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Avg Loss</div>
            <div className="text-sm font-mono text-red-400">
              {formatPips(avgLoss)}
            </div>
          </div>
        </div>

        {/* Trade list */}
        <div className="space-y-1 max-h-48 overflow-y-auto">
          <AnimatePresence mode="popLayout">
            {recentTrades.length > 0 ? (
              recentTrades.map((trade, i) => (
                <motion.div
                  key={`${trade.timestamp}-${i}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center justify-between py-1 px-2 rounded bg-slate-800/50 text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-12 font-medium ${
                        trade.direction === 1
                          ? 'text-green-400'
                          : trade.direction === -1
                          ? 'text-red-400'
                          : 'text-slate-400'
                      }`}
                    >
                      {trade.direction === 1 ? 'LONG' : trade.direction === -1 ? 'SHORT' : 'FLAT'}
                    </span>
                    <span className="text-slate-400">
                      {trade.is_entry ? 'ENTRY' : trade.close_reason?.toUpperCase() ?? 'EXIT'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-300 font-mono">
                      {trade.price.toFixed(5)}
                    </span>
                    {!trade.is_entry && trade.pnl !== undefined && (
                      <span
                        className={`font-mono font-medium ${
                          trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}
                      >
                        {formatPips(trade.pnl)}
                      </span>
                    )}
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center text-slate-500 py-4">No trades yet</div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </Card>
  );
}
