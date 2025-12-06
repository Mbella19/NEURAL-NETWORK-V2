'use client';

import { Card } from '@/components/ui/Card';
import { useReward, useRewardHistory } from '@/hooks/useTrainingData';
import { motion } from 'framer-motion';
import { useMemo } from 'react';

export function RewardBreakdown() {
  const reward = useReward();
  const rewardHistory = useRewardHistory();

  const components = [
    { key: 'pnl_delta', label: 'PnL Delta', value: reward?.pnl_delta ?? 0, color: '#22c55e' },
    { key: 'transaction_cost', label: 'Transaction', value: -(reward?.transaction_cost ?? 0), color: '#ef4444' },
    { key: 'direction_bonus', label: 'Direction', value: reward?.direction_bonus ?? 0, color: '#3b82f6' },
    { key: 'confidence_bonus', label: 'Confidence', value: reward?.confidence_bonus ?? 0, color: '#8b5cf6' },
    { key: 'fomo_penalty', label: 'FOMO', value: reward?.fomo_penalty ?? 0, color: '#f59e0b' },
    { key: 'chop_penalty', label: 'Chop', value: reward?.chop_penalty ?? 0, color: '#eab308' },
  ];

  const total = reward?.total ?? 0;
  const maxAbs = Math.max(...components.map((c) => Math.abs(c.value)), 0.1);

  // Recent average reward
  const recentAvg = useMemo(() => {
    if (rewardHistory.length === 0) return 0;
    const recent = rewardHistory.slice(-100);
    return recent.reduce((a, b) => a + b, 0) / recent.length;
  }, [rewardHistory]);

  return (
    <Card title="Reward Breakdown">
      <div className="space-y-3">
        {/* Waterfall bars */}
        {components.map((comp) => {
          const width = Math.abs(comp.value) / maxAbs * 50;
          const isPositive = comp.value >= 0;

          return (
            <div key={comp.key} className="flex items-center gap-2">
              <span className="text-xs text-slate-400 w-20 truncate">{comp.label}</span>
              <div className="flex-1 h-4 flex items-center">
                <div className="w-1/2 flex justify-end">
                  {!isPositive && (
                    <motion.div
                      className="h-full rounded-l"
                      style={{ backgroundColor: comp.color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${width}%` }}
                      transition={{ type: 'spring', stiffness: 100 }}
                    />
                  )}
                </div>
                <div className="w-px h-full bg-slate-600" />
                <div className="w-1/2">
                  {isPositive && comp.value > 0 && (
                    <motion.div
                      className="h-full rounded-r"
                      style={{ backgroundColor: comp.color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${width}%` }}
                      transition={{ type: 'spring', stiffness: 100 }}
                    />
                  )}
                </div>
              </div>
              <span
                className={`text-xs font-mono w-16 text-right ${
                  comp.value >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {comp.value >= 0 ? '+' : ''}{comp.value.toFixed(3)}
              </span>
            </div>
          );
        })}

        {/* Total */}
        <div className="pt-3 border-t border-slate-700/50 flex items-center justify-between">
          <span className="text-sm font-medium text-slate-300">Total Reward</span>
          <motion.span
            key={total}
            initial={{ scale: 1.1 }}
            animate={{ scale: 1 }}
            className={`text-lg font-mono font-bold ${
              total >= 0 ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {total >= 0 ? '+' : ''}{total.toFixed(3)}
          </motion.span>
        </div>

        {/* Recent average */}
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-400">Avg (100 steps)</span>
          <span
            className={`font-mono ${
              recentAvg >= 0 ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {recentAvg >= 0 ? '+' : ''}{recentAvg.toFixed(4)}
          </span>
        </div>
      </div>
    </Card>
  );
}
