'use client';

import { Card } from '@/components/ui/Card';
import { useAgent } from '@/hooks/useTrainingData';
import { motion } from 'framer-motion';
import { cn, getActionLabel, getSizeLabel, CHART_COLORS } from '@/lib/utils';

export function ActionDistribution() {
  const agent = useAgent();

  const actionProbs = agent?.action_probs ?? [0.33, 0.33, 0.34];
  const sizeProbs = agent?.size_probs ?? [0.25, 0.25, 0.25, 0.25];
  const lastAction = agent?.last_action_direction ?? 0;
  const lastSize = agent?.last_action_size ?? 0;

  const actionColors = [CHART_COLORS.flat, CHART_COLORS.long, CHART_COLORS.short];
  const sizeColors = ['#475569', '#64748b', '#94a3b8', '#cbd5e1'];

  return (
    <Card title="Action Distribution">
      <div className="space-y-4">
        {/* Direction Distribution */}
        <div>
          <div className="text-xs text-slate-400 mb-2">Direction</div>
          <div className="flex gap-2 h-8">
            {actionProbs.map((prob, i) => (
              <motion.div
                key={i}
                className={cn(
                  'flex-1 rounded flex items-center justify-center text-xs font-medium',
                  lastAction === i && 'ring-2 ring-white/50'
                )}
                style={{ backgroundColor: actionColors[i] }}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ delay: i * 0.1 }}
              >
                <span className="text-white/90">
                  {getActionLabel(i as 0 | 1 | 2)} {(prob * 100).toFixed(0)}%
                </span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Size Distribution */}
        <div>
          <div className="text-xs text-slate-400 mb-2">Size</div>
          <div className="flex gap-2 h-6">
            {sizeProbs.map((prob, i) => (
              <motion.div
                key={i}
                className={cn(
                  'flex-1 rounded flex items-center justify-center text-xs',
                  lastSize === i && 'ring-2 ring-white/50'
                )}
                style={{ backgroundColor: sizeColors[i] }}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ delay: i * 0.1 }}
              >
                <span className="text-slate-900 font-medium">
                  {getSizeLabel(i as 0 | 1 | 2 | 3)}
                </span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Probability bars */}
        <div className="space-y-2">
          {actionProbs.map((prob, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-xs text-slate-400 w-12">
                {getActionLabel(i as 0 | 1 | 2)}
              </span>
              <div className="flex-1 h-3 bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: actionColors[i] }}
                  initial={{ width: 0 }}
                  animate={{ width: `${prob * 100}%` }}
                  transition={{ type: 'spring', stiffness: 100 }}
                />
              </div>
              <span className="text-xs font-mono text-white w-12 text-right">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>

        {/* Entropy indicator */}
        <div className="pt-2 border-t border-slate-700/50 flex justify-between items-center">
          <span className="text-xs text-slate-400">Policy Entropy</span>
          <span className="text-sm font-mono text-blue-400">
            {agent?.entropy?.toFixed(3) ?? '0.000'}
          </span>
        </div>
      </div>
    </Card>
  );
}
