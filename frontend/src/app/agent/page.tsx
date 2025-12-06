'use client';

import { useState, useEffect } from 'react';

import { Card } from '@/components/ui/Card';
import { useAgent, useReward } from '@/hooks/useTrainingData';
import { cn, formatPercent } from '@/lib/utils';
import { getPnlClass } from '@/lib/colors';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { colors } from '@/lib/colors';

export default function AgentPage() {
  const agent = useAgent();
  const reward = useReward();
  const [mounted, setMounted] = useState(false);
  const [rewardHistory, setRewardHistory] = useState<any[]>([]);
  const [metrics, setMetrics] = useState({ kl: 0, clip: 0 });

  useEffect(() => {
    setMounted(true);
    setRewardHistory(Array.from({ length: 100 }, (_, i) => ({
      episode: i + 1,
      reward: -20 + i * 0.5 + Math.random() * 30 - 15,
      avg: -20 + i * 0.6,
    })));
    setMetrics({
      kl: 0.015 + Math.random() * 0.01,
      clip: 0.05 + Math.random() * 0.05,
    });
  }, []);

  if (!mounted) return null;

  const actionProbs = agent?.action_probs ?? [0.33, 0.34, 0.33];
  const sizeProbs = agent?.size_probs ?? [0.25, 0.25, 0.25, 0.25];

  const getExplorationLevel = () => {
    const entropy = agent?.entropy ?? 0;
    if (entropy > 1.0) return 'HIGH';
    if (entropy > 0.5) return 'MODERATE';
    return 'LOW';
  };

  return (
    <div className="page-container p-4 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">PPO Sniper Agent</h1>
          <p className="text-xs text-zinc-500">Reinforcement learning phase - Policy optimization</p>
        </div>
        <div className="text-right">
          <div className="text-xs text-zinc-500">Timestep</div>
          <div className="text-xl font-mono text-zinc-100">
            {(agent?.timestep ?? 0).toLocaleString()}
          </div>
        </div>
      </div>

      {/* Main Content Grid - 2 rows */}
      <div className="flex-1 grid grid-rows-2 gap-3 min-h-0">
        {/* Top Row - Chart and Network */}
        <div className="grid grid-cols-1 gap-3 min-h-0">
          {/* Episode Rewards Chart */}
          <Card title="EPISODE REWARDS" className="flex flex-col min-h-0">
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={rewardHistory} margin={{ top: 5, right: 5, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorAvg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={colors.chart.line1} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={colors.chart.line1} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="episode" stroke={colors.text.muted} fontSize={9} tickLine={false} axisLine={false} minTickGap={40} />
                  <YAxis stroke={colors.text.muted} fontSize={9} tickLine={false} axisLine={false} />
                  <ReferenceLine y={0} stroke={colors.border.muted} strokeDasharray="3 3" />
                  <Area type="monotone" dataKey="reward" stroke={colors.text.muted} strokeWidth={1} fill="transparent" name="Raw" />
                  <Area type="monotone" dataKey="avg" stroke={colors.chart.line1} strokeWidth={2} fill="url(#colorAvg)" name="Avg" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="text-xs flex justify-between border-t border-zinc-800 pt-1 mt-1">
              <span className="text-zinc-400">Ep {agent?.episode ?? 0}</span>
              <span>
                Current: <span className={cn('font-mono', getPnlClass(agent?.episode_reward ?? 0))}>{(agent?.episode_reward ?? 0).toFixed(1)}</span>
                {' | '}
                Best: <span className="font-mono text-green-500">{(agent?.best_reward ?? 0).toFixed(1)}</span>
              </span>
            </div>
          </Card>
        </div>

        {/* Bottom Row - Distributions and Metrics */}
        <div className="grid grid-cols-3 gap-3 min-h-0">
          {/* Action Distribution */}
          <Card title="ACTION DISTRIBUTION" className="flex flex-col min-h-0">
            <div className="flex-1 space-y-2 overflow-auto">
              {['FLAT', 'LONG', 'SHORT'].map((action, idx) => (
                <div key={action}>
                  <div className="flex justify-between items-center mb-0.5">
                    <span className={cn('text-xs font-medium', idx === 0 ? 'text-zinc-400' : idx === 1 ? 'text-green-500' : 'text-red-600')}>
                      {action}
                    </span>
                    <span className="text-xs font-mono text-zinc-300">{(actionProbs[idx] * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-zinc-800 rounded overflow-hidden">
                    <div className={cn('h-full', idx === 0 ? 'bg-zinc-500' : idx === 1 ? 'bg-green-500' : 'bg-red-600')} style={{ width: `${actionProbs[idx] * 100}%` }} />
                  </div>
                </div>
              ))}
              <div className="pt-2 border-t border-zinc-800">
                <div className="text-[10px] text-zinc-500 mb-1">SIZE</div>
                {['0.25x', '0.50x', '0.75x', '1.00x'].map((size, idx) => (
                  <div key={size} className="flex items-center gap-1 mb-0.5">
                    <span className="text-[10px] text-zinc-400 w-8">{size}</span>
                    <div className="flex-1 h-1 bg-zinc-800 rounded overflow-hidden">
                      <div className="h-full bg-blue-500" style={{ width: `${sizeProbs[idx] * 100}%` }} />
                    </div>
                    <span className="text-[10px] font-mono text-zinc-400 w-6">{(sizeProbs[idx] * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          {/* Policy Metrics */}
          <Card title="POLICY METRICS" className="flex flex-col min-h-0">
            <div className="flex-1 space-y-2 overflow-auto">
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Value Est.</span>
                <span className="text-sm font-mono text-blue-500">{(agent?.value_estimate ?? 0).toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Entropy</span>
                <span className="text-sm font-mono text-zinc-200">{(agent?.entropy ?? 0).toFixed(3)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">KL Div</span>
                <span className="text-sm font-mono text-zinc-200">{metrics.kl.toFixed(4)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Clip Frac</span>
                <span className="text-sm font-mono text-zinc-200">{metrics.clip.toFixed(2)}</span>
              </div>
              <div className="pt-2 border-t border-zinc-800 flex justify-between items-center">
                <span className="text-xs text-zinc-400">Exploration</span>
                <span className={cn('text-xs font-medium', getExplorationLevel() === 'HIGH' ? 'text-green-500' : getExplorationLevel() === 'MODERATE' ? 'text-amber-500' : 'text-red-600')}>
                  {getExplorationLevel()}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">LR</span>
                <span className="text-xs font-mono text-zinc-300">{(agent?.learning_rate ?? 3e-4).toExponential(1)}</span>
              </div>
            </div>
          </Card>

          {/* Reward Breakdown */}
          <Card title="REWARD BREAKDOWN" className="flex flex-col min-h-0">
            <div className="flex-1 space-y-1.5 overflow-auto">
              {[
                { label: 'PnL Delta', value: reward?.pnl_delta ?? 0, color: 'bg-green-500' },
                { label: 'Conf', value: reward?.confidence_bonus ?? 0, color: 'bg-blue-500' },
                { label: 'Dir', value: reward?.direction_bonus ?? 0, color: 'bg-purple-500' },
                { label: 'Trans', value: -(reward?.transaction_cost ?? 0), color: 'bg-red-600' },
                { label: 'FOMO', value: -(reward?.fomo_penalty ?? 0), color: 'bg-amber-500' },
                { label: 'Chop', value: -(reward?.chop_penalty ?? 0), color: 'bg-amber-600' },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-1">
                  <span className="text-xs text-zinc-400 w-16">{item.label}</span>
                  <span className={cn('text-xs font-mono w-14', getPnlClass(item.value))}>
                    {item.value >= 0 ? '+' : ''}{item.value.toFixed(2)}
                  </span>
                  <div className="flex-1 h-1 bg-zinc-800 rounded overflow-hidden">
                    <div className={cn('h-full', item.color)} style={{ width: `${Math.min(100, Math.abs(item.value) * 50)}%`, opacity: item.value >= 0 ? 1 : 0.5 }} />
                  </div>
                </div>
              ))}
              <div className="pt-1 border-t border-zinc-800 flex justify-between items-center">
                <span className="text-xs font-medium text-zinc-300">TOTAL</span>
                <span className={cn('text-sm font-mono', getPnlClass(reward?.total ?? 0))}>
                  {(reward?.total ?? 0) >= 0 ? '+' : ''}{(reward?.total ?? 0).toFixed(2)}
                </span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
