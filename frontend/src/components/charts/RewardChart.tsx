'use client';

import { useMemo } from 'react';
import { Card } from '@/components/ui/Card';
import { useRewardHistory, useAgent } from '@/hooks/useTrainingData';
import { rollingAverage, CHART_COLORS } from '@/lib/utils';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export function RewardChart() {
  const rewardHistory = useRewardHistory();
  const agent = useAgent();

  const chartData = useMemo(() => {
    if (rewardHistory.length === 0) return [];

    // Downsample if too many points
    const maxPoints = 500;
    const step = Math.max(1, Math.floor(rewardHistory.length / maxPoints));
    const sampled = rewardHistory.filter((_, i) => i % step === 0);

    // Calculate rolling average
    const smoothed = rollingAverage(sampled, 50);

    return sampled.map((reward, i) => ({
      index: i * step,
      reward,
      smoothed: smoothed[i],
    }));
  }, [rewardHistory]);

  const avgReward = useMemo(() => {
    if (rewardHistory.length === 0) return 0;
    return rewardHistory.reduce((a, b) => a + b, 0) / rewardHistory.length;
  }, [rewardHistory]);

  return (
    <Card title="Episode Rewards">
      <div className="h-40">
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
              <XAxis
                dataKey="index"
                axisLine={false}
                tickLine={false}
                tick={{ fill: CHART_COLORS.muted, fontSize: 10 }}
                tickFormatter={(v) => (v % 100 === 0 ? v : '')}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: CHART_COLORS.muted, fontSize: 10 }}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: CHART_COLORS.background,
                  border: `1px solid ${CHART_COLORS.grid}`,
                  borderRadius: '8px',
                }}
                labelStyle={{ color: CHART_COLORS.muted }}
              />
              <ReferenceLine y={0} stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
              {/* Raw rewards (faded) */}
              <Line
                type="monotone"
                dataKey="reward"
                stroke={CHART_COLORS.primary}
                strokeWidth={1}
                dot={false}
                opacity={0.3}
              />
              {/* Smoothed rewards */}
              <Line
                type="monotone"
                dataKey="smoothed"
                stroke={CHART_COLORS.primary}
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500">
            Waiting for data...
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="mt-2 pt-2 border-t border-slate-700/50 grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="text-xs text-slate-400">Episode</div>
          <div className="text-sm font-mono text-white">{agent?.episode ?? 0}</div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Avg Reward</div>
          <div
            className={`text-sm font-mono ${
              avgReward >= 0 ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {avgReward.toFixed(3)}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Last Reward</div>
          <div
            className={`text-sm font-mono ${
              (agent?.episode_reward ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {(agent?.episode_reward ?? 0).toFixed(2)}
          </div>
        </div>
      </div>
    </Card>
  );
}
