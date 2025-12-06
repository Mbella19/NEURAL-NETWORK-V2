'use client';

import { useMemo } from 'react';
import { Card } from '@/components/ui/Card';
import { usePnlHistory, useMarket } from '@/hooks/useTrainingData';
import { calculateDrawdown, formatPips, CHART_COLORS } from '@/lib/utils';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export function EquityCurve() {
  const pnlHistory = usePnlHistory();
  const market = useMarket();

  const chartData = useMemo(() => {
    if (pnlHistory.length === 0) return [];

    const drawdown = calculateDrawdown(pnlHistory);
    return pnlHistory.map((pnl, i) => ({
      index: i,
      pnl,
      drawdown: drawdown[i],
    }));
  }, [pnlHistory]);

  const currentPnl = market?.total_pnl ?? 0;
  const maxDrawdown = useMemo(() => {
    if (chartData.length === 0) return 0;
    return Math.max(...chartData.map((d) => d.drawdown));
  }, [chartData]);

  const pnlColor = currentPnl >= 0 ? CHART_COLORS.success : CHART_COLORS.error;

  return (
    <Card title="Equity Curve">
      <div className="h-48">
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
              <defs>
                <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="5%"
                    stopColor={pnlColor}
                    stopOpacity={0.3}
                  />
                  <stop
                    offset="95%"
                    stopColor={pnlColor}
                    stopOpacity={0}
                  />
                </linearGradient>
              </defs>
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
                tickFormatter={(v) => `${v.toFixed(0)}`}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: CHART_COLORS.background,
                  border: `1px solid ${CHART_COLORS.grid}`,
                  borderRadius: '8px',
                }}
                labelStyle={{ color: CHART_COLORS.muted }}
                itemStyle={{ color: pnlColor }}
                formatter={(value: number) => [`${value.toFixed(1)} pips`, 'PnL']}
              />
              <ReferenceLine y={0} stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
              <Area
                type="monotone"
                dataKey="pnl"
                stroke={pnlColor}
                fill="url(#pnlGradient)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500">
            Waiting for data...
          </div>
        )}
      </div>

      {/* Stats row */}
      <div className="mt-4 pt-4 border-t border-slate-700/50 grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-xs text-slate-400">Total PnL</div>
          <div
            className={`text-lg font-mono font-bold ${
              currentPnl >= 0 ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {formatPips(currentPnl)}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Max Drawdown</div>
          <div className="text-lg font-mono font-bold text-red-400">
            -{maxDrawdown.toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Win Rate</div>
          <div className="text-lg font-mono font-bold text-blue-400">
            {((market?.n_trades ?? 0) > 0
              ? (50).toFixed(1)  // Placeholder - would need actual win tracking
              : '0.0')}%
          </div>
        </div>
      </div>
    </Card>
  );
}
