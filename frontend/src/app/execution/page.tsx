'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { PriceChart } from '@/components/charts/PriceChart';
import { useMarket, useTrades, useEquityCurve } from '@/hooks/useTrainingData';
import { cn, formatPrice, formatDuration } from '@/lib/utils';
import { getPnlClass, getPositionBgClass } from '@/lib/colors';
import {
  ResponsiveContainer,
  Area,
  AreaChart,
  YAxis,
} from 'recharts';
import { colors } from '@/lib/colors';

// Fixed mock data for SSR (no Math.random())
const mockEquityData = Array.from({ length: 100 }, (_, i) => ({
  step: i,
  equity: 100 + i * 0.5,
  drawdown: 2.5,
}));

const mockTrades = [
  { id: 47, direction: 1 as const, size: 2, entry: 1.08456, exit: null, pnl: 8.6, status: 'OPEN', duration: 47 * 60 + 23 },
  { id: 46, direction: -1 as const, size: 1, entry: 1.08234, exit: 1.08189, pnl: 4.5, status: 'EXIT', duration: 2 * 3600 + 14 * 60 },
  { id: 45, direction: 1 as const, size: 3, entry: 1.07892, exit: 1.07834, pnl: -5.8, status: 'SL', duration: 1 * 3600 + 23 * 60 },
  { id: 44, direction: -1 as const, size: 2, entry: 1.08123, exit: 1.08067, pnl: 5.6, status: 'TP', duration: 3 * 3600 + 45 * 60 },
];

export default function ExecutionPage() {
  const market = useMarket();
  const trades = useTrades();
  const equityCurve = useEquityCurve();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const position = market?.position ?? 0;
  const getPositionLabel = () => {
    if (position === 1) return 'LONG';
    if (position === -1) return 'SHORT';
    return 'FLAT';
  };

  const getSizeLabel = () => {
    const sizes = ['0.25x', '0.5x', '0.75x', '1.0x'];
    return sizes[Math.min(market?.position_size ?? 0, 3)];
  };

  // Use real data if available, otherwise use fixed mock data
  const equityData = equityCurve.length > 0 ? equityCurve : mockEquityData;

  // Calculate max drawdown
  let maxEquity = 0;
  let maxDrawdown = 0;
  equityData.forEach(d => {
    maxEquity = Math.max(maxEquity, d.equity);
    const dd = (maxEquity - d.equity) / maxEquity * 100;
    maxDrawdown = Math.max(maxDrawdown, dd);
  });

  // Recent trades (use real or mock)
  const recentTrades = trades.length > 0 ? trades.slice(-10).reverse() : mockTrades;

  if (!mounted) return null;

  return (
    <div className="page-container p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">Live Execution</h1>
          <p className="text-sm text-zinc-500">Real-time trading and position monitoring</p>
        </div>
        <div className="text-right">
          <div className="text-xs text-zinc-500">EURUSD</div>
          <div className="text-2xl font-mono font-semibold text-zinc-100">
            {formatPrice(market?.price ?? 1.08542)}
          </div>
        </div>
      </div>

      {/* Price Chart Placeholder */}
      <div className="grid grid-cols-[3fr_2fr] gap-6 mb-6">
        <Card title="PRICE CHART" className="h-full">
          <div className="h-[30rem] bg-zinc-800/30 rounded overflow-hidden relative">
            <PriceChart headless height="100%" />
          </div>
          <div className="mt-2 pt-2 border-t border-zinc-800 flex justify-between text-sm">
            <span className="text-zinc-400">
              TP: <span className="text-green-500 font-mono">{formatPrice(market?.tp_level ?? 1.08702)}</span>
              <span className="text-zinc-500 ml-1">(+{((market?.tp_level ?? 1.08702) - (market?.entry_price ?? 1.08456) * 10000).toFixed(1)}p)</span>
            </span>
            <span className="text-zinc-400">
              SL: <span className="text-red-600 font-mono">{formatPrice(market?.sl_level ?? 1.08374)}</span>
              <span className="text-zinc-500 ml-1">(-{(((market?.entry_price ?? 1.08456) - (market?.sl_level ?? 1.08374)) * 10000).toFixed(1)}p)</span>
            </span>
            <span className="text-zinc-400">
              RR: <span className="text-blue-500 font-mono">3.0:1</span>
            </span>
          </div>
        </Card>

        {/* Position and Equity Row */}
        <div className="grid grid-cols-1 gap-4 h-fit">
          {/* Equity Curve */}
          <Card title="EQUITY CURVE">
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityData}>
                  <defs>
                    <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={colors.chart.line1} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={colors.chart.line1} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <YAxis domain={['dataMin', 'dataMax']} hide />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke={colors.chart.line1}
                    strokeWidth={2}
                    fill="url(#equityGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-between text-sm mt-2 pt-2 border-t border-zinc-800">
              <span className="text-zinc-400">
                Total: <span className={cn('font-mono', getPnlClass(market?.total_pnl ?? 0))}>
                  {(market?.total_pnl ?? 0) >= 0 ? '+' : ''}{(market?.total_pnl ?? 156.3).toFixed(1)}
                </span>
              </span>
              <span className="text-zinc-400">
                Drawdown: <span className="font-mono text-red-600">-{maxDrawdown.toFixed(1)}%</span>
              </span>
            </div>
          </Card>

          {/* Current Position */}
          <Card title="POSITION">
            {position !== 0 ? (
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className={cn(
                    'px-4 py-2 rounded text-sm font-semibold',
                    getPositionBgClass(position as -1 | 0 | 1)
                  )}>
                    {getPositionLabel()}
                  </div>
                  <span className="text-zinc-300 font-mono">{getSizeLabel()}</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-zinc-500">Entry</div>
                    <div className="font-mono text-zinc-200">{formatPrice(market?.entry_price ?? 0)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-zinc-500">Current</div>
                    <div className="font-mono text-zinc-200">{formatPrice(market?.price ?? 0)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-zinc-500">Unrealized</div>
                    <div className={cn('font-mono text-lg', getPnlClass(market?.unrealized_pnl ?? 0))}>
                      {(market?.unrealized_pnl ?? 0) >= 0 ? '+' : ''}{(market?.unrealized_pnl ?? 0).toFixed(1)} pips
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-zinc-500">Duration</div>
                    <div className="font-mono text-zinc-200">00:47:23</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-48">
                <div className={cn('px-4 py-1 rounded text-xs', getPositionBgClass(0))}>
                  FLAT - No Open Position
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Trade Log */}
      <Card title="TRADE LOG" noPadding>
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Direction</th>
              <th>Size</th>
              <th>Entry</th>
              <th>Exit</th>
              <th>PnL</th>
              <th>Status</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {recentTrades.map((trade, idx) => (
              <tr key={trade.id ?? idx}>
                <td className="font-mono text-zinc-500">{trade.id ?? idx + 1}</td>
                <td className={cn(
                  'font-medium',
                  trade.direction === 1 ? 'text-green-500' : 'text-red-600'
                )}>
                  {trade.direction === 1 ? 'LONG' : 'SHORT'}
                </td>
                <td className="font-mono text-zinc-300">
                  {['0.25x', '0.50x', '0.75x', '1.00x'][trade.size]}
                </td>
                <td className="font-mono text-zinc-300">{formatPrice(trade.entry ?? 0)}</td>
                <td className="font-mono text-zinc-300">
                  {trade.exit ? formatPrice(trade.exit) : '...'}
                </td>
                <td className={cn('font-mono', getPnlClass(trade.pnl ?? 0))}>
                  {(trade.pnl ?? 0) >= 0 ? '+' : ''}{(trade.pnl ?? 0).toFixed(1)}p
                </td>
                <td className={cn(
                  'text-xs font-medium',
                  trade.status === 'OPEN' ? 'text-blue-500' :
                    trade.status === 'TP' ? 'text-green-500' :
                      trade.status === 'SL' ? 'text-red-600' : 'text-zinc-400'
                )}>
                  {trade.status ?? 'CLOSED'}
                </td>
                <td className="font-mono text-zinc-400 text-sm">
                  {formatDuration(trade.duration ?? 0)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
