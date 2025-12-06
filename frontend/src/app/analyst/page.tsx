'use client';

import { useState, useEffect, useMemo } from 'react';
import { Card } from '@/components/ui/Card';
import { useAnalyst } from '@/hooks/useTrainingData';
import { cn, formatPercent } from '@/lib/utils';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from 'recharts';
import { colors } from '@/lib/colors';

// Fixed mock data for SSR (no Math.random())
const mockLossHistory = Array.from({ length: 50 }, (_, i) => ({
  epoch: i + 1,
  train: 0.8 - i * 0.01,
  val: 0.85 - i * 0.008,
}));

const mockEncoderBars = Array.from({ length: 16 }, (_, i) => 30 + (i % 5) * 10);
const mockGradientBars = Array.from({ length: 25 }, (_, i) => 30 + (i % 4) * 15);
const mockEncoderStd = [0.18, 0.20, 0.17];

export default function AnalystPage() {
  const analyst = useAnalyst();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const attention = analyst?.attention_weights ?? [0.5, 0.5];
  const attentionStability = Math.abs(attention[0] - attention[1]) < 0.3 ? 'HIGH' : 'LOW';

  if (!mounted) return null;

  return (
    <div className="page-container p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">Market Analyst Training</h1>
          <p className="text-sm text-zinc-500">Supervised learning phase - Multi-timeframe encoder</p>
        </div>
        <div className="text-right">
          <div className="text-xs text-zinc-500">Epoch</div>
          <div className="text-xl font-mono text-zinc-100">
            {analyst?.epoch ?? 0} / 100
          </div>
        </div>
      </div>

      {/* Top Row - Loss Curves and Accuracy */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Loss Curves Chart */}
        <Card title="LOSS CURVES">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockLossHistory}>
                <XAxis
                  dataKey="epoch"
                  stroke={colors.text.muted}
                  fontSize={10}
                  tickLine={false}
                  axisLine={{ stroke: colors.border.subtle }}
                />
                <YAxis
                  stroke={colors.text.muted}
                  fontSize={10}
                  tickLine={false}
                  axisLine={{ stroke: colors.border.subtle }}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: colors.bg.elevated,
                    border: `1px solid ${colors.border.subtle}`,
                    borderRadius: '4px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line
                  type="monotone"
                  dataKey="train"
                  stroke={colors.chart.line1}
                  strokeWidth={2}
                  dot={false}
                  name="Train Loss"
                />
                <Line
                  type="monotone"
                  dataKey="val"
                  stroke={colors.semantic.warning}
                  strokeWidth={2}
                  dot={false}
                  name="Val Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 pt-2 border-t border-zinc-800 flex justify-between text-sm">
            <span className="text-zinc-500">
              Current: <span className="text-blue-500 font-mono">T={(analyst?.train_loss ?? 0).toFixed(3)}</span>
              {' '}
              <span className="text-amber-500 font-mono">V={(analyst?.val_loss ?? 0).toFixed(3)}</span>
            </span>
          </div>
        </Card>

        {/* Accuracy Metrics */}
        <Card title="ACCURACY METRICS">
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Direction Accuracy</span>
              <span className="text-lg font-mono text-green-500">
                {formatPercent(analyst?.direction_acc ?? 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Train Accuracy</span>
              <span className="text-lg font-mono text-zinc-200">
                {formatPercent(analyst?.train_acc ?? 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Validation Accuracy</span>
              <span className="text-lg font-mono text-zinc-200">
                {formatPercent(analyst?.val_acc ?? 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Overfit Gap</span>
              <span className={cn(
                'text-lg font-mono',
                (analyst?.train_acc ?? 0) - (analyst?.val_acc ?? 0) > 0.05 ? 'text-amber-500' : 'text-zinc-200'
              )}>
                +{(((analyst?.train_acc ?? 0) - (analyst?.val_acc ?? 0)) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="pt-2 border-t border-zinc-800">
              <div className="text-xs text-zinc-500">Early Stopping</div>
              <div className="text-sm text-zinc-300">
                {analyst?.epochs_without_improvement ?? 0} epochs without improvement
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Encoder Activations */}
      <Card title="ENCODER ACTIVATIONS" className="mb-6">
        <div className="grid grid-cols-3 gap-6">
          {['15M', '1H', '4H'].map((tf, idx) => {
            const norm = idx === 0 ? analyst?.encoder_15m_norm : idx === 1 ? analyst?.encoder_1h_norm : analyst?.encoder_4h_norm;
            return (
              <div key={tf} className="text-center">
                <div className="text-sm font-medium text-zinc-300 mb-2">{tf} ENCODER</div>
                <div className="h-4 bg-zinc-800 rounded overflow-hidden mb-2">
                  <div
                    className="h-full bg-blue-500"
                    style={{ width: `${(norm ?? 0.5) * 100}%` }}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-zinc-500">Mean: </span>
                    <span className="text-zinc-300 font-mono">{(norm ?? 0.5).toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-zinc-500">Std: </span>
                    <span className="text-zinc-300 font-mono">{mockEncoderStd[idx].toFixed(2)}</span>
                  </div>
                </div>
                {/* Sparkline placeholder */}
                <div className="mt-2 h-8 bg-zinc-800/50 rounded flex items-end justify-around px-1">
                  {mockEncoderBars.map((h, i) => (
                    <div
                      key={i}
                      className="w-1 bg-blue-500/60"
                      style={{ height: `${h}%` }}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Bottom Row - Attention and Gradients */}
      <div className="grid grid-cols-2 gap-4">
        {/* Attention Weights */}
        <Card title="ATTENTION WEIGHTS">
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-zinc-400">1H Weight</span>
                <span className="text-sm font-mono text-zinc-300">{(attention[0] * 100).toFixed(0)}%</span>
              </div>
              <div className="h-3 bg-zinc-800 rounded overflow-hidden">
                <div
                  className="h-full bg-purple-500"
                  style={{ width: `${attention[0] * 100}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-zinc-400">4H Weight</span>
                <span className="text-sm font-mono text-zinc-300">{(attention[1] * 100).toFixed(0)}%</span>
              </div>
              <div className="h-3 bg-zinc-800 rounded overflow-hidden">
                <div
                  className="h-full bg-pink-500"
                  style={{ width: `${attention[1] * 100}%` }}
                />
              </div>
            </div>
            <div className="pt-2 border-t border-zinc-800">
              <div className="flex justify-between">
                <span className="text-xs text-zinc-500">Stability</span>
                <span className={cn(
                  'text-xs font-medium',
                  attentionStability === 'HIGH' ? 'text-green-500' : 'text-amber-500'
                )}>
                  {attentionStability}
                </span>
              </div>
            </div>
          </div>
        </Card>

        {/* Gradient Health */}
        <Card title="GRADIENT HEALTH">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Norm</span>
              <span className={cn(
                'text-lg font-mono',
                (analyst?.grad_norm ?? 0) < 1.0 ? 'text-green-500' : (analyst?.grad_norm ?? 0) < 5.0 ? 'text-amber-500' : 'text-red-600'
              )}>
                {(analyst?.grad_norm ?? 0).toFixed(2)}
                <span className="text-xs text-zinc-500 ml-1">
                  ({(analyst?.grad_norm ?? 0) < 1.0 ? 'healthy' : (analyst?.grad_norm ?? 0) < 5.0 ? 'elevated' : 'exploding'})
                </span>
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Max</span>
              <span className="text-lg font-mono text-zinc-300">
                {((analyst?.grad_norm ?? 0) * 2.5 + 0.5).toFixed(2)}
              </span>
            </div>
            {/* Gradient history sparkline */}
            <div className="pt-2 border-t border-zinc-800">
              <div className="text-xs text-zinc-500 mb-2">History (last 50 batches)</div>
              <div className="h-12 bg-zinc-800/50 rounded flex items-end justify-around px-1">
                {mockGradientBars.map((h, i) => (
                  <div
                    key={i}
                    className={cn(
                      'w-1',
                      i > 20 ? 'bg-amber-500/60' : 'bg-green-500/60'
                    )}
                    style={{ height: `${h}%` }}
                  />
                ))}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
