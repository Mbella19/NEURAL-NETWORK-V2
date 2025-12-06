'use client';

import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { BarGauge } from '@/components/ui/Gauge';
import { useSystem, useConnected } from '@/hooks/useTrainingData';
import { formatDuration, formatNumber } from '@/lib/utils';

export function SystemStatus() {
  const system = useSystem();
  const connected = useConnected();

  const phase = system?.phase ?? 'idle';
  const memoryUsed = system?.memory_used_mb ?? 0;
  const memoryTotal = system?.memory_total_mb ?? 8192;
  const stepsPerSec = system?.steps_per_second ?? 0;
  const epsPerHour = system?.episodes_per_hour ?? 0;
  const elapsed = system?.elapsed_seconds ?? 0;
  const device = system?.device ?? 'mps';

  const phaseLabels: Record<string, string> = {
    idle: 'Idle',
    analyst_training: 'Training Analyst',
    agent_training: 'Training Agent',
    backtest: 'Backtesting',
  };

  const phaseVariants: Record<string, 'default' | 'info' | 'success' | 'warning'> = {
    idle: 'default',
    analyst_training: 'info',
    agent_training: 'success',
    backtest: 'warning',
  };

  return (
    <Card title="System Status">
      <div className="space-y-4">
        {/* Connection Status */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Connection</span>
          <Badge variant={connected ? 'success' : 'error'} pulse={connected}>
            {connected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>

        {/* Training Phase */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Phase</span>
          <Badge variant={phaseVariants[phase]} pulse={phase !== 'idle'}>
            {phaseLabels[phase] ?? phase}
          </Badge>
        </div>

        {/* Device */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-sm">Device</span>
          <span className="font-mono text-blue-400 text-sm uppercase">{device}</span>
        </div>

        {/* Memory Usage */}
        <BarGauge
          value={memoryUsed}
          max={memoryTotal}
          label={`Memory (${formatNumber(memoryUsed, 0)} / ${formatNumber(memoryTotal, 0)} MB)`}
          color={memoryUsed / memoryTotal > 0.8 ? '#ef4444' : '#3b82f6'}
        />

        {/* Performance Metrics */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-slate-700/50">
          <div>
            <div className="text-xs text-slate-400">Steps/sec</div>
            <div className="text-sm font-mono text-white">
              {formatNumber(stepsPerSec, 1)}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Episodes/hr</div>
            <div className="text-sm font-mono text-white">
              {formatNumber(epsPerHour, 1)}
            </div>
          </div>
        </div>

        {/* Elapsed Time */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-700/50">
          <span className="text-slate-400 text-sm">Elapsed</span>
          <span className="font-mono text-white">{formatDuration(elapsed)}</span>
        </div>
      </div>
    </Card>
  );
}
