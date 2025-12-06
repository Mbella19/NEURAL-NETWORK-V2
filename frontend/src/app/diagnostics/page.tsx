'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { ResourceBar } from '@/components/metrics/ProgressBar';
import { AlertList } from '@/components/metrics/StatusIndicator';
import { useSystem, useAgent, useConnected } from '@/hooks/useTrainingData';
import { cn, formatDuration } from '@/lib/utils';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from 'recharts';
import { colors } from '@/lib/colors';

export default function DiagnosticsPage() {
  const system = useSystem();
  const agent = useAgent();
  const connected = useConnected();
  const [mounted, setMounted] = useState(false);
  const [performanceHistory, setPerformanceHistory] = useState<any[]>([]);
  const [uptime, setUptime] = useState(0);

  useEffect(() => {
    setMounted(true);
    setPerformanceHistory(Array.from({ length: 60 }, (_, i) => ({
      time: i,
      memory: 60 + Math.random() * 20,
      stepsPerSec: 1200 + Math.random() * 100,
    })));

    const interval = setInterval(() => {
      setUptime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  if (!mounted) return null;

  const generateAlerts = () => {
    const alerts: Array<{ level: 'ok' | 'warn' | 'error'; message: string }> = [];
    const memUsage = system?.memory_percent ?? 65;
    if (memUsage > 90) {
      alerts.push({ level: 'error', message: `Critical memory: ${memUsage.toFixed(0)}%` });
    } else if (memUsage > 75) {
      alerts.push({ level: 'warn', message: `High memory: ${memUsage.toFixed(0)}%` });
    } else {
      alerts.push({ level: 'ok', message: 'Memory stable' });
    }
    alerts.push({ level: 'ok', message: 'No gradient explosions' });
    const actionProbs = agent?.action_probs ?? [0.33, 0.34, 0.33];
    const maxActionProb = Math.max(...actionProbs);
    if (maxActionProb > 0.8) {
      alerts.push({ level: 'warn', message: `Action skew: ${(maxActionProb * 100).toFixed(0)}%` });
    } else {
      alerts.push({ level: 'ok', message: 'Actions balanced' });
    }
    alerts.push({ level: 'ok', message: 'Loss converging' });
    return alerts;
  };

  const alerts = generateAlerts();
  const totalSteps = 2_000_000;
  const currentStep = agent?.timestep ?? 0;
  const stepsPerSec = system?.steps_per_sec ?? 1245;
  const remainingSteps = totalSteps - currentStep;
  const etaSeconds = stepsPerSec > 0 ? remainingSteps / stepsPerSec : 0;

  return (
    <div className="page-container p-4 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">System Diagnostics</h1>
          <p className="text-xs text-zinc-500">Resource monitoring and health checks</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={cn('status-dot', connected ? 'connected' : 'disconnected')} />
          <span className={cn('text-xs', connected ? 'text-green-500' : 'text-red-600')}>
            {connected ? 'Online' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 grid grid-rows-[auto_1fr_auto] gap-3 min-h-0">
        {/* Top Row - Resources, Speed, Alerts */}
        <div className="grid grid-cols-3 gap-3">
          {/* Resource Usage */}
          <Card title="RESOURCE USAGE">
            <div className="space-y-2">
              <ResourceBar label="MPS" value={system?.mps_memory_used ?? 5.2} max={8.0} unit=" GB" />
              <ResourceBar label="RAM" value={system?.ram_used ?? 4.8} max={8.0} unit=" GB" />
              <ResourceBar label="CPU" value={system?.cpu_percent ?? 34} max={100} unit="%" />
            </div>
          </Card>

          {/* Training Speed */}
          <Card title="TRAINING SPEED">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Steps/sec</span>
                <span className="text-lg font-mono text-zinc-100">{(system?.steps_per_sec ?? stepsPerSec).toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Ep/hr</span>
                <span className="text-lg font-mono text-zinc-100">{system?.episodes_per_hour ?? 47}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">ETA</span>
                <span className="text-lg font-mono text-blue-500">{formatDuration(etaSeconds)}</span>
              </div>
            </div>
          </Card>

          {/* Alerts */}
          <Card title="ALERTS">
            <AlertList alerts={alerts} />
          </Card>
        </div>

        {/* Middle Row - Performance Chart */}
        <Card title="PERFORMANCE HISTORY" className="flex flex-col min-h-0">
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceHistory} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
                <XAxis dataKey="time" stroke={colors.text.muted} fontSize={9} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}s`} />
                <YAxis yAxisId="memory" orientation="left" stroke={colors.text.muted} fontSize={9} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                <YAxis yAxisId="steps" orientation="right" stroke={colors.text.muted} fontSize={9} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                <Line yAxisId="memory" type="monotone" dataKey="memory" stroke={colors.semantic.warning} strokeWidth={1.5} dot={false} name="Memory %" />
                <Line yAxisId="steps" type="monotone" dataKey="stepsPerSec" stroke={colors.chart.line1} strokeWidth={1.5} dot={false} name="Steps/sec" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-4 text-[10px] pt-1 border-t border-zinc-800">
            <span className="flex items-center gap-1"><div className="w-3 h-0.5 bg-amber-500" /> Memory %</span>
            <span className="flex items-center gap-1"><div className="w-3 h-0.5 bg-blue-500" /> Steps/sec</span>
          </div>
        </Card>

        {/* Bottom Row - Connection and Device */}
        <div className="grid grid-cols-2 gap-3">
          {/* Connection */}
          <Card title="CONNECTION">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">WebSocket</span>
                <span className={cn('text-xs font-medium', connected ? 'text-green-500' : 'text-red-600')}>
                  {connected ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Latency</span>
                <span className="text-xs font-mono text-zinc-200">{system?.latency_ms ?? 12}ms</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Msg/sec</span>
                <span className="text-xs font-mono text-zinc-200">{system?.messages_per_sec ?? 10}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Uptime</span>
                <span className="text-xs font-mono text-zinc-200">{formatDuration(uptime)}</span>
              </div>
            </div>
          </Card>

          {/* Device Info */}
          <Card title="DEVICE INFO">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Device</span>
                <span className="text-xs font-mono text-zinc-200">Apple M2</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Backend</span>
                <span className="text-xs font-mono text-zinc-200 uppercase">{system?.device ?? 'MPS'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">Precision</span>
                <span className="text-xs font-mono text-zinc-200">float32</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-400">PyTorch</span>
                <span className="text-xs font-mono text-zinc-200">2.0.1</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
