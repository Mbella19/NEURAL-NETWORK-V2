'use client';

import { cn } from '@/lib/utils';

type AlertLevel = 'ok' | 'warn' | 'error';

interface StatusIndicatorProps {
  level: AlertLevel;
  message: string;
}

export function StatusIndicator({ level, message }: StatusIndicatorProps) {
  return (
    <div className="alert-row">
      <div className={cn('alert-icon', level)} />
      <span className="text-zinc-300">{message}</span>
    </div>
  );
}

interface AlertListProps {
  alerts: Array<{ level: AlertLevel; message: string }>;
}

export function AlertList({ alerts }: AlertListProps) {
  if (alerts.length === 0) {
    return (
      <div className="text-sm text-zinc-500 text-center py-4">
        No alerts
      </div>
    );
  }

  return (
    <div>
      {alerts.map((alert, i) => (
        <StatusIndicator key={i} level={alert.level} message={alert.message} />
      ))}
    </div>
  );
}

// Connection badge
interface ConnectionBadgeProps {
  connected: boolean;
  className?: string;
}

export function ConnectionBadge({ connected, className }: ConnectionBadgeProps) {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className={cn('status-dot', connected ? 'connected' : 'disconnected')} />
      <span className={cn('text-xs', connected ? 'text-green-500' : 'text-red-600')}>
        {connected ? 'CONNECTED' : 'DISCONNECTED'}
      </span>
    </div>
  );
}
