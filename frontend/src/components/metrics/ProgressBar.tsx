'use client';

import { cn } from '@/lib/utils';

interface ProgressBarProps {
  value: number;
  max: number;
  label?: string;
  showPercent?: boolean;
  className?: string;
  barColor?: string;
}

export function ProgressBar({
  value,
  max,
  label,
  showPercent = true,
  className,
  barColor = 'bg-blue-500',
}: ProgressBarProps) {
  const percent = max > 0 ? Math.min(100, (value / max) * 100) : 0;

  return (
    <div className={cn('w-full', className)}>
      {(label || showPercent) && (
        <div className="flex justify-between items-center mb-1">
          {label && <span className="text-xs text-zinc-500">{label}</span>}
          {showPercent && (
            <span className="text-xs text-zinc-400 font-mono">{percent.toFixed(0)}%</span>
          )}
        </div>
      )}
      <div className="distribution-bar">
        <div
          className={cn('distribution-fill progress-bar', barColor)}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

interface ResourceBarProps {
  label: string;
  value: number;
  max: number;
  unit?: string;
  className?: string;
}

export function ResourceBar({ label, value, max, unit = '', className }: ResourceBarProps) {
  const percent = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const barColor = percent > 90 ? 'bg-red-600' : percent > 70 ? 'bg-amber-500' : 'bg-blue-500';

  return (
    <div className={cn('', className)}>
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm text-zinc-400">{label}</span>
        <span className="text-xs text-zinc-500 font-mono">
          {value.toFixed(1)}{unit} / {max}{unit}
        </span>
      </div>
      <div className="distribution-bar">
        <div
          className={cn('distribution-fill progress-bar', barColor)}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
