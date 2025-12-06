'use client';

import { cn } from '@/lib/utils';
import { ReactNode } from 'react';

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
  headerRight?: ReactNode;
  noPadding?: boolean;
}

export function Card({ title, children, className, headerRight, noPadding }: CardProps) {
  const isFlexCol = className?.includes('flex-col');
  return (
    <div className={cn('card', className)}>
      {title && (
        <div className="card-header flex items-center justify-between flex-shrink-0">
          <span>{title}</span>
          {headerRight}
        </div>
      )}
      <div className={cn(
        noPadding ? '' : 'card-body',
        isFlexCol && 'flex-1 flex flex-col min-h-0'
      )}>
        {children}
      </div>
    </div>
  );
}

// Metric card for key numbers
interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricCard({ label, value, subValue, trend, className }: MetricCardProps) {
  const trendColor = trend === 'up' ? 'text-green-500' : trend === 'down' ? 'text-red-600' : 'text-zinc-50';

  return (
    <div className={cn('card p-4', className)}>
      <div className="metric-label mb-1">{label}</div>
      <div className={cn('metric-value data-value', trendColor)}>{value}</div>
      {subValue && <div className="text-xs text-zinc-500 mt-1">{subValue}</div>}
    </div>
  );
}

// Section header within pages
interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}

export function SectionHeader({ title, subtitle, right }: SectionHeaderProps) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div>
        <h2 className="text-sm font-medium text-zinc-50 uppercase tracking-wider">{title}</h2>
        {subtitle && <p className="text-xs text-zinc-500 mt-0.5">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

// Stat card for inline stats (replaces old StatCard)
interface StatCardProps {
  label: string;
  value: string | number;
  change?: number;
  color?: 'positive' | 'negative' | 'info' | 'warning' | 'default';
}

export function StatCard({ label, value, change, color = 'default' }: StatCardProps) {
  const colorClasses = {
    positive: 'text-green-500',
    negative: 'text-red-600',
    info: 'text-blue-500',
    warning: 'text-amber-500',
    default: 'text-zinc-50',
  };

  return (
    <div className="flex flex-col space-y-1">
      <span className="text-xs text-zinc-500 uppercase tracking-wide">{label}</span>
      <span className={cn('text-lg font-mono font-semibold data-value', colorClasses[color])}>
        {value}
      </span>
      {change !== undefined && (
        <span className={cn('text-xs', change >= 0 ? 'text-green-500' : 'text-red-600')}>
          {change >= 0 ? '+' : ''}
          {change.toFixed(2)}
        </span>
      )}
    </div>
  );
}
