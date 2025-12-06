'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
}

export function Badge({
  children,
  variant = 'default',
  size = 'md',
  pulse = false,
}: BadgeProps) {
  const variantClasses = {
    default: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
    success: 'bg-green-500/20 text-green-400 border-green-500/30',
    warning: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    error: 'bg-red-500/20 text-red-400 border-red-500/30',
    info: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  };

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  };

  return (
    <motion.span
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={cn(
        'inline-flex items-center font-medium rounded-md border',
        variantClasses[variant],
        sizeClasses[size]
      )}
    >
      {pulse && (
        <span className="relative flex h-2 w-2 mr-2">
          <span
            className={cn(
              'animate-ping absolute inline-flex h-full w-full rounded-full opacity-75',
              variant === 'success' && 'bg-green-400',
              variant === 'error' && 'bg-red-400',
              variant === 'warning' && 'bg-yellow-400',
              variant === 'info' && 'bg-blue-400',
              variant === 'default' && 'bg-slate-400'
            )}
          />
          <span
            className={cn(
              'relative inline-flex rounded-full h-2 w-2',
              variant === 'success' && 'bg-green-500',
              variant === 'error' && 'bg-red-500',
              variant === 'warning' && 'bg-yellow-500',
              variant === 'info' && 'bg-blue-500',
              variant === 'default' && 'bg-slate-500'
            )}
          />
        </span>
      )}
      {children}
    </motion.span>
  );
}

interface PositionBadgeProps {
  position: -1 | 0 | 1;
  size?: number;
}

export function PositionBadge({ position, size = 0 }: PositionBadgeProps) {
  const labels = {
    [-1]: 'SHORT',
    [0]: 'FLAT',
    [1]: 'LONG',
  };

  const variants: Record<number, 'success' | 'error' | 'default'> = {
    [-1]: 'error',
    [0]: 'default',
    [1]: 'success',
  };

  const sizeLabel = size > 0 ? ` ${(size * 100).toFixed(0)}%` : '';

  return (
    <Badge variant={variants[position]} size="lg" pulse={position !== 0}>
      {labels[position]}
      {sizeLabel}
    </Badge>
  );
}
