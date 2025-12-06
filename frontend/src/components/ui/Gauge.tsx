'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface GaugeProps {
  value: number;
  min: number;
  max: number;
  label: string;
  thresholds?: { low: number; high: number };
  colors?: { low: string; mid: string; high: string };
  showValue?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export function Gauge({
  value,
  min,
  max,
  label,
  thresholds = { low: 33, high: 66 },
  colors = { low: '#22c55e', mid: '#f59e0b', high: '#ef4444' },
  showValue = true,
  size = 'md',
}: GaugeProps) {
  const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const angle = -135 + normalized * 270;

  const getColor = () => {
    const pct = ((value - min) / (max - min)) * 100;
    if (pct < thresholds.low) return colors.low;
    if (pct > thresholds.high) return colors.high;
    return colors.mid;
  };

  const sizeClasses = {
    sm: 'w-16 h-10',
    md: 'w-24 h-14',
    lg: 'w-32 h-20',
  };

  const fontSize = {
    sm: '8px',
    md: '10px',
    lg: '12px',
  };

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 60" className={sizeClasses[size]}>
        {/* Background arc */}
        <path
          d="M10,50 A40,40 0 0,1 90,50"
          fill="none"
          stroke="#1e293b"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Value arc */}
        <motion.path
          d="M10,50 A40,40 0 0,1 90,50"
          fill="none"
          stroke={getColor()}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${normalized * 126} 126`}
          initial={{ strokeDasharray: '0 126' }}
          animate={{ strokeDasharray: `${normalized * 126} 126` }}
          transition={{ type: 'spring', stiffness: 50, damping: 15 }}
        />
        {/* Needle */}
        <motion.line
          x1="50"
          y1="50"
          x2="50"
          y2="20"
          stroke={getColor()}
          strokeWidth="2"
          strokeLinecap="round"
          style={{ transformOrigin: '50px 50px' }}
          animate={{ rotate: angle }}
          transition={{ type: 'spring', stiffness: 100, damping: 15 }}
        />
        {/* Center dot */}
        <circle cx="50" cy="50" r="4" fill={getColor()} />
      </svg>
      <span className="text-xs text-slate-400 mt-1">{label}</span>
      {showValue && (
        <motion.span
          key={value}
          initial={{ scale: 1.1 }}
          animate={{ scale: 1 }}
          className="text-sm font-mono text-white"
          style={{ fontSize: fontSize[size] }}
        >
          {value.toFixed(1)}
        </motion.span>
      )}
    </div>
  );
}

interface BarGaugeProps {
  value: number;
  max: number;
  label: string;
  color?: string;
  showValue?: boolean;
}

export function BarGauge({
  value,
  max,
  label,
  color = '#3b82f6',
  showValue = true,
}: BarGaugeProps) {
  const percentage = Math.max(0, Math.min(100, (value / max) * 100));

  return (
    <div className="flex flex-col space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        {showValue && <span className="text-white font-mono">{value.toFixed(1)}</span>}
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ type: 'spring', stiffness: 100, damping: 15 }}
        />
      </div>
    </div>
  );
}
