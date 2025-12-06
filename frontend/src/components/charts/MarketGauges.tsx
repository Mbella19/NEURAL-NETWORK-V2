'use client';

import { Card } from '@/components/ui/Card';
import { Gauge } from '@/components/ui/Gauge';
import { useMarket } from '@/hooks/useTrainingData';
import { cn, getRegimeLabel, getRegimeColor } from '@/lib/utils';
import { motion } from 'framer-motion';

export function MarketGauges() {
  const market = useMarket();

  const atr = market?.atr ?? 0;
  const chop = market?.chop ?? 50;
  const adx = market?.adx ?? 25;
  const regime = market?.regime ?? 1;

  // Convert ATR to pips (EURUSD)
  const atrPips = atr * 10000;

  return (
    <Card title="Market State">
      <div className="grid grid-cols-4 gap-4">
        {/* ATR Gauge */}
        <Gauge
          value={atrPips}
          min={0}
          max={30}
          label="ATR (pips)"
          thresholds={{ low: 8, high: 20 }}
          colors={{ low: '#22c55e', mid: '#f59e0b', high: '#ef4444' }}
        />

        {/* Choppiness Index */}
        <Gauge
          value={chop}
          min={0}
          max={100}
          label="Choppiness"
          thresholds={{ low: 38.2, high: 61.8 }}
          colors={{ low: '#22c55e', mid: '#f59e0b', high: '#ef4444' }}
        />

        {/* ADX */}
        <Gauge
          value={adx}
          min={0}
          max={60}
          label="ADX"
          thresholds={{ low: 20, high: 40 }}
          colors={{ low: '#ef4444', mid: '#f59e0b', high: '#22c55e' }}
        />

        {/* Regime Indicator */}
        <div className="flex flex-col items-center justify-center">
          <motion.div
            key={regime}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className={cn(
              'text-xl font-bold',
              getRegimeColor(regime as 0 | 1 | 2)
            )}
          >
            {getRegimeLabel(regime as 0 | 1 | 2)}
          </motion.div>
          <span className="text-xs text-slate-400 mt-1">Regime</span>
        </div>
      </div>

      {/* Additional market info */}
      <div className="mt-4 pt-4 border-t border-slate-700/50 grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-xs text-slate-400">Price</div>
          <div className="text-sm font-mono text-white">
            {market?.current_price?.toFixed(5) ?? '-.-----'}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400">SMA Distance</div>
          <div className="text-sm font-mono text-white">
            {((market?.sma_distance ?? 0) * 10000).toFixed(1)} pips
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Trades</div>
          <div className="text-sm font-mono text-white">
            {market?.n_trades ?? 0}
          </div>
        </div>
      </div>
    </Card>
  );
}
