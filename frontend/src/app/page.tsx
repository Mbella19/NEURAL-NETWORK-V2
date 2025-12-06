'use client';

import { Card, MetricCard } from '@/components/ui/Card';
import { ProgressBar } from '@/components/metrics/ProgressBar';
import { ConnectionBadge } from '@/components/metrics/StatusIndicator';
import { useAgent, useMarket, useSystem, useConnected } from '@/hooks/useTrainingData';
import { formatNumber, formatPrice, formatPercent, cn } from '@/lib/utils';
import { getPnlClass, getPositionBgClass } from '@/lib/colors';

export default function OverviewPage() {
  const agent = useAgent();
  const market = useMarket();
  const system = useSystem();
  const connected = useConnected();

  const getPhaseLabel = () => {
    switch (system?.phase) {
      case 'analyst_training':
        return 'ANALYST';
      case 'agent_training':
        return 'AGENT';
      case 'backtest':
        return 'BACKTEST';
      default:
        return 'IDLE';
    }
  };

  const getPositionLabel = () => {
    if (!market) return 'FLAT';
    if (market.position === 1) return 'LONG';
    if (market.position === -1) return 'SHORT';
    return 'FLAT';
  };

  const getSizeLabel = () => {
    const sizes = ['0.25x', '0.5x', '0.75x', '1.0x'];
    const sizeIdx = market?.position_size ?? 0;
    return sizes[Math.min(sizeIdx, 3)];
  };

  return (
    <div className="page-container p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">EURUSD Hybrid Trading System</h1>
          <p className="text-sm text-zinc-500">Real-time training dashboard</p>
        </div>
        <ConnectionBadge connected={connected} />
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="Phase"
          value={getPhaseLabel()}
        />
        <MetricCard
          label="Timestep"
          value={agent?.timestep?.toLocaleString() ?? '0'}
          subValue={`Episode ${agent?.episode ?? 0}`}
        />
        <MetricCard
          label="Total PnL"
          value={formatNumber(market?.total_pnl ?? 0, 1)}
          trend={market?.total_pnl && market.total_pnl > 0 ? 'up' : market?.total_pnl && market.total_pnl < 0 ? 'down' : 'neutral'}
          subValue={`${market?.n_trades ?? 0} trades`}
        />
        <MetricCard
          label="Win Rate"
          value={formatPercent(agent?.win_rate ?? 0)}
          trend={agent?.win_rate && agent.win_rate > 0.5 ? 'up' : 'down'}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Market Regime */}
        <Card title="MARKET REGIME">
          <div className="flex items-center justify-between">
            <div className="flex gap-6">
              <div className="text-center">
                <div className="text-xs text-zinc-500 mb-1">ATR</div>
                <div className="w-16 h-2 bg-zinc-800 rounded overflow-hidden">
                  <div
                    className="h-full bg-blue-500"
                    style={{ width: `${Math.min(100, (market?.atr ?? 0) * 10000)}%` }}
                  />
                </div>
                <div className="text-xs text-zinc-400 mt-1 font-mono">
                  {(market?.atr ?? 0).toFixed(5)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-zinc-500 mb-1">CHOP</div>
                <div className="w-16 h-2 bg-zinc-800 rounded overflow-hidden">
                  <div
                    className={cn(
                      'h-full',
                      (market?.chop ?? 50) > 61.8 ? 'bg-amber-500' : (market?.chop ?? 50) < 38.2 ? 'bg-green-500' : 'bg-zinc-500'
                    )}
                    style={{ width: `${market?.chop ?? 50}%` }}
                  />
                </div>
                <div className="text-xs text-zinc-400 mt-1 font-mono">
                  {(market?.chop ?? 0).toFixed(1)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-zinc-500 mb-1">ADX</div>
                <div className="w-16 h-2 bg-zinc-800 rounded overflow-hidden">
                  <div
                    className={cn(
                      'h-full',
                      (market?.adx ?? 0) > 25 ? 'bg-green-500' : 'bg-red-600'
                    )}
                    style={{ width: `${Math.min(100, (market?.adx ?? 0) * 2)}%` }}
                  />
                </div>
                <div className="text-xs text-zinc-400 mt-1 font-mono">
                  {(market?.adx ?? 0).toFixed(1)}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-zinc-500 mb-1">Regime</div>
              <div className={cn(
                'text-lg font-semibold',
                market?.regime === 0 ? 'text-green-500' : market?.regime === 2 ? 'text-red-600' : 'text-amber-500'
              )}>
                {market?.regime === 0 ? 'BULLISH' : market?.regime === 2 ? 'BEARISH' : 'RANGING'}
              </div>
            </div>
          </div>
        </Card>

        {/* Current Position */}
        <Card title="CURRENT POSITION">
          {market?.position !== 0 ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className={cn(
                  'px-3 py-1 rounded text-sm font-semibold',
                  getPositionBgClass(market?.position ?? 0 as -1 | 0 | 1)
                )}>
                  {getPositionLabel()} {getSizeLabel()}
                </div>
                <div>
                  <div className="text-xs text-zinc-500">Entry</div>
                  <div className="font-mono text-zinc-200">{formatPrice(market?.entry_price ?? 0)}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-zinc-500">Unrealized</div>
                <div className={cn('text-lg font-semibold font-mono', getPnlClass(market?.unrealized_pnl ?? 0))}>
                  {market?.unrealized_pnl && market.unrealized_pnl >= 0 ? '+' : ''}{(market?.unrealized_pnl ?? 0).toFixed(1)} pips
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-12">
              <div className={cn('px-4 py-2 rounded text-sm', getPositionBgClass(0))}>
                FLAT - No Position
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* Training Progress */}
      <Card title="TRAINING PROGRESS">
        <div className="mb-4">
          <ProgressBar
            value={agent?.episode ?? 0}
            max={500}
            showPercent
          />
        </div>
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-xs text-zinc-500">Episode</div>
            <div className="text-sm font-mono text-zinc-200">
              {agent?.episode ?? 0} / 500
            </div>
          </div>
          <div>
            <div className="text-xs text-zinc-500">Current Reward</div>
            <div className={cn('text-sm font-mono', getPnlClass(agent?.episode_reward ?? 0))}>
              {(agent?.episode_reward ?? 0).toFixed(1)}
            </div>
          </div>
          <div>
            <div className="text-xs text-zinc-500">Avg Reward</div>
            <div className="text-sm font-mono text-zinc-200">
              {(agent?.avg_reward ?? 0).toFixed(1)}
            </div>
          </div>
          <div>
            <div className="text-xs text-zinc-500">Best Reward</div>
            <div className="text-sm font-mono text-green-500">
              {(agent?.best_reward ?? 0).toFixed(1)}
            </div>
          </div>
        </div>
      </Card>

      {/* Bottom Stats Row */}
      <div className="grid grid-cols-4 gap-4 mt-6">
        <Card className="p-4">
          <div className="text-xs text-zinc-500 mb-1">Current Price</div>
          <div className="text-xl font-mono font-semibold text-zinc-100">
            {formatPrice(market?.price ?? 0)}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-zinc-500 mb-1">Value Estimate</div>
          <div className="text-xl font-mono font-semibold text-blue-500">
            {(agent?.value_estimate ?? 0).toFixed(2)}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-zinc-500 mb-1">Policy Entropy</div>
          <div className="text-xl font-mono font-semibold text-zinc-100">
            {(agent?.entropy ?? 0).toFixed(3)}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-zinc-500 mb-1">Device</div>
          <div className="text-xl font-mono font-semibold text-zinc-100 uppercase">
            {system?.device ?? 'mps'}
          </div>
        </Card>
      </div>
    </div>
  );
}
