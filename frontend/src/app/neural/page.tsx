'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { useAnalyst, useAgent } from '@/hooks/useTrainingData';
import { cn, formatPercent } from '@/lib/utils';
import { getHeatmapColor, getPnlClass } from '@/lib/colors';

export default function NeuralPage() {
  const analyst = useAnalyst();
  const agent = useAgent();
  const [mounted, setMounted] = useState(false);
  const [contextVector, setContextVector] = useState<number[]>([]);
  const [layerActivations, setLayerActivations] = useState<number[][][]>([]);

  useEffect(() => {
    setMounted(true);
    setContextVector(Array.from({ length: 64 }, () => Math.random() * 2 - 1));
    setLayerActivations(
      Array.from({ length: 3 }, () =>
        Array.from({ length: 2 }, () =>
          Array.from({ length: 16 }, () => Math.random())
        )
      )
    );
  }, []);

  if (!mounted) return null;

  const contextMin = Math.min(...contextVector);
  const contextMax = Math.max(...contextVector);
  const contextMean = contextVector.reduce((a, b) => a + b, 0) / contextVector.length;
  const activeCount = contextVector.filter(v => Math.abs(v) > 0.5).length;

  const attention = analyst?.attention_weights ?? [0.5, 0.5];
  const pUp = analyst?.p_up ?? 0.5;
  const pDown = analyst?.p_down ?? 0.5;
  const edge = pUp - pDown;
  const confidence = Math.max(pUp, pDown);

  const getActionLabel = () => {
    const probs = agent?.action_probs ?? [0.33, 0.34, 0.33];
    const maxIdx = probs.indexOf(Math.max(...probs));
    const sizes = ['0.25x', '0.5x', '0.75x', '1.0x'];
    const sizeProbs = agent?.size_probs ?? [0.25, 0.25, 0.25, 0.25];
    const sizeIdx = sizeProbs.indexOf(Math.max(...sizeProbs));
    if (maxIdx === 0) return 'FLAT';
    if (maxIdx === 1) return `LONG ${sizes[sizeIdx]}`;
    return `SHORT ${sizes[sizeIdx]}`;
  };

  return (
    <div className="page-container p-4 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">Neural Network Introspection</h1>
          <p className="text-xs text-zinc-500">Real-time visualization of network internals</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-xs text-zinc-400">Real-time</span>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 grid grid-rows-[auto_1fr_auto] gap-3 min-h-0">
        {/* Network Architecture - Compact */}
        <Card title="NETWORK ARCHITECTURE" className="min-h-0">
          <div className="h-80">
            <svg viewBox="0 0 600 100" className="w-full h-full select-none">
              <defs>
                <filter id="glow-sm" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>

              {/* TCN Encoders */}
              <g transform="translate(40, 50)">
                <rect x="-25" y="-40" width="50" height="22" rx="2" fill="#0f172a" stroke="#06b6d4" strokeWidth="1.5" />
                <text x="0" y="-26" textAnchor="middle" fill="#22d3ee" fontSize="8" fontWeight="bold">15m</text>
                <rect x="-25" y="-11" width="50" height="22" rx="2" fill="#0f172a" stroke="#3b82f6" strokeWidth="1.5" />
                <text x="0" y="3" textAnchor="middle" fill="#60a5fa" fontSize="8" fontWeight="bold">1H</text>
                <rect x="-25" y="18" width="50" height="22" rx="2" fill="#0f172a" stroke="#8b5cf6" strokeWidth="1.5" />
                <text x="0" y="32" textAnchor="middle" fill="#a78bfa" fontSize="8" fontWeight="bold">4H</text>
              </g>

              {/* Connections */}
              <path d="M65,10 C100,10 100,50 135,50" fill="none" stroke="#06b6d4" strokeWidth="1" opacity="0.5" />
              <path d="M65,50 L135,50" fill="none" stroke="#3b82f6" strokeWidth="1" opacity="0.5" />
              <path d="M65,90 C100,90 100,50 135,50" fill="none" stroke="#8b5cf6" strokeWidth="1" opacity="0.5" />

              {/* Fusion Core */}
              <g transform="translate(170, 50)">
                <circle r="28" fill="#18181b" stroke="#fbbf24" strokeWidth="2" filter="url(#glow-sm)" />
                <circle r="10" fill="#fbbf24" opacity="0.8">
                  <animate attributeName="opacity" values="0.6;1;0.6" dur="2s" repeatCount="indefinite" />
                </circle>
                <text x="0" y="42" textAnchor="middle" fill="#fbbf24" fontSize="7" fontWeight="bold">ATTENTION</text>
              </g>

              {/* Context Vector */}
              <rect x="220" y="38" width="70" height="24" rx="12" fill="#18181b" stroke="#fbbf24" strokeWidth="1.5" />
              <text x="255" y="54" textAnchor="middle" fill="#fbbf24" fontSize="7" fontWeight="bold">CONTEXT</text>

              {/* Split paths */}
              <path d="M290,50 C310,50 310,25 330,25" fill="none" stroke="#f97316" strokeWidth="1.5" />
              <path d="M290,50 C310,50 310,75 330,75" fill="none" stroke="#8b5cf6" strokeWidth="1.5" />

              {/* Actor/Critic */}
              <g transform="translate(360, 25)">
                <circle r="18" fill="#18181b" stroke="#f97316" strokeWidth="1.5" />
                <text x="0" y="3" textAnchor="middle" fill="#fdba74" fontSize="7" fontWeight="bold">ACTOR</text>
              </g>
              <g transform="translate(360, 75)">
                <circle r="18" fill="#18181b" stroke="#8b5cf6" strokeWidth="1.5" />
                <text x="0" y="3" textAnchor="middle" fill="#d8b4fe" fontSize="7" fontWeight="bold">CRITIC</text>
              </g>

              {/* Outputs */}
              <path d="M378,25 L410,25" stroke="#f97316" strokeWidth="1.5" />
              <path d="M378,75 L410,75" stroke="#8b5cf6" strokeWidth="1.5" />

              <g transform="translate(435, 25)">
                <rect x="-20" y="-12" width="40" height="24" rx="4" fill="#18181b" stroke="#f97316" strokeWidth="1.5" />
                <text x="0" y="4" textAnchor="middle" fill="#fdba74" fontSize="7" fontWeight="bold">ACTION</text>
              </g>
              <g transform="translate(435, 75)">
                <rect x="-20" y="-12" width="40" height="24" rx="4" fill="#18181b" stroke="#8b5cf6" strokeWidth="1.5" />
                <text x="0" y="4" textAnchor="middle" fill="#d8b4fe" fontSize="7" fontWeight="bold">VALUE</text>
              </g>

              {/* Output values */}
              <text x="500" y="28" textAnchor="start" fill="#fdba74" fontSize="9">{getActionLabel()}</text>
              <text x="500" y="78" textAnchor="start" fill="#d8b4fe" fontSize="9">{formatPercent(confidence)}</text>
            </svg>
          </div>
        </Card>

        {/* Middle Row - Context Heatmap + Encoders */}
        <div className="grid grid-cols-4 gap-3 min-h-0 h-full">
          {/* Context Vector Heatmap */}
          <Card title="CONTEXT VECTOR (64D)" className="col-span-2 flex flex-col min-h-0">
            <div className="flex-1 flex flex-wrap gap-0.5 content-start overflow-hidden">
              {contextVector.map((value, idx) => {
                const normalized = (value - contextMin) / (contextMax - contextMin);
                return (
                  <div
                    key={idx}
                    className="w-2.5 h-5 rounded-sm"
                    style={{ backgroundColor: getHeatmapColor(normalized) }}
                    title={`Dim ${idx}: ${value.toFixed(3)}`}
                  />
                );
              })}
            </div>
            <div className="flex justify-between text-[10px] text-zinc-400 border-t border-zinc-800 pt-1 mt-1">
              <span>Min: <span className="font-mono text-zinc-300">{contextMin.toFixed(2)}</span></span>
              <span>Max: <span className="font-mono text-zinc-300">{contextMax.toFixed(2)}</span></span>
              <span>Mean: <span className="font-mono text-zinc-300">{contextMean.toFixed(2)}</span></span>
              <span>Active: <span className="font-mono text-zinc-300">{activeCount}/64</span></span>
            </div>
          </Card>

          {/* Encoder Panels - Compact side by side */}
          <Card title="ENCODER ACTIVATIONS" className="col-span-2 flex flex-col min-h-0">
            <div className="flex-1 grid grid-cols-3 gap-2 min-h-0 overflow-hidden">
              {['15M', '1H', '4H'].map((tf, idx) => {
                const layers = layerActivations[idx] || [];
                const dominantFeature = ['Price', 'Trend', 'Structure'][idx];
                return (
                  <div key={tf} className="flex flex-col min-h-0">
                    <div className="text-[10px] text-zinc-400 mb-1">{tf}</div>
                    {layers.map((layer, layerIdx) => (
                      <div key={layerIdx} className="flex gap-px mb-0.5">
                        {layer.map((val, i) => (
                          <div
                            key={i}
                            className="w-1.5 h-3 rounded-sm"
                            style={{ backgroundColor: getHeatmapColor(val) }}
                          />
                        ))}
                      </div>
                    ))}
                    <div className="text-[9px] text-zinc-500 mt-auto">{dominantFeature}</div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>

        {/* Decision Pathway - Compact */}
        <Card title="DECISION PATHWAY" className="min-h-0">
          <div className="flex items-center gap-2 text-xs h-12">
            <div className="px-2 py-1 bg-zinc-800 rounded text-zinc-300">Input</div>
            <span className="text-zinc-600">→</span>
            <div className="px-2 py-1 bg-zinc-800 rounded text-zinc-300">Encode</div>
            <span className="text-zinc-600">→</span>
            <div className="px-2 py-1 bg-zinc-800 rounded text-zinc-300">Fuse</div>
            <span className="text-zinc-600">→</span>
            <div className={cn(
              'px-2 py-1 rounded font-medium',
              edge > 0 ? 'bg-green-500/20 text-green-500' : edge < 0 ? 'bg-red-600/20 text-red-600' : 'bg-zinc-700 text-zinc-400'
            )}>
              {edge > 0 ? '▲' : edge < 0 ? '▼' : '●'} {edge > 0 ? 'UP' : edge < 0 ? 'DOWN' : 'NEUTRAL'}
            </div>
            <span className="text-zinc-600">→</span>
            <div className={cn(
              'px-3 py-1 rounded font-semibold',
              getActionLabel().includes('LONG') ? 'bg-green-500/20 text-green-500' :
                getActionLabel().includes('SHORT') ? 'bg-red-600/20 text-red-600' : 'bg-zinc-700 text-zinc-400'
            )}>
              {getActionLabel()}
            </div>
            <div className="flex-1" />
            <div className="flex gap-4 text-[10px]">
              <div>
                <span className="text-zinc-500">Conf: </span>
                <span className="font-mono text-blue-500">{formatPercent(confidence)}</span>
              </div>
              <div>
                <span className="text-zinc-500">Edge: </span>
                <span className={cn('font-mono', getPnlClass(edge))}>{edge >= 0 ? '+' : ''}{formatPercent(edge)}</span>
              </div>
              <div>
                <span className="text-zinc-500">1H: </span>
                <span className="font-mono text-purple-500">{formatPercent(attention[0])}</span>
              </div>
              <div>
                <span className="text-zinc-500">4H: </span>
                <span className="font-mono text-pink-500">{formatPercent(attention[1])}</span>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
