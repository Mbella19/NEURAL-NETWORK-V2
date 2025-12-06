'use client';

import { Card } from '@/components/ui/Card';
import { useAnalyst } from '@/hooks/useTrainingData';
import { motion } from 'framer-motion';
import { CHART_COLORS } from '@/lib/utils';

export function AttentionFlow() {
  const analyst = useAnalyst();

  const attention = analyst?.attention_weights ?? [0.5, 0.5];
  const pDown = analyst?.p_down ?? 0.5;
  const pUp = analyst?.p_up ?? 0.5;
  const confidence = analyst?.confidence ?? 0.5;
  const edge = analyst?.edge ?? 0;

  // Encoder activation levels (normalized)
  const enc15m = analyst?.encoder_15m_norm ?? 0.5;
  const enc1h = analyst?.encoder_1h_norm ?? 0.5;
  const enc4h = analyst?.encoder_4h_norm ?? 0.5;

  // Direction arrow rotation
  const arrowRotation = edge * 90; // -90 to +90 degrees

  return (
    <Card title="Neural Network Flow">
      <svg viewBox="0 0 400 280" className="w-full h-56">
        {/* Background grid */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path
              d="M 20 0 L 0 0 0 20"
              fill="none"
              stroke="#1e293b"
              strokeWidth="0.5"
            />
          </pattern>
          {/* Glow filters */}
          <filter id="glow-blue" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glow-purple" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <rect width="400" height="280" fill="url(#grid)" opacity="0.5" />

        {/* Encoder Nodes */}
        {/* 15m Encoder */}
        <g transform="translate(60, 60)">
          <motion.circle
            r="25"
            fill="#3b82f6"
            opacity={0.3 + enc15m * 0.7}
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ repeat: Infinity, duration: 2 }}
            filter="url(#glow-blue)"
          />
          <circle r="20" fill="#1e293b" stroke="#3b82f6" strokeWidth="2" />
          <text y="5" textAnchor="middle" fill="#3b82f6" fontSize="12" fontWeight="bold">
            15m
          </text>
        </g>

        {/* 1H Encoder */}
        <g transform="translate(60, 140)">
          <motion.circle
            r="25"
            fill="#8b5cf6"
            opacity={0.3 + enc1h * 0.7}
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ repeat: Infinity, duration: 2, delay: 0.3 }}
            filter="url(#glow-purple)"
          />
          <circle r="20" fill="#1e293b" stroke="#8b5cf6" strokeWidth="2" />
          <text y="5" textAnchor="middle" fill="#8b5cf6" fontSize="12" fontWeight="bold">
            1H
          </text>
        </g>

        {/* 4H Encoder */}
        <g transform="translate(60, 220)">
          <motion.circle
            r="25"
            fill="#ec4899"
            opacity={0.3 + enc4h * 0.7}
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ repeat: Infinity, duration: 2, delay: 0.6 }}
          />
          <circle r="20" fill="#1e293b" stroke="#ec4899" strokeWidth="2" />
          <text y="5" textAnchor="middle" fill="#ec4899" fontSize="12" fontWeight="bold">
            4H
          </text>
        </g>

        {/* Flow lines to Fusion */}
        {/* 15m -> Fusion (always full weight as query) */}
        <motion.path
          d="M85,60 Q140,60 180,120"
          stroke="#3b82f6"
          strokeWidth="3"
          fill="none"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1 }}
        />

        {/* 1H -> Fusion (weighted) */}
        <motion.path
          d="M85,140 L180,140"
          stroke="#8b5cf6"
          strokeWidth={2 + attention[0] * 4}
          fill="none"
          strokeLinecap="round"
          opacity={0.3 + attention[0] * 0.7}
          className="flow-line"
        />

        {/* 4H -> Fusion (weighted) */}
        <motion.path
          d="M85,220 Q140,220 180,160"
          stroke="#ec4899"
          strokeWidth={2 + attention[1] * 4}
          fill="none"
          strokeLinecap="round"
          opacity={0.3 + attention[1] * 0.7}
          className="flow-line"
        />

        {/* Fusion Node */}
        <g transform="translate(200, 140)">
          <motion.circle
            r="35"
            fill="#0ea5e9"
            opacity="0.2"
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          />
          <circle r="28" fill="#1e293b" stroke="#0ea5e9" strokeWidth="2" />
          <text y="-5" textAnchor="middle" fill="#0ea5e9" fontSize="10">
            Attention
          </text>
          <text y="10" textAnchor="middle" fill="#0ea5e9" fontSize="10">
            Fusion
          </text>
        </g>

        {/* Fusion -> Context */}
        <motion.path
          d="M235,140 L280,140"
          stroke="#22c55e"
          strokeWidth="4"
          fill="none"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5, delay: 1 }}
        />

        {/* Context Vector Node */}
        <g transform="translate(310, 140)">
          <rect
            x="-30"
            y="-25"
            width="60"
            height="50"
            rx="8"
            fill="#1e293b"
            stroke="#22c55e"
            strokeWidth="2"
          />
          <text y="-5" textAnchor="middle" fill="#22c55e" fontSize="10">
            Context
          </text>
          <text y="10" textAnchor="middle" fill="#94a3b8" fontSize="9">
            64-dim
          </text>
        </g>

        {/* Direction Arrow */}
        <g transform="translate(370, 140)">
          <motion.g
            animate={{ rotate: arrowRotation }}
            transition={{ type: 'spring', stiffness: 100 }}
          >
            <path
              d="M0,-30 L10,0 L5,0 L5,30 L-5,30 L-5,0 L-10,0 Z"
              fill={edge > 0 ? '#22c55e' : edge < 0 ? '#ef4444' : '#64748b'}
            />
          </motion.g>
        </g>

        {/* Attention Weight Labels */}
        <g transform="translate(130, 170)">
          <text fill="#8b5cf6" fontSize="9">
            1H: {(attention[0] * 100).toFixed(0)}%
          </text>
        </g>
        <g transform="translate(130, 185)">
          <text fill="#ec4899" fontSize="9">
            4H: {(attention[1] * 100).toFixed(0)}%
          </text>
        </g>

        {/* Probability Display */}
        <g transform="translate(370, 60)">
          <text textAnchor="middle" fill="#ef4444" fontSize="11">
            Down
          </text>
          <text y="15" textAnchor="middle" fill="#ef4444" fontSize="13" fontWeight="bold">
            {(pDown * 100).toFixed(1)}%
          </text>
        </g>
        <g transform="translate(370, 220)">
          <text textAnchor="middle" fill="#22c55e" fontSize="11">
            Up
          </text>
          <text y="15" textAnchor="middle" fill="#22c55e" fontSize="13" fontWeight="bold">
            {(pUp * 100).toFixed(1)}%
          </text>
        </g>
      </svg>

      {/* Stats row */}
      <div className="mt-2 pt-2 border-t border-slate-700/50 grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="text-xs text-slate-400">Confidence</div>
          <div className="text-sm font-mono text-blue-400">
            {(confidence * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Edge</div>
          <div
            className={`text-sm font-mono ${
              edge > 0 ? 'text-green-400' : edge < 0 ? 'text-red-400' : 'text-slate-400'
            }`}
          >
            {edge >= 0 ? '+' : ''}{(edge * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Uncertainty</div>
          <div className="text-sm font-mono text-yellow-400">
            {(analyst?.uncertainty ?? 0).toFixed(3)}
          </div>
        </div>
      </div>
    </Card>
  );
}
