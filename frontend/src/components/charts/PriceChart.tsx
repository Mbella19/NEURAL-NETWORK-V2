'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { Card } from '@/components/ui/Card';
import { usePriceHistory, useTradeHistory, useMarket, useAnalyst } from '@/hooks/useTrainingData';
import { CHART_COLORS } from '@/lib/utils';

// Dynamic import for lightweight-charts (client-side only)
let createChart: any;
let CrosshairMode: any;

if (typeof window !== 'undefined') {
  import('lightweight-charts').then((module) => {
    createChart = module.createChart;
    CrosshairMode = module.CrosshairMode;
  });
}

export interface PriceChartProps {
  headless?: boolean;
  height?: number | string;
  className?: string;
}

export function PriceChart({ headless = false, height = 350, className }: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candleSeriesRef = useRef<any>(null);
  const markerSeriesRef = useRef<any>(null);
  const [isLibraryLoaded, setIsLibraryLoaded] = useState(false);
  const hasInitialFit = useRef(false); // Track if we've done initial fit
  const lastDataCount = useRef(0); // Track data count to detect clears

  const priceHistory = usePriceHistory();
  const tradeHistory = useTradeHistory();
  const market = useMarket();
  const analyst = useAnalyst();

  // Load library
  useEffect(() => {
    if (typeof window !== 'undefined' && !createChart) {
      import('lightweight-charts').then((module) => {
        createChart = module.createChart;
        CrosshairMode = module.CrosshairMode;
        setIsLibraryLoaded(true);
      });
    } else if (createChart) {
      setIsLibraryLoaded(true);
    }
  }, []);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current || !createChart) return;

    // Get dimensions from parent container
    const parent = chartContainerRef.current.parentElement;
    const containerWidth = chartContainerRef.current.clientWidth || parent?.clientWidth || 800;
    const containerHeight = chartContainerRef.current.clientHeight || parent?.clientHeight || 480;

    // Create chart with TradingView-like interactivity
    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
      height: containerHeight,
      layout: {
        background: { color: CHART_COLORS.background },
        textColor: CHART_COLORS.muted,
      },
      grid: {
        vertLines: { color: CHART_COLORS.grid },
        horzLines: { color: CHART_COLORS.grid },
      },
      crosshair: {
        mode: CrosshairMode?.Normal,
        vertLine: {
          labelVisible: true,
        },
        horzLine: {
          labelVisible: true,
        },
      },
      rightPriceScale: {
        borderColor: CHART_COLORS.grid,
        autoScale: true,  // Auto-adjust price scale
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: CHART_COLORS.grid,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,  // Space on the right for new bars
        barSpacing: 8,   // Initial bar spacing (adjustable by zoom)
        minBarSpacing: 2,  // Minimum zoom out
        fixLeftEdge: false,  // Allow scrolling past left edge
        fixRightEdge: false, // Allow scrolling past right edge
      },
      handleScroll: {
        mouseWheel: true,      // Mouse wheel to scroll horizontally
        pressedMouseMove: true, // Drag to scroll
        horzTouchDrag: true,   // Touch drag horizontal
        vertTouchDrag: true,   // Touch drag vertical (for price scale)
      },
      handleScale: {
        axisPressedMouseMove: true, // Drag on axis to scale
        mouseWheel: true,           // Mouse wheel to zoom
        pinch: true,                // Pinch to zoom on touch devices
      },
    });

    chartRef.current = chart;

    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: CHART_COLORS.long,
      downColor: CHART_COLORS.short,
      borderVisible: false,
      wickUpColor: CHART_COLORS.long,
      wickDownColor: CHART_COLORS.short,
      priceFormat: {
        type: 'price',
        precision: 5,
        minMove: 0.00001,
      },
    });

    candleSeriesRef.current = candleSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        const parent = chartContainerRef.current.parentElement;
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth || parent?.clientWidth || 800,
          height: chartContainerRef.current.clientHeight || parent?.clientHeight || 480,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, [isLibraryLoaded]);

  // Update price data
  useEffect(() => {
    if (!candleSeriesRef.current) return;

    if (priceHistory.length > 0) {
      // Filter out invalid bars (with 0 values which break chart scale)
      const validBars = priceHistory.filter(bar =>
        bar.open > 0 && bar.high > 0 && bar.low > 0 && bar.close > 0
      );

      if (validBars.length === 0) return;

      const dataToRender = validBars.map((bar) => ({
        time: bar.timestamp as any,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }));

      // Sort data by time in ascending order (required by lightweight-charts)
      dataToRender.sort((a, b) => (a.time as number) - (b.time as number));

      // Deduplicate bars with same timestamp (keep the latest/last one)
      const uniqueData: typeof dataToRender = [];
      const seenTimes = new Set<number>();
      for (let i = dataToRender.length - 1; i >= 0; i--) {
        const bar = dataToRender[i];
        const time = bar.time as number;
        if (!seenTimes.has(time)) {
          seenTimes.add(time);
          uniqueData.unshift(bar);
        }
      }

      candleSeriesRef.current.setData(uniqueData);

      // Detect episode clear (data count dropped significantly) and reset fit
      if (uniqueData.length < lastDataCount.current * 0.5) {
        hasInitialFit.current = false; // Reset so we re-fit to new episode
      }
      lastDataCount.current = uniqueData.length;

      // Force price scale to auto-adjust to the data range
      if (chartRef.current) {
        // Reset price scale to auto mode
        chartRef.current.priceScale('right').applyOptions({
          autoScale: true,
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        });

        // Fit content on first load or after episode clear
        if (!hasInitialFit.current && uniqueData.length > 5) {
          chartRef.current.timeScale().fitContent();
          hasInitialFit.current = true;
        }
      }
    } else {
      // Clear chart when no data
      candleSeriesRef.current.setData([]);
      lastDataCount.current = 0;
      hasInitialFit.current = false;
    }
  }, [priceHistory, isLibraryLoaded]);

  // Update trade markers
  useEffect(() => {
    if (!candleSeriesRef.current || tradeHistory.length === 0) return;

    const markers = tradeHistory.map((trade) => {
      if (trade.is_entry) {
        // Entry markers - clearly labeled LONG or SHORT
        const isLong = trade.direction === 1;
        return {
          time: trade.timestamp as any,
          position: isLong ? 'belowBar' : 'aboveBar',
          color: isLong ? CHART_COLORS.long : CHART_COLORS.short,
          shape: isLong ? 'arrowUp' : 'arrowDown',
          text: isLong ? 'LONG' : 'SHORT',
        };
      } else {
        // Exit markers - show EXIT with PnL
        const pnl = trade.pnl ?? 0;
        const isProfitable = pnl >= 0;
        return {
          time: trade.timestamp as any,
          position: 'inBar',
          color: isProfitable ? '#06b6d4' : '#f97316', // cyan for profit, orange for loss
          shape: 'circle',
          text: `EXIT ${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}`,
        };
      }
    }).sort((a, b) => (a.time as number) - (b.time as number));

    candleSeriesRef.current.setMarkers(markers);
  }, [tradeHistory]);

  // Calculate analyst prediction display
  const analystDirection = analyst && analyst.p_up > analyst.p_down ? 'BULLISH' : 'BEARISH';
  const analystConfidence = analyst ? Math.abs(analyst.p_up - analyst.p_down) * 100 : 0;
  const analystEdge = analyst ? analyst.edge * 100 : 0;

  const chartContent = (
    <div className="relative h-full w-full">
      {/* Analyst Overlay - Top Right Corner */}
      <div className="absolute top-2 right-2 z-10 bg-slate-900/80 backdrop-blur-sm border border-slate-700/50 rounded-lg p-2 text-xs">
        <div className="text-slate-400 text-[10px] mb-1">ANALYST SAYS:</div>
        <div className={`font-bold text-sm ${(analyst?.p_up || 0) > (analyst?.p_down || 0) ? 'text-green-400' : 'text-red-400'}`}>
          {analyst ? analystDirection : 'WAITING...'}
        </div>
        <div className="grid grid-cols-2 gap-x-2 mt-1 text-[10px]">
          <div>
            <span className="text-green-400">↑</span> {((analyst?.p_up || 0.5) * 100).toFixed(1)}%
          </div>
          <div>
            <span className="text-red-400">↓</span> {((analyst?.p_down || 0.5) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="mt-1 text-[10px]">
          <span className="text-slate-400">Conf:</span>{' '}
          <span className={analystConfidence > 10 ? 'text-cyan-400' : 'text-slate-500'}>
            {analystConfidence.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Empty state message */}
      {priceHistory.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-slate-500 text-sm">Waiting for price data...</div>
        </div>
      )}

      <div
        ref={chartContainerRef}
        className="w-full h-full absolute inset-0"
      />
    </div>
  );

  if (headless) {
    return chartContent;
  }

  return (
    <Card title="Price Action">
      {chartContent}

      {/* Price info bar */}
      <div className="mt-2 pt-2 border-t border-slate-700/50 grid grid-cols-5 gap-2 text-center text-xs">
        <div>
          <span className="text-slate-400">O </span>
          <span className="text-white font-mono">
            {priceHistory[priceHistory.length - 1]?.open?.toFixed(5) ?? '-.-----'}
          </span>
        </div>
        <div>
          <span className="text-slate-400">H </span>
          <span className="text-green-400 font-mono">
            {priceHistory[priceHistory.length - 1]?.high?.toFixed(5) ?? '-.-----'}
          </span>
        </div>
        <div>
          <span className="text-slate-400">L </span>
          <span className="text-red-400 font-mono">
            {priceHistory[priceHistory.length - 1]?.low?.toFixed(5) ?? '-.-----'}
          </span>
        </div>
        <div>
          <span className="text-slate-400">C </span>
          <span className="text-white font-mono">
            {priceHistory[priceHistory.length - 1]?.close?.toFixed(5) ?? '-.-----'}
          </span>
        </div>
        <div>
          <span className="text-slate-400">Position </span>
          <span
            className={`font-mono ${market?.position === 1
              ? 'text-green-400'
              : market?.position === -1
                ? 'text-red-400'
                : 'text-slate-400'
              }`}
          >
            {market?.position === 1 ? 'LONG' : market?.position === -1 ? 'SHORT' : 'FLAT'}
          </span>
        </div>
      </div>
    </Card>
  );
}
