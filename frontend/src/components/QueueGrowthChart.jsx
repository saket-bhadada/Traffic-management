import React from 'react';
import { LANES, LANE_NAMES } from '../simulation/stateMachine';

export default function QueueGrowthChart({
  dcDt = null,
  hasData = true,
  title = 'Queue growth rate (dC/dt)',
}) {
  const maxAbs = hasData && dcDt
    ? Math.max(
        1.0,
        ...LANES.map((lane) => Math.abs(Number(dcDt[`dC_${lane}_dt`] ?? 0)))
      )
    : 1.0;

  return (
    <div className="panel qgrowth-panel">
      <div className="panel-header">
        <div>
          <h3>{title}</h3>
          <p className="panel-sub">
            Per-approach queue delta (veh/s) &middot; positive indicates queue build-up
          </p>
        </div>
        <span className="mono badge-neutral">
          {hasData ? `SCALE ±${maxAbs.toFixed(1)} veh/s` : 'NO DATA'}
        </span>
      </div>

      {!hasData || !dcDt ? (
        <div className="empty-state-box">
          No queue growth telemetry — model offline or awaiting stream ticks.
        </div>
      ) : (
        <div className="qgrowth-list">
          {LANES.map((lane) => {
            const raw = Number(dcDt[`dC_${lane}_dt`] ?? 0);
            const pct = Math.min(100, Math.round((Math.abs(raw) / maxAbs) * 100));
            const isPositive = raw > 0.01;
            const isNegative = raw < -0.01;
            const formatted = raw > 0 ? `+${raw.toFixed(2)}` : raw.toFixed(2);

            return (
              <div key={lane} className="qgrowth-row">
                <div className="qgrowth-label">
                  <span className="mono qgrowth-lane">{lane}</span>
                  <span className="qgrowth-name">{LANE_NAMES[lane]}</span>
                </div>

                <div className="qgrowth-bar-area" aria-label={`${lane} queue growth ${formatted} veh/s`}>
                  <div className="qgrowth-half qgrowth-neg">
                    {isNegative && (
                      <div
                        className="qgrowth-fill fill-neg"
                        style={{ width: `${pct}%` }}
                      />
                    )}
                  </div>
                  <div className="qgrowth-axis" />
                  <div className="qgrowth-half qgrowth-pos">
                    {isPositive && (
                      <div
                        className="qgrowth-fill fill-pos"
                        style={{ width: `${pct}%` }}
                      />
                    )}
                  </div>
                </div>

                <div
                  className={`mono qgrowth-val ${
                    isPositive ? 'val-pos' : isNegative ? 'val-neg' : ''
                  }`}
                >
                  {formatted} veh/s
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
