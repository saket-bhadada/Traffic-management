import React, { useEffect, useRef, useState } from 'react';
import { Pause, Play, RotateCcw, ShieldAlert, StepForward } from 'lucide-react';
import { LANES, LightColor, SignalState } from '../simulation/stateMachine';

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/traffic';

const SHORT_LANE = {
  N: 'North',
  E: 'East',
  S: 'South',
  W: 'West',
};

function formatStateLabel(state) {
  switch (state) {
    case SignalState.ADAPTIVE_GREEN:
      return 'Green';
    case SignalState.TRUNCATED_GREEN:
      return 'Truncating';
    case SignalState.AMBER:
      return 'Amber';
    case SignalState.ALL_RED:
      return 'All red';
    case SignalState.EMERGENCY_GREEN:
      return 'Emergency';
    default:
      return state || '';
  }
}

export default function LiveDashboard({ headerLeft }) {
  const [data, setData] = useState(null);
  const [connectionState, setConnectionState] = useState('loading');
  const [isPaused, setIsPaused] = useState(false);
  const isPausedRef = useRef(false);

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  useEffect(() => {
    let ws;
    let reconnectTimer = null;
    let unmounted = false;

    const connect = () => {
      if (unmounted) return;
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        if (!unmounted) setConnectionState('connected');
      };

      ws.onmessage = (event) => {
        if (!isPausedRef.current) {
          try {
            const payload = JSON.parse(event.data);
            setData(payload);
            setConnectionState('connected');
          } catch (e) {
            console.error('Error parsing WS message:', e);
          }
        }
      };

      ws.onclose = () => {
        if (unmounted) return;
        setConnectionState('disconnected');
        reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        if (!unmounted) setConnectionState('disconnected');
        ws.close();
      };
    };

    connect();

    return () => {
      unmounted = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, []);

  const handleStep = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/step`, { method: 'POST' });
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error('Failed to step:', err);
    }
  };

  const handleReset = async () => {
    try {
      await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
      handleStep();
    } catch (err) {
      console.error('Failed to reset:', err);
    }
  };

  const hasTelemetry = Boolean(data && data.x_t);
  const activeLane = data?.active_lane || 'N';
  const activeLight = hasTelemetry ? data?.lights?.[activeLane] || LightColor.RED : null;
  const connDotClass =
    connectionState === 'connected'
      ? 'dot-green'
      : connectionState === 'loading'
        ? 'dot-amber'
        : 'dot-offline';

  return (
    <>
      <header className="topbar">
        {headerLeft}

        <div className="topbar-right">
          <span className="conn-indicator">
            <span className={`sig-dot ${connDotClass}`} />
            <span>
              {connectionState === 'connected'
                ? 'Connected'
                : connectionState === 'loading'
                  ? 'Connecting'
                  : 'Offline'}
            </span>
          </span>

          <span className="bar-sep" aria-hidden="true" />

          <button
            type="button"
            className={`ghost-btn ${isPaused ? 'active' : ''}`}
            onClick={() => setIsPaused(!isPaused)}
          >
            {isPaused ? <Play size={13} strokeWidth={2} /> : <Pause size={13} strokeWidth={2} />}
            {isPaused ? 'Resume' : 'Pause'}
          </button>

          <button
            type="button"
            className="ghost-btn"
            onClick={handleStep}
            disabled={!isPaused}
          >
            <StepForward size={13} strokeWidth={2} /> Step
          </button>

          <button type="button" className="ghost-btn" onClick={handleReset}>
            <RotateCcw size={13} strokeWidth={2} /> Reset
          </button>
        </div>
      </header>

      <div className="split-workspace">
        <div className="camera-stage">
          {LANES.map((lane) => {
            const lightColor = hasTelemetry ? data.lights?.[lane] || 'RED' : null;
            const imgBase64 = data?.images?.[lane];

            return (
              <div key={lane} className="camera-cell">
                <div className="cam-label">
                  <span
                    className={`sig-dot ${
                      lightColor ? `dot-${lightColor.toLowerCase()}` : 'dot-offline'
                    }`}
                  />
                  <span>{SHORT_LANE[lane]}</span>
                </div>

                {imgBase64 ? (
                  <img
                    src={`data:image/jpeg;base64,${imgBase64}`}
                    alt={`${SHORT_LANE[lane]} camera`}
                    className="feed-img"
                  />
                ) : (
                  <span className="cam-empty">
                    {connectionState === 'loading' ? 'Connecting…' : 'Offline'}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <aside className="side-pane">
          {hasTelemetry && data.is_emergency_preemption && (
            <div className="emg-bar" role="status">
              <ShieldAlert size={15} strokeWidth={2} />
              <span>Emergency &middot; {SHORT_LANE[activeLane] || activeLane}</span>
            </div>
          )}

          {/* Active Signal */}
          <div className="pane-block">
            <div className="phase-row">
              <div className="phase-left">
                <span
                  className={`sig-dot sig-dot-lg ${
                    activeLight ? `dot-${activeLight.toLowerCase()}` : 'dot-offline'
                  }`}
                />
                <span className="phase-name">
                  {hasTelemetry ? SHORT_LANE[activeLane] || activeLane : 'Offline'}
                </span>
                {hasTelemetry && (
                  <span className="phase-state">{formatStateLabel(data.state)}</span>
                )}
              </div>
              <span className="phase-countdown num">
                {hasTelemetry
                  ? data.state === SignalState.EMERGENCY_GREEN
                    ? 'HOLD'
                    : `${Math.ceil(data.remaining_time)}s`
                  : '--'}
              </span>
            </div>
          </div>

          {/* Per-Approach Counts */}
          <div className="pane-block">
            <div className="approach-list">
              {LANES.map((lane) => {
                const light = hasTelemetry ? data.lights?.[lane] || 'RED' : null;
                const cVal = hasTelemetry ? data.x_t[`C_${lane}`] : '--';
                const eVal = hasTelemetry ? Number(data.x_t[`E_${lane}`] ?? 0) : 0;

                return (
                  <div key={lane} className="approach-row">
                    <div className={`approach-id ${light === 'GREEN' ? 'is-green' : ''}`}>
                      <span
                        className={`sig-dot ${
                          light ? `dot-${light.toLowerCase()}` : 'dot-offline'
                        }`}
                      />
                      <span>{SHORT_LANE[lane]}</span>
                    </div>

                    <div className="approach-count align-end num">
                      {cVal}
                      {eVal > 0 && <span className="emg-tag">+{eVal}E</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}
