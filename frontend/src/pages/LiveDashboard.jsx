import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Car,
  Layers,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldAlert,
  WifiOff,
} from 'lucide-react';
import IntersectionDiagram from '../components/IntersectionDiagram';
import QueueGrowthChart from '../components/QueueGrowthChart';
import { LANES, LANE_NAMES, LightColor, SignalState } from '../simulation/stateMachine';

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/traffic';

const XT_KEYS = ['C_N', 'C_E', 'C_S', 'C_W', 'E_N', 'E_E', 'E_S', 'E_W'];

function buildOfflineSnapshot() {
  return {
    timestamp: 0,
    state: SignalState.ADAPTIVE_GREEN,
    activeLane: 'N',
    remainingTime: 0,
    lights: {
      N: LightColor.RED,
      E: LightColor.RED,
      S: LightColor.RED,
      W: LightColor.RED,
    },
    queues: { N: 0, E: 0, S: 0, W: 0 },
    emergencies: { N: 0, E: 0, S: 0, W: 0 },
    discharging: { N: 0, E: 0, S: 0, W: 0 },
    emergencyDischarging: { N: 0, E: 0, S: 0, W: 0 },
  };
}

function buildSchematicSnapshot(data) {
  if (!data) return buildOfflineSnapshot();
  const queues = {};
  const emergencies = {};
  const lights = {};
  for (const lane of LANES) {
    queues[lane] = Number(data.x_t?.[`C_${lane}`] ?? 0);
    emergencies[lane] = Number(data.x_t?.[`E_${lane}`] ?? 0);
    lights[lane] = data.lights?.[lane] || LightColor.RED;
  }
  return {
    timestamp: Number(data.timestamp ?? 0),
    state: data.state || SignalState.ADAPTIVE_GREEN,
    activeLane: data.active_lane || 'N',
    remainingTime: Number(data.remaining_time ?? 0),
    lights,
    queues,
    emergencies,
    discharging: { N: 0, E: 0, S: 0, W: 0 },
    emergencyDischarging: { N: 0, E: 0, S: 0, W: 0 },
  };
}

export default function LiveDashboard() {
  const [data, setData] = useState(null);
  const [connectionState, setConnectionState] = useState('loading');
  const [isPaused, setIsPaused] = useState(false);
  const isPausedRef = useRef(false);
  const wsRef = useRef(null);

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
      wsRef.current = ws;

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

  const hasTelemetry = Boolean(data && data.x_t && data.dc_dt);
  const schematicSnapshot = buildSchematicSnapshot(data);
  const transitions = data?.recent_transitions ?? [];

  const activeLane = data?.active_lane || 'N';
  const activeLaneLabel = LANE_NAMES[activeLane] || activeLane;
  const activeAspect = (data?.lights?.[activeLane] || 'RED').toLowerCase();

  let bannerClass = 'alert-offline';
  if (hasTelemetry) {
    if (data.is_emergency_preemption) {
      bannerClass = 'alert-emergency';
    } else if (data.state === SignalState.AMBER) {
      bannerClass = 'alert-amber';
    } else {
      bannerClass = 'alert-normal';
    }
  }

  return (
    <div className="live-dashboard-page">
      {/* Toolbar */}
      <div className="live-toolbar">
        <div className="live-toolbar-meta">
          <span
            className={`status-pill ${
              connectionState === 'connected'
                ? 'online'
                : connectionState === 'loading'
                  ? 'loading'
                  : 'offline'
            }`}
          >
            {connectionState === 'disconnected' ? (
              <WifiOff size={14} strokeWidth={1.75} />
            ) : (
              <Radio size={14} strokeWidth={1.75} />
            )}
            {connectionState === 'connected'
              ? 'Live WebSocket'
              : connectionState === 'loading'
                ? 'Connecting...'
                : 'Disconnected'}
          </span>

          {!hasTelemetry && (
            <span className="badge-neutral">
              Model offline &middot; YOLOv8 / LSTM unavailable
            </span>
          )}
        </div>

        <div className="live-toolbar-actions">
          <button type="button" className="btn" onClick={() => setIsPaused(!isPaused)}>
            {isPaused ? (
              <Play size={15} strokeWidth={1.75} />
            ) : (
              <Pause size={15} strokeWidth={1.75} />
            )}{' '}
            {isPaused ? 'Resume Stream' : 'Pause'}
          </button>
          <button type="button" className="btn" onClick={handleStep} disabled={!isPaused}>
            Step (1s)
          </button>
          <button type="button" className="btn btn-danger" onClick={handleReset}>
            <RotateCcw size={15} strokeWidth={1.75} /> Reset
          </button>
        </div>
      </div>

      {/* Operational Signal Status Banner */}
      <div className={`alert-banner ${bannerClass}`}>
        {!hasTelemetry ? (
          <>
            <WifiOff size={20} strokeWidth={1.75} className="banner-icon" />
            <div className="banner-text">
              {connectionState === 'loading' ? (
                <span>
                  <b>Controller connecting</b> &middot; awaiting telemetry from{' '}
                  <code>{WS_URL}</code>
                </span>
              ) : (
                <span>
                  <b>Controller offline</b> &middot; no telemetry stream at{' '}
                  <code>{WS_URL}</code> (values shown as <code>--</code>)
                </span>
              )}
            </div>
          </>
        ) : data.is_emergency_preemption ? (
          <>
            <ShieldAlert size={20} strokeWidth={1.75} className="banner-icon" />
            <div className="banner-text">
              <strong>
                Preempt active: {activeLaneLabel} ({activeLane})
              </strong>
              <span>
                {' '}
                &middot; State: <b>{data.state}</b>
                {data.remaining_time > 0
                  ? ` · ${data.remaining_time}s remaining`
                  : ' · Holding green until E == 0'}
              </span>
            </div>
          </>
        ) : (
          <>
            <Car size={20} strokeWidth={1.75} className="banner-icon" />
            <div className="banner-text">
              <strong>
                {activeLaneLabel} ({activeLane}) {activeAspect}
              </strong>
              <span>
                {' '}
                &mdash; <b>{data.remaining_time}s</b> remaining &middot; State:{' '}
                <b>{data.state}</b>
              </span>
            </div>
          </>
        )}
      </div>

      {/* Main Operations Workspace: Camera Feeds + Top-Down Schematic + Side Timeline */}
      <div className="live-workspace">
        <div className="live-main-col">
          <div className="live-visual-split">
            {/* 4-Way Camera Area */}
            <div className="camera-grid">
              {LANES.map((lane) => {
                const lightColor = hasTelemetry ? data.lights?.[lane] || 'RED' : null;
                const imgBase64 = data?.images?.[lane];
                const stdCount = hasTelemetry ? data.x_t[`C_${lane}`] : '--';
                const emgCount = hasTelemetry ? data.x_t[`E_${lane}`] : '--';
                const growthVal = hasTelemetry ? data.dc_dt[`dC_${lane}_dt`] : null;
                const growthText =
                  growthVal === null
                    ? '--'
                    : growthVal > 0
                      ? `+${growthVal}/s`
                      : `${growthVal}/s`;

                const borderClass = lightColor
                  ? `${lightColor.toLowerCase()}-border`
                  : 'offline-border';
                const badgeClass = lightColor
                  ? `light-${lightColor.toLowerCase()}`
                  : 'light-offline';

                return (
                  <div key={lane} className={`camera-card ${borderClass}`}>
                    <div className="camera-header">
                      <span className="camera-title">
                        {LANE_NAMES[lane]} ({lane})
                      </span>
                      <span className={`light-badge ${badgeClass}`}>
                        {lightColor ?? '--'}
                      </span>
                    </div>

                    <div className="video-viewport">
                      {imgBase64 ? (
                        <>
                          <img
                            src={`data:image/jpeg;base64,${imgBase64}`}
                            alt={`Approach ${lane}`}
                            className="feed-img"
                          />
                          <div className="camera-overlay-counts">
                            <span>C_{lane}: {stdCount}</span>
                            <span
                              className={
                                typeof emgCount === 'number' && emgCount > 0
                                  ? 'camera-overlay-emg'
                                  : ''
                              }
                            >
                              E_{lane}: {emgCount}
                            </span>
                          </div>
                        </>
                      ) : (
                        <div className="placeholder-box">
                          <span>
                            {connectionState === 'loading'
                              ? 'Connecting to camera feed...'
                              : 'Camera disconnected'}
                          </span>
                          <span className="placeholder-sub">
                            {connectionState === 'loading'
                              ? `Awaiting frame (${lane})`
                              : 'No stream frame received'}
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="camera-stats">
                      <div className="stat-item">
                        <span className="stat-label">Vehicles (C)</span>
                        <span className="stat-val">{stdCount}</span>
                      </div>
                      <div
                        className={`stat-item ${
                          typeof emgCount === 'number' && emgCount > 0 ? 'highlight-emg' : ''
                        }`}
                      >
                        <span className="stat-label">Emergency (E)</span>
                        <span className="stat-val">{emgCount}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">dC/dt (LSTM)</span>
                        <span className="stat-val">{growthText}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Center Top-Down 4-Way Junction Schematic */}
            <IntersectionDiagram
              snapshot={schematicSnapshot}
              offline={!hasTelemetry}
            />
          </div>
        </div>

        {/* Right Side Panel: Vector X_t + Phase Timeline */}
        <div className="live-side-col">
          <div className="panel">
            <div className="panel-header">
              <div>
                <h3>
                  <Layers size={16} strokeWidth={1.75} /> State Vector X_t
                </h3>
                <p className="panel-sub">YOLOv8 per-approach vehicle &amp; emergency counts</p>
              </div>
              <span className="badge-neutral">
                {hasTelemetry ? `T+${data.timestamp}s` : 'OFFLINE'}
              </span>
            </div>

            <div className="xt-grid">
              {XT_KEYS.map((key) => {
                const val = hasTelemetry ? data.x_t[key] : '--';
                const isEmgHot =
                  key.startsWith('E_') && typeof val === 'number' && val > 0;

                return (
                  <div key={key} className={`xt-badge ${isEmgHot ? 'emg-bg' : ''}`}>
                    <span className="xt-k">{key}</span>
                    <span className="xt-v">{val}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <h3>
                  <AlertTriangle size={16} strokeWidth={1.75} /> Phase Timeline &amp; Log
                </h3>
                <p className="panel-sub">Green 10-60s &middot; Amber 3s &middot; All-Red 1s</p>
              </div>
              <span className="badge-neutral">{transitions.length} events</span>
            </div>

            <div className="phase-spec-bar">
              <span>State: {hasTelemetry ? data.state : '--'}</span>
              <span>|</span>
              <span>Active: {hasTelemetry ? data.active_lane : '--'}</span>
              <span>|</span>
              <span>Rem: {hasTelemetry ? `${data.remaining_time}s` : '--'}</span>
            </div>

            <div className="log-list">
              {transitions.length > 0 ? (
                transitions.map((t, idx) => (
                  <div key={idx} className="log-entry">
                    <span className="log-time">[{t.timestamp}s]</span>
                    <span className="log-states">
                      {t.from_state} &rarr; {t.to_state}
                    </span>
                    <span className="log-lane">Lane {t.active_lane}</span>
                    <span className="log-reason">{t.reason}</span>
                  </div>
                ))
              ) : (
                <div className="empty-state-box">No phase transitions recorded</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Strip: Queue Growth Chart */}
      <QueueGrowthChart
        dcDt={data?.dc_dt ?? null}
        hasData={hasTelemetry}
        title="Queue growth rate (dC/dt)"
      />
    </div>
  );
}
