import React, { useState, useEffect, useRef } from 'react';
import {
  AlertTriangle,
  Play,
  Pause,
  RotateCcw,
  Activity,
  Layers,
  Radio,
  Clock,
  ShieldAlert,
  Car
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/traffic';

export default function App() {
  const [data, setData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    let ws;
    const connect = () => {
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        if (!isPaused) {
          try {
            const payload = JSON.parse(event.data);
            setData(payload);
          } catch (e) {
            console.error('Error parsing WS message:', e);
          }
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      if (ws) ws.close();
    };
  }, [isPaused]);

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

  const lanes = ['N', 'E', 'S', 'W'];
  const laneNames = { N: 'North Approach', E: 'East Approach', S: 'South Approach', W: 'West Approach' };

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <Activity className="icon-pulse" />
          <div>
            <h1>Adaptive Traffic Signal Control with Emergency Preemption</h1>
            <p className="subtitle">Indian 4-Way Intersection (Chowk) | YOLOv8 + 2-Layer LSTM + Deterministic State Machine</p>
          </div>
        </div>
        <div className="header-controls">
          <span className={`status-pill ${isConnected ? 'online' : 'offline'}`}>
            <Radio size={14} /> {isConnected ? 'Live WebSocket' : 'Connecting...'}
          </span>
          <button className="btn" onClick={() => setIsPaused(!isPaused)}>
            {isPaused ? <Play size={16} /> : <Pause size={16} />} {isPaused ? 'Resume Stream' : 'Pause'}
          </button>
          <button className="btn" onClick={handleStep} disabled={!isPaused}>
            Step (1s)
          </button>
          <button className="btn btn-danger" onClick={handleReset}>
            <RotateCcw size={16} /> Reset
          </button>
        </div>
      </header>

      {/* Emergency Preemption Alert Banner */}
      {data && (
        <div className={`alert-banner ${data.is_emergency_preemption ? 'alert-emergency' : 'alert-normal'}`}>
          {data.is_emergency_preemption ? (
            <>
              <ShieldAlert size={28} className="banner-icon" />
              <div>
                <strong>EMERGENCY VEHICLE PREEMPTION ACTIVE!</strong>
                <span> State: {data.state} | Active Approach: {data.active_lane} | Clearing emergency vehicle...</span>
              </div>
            </>
          ) : (
            <>
              <Car size={24} className="banner-icon" />
              <div>
                <strong>Normal Clockwise Adaptive Cycle</strong>
                <span> State: {data.state} | Active Lane: {data.active_lane} | Remaining: {data.remaining_time}s</span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Main Grid: 4-Way Synchronized Video Feeds */}
      <div className="camera-grid">
        {lanes.map((lane) => {
          const lightColor = data?.lights?.[lane] || 'RED';
          const imgBase64 = data?.images?.[lane];
          const stdCount = data?.x_t?.[`C_${lane}`] ?? 0;
          const emgCount = data?.x_t?.[`E_${lane}`] ?? 0;
          const growthRate = data?.dc_dt?.[`dC_${lane}_dt`] ?? 0;

          return (
            <div key={lane} className={`camera-card ${lightColor.toLowerCase()}-border`}>
              <div className="camera-header">
                <span className="camera-title">{laneNames[lane]} ({lane})</span>
                <span className={`light-badge light-${lightColor.toLowerCase()}`}>
                  ● {lightColor}
                </span>
              </div>

              <div className="video-viewport">
                {imgBase64 ? (
                  <img
                    src={`data:image/jpeg;base64,${imgBase64}`}
                    alt={`Approach ${lane}`}
                    className="feed-img"
                  />
                ) : (
                  <div className="placeholder-box">Loading camera feed...</div>
                )}
              </div>

              <div className="camera-stats">
                <div className="stat-item">
                  <span className="stat-label">Vehicles (C)</span>
                  <span className="stat-val">{stdCount}</span>
                </div>
                <div className={`stat-item ${emgCount > 0 ? 'highlight-emg' : ''}`}>
                  <span className="stat-label">Emergency (E)</span>
                  <span className="stat-val">{emgCount}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">dC/dt (LSTM)</span>
                  <span className="stat-val">{growthRate > 0 ? `+${growthRate}` : growthRate}/s</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Real-time Telemetry & State Machine Transitions */}
      {data && (
        <div className="telemetry-section">
          {/* Feature Vector Table */}
          <div className="telemetry-card">
            <h3><Layers size={18} /> Live Feature Vector X_t</h3>
            <div className="xt-grid">
              {Object.entries(data.x_t || {}).map(([key, val]) => (
                <div key={key} className={`xt-badge ${key.startsWith('E_') && val > 0 ? 'emg-bg' : ''}`}>
                  <span className="xt-k">{key}</span>
                  <span className="xt-v">{val}</span>
                </div>
              ))}
            </div>

            <h3 style={{ marginTop: '1.2rem' }}><Clock size={18} /> Predicted Queue Growth Rates (dC/dt)</h3>
            <div className="xt-grid">
              {Object.entries(data.dc_dt || {}).map(([key, val]) => (
                <div key={key} className="xt-badge">
                  <span className="xt-k">{key.replace('_dt', '')}</span>
                  <span className="xt-v">{val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* State Machine Transition Logs */}
          <div className="telemetry-card">
            <h3><AlertTriangle size={18} /> Controller Audit Logs</h3>
            <div className="log-list">
              {data.recent_transitions && data.recent_transitions.length > 0 ? (
                data.recent_transitions.map((t, idx) => (
                  <div key={idx} className="log-entry">
                    <span className="log-time">[{t.timestamp}s]</span>
                    <span className="log-states">{t.from_state} &rarr; {t.to_state}</span>
                    <span className="log-lane">Lane: {t.active_lane}</span>
                    <span className="log-reason">{t.reason}</span>
                  </div>
                ))
              ) : (
                <div className="log-empty">No transitions recorded yet.</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
