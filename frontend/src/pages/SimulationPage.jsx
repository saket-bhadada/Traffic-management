import React, { useEffect, useState } from 'react';
import { Pause, Play, RotateCcw, ShieldAlert, StepForward } from 'lucide-react';
import ThreeIntersectionScene from '../components/ThreeIntersectionScene';
import {
  IntersectionSimulator,
  LANES,
  LightColor,
  SignalState,
} from '../simulation/stateMachine';
import './SimulationPage.css';

const SHORT_LANE = {
  N: 'North',
  E: 'East',
  S: 'South',
  W: 'West',
};

const SPEED_OPTIONS = [0.5, 1, 2, 4];

const DEMAND_OPTIONS = [
  { id: 'manual', label: 'Manual' },
  { id: 'light', label: 'Light' },
  { id: 'normal', label: 'Normal' },
  { id: 'rush_hour', label: 'Rush hour' },
];

const THRESHOLD_OPTIONS = [6, 8, 10, 12, 15, 20];
const BUFFER_OPTIONS = [4, 6, 8, 10, 12];

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
      return state;
  }
}

export default function SimulationPage({ headerLeft }) {
  const [sim] = useState(() => new IntersectionSimulator());
  const [snapshot, setSnapshot] = useState(() => sim.getSnapshot());
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const intervalMs = Math.round(1000 / speed);
    const timer = setInterval(() => {
      setSnapshot(sim.step());
    }, intervalMs);
    return () => clearInterval(timer);
  }, [isPlaying, speed, sim]);

  const handleStep = () => {
    setSnapshot(sim.step());
  };

  const handleReset = () => {
    sim.reset(sim.demandMode);
    setSnapshot(sim.getSnapshot());
  };

  const handleDemandChange = (e) => {
    sim.setDemandMode(e.target.value);
    setSnapshot(sim.getSnapshot());
  };

  const handleInjectCars = (lane, count = 5) => {
    sim.injectVehicles(lane, count);
    setSnapshot(sim.getSnapshot());
  };

  const handleInjectEmergency = (lane) => {
    sim.injectEmergency(lane);
    setSnapshot(sim.getSnapshot());
  };

  const handleClearEmergency = (lane) => {
    sim.clearEmergency(lane);
    setSnapshot(sim.getSnapshot());
  };

  const handleCarCountConfig = (patch) => {
    sim.setCarCountConfig(patch);
    setSnapshot(sim.getSnapshot());
  };

  const {
    state,
    activeLane,
    preemptTargetLane,
    remainingTime,
    elapsedInState,
    targetGreen,
    isEmergencyPreemption,
    queues,
    emergencies,
    demandMode,
    carCountSwitch,
  } = snapshot;

  const activeLight = snapshot.lights[activeLane] ?? LightColor.RED;
  const stateWord = formatStateLabel(state);
  const timerLabel =
    state === SignalState.EMERGENCY_GREEN
      ? 'HOLD'
      : `${Math.ceil(remainingTime)}s`;

  const totalPhaseDur =
    state === SignalState.ADAPTIVE_GREEN
      ? targetGreen
      : elapsedInState + remainingTime;
  const remainingPct =
    state === SignalState.EMERGENCY_GREEN
      ? 100
      : totalPhaseDur > 0
        ? Math.min(100, Math.max(0, Math.round((remainingTime / totalPhaseDur) * 100)))
        : 0;

  const progressColorClass =
    state === SignalState.EMERGENCY_GREEN || state === SignalState.ALL_RED
      ? 'bar-red'
      : state === SignalState.AMBER || state === SignalState.TRUNCATED_GREEN
        ? 'bar-amber'
        : 'bar-green';

  const isDemandBuffering = !isEmergencyPreemption && Boolean(carCountSwitch.targetLane);
  const bufferPct =
    carCountSwitch.bufferDuration > 0
      ? Math.min(100, Math.round((carCountSwitch.bufferElapsed / carCountSwitch.bufferDuration) * 100))
      : 100;

  return (
    <>
      <header className="topbar">
        {headerLeft}

        <div className="topbar-right">
          <button
            type="button"
            className="ghost-btn active"
            onClick={() => setIsPlaying((prev) => !prev)}
          >
            {isPlaying ? <Pause size={13} strokeWidth={2} /> : <Play size={13} strokeWidth={2} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>

          <button
            type="button"
            className="ghost-btn"
            onClick={handleStep}
            disabled={isPlaying}
            title={isPlaying ? 'Pause to step 1s' : 'Advance 1s'}
          >
            <StepForward size={13} strokeWidth={2} /> Step
          </button>

          <button type="button" className="ghost-btn" onClick={handleReset}>
            <RotateCcw size={13} strokeWidth={2} /> Reset
          </button>

          <span className="bar-sep" aria-hidden="true" />

          <div className="speed-strip" role="group" aria-label="Playback speed">
            {SPEED_OPTIONS.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`speed-item num ${speed === opt ? 'active' : ''}`}
                onClick={() => setSpeed(opt)}
              >
                {opt}x
              </button>
            ))}
          </div>

          <span className="bar-sep" aria-hidden="true" />

          <select
            aria-label="Traffic flow"
            className="bare-select"
            value={demandMode}
            onChange={handleDemandChange}
          >
            {DEMAND_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </header>

      <div className="split-workspace">
        <ThreeIntersectionScene snapshot={snapshot} />

        <aside className="side-pane">
          {isEmergencyPreemption && (
            <div className="emg-bar" role="status">
              <ShieldAlert size={15} strokeWidth={2} />
              <span>Emergency &middot; {SHORT_LANE[preemptTargetLane || activeLane]}</span>
            </div>
          )}

          {/* Active Signal */}
          <div className="pane-block">
            <div className="phase-row">
              <div className="phase-left">
                <span className={`sig-dot sig-dot-lg dot-${activeLight.toLowerCase()}`} />
                <span className="phase-name">
                  {state === SignalState.ALL_RED ? 'All red' : SHORT_LANE[activeLane]}
                </span>
                {state !== SignalState.ALL_RED && (
                  <span
                    className={`phase-state ${
                      isEmergencyPreemption
                        ? 'is-emg'
                        : activeLight === LightColor.AMBER
                          ? 'is-amber'
                          : ''
                    }`}
                  >
                    {stateWord}
                  </span>
                )}
              </div>
              <span className="phase-countdown num">{timerLabel}</span>
            </div>

            <div className="phase-line">
              <div
                className={`phase-line-fill ${progressColorClass}`}
                style={{ width: `${remainingPct}%` }}
              />
            </div>

            {isDemandBuffering && (
              <div className="priority-notice num">
                Priority &rarr; {SHORT_LANE[carCountSwitch.targetLane]} in{' '}
                {Math.ceil(carCountSwitch.bufferRemaining)}s
              </div>
            )}
          </div>

          {/* Per-Approach Counts & Triggers */}
          <div className="pane-block">
            <div className="approach-list">
              {LANES.map((lane) => {
                const qLen = queues[lane] ?? 0;
                const eLen = emergencies[lane] ?? 0;
                const light = snapshot.lights[lane];
                const reachedThreshold =
                  carCountSwitch.enabled &&
                  light !== LightColor.GREEN &&
                  qLen >= carCountSwitch.threshold;

                return (
                  <div key={lane} className="approach-row">
                    <div
                      className={`approach-id ${
                        light === LightColor.GREEN ? 'is-green' : ''
                      }`}
                    >
                      <span className={`sig-dot dot-${light.toLowerCase()}`} />
                      <span>{SHORT_LANE[lane]}</span>
                    </div>

                    <div
                      className={`approach-count num ${
                        reachedThreshold ? 'over-threshold' : ''
                      }`}
                    >
                      {qLen}
                      {eLen > 0 && <span className="emg-tag">+{eLen}E</span>}
                    </div>

                    <div className="row-actions">
                      <button
                        type="button"
                        className="row-btn"
                        onClick={() => handleInjectCars(lane, 5)}
                        title={`Add 5 cars to ${SHORT_LANE[lane]}`}
                      >
                        +5
                      </button>
                      <button
                        type="button"
                        className="row-btn row-btn-emg"
                        onClick={() => handleInjectEmergency(lane)}
                        title={`Trigger emergency on ${SHORT_LANE[lane]}`}
                      >
                        Emg
                      </button>
                      {eLen > 0 && (
                        <button
                          type="button"
                          className="row-btn row-btn-clear"
                          onClick={() => handleClearEmergency(lane)}
                        >
                          Clear
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Priority Threshold & Buffer */}
          <div className="pane-block">
            <button
              type="button"
              role="switch"
              aria-checked={carCountSwitch.enabled}
              className="switch-control"
              onClick={() => handleCarCountConfig({ enabled: !carCountSwitch.enabled })}
            >
              <span className={`switch-track ${carCountSwitch.enabled ? 'on' : ''}`}>
                <span className="switch-thumb" />
              </span>
              <span>Priority by count</span>
            </button>

            <div className="param-grid">
              <label className="param-field">
                <span>Threshold</span>
                <select
                  aria-label="Count threshold"
                  className="num"
                  value={carCountSwitch.threshold}
                  disabled={!carCountSwitch.enabled}
                  onChange={(e) => handleCarCountConfig({ threshold: Number(e.target.value) })}
                >
                  {THRESHOLD_OPTIONS.map((th) => (
                    <option key={th} value={th}>
                      {th} cars
                    </option>
                  ))}
                </select>
              </label>

              <label className="param-field">
                <span>Buffer</span>
                <select
                  aria-label="Switch buffer"
                  className="num"
                  value={carCountSwitch.bufferDuration}
                  disabled={!carCountSwitch.enabled}
                  onChange={(e) => handleCarCountConfig({ buffer: Number(e.target.value) })}
                >
                  {BUFFER_OPTIONS.map((sec) => (
                    <option key={sec} value={sec}>
                      {sec}s
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {carCountSwitch.enabled && carCountSwitch.targetLane && (
              <div className="buffer-live-bar">
                <div className="buffer-live-fill" style={{ width: `${bufferPct}%` }} />
              </div>
            )}
          </div>
        </aside>
      </div>
    </>
  );
}
