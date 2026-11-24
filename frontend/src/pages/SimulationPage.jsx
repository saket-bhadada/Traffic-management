import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Car,
  Clock,
  FastForward,
  Pause,
  Play,
  Plus,
  RotateCcw,
  ShieldAlert,
  Siren,
  StepForward,
  Zap,
} from 'lucide-react';
import PreemptionFlowchart from '../components/PreemptionFlowchart';
import QueueGrowthChart from '../components/QueueGrowthChart';
import ThreeIntersectionScene from '../components/ThreeIntersectionScene';
import {
  IntersectionSimulator,
  LANES,
  LANE_NAMES,
  LightColor,
  SignalState,
} from '../simulation/stateMachine';
import './SimulationPage.css';

const SPEED_OPTIONS = [0.5, 1, 2, 4];

const DEMAND_OPTIONS = [
  { id: 'manual', label: 'Manual Only' },
  { id: 'light', label: 'Light Traffic' },
  { id: 'normal', label: 'Normal Chowk' },
  { id: 'rush_hour', label: 'Rush Hour' },
];

const SCENARIO_PRESETS = [
  {
    id: 'truncate_gt4',
    title: 'Truncation (> 4s)',
    desc: 'Lane N has >4s green left; emergency on S triggers 2s truncated green -> 3s amber -> 1s all-red.',
  },
  {
    id: 'expire_le4',
    title: 'Normal Expiry (<= 4s)',
    desc: 'Lane N has 3s green left; emergency on W lets green finish naturally before amber/all-red.',
  },
  {
    id: 'same_lane',
    title: 'Same-Lane (j == i)',
    desc: 'Emergency arrives on active green Lane N; transitions straight to EMERGENCY_GREEN.',
  },
  {
    id: 'rush_hour',
    title: 'Rush Hour Burst',
    desc: 'Asymmetric N/S arterial platoons demonstrate dynamic adaptive green scaling vs 30s fixed timer.',
  },
  {
    id: 'car_count_switch',
    title: 'Car-Count Switch (6s Buffer)',
    desc: 'Lane N (10 cars) yields to heavy Lane S (20 cars) after 10s min-green + 6s buffer.',
  },
];

const BUFFER_OPTIONS = [4, 6, 8, 10, 12];
const MARGIN_OPTIONS = [3, 5, 8];

export default function SimulationPage() {
  const [sim] = useState(() => new IntersectionSimulator());
  const [snapshot, setSnapshot] = useState(() => sim.getSnapshot());
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [activeScenario, setActiveScenario] = useState(null);

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
    setActiveScenario(null);
    setSnapshot(sim.getSnapshot());
  };

  const handleDemandChange = (e) => {
    const mode = e.target.value;
    sim.setDemandMode(mode);
    setActiveScenario(null);
    setSnapshot(sim.getSnapshot());
  };

  const handleScenario = (scenarioId) => {
    sim.loadScenario(scenarioId);
    setActiveScenario(scenarioId);
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
    clockwiseLane,
    resumeLane,
    preemptTargetLane,
    postEmergencyClearing,
    remainingTime,
    elapsedInState,
    targetGreen,
    rawGreen,
    isEmergencyPreemption,
    queues,
    emergencies,
    dc_dt: dcDtMap,
    transitions,
    demandMode,
    config,
    carCountSwitch,
    metrics,
  } = snapshot;

  const activeC = queues[clockwiseLane] ?? 0;
  const activeDcDt = dcDtMap[`dC_${clockwiseLane}_dt`] ?? 0;
  const activePositiveDcDt = Math.max(activeDcDt, 0);
  const timerLabel =
    state === SignalState.EMERGENCY_GREEN ? 'HOLD (until E_j == 0)' : `${remainingTime.toFixed(1)}s`;
  const greenProgressPct =
    targetGreen > 0 ? Math.min(100, Math.round((elapsedInState / targetGreen) * 100)) : 0;

  const isDemandBuffering = !isEmergencyPreemption && Boolean(carCountSwitch.targetLane);
  const isDemandClearing = !isEmergencyPreemption && Boolean(carCountSwitch.activeDemandSwitch);
  const bufferPct =
    carCountSwitch.bufferDuration > 0
      ? Math.min(100, Math.round((carCountSwitch.bufferElapsed / carCountSwitch.bufferDuration) * 100))
      : 100;
  const minGreenElapsed = Math.min(config.minGreen, elapsedInState);
  const activeAspect = (snapshot.lights[activeLane] ?? LightColor.RED).toLowerCase();
  const hasVehicleData =
    snapshot.timestamp > 0 ||
    metrics.adaptive.totalQueued > 0 ||
    metrics.adaptive.throughput > 0;
  const hasQueueGrowthData =
    snapshot.timestamp > 0 || Object.values(dcDtMap).some((v) => Number(v) !== 0);

  let bannerClass = 'alert-normal';
  if (isEmergencyPreemption) {
    bannerClass = 'alert-emergency';
  } else if (isDemandBuffering || isDemandClearing || state === SignalState.AMBER) {
    bannerClass = 'alert-amber';
  }

  return (
    <div className="sim-page">
      {/* Header & Transport Controls */}
      <div className="sim-toolbar">
        <div className="sim-title-block">
          <span className="sim-op-code">SIM 1Hz</span>
          <div>
            <h2>Signal Simulation</h2>
            <p className="subtitle">
              1s discrete-time state machine &middot; Adaptive timer vs. 30s fixed baseline
            </p>
          </div>
        </div>

        <div className="sim-controls-group">
          <button
            type="button"
            className={`btn ${isPlaying ? 'btn-active-play' : 'btn-primary-sim'}`}
            onClick={() => setIsPlaying((prev) => !prev)}
          >
            {isPlaying ? <Pause size={16} strokeWidth={1.75} /> : <Play size={16} strokeWidth={1.75} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>

          <button
            type="button"
            className="btn"
            onClick={handleStep}
            disabled={isPlaying}
            title={isPlaying ? 'Pause simulation to step 1s manually' : 'Advance simulation by 1 second'}
          >
            <StepForward size={16} strokeWidth={1.75} /> Step (+1s)
          </button>

          <button type="button" className="btn btn-danger" onClick={handleReset}>
            <RotateCcw size={16} strokeWidth={1.75} /> Reset
          </button>

          <div className="sim-speed-pills" role="group" aria-label="Simulation speed">
            <FastForward size={14} strokeWidth={1.75} className="speed-icon" />
            {SPEED_OPTIONS.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`speed-pill ${speed === opt ? 'active' : ''}`}
                onClick={() => setSpeed(opt)}
              >
                {opt}x
              </button>
            ))}
          </div>

          <label className="sim-demand-select">
            <span>Demand:</span>
            <select value={demandMode} onChange={handleDemandChange}>
              {DEMAND_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {/* Live Status Banner */}
      <div className={`alert-banner ${bannerClass}`}>
        {isEmergencyPreemption ? (
          <>
            <ShieldAlert size={20} strokeWidth={1.75} className="banner-icon" />
            <div className="sim-banner-text">
              <strong>
                Preempt active: {LANE_NAMES[activeLane]} ({activeLane})
              </strong>
              <span>
                {' '}
                &middot; State: <b>{state}</b> &middot; <b>{timerLabel}</b>
                {preemptTargetLane ? ` · Target j: ${preemptTargetLane}` : ''}
                {postEmergencyClearing
                  ? ` · Clearing to Resume: ${resumeLane}`
                  : ` · Resume: ${resumeLane}`}
              </span>
            </div>
          </>
        ) : isDemandBuffering ? (
          <>
            <AlertTriangle size={20} strokeWidth={1.75} className="banner-icon" />
            <div className="sim-banner-text">
              <strong>
                Demand switch queued: {LANE_NAMES[carCountSwitch.targetLane]} (
                {carCountSwitch.targetLane})
              </strong>
              <span>
                {' '}
                &middot; C_{carCountSwitch.targetLane}={queues[carCountSwitch.targetLane] ?? 0} vs C_
                {activeLane}={queues[activeLane] ?? 0} &middot; Buffer:{' '}
                <b>{carCountSwitch.bufferRemaining.toFixed(1)}s</b> remaining &middot; Min green hold:{' '}
                <b>{carCountSwitch.minGreenRemaining.toFixed(1)}s</b>
              </span>
            </div>
          </>
        ) : isDemandClearing ? (
          <>
            <Car size={20} strokeWidth={1.75} className="banner-icon" />
            <div className="sim-banner-text">
              <strong>
                Demand switch active: {activeLane} &rarr; {resumeLane}
              </strong>
              <span>
                {' '}
                &middot; State: <b>{state}</b> &middot; <b>{timerLabel}</b>
              </span>
            </div>
          </>
        ) : (
          <>
            <Car size={20} strokeWidth={1.75} className="banner-icon" />
            <div className="sim-banner-text">
              <strong>
                {LANE_NAMES[activeLane]} ({activeLane}) {activeAspect}
              </strong>
              <span>
                {' '}
                &mdash; <b>{timerLabel}</b> &middot; State: <b>{state}</b> &middot; Next:{' '}
                <b>{resumeLane}</b>
              </span>
            </div>
          </>
        )}
      </div>

      {/* Guided Scenario Presets */}
      <div className="sim-scenarios-bar">
        <div className="sim-scenarios-heading">
          <Zap size={14} strokeWidth={1.75} />
          <span>Guided Scenarios</span>
        </div>
        <div className="sim-scenarios-grid">
          {SCENARIO_PRESETS.map((sc, idx) => (
            <button
              key={sc.id}
              type="button"
              className={`scenario-btn ${activeScenario === sc.id ? 'selected' : ''}`}
              onClick={() => handleScenario(sc.id)}
            >
              <span className="scenario-head">
                <span className="scenario-idx">0{idx + 1}</span>
                <span className="scenario-title">{sc.title}</span>
              </span>
              <span className="scenario-desc">{sc.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main 2-Column Stage: Left 3D Digital Twin + Injection | Right Flowchart + Formula */}
      <div className="sim-main-grid">
        <div className="sim-col-left">
          <ThreeIntersectionScene snapshot={snapshot} />

          {/* Interactive Lane Injection Strip */}
          <div className="sim-card sim-inject-card">
            <div className="sim-card-header">
              <div>
                <h3>Interactive Traffic &amp; Emergency Injection</h3>
                <p className="sim-card-sub">
                  Inject vehicles or trigger an ambulance on any approach to watch the controller react live
                </p>
              </div>
            </div>

            <div className="inject-grid">
              {LANES.map((lane) => {
                const qLen = queues[lane] ?? 0;
                const eLen = emergencies[lane] ?? 0;
                const light = snapshot.lights[lane];

                return (
                  <div
                    key={lane}
                    className={`inject-lane-box light-border-${light.toLowerCase()} ${
                      eLen > 0 ? 'has-emg' : ''
                    }`}
                  >
                    <div className="inject-lane-top">
                      <span className="inject-lane-name">{LANE_NAMES[lane]}</span>
                      <span className={`light-badge light-${light.toLowerCase()}`}>{light}</span>
                    </div>

                    <div className="inject-lane-counts">
                      <span>C_{lane}: <b>{qLen}</b></span>
                      <span className={eLen > 0 ? 'emg-count-hot' : ''}>
                        E_{lane}: <b>{eLen}</b>
                      </span>
                    </div>

                    <div className="inject-actions">
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => handleInjectCars(lane, 5)}
                      >
                        <Plus size={14} strokeWidth={1.75} /> 5 Cars
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm btn-emg"
                        onClick={() => handleInjectEmergency(lane)}
                      >
                        <Siren size={14} strokeWidth={1.75} /> Emergency
                      </button>
                      {eLen > 0 && (
                        <button
                          type="button"
                          className="btn btn-sm btn-clear-emg"
                          onClick={() => handleClearEmergency(lane)}
                        >
                          Clear E_{lane}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="sim-col-right">
          <PreemptionFlowchart snapshot={snapshot} />

          {/* Adaptive Green Formula Live Inspector */}
          <div className="sim-card sim-formula-card">
            <div className="sim-card-header">
              <div>
                <h3><Clock size={16} strokeWidth={1.75} /> Adaptive Green Timer Formula</h3>
                <p className="sim-card-sub">
                  Recomputed every 1s tick during <code>ADAPTIVE_GREEN</code> on active clockwise lane {clockwiseLane}
                </p>
              </div>
            </div>

            <div className="formula-equation">
              <code>
                green = clamp({config.baseGreen} + {config.k1}&middot;C_{clockwiseLane} + {config.k2}&middot;max(dC_{clockwiseLane}/dt, 0), {config.minGreen}, {config.maxGreen})
              </code>
            </div>

            <div className="formula-substitution">
              <div className="formula-term">
                <span className="term-label">Base Green</span>
                <span className="term-val">{config.baseGreen.toFixed(1)}s</span>
              </div>
              <div className="formula-op">+</div>
              <div className="formula-term">
                <span className="term-label">Queue ({config.k1} &times; C_{clockwiseLane}={activeC})</span>
                <span className="term-val">{(config.k1 * activeC).toFixed(1)}s</span>
              </div>
              <div className="formula-op">+</div>
              <div className="formula-term">
                <span className="term-label">Growth ({config.k2} &times; {activePositiveDcDt.toFixed(2)})</span>
                <span className="term-val">{(config.k2 * activePositiveDcDt).toFixed(1)}s</span>
              </div>
              <div className="formula-op">=</div>
              <div className="formula-term term-result">
                <span className="term-label">Target (raw {rawGreen.toFixed(1)}s)</span>
                <span className="term-val">{targetGreen.toFixed(1)}s</span>
              </div>
            </div>

            <div className="formula-progress-wrap">
              <div className="formula-progress-labels">
                <span>
                  Elapsed in {state}: <b>{elapsedInState.toFixed(1)}s</b>
                </span>
                <span>
                  Remaining: <b>{timerLabel}</b>
                </span>
              </div>
              <div className="formula-progress-track">
                <div
                  className="formula-progress-fill"
                  style={{ width: `${state === SignalState.ADAPTIVE_GREEN ? greenProgressPct : 100}%` }}
                />
              </div>
            </div>

            {/* Car-Count Switching & Buffer Controls */}
            <div className="demand-controls-divider">
              <div className="demand-controls-header">
                <div>
                  <h4>Car-Count Switching &amp; Buffer Controls</h4>
                  <p className="sim-card-sub">
                    Switches to the busiest waiting lane only after min green ({config.minGreen.toFixed(0)}s) and sustained buffer hold
                  </p>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={carCountSwitch.enabled}
                  className={`btn btn-sm demand-toggle-btn ${carCountSwitch.enabled ? 'is-on' : 'is-off'}`}
                  onClick={() => handleCarCountConfig({ enabled: !carCountSwitch.enabled })}
                >
                  Car-Count Switching: {carCountSwitch.enabled ? 'ON' : 'OFF'}
                </button>
              </div>

              <div className="demand-controls-row">
                <label className="sim-demand-select">
                  <span>Switch Buffer:</span>
                  <select
                    aria-label="Switch Buffer"
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

                <label className="sim-demand-select">
                  <span>Car Count Margin:</span>
                  <select
                    aria-label="Car Count Margin"
                    value={carCountSwitch.margin}
                    disabled={!carCountSwitch.enabled}
                    onChange={(e) => handleCarCountConfig({ margin: Number(e.target.value) })}
                  >
                    {MARGIN_OPTIONS.map((m) => (
                      <option key={m} value={m}>
                        +{m} cars
                      </option>
                    ))}
                  </select>
                </label>

                <span className="demand-guard-pill">
                  Starvation guard: {carCountSwitch.maxLaneRedWait.toFixed(0)}s
                </span>
              </div>

              <div className="formula-progress-wrap demand-buffer-progress">
                <div className="formula-progress-labels">
                  <span>
                    {carCountSwitch.targetLane ? (
                      <>
                        Buffer &rarr; Lane <b>{carCountSwitch.targetLane}</b>:{' '}
                        <b>
                          {carCountSwitch.bufferElapsed.toFixed(1)}s /{' '}
                          {carCountSwitch.bufferDuration.toFixed(1)}s
                        </b>
                      </>
                    ) : (
                      <>
                        Buffer: <b>0.0s / {carCountSwitch.bufferDuration.toFixed(1)}s</b> (waiting for margin +{carCountSwitch.margin})
                      </>
                    )}
                  </span>
                  <span>
                    Min Green Hold:{' '}
                    <b>
                      {minGreenElapsed.toFixed(1)}s / {config.minGreen.toFixed(1)}s
                    </b>
                  </span>
                </div>
                <div className="formula-progress-track">
                  <div
                    className="formula-progress-fill demand-progress-fill"
                    style={{ width: `${carCountSwitch.targetLane ? bufferPct : 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Strip: Queue Growth Chart + Comparison Table + Phase Timeline Log */}
      <QueueGrowthChart
        dcDt={dcDtMap}
        hasData={hasQueueGrowthData}
        title="Per-lane queue growth (dC/dt)"
      />

      <div className="sim-bottom-grid">
        <div className="sim-card">
          <div className="sim-card-header">
            <div>
              <h3>Comparison vs. Fixed-Time (30s)</h3>
              <p className="sim-card-sub">
                Identical arrival stream &amp; 1.5 veh/s discharge rate
              </p>
            </div>
            <span className="fixed-status-pill">
              Fixed:{' '}
              <b className={`txt-${(metrics.fixed.fixedLights[metrics.fixed.fixedActiveLane] ?? LightColor.RED).toLowerCase()}`}>
                {metrics.fixed.fixedActiveLane} ({metrics.fixed.fixedRemainingTime.toFixed(0)}s)
              </b>
            </span>
          </div>

          <div className="bench-table-wrap">
            <table className="bench-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Fixed (30s)</th>
                  <th>Adaptive</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Avg wait time</td>
                  <td>{hasVehicleData ? `${metrics.fixed.avgWaitSec.toFixed(2)}s` : '--'}</td>
                  <td className="val-bold">
                    {hasVehicleData ? `${metrics.adaptive.avgWaitSec.toFixed(2)}s` : '--'}
                  </td>
                  <td
                    className={
                      hasVehicleData
                        ? metrics.adaptive.avgWaitSec <= metrics.fixed.avgWaitSec
                          ? 'delta-good'
                          : 'delta-warn'
                        : ''
                    }
                  >
                    {hasVehicleData
                      ? `${(metrics.adaptive.avgWaitSec - metrics.fixed.avgWaitSec).toFixed(2)}s`
                      : '--'}
                  </td>
                </tr>
                <tr>
                  <td>Idle green time</td>
                  <td>{snapshot.timestamp > 0 ? `${metrics.fixed.idleGreenSec.toFixed(1)}s` : '--'}</td>
                  <td className="val-bold">
                    {snapshot.timestamp > 0 ? `${metrics.adaptive.idleGreenSec.toFixed(1)}s` : '--'}
                  </td>
                  <td
                    className={
                      snapshot.timestamp > 0
                        ? metrics.adaptive.idleGreenSec <= metrics.fixed.idleGreenSec
                          ? 'delta-good'
                          : 'delta-warn'
                        : ''
                    }
                  >
                    {snapshot.timestamp > 0
                      ? `${(metrics.adaptive.idleGreenSec - metrics.fixed.idleGreenSec).toFixed(1)}s`
                      : '--'}
                  </td>
                </tr>
                <tr>
                  <td>Throughput (veh)</td>
                  <td>{snapshot.timestamp > 0 ? metrics.fixed.throughput : '--'}</td>
                  <td className="val-bold">
                    {snapshot.timestamp > 0 ? metrics.adaptive.throughput : '--'}
                  </td>
                  <td
                    className={
                      snapshot.timestamp > 0
                        ? metrics.adaptive.throughput >= metrics.fixed.throughput
                          ? 'delta-good'
                          : 'delta-warn'
                        : ''
                    }
                  >
                    {snapshot.timestamp > 0
                      ? `${metrics.adaptive.throughput - metrics.fixed.throughput >= 0 ? '+' : ''}${metrics.adaptive.throughput - metrics.fixed.throughput}`
                      : '--'}
                  </td>
                </tr>
                <tr>
                  <td>Emergency time-to-green</td>
                  <td>
                    {metrics.fixed.emgEvents > 0
                      ? `${metrics.fixed.emgAvgTimeToGreenSec.toFixed(2)}s`
                      : '--'}
                  </td>
                  <td className="val-bold">
                    {metrics.adaptive.emgEvents > 0
                      ? `${metrics.adaptive.emgAvgTimeToGreenSec.toFixed(2)}s`
                      : '--'}
                  </td>
                  <td
                    className={
                      metrics.adaptive.emgEvents > 0 &&
                      metrics.adaptive.emgAvgTimeToGreenSec <= metrics.fixed.emgAvgTimeToGreenSec
                        ? 'delta-good'
                        : ''
                    }
                  >
                    {metrics.adaptive.emgEvents > 0
                      ? `${(metrics.adaptive.emgAvgTimeToGreenSec - metrics.fixed.emgAvgTimeToGreenSec).toFixed(2)}s`
                      : '--'}
                  </td>
                </tr>
                <tr>
                  <td>Queued vehicles</td>
                  <td>{metrics.fixed.totalQueued}</td>
                  <td className="val-bold">{metrics.adaptive.totalQueued}</td>
                  <td className={metrics.adaptive.totalQueued <= metrics.fixed.totalQueued ? 'delta-good' : 'delta-warn'}>
                    {metrics.adaptive.totalQueued - metrics.fixed.totalQueued}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="sim-card">
          <div className="sim-card-header">
            <div>
              <h3><AlertTriangle size={16} strokeWidth={1.75} /> Phase Timeline &amp; Transition Log</h3>
              <p className="sim-card-sub">Latest state transitions (newest first)</p>
            </div>
            <span className="badge-neutral">{transitions.length} events</span>
          </div>

          <div className="log-list sim-log-list">
            {transitions.length > 0 ? (
              transitions.map((t, idx) => (
                <div key={`${t.timestamp}-${t.fromState}-${t.toState}-${idx}`} className="log-entry">
                  <span className="log-time">[{t.timestamp.toFixed(0)}s]</span>
                  <span className="log-states">
                    {t.fromState} &rarr; {t.toState}
                  </span>
                  <span className="log-lane">
                    {t.activeLane} &rarr; {t.nextLane}
                  </span>
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
  );
}
