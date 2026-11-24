/**
 * Deterministic 4-Way Traffic Signal Controller, Fixed-Timer Baseline,
 * and 1-Second Intersection Simulator.
 *
 * Mirrors traffic_signal_ml/src/control/state_machine.py,
 * traffic_signal_ml/src/sim/baseline_fixed.py, and
 * traffic_signal_ml/configs/controller.yaml.
 */

export const LANES = ['N', 'E', 'S', 'W'];

export const LANE_NAMES = Object.freeze({
  N: 'North Approach',
  E: 'East Approach',
  S: 'South Approach',
  W: 'West Approach',
});

export const SignalState = Object.freeze({
  ADAPTIVE_GREEN: 'ADAPTIVE_GREEN',
  TRUNCATED_GREEN: 'TRUNCATED_GREEN',
  AMBER: 'AMBER',
  ALL_RED: 'ALL_RED',
  EMERGENCY_GREEN: 'EMERGENCY_GREEN',
});

export const LightColor = Object.freeze({
  GREEN: 'GREEN',
  AMBER: 'AMBER',
  RED: 'RED',
});

export const PreemptBranch = Object.freeze({
  NONE: 'NONE',
  TRUNCATE: 'TRUNCATE',
  EXPIRE_NORMAL: 'EXPIRE_NORMAL',
  SAME_LANE: 'SAME_LANE',
});

export const DEFAULT_CONFIG = Object.freeze({
  phaseOrder: ['N', 'E', 'S', 'W'],
  baseGreen: 15.0,
  k1: 1.5,
  k2: 4.0,
  minGreen: 10.0,
  maxGreen: 60.0,
  amberDuration: 3.0,
  allRedDuration: 1.0,
  preemptRemainingThreshold: 4.0,
  truncatedGreenDuration: 2.0,
  fixedGreenDuration: 30.0,
  saturationDischargeRate: 1.5,
  emergencyClearTicks: 3,
  carCountSwitchingEnabled: true,
  carCountPriorityThreshold: 10.0,
  carCountSwitchMargin: 5.0,
  carCountSwitchBuffer: 6.0,
  maxLaneRedWait: 120.0,
});

const DEMAND_RATES = Object.freeze({
  manual: { N: 0.0, E: 0.0, S: 0.0, W: 0.0 },
  light: { N: 0.14, E: 0.08, S: 0.12, W: 0.07 },
  normal: { N: 0.32, E: 0.15, S: 0.28, W: 0.13 },
  rush_hour: { N: 0.65, E: 0.22, S: 0.60, W: 0.20 },
});

function samplePoisson(lambda) {
  if (lambda <= 0) return 0;
  const limit = Math.exp(-lambda);
  let k = 0;
  let prod = 1.0;
  do {
    k += 1;
    prod *= Math.random();
  } while (prod > limit);
  return k - 1;
}

/**
 * Compute clamped adaptive green duration (seconds) for the active lane:
 * green = clamp(baseGreen + k1 * C_lane + k2 * max(dC/dt, 0), minGreen, maxGreen)
 */
export function computeAdaptiveGreen(cLane, dcDt, cfg = DEFAULT_CONFIG) {
  const base = Number(cfg.baseGreen);
  const k1 = Number(cfg.k1);
  const k2 = Number(cfg.k2);
  const minG = Number(cfg.minGreen);
  const maxG = Number(cfg.maxGreen);
  const raw = base + k1 * Number(cLane) + k2 * Math.max(Number(dcDt), 0.0);
  const clamped = Math.max(minG, Math.min(maxG, raw));
  return { raw, clamped };
}

/**
 * Explicit deterministic state machine for clockwise phase control and emergency preemption.
 */
export class PreemptionStateMachine {
  constructor(config = DEFAULT_CONFIG) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.reset();
  }

  reset() {
    this.state = SignalState.ADAPTIVE_GREEN;
    this.clockwiseIndex = 0;
    this.activeLane = this.config.phaseOrder[0];
    this.remainingTime = Number(this.config.baseGreen);
    this.elapsedInState = 0.0;
    this.preemptTargetLane = null;
    this.resumeClockwiseIndex = null;
    this.laneRedWait = createEmptyLaneMap(0.0);
    this.demandSwitchTarget = null;
    this.demandBufferElapsed = 0.0;
    this.nextDemandLane = null;
    this.activeDemandSwitch = false;
    this.transitions = [];

    // UI diagnostic tracking fields
    this.lastBranch = PreemptBranch.NONE;
    this.preemptOriginLane = null;
    this.lastRawGreen = Number(this.config.baseGreen);
    this.lastTargetGreen = Number(this.config.baseGreen);
    this.postEmergencyClearing = false;
  }

  _nextClockwiseIndex(idx) {
    return (idx + 1) % this.config.phaseOrder.length;
  }

  _clearDemandSwitch(clearNext = false) {
    this.demandSwitchTarget = null;
    this.demandBufferElapsed = 0.0;
    if (clearNext) {
      this.nextDemandLane = null;
      this.activeDemandSwitch = false;
    }
  }

  _updateLaneRedWaits(standardCounts, dt) {
    const isGreenState =
      this.state === SignalState.ADAPTIVE_GREEN ||
      this.state === SignalState.TRUNCATED_GREEN ||
      this.state === SignalState.EMERGENCY_GREEN;

    for (const lane of this.config.phaseOrder) {
      const isGreen = isGreenState && lane === this.activeLane;
      if (isGreen || Number(standardCounts[lane] ?? 0) <= 0) {
        this.laneRedWait[lane] = 0.0;
      } else {
        this.laneRedWait[lane] = (this.laneRedWait[lane] ?? 0.0) + dt;
      }
    }
  }

  _selectHighestCountLane(standardCounts, excludeLane = null) {
    const n = this.config.phaseOrder.length;
    const startIdx = (this.clockwiseIndex + 1) % n;
    const threshold = Number(this.config.carCountPriorityThreshold);
    const candidates = [];
    for (let offset = 0; offset < n; offset += 1) {
      const lane = this.config.phaseOrder[(startIdx + offset) % n];
      if (lane !== excludeLane && Number(standardCounts[lane] ?? 0) >= threshold) {
        candidates.push(lane);
      }
    }
    if (candidates.length === 0) return null;

    const starving = candidates.filter(
      (lane) => (this.laneRedWait[lane] ?? 0.0) >= this.config.maxLaneRedWait
    );
    if (starving.length > 0) {
      return starving.reduce((best, lane) => {
        const bestWait = this.laneRedWait[best] ?? 0.0;
        const laneWait = this.laneRedWait[lane] ?? 0.0;
        if (laneWait > bestWait) return lane;
        if (laneWait === bestWait && Number(standardCounts[lane] ?? 0) > Number(standardCounts[best] ?? 0)) {
          return lane;
        }
        return best;
      });
    }

    return candidates.reduce((best, lane) => (
      Number(standardCounts[lane] ?? 0) > Number(standardCounts[best] ?? 0) ? lane : best
    ));
  }

  _evaluateDemandSwitch(standardCounts, dt) {
    if (!this.config.carCountSwitchingEnabled) {
      this._clearDemandSwitch();
      return null;
    }

    const cActive = Number(standardCounts[this.activeLane] ?? 0);
    const bestWaiting = this._selectHighestCountLane(standardCounts, this.activeLane);
    if (bestWaiting === null) {
      this._clearDemandSwitch();
      return null;
    }

    const cBest = Number(standardCounts[bestWaiting] ?? 0);
    const bestWait = this.laneRedWait[bestWaiting] ?? 0.0;
    const threshold = Number(this.config.carCountPriorityThreshold);
    const conditionHolds =
      cActive < threshold ||
      cBest >= cActive + this.config.carCountSwitchMargin ||
      bestWait >= this.config.maxLaneRedWait;

    if (!conditionHolds) {
      this._clearDemandSwitch();
      return null;
    }

    this.demandSwitchTarget = bestWaiting;
    this.demandBufferElapsed += dt;
    return bestWaiting;
  }

  _selectEmergencyLane(emergencyCounts) {
    if ((emergencyCounts[this.activeLane] ?? 0) > 0) {
      return this.activeLane;
    }
    const n = this.config.phaseOrder.length;
    const startIdx = this.config.phaseOrder.indexOf(this.activeLane);
    for (let offset = 1; offset <= n; offset += 1) {
      const lane = this.config.phaseOrder[(startIdx + offset) % n];
      if ((emergencyCounts[lane] ?? 0) > 0) {
        return lane;
      }
    }
    return null;
  }

  _transition(timestamp, newState, newActiveLane, duration, reason) {
    const oldState = this.state;
    this.transitions.push({
      timestamp: Number(timestamp),
      fromState: oldState,
      toState: newState,
      activeLane: this.activeLane,
      nextLane: newActiveLane,
      reason,
    });
    this.state = newState;
    this.activeLane = newActiveLane;
    this.remainingTime = Number(duration);
    this.elapsedInState = 0.0;
    if (newState === SignalState.ADAPTIVE_GREEN || newState === SignalState.EMERGENCY_GREEN) {
      this.laneRedWait[newActiveLane] = 0.0;
    }
  }

  _beginPreemption(emgLane) {
    this._clearDemandSwitch(true);
    this.resumeClockwiseIndex = this._nextClockwiseIndex(this.clockwiseIndex);
    this.preemptTargetLane = emgLane;
    this.preemptOriginLane = this.activeLane;
    this.postEmergencyClearing = false;
  }

  step(timestamp, standardCounts, emergencyCounts, predictedDcDt, dt = 1.0) {
    this._updateLaneRedWaits(standardCounts, dt);
    const emgLane = this._selectEmergencyLane(emergencyCounts);

    // 1. Handle ADAPTIVE_GREEN
    if (this.state === SignalState.ADAPTIVE_GREEN) {
      if (this.preemptTargetLane === null) {
        const cActive = Number(standardCounts[this.activeLane] ?? 0);
        const dcActive = Number(predictedDcDt[this.activeLane] ?? 0);
        const { raw, clamped } = computeAdaptiveGreen(cActive, dcActive, this.config);
        this.lastRawGreen = raw;
        this.lastTargetGreen = clamped;
        this.remainingTime = Math.max(0.0, clamped - this.elapsedInState);
      }

      if (emgLane !== null && this.preemptTargetLane === null) {
        this._beginPreemption(emgLane);

        if (emgLane === this.activeLane) {
          this.lastBranch = PreemptBranch.SAME_LANE;
          this._transition(
            timestamp,
            SignalState.EMERGENCY_GREEN,
            emgLane,
            0.0,
            `Emergency detected on active green lane ${emgLane}; holding green until E_${emgLane}==0`
          );
          return this.state;
        }

        if (this.remainingTime > this.config.preemptRemainingThreshold) {
          this.lastBranch = PreemptBranch.TRUNCATE;
          this._transition(
            timestamp,
            SignalState.TRUNCATED_GREEN,
            this.activeLane,
            this.config.truncatedGreenDuration,
            `Emergency on lane ${emgLane}; remaining green (${this.remainingTime.toFixed(1)}s) > ${this.config.preemptRemainingThreshold.toFixed(1)}s -> truncating green to ${this.config.truncatedGreenDuration.toFixed(1)}s`
          );
          return this.state;
        }

        this.lastBranch = PreemptBranch.EXPIRE_NORMAL;
        this.transitions.push({
          timestamp: Number(timestamp),
          fromState: SignalState.ADAPTIVE_GREEN,
          toState: SignalState.ADAPTIVE_GREEN,
          activeLane: this.activeLane,
          nextLane: emgLane,
          reason: `Emergency on lane ${emgLane}; remaining green (${this.remainingTime.toFixed(1)}s) <= ${this.config.preemptRemainingThreshold.toFixed(1)}s -> letting green expire normally`,
        });
      }

      const demandCandidate =
        this.preemptTargetLane === null
          ? this._evaluateDemandSwitch(standardCounts, dt)
          : null;

      this.elapsedInState += dt;
      this.remainingTime = Math.max(0.0, this.remainingTime - dt);

      if (
        demandCandidate !== null &&
        this.preemptTargetLane === null &&
        this.elapsedInState >= this.config.minGreen &&
        this.demandBufferElapsed >= this.config.carCountSwitchBuffer
      ) {
        const cActive = Number(standardCounts[this.activeLane] ?? 0);
        const cBest = Number(standardCounts[demandCandidate] ?? 0);
        this.nextDemandLane = demandCandidate;
        this.activeDemandSwitch = true;
        this._clearDemandSwitch(false);
        this._transition(
          timestamp,
          SignalState.AMBER,
          this.activeLane,
          this.config.amberDuration,
          `Car-count switch buffer (${this.config.carCountSwitchBuffer.toFixed(1)}s) elapsed on ${this.activeLane} -> switching to lane ${demandCandidate} (C_${demandCandidate}=${cBest.toFixed(0)}, C_${this.activeLane}=${cActive.toFixed(0)})`
        );
        return this.state;
      }

      if (this.remainingTime <= 0.0) {
        this._clearDemandSwitch(false);
        const reason =
          this.preemptTargetLane !== null
            ? `Preemption expiry on ${this.activeLane} for emergency on ${this.preemptTargetLane}`
            : `Adaptive green expired normally on lane ${this.activeLane}`;
        this._transition(
          timestamp,
          SignalState.AMBER,
          this.activeLane,
          this.config.amberDuration,
          reason
        );
      }
      return this.state;
    }

    // 2. Handle TRUNCATED_GREEN
    if (this.state === SignalState.TRUNCATED_GREEN) {
      this.elapsedInState += dt;
      this.remainingTime = Math.max(0.0, this.remainingTime - dt);
      if (this.remainingTime <= 0.0) {
        this._transition(
          timestamp,
          SignalState.AMBER,
          this.activeLane,
          this.config.amberDuration,
          `Truncated green expired on ${this.activeLane}; entering AMBER before emergency lane ${this.preemptTargetLane}`
        );
      }
      return this.state;
    }

    // 3. Handle AMBER
    if (this.state === SignalState.AMBER) {
      if (emgLane !== null && this.preemptTargetLane === null) {
        this._beginPreemption(emgLane);
        this.lastBranch = PreemptBranch.EXPIRE_NORMAL;
      }

      this.elapsedInState += dt;
      this.remainingTime = Math.max(0.0, this.remainingTime - dt);
      if (this.remainingTime <= 0.0) {
        this._transition(
          timestamp,
          SignalState.ALL_RED,
          this.activeLane,
          this.config.allRedDuration,
          `Amber expired on ${this.activeLane}; entering ALL_RED clearance`
        );
      }
      return this.state;
    }

    // 4. Handle ALL_RED
    if (this.state === SignalState.ALL_RED) {
      if (emgLane !== null && this.preemptTargetLane === null) {
        this._beginPreemption(emgLane);
        this.lastBranch = PreemptBranch.EXPIRE_NORMAL;
      }

      this.elapsedInState += dt;
      this.remainingTime = Math.max(0.0, this.remainingTime - dt);
      if (this.remainingTime <= 0.0) {
        if (this.preemptTargetLane !== null) {
          const target = this.preemptTargetLane;
          this.postEmergencyClearing = false;
          this._transition(
            timestamp,
            SignalState.EMERGENCY_GREEN,
            target,
            0.0,
            `ALL_RED expired; granting EMERGENCY_GREEN to lane ${target}`
          );
        } else {
          let reasonPrefix;
          if (this.resumeClockwiseIndex !== null) {
            this.clockwiseIndex = this.resumeClockwiseIndex;
            this.resumeClockwiseIndex = null;
            this._clearDemandSwitch(true);
            reasonPrefix = 'Resuming clockwise cycle after preemption at phase';
          } else if (this.config.carCountSwitchingEnabled) {
            let chosen = null;
            if (
              this.nextDemandLane !== null &&
              Number(standardCounts[this.nextDemandLane] ?? 0) >= this.config.carCountPriorityThreshold
            ) {
              chosen = this.nextDemandLane;
            } else {
              chosen = this._selectHighestCountLane(standardCounts, this.activeLane);
            }
            this._clearDemandSwitch(true);
            if (chosen !== null) {
              this.clockwiseIndex = this.config.phaseOrder.indexOf(chosen);
              const cChosen = Number(standardCounts[chosen] ?? 0);
              reasonPrefix = `Switching by car-count demand (C_${chosen}=${cChosen.toFixed(0)}) to phase`;
            } else {
              this.clockwiseIndex = this._nextClockwiseIndex(this.clockwiseIndex);
              reasonPrefix = 'Advancing clockwise cycle to phase';
            }
          } else {
            this._clearDemandSwitch(true);
            this.clockwiseIndex = this._nextClockwiseIndex(this.clockwiseIndex);
            reasonPrefix = 'Advancing clockwise cycle to phase';
          }
          this.lastBranch = PreemptBranch.NONE;
          this.preemptOriginLane = null;
          this.postEmergencyClearing = false;

          const nextLane = this.config.phaseOrder[this.clockwiseIndex];
          const cNext = Number(standardCounts[nextLane] ?? 0);
          const dcNext = Number(predictedDcDt[nextLane] ?? 0);
          const { raw, clamped } = computeAdaptiveGreen(cNext, dcNext, this.config);
          this.lastRawGreen = raw;
          this.lastTargetGreen = clamped;
          this._transition(
            timestamp,
            SignalState.ADAPTIVE_GREEN,
            nextLane,
            clamped,
            `${reasonPrefix} ${nextLane} (initial adaptive green=${clamped.toFixed(1)}s)`
          );
        }
      }
      return this.state;
    }

    // 5. Handle EMERGENCY_GREEN
    if (this.state === SignalState.EMERGENCY_GREEN) {
      this.elapsedInState += dt;
      const ej = Number(emergencyCounts[this.activeLane] ?? 0);
      if (ej <= 0.0) {
        const clearedLane = this.activeLane;
        this.preemptTargetLane = null;
        if (this.resumeClockwiseIndex === null) {
          this.resumeClockwiseIndex = this._nextClockwiseIndex(this.clockwiseIndex);
        }
        this.postEmergencyClearing = true;
        const resumeLane = this.config.phaseOrder[this.resumeClockwiseIndex];
        this._transition(
          timestamp,
          SignalState.AMBER,
          clearedLane,
          this.config.amberDuration,
          `Emergency cleared on lane ${clearedLane} (E_${clearedLane}==0); transitioning via AMBER/ALL_RED to resume clockwise phase ${resumeLane}`
        );
      }
      return this.state;
    }

    return this.state;
  }

  getLaneLights() {
    const lights = {};
    for (const lane of this.config.phaseOrder) {
      lights[lane] = LightColor.RED;
    }
    if (
      this.state === SignalState.ADAPTIVE_GREEN ||
      this.state === SignalState.TRUNCATED_GREEN ||
      this.state === SignalState.EMERGENCY_GREEN
    ) {
      lights[this.activeLane] = LightColor.GREEN;
    } else if (this.state === SignalState.AMBER) {
      lights[this.activeLane] = LightColor.AMBER;
    }
    return lights;
  }
}

/**
 * Fixed-timer clockwise baseline controller (30s green, 3s amber, 1s all-red).
 */
export class FixedTimerBaselineController {
  constructor(config = DEFAULT_CONFIG) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.reset();
  }

  reset() {
    this.state = SignalState.ADAPTIVE_GREEN;
    this.clockwiseIndex = 0;
    this.activeLane = this.config.phaseOrder[0];
    this.remainingTime = Number(this.config.fixedGreenDuration);
  }

  step(dt = 1.0) {
    this.remainingTime = Math.max(0.0, this.remainingTime - dt);
    if (this.remainingTime <= 0.0) {
      if (this.state === SignalState.ADAPTIVE_GREEN) {
        this.state = SignalState.AMBER;
        this.remainingTime = Number(this.config.amberDuration);
      } else if (this.state === SignalState.AMBER) {
        this.state = SignalState.ALL_RED;
        this.remainingTime = Number(this.config.allRedDuration);
      } else if (this.state === SignalState.ALL_RED) {
        this.clockwiseIndex = (this.clockwiseIndex + 1) % this.config.phaseOrder.length;
        this.activeLane = this.config.phaseOrder[this.clockwiseIndex];
        this.state = SignalState.ADAPTIVE_GREEN;
        this.remainingTime = Number(this.config.fixedGreenDuration);
      }
    }
    return this.getLaneLights();
  }

  getLaneLights() {
    const lights = {};
    for (const lane of this.config.phaseOrder) {
      lights[lane] = LightColor.RED;
    }
    if (this.state === SignalState.ADAPTIVE_GREEN) {
      lights[this.activeLane] = LightColor.GREEN;
    } else if (this.state === SignalState.AMBER) {
      lights[this.activeLane] = LightColor.AMBER;
    }
    return lights;
  }
}

function createEmptyLaneMap(initialValue) {
  return { N: initialValue, E: initialValue, S: initialValue, W: initialValue };
}

function createEmptyLaneArrays() {
  return { N: [], E: [], S: [], W: [] };
}

function createSideState() {
  return {
    queues: createEmptyLaneArrays(),
    emergencies: createEmptyLaneArrays(),
    allEmergencyRecords: [],
    dischargeCarry: createEmptyLaneMap(0.0),
    completedWaitSum: 0.0,
    completedCount: 0,
    idleGreenSec: 0.0,
    totalGreenSec: 0.0,
    throughput: 0,
  };
}

function computeAvgWait(side, currentTime) {
  let activeWaitSum = 0.0;
  let queuedCount = 0;
  for (const lane of LANES) {
    for (const arrSec of side.queues[lane]) {
      activeWaitSum += Math.max(0.0, currentTime - arrSec);
      queuedCount += 1;
    }
  }
  const totalVehicles = side.completedCount + queuedCount;
  if (totalVehicles === 0) return 0.0;
  return (side.completedWaitSum + activeWaitSum) / totalVehicles;
}

function computeAvgEmergencyTimeToGreen(records, currentTime) {
  if (records.length === 0) return 0.0;
  let sum = 0.0;
  for (const rec of records) {
    const granted = rec.greenGrantedSec !== null ? rec.greenGrantedSec : currentTime;
    sum += Math.max(0.0, granted - rec.arrivalSec);
  }
  return sum / records.length;
}

function dischargeSide(t, lights, side, rate, clearTicks) {
  const departing = createEmptyLaneMap(0);
  const emergencyDeparting = createEmptyLaneMap(0);

  for (const lane of LANES) {
    if (lights[lane] === LightColor.GREEN) {
      side.totalGreenSec += 1.0;

      if (side.queues[lane].length === 0 && side.emergencies[lane].length === 0) {
        side.idleGreenSec += 1.0;
        side.dischargeCarry[lane] = 0.0;
        continue;
      }

      if (side.emergencies[lane].length > 0) {
        const remaining = [];
        for (const emg of side.emergencies[lane]) {
          if (emg.greenGrantedSec === null) {
            emg.greenGrantedSec = t;
          }
          emg.greenTicksServed += 1;
          if (emg.greenTicksServed >= clearTicks) {
            emg.clearedSec = t + 1;
            emergencyDeparting[lane] += 1;
          } else {
            remaining.push(emg);
          }
        }
        side.emergencies[lane] = remaining;
      }

      const capacity = side.dischargeCarry[lane] + rate;
      const nDepart = Math.min(Math.floor(capacity), side.queues[lane].length);
      side.dischargeCarry[lane] = capacity - Math.floor(capacity);

      for (let i = 0; i < nDepart; i += 1) {
        const arrT = side.queues[lane].shift();
        side.completedWaitSum += Math.max(0.0, t - arrT);
        side.completedCount += 1;
        side.throughput += 1;
        departing[lane] += 1;
      }
    } else {
      side.dischargeCarry[lane] = 0.0;
    }
  }

  return { departing, emergencyDeparting };
}

/**
 * Runs Adaptive+Preemption and Fixed-Timer Baseline in lockstep on identical arrivals.
 */
export class IntersectionSimulator {
  constructor(config = DEFAULT_CONFIG) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.sm = new PreemptionStateMachine(this.config);
    this.fixed = new FixedTimerBaselineController(this.config);
    this.demandMode = 'normal';
    this.reset('normal');
  }

  reset(demandMode = this.demandMode) {
    this.demandMode = demandMode;
    this.timestamp = 0;
    this.sm.reset();
    this.fixed.reset();
    this.adaptiveSide = createSideState();
    this.fixedSide = createSideState();
    this.prevCounts = createEmptyLaneMap(0.0);
    this.dcDt = createEmptyLaneMap(0.0);
    this.lastDischarging = createEmptyLaneMap(0);
    this.lastEmergencyDischarging = createEmptyLaneMap(0);
    this.nextAutoEmergencyAt = 40;
  }

  setDemandMode(mode) {
    if (Object.prototype.hasOwnProperty.call(DEMAND_RATES, mode)) {
      this.demandMode = mode;
    }
  }

  setCarCountConfig({ enabled, threshold, buffer, margin } = {}) {
    const nextConfig = { ...this.config };
    if (typeof enabled === 'boolean') {
      nextConfig.carCountSwitchingEnabled = enabled;
    }
    if (threshold !== undefined) {
      nextConfig.carCountPriorityThreshold = Math.max(1.0, Number(threshold));
    }
    if (buffer !== undefined) {
      nextConfig.carCountSwitchBuffer = Math.max(0.0, Number(buffer));
    }
    if (margin !== undefined) {
      nextConfig.carCountSwitchMargin = Math.max(1.0, Number(margin));
    }
    this.config = nextConfig;
    this.sm.config = nextConfig;
    if (!nextConfig.carCountSwitchingEnabled) {
      this.sm._clearDemandSwitch(true);
    } else if (this.sm.demandSwitchTarget !== null) {
      const targetLen = this.adaptiveSide.queues[this.sm.demandSwitchTarget]?.length ?? 0;
      if (targetLen < nextConfig.carCountPriorityThreshold) {
        this.sm._clearDemandSwitch(true);
      }
    }
  }

  injectVehicles(lane, count = 5) {
    if (!LANES.includes(lane) || count <= 0) return;
    for (let i = 0; i < count; i += 1) {
      this.adaptiveSide.queues[lane].push(this.timestamp);
      this.fixedSide.queues[lane].push(this.timestamp);
    }
  }

  injectEmergency(lane) {
    if (!LANES.includes(lane)) return;
    const makeRec = () => ({
      lane,
      arrivalSec: this.timestamp,
      greenGrantedSec: null,
      clearedSec: null,
      greenTicksServed: 0,
    });
    const recA = makeRec();
    const recF = makeRec();
    this.adaptiveSide.emergencies[lane].push(recA);
    this.adaptiveSide.allEmergencyRecords.push(recA);
    this.fixedSide.emergencies[lane].push(recF);
    this.fixedSide.allEmergencyRecords.push(recF);
  }

  clearEmergency(lane) {
    if (!LANES.includes(lane)) return;
    for (const emg of this.adaptiveSide.emergencies[lane]) {
      emg.clearedSec = this.timestamp;
    }
    this.adaptiveSide.emergencies[lane] = [];
  }

  loadScenario(scenarioId) {
    if (scenarioId === 'truncate_gt4') {
      this.reset('manual');
      this.injectVehicles('N', 4);
      this.injectVehicles('E', 2);
      this.injectVehicles('S', 3);
      this.injectVehicles('W', 1);
      // Sync prevCounts so dC/dt starts at 0 and remaining green is ~19s (> 4s threshold)
      this.prevCounts = { N: 4, E: 2, S: 3, W: 1 };
      this.sm.elapsedInState = 2.0;
      const { raw, clamped } = computeAdaptiveGreen(4, 0, this.config);
      this.sm.lastRawGreen = raw;
      this.sm.lastTargetGreen = clamped;
      this.sm.remainingTime = Math.max(0.0, clamped - this.sm.elapsedInState);
      this.injectEmergency('S');
      return;
    }

    if (scenarioId === 'expire_le4') {
      this.reset('manual');
      // 0 vehicles on N -> targetGreen = 15.0s; elapsed = 12.0s -> remaining = 3.0s (<= 4.0s)
      this.injectVehicles('E', 3);
      this.injectVehicles('W', 4);
      this.prevCounts = { N: 0, E: 3, S: 0, W: 4 };
      this.sm.elapsedInState = 12.0;
      this.sm.lastRawGreen = this.config.baseGreen;
      this.sm.lastTargetGreen = this.config.baseGreen;
      this.sm.remainingTime = 3.0;
      this.injectEmergency('W');
      return;
    }

    if (scenarioId === 'same_lane') {
      this.reset('manual');
      this.injectVehicles('N', 3);
      this.injectVehicles('E', 4);
      this.prevCounts = { N: 3, E: 4, S: 0, W: 0 };
      const { raw, clamped } = computeAdaptiveGreen(3, 0, this.config);
      this.sm.lastRawGreen = raw;
      this.sm.lastTargetGreen = clamped;
      this.sm.remainingTime = clamped;
      this.injectEmergency('N');
      return;
    }

    if (scenarioId === 'rush_hour') {
      this.reset('rush_hour');
      this.injectVehicles('N', 12);
      this.injectVehicles('E', 2);
      this.injectVehicles('S', 18);
      this.injectVehicles('W', 3);
      this.prevCounts = { N: 12, E: 2, S: 18, W: 3 };
      const { raw, clamped } = computeAdaptiveGreen(12, 0, this.config);
      this.sm.lastRawGreen = raw;
      this.sm.lastTargetGreen = clamped;
      this.sm.remainingTime = clamped;
      this.nextAutoEmergencyAt = 45;
      return;
    }

    if (scenarioId === 'car_count_switch') {
      this.reset('manual');
      this.setCarCountConfig({ enabled: true, threshold: 10.0, buffer: 6.0, margin: 5.0 });
      this.injectVehicles('N', 8);
      this.injectVehicles('E', 2);
      this.injectVehicles('S', 14);
      this.injectVehicles('W', 2);
      this.prevCounts = { N: 8, E: 2, S: 14, W: 2 };
      this.sm.elapsedInState = this.config.minGreen;
      const { raw, clamped } = computeAdaptiveGreen(8, 0, this.config);
      this.sm.lastRawGreen = raw;
      this.sm.lastTargetGreen = clamped;
      this.sm.remainingTime = Math.max(0.0, clamped - this.sm.elapsedInState);
    }
  }

  _generateAutoArrivals() {
    if (this.demandMode === 'manual') return;
    const rates = DEMAND_RATES[this.demandMode] ?? DEMAND_RATES.normal;
    for (const lane of LANES) {
      const n = samplePoisson(rates[lane]);
      if (n > 0) {
        this.injectVehicles(lane, n);
      }
    }

    if (
      (this.demandMode === 'normal' || this.demandMode === 'rush_hour') &&
      this.timestamp >= this.nextAutoEmergencyAt
    ) {
      const hasActiveEmg = LANES.some((l) => this.adaptiveSide.emergencies[l].length > 0);
      if (!hasActiveEmg) {
        const nonActiveLanes = LANES.filter((l) => l !== this.sm.activeLane);
        const chosen = nonActiveLanes[Math.floor(Math.random() * nonActiveLanes.length)];
        this.injectEmergency(chosen);
        this.nextAutoEmergencyAt = this.timestamp + 38 + Math.floor(Math.random() * 20);
      }
    }
  }

  step() {
    const t = this.timestamp;
    this._generateAutoArrivals();

    const stdCounts = {};
    const emgCounts = {};
    for (const lane of LANES) {
      stdCounts[lane] = this.adaptiveSide.queues[lane].length;
      emgCounts[lane] = this.adaptiveSide.emergencies[lane].length;
    }

    for (const lane of LANES) {
      const rawDelta = stdCounts[lane] - this.prevCounts[lane];
      this.dcDt[lane] = t === 0 && rawDelta === 0
        ? 0.0
        : Number((0.5 * rawDelta + 0.5 * this.dcDt[lane]).toFixed(2));
      this.prevCounts[lane] = stdCounts[lane];
    }

    // Record green_granted_sec if an emergency vehicle is already on a currently GREEN lane
    const preLights = this.sm.getLaneLights();
    const fixedPreLights = this.fixed.getLaneLights();
    for (const lane of LANES) {
      if (preLights[lane] === LightColor.GREEN) {
        for (const emg of this.adaptiveSide.emergencies[lane]) {
          if (emg.greenGrantedSec === null) emg.greenGrantedSec = t;
        }
      }
      if (fixedPreLights[lane] === LightColor.GREEN) {
        for (const emg of this.fixedSide.emergencies[lane]) {
          if (emg.greenGrantedSec === null) emg.greenGrantedSec = t;
        }
      }
    }

    this.sm.step(t, stdCounts, emgCounts, this.dcDt, 1.0);
    const adaptiveLights = this.sm.getLaneLights();
    const fixedLights = this.fixed.step(1.0);

    const adaptDischarge = dischargeSide(
      t,
      adaptiveLights,
      this.adaptiveSide,
      this.config.saturationDischargeRate,
      this.config.emergencyClearTicks
    );
    dischargeSide(
      t,
      fixedLights,
      this.fixedSide,
      this.config.saturationDischargeRate,
      this.config.emergencyClearTicks
    );

    this.lastDischarging = adaptDischarge.departing;
    this.lastEmergencyDischarging = adaptDischarge.emergencyDeparting;
    this.timestamp = t + 1;

    return this.getSnapshot();
  }

  getSnapshot() {
    const queues = {};
    const emergencies = {};
    let totalQueuedAdaptive = 0;
    let totalQueuedFixed = 0;

    for (const lane of LANES) {
      queues[lane] = this.adaptiveSide.queues[lane].length;
      emergencies[lane] = this.adaptiveSide.emergencies[lane].length;
      totalQueuedAdaptive += queues[lane];
      totalQueuedFixed += this.fixedSide.queues[lane].length;
    }

    const clockwiseLane = this.config.phaseOrder[this.sm.clockwiseIndex];
    const resumeIdx =
      this.sm.resumeClockwiseIndex !== null
        ? this.sm.resumeClockwiseIndex
        : (this.sm.clockwiseIndex + 1) % this.config.phaseOrder.length;
    const pendingPriorityLane = this.config.carCountSwitchingEnabled
      ? this.sm.nextDemandLane ??
        this.sm.demandSwitchTarget ??
        this.sm._selectHighestCountLane(queues, this.sm.activeLane)
      : null;
    const resumeLane =
      this.sm.resumeClockwiseIndex !== null
        ? this.config.phaseOrder[resumeIdx]
        : pendingPriorityLane ?? this.config.phaseOrder[resumeIdx];

    const bufferDuration = Number(this.config.carCountSwitchBuffer);
    const bufferElapsed = Number(this.sm.demandBufferElapsed.toFixed(1));
    const bufferRemaining = Number(Math.max(0.0, bufferDuration - this.sm.demandBufferElapsed).toFixed(1));
    const minGreenRemaining = Number(
      Math.max(0.0, Number(this.config.minGreen) - this.sm.elapsedInState).toFixed(1)
    );
    const laneRedWait = {};
    for (const lane of LANES) {
      laneRedWait[lane] = Number((this.sm.laneRedWait[lane] ?? 0.0).toFixed(1));
    }
    const hasAnyEmergency = LANES.some((lane) => emergencies[lane] > 0);
    const isEmergencyPreemption =
      this.sm.state === SignalState.TRUNCATED_GREEN ||
      this.sm.state === SignalState.EMERGENCY_GREEN ||
      this.sm.preemptTargetLane !== null ||
      this.sm.postEmergencyClearing ||
      hasAnyEmergency;

    const xT = {
      C_N: queues.N,
      C_E: queues.E,
      C_S: queues.S,
      C_W: queues.W,
      E_N: emergencies.N,
      E_E: emergencies.E,
      E_S: emergencies.S,
      E_W: emergencies.W,
    };

    const dcDtMap = {
      dC_N_dt: Number(this.dcDt.N.toFixed(2)),
      dC_E_dt: Number(this.dcDt.E.toFixed(2)),
      dC_S_dt: Number(this.dcDt.S.toFixed(2)),
      dC_W_dt: Number(this.dcDt.W.toFixed(2)),
    };

    return {
      timestamp: this.timestamp,
      state: this.sm.state,
      activeLane: this.sm.activeLane,
      clockwiseLane,
      resumeLane,
      preemptTargetLane: this.sm.preemptTargetLane,
      preemptOriginLane: this.sm.preemptOriginLane ?? clockwiseLane,
      lastBranch: this.sm.lastBranch,
      postEmergencyClearing: this.sm.postEmergencyClearing,
      remainingTime: Number(this.sm.remainingTime.toFixed(1)),
      elapsedInState: Number(this.sm.elapsedInState.toFixed(1)),
      targetGreen: Number(this.sm.lastTargetGreen.toFixed(1)),
      rawGreen: Number(this.sm.lastRawGreen.toFixed(1)),
      lights: this.sm.getLaneLights(),
      isEmergencyPreemption,
      queues,
      emergencies,
      discharging: { ...this.lastDischarging },
      emergencyDischarging: { ...this.lastEmergencyDischarging },
      x_t: xT,
      dc_dt: dcDtMap,
      transitions: this.sm.transitions.slice(-20).reverse(),
      demandMode: this.demandMode,
      config: this.config,
      carCountSwitch: {
        enabled: Boolean(this.config.carCountSwitchingEnabled),
        threshold: Number(this.config.carCountPriorityThreshold),
        targetLane: this.sm.demandSwitchTarget,
        nextDemandLane: this.sm.nextDemandLane,
        activeDemandSwitch: this.sm.activeDemandSwitch,
        bufferElapsed,
        bufferDuration,
        bufferRemaining,
        minGreenRemaining,
        margin: Number(this.config.carCountSwitchMargin),
        maxLaneRedWait: Number(this.config.maxLaneRedWait),
        laneRedWait,
      },
      metrics: {
        adaptive: {
          avgWaitSec: Number(computeAvgWait(this.adaptiveSide, this.timestamp).toFixed(2)),
          idleGreenSec: Number(this.adaptiveSide.idleGreenSec.toFixed(1)),
          throughput: this.adaptiveSide.throughput,
          emgAvgTimeToGreenSec: Number(
            computeAvgEmergencyTimeToGreen(this.adaptiveSide.allEmergencyRecords, this.timestamp).toFixed(2)
          ),
          emgEvents: this.adaptiveSide.allEmergencyRecords.length,
          totalQueued: totalQueuedAdaptive,
        },
        fixed: {
          avgWaitSec: Number(computeAvgWait(this.fixedSide, this.timestamp).toFixed(2)),
          idleGreenSec: Number(this.fixedSide.idleGreenSec.toFixed(1)),
          throughput: this.fixedSide.throughput,
          emgAvgTimeToGreenSec: Number(
            computeAvgEmergencyTimeToGreen(this.fixedSide.allEmergencyRecords, this.timestamp).toFixed(2)
          ),
          emgEvents: this.fixedSide.allEmergencyRecords.length,
          totalQueued: totalQueuedFixed,
          fixedLights: this.fixed.getLaneLights(),
          fixedActiveLane: this.fixed.activeLane,
          fixedRemainingTime: Number(this.fixed.remainingTime.toFixed(1)),
        },
      },
    };
  }
}
