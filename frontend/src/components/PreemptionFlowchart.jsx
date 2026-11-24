import React from 'react';
import {
  LANES,
  PreemptBranch,
  SignalState,
} from '../simulation/stateMachine';

function getStepStates(snapshot) {
  const {
    state,
    activeLane,
    clockwiseLane,
    resumeLane,
    preemptTargetLane,
    lastBranch,
    postEmergencyClearing,
    emergencies,
    queues,
    remainingTime,
    targetGreen,
    config,
    carCountSwitch,
  } = snapshot;

  const detectedLane = preemptTargetLane ?? LANES.find((l) => emergencies[l] > 0) ?? null;
  const hasEmergency = detectedLane !== null;
  const isSameLane = lastBranch === PreemptBranch.SAME_LANE;
  const isTruncate = lastBranch === PreemptBranch.TRUNCATE;
  const isExpireNormal = lastBranch === PreemptBranch.EXPIRE_NORMAL;

  const demandTarget = carCountSwitch?.targetLane ?? null;
  const isDemandBuffering =
    state === SignalState.ADAPTIVE_GREEN &&
    !hasEmergency &&
    preemptTargetLane === null &&
    demandTarget !== null;
  const isDemandSwitchActive = Boolean(carCountSwitch?.activeDemandSwitch);

  const active = new Set();
  const passed = new Set();

  const addPreemptionPath = () => {
    passed.add('phase_i');
    passed.add('emg_check');
    if (!isSameLane) {
      passed.add('rem_check');
      passed.add(isTruncate ? 'truncate_2s' : 'expire_normal');
      passed.add('amber_3s');
      passed.add('all_red_1s');
    }
  };

  if (state === SignalState.ADAPTIVE_GREEN) {
    if (isExpireNormal) {
      passed.add('phase_i');
      passed.add('emg_check');
      passed.add('rem_check');
      active.add('expire_normal');
    } else if (hasEmergency) {
      passed.add('phase_i');
      active.add('emg_check');
    } else {
      active.add('phase_i');
      active.add('continue_adaptive');
      if (isDemandBuffering) {
        active.add('demand_buffer');
      }
    }
  } else if (state === SignalState.TRUNCATED_GREEN) {
    passed.add('phase_i');
    passed.add('emg_check');
    passed.add('rem_check');
    active.add('truncate_2s');
  } else if (state === SignalState.AMBER) {
    if (postEmergencyClearing) {
      addPreemptionPath();
      passed.add('emg_green');
      passed.add('emg_cleared');
      active.add('resume_next');
    } else if (preemptTargetLane !== null) {
      passed.add('phase_i');
      passed.add('emg_check');
      passed.add('rem_check');
      passed.add(isTruncate ? 'truncate_2s' : 'expire_normal');
      active.add('amber_3s');
    } else {
      passed.add('phase_i');
      passed.add('continue_adaptive');
      if (isDemandSwitchActive) {
        passed.add('demand_buffer');
      }
      active.add('amber_3s');
    }
  } else if (state === SignalState.ALL_RED) {
    if (postEmergencyClearing) {
      addPreemptionPath();
      passed.add('emg_green');
      passed.add('emg_cleared');
      active.add('resume_next');
    } else {
      if (preemptTargetLane !== null) {
        passed.add('phase_i');
        passed.add('emg_check');
        passed.add('rem_check');
        passed.add(isTruncate ? 'truncate_2s' : 'expire_normal');
      } else {
        passed.add('phase_i');
        passed.add('continue_adaptive');
        if (isDemandSwitchActive) {
          passed.add('demand_buffer');
        }
      }
      passed.add('amber_3s');
      active.add('all_red_1s');
    }
  } else if (state === SignalState.EMERGENCY_GREEN) {
    addPreemptionPath();
    active.add('emg_green');
    active.add('emg_cleared');
  }

  const cls = (id) => {
    if (id === 'demand_buffer' && active.has(id)) return 'flow-node is-demand-active';
    if (active.has(id)) return 'flow-node is-active';
    if (passed.has(id)) return 'flow-node is-passed';
    return 'flow-node is-idle';
  };

  return {
    cls,
    detectedLane,
    hasEmergency,
    isSameLane,
    isTruncate,
    isExpireNormal,
    clockwiseLane,
    resumeLane,
    activeLane,
    remainingTime,
    targetGreen,
    config,
    state,
    postEmergencyClearing,
    emergencies,
    preemptTargetLane,
    carCountEnabled: Boolean(carCountSwitch?.enabled),
    isDemandBuffering,
    isDemandSwitchActive,
    demandTarget,
    nextDemandLane: carCountSwitch?.nextDemandLane ?? null,
    targetCount: demandTarget ? (queues[demandTarget] ?? 0) : 0,
    activeCount: queues[activeLane] ?? 0,
    bufferElapsed: carCountSwitch?.bufferElapsed ?? 0,
    bufferDuration: carCountSwitch?.bufferDuration ?? config.carCountSwitchBuffer ?? 6,
    minGreenRemaining: carCountSwitch?.minGreenRemaining ?? 0,
    margin: carCountSwitch?.margin ?? config.carCountSwitchMargin ?? 5,
  };
}

export default function PreemptionFlowchart({ snapshot }) {
  const s = getStepStates(snapshot);
  const emgLaneLabel = s.preemptTargetLane ?? s.detectedLane ?? 'j';
  const ejCount =
    s.state === SignalState.EMERGENCY_GREEN ? s.emergencies[s.activeLane] ?? 0 : null;

  return (
    <div className="sim-card sim-flowchart-card">
      <div className="sim-card-header">
        <div>
          <h3>Signal Control &amp; Preemption Flowchart</h3>
          <p className="sim-card-sub">
            Live execution trace of <code>PreemptionStateMachine</code>
          </p>
        </div>
        <span className="flow-state-badge">{s.state}</span>
      </div>

      <div className="flow-diagram">
        {/* 1. Current Phase */}
        <div className={s.cls('phase_i')}>
          <div className="flow-title">Current Phase: Lane i ({s.clockwiseLane}) is GREEN</div>
          <div className="flow-meta">
            Clockwise index: {s.clockwiseLane} &rarr; Next: {s.resumeLane}
          </div>
        </div>

        <div className="flow-arrow">&darr;</div>

        {/* 2. Emergency Detected? */}
        <div className={`${s.cls('emg_check')} flow-decision`}>
          <div className="flow-title">Emergency detected on Lane j?</div>
          <div className="flow-meta">
            {s.hasEmergency || s.preemptTargetLane
              ? `YES: j = Lane ${emgLaneLabel}${s.isSameLane ? ' (j == i: Same Lane!)' : ''}`
              : 'NO: All E_N, E_E, E_S, E_W == 0'}
          </div>
        </div>

        {/* Split NO vs YES */}
        <div className="flow-split-2">
          <div className="flow-branch-col">
            <div className={`flow-branch-label ${!s.hasEmergency && !s.preemptTargetLane ? 'lit-green' : ''}`}>
              NO
            </div>
            <div className={s.cls('continue_adaptive')}>
              <div className="flow-title">Continue Adaptive Timer</div>
              <div className="flow-meta">
                Target {s.targetGreen.toFixed(1)}s &middot; Rem {s.remainingTime.toFixed(1)}s
              </div>
            </div>

            <div className={s.cls('demand_buffer')}>
              <div className="flow-title">Car-Count Demand &amp; Buffer</div>
              <div className="flow-meta">
                {!s.carCountEnabled
                  ? 'Disabled (clockwise cycle only)'
                  : s.isDemandBuffering
                    ? `Target Lane ${s.demandTarget} (${s.targetCount} vs ${s.activeCount}) · Buffer ${s.bufferElapsed.toFixed(1)}s/${s.bufferDuration.toFixed(1)}s${
                        s.minGreenRemaining > 0
                          ? ` · Min green: ${s.minGreenRemaining.toFixed(1)}s`
                          : ''
                      }`
                    : s.isDemandSwitchActive
                      ? `Switching to Lane ${s.nextDemandLane ?? s.resumeLane} after clearance`
                      : `No higher-count override (margin +${s.margin}, buffer ${s.bufferDuration}s)`}
              </div>
            </div>
          </div>

          <div className="flow-branch-col">
            <div className={`flow-branch-label ${s.hasEmergency || s.preemptTargetLane ? 'lit-red' : ''}`}>
              YES
            </div>
            <div className={`${s.cls('rem_check')} flow-decision`}>
              <div className="flow-title">
                Remaining Green &gt; {s.config.preemptRemainingThreshold}s?
              </div>
              <div className="flow-meta">
                {s.isSameLane
                  ? 'Bypassed (j == i): Direct Emergency Green'
                  : s.isTruncate
                    ? 'YES (> 4s): Truncate immediately'
                    : s.isExpireNormal
                      ? 'NO (<= 4s): Let green finish'
                      : `Current rem: ${s.remainingTime.toFixed(1)}s`}
              </div>
            </div>

            <div className="flow-split-2 flow-subsplit">
              <div className="flow-branch-col">
                <div className={`flow-branch-label ${s.isTruncate ? 'lit-amber' : ''}`}>YES</div>
                <div className={s.cls('truncate_2s')}>
                  <div className="flow-title">Truncate Green to {s.config.truncatedGreenDuration}s</div>
                  <div className="flow-meta">
                    {s.state === SignalState.TRUNCATED_GREEN
                      ? `Counting: ${s.remainingTime.toFixed(1)}s`
                      : 'TRUNCATED_GREEN'}
                  </div>
                </div>
              </div>

              <div className="flow-branch-col">
                <div className={`flow-branch-label ${s.isExpireNormal ? 'lit-blue' : ''}`}>NO</div>
                <div className={s.cls('expire_normal')}>
                  <div className="flow-title">Expire Green Normally</div>
                  <div className="flow-meta">
                    {s.state === SignalState.ADAPTIVE_GREEN && s.isExpireNormal
                      ? `Counting: ${s.remainingTime.toFixed(1)}s`
                      : 'Stay ADAPTIVE_GREEN'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flow-arrow">&darr;</div>

        {/* 3 & 4. Intergreen clearance */}
        <div className="flow-inline-pair">
          <div className={s.cls('amber_3s')}>
            <div className="flow-title">Trigger Amber Phase: {s.config.amberDuration}s</div>
            <div className="flow-meta">
              {s.state === SignalState.AMBER && !s.postEmergencyClearing
                ? `Active (${s.remainingTime.toFixed(1)}s left)`
                : 'AMBER'}
            </div>
          </div>

          <div className="flow-arrow-h">&rarr;</div>

          <div className={s.cls('all_red_1s')}>
            <div className="flow-title">All-Red Clearance: {s.config.allRedDuration}s</div>
            <div className="flow-meta">
              {s.state === SignalState.ALL_RED && !s.postEmergencyClearing
                ? `Active (${s.remainingTime.toFixed(1)}s left)`
                : 'ALL_RED'}
            </div>
          </div>
        </div>

        <div className="flow-arrow">&darr;</div>

        {/* 5 & 6. Emergency Green & Clearance */}
        <div className="flow-inline-pair">
          <div className={s.cls('emg_green')}>
            <div className="flow-title">Grant Emergency Green to Lane {emgLaneLabel}</div>
            <div className="flow-meta">
              {s.state === SignalState.EMERGENCY_GREEN
                ? `Active on Lane ${s.activeLane}`
                : 'EMERGENCY_GREEN'}
            </div>
          </div>

          <div className="flow-arrow-h">&rarr;</div>

          <div className={`${s.cls('emg_cleared')} flow-decision`}>
            <div className="flow-title">Emergency Cleared (E_j == 0)?</div>
            <div className="flow-meta">
              {s.state === SignalState.EMERGENCY_GREEN
                ? ejCount > 0
                  ? `NO (E_${s.activeLane}=${ejCount}) — Holding`
                  : `YES (E_${s.activeLane}=0) — Clearing`
                : s.postEmergencyClearing
                  ? `YES — ${s.state} (${s.remainingTime.toFixed(1)}s)`
                  : 'Hold until E_j == 0'}
            </div>
          </div>
        </div>

        <div className="flow-arrow">&darr;</div>

        {/* 7. Resume */}
        <div className={s.cls('resume_next')}>
          <div className="flow-title">
            Resume Clockwise Cycle at Next Phase: (i+1) = Lane {s.resumeLane}
          </div>
          <div className="flow-meta">
            {s.postEmergencyClearing
              ? `Clearing via ${s.state} (${s.remainingTime.toFixed(1)}s) -> Lane ${s.resumeLane}`
              : `Always continues clockwise after preemption (never repeats or skips)`}
          </div>
        </div>
      </div>
    </div>
  );
}
