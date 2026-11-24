import React from 'react';
import { LANES, LightColor, SignalState } from '../simulation/stateMachine';

const CORNER_SIGNAL_POS = {
  N: { x: 432, y: 84, badgeX: 432, badgeY: 50, label: 'NORTH (N)' },
  E: { x: 432, y: 466, badgeX: 432, badgeY: 444, label: 'EAST (E)' },
  S: { x: 146, y: 466, badgeX: 146, badgeY: 444, label: 'SOUTH (S)' },
  W: { x: 146, y: 84, badgeX: 146, badgeY: 50, label: 'WEST (W)' },
};

const MAX_VISIBLE_SLOTS = 10;

function getVehicleSlotTransform(lane, slotIndex) {
  const col = slotIndex % 2;
  const row = Math.floor(slotIndex / 2);

  if (lane === 'N') {
    const cx = col === 0 ? 344 : 386;
    const cy = 176 - row * 34;
    return { x: cx, y: cy, angle: 180 };
  }
  if (lane === 'S') {
    const cx = col === 0 ? 296 : 254;
    const cy = 464 + row * 34;
    return { x: cx, y: cy, angle: 0 };
  }
  if (lane === 'E') {
    const cx = 464 + row * 34;
    const cy = col === 0 ? 344 : 386;
    return { x: cx, y: cy, angle: -90 };
  }
  // W
  const cx = 176 - row * 34;
  const cy = col === 0 ? 296 : 254;
  return { x: cx, y: cy, angle: 90 };
}

function getOverflowBadgePos(lane) {
  if (lane === 'N') return { x: 365, y: 18 };
  if (lane === 'S') return { x: 275, y: 624 };
  if (lane === 'E') return { x: 618, y: 365 };
  return { x: 22, y: 275 };
}

function SignalHousing({ lane, lightColor, isActive, remainingTime, state }) {
  const pos = CORNER_SIGNAL_POS[lane];
  const isRed = lightColor === LightColor.RED;
  const isAmber = lightColor === LightColor.AMBER;
  const isGreen = lightColor === LightColor.GREEN;

  const borderStroke = isGreen
    ? '#22c55e'
    : isAmber
      ? '#f59e0b'
      : '#475569';

  return (
    <g>
      <text
        x={pos.badgeX + 31}
        y={pos.badgeY}
        textAnchor="middle"
        fill="#cbd5e1"
        fontSize="12"
        fontWeight="700"
        letterSpacing="0.06em"
      >
        {pos.label}
      </text>

      <rect
        x={pos.x}
        y={pos.y - 14}
        width="62"
        height="108"
        rx="12"
        fill="#0b1120"
        stroke={borderStroke}
        strokeWidth={isActive ? '2.5' : '1.5'}
      />

      {/* Red lamp */}
      <circle
        cx={pos.x + 31}
        cy={pos.y + 8}
        r="12"
        fill={isRed ? '#ef4444' : '#3b1518'}
        filter={isRed ? 'url(#glow-red)' : undefined}
      />
      {/* Amber lamp */}
      <circle
        cx={pos.x + 31}
        cy={pos.y + 40}
        r="12"
        fill={isAmber ? '#f59e0b' : '#3d2c0b'}
        filter={isAmber ? 'url(#glow-amber)' : undefined}
      />
      {/* Green lamp */}
      <circle
        cx={pos.x + 31}
        cy={pos.y + 72}
        r="12"
        fill={isGreen ? '#22c55e' : '#0f361e'}
        filter={isGreen ? 'url(#glow-green)' : undefined}
      />

      {isActive && (
        <g transform={`translate(${pos.x + 31}, ${pos.y + 112})`}>
          <rect
            x="-34"
            y="-10"
            width="68"
            height="22"
            rx="6"
            fill={
              state === SignalState.EMERGENCY_GREEN
                ? 'rgba(239, 68, 68, 0.28)'
                : isAmber
                  ? 'rgba(245, 158, 11, 0.25)'
                  : 'rgba(34, 197, 94, 0.22)'
            }
            stroke={borderStroke}
            strokeWidth="1"
          />
          <text
            x="0"
            y="5"
            textAnchor="middle"
            fill="#f8fafc"
            fontSize="11"
            fontWeight="700"
          >
            {state === SignalState.EMERGENCY_GREEN
              ? 'HOLD'
              : `${remainingTime.toFixed(0)}s`}
          </text>
        </g>
      )}
    </g>
  );
}

function StandardCarIcon({ x, y, angle }) {
  return (
    <g transform={`translate(${x}, ${y}) rotate(${angle})`}>
      <rect
        x="-10"
        y="-14"
        width="20"
        height="28"
        rx="5"
        fill="#38bdf8"
        stroke="#0284c7"
        strokeWidth="1.5"
      />
      <rect x="-7" y="-8" width="14" height="7" rx="2" fill="#0f172a" opacity="0.75" />
      <rect x="-7" y="4" width="14" height="5" rx="1.5" fill="#0f172a" opacity="0.6" />
      <circle cx="-6" cy="-13" r="1.8" fill="#fef08a" />
      <circle cx="6" cy="-13" r="1.8" fill="#fef08a" />
    </g>
  );
}

function AmbulanceIcon({ x, y, angle }) {
  return (
    <g transform={`translate(${x}, ${y}) rotate(${angle})`}>
      <circle r="20" fill="rgba(239, 68, 68, 0.22)" className="sim-siren-ring" />
      <rect
        x="-11"
        y="-16"
        width="22"
        height="32"
        rx="5"
        fill="#f8fafc"
        stroke="#ef4444"
        strokeWidth="2"
      />
      <rect x="-8" y="-12" width="16" height="6" rx="1.5" fill="#1e293b" />
      <rect x="-2" y="-1" width="4" height="10" rx="0.8" fill="#ef4444" />
      <rect x="-5" y="2" width="10" height="4" rx="0.8" fill="#ef4444" />
      <circle cx="-5" cy="-15" r="2.2" fill="#ef4444" />
      <circle cx="5" cy="-15" r="2.2" fill="#3b82f6" />
    </g>
  );
}

function CrossingVehicles({ lane, count, hasEmergencyCrossing, timestamp }) {
  const nodes = [];
  if (hasEmergencyCrossing) {
    const coords =
      lane === 'N'
        ? { x: 344, y: 270, angle: 180, cls: 'cross-south' }
        : lane === 'S'
          ? { x: 296, y: 370, angle: 0, cls: 'cross-north' }
          : lane === 'E'
            ? { x: 370, y: 344, angle: -90, cls: 'cross-west' }
            : { x: 270, y: 296, angle: 90, cls: 'cross-east' };

    nodes.push(
      <g key={`emg-cross-${lane}-${timestamp}`} className={coords.cls}>
        <AmbulanceIcon x={coords.x} y={coords.y} angle={coords.angle} />
      </g>
    );
  }

  const visibleCrossing = Math.min(count, 2);
  for (let i = 0; i < visibleCrossing; i += 1) {
    const offset = i * 36;
    const coords =
      lane === 'N'
        ? { x: i === 0 ? 386 : 344, y: 255 + offset, angle: 180, cls: 'cross-south' }
        : lane === 'S'
          ? { x: i === 0 ? 254 : 296, y: 385 - offset, angle: 0, cls: 'cross-north' }
          : lane === 'E'
            ? { x: 385 - offset, y: i === 0 ? 386 : 344, angle: -90, cls: 'cross-west' }
            : { x: 255 + offset, y: i === 0 ? 254 : 296, angle: 90, cls: 'cross-east' };

    nodes.push(
      <g key={`car-cross-${lane}-${i}-${timestamp}`} className={coords.cls}>
        <StandardCarIcon x={coords.x} y={coords.y} angle={coords.angle} />
      </g>
    );
  }
  return nodes;
}

export default function IntersectionDiagram({ snapshot, embedded = false }) {
  const {
    timestamp,
    state,
    activeLane,
    remainingTime,
    lights,
    queues,
    emergencies,
    discharging,
    emergencyDischarging,
  } = snapshot;

  const isEmergencyCenter =
    state === SignalState.EMERGENCY_GREEN && emergencies[activeLane] > 0;

  const svgBlock = (
    <div className="sim-svg-wrap">
      <svg viewBox="0 0 640 640" className="sim-svg" role="img" aria-label="4-way traffic light intersection simulation">
          <defs>
            <filter id="glow-green" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glow-amber" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background urban blocks / curbs */}
          <rect x="0" y="0" width="640" height="640" fill="#090b0e" />
          <rect x="14" y="14" width="202" height="202" rx="6" fill="#10141b" stroke="#212630" />
          <rect x="424" y="14" width="202" height="202" rx="6" fill="#10141b" stroke="#212630" />
          <rect x="14" y="424" width="202" height="202" rx="6" fill="#10141b" stroke="#212630" />
          <rect x="424" y="424" width="202" height="202" rx="6" fill="#10141b" stroke="#212630" />

          {/* Asphalt crossroad */}
          <rect x="230" y="0" width="180" height="640" fill="#171b22" />
          <rect x="0" y="230" width="640" height="180" fill="#171b22" />
          <rect x="230" y="230" width="180" height="180" fill="#1b2029" />

          {/* Road borders */}
          <line x1="230" y1="0" x2="230" y2="230" stroke="#475569" strokeWidth="2" />
          <line x1="410" y1="0" x2="410" y2="230" stroke="#475569" strokeWidth="2" />
          <line x1="230" y1="410" x2="230" y2="640" stroke="#475569" strokeWidth="2" />
          <line x1="410" y1="410" x2="410" y2="640" stroke="#475569" strokeWidth="2" />
          <line x1="0" y1="230" x2="230" y2="230" stroke="#475569" strokeWidth="2" />
          <line x1="0" y1="410" x2="230" y2="410" stroke="#475569" strokeWidth="2" />
          <line x1="410" y1="230" x2="640" y2="230" stroke="#475569" strokeWidth="2" />
          <line x1="410" y1="410" x2="640" y2="410" stroke="#475569" strokeWidth="2" />

          {/* Center yellow median dividers */}
          <line x1="320" y1="0" x2="320" y2="225" stroke="#eab308" strokeWidth="2.5" strokeDasharray="10 8" />
          <line x1="320" y1="415" x2="320" y2="640" stroke="#eab308" strokeWidth="2.5" strokeDasharray="10 8" />
          <line x1="0" y1="320" x2="225" y2="320" stroke="#eab308" strokeWidth="2.5" strokeDasharray="10 8" />
          <line x1="415" y1="320" x2="640" y2="320" stroke="#eab308" strokeWidth="2.5" strokeDasharray="10 8" />

          {/* Crosswalks */}
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <g key={`cw-${i}`} fill="#cbd5e1" opacity="0.32">
              <rect x={242 + i * 28} y="206" width="14" height="18" rx="1" />
              <rect x={242 + i * 28} y="416" width="14" height="18" rx="1" />
              <rect x="206" y={242 + i * 28} width="18" height="14" rx="1" />
              <rect x="416" y={242 + i * 28} width="18" height="14" rx="1" />
            </g>
          ))}

          {/* White stop lines on incoming half of each approach */}
          <line x1="324" y1="198" x2="406" y2="198" stroke="#f8fafc" strokeWidth="4" />
          <line x1="234" y1="442" x2="316" y2="442" stroke="#f8fafc" strokeWidth="4" />
          <line x1="442" y1="324" x2="442" y2="406" stroke="#f8fafc" strokeWidth="4" />
          <line x1="198" y1="234" x2="198" y2="316" stroke="#f8fafc" strokeWidth="4" />

          {/* Directional flow arrows on pavement */}
          <g fill="#64748b" opacity="0.55">
            {/* N incoming (Southbound) */}
            <path d="M365 212 L357 201 L362 201 L362 189 L368 189 L368 201 L373 201 Z" />
            {/* S incoming (Northbound) */}
            <path d="M275 428 L267 439 L272 439 L272 451 L278 451 L278 439 L283 439 Z" />
            {/* E incoming (Westbound) */}
            <path d="M428 365 L439 357 L439 362 L451 362 L451 368 L439 368 L439 373 Z" />
            {/* W incoming (Eastbound) */}
            <path d="M212 275 L201 267 L201 272 L189 272 L189 278 L201 278 L201 283 Z" />
          </g>

          {/* Queued vehicles on each approach */}
          {LANES.map((lane) => {
            const emgCount = emergencies[lane] ?? 0;
            const stdCount = queues[lane] ?? 0;
            const visibleEmg = Math.min(emgCount, 2);
            const visibleStd = Math.min(stdCount, MAX_VISIBLE_SLOTS - visibleEmg);
            const hiddenCount = emgCount + stdCount - (visibleEmg + visibleStd);
            const items = [];

            for (let i = 0; i < visibleEmg; i += 1) {
              const slot = getVehicleSlotTransform(lane, i);
              items.push(
                <AmbulanceIcon key={`emg-${lane}-${i}`} x={slot.x} y={slot.y} angle={slot.angle} />
              );
            }
            for (let i = 0; i < visibleStd; i += 1) {
              const slot = getVehicleSlotTransform(lane, visibleEmg + i);
              items.push(
                <StandardCarIcon key={`std-${lane}-${i}`} x={slot.x} y={slot.y} angle={slot.angle} />
              );
            }

            if (hiddenCount > 0) {
              const pos = getOverflowBadgePos(lane);
              items.push(
                <g key={`overflow-${lane}`} transform={`translate(${pos.x}, ${pos.y})`}>
                  <rect x="-22" y="-11" width="44" height="20" rx="6" fill="#0f172a" stroke="#38bdf8" />
                  <text x="0" y="3" textAnchor="middle" fill="#e0f2fe" fontSize="11" fontWeight="700">
                    +{hiddenCount}
                  </text>
                </g>
              );
            }
            return items;
          })}

          {/* Vehicles actively crossing the intersection box */}
          {LANES.map((lane) => (
            <CrossingVehicles
              key={`cross-${lane}`}
              lane={lane}
              count={discharging[lane] ?? 0}
              hasEmergencyCrossing={
                (emergencyDischarging[lane] ?? 0) > 0 ||
                (isEmergencyCenter && activeLane === lane)
              }
              timestamp={timestamp}
            />
          ))}

          {/* 4 Traffic Signal Housings */}
          {LANES.map((lane) => (
            <SignalHousing
              key={`sig-${lane}`}
              lane={lane}
              lightColor={lights[lane]}
              isActive={lane === activeLane || state === SignalState.ALL_RED}
              remainingTime={remainingTime}
              state={state}
            />
          ))}

          {/* Center intersection phase overlay */}
          {state === SignalState.ALL_RED && (
            <g transform="translate(320, 320)">
              <rect
                x="-68"
                y="-18"
                width="136"
                height="36"
                rx="8"
                fill="rgba(127, 29, 29, 0.92)"
                stroke="#ef4444"
                strokeWidth="2"
              />
              <text x="0" y="4" textAnchor="middle" fill="#fecaca" fontSize="12" fontWeight="800">
                ALL-RED CLEAR ({remainingTime.toFixed(0)}s)
              </text>
            </g>
          )}

          {state === SignalState.TRUNCATED_GREEN && (
            <g transform="translate(320, 320)">
              <rect
                x="-76"
                y="-18"
                width="152"
                height="36"
                rx="8"
                fill="rgba(120, 53, 15, 0.9)"
                stroke="#f59e0b"
                strokeWidth="2"
              />
              <text x="0" y="4" textAnchor="middle" fill="#fde68a" fontSize="11" fontWeight="800">
                TRUNCATING {activeLane} ({remainingTime.toFixed(0)}s)
              </text>
            </g>
          )}

          {state === SignalState.EMERGENCY_GREEN && (
            <g transform="translate(320, 320)">
              <rect
                x="-76"
                y="-18"
                width="152"
                height="36"
                rx="8"
                fill="rgba(127, 29, 29, 0.9)"
                stroke="#ef4444"
                strokeWidth="2"
              />
              <text x="0" y="4" textAnchor="middle" fill="#fecaca" fontSize="11" fontWeight="800">
                PRIORITY GREEN: {activeLane}
              </text>
            </g>
          )}
        </svg>
    </div>
  );

  if (embedded) {
    return svgBlock;
  }

  return (
    <div className="sim-card sim-intersection-card">
      <div className="sim-card-header">
        <div>
          <h3>4-Way Chowk Schematic</h3>
          <p className="sim-card-sub">
            Clockwise Cycle: N &rarr; E &rarr; S &rarr; W | Left-hand Indian intersection geometry
          </p>
        </div>
        <span className="sim-clock-pill">T+{timestamp}s</span>
      </div>
      {svgBlock}
    </div>
  );
}
