import React from 'react';
import { LANES, LightColor, SignalState } from '../simulation/stateMachine';

const CORNER_SIGNAL_POS = {
  N: { x: 432, y: 84, badgeX: 432, badgeY: 50, label: 'North' },
  E: { x: 432, y: 466, badgeX: 432, badgeY: 444, label: 'East' },
  S: { x: 146, y: 466, badgeX: 146, badgeY: 444, label: 'South' },
  W: { x: 146, y: 84, badgeX: 146, badgeY: 50, label: 'West' },
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

function SignalHousing({ lane, lightColor, isActive, remainingTime, state, offline }) {
  const pos = CORNER_SIGNAL_POS[lane];
  const isRed = !offline && lightColor === LightColor.RED;
  const isAmber = !offline && lightColor === LightColor.AMBER;
  const isGreen = !offline && lightColor === LightColor.GREEN;

  const borderStroke = isGreen
    ? '#16a34a'
    : isAmber
      ? '#d97706'
      : '#2e3542';

  return (
    <g>
      <text
        x={pos.badgeX + 31}
        y={pos.badgeY}
        textAnchor="middle"
        fill="#949eae"
        fontSize="12"
        fontWeight="600"
        letterSpacing="0.05em"
      >
        {pos.label}
      </text>

      <rect
        x={pos.x}
        y={pos.y - 14}
        width="62"
        height="108"
        rx="4"
        fill="#0b0d10"
        stroke={borderStroke}
        strokeWidth={isActive && !offline ? '2' : '1'}
      />

      {/* Red lamp */}
      <circle
        cx={pos.x + 31}
        cy={pos.y + 8}
        r="11"
        fill={isRed ? '#dc2626' : '#1d222b'}
      />
      {/* Amber lamp */}
      <circle
        cx={pos.x + 31}
        cy={pos.y + 40}
        r="11"
        fill={isAmber ? '#d97706' : '#1d222b'}
      />
      {/* Green lamp */}
      <circle
        cx={pos.x + 31}
        cy={pos.y + 72}
        r="11"
        fill={isGreen ? '#16a34a' : '#1d222b'}
      />

      {!offline && isActive && (
        <g transform={`translate(${pos.x + 31}, ${pos.y + 112})`}>
          <rect
            x="-34"
            y="-10"
            width="68"
            height="22"
            rx="3"
            fill={
              state === SignalState.EMERGENCY_GREEN
                ? 'rgba(220, 38, 38, 0.22)'
                : isAmber
                  ? 'rgba(217, 119, 6, 0.22)'
                  : 'rgba(22, 163, 74, 0.22)'
            }
            stroke={borderStroke}
            strokeWidth="1"
          />
          <text
            x="0"
            y="5"
            textAnchor="middle"
            fill="#e6eaf0"
            fontSize="12"
            fontWeight="600"
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
        rx="3"
        fill="#38bdf8"
        stroke="#0284c7"
        strokeWidth="1"
      />
      <rect x="-7" y="-8" width="14" height="7" rx="1" fill="#0b0d10" opacity="0.75" />
      <rect x="-7" y="4" width="14" height="5" rx="1" fill="#0b0d10" opacity="0.6" />
      <circle cx="-6" cy="-13" r="1.6" fill="#e6eaf0" />
      <circle cx="6" cy="-13" r="1.6" fill="#e6eaf0" />
    </g>
  );
}

function AmbulanceIcon({ x, y, angle }) {
  return (
    <g transform={`translate(${x}, ${y}) rotate(${angle})`}>
      <circle r="18" fill="none" stroke="#dc2626" strokeWidth="1.5" opacity="0.65" />
      <rect
        x="-11"
        y="-16"
        width="22"
        height="32"
        rx="3"
        fill="#e6eaf0"
        stroke="#dc2626"
        strokeWidth="2"
      />
      <rect x="-8" y="-12" width="16" height="6" rx="1" fill="#171b22" />
      <rect x="-2" y="-1" width="4" height="10" rx="0.5" fill="#dc2626" />
      <rect x="-5" y="2" width="10" height="4" rx="0.5" fill="#dc2626" />
      <circle cx="-5" cy="-15" r="2" fill="#dc2626" />
      <circle cx="5" cy="-15" r="2" fill="#38bdf8" />
    </g>
  );
}

function CrossingVehicles({ lane, count, hasEmergencyCrossing, timestamp }) {
  const nodes = [];
  if (hasEmergencyCrossing) {
    const coords =
      lane === 'N'
        ? { x: 344, y: 295, angle: 180 }
        : lane === 'S'
          ? { x: 296, y: 345, angle: 0 }
          : lane === 'E'
            ? { x: 345, y: 344, angle: -90 }
            : { x: 295, y: 296, angle: 90 };

    nodes.push(
      <g key={`emg-cross-${lane}-${timestamp}`}>
        <AmbulanceIcon x={coords.x} y={coords.y} angle={coords.angle} />
      </g>
    );
  }

  const visibleCrossing = Math.min(count, 2);
  for (let i = 0; i < visibleCrossing; i += 1) {
    const offset = i * 36;
    const coords =
      lane === 'N'
        ? { x: i === 0 ? 386 : 344, y: 270 + offset, angle: 180 }
        : lane === 'S'
          ? { x: i === 0 ? 254 : 296, y: 370 - offset, angle: 0 }
          : lane === 'E'
            ? { x: 370 - offset, y: i === 0 ? 386 : 344, angle: -90 }
            : { x: 270 + offset, y: i === 0 ? 254 : 296, angle: 90 };

    nodes.push(
      <g key={`car-cross-${lane}-${i}-${timestamp}`}>
        <StandardCarIcon x={coords.x} y={coords.y} angle={coords.angle} />
      </g>
    );
  }
  return nodes;
}

export default function IntersectionDiagram({ snapshot, embedded = false, offline = false }) {
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
      <svg viewBox="0 0 640 640" className="sim-svg" role="img" aria-label="4-way traffic light intersection schematic">
          {/* Background urban blocks / curbs */}
          <rect x="0" y="0" width="640" height="640" fill="#0b0d10" />
          <rect x="14" y="14" width="202" height="202" rx="4" fill="#111419" stroke="#222730" />
          <rect x="424" y="14" width="202" height="202" rx="4" fill="#111419" stroke="#222730" />
          <rect x="14" y="424" width="202" height="202" rx="4" fill="#111419" stroke="#222730" />
          <rect x="424" y="424" width="202" height="202" rx="4" fill="#111419" stroke="#222730" />

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
              offline={offline}
            />
          ))}

          {/* Center intersection phase overlay */}
          {offline && (
            <g transform="translate(320, 320)">
              <rect
                x="-76"
                y="-16"
                width="152"
                height="32"
                rx="4"
                fill="#111419"
                stroke="#2e3542"
                strokeWidth="1"
              />
              <text x="0" y="4" textAnchor="middle" fill="#949eae" fontSize="12" fontWeight="600">
                CONTROLLER OFFLINE
              </text>
            </g>
          )}

          {!offline && state === SignalState.ALL_RED && (
            <g transform="translate(320, 320)">
              <rect
                x="-68"
                y="-16"
                width="136"
                height="32"
                rx="4"
                fill="rgba(127, 29, 29, 0.9)"
                stroke="#dc2626"
                strokeWidth="1.5"
              />
              <text x="0" y="4" textAnchor="middle" fill="#fecaca" fontSize="12" fontWeight="700">
                ALL-RED ({remainingTime.toFixed(0)}s)
              </text>
            </g>
          )}

          {!offline && state === SignalState.TRUNCATED_GREEN && (
            <g transform="translate(320, 320)">
              <rect
                x="-76"
                y="-16"
                width="152"
                height="32"
                rx="4"
                fill="rgba(120, 53, 15, 0.9)"
                stroke="#d97706"
                strokeWidth="1.5"
              />
              <text x="0" y="4" textAnchor="middle" fill="#fde68a" fontSize="12" fontWeight="700">
                TRUNCATE {activeLane} ({remainingTime.toFixed(0)}s)
              </text>
            </g>
          )}

          {!offline && state === SignalState.EMERGENCY_GREEN && (
            <g transform="translate(320, 320)">
              <rect
                x="-76"
                y="-16"
                width="152"
                height="32"
                rx="4"
                fill="rgba(127, 29, 29, 0.9)"
                stroke="#dc2626"
                strokeWidth="1.5"
              />
              <text x="0" y="4" textAnchor="middle" fill="#fecaca" fontSize="12" fontWeight="700">
                PREEMPT: {activeLane}
              </text>
            </g>
          )}
        </svg>
    </div>
  );

  if (embedded) {
    return svgBlock;
  }

  return <div className="schematic-standalone">{svgBlock}</div>;
}
