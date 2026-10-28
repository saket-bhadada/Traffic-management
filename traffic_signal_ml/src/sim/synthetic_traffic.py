"""Synthetic traffic arrival generator with varied arrival rates, bursts, lulls, and emergency events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

LANES: Tuple[str, str, str, str] = ("N", "E", "S", "W")


@dataclass(frozen=True)
class EmergencyEventSpec:
    """Scheduled or stochastic emergency arrival on a specific approach lane."""

    arrival_sec: int
    lane: str


@dataclass(frozen=True)
class SyntheticTrafficConfig:
    """Configuration for generating realistic asymmetric 4-way chowk traffic arrivals."""

    duration_seconds: int = 1800
    seed: int = 42
    # Asymmetric base arrival rates (vehicles/sec) so N/S arterial is busier than E/W cross-street
    base_arrival_rates: Dict[str, float] = field(
        default_factory=lambda: {"N": 0.34, "E": 0.14, "S": 0.30, "W": 0.12}
    )
    burst_multiplier: float = 2.4
    lull_multiplier: float = 0.25
    emergency_interval_mean_sec: int = 180


@dataclass(frozen=True)
class TrafficTrace:
    """Pre-generated deterministic arrival trace for reproducible controller vs. baseline comparison."""

    duration_seconds: int
    standard_arrivals: np.ndarray  # shape (T, 4) int32 arrivals per second for (N, E, S, W)
    emergency_arrivals: List[EmergencyEventSpec]


def generate_traffic_trace(config: SyntheticTrafficConfig) -> TrafficTrace:
    """Generate a reproducible 1-second traffic arrival trace with bursts, lulls, and emergency vehicles."""
    rng = np.random.default_rng(config.seed)
    t_total = config.duration_seconds
    std_arrivals = np.zeros((t_total, 4), dtype=np.int32)

    for t in range(t_total):
        # Create alternating burst and lull windows across approaches to mimic platoon waves
        cycle_pos = t % 240
        for idx, lane in enumerate(LANES):
            rate = float(config.base_arrival_rates.get(lane, 0.2))
            if idx in (0, 2):  # N/S arterial burst during first 80s of 240s wave
                if 20 <= cycle_pos < 90:
                    rate *= config.burst_multiplier
                elif 140 <= cycle_pos < 200:
                    rate *= config.lull_multiplier
            else:  # E/W cross-street burst during middle window
                if 100 <= cycle_pos < 150:
                    rate *= config.burst_multiplier
                else:
                    rate *= config.lull_multiplier

            std_arrivals[t, idx] = int(rng.poisson(lam=rate))

    # Generate emergency vehicle arrivals spaced across the simulation
    emergency_events: List[EmergencyEventSpec] = []
    next_emg_t = int(rng.integers(35, 75))
    while next_emg_t < t_total - 40:
        lane_choice = str(LANES[int(rng.integers(0, 4))])
        emergency_events.append(EmergencyEventSpec(arrival_sec=next_emg_t, lane=lane_choice))
        gap = int(rng.integers(int(config.emergency_interval_mean_sec * 0.6), int(config.emergency_interval_mean_sec * 1.4)))
        next_emg_t += max(60, gap)

    return TrafficTrace(
        duration_seconds=t_total,
        standard_arrivals=std_arrivals,
        emergency_arrivals=emergency_events,
    )


def generate_synthetic_xt_series(config: SyntheticTrafficConfig) -> np.ndarray:
    """Generate an (N, 8) array of X_t = [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W] for LSTM training."""
    trace = generate_traffic_trace(config)
    x_series = np.zeros((config.duration_seconds, 8), dtype=np.float32)
    queues = np.zeros(4, dtype=np.float32)
    emg_queues = np.zeros(4, dtype=np.float32)

    emg_by_sec: Dict[int, List[int]] = {}
    for ev in trace.emergency_arrivals:
        lane_idx = LANES.index(ev.lane)
        emg_by_sec.setdefault(ev.arrival_sec, []).append(lane_idx)

    active_phase = 0
    phase_timer = 0
    discharge_rate = 1.5

    for t in range(config.duration_seconds):
        queues += trace.standard_arrivals[t].astype(np.float32)
        for l_idx in emg_by_sec.get(t, []):
            emg_queues[l_idx] += 1.0

        # Simple synthetic signal cycling so LSTM sees both queue build-up and green discharge
        phase_timer += 1
        if phase_timer <= 22:
            # Green on active_phase
            queues[active_phase] = max(0.0, queues[active_phase] - discharge_rate)
            if emg_queues[active_phase] > 0:
                emg_queues[active_phase] = max(0.0, emg_queues[active_phase] - 1.0)
        elif phase_timer >= 26:
            active_phase = (active_phase + 1) % 4
            phase_timer = 0

        x_series[t, :4] = queues
        x_series[t, 4:] = emg_queues

    return x_series
