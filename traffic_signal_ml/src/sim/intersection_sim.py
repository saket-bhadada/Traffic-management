"""1-second-step intersection simulator for both Adaptive+Preemption and Fixed-Timer Baseline controllers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.control.controller import TrafficSignalController
from src.control.state_machine import LightColor, StateMachineConfig
from src.features.feature_vector import FeatureCSVLogger, FeatureVectorRecord
from src.prediction.lstm_model import OnlineQueuePredictor
from src.sim.baseline_fixed import FixedTimerBaselineController, FixedTimerConfig
from src.sim.synthetic_traffic import LANES, TrafficTrace


@dataclass
class EmergencyVehicleRecord:
    """Tracks an individual emergency vehicle's arrival, first green timestamp, and clearance."""

    lane: str
    arrival_sec: float
    green_granted_sec: Optional[float] = None
    cleared_sec: Optional[float] = None


@dataclass
class SimulationRunResult:
    """Complete telemetry and raw event records from a 1-second-step intersection simulation."""

    controller_name: str
    duration_seconds: int
    vehicle_wait_times: List[float]
    idle_green_seconds: float
    total_green_seconds: float
    throughput_vehicles: int
    remaining_queued_vehicles: int
    emergency_records: List[EmergencyVehicleRecord]
    xt_history: np.ndarray  # shape (T, 8)
    dc_dt_history: np.ndarray  # shape (T, 4)
    active_lane_history: List[str]
    state_history: List[str]


def _run_queue_step(
    t: int,
    trace: TrafficTrace,
    emg_lookup: Dict[int, List[str]],
    std_queues: Dict[str, Deque[float]],
    active_emergencies: Dict[str, List[EmergencyVehicleRecord]],
    all_emergency_records: List[EmergencyVehicleRecord],
) -> np.ndarray:
    """Add new arrivals at second t and return current X_t = [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W]."""
    for idx, lane in enumerate(LANES):
        n_new = int(trace.standard_arrivals[t, idx])
        for _ in range(n_new):
            std_queues[lane].append(float(t))

    for emg_lane in emg_lookup.get(t, []):
        rec = EmergencyVehicleRecord(lane=emg_lane, arrival_sec=float(t))
        active_emergencies[emg_lane].append(rec)
        all_emergency_records.append(rec)

    c_vals = [float(len(std_queues[lane])) for lane in LANES]
    e_vals = [float(len(active_emergencies[lane])) for lane in LANES]
    return np.asarray(c_vals + e_vals, dtype=np.float32)


def _discharge_green_lane(
    t: int,
    lights: Dict[str, LightColor],
    std_queues: Dict[str, Deque[float]],
    active_emergencies: Dict[str, List[EmergencyVehicleRecord]],
    discharge_carry: Dict[str, float],
    saturation_discharge_rate: float,
    wait_times: List[float],
) -> Tuple[int, float, float]:
    """Discharge vehicles on any lane showing GREEN.

    Returns:
        (discharged_count, green_sec_this_tick, idle_green_sec_this_tick)
    """
    discharged = 0
    green_sec = 0.0
    idle_green_sec = 0.0

    for lane in LANES:
        if lights.get(lane) == LightColor.GREEN:
            green_sec += 1.0
            # Check if queue (both standard and emergency) is completely empty at start of green tick
            if len(std_queues[lane]) == 0 and len(active_emergencies[lane]) == 0:
                idle_green_sec += 1.0
                discharge_carry[lane] = 0.0
                continue

            # Record time-to-green and clear emergency vehicles on green lane
            if active_emergencies[lane]:
                for emg in active_emergencies[lane]:
                    if emg.green_granted_sec is None:
                        emg.green_granted_sec = float(t)
                    emg.cleared_sec = float(t + 1)
                active_emergencies[lane].clear()

            # Discharge standard vehicles at fixed saturation rate
            capacity = discharge_carry[lane] + saturation_discharge_rate
            n_depart = int(np.floor(capacity))
            discharge_carry[lane] = capacity - n_depart

            for _ in range(min(n_depart, len(std_queues[lane]))):
                arr_t = std_queues[lane].popleft()
                wait_times.append(float(t) - arr_t)
                discharged += 1
        else:
            discharge_carry[lane] = 0.0

    return discharged, green_sec, idle_green_sec


def run_adaptive_simulation(
    trace: TrafficTrace,
    sm_config: StateMachineConfig,
    predictor: Optional[OnlineQueuePredictor] = None,
    saturation_discharge_rate: float = 1.5,
    feature_log_csv: Optional[Path] = None,
    transitions_log_csv: Optional[Path] = None,
) -> SimulationRunResult:
    """Run the 1-second-step intersection simulator using the Adaptive + Emergency Preemption controller."""
    controller = TrafficSignalController(config=sm_config, transitions_csv_path=transitions_log_csv)
    csv_logger = FeatureCSVLogger(feature_log_csv, overwrite=True) if feature_log_csv else None

    std_queues: Dict[str, Deque[float]] = {lane: deque() for lane in LANES}
    active_emergencies: Dict[str, List[EmergencyVehicleRecord]] = {lane: [] for lane in LANES}
    all_emergency_records: List[EmergencyVehicleRecord] = []
    discharge_carry: Dict[str, float] = {lane: 0.0 for lane in LANES}

    emg_lookup: Dict[int, List[str]] = {}
    for ev in trace.emergency_arrivals:
        emg_lookup.setdefault(ev.arrival_sec, []).append(ev.lane)

    wait_times: List[float] = []
    idle_green_total = 0.0
    total_green = 0.0
    throughput = 0

    xt_history = np.zeros((trace.duration_seconds, 8), dtype=np.float32)
    dc_dt_history = np.zeros((trace.duration_seconds, 4), dtype=np.float32)
    active_lanes: List[str] = []
    states: List[str] = []

    prev_c = np.zeros(4, dtype=np.float32)

    for t in range(trace.duration_seconds):
        x_t = _run_queue_step(
            t, trace, emg_lookup, std_queues, active_emergencies, all_emergency_records
        )
        xt_history[t] = x_t

        if csv_logger is not None:
            csv_logger.log(FeatureVectorRecord(timestamp=float(t), x_t=x_t))

        if predictor is not None:
            pred_dc_dt = predictor.push_and_predict(x_t)
        else:
            pred_dc_dt = (x_t[:4] - prev_c) if t > 0 else np.zeros(4, dtype=np.float32)
        prev_c = x_t[:4].copy()
        dc_dt_history[t] = pred_dc_dt

        # First check if an emergency vehicle is already on a currently GREEN lane before stepping
        current_lights = controller.state_machine.get_lane_lights()
        for lane in LANES:
            if current_lights[lane] == LightColor.GREEN and active_emergencies[lane]:
                for emg in active_emergencies[lane]:
                    if emg.green_granted_sec is None:
                        emg.green_granted_sec = float(t)

        decision = controller.update(timestamp=float(t), x_t=x_t, predicted_dc_dt=pred_dc_dt, dt=1.0)
        active_lanes.append(decision.active_lane)
        states.append(decision.state.value)

        dep, g_sec, idle_sec = _discharge_green_lane(
            t=t,
            lights=decision.lights,
            std_queues=std_queues,
            active_emergencies=active_emergencies,
            discharge_carry=discharge_carry,
            saturation_discharge_rate=saturation_discharge_rate,
            wait_times=wait_times,
        )
        throughput += dep
        total_green += g_sec
        idle_green_total += idle_sec

    # Include accrued wait times of vehicles still in queue at end of horizon
    remaining_q = 0
    for lane in LANES:
        remaining_q += len(std_queues[lane])
        for arr_t in std_queues[lane]:
            wait_times.append(float(trace.duration_seconds) - arr_t)

    return SimulationRunResult(
        controller_name="Adaptive + Emergency Preemption",
        duration_seconds=trace.duration_seconds,
        vehicle_wait_times=wait_times,
        idle_green_seconds=idle_green_total,
        total_green_seconds=total_green,
        throughput_vehicles=throughput,
        remaining_queued_vehicles=remaining_q,
        emergency_records=all_emergency_records,
        xt_history=xt_history,
        dc_dt_history=dc_dt_history,
        active_lane_history=active_lanes,
        state_history=states,
    )


def run_fixed_baseline_simulation(
    trace: TrafficTrace,
    fixed_config: FixedTimerConfig = FixedTimerConfig(),
    saturation_discharge_rate: float = 1.5,
) -> SimulationRunResult:
    """Run the 1-second-step intersection simulator using the Fixed-Timer Baseline controller."""
    baseline = FixedTimerBaselineController(config=fixed_config)

    std_queues: Dict[str, Deque[float]] = {lane: deque() for lane in LANES}
    active_emergencies: Dict[str, List[EmergencyVehicleRecord]] = {lane: [] for lane in LANES}
    all_emergency_records: List[EmergencyVehicleRecord] = []
    discharge_carry: Dict[str, float] = {lane: 0.0 for lane in LANES}

    emg_lookup: Dict[int, List[str]] = {}
    for ev in trace.emergency_arrivals:
        emg_lookup.setdefault(ev.arrival_sec, []).append(ev.lane)

    wait_times: List[float] = []
    idle_green_total = 0.0
    total_green = 0.0
    throughput = 0

    xt_history = np.zeros((trace.duration_seconds, 8), dtype=np.float32)
    dc_dt_history = np.zeros((trace.duration_seconds, 4), dtype=np.float32)
    active_lanes: List[str] = []
    states: List[str] = []

    for t in range(trace.duration_seconds):
        x_t = _run_queue_step(
            t, trace, emg_lookup, std_queues, active_emergencies, all_emergency_records
        )
        xt_history[t] = x_t

        lights = baseline.step(dt=1.0)
        active_lanes.append(baseline.active_lane)
        states.append(baseline.state.value)

        dep, g_sec, idle_sec = _discharge_green_lane(
            t=t,
            lights=lights,
            std_queues=std_queues,
            active_emergencies=active_emergencies,
            discharge_carry=discharge_carry,
            saturation_discharge_rate=saturation_discharge_rate,
            wait_times=wait_times,
        )
        throughput += dep
        total_green += g_sec
        idle_green_total += idle_sec

    remaining_q = 0
    for lane in LANES:
        remaining_q += len(std_queues[lane])
        for arr_t in std_queues[lane]:
            wait_times.append(float(trace.duration_seconds) - arr_t)

    return SimulationRunResult(
        controller_name="Fixed-Timer Baseline (30s)",
        duration_seconds=trace.duration_seconds,
        vehicle_wait_times=wait_times,
        idle_green_seconds=idle_green_total,
        total_green_seconds=total_green,
        throughput_vehicles=throughput,
        remaining_queued_vehicles=remaining_q,
        emergency_records=all_emergency_records,
        xt_history=xt_history,
        dc_dt_history=dc_dt_history,
        active_lane_history=active_lanes,
        state_history=states,
    )
