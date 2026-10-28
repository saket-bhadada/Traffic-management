"""Compute traffic control performance metrics: average wait time, idle green time, throughput, and emergency time-to-green."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.sim.intersection_sim import EmergencyVehicleRecord, SimulationRunResult


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregated evaluation metrics for a single simulation run."""

    controller_name: str
    average_wait_time_sec: float
    idle_green_time_sec: float
    throughput_vehicles: int
    emergency_time_to_green_sec: float
    emergency_events_count: int


def compute_emergency_time_to_green(
    records: List[EmergencyVehicleRecord],
    default_horizon_sec: float = 1800.0,
) -> float:
    """Calculate mean time-to-green (in seconds) across all emergency vehicle arrivals."""
    if not records:
        return 0.0
    delays: List[float] = []
    for rec in records:
        granted = rec.green_granted_sec if rec.green_granted_sec is not None else default_horizon_sec
        delays.append(max(0.0, float(granted) - float(rec.arrival_sec)))
    return float(np.mean(delays))


def evaluate_simulation_run(result: SimulationRunResult) -> EvaluationMetrics:
    """Compute EvaluationMetrics from a SimulationRunResult."""
    avg_wait = (
        float(np.mean(result.vehicle_wait_times))
        if len(result.vehicle_wait_times) > 0
        else 0.0
    )
    emg_ttg = compute_emergency_time_to_green(
        result.emergency_records, default_horizon_sec=float(result.duration_seconds)
    )
    return EvaluationMetrics(
        controller_name=result.controller_name,
        average_wait_time_sec=round(avg_wait, 2),
        idle_green_time_sec=round(float(result.idle_green_seconds), 2),
        throughput_vehicles=int(result.throughput_vehicles),
        emergency_time_to_green_sec=round(emg_ttg, 2),
        emergency_events_count=len(result.emergency_records),
    )
