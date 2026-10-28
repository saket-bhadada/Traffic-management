"""High-level TrafficSignalController wrapping PreemptionStateMachine and CSV transition logging (No ML imports)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.control.state_machine import (
    LightColor,
    PreemptionStateMachine,
    SignalState,
    StateMachineConfig,
    TransitionLog,
)


@dataclass(frozen=True)
class ControllerDecision:
    """Output snapshot produced by TrafficSignalController at each 1-second tick."""

    timestamp: float
    state: SignalState
    active_lane: str
    remaining_time: float
    lights: Dict[str, LightColor]
    is_emergency_preemption: bool


class TrafficSignalController:
    """Deterministic traffic signal controller consuming vehicle counts and queue growth rates.

    Strictly free of any ML framework imports.
    """

    def __init__(
        self,
        config: StateMachineConfig = StateMachineConfig(),
        transitions_csv_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.state_machine = PreemptionStateMachine(config=config)
        self.transitions_csv_path = transitions_csv_path
        self._logged_transition_count = 0

        if self.transitions_csv_path is not None:
            self.transitions_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.transitions_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "from_state", "to_state", "active_lane", "next_lane", "reason"]
                )

    def update(
        self,
        timestamp: float,
        x_t: Sequence[float],
        predicted_dc_dt: Sequence[float],
        dt: float = 1.0,
    ) -> ControllerDecision:
        """Step the controller for one second using X_t (8-dim) and predicted dC/dt (4-dim).

        Args:
            timestamp: Current timestamp in seconds.
            x_t: Sequence of 8 floats [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W].
            predicted_dc_dt: Sequence of 4 floats [dC_N/dt, dC_E/dt, dC_S/dt, dC_W/dt].
            dt: Step duration in seconds (default 1.0).

        Returns:
            ControllerDecision for the current tick.
        """
        if len(x_t) != 8:
            raise ValueError(f"Expected X_t of length 8, got {len(x_t)}")
        if len(predicted_dc_dt) != 4:
            raise ValueError(f"Expected predicted_dc_dt of length 4, got {len(predicted_dc_dt)}")

        lanes = self.config.phase_order
        std_counts = {lanes[i]: float(x_t[i]) for i in range(4)}
        emg_counts = {lanes[i]: float(x_t[i + 4]) for i in range(4)}
        dc_dt_map = {lanes[i]: float(predicted_dc_dt[i]) for i in range(4)}

        state = self.state_machine.step(
            timestamp=timestamp,
            standard_counts=std_counts,
            emergency_counts=emg_counts,
            predicted_dc_dt=dc_dt_map,
            dt=dt,
        )

        self._flush_new_transitions()

        is_preempt = (
            state in (SignalState.TRUNCATED_GREEN, SignalState.EMERGENCY_GREEN)
            or self.state_machine.preempt_target_lane is not None
        )
        return ControllerDecision(
            timestamp=float(timestamp),
            state=state,
            active_lane=self.state_machine.active_lane,
            remaining_time=float(self.state_machine.remaining_time),
            lights=self.state_machine.get_lane_lights(),
            is_emergency_preemption=is_preempt,
        )

    def _flush_new_transitions(self) -> None:
        if self.transitions_csv_path is None:
            return
        all_t = self.state_machine.transitions
        if len(all_t) <= self._logged_transition_count:
            return
        new_items = all_t[self._logged_transition_count :]
        with open(self.transitions_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for t in new_items:
                writer.writerow(
                    [
                        f"{t.timestamp:.2f}",
                        t.from_state.value,
                        t.to_state.value,
                        t.active_lane,
                        t.next_lane,
                        t.reason,
                    ]
                )
        self._logged_transition_count = len(all_t)

    @property
    def transition_logs(self) -> List[TransitionLog]:
        """Return all state transitions recorded so far."""
        return list(self.state_machine.transitions)
