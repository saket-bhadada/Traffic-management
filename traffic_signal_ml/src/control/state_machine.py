"""Explicit Emergency Preemption and Clockwise Signal State Machine (No ML imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SignalState(str, Enum):
    """Explicit signal controller states."""

    ADAPTIVE_GREEN = "ADAPTIVE_GREEN"
    TRUNCATED_GREEN = "TRUNCATED_GREEN"
    AMBER = "AMBER"
    ALL_RED = "ALL_RED"
    EMERGENCY_GREEN = "EMERGENCY_GREEN"


class LightColor(str, Enum):
    """Per-approach signal aspect shown to drivers."""

    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


@dataclass(frozen=True)
class TransitionLog:
    """Audit record logged on every state machine transition."""

    timestamp: float
    from_state: SignalState
    to_state: SignalState
    active_lane: str
    next_lane: str
    reason: str


@dataclass(frozen=True)
class StateMachineConfig:
    """Timing and preemption thresholds for the intersection state machine."""

    phase_order: Tuple[str, str, str, str] = ("N", "E", "S", "W")
    base_green: float = 15.0
    k1: float = 1.5
    k2: float = 4.0
    min_green: float = 10.0
    max_green: float = 60.0
    amber_duration: float = 3.0
    all_red_duration: float = 1.0
    preempt_remaining_threshold: float = 4.0
    truncated_green_duration: float = 2.0


def compute_adaptive_green(
    c_lane: float,
    dc_dt: float,
    base_green: float = 15.0,
    k1: float = 1.5,
    k2: float = 4.0,
    min_green: float = 10.0,
    max_green: float = 60.0,
) -> float:
    """Compute clamped adaptive green duration (seconds) for the active lane:
    green = clamp(base + k1*C_lane + k2*max(dC/dt, 0), min_green, max_green)
    """
    raw = float(base_green) + float(k1) * float(c_lane) + float(k2) * max(float(dc_dt), 0.0)
    return max(float(min_green), min(float(max_green), raw))


@dataclass
class PreemptionStateMachine:
    """Explicit deterministic state machine for clockwise phase control and emergency preemption.

    Rules implemented:
    - Clockwise cycle: N -> E -> S -> W -> N.
    - During ADAPTIVE_GREEN on lane i:
      Recomputes target green duration every second using compute_adaptive_green, and updates
      remaining green as `target_green - elapsed_in_green`.
      Emergency detected on lane j?
        NO  -> continue adaptive timer
        YES ->
          If j == i: immediately transition to EMERGENCY_GREEN on lane i (holding until E_i == 0)
          If j != i:
            remaining green > 4s?
              YES -> transition to TRUNCATED_GREEN with remaining timer = 2s
              NO  -> let green expire normally (stay in ADAPTIVE_GREEN with current remaining timer,
                     routing next phase to emergency lane j)
            -> AMBER 3s
            -> ALL_RED 1s
            -> EMERGENCY_GREEN to lane j
            -> hold until E_j == 0
            -> AMBER 3s -> ALL_RED 1s -> resume clockwise cycle at phase (i+1)
    """

    config: StateMachineConfig = field(default_factory=StateMachineConfig)
    state: SignalState = SignalState.ADAPTIVE_GREEN
    active_lane: str = "N"
    remaining_time: float = 15.0
    elapsed_in_state: float = 0.0

    # Tracks the clockwise phase index i that was active before preemption
    clockwise_index: int = 0
    preempt_target_lane: Optional[str] = None
    resume_clockwise_index: Optional[int] = None
    transitions: List[TransitionLog] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.active_lane = self.config.phase_order[self.clockwise_index]
        self.remaining_time = float(self.config.base_green)

    def _next_clockwise_index(self, idx: int) -> int:
        return (idx + 1) % len(self.config.phase_order)

    def _select_emergency_lane(self, emergency_counts: Dict[str, float]) -> Optional[str]:
        """Select lane j with E_j > 0. Prioritizes active lane if it has an emergency, else clockwise from active."""
        if emergency_counts.get(self.active_lane, 0.0) > 0.0:
            return self.active_lane
        n = len(self.config.phase_order)
        start_idx = self.config.phase_order.index(self.active_lane)
        for offset in range(1, n + 1):
            lane = self.config.phase_order[(start_idx + offset) % n]
            if emergency_counts.get(lane, 0.0) > 0.0:
                return lane
        return None

    def _transition(
        self,
        timestamp: float,
        new_state: SignalState,
        new_active_lane: str,
        duration: float,
        reason: str,
    ) -> None:
        old_state = self.state
        self.transitions.append(
            TransitionLog(
                timestamp=float(timestamp),
                from_state=old_state,
                to_state=new_state,
                active_lane=self.active_lane,
                next_lane=new_active_lane,
                reason=reason,
            )
        )
        self.state = new_state
        self.active_lane = new_active_lane
        self.remaining_time = float(duration)
        self.elapsed_in_state = 0.0

    def step(
        self,
        timestamp: float,
        standard_counts: Dict[str, float],
        emergency_counts: Dict[str, float],
        predicted_dc_dt: Dict[str, float],
        dt: float = 1.0,
    ) -> SignalState:
        """Advance the state machine by one second (dt=1.0) and return current SignalState."""
        emg_lane = self._select_emergency_lane(emergency_counts)

        # 1. Handle ADAPTIVE_GREEN
        if self.state == SignalState.ADAPTIVE_GREEN:
            # If no preemption is pending yet, recompute adaptive target green every second
            if self.preempt_target_lane is None:
                c_active = float(standard_counts.get(self.active_lane, 0.0))
                dc_active = float(predicted_dc_dt.get(self.active_lane, 0.0))
                target_green = compute_adaptive_green(
                    c_lane=c_active,
                    dc_dt=dc_active,
                    base_green=self.config.base_green,
                    k1=self.config.k1,
                    k2=self.config.k2,
                    min_green=self.config.min_green,
                    max_green=self.config.max_green,
                )
                self.remaining_time = max(0.0, target_green - self.elapsed_in_state)

            # Check for emergency preemption trigger
            if emg_lane is not None and self.preempt_target_lane is None:
                self.resume_clockwise_index = self._next_clockwise_index(self.clockwise_index)
                self.preempt_target_lane = emg_lane

                if emg_lane == self.active_lane:
                    # Emergency is already on the green lane -> switch directly to EMERGENCY_GREEN
                    self._transition(
                        timestamp=timestamp,
                        new_state=SignalState.EMERGENCY_GREEN,
                        new_active_lane=emg_lane,
                        duration=0.0,
                        reason=f"Emergency detected on active green lane {emg_lane}; holding green until E_{emg_lane}==0",
                    )
                    return self.state

                # Emergency on different lane j != i
                if self.remaining_time > self.config.preempt_remaining_threshold:
                    self._transition(
                        timestamp=timestamp,
                        new_state=SignalState.TRUNCATED_GREEN,
                        new_active_lane=self.active_lane,
                        duration=self.config.truncated_green_duration,
                        reason=(
                            f"Emergency on lane {emg_lane}; remaining green ({self.remaining_time:.1f}s) > "
                            f"{self.config.preempt_remaining_threshold:.1f}s -> truncating green to "
                            f"{self.config.truncated_green_duration:.1f}s"
                        ),
                    )
                    return self.state
                else:
                    # <= 4s remaining: let green expire normally without resetting remaining_time
                    self.transitions.append(
                        TransitionLog(
                            timestamp=float(timestamp),
                            from_state=SignalState.ADAPTIVE_GREEN,
                            to_state=SignalState.ADAPTIVE_GREEN,
                            active_lane=self.active_lane,
                            next_lane=emg_lane,
                            reason=(
                                f"Emergency on lane {emg_lane}; remaining green ({self.remaining_time:.1f}s) <= "
                                f"{self.config.preempt_remaining_threshold:.1f}s -> letting green expire normally"
                            ),
                        )
                    )

            # Advance timer in ADAPTIVE_GREEN
            self.elapsed_in_state += dt
            self.remaining_time = max(0.0, self.remaining_time - dt)
            if self.remaining_time <= 0.0:
                reason = (
                    f"Preemption expiry on {self.active_lane} for emergency on {self.preempt_target_lane}"
                    if self.preempt_target_lane is not None
                    else f"Adaptive green expired normally on lane {self.active_lane}"
                )
                self._transition(
                    timestamp=timestamp,
                    new_state=SignalState.AMBER,
                    new_active_lane=self.active_lane,
                    duration=self.config.amber_duration,
                    reason=reason,
                )
            return self.state

        # 2. Handle TRUNCATED_GREEN
        if self.state == SignalState.TRUNCATED_GREEN:
            self.elapsed_in_state += dt
            self.remaining_time = max(0.0, self.remaining_time - dt)
            if self.remaining_time <= 0.0:
                self._transition(
                    timestamp=timestamp,
                    new_state=SignalState.AMBER,
                    new_active_lane=self.active_lane,
                    duration=self.config.amber_duration,
                    reason=f"Truncated green expired on {self.active_lane}; entering AMBER before emergency lane {self.preempt_target_lane}",
                )
            return self.state

        # 3. Handle AMBER
        if self.state == SignalState.AMBER:
            # If an emergency arrives during normal AMBER when not already preempting:
            if emg_lane is not None and self.preempt_target_lane is None:
                self.preempt_target_lane = emg_lane
                self.resume_clockwise_index = self._next_clockwise_index(self.clockwise_index)

            self.elapsed_in_state += dt
            self.remaining_time = max(0.0, self.remaining_time - dt)
            if self.remaining_time <= 0.0:
                self._transition(
                    timestamp=timestamp,
                    new_state=SignalState.ALL_RED,
                    new_active_lane=self.active_lane,
                    duration=self.config.all_red_duration,
                    reason=f"Amber expired on {self.active_lane}; entering ALL_RED clearance",
                )
            return self.state

        # 4. Handle ALL_RED
        if self.state == SignalState.ALL_RED:
            if emg_lane is not None and self.preempt_target_lane is None:
                self.preempt_target_lane = emg_lane
                self.resume_clockwise_index = self._next_clockwise_index(self.clockwise_index)

            self.elapsed_in_state += dt
            self.remaining_time = max(0.0, self.remaining_time - dt)
            if self.remaining_time <= 0.0:
                if self.preempt_target_lane is not None:
                    target = self.preempt_target_lane
                    self._transition(
                        timestamp=timestamp,
                        new_state=SignalState.EMERGENCY_GREEN,
                        new_active_lane=target,
                        duration=0.0,
                        reason=f"ALL_RED expired; granting EMERGENCY_GREEN to lane {target}",
                    )
                else:
                    # Normal or post-preemption clockwise transition
                    if self.resume_clockwise_index is not None:
                        self.clockwise_index = self.resume_clockwise_index
                        self.resume_clockwise_index = None
                        reason_prefix = "Resuming clockwise cycle after preemption at phase"
                    else:
                        self.clockwise_index = self._next_clockwise_index(self.clockwise_index)
                        reason_prefix = "Advancing clockwise cycle to phase"

                    next_lane = self.config.phase_order[self.clockwise_index]
                    c_next = float(standard_counts.get(next_lane, 0.0))
                    dc_next = float(predicted_dc_dt.get(next_lane, 0.0))
                    init_green = compute_adaptive_green(
                        c_lane=c_next,
                        dc_dt=dc_next,
                        base_green=self.config.base_green,
                        k1=self.config.k1,
                        k2=self.config.k2,
                        min_green=self.config.min_green,
                        max_green=self.config.max_green,
                    )
                    self._transition(
                        timestamp=timestamp,
                        new_state=SignalState.ADAPTIVE_GREEN,
                        new_active_lane=next_lane,
                        duration=init_green,
                        reason=f"{reason_prefix} {next_lane} (initial adaptive green={init_green:.1f}s)",
                    )
            return self.state

        # 5. Handle EMERGENCY_GREEN
        if self.state == SignalState.EMERGENCY_GREEN:
            self.elapsed_in_state += dt
            e_j = float(emergency_counts.get(self.active_lane, 0.0))
            if e_j <= 0.0:
                cleared_lane = self.active_lane
                self.preempt_target_lane = None
                # Ensure resume_clockwise_index is set to (i+1)
                if self.resume_clockwise_index is None:
                    self.resume_clockwise_index = self._next_clockwise_index(self.clockwise_index)
                self._transition(
                    timestamp=timestamp,
                    new_state=SignalState.AMBER,
                    new_active_lane=cleared_lane,
                    duration=self.config.amber_duration,
                    reason=(
                        f"Emergency cleared on lane {cleared_lane} (E_{cleared_lane}==0); "
                        f"transitioning via AMBER/ALL_RED to resume clockwise phase "
                        f"{self.config.phase_order[self.resume_clockwise_index]}"
                    ),
                )
            return self.state

        return self.state

    def get_lane_lights(self) -> Dict[str, LightColor]:
        """Return the current traffic signal light color (GREEN, AMBER, RED) for all 4 approaches."""
        lights: Dict[str, LightColor] = {lane: LightColor.RED for lane in self.config.phase_order}
        if self.state in (
            SignalState.ADAPTIVE_GREEN,
            SignalState.TRUNCATED_GREEN,
            SignalState.EMERGENCY_GREEN,
        ):
            lights[self.active_lane] = LightColor.GREEN
        elif self.state == SignalState.AMBER:
            lights[self.active_lane] = LightColor.AMBER
        return lights
