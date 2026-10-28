"""Fixed-timer clockwise baseline controller with identical amber (3s) and all-red (1s) intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from src.control.state_machine import LightColor, SignalState


@dataclass(frozen=True)
class FixedTimerConfig:
    """Configuration for the fixed-time clockwise baseline controller."""

    phase_order: Tuple[str, str, str, str] = ("N", "E", "S", "W")
    fixed_green_duration: float = 30.0
    amber_duration: float = 3.0
    all_red_duration: float = 1.0


@dataclass
class FixedTimerBaselineController:
    """Fixed-timer signal controller cycling clockwise N -> E -> S -> W with fixed green, amber, and all-red."""

    config: FixedTimerConfig = field(default_factory=FixedTimerConfig)
    state: SignalState = SignalState.ADAPTIVE_GREEN
    clockwise_index: int = 0
    active_lane: str = "N"
    remaining_time: float = 30.0

    def __post_init__(self) -> None:
        self.active_lane = self.config.phase_order[self.clockwise_index]
        self.remaining_time = float(self.config.fixed_green_duration)

    def step(self, dt: float = 1.0) -> Dict[str, LightColor]:
        """Advance fixed timer by dt seconds and return per-approach LightColor."""
        self.remaining_time = max(0.0, self.remaining_time - dt)

        if self.remaining_time <= 0.0:
            if self.state == SignalState.ADAPTIVE_GREEN:
                self.state = SignalState.AMBER
                self.remaining_time = float(self.config.amber_duration)
            elif self.state == SignalState.AMBER:
                self.state = SignalState.ALL_RED
                self.remaining_time = float(self.config.all_red_duration)
            elif self.state == SignalState.ALL_RED:
                self.clockwise_index = (self.clockwise_index + 1) % len(self.config.phase_order)
                self.active_lane = self.config.phase_order[self.clockwise_index]
                self.state = SignalState.ADAPTIVE_GREEN
                self.remaining_time = float(self.config.fixed_green_duration)

        lights: Dict[str, LightColor] = {lane: LightColor.RED for lane in self.config.phase_order}
        if self.state == SignalState.ADAPTIVE_GREEN:
            lights[self.active_lane] = LightColor.GREEN
        elif self.state == SignalState.AMBER:
            lights[self.active_lane] = LightColor.AMBER
        return lights
