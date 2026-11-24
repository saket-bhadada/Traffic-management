"""Pytest suite for the deterministic signal controller and emergency preemption state machine."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.control.controller import TrafficSignalController
from src.control.state_machine import (
    LightColor,
    PreemptionStateMachine,
    SignalState,
    StateMachineConfig,
    compute_adaptive_green,
)


def test_no_ml_imports_in_control_package() -> None:
    """Verify that src/control/ contains zero ML library imports (torch, ultralytics, cv2, etc.)."""
    control_dir = Path(__file__).resolve().parents[1] / "src" / "control"
    forbidden = {"torch", "ultralytics", "cv2", "sklearn", "tensorflow", "roboflow"}
    for py_file in control_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    assert root_pkg not in forbidden, f"Forbidden import {root_pkg} in {py_file}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_pkg = node.module.split(".")[0]
                assert root_pkg not in forbidden, f"Forbidden import {root_pkg} in {py_file}"


def test_adaptive_green_timer_clamping_min_and_max() -> None:
    """Verify timer clamping at min_green (10s) and max_green (60s) and negative dC/dt handling."""
    # Negative dC/dt and 0 vehicles -> base=15s
    g_base = compute_adaptive_green(c_lane=0.0, dc_dt=-3.0, base_green=15.0, k1=1.5, k2=4.0, min_green=10.0, max_green=60.0)
    assert g_base == pytest.approx(15.0)

    # Low base_green so raw value falls below min_green -> clamps to min_green
    g_min = compute_adaptive_green(c_lane=0.0, dc_dt=0.0, base_green=5.0, k1=1.5, k2=4.0, min_green=10.0, max_green=60.0)
    assert g_min == pytest.approx(10.0)

    # Large vehicle count and positive growth rate -> clamps to max_green (60s)
    g_max = compute_adaptive_green(c_lane=40.0, dc_dt=5.0, base_green=15.0, k1=1.5, k2=4.0, min_green=10.0, max_green=60.0)
    assert g_max == pytest.approx(60.0)

    # Exact mid-range formula check: 15 + 1.5*10 + 4.0*2.0 = 15 + 15 + 8 = 38.0s
    g_mid = compute_adaptive_green(c_lane=10.0, dc_dt=2.0, base_green=15.0, k1=1.5, k2=4.0, min_green=10.0, max_green=60.0)
    assert g_mid == pytest.approx(38.0)


def test_normal_clockwise_cycle_progression() -> None:
    """Verify normal clockwise cycle N -> E -> S -> W -> N without emergency preemption."""
    cfg = StateMachineConfig(
        base_green=10.0,
        min_green=10.0,
        max_green=60.0,
        amber_duration=3.0,
        all_red_duration=1.0,
    )
    sm = PreemptionStateMachine(config=cfg)
    zero_c = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    zero_e = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    zero_dc = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}

    t = 0.0
    expected_cycle = ["N", "E", "S", "W", "N"]
    visited_greens = [sm.active_lane]

    for expected_next in expected_cycle[1:]:
        # Step through 10s of ADAPTIVE_GREEN
        for _ in range(10):
            state = sm.step(t, zero_c, zero_e, zero_dc)
            t += 1.0
        assert state == SignalState.AMBER

        # Step through 3s of AMBER
        for _ in range(3):
            state = sm.step(t, zero_c, zero_e, zero_dc)
            t += 1.0
        assert state == SignalState.ALL_RED

        # Step through 1s of ALL_RED -> transitions to next clockwise phase in ADAPTIVE_GREEN
        state = sm.step(t, zero_c, zero_e, zero_dc)
        t += 1.0
        assert state == SignalState.ADAPTIVE_GREEN
        assert sm.active_lane == expected_next
        visited_greens.append(sm.active_lane)

    assert visited_greens == expected_cycle


def test_preemption_truncation_branch_when_remaining_greater_than_4s() -> None:
    """Verify >4s remaining green branch:
    ADAPTIVE_GREEN (on N, rem > 4s) -> TRUNCATED_GREEN (2s) -> AMBER (3s) -> ALL_RED (1s)
    -> EMERGENCY_GREEN (on S, holds while E_S > 0) -> release when E_S == 0 -> AMBER (3s)
    -> ALL_RED (1s) -> resumes clockwise cycle at phase i+1 ('E').
    """
    cfg = StateMachineConfig(
        base_green=15.0,
        min_green=10.0,
        max_green=60.0,
        amber_duration=3.0,
        all_red_duration=1.0,
        preempt_remaining_threshold=4.0,
        truncated_green_duration=2.0,
    )
    sm = PreemptionStateMachine(config=cfg)
    std_c = {"N": 4.0, "E": 2.0, "S": 3.0, "W": 1.0}
    zero_dc = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}

    # Step 1s on N with no emergency (remaining green is ~20s > 4s)
    sm.step(0.0, std_c, {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.active_lane == "N"
    assert sm.remaining_time > 4.0

    # At t=1.0, emergency detected on lane S -> must transition to TRUNCATED_GREEN (2s)
    emg_s = {"N": 0.0, "E": 0.0, "S": 1.0, "W": 0.0}
    state = sm.step(1.0, std_c, emg_s, zero_dc)
    assert state == SignalState.TRUNCATED_GREEN
    assert sm.active_lane == "N"
    assert sm.remaining_time == pytest.approx(2.0)

    # Step 2s through TRUNCATED_GREEN -> transitions to AMBER (3s)
    sm.step(2.0, std_c, emg_s, zero_dc)
    assert sm.state == SignalState.TRUNCATED_GREEN
    sm.step(3.0, std_c, emg_s, zero_dc)
    assert sm.state == SignalState.AMBER
    assert sm.remaining_time == pytest.approx(3.0)

    # Step 3s through AMBER -> transitions to ALL_RED (1s)
    sm.step(4.0, std_c, emg_s, zero_dc)
    sm.step(5.0, std_c, emg_s, zero_dc)
    sm.step(6.0, std_c, emg_s, zero_dc)
    assert sm.state == SignalState.ALL_RED
    assert sm.remaining_time == pytest.approx(1.0)

    # Step 1s through ALL_RED -> transitions to EMERGENCY_GREEN on lane S
    sm.step(7.0, std_c, emg_s, zero_dc)
    assert sm.state == SignalState.EMERGENCY_GREEN
    assert sm.active_lane == "S"
    assert sm.get_lane_lights()["S"] == LightColor.GREEN

    # Hold EMERGENCY_GREEN as long as E_S > 0
    for tick in (8.0, 9.0, 10.0):
        sm.step(tick, std_c, emg_s, zero_dc)
        assert sm.state == SignalState.EMERGENCY_GREEN
        assert sm.active_lane == "S"

    # At t=11.0, E_S returns to 0 -> transitions to AMBER (3s)
    no_emg = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    sm.step(11.0, std_c, no_emg, zero_dc)
    assert sm.state == SignalState.AMBER
    assert sm.active_lane == "S"

    # Step 3s AMBER + 1s ALL_RED -> resumes clockwise at i+1 = 'E' (since i was 'N')
    sm.step(12.0, std_c, no_emg, zero_dc)
    sm.step(13.0, std_c, no_emg, zero_dc)
    sm.step(14.0, std_c, no_emg, zero_dc)
    assert sm.state == SignalState.ALL_RED
    sm.step(15.0, std_c, no_emg, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.active_lane == "E"


def test_preemption_less_than_or_equal_4s_branch_expires_normally() -> None:
    """Verify <=4s remaining green branch:
    When emergency arrives with remaining green <= 4s, controller does NOT enter TRUNCATED_GREEN;
    it lets green expire normally, then goes AMBER (3s) -> ALL_RED (1s) -> EMERGENCY_GREEN on lane j.
    """
    cfg = StateMachineConfig(
        base_green=10.0,
        min_green=10.0,
        max_green=60.0,
        amber_duration=3.0,
        all_red_duration=1.0,
        preempt_remaining_threshold=4.0,
        truncated_green_duration=2.0,
    )
    sm = PreemptionStateMachine(config=cfg)
    zero_c = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    zero_e = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    zero_dc = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}

    # Advance 7 seconds on lane N (10s total green -> 3s remaining at start of t=7)
    for t in range(7):
        sm.step(float(t), zero_c, zero_e, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.remaining_time == pytest.approx(3.0)

    # Emergency detected on W at t=7.0 with 3.0s remaining (<= 4.0s threshold)
    emg_w = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 1.0}
    sm.step(7.0, zero_c, emg_w, zero_dc)
    # Must stay in ADAPTIVE_GREEN (not TRUNCATED_GREEN) and count down remaining 3s -> 2s
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.remaining_time == pytest.approx(2.0)

    sm.step(8.0, zero_c, emg_w, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.remaining_time == pytest.approx(1.0)

    # At t=9.0, the 3s remaining green expires normally -> transitions to AMBER
    sm.step(9.0, zero_c, emg_w, zero_dc)
    assert sm.state == SignalState.AMBER

    # 3s AMBER + 1s ALL_RED -> EMERGENCY_GREEN on W
    sm.step(10.0, zero_c, emg_w, zero_dc)
    sm.step(11.0, zero_c, emg_w, zero_dc)
    sm.step(12.0, zero_c, emg_w, zero_dc)
    assert sm.state == SignalState.ALL_RED
    sm.step(13.0, zero_c, emg_w, zero_dc)
    assert sm.state == SignalState.EMERGENCY_GREEN
    assert sm.active_lane == "W"

    # Release when E_W == 0 and verify clockwise resumption at phase (i+1) = 'E'
    sm.step(14.0, zero_c, zero_e, zero_dc)
    assert sm.state == SignalState.AMBER
    sm.step(15.0, zero_c, zero_e, zero_dc)
    sm.step(16.0, zero_c, zero_e, zero_dc)
    sm.step(17.0, zero_c, zero_e, zero_dc)
    assert sm.state == SignalState.ALL_RED
    sm.step(18.0, zero_c, zero_e, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.active_lane == "E"


def test_car_count_buffered_switch_waits_for_min_green_and_buffer() -> None:
    """Verify car-count switch enforces min_green + sustained buffer, resets on transient blips,
    and switches via AMBER (3s) -> ALL_RED (1s) to the highest-count lane."""
    cfg = StateMachineConfig(
        base_green=15.0,
        k1=1.5,
        min_green=10.0,
        max_green=60.0,
        amber_duration=3.0,
        all_red_duration=1.0,
        car_count_switching_enabled=True,
        car_count_switch_margin=5.0,
        car_count_switch_buffer=6.0,
    )
    sm = PreemptionStateMachine(config=cfg)
    zero_e = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    zero_dc = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}

    # Transient spike on S for 3 seconds (below 6s buffer) -> must NOT switch and must reset buffer
    heavy_s = {"N": 6.0, "E": 1.0, "S": 14.0, "W": 1.0}
    balanced = {"N": 6.0, "E": 1.0, "S": 6.0, "W": 1.0}
    for t in range(3):
        state = sm.step(float(t), heavy_s, zero_e, zero_dc)
        assert state == SignalState.ADAPTIVE_GREEN
        assert sm.demand_buffer_elapsed == pytest.approx(float(t + 1))

    # Drop counts back within margin at t=3 -> buffer must reset to 0
    sm.step(3.0, balanced, zero_e, zero_dc)
    assert sm.demand_buffer_elapsed == pytest.approx(0.0)
    assert sm.demand_switch_target is None

    # Advance up to t=7 (elapsed=8) with balanced counts, then hold heavy_s across t=8..13
    for t in range(4, 8):
        sm.step(float(t), balanced, zero_e, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN

    # t=8,9: crosses min_green=10 at t=9, but buffer only at 2s -> still GREEN
    sm.step(8.0, heavy_s, zero_e, zero_dc)
    sm.step(9.0, heavy_s, zero_e, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.elapsed_in_state == pytest.approx(10.0)
    assert sm.demand_buffer_elapsed == pytest.approx(2.0)

    # t=10,11,12: buffer at 3,4,5s -> still GREEN
    for t in (10, 11, 12):
        state = sm.step(float(t), heavy_s, zero_e, zero_dc)
        assert state == SignalState.ADAPTIVE_GREEN

    # t=13: buffer reaches 6s (>= 6s) and elapsed >= 10s -> triggers AMBER (3s)
    state = sm.step(13.0, heavy_s, zero_e, zero_dc)
    assert state == SignalState.AMBER
    assert sm.active_lane == "N"
    assert sm.remaining_time == pytest.approx(3.0)

    # 3s AMBER + 1s ALL_RED -> switches directly to S (skipping E)
    for t in (14.0, 15.0, 16.0):
        sm.step(t, heavy_s, zero_e, zero_dc)
    assert sm.state == SignalState.ALL_RED
    sm.step(17.0, heavy_s, zero_e, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.active_lane == "S"


def test_car_count_empty_lane_switches_early_after_buffer() -> None:
    """Verify an empty green lane (C_i == 0) with waiting cars on S switches early after min_green + buffer."""
    cfg = StateMachineConfig(
        base_green=25.0,
        min_green=10.0,
        max_green=60.0,
        amber_duration=3.0,
        all_red_duration=1.0,
        car_count_switching_enabled=True,
        car_count_switch_margin=5.0,
        car_count_switch_buffer=6.0,
    )
    sm = PreemptionStateMachine(config=cfg)
    zero_e = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    zero_dc = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    counts = {"N": 0.0, "E": 1.0, "S": 3.0, "W": 0.0}

    # During first 9 ticks (t=0..8), must remain GREEN due to min_green=10s
    for t in range(9):
        state = sm.step(float(t), counts, zero_e, zero_dc)
        assert state == SignalState.ADAPTIVE_GREEN

    # At t=9 (10th tick), elapsed=10s >= min_green and buffer=10s >= 6s -> enters AMBER
    state = sm.step(9.0, counts, zero_e, zero_dc)
    assert state == SignalState.AMBER

    for t in (10.0, 11.0, 12.0):
        sm.step(t, counts, zero_e, zero_dc)
    assert sm.state == SignalState.ALL_RED
    sm.step(13.0, counts, zero_e, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.active_lane == "S"


def test_car_count_starvation_guard_prioritizes_long_waiting_lane() -> None:
    """Verify a waiting lane reaching max_lane_red_wait is prioritized ahead of a slightly busier lane."""
    cfg = StateMachineConfig(
        base_green=15.0,
        min_green=10.0,
        max_green=60.0,
        amber_duration=3.0,
        all_red_duration=1.0,
        car_count_switching_enabled=True,
        car_count_switch_margin=5.0,
        car_count_switch_buffer=6.0,
        max_lane_red_wait=20.0,
    )
    sm = PreemptionStateMachine(config=cfg)
    sm.lane_red_wait["W"] = 19.0
    zero_e = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    zero_dc = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    # S has more cars (8 vs W's 2), while N has 6 cars (so neither beats margin +5 on raw count),
    # yet W hits max_lane_red_wait (20s) on the first tick!
    counts = {"N": 6.0, "E": 1.0, "S": 8.0, "W": 2.0}

    for t in range(10):
        state = sm.step(float(t), counts, zero_e, zero_dc)
        if t < 9:
            assert state == SignalState.ADAPTIVE_GREEN
    assert sm.state == SignalState.AMBER

    for t in (10.0, 11.0, 12.0):
        sm.step(t, counts, zero_e, zero_dc)
    sm.step(13.0, counts, zero_e, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.active_lane == "W"


def test_emergency_preemption_overrides_pending_car_count_buffer() -> None:
    """Verify emergency preemption clears pending car-count buffer and resumes clockwise at (i+1)."""
    cfg = StateMachineConfig(
        base_green=15.0,
        min_green=10.0,
        max_green=60.0,
        amber_duration=3.0,
        all_red_duration=1.0,
        preempt_remaining_threshold=4.0,
        truncated_green_duration=2.0,
        car_count_switching_enabled=True,
        car_count_switch_margin=5.0,
        car_count_switch_buffer=6.0,
    )
    sm = PreemptionStateMachine(config=cfg)
    heavy_s = {"N": 6.0, "E": 1.0, "S": 18.0, "W": 1.0}
    zero_e = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    zero_dc = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}

    # Build up 3s of car-count switch buffer targeting S
    for t in range(3):
        sm.step(float(t), heavy_s, zero_e, zero_dc)
    assert sm.demand_switch_target == "S"
    assert sm.demand_buffer_elapsed == pytest.approx(3.0)

    # Emergency arrives on W at t=3 -> clears car-count buffer and enters TRUNCATED_GREEN
    emg_w = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 1.0}
    state = sm.step(3.0, heavy_s, emg_w, zero_dc)
    assert state == SignalState.TRUNCATED_GREEN
    assert sm.demand_switch_target is None
    assert sm.demand_buffer_elapsed == pytest.approx(0.0)
    assert sm.next_demand_lane is None

    # Complete truncation (2s) -> AMBER (3s) -> ALL_RED (1s) -> EMERGENCY_GREEN on W
    for t in range(4, 10):
        sm.step(float(t), heavy_s, emg_w, zero_dc)
    assert sm.state == SignalState.EMERGENCY_GREEN
    assert sm.active_lane == "W"

    # Clear emergency on W -> AMBER (3s) -> ALL_RED (1s) -> strictly resumes (i+1)='E' despite S=18
    sm.step(10.0, heavy_s, zero_e, zero_dc)
    assert sm.state == SignalState.AMBER
    for t in (11.0, 12.0, 13.0):
        sm.step(t, heavy_s, zero_e, zero_dc)
    assert sm.state == SignalState.ALL_RED
    sm.step(14.0, heavy_s, zero_e, zero_dc)
    assert sm.state == SignalState.ADAPTIVE_GREEN
    assert sm.active_lane == "E"
