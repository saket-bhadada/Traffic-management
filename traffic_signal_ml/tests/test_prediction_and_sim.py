"""Pytest suite for data unification, leak-free split grouping, LSTM predictor, and simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.control.state_machine import StateMachineConfig
from src.data.build_splits import SplitRatios, assign_groups_to_splits, extract_source_group_id
from src.data.unify_labels import ClassMapConfig, remap_yolo_label_lines
from src.evaluation.metrics import evaluate_simulation_run
from src.prediction.dataset import QueueGrowthDataset, build_sliding_windows
from src.prediction.lstm_model import LSTMConfig, OnlineQueuePredictor, QueueGrowthLSTM
from src.sim.baseline_fixed import FixedTimerConfig
from src.sim.intersection_sim import run_adaptive_simulation, run_fixed_baseline_simulation
from src.sim.synthetic_traffic import SyntheticTrafficConfig, generate_traffic_trace


def test_augmentation_group_leakage_prevention_and_emergency_in_every_split() -> None:
    """Verify augmented copies of the same source image stay in the same split and every split gets emergencies."""
    stem_a1 = "ph_ambulances__amb_012_jpg.rf.1234567890abcdef"
    stem_a2 = "ph_ambulances__amb_012_jpg.rf.fedcba0987654321"
    assert extract_source_group_id(stem_a1) == extract_source_group_id(stem_a2)

    group_emg = {
        f"emg_group_{i}": True for i in range(6)
    }
    group_std = {
        f"std_group_{i}": False for i in range(20)
    }
    combined = {**group_emg, **group_std}
    assignments = assign_groups_to_splits(combined, SplitRatios(0.7, 0.15, 0.15), seed=42)

    # Check that train, val, and test each have at least 1 emergency group
    splits_with_emg = {assignments[g] for g in group_emg}
    assert splits_with_emg == {"train", "val", "test"}


def test_class_map_unification_drops_unmapped() -> None:
    """Verify class_map.yaml maps standard vehicles to 0, emergency vehicles to 1, and drops unmapped."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "class_map.yaml"
    cmap = ClassMapConfig.from_yaml(cfg_path)
    assert cmap.map_label("car") == 0
    assert cmap.map_label("auto-rickshaw") == 0
    assert cmap.map_label("multi-axle truck") == 0
    assert cmap.map_label("ambulance") == 1
    assert cmap.map_label("fire engine") == 1
    assert cmap.map_label("pedestrian") is None

    src_names = {0: "car", 1: "pedestrian", 2: "ambulance"}
    raw_lines = [
        "0 0.5 0.5 0.2 0.2\n",
        "1 0.3 0.3 0.1 0.1\n",
        "2 0.6 0.6 0.3 0.3\n",
    ]
    dropped: dict[str, int] = {}
    remapped = remap_yolo_label_lines(raw_lines, src_names, cmap, dropped)
    assert len(remapped) == 2
    assert remapped[0].startswith("0 ")
    assert remapped[1].startswith("1 ")
    assert dropped == {"pedestrian": 1}


def test_lstm_sliding_window_and_forward_shapes() -> None:
    """Verify 2-layer LSTM accepts (B, W, 8) windows and outputs (B, 4) dC/dt predictions."""
    x_series = np.random.default_rng(42).uniform(0, 10, size=(35, 8)).astype(np.float32)
    windows, targets = build_sliding_windows(x_series, window_size=10)
    assert windows.shape == (25, 10, 8)
    assert targets.shape == (25, 4)

    ds = QueueGrowthDataset(x_series, window_size=10)
    assert len(ds) == 25

    model = QueueGrowthLSTM(input_size=8, hidden_size=32, num_layers=2, output_size=4)
    out = model(torch.from_numpy(windows[:4]))
    assert out.shape == (4, 4)

    online = OnlineQueuePredictor(LSTMConfig(window_size=10, hidden_size=32), model=model)
    pred_single = online.push_and_predict(x_series[0])
    assert pred_single.shape == (4,)


def test_simulation_adaptive_vs_fixed_baseline() -> None:
    """Verify 1s simulator runs both controllers deterministically and Adaptive improves emergency time-to-green."""
    trace = generate_traffic_trace(SyntheticTrafficConfig(duration_seconds=300, seed=42))
    sm_cfg = StateMachineConfig()
    fixed_cfg = FixedTimerConfig()

    adapt_res = run_adaptive_simulation(trace, sm_cfg)
    base_res = run_fixed_baseline_simulation(trace, fixed_cfg)

    m_adapt = evaluate_simulation_run(adapt_res)
    m_base = evaluate_simulation_run(base_res)

    assert m_adapt.throughput_vehicles > 0
    assert m_base.throughput_vehicles > 0
    assert m_adapt.emergency_time_to_green_sec <= m_base.emergency_time_to_green_sec
