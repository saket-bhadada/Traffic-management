"""Compare Adaptive + Emergency Preemption controller vs Fixed-Timer Baseline and export CSV + Markdown summary."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml

from src.control.state_machine import StateMachineConfig
from src.evaluation.metrics import EvaluationMetrics, evaluate_simulation_run
from src.prediction.lstm_model import OnlineQueuePredictor
from src.prediction.train_lstm import load_lstm_config, train_lstm_model
from src.sim.baseline_fixed import FixedTimerConfig
from src.sim.intersection_sim import (
    SimulationRunResult,
    run_adaptive_simulation,
    run_fixed_baseline_simulation,
)
from src.sim.synthetic_traffic import SyntheticTrafficConfig, generate_traffic_trace

logger = logging.getLogger(__name__)


def load_controller_configs(
    yaml_path: Path, base_dir: Path | None = None
) -> Tuple[StateMachineConfig, FixedTimerConfig, Dict[str, Any]]:
    """Load StateMachineConfig, FixedTimerConfig, and simulation settings from configs/controller.yaml."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    root = base_dir if base_dir is not None else yaml_path.parent.parent
    phases = tuple(str(p) for p in raw.get("phase_order", ["N", "E", "S", "W"]))

    sm_cfg = StateMachineConfig(
        phase_order=phases,  # type: ignore[arg-type]
        base_green=float(raw.get("base_green", 15.0)),
        k1=float(raw.get("k1", 1.5)),
        k2=float(raw.get("k2", 4.0)),
        min_green=float(raw.get("min_green", 10.0)),
        max_green=float(raw.get("max_green", 60.0)),
        amber_duration=float(raw.get("amber_duration", 3.0)),
        all_red_duration=float(raw.get("all_red_duration", 1.0)),
        preempt_remaining_threshold=float(raw.get("preempt_remaining_threshold", 4.0)),
        truncated_green_duration=float(raw.get("truncated_green_duration", 2.0)),
    )

    fixed_cfg = FixedTimerConfig(
        phase_order=phases,  # type: ignore[arg-type]
        fixed_green_duration=float(raw.get("fixed_green_duration", 30.0)),
        amber_duration=float(raw.get("amber_duration", 3.0)),
        all_red_duration=float(raw.get("all_red_duration", 1.0)),
    )

    sim_raw = raw.get("simulation", {})
    sim_settings = {
        "seed": int(raw.get("seed", 42)),
        "duration_seconds": int(sim_raw.get("duration_seconds", 1800)),
        "saturation_discharge_rate": float(sim_raw.get("saturation_discharge_rate", 1.5)),
        "comparison_csv_path": (
            root / str(sim_raw.get("comparison_csv_path", "artifacts/sim_comparison.csv"))
        ).resolve(),
        "transitions_log_path": (
            root / str(sim_raw.get("transitions_log_path", "artifacts/controller_transitions.csv"))
        ).resolve(),
        "feature_log_csv": (root / "artifacts" / "feature_vectors_xt.csv").resolve(),
    }
    return sm_cfg, fixed_cfg, sim_settings


def run_comparison(
    controller_yaml: Path,
    ensure_lstm_trained: bool = True,
) -> Tuple[pd.DataFrame, SimulationRunResult, SimulationRunResult]:
    """Run both Adaptive+Preemption and Fixed-Timer Baseline on identical seeded traffic traces and save CSV."""
    sm_cfg, fixed_cfg, sim_settings = load_controller_configs(controller_yaml)
    lstm_cfg = load_lstm_config(controller_yaml)

    if ensure_lstm_trained and not lstm_cfg.weights_path.exists():
        logger.info("Training LSTM predictor on synthetic data before running simulation...")
        train_lstm_model(lstm_cfg)

    predictor = OnlineQueuePredictor(config=lstm_cfg)

    trace = generate_traffic_trace(
        SyntheticTrafficConfig(
            duration_seconds=int(sim_settings["duration_seconds"]),
            seed=int(sim_settings["seed"]),
        )
    )

    adaptive_res = run_adaptive_simulation(
        trace=trace,
        sm_config=sm_cfg,
        predictor=predictor,
        saturation_discharge_rate=float(sim_settings["saturation_discharge_rate"]),
        feature_log_csv=sim_settings["feature_log_csv"],
        transitions_log_csv=sim_settings["transitions_log_path"],
    )

    baseline_res = run_fixed_baseline_simulation(
        trace=trace,
        fixed_config=fixed_cfg,
        saturation_discharge_rate=float(sim_settings["saturation_discharge_rate"]),
    )

    m_adapt: EvaluationMetrics = evaluate_simulation_run(adaptive_res)
    m_base: EvaluationMetrics = evaluate_simulation_run(baseline_res)

    rows = [
        {
            "controller": m_base.controller_name,
            "average_wait_time_s": m_base.average_wait_time_sec,
            "idle_green_time_s": m_base.idle_green_time_sec,
            "throughput_vehicles": m_base.throughput_vehicles,
            "emergency_time_to_green_s": m_base.emergency_time_to_green_sec,
            "emergency_events": m_base.emergency_events_count,
        },
        {
            "controller": m_adapt.controller_name,
            "average_wait_time_s": m_adapt.average_wait_time_sec,
            "idle_green_time_s": m_adapt.idle_green_time_sec,
            "throughput_vehicles": m_adapt.throughput_vehicles,
            "emergency_time_to_green_s": m_adapt.emergency_time_to_green_sec,
            "emergency_events": m_adapt.emergency_events_count,
        },
    ]
    df = pd.DataFrame(rows)
    csv_out: Path = sim_settings["comparison_csv_path"]
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    logger.info("Saved simulation comparison CSV to %s", csv_out)
    return df, adaptive_res, baseline_res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "controller.yaml"
    df_comp, _, _ = run_comparison(cfg_path)
    print(df_comp.to_string(index=False))
