"""Main CLI entry point for Adaptive Traffic Signal Control with Emergency Preemption.

Usage:
  python main.py --mode sim
  python main.py --mode video
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.control.controller import TrafficSignalController
from src.detection.detector import Detector, DetectorConfig
from src.evaluation.compare import load_controller_configs, run_comparison
from src.features.feature_vector import FeatureCSVLogger, extract_x_t_from_detections
from src.features.lane_density import LaneConfig
from src.feeds.base import LANES
from src.feeds.video_file_feed import (
    VideoFeedConfig,
    VideoFileFeed,
    generate_synthetic_four_way_videos,
)
from src.prediction.lstm_model import OnlineQueuePredictor
from src.prediction.train_lstm import load_lstm_config, train_lstm_model

logger = logging.getLogger(__name__)


def seed_all(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sim_mode(root_dir: Path) -> None:
    """Execute 1-second-step intersection simulation and compare against fixed-timer baseline."""
    controller_yaml = root_dir / "configs" / "controller.yaml"
    df_comp, adaptive_res, _ = run_comparison(controller_yaml, ensure_lstm_trained=True)
    logger.info("Simulation complete (%d seconds).", adaptive_res.duration_seconds)
    print("\n=== Simulation Comparison: Fixed-Timer Baseline vs Adaptive + Preemption ===")
    print(df_comp.to_string(index=False))


def run_video_mode(
    root_dir: Path,
    video_dir: Path | None = None,
    max_ticks: int = 30,
) -> None:
    """Run synchronized 1 FPS 4-way video pipeline: YOLOv8 Detector -> ROI Counts -> X_t -> LSTM -> Controller."""
    datasets_yaml = root_dir / "configs" / "datasets.yaml"
    lanes_yaml = root_dir / "configs" / "lanes.yaml"
    controller_yaml = root_dir / "configs" / "controller.yaml"

    det_cfg = DetectorConfig.from_yaml(datasets_yaml, base_dir=root_dir)
    lane_cfg = LaneConfig.from_yaml(lanes_yaml, base_dir=root_dir)
    sm_cfg, _, sim_settings = load_controller_configs(controller_yaml, base_dir=root_dir)
    lstm_cfg = load_lstm_config(controller_yaml, base_dir=root_dir)

    seed_all(det_cfg.seed)

    if not lstm_cfg.weights_path.exists():
        train_lstm_model(lstm_cfg)

    target_vdir = video_dir if video_dir is not None else (root_dir / "artifacts" / "demo_videos")
    video_paths: Dict[str, Path] = {
        lane: target_vdir / f"approach_{lane}.mp4" for lane in LANES
    }
    if not all(p.exists() for p in video_paths.values()):
        logger.info("Generating 4 synchronized sample approach videos in %s...", target_vdir)
        generate_synthetic_four_way_videos(
            output_dir=target_vdir,
            duration_seconds=max_ticks,
            fps=5,
            seed=det_cfg.seed,
        )

    detector = Detector(det_cfg)
    predictor = OnlineQueuePredictor(lstm_cfg)
    csv_logger = FeatureCSVLogger(lane_cfg.feature_log_csv, overwrite=True)
    controller = TrafficSignalController(
        config=sm_cfg,
        transitions_csv_path=sim_settings["transitions_log_path"],
    )

    feed = VideoFileFeed(
        VideoFeedConfig(video_paths=video_paths, target_fps=1.0, max_ticks=max_ticks)
    )

    print("\n=== Running 4-Way Video Mode (1 FPS Synchronized) ===")
    try:
        for timestamp, frames_by_lane in feed:
            detections_by_lane = {
                lane: detector.detect(frames_by_lane[lane]) for lane in LANES
            }
            record = extract_x_t_from_detections(
                detections_by_lane=detections_by_lane,
                lane_config=lane_cfg,
                timestamp=timestamp,
            )
            csv_logger.log(record)

            pred_dc_dt = predictor.push_and_predict(record.x_t)
            decision = controller.update(
                timestamp=timestamp,
                x_t=record.x_t,
                predicted_dc_dt=pred_dc_dt,
                dt=1.0,
            )
            c_str = ", ".join(f"{lane}:{int(record.x_t[i])}" for i, lane in enumerate(LANES))
            e_str = ", ".join(f"{lane}:{int(record.x_t[i+4])}" for i, lane in enumerate(LANES))
            print(
                f"[t={timestamp:5.1f}s] State={decision.state.value:<16} "
                f"Active={decision.active_lane} (rem={decision.remaining_time:4.1f}s) | "
                f"C=[{c_str}] | E=[{e_str}]"
            )
    finally:
        feed.close()

    logger.info("Logged X_t trajectory to %s", lane_cfg.feature_log_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Traffic Signal Control with Emergency Preemption"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sim", "video"],
        help="Execution mode: 'sim' (1s intersection simulation) or 'video' (4-way video detector pipeline)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Optional directory containing approach_N.mp4, approach_E.mp4, approach_S.mp4, approach_W.mp4",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=25,
        help="Number of 1-FPS ticks to process in video mode (default: 25)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.mode == "sim":
        run_sim_mode(ROOT_DIR)
    elif args.mode == "video":
        vdir = Path(args.video_dir).resolve() if args.video_dir else None
        run_video_mode(ROOT_DIR, video_dir=vdir, max_ticks=args.max_ticks)
