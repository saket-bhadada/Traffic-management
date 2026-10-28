"""Streamlit dashboard: 4 camera feeds with boxes, current signal per approach, live X_t, predicted dC/dt, and emergency preemption banner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.control.controller import ControllerDecision, TrafficSignalController
from src.control.state_machine import LightColor, SignalState
from src.detection.detector import DetectionOutput, Detector, DetectorConfig
from src.evaluation.compare import load_controller_configs
from src.features.feature_vector import XT_COLUMNS, extract_x_t_from_detections
from src.features.lane_density import LaneConfig
from src.feeds.base import LANES
from src.feeds.video_file_feed import VideoFeedConfig, VideoFileFeed, generate_synthetic_four_way_videos
from src.prediction.lstm_model import OnlineQueuePredictor
from src.prediction.train_lstm import load_lstm_config


def annotate_frame(
    frame: np.ndarray,
    det: DetectionOutput,
    polygon: np.ndarray,
    lane: str,
    light: LightColor,
) -> np.ndarray:
    """Draw lane ROI polygon, detected bounding boxes, and signal aspect indicator onto a frame."""
    vis = frame.copy()
    poly_i32 = polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(vis, [poly_i32], isClosed=True, color=(255, 255, 0), thickness=2)

    for i in range(len(det.boxes)):
        x1, y1, x2, y2 = [int(v) for v in det.boxes[i]]
        cls_id = int(det.class_ids[i])
        conf = float(det.confidences[i])
        color = (0, 0, 255) if cls_id == 1 else (0, 215, 255)
        label = f"{'EMERGENCY' if cls_id == 1 else 'VEHICLE'} {conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )

    sig_bgr = {
        LightColor.GREEN: (40, 220, 40),
        LightColor.AMBER: (0, 190, 255),
        LightColor.RED: (40, 40, 230),
    }[light]
    cv2.circle(vis, (40, 40), 22, sig_bgr, -1)
    cv2.putText(
        vis,
        f"Lane {lane}: {light.value}",
        (75, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(
        page_title="Adaptive Traffic Signal Control - Indian Chowk",
        layout="wide",
    )
    st.title("Adaptive Traffic Signal Control with Emergency Preemption")
    st.caption("4-Way Indian Intersection (Chowk) | 1 FPS Synchronized Feeds + YOLOv8 + 2-Layer LSTM + Explicit State Machine")

    datasets_yaml = ROOT_DIR / "configs" / "datasets.yaml"
    lanes_yaml = ROOT_DIR / "configs" / "lanes.yaml"
    controller_yaml = ROOT_DIR / "configs" / "controller.yaml"

    det_cfg = DetectorConfig.from_yaml(datasets_yaml, base_dir=ROOT_DIR)
    lane_cfg = LaneConfig.from_yaml(lanes_yaml, base_dir=ROOT_DIR)
    sm_cfg, _, _ = load_controller_configs(controller_yaml, base_dir=ROOT_DIR)
    lstm_cfg = load_lstm_config(controller_yaml, base_dir=ROOT_DIR)

    video_dir = ROOT_DIR / "artifacts" / "demo_videos"
    video_paths = {lane: video_dir / f"approach_{lane}.mp4" for lane in LANES}
    if not all(p.exists() for p in video_paths.values()):
        generate_synthetic_four_way_videos(video_dir, duration_seconds=30, fps=5, seed=42)

    tick_slider = st.sidebar.slider("Replay Tick (seconds @ 1 FPS)", min_value=0, max_value=29, value=9, step=1)
    normalize_flag = st.sidebar.checkbox("Normalize counts by ROI area", value=lane_cfg.normalize_by_area)

    active_lane_cfg = LaneConfig(
        polygons=lane_cfg.polygons,
        normalize_by_area=normalize_flag,
        anchor_point=lane_cfg.anchor_point,
        feature_log_csv=lane_cfg.feature_log_csv,
    )

    detector = Detector(det_cfg)
    predictor = OnlineQueuePredictor(lstm_cfg)
    controller = TrafficSignalController(config=sm_cfg)

    feed = VideoFileFeed(VideoFeedConfig(video_paths=video_paths, target_fps=1.0, max_ticks=tick_slider + 1))

    last_frames: Dict[str, np.ndarray] = {}
    last_dets: Dict[str, DetectionOutput] = {}
    last_x_t = np.zeros(8, dtype=np.float32)
    last_dc_dt = np.zeros(4, dtype=np.float32)
    last_decision: ControllerDecision | None = None

    for ts, frames in feed:
        dets = {lane: detector.detect(frames[lane]) for lane in LANES}
        rec = extract_x_t_from_detections(dets, active_lane_cfg, timestamp=ts)
        pred_dc_dt = predictor.push_and_predict(rec.x_t)
        decision = controller.update(timestamp=ts, x_t=rec.x_t, predicted_dc_dt=pred_dc_dt, dt=1.0)

        last_frames = frames
        last_dets = dets
        last_x_t = rec.x_t
        last_dc_dt = pred_dc_dt
        last_decision = decision

    feed.close()

    if last_decision is None:
        st.warning("No frames available.")
        return

    # Emergency Preemption Banner
    if last_decision.is_emergency_preemption or np.any(last_x_t[4:] > 0):
        emg_lanes = [LANES[i] for i in range(4) if last_x_t[i + 4] > 0]
        st.error(
            f"🚨 EMERGENCY PREEMPTION ACTIVE | State: {last_decision.state.value} | "
            f"Active Green/Clearing Approach: {last_decision.active_lane} | "
            f"Emergency Vehicle Detected on: {', '.join(emg_lanes) if emg_lanes else last_decision.active_lane}"
        )
    else:
        st.success(
            f"✅ Normal Clockwise Adaptive Cycle | State: {last_decision.state.value} | "
            f"Active Approach: {last_decision.active_lane} | Remaining Timer: {last_decision.remaining_time:.1f}s"
        )

    # Four synchronized camera feeds with bounding boxes and signal states
    col_n, col_e = st.columns(2)
    col_s, col_w = st.columns(2)
    grid_cols = {"N": col_n, "E": col_e, "S": col_s, "W": col_w}

    for lane in LANES:
        with grid_cols[lane]:
            light = last_decision.lights[lane]
            badge = {"GREEN": "🟢 GREEN", "AMBER": "🟡 AMBER", "RED": "🔴 RED"}[light.value]
            st.subheader(f"Approach {lane} — {badge}")
            annotated = annotate_frame(
                last_frames[lane],
                last_dets[lane],
                active_lane_cfg.polygons[lane],
                lane,
                light,
            )
            st.image(annotated, use_container_width=True)

    # Live X_t and Predicted dC/dt tables
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Live Feature Vector $X_t$")
        xt_df = pd.DataFrame([last_x_t], columns=XT_COLUMNS)
        st.dataframe(xt_df, use_container_width=True)

    with c2:
        st.subheader("LSTM Predicted Queue Growth Rate $dC/dt$ (veh/s)")
        dc_df = pd.DataFrame(
            [last_dc_dt],
            columns=["dC_N/dt", "dC_E/dt", "dC_S/dt", "dC_W/dt"],
        )
        st.dataframe(dc_df, use_container_width=True)

    if controller.transition_logs:
        st.subheader("State Machine Transition Log")
        t_df = pd.DataFrame(
            [
                {
                    "timestamp_s": t.timestamp,
                    "from_state": t.from_state.value,
                    "to_state": t.to_state.value,
                    "active_lane": t.active_lane,
                    "next_lane": t.next_lane,
                    "reason": t.reason,
                }
                for t in controller.transition_logs
            ]
        )
        st.dataframe(t_df, use_container_width=True)


if __name__ == "__main__":
    main()
