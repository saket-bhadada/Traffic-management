"""FastAPI Backend Server for Adaptive Traffic Signal Control with Emergency Preemption.

Provides REST and WebSocket endpoints for real-time traffic signal telemetry,
video frame streaming with detections, and controller state management.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.control.controller import ControllerDecision, TrafficSignalController
from src.control.state_machine import LightColor, SignalState
from src.detection.detector import DetectionOutput, Detector, DetectorConfig
from src.evaluation.compare import load_controller_configs
from src.features.feature_vector import XT_COLUMNS, extract_x_t_from_detections
from src.features.lane_density import LaneConfig
from src.feeds.base import LANES
from src.feeds.video_file_feed import (
    VideoFeedConfig,
    VideoFileFeed,
    generate_synthetic_four_way_videos,
)
from src.prediction.lstm_model import OnlineQueuePredictor
from src.prediction.train_lstm import load_lstm_config

logger = logging.getLogger("traffic_fastapi")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="Adaptive Traffic Signal Control API",
    description="FastAPI Backend for 4-Way Chowk Traffic Management with Emergency Preemption",
    version="1.0.0",
)

# Enable CORS for React frontend (Vite default: http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- App State & Pipeline Singletons -----------------

class TrafficEngine:
    """Manages the 1 FPS locked feed, detector, LSTM predictor, and state machine controller."""

    def __init__(self) -> None:
        self.root_dir = ROOT_DIR
        datasets_yaml = self.root_dir / "configs" / "datasets.yaml"
        lanes_yaml = self.root_dir / "configs" / "lanes.yaml"
        controller_yaml = self.root_dir / "configs" / "controller.yaml"

        self.det_cfg = DetectorConfig.from_yaml(datasets_yaml, base_dir=self.root_dir)
        self.lane_cfg = LaneConfig.from_yaml(lanes_yaml, base_dir=self.root_dir)
        self.sm_cfg, _, self.sim_settings = load_controller_configs(
            controller_yaml, base_dir=self.root_dir
        )
        self.lstm_cfg = load_lstm_config(controller_yaml, base_dir=self.root_dir)

        self.video_dir = self.root_dir / "artifacts" / "demo_videos"
        self.video_paths = {lane: self.video_dir / f"approach_{lane}.mp4" for lane in LANES}
        if not all(p.exists() for p in self.video_paths.values()):
            generate_synthetic_four_way_videos(self.video_dir, duration_seconds=30, fps=5, seed=42)

        self.detector = Detector(self.det_cfg)
        self.predictor = OnlineQueuePredictor(self.lstm_cfg)
        self.controller = TrafficSignalController(config=self.sm_cfg)
        self.feed = VideoFileFeed(VideoFeedConfig(video_paths=self.video_paths, target_fps=1.0))
        self.feed_iter = iter(self.feed)

        self.last_timestamp: float = 0.0
        self.last_x_t: List[float] = [0.0] * 8
        self.last_dc_dt: List[float] = [0.0] * 4
        self.last_decision: Optional[ControllerDecision] = None
        self.last_images_b64: Dict[str, str] = {}
        self.last_detections_summary: Dict[str, Dict[str, Any]] = {}
        self.is_running: bool = True
        self.step_delay: float = 1.0  # 1 FPS default

    def reset(self) -> None:
        """Reset feed and controller back to tick 0."""
        self.feed.close()
        self.feed = VideoFileFeed(VideoFeedConfig(video_paths=self.video_paths, target_fps=1.0))
        self.feed_iter = iter(self.feed)
        self.controller = TrafficSignalController(config=self.sm_cfg)
        self.predictor = OnlineQueuePredictor(self.lstm_cfg)

    def step(self) -> Dict[str, Any]:
        """Perform one 1-second step across the 4 cameras, detect, predict, and control."""
        try:
            timestamp, frames_by_lane = next(self.feed_iter)
        except StopIteration:
            self.reset()
            timestamp, frames_by_lane = next(self.feed_iter)

        self.last_timestamp = timestamp
        dets_by_lane: Dict[str, DetectionOutput] = {}
        images_b64: Dict[str, str] = {}
        dets_summary: Dict[str, Dict[str, Any]] = {}

        for lane in LANES:
            frame = frames_by_lane[lane]
            det = self.detector.detect(frame)
            dets_by_lane[lane] = det

            # Annotate frame
            annotated = self._draw_annotations(frame, det, lane)
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            images_b64[lane] = base64.b64encode(buffer).decode("utf-8")

            dets_summary[lane] = {
                "boxes": det.boxes.tolist(),
                "class_ids": det.class_ids.tolist(),
                "confidences": [round(float(c), 2) for c in det.confidences],
            }

        rec = extract_x_t_from_detections(dets_by_lane, self.lane_cfg, timestamp=timestamp)
        self.last_x_t = [round(float(v), 2) for v in rec.x_t]

        pred_dc_dt = self.predictor.push_and_predict(rec.x_t)
        self.last_dc_dt = [round(float(v), 2) for v in pred_dc_dt]

        decision = self.controller.update(
            timestamp=timestamp,
            x_t=rec.x_t,
            predicted_dc_dt=pred_dc_dt,
            dt=1.0,
        )
        self.last_decision = decision
        self.last_images_b64 = images_b64
        self.last_detections_summary = dets_summary

        return self.get_state_payload()

    def _draw_annotations(self, frame: np.ndarray, det: DetectionOutput, lane: str) -> np.ndarray:
        vis = frame.copy()
        poly = self.lane_cfg.polygons[lane].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(vis, [poly], isClosed=True, color=(255, 255, 0), thickness=2)

        for i in range(len(det.boxes)):
            x1, y1, x2, y2 = [int(v) for v in det.boxes[i]]
            cls_id = int(det.class_ids[i])
            conf = float(det.confidences[i])
            color = (0, 0, 255) if cls_id == 1 else (0, 215, 255)
            label = f"{'EMERGENCY' if cls_id == 1 else 'VEHICLE'} {conf:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        return vis

    def get_state_payload(self) -> Dict[str, Any]:
        lights_map = {}
        if self.last_decision is not None:
            lights_map = {k: v.value for k, v in self.last_decision.lights.items()}

        recent_transitions = [
            {
                "timestamp": round(t.timestamp, 1),
                "from_state": t.from_state.value,
                "to_state": t.to_state.value,
                "active_lane": t.active_lane,
                "next_lane": t.next_lane,
                "reason": t.reason,
            }
            for t in self.controller.transition_logs[-10:]
        ]

        has_emergency = any(self.last_x_t[i + 4] > 0 for i in range(4))
        is_preempting = (
            self.last_decision.is_emergency_preemption if self.last_decision else False
        ) or has_emergency

        return {
            "timestamp": round(self.last_timestamp, 1),
            "state": self.last_decision.state.value if self.last_decision else "IDLE",
            "active_lane": self.last_decision.active_lane if self.last_decision else "N",
            "remaining_time": round(self.last_decision.remaining_time, 1) if self.last_decision else 0.0,
            "lights": lights_map,
            "is_emergency_preemption": is_preempting,
            "x_t": dict(zip(XT_COLUMNS, self.last_x_t)),
            "dc_dt": dict(zip(["dC_N_dt", "dC_E_dt", "dC_S_dt", "dC_W_dt"], self.last_dc_dt)),
            "images": self.last_images_b64,
            "detections": self.last_detections_summary,
            "recent_transitions": recent_transitions,
        }


engine = TrafficEngine()

# ----------------- REST Endpoints -----------------

@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "Traffic Signal FastAPI Backend"}

@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    """Get latest intersection state."""
    return engine.get_state_payload()

@app.post("/api/step")
def step_simulation() -> Dict[str, Any]:
    """Advance simulation by 1 second."""
    return engine.step()

@app.post("/api/reset")
def reset_simulation() -> Dict[str, str]:
    """Reset feeds and controller."""
    engine.reset()
    return {"status": "reset_successful"}

# ----------------- WebSocket for Real-Time Streaming -----------------

@app.websocket("/ws/traffic")
async def websocket_traffic(websocket: WebSocket) -> None:
    """Streams live 1 FPS ticks with image frames and telemetry to React."""
    await websocket.accept()
    logger.info("React WebSocket client connected.")
    try:
        while True:
            payload = engine.step()
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(engine.step_delay)
    except WebSocketDisconnect:
        logger.info("React WebSocket client disconnected.")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
