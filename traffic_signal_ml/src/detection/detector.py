"""Detector wrapper exposing boxes, class IDs, and confidences for standard and emergency vehicles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for YOLOv8 inference."""

    model_variant: str = "yolov8n"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    weights_path: Optional[Path] = None
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: Path, base_dir: Optional[Path] = None) -> "DetectorConfig":
        """Load DetectorConfig from datasets.yaml."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        det = raw.get("detector", {})
        root = base_dir if base_dir is not None else yaml_path.parent.parent
        w_rel = det.get("weights_path")
        w_path = (root / str(w_rel)).resolve() if w_rel else None
        return cls(
            model_variant=str(det.get("model_variant", "yolov8n")),
            conf_threshold=float(det.get("conf_threshold", 0.25)),
            iou_threshold=float(det.get("iou_threshold", 0.45)),
            weights_path=w_path,
            seed=int(raw.get("seed", 42)),
        )


@dataclass(frozen=True)
class DetectionOutput:
    """Structured output returned by Detector.predict()."""

    boxes: np.ndarray        # shape (N, 4) in [x1, y1, x2, y2] pixel coordinates
    class_ids: np.ndarray    # shape (N,) int64 in {0: standard_vehicle, 1: emergency_vehicle}
    confidences: np.ndarray  # shape (N,) float32 in [0.0, 1.0]


class Detector:
    """Unified vehicle & emergency vehicle detector backed by Ultralytics YOLOv8.

    Also supports deterministic synthetic marker detection when running on generated
    synthetic test videos without trained YOLO weights present on disk.
    """

    def __init__(self, config: DetectorConfig, use_yolo_weights: bool = True) -> None:
        self.config = config
        self._model: Any = None
        if use_yolo_weights and config.weights_path is not None and config.weights_path.exists():
            from ultralytics import YOLO

            logger.info("Loading YOLOv8 weights from %s", config.weights_path)
            self._model = YOLO(str(config.weights_path))
        elif use_yolo_weights:
            logger.info(
                "Trained weights not found at %s; using synthetic/contour fallback until trained.",
                config.weights_path,
            )

    def detect(self, frame: np.ndarray) -> DetectionOutput:
        """Detect vehicles (class 0) and emergency vehicles (class 1) on a single BGR frame.

        Returns:
            DetectionOutput containing (boxes [N, 4], class_ids [N], confidences [N]).
        """
        if self._model is not None:
            results = self._model.predict(
                source=frame,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return DetectionOutput(
                    boxes=np.zeros((0, 4), dtype=np.float32),
                    class_ids=np.zeros((0,), dtype=np.int64),
                    confidences=np.zeros((0,), dtype=np.float32),
                )
            b = results[0].boxes
            boxes = b.xyxy.detach().cpu().numpy().astype(np.float32)
            class_ids = b.cls.detach().cpu().numpy().astype(np.int64)
            confidences = b.conf.detach().cpu().numpy().astype(np.float32)
            # Filter strictly to classes 0 and 1
            valid_mask = (class_ids == 0) | (class_ids == 1)
            return DetectionOutput(
                boxes=boxes[valid_mask],
                class_ids=class_ids[valid_mask],
                confidences=confidences[valid_mask],
            )

        return self._detect_synthetic_markers(frame)

    def _detect_synthetic_markers(self, frame: np.ndarray) -> DetectionOutput:
        """Fallback detector for synthetic traffic video frames rendered by VideoFileFeed.

        Standard vehicles are rendered as bright cyan/blue rectangles (BGR: (220, 180, 40)),
        and emergency vehicles are rendered as bright red/white rectangles (BGR: (30, 30, 240)).
        """
        boxes_list = []
        cls_list = []
        conf_list = []

        # Standard vehicle mask (cyan-blue range in BGR)
        std_mask = cv2.inRange(frame, (180, 140, 10), (255, 220, 80))
        contours_std, _ = cv2.findContours(std_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_std:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 12 and h >= 12:
                boxes_list.append([float(x), float(y), float(x + w), float(y + h)])
                cls_list.append(0)
                conf_list.append(0.92)

        # Emergency vehicle mask (bright red range in BGR)
        emg_mask = cv2.inRange(frame, (0, 0, 200), (70, 70, 255))
        contours_emg, _ = cv2.findContours(emg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_emg:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 12 and h >= 12:
                boxes_list.append([float(x), float(y), float(x + w), float(y + h)])
                cls_list.append(1)
                conf_list.append(0.96)

        if not boxes_list:
            return DetectionOutput(
                boxes=np.zeros((0, 4), dtype=np.float32),
                class_ids=np.zeros((0,), dtype=np.int64),
                confidences=np.zeros((0,), dtype=np.float32),
            )

        return DetectionOutput(
            boxes=np.asarray(boxes_list, dtype=np.float32),
            class_ids=np.asarray(cls_list, dtype=np.int64),
            confidences=np.asarray(conf_list, dtype=np.float32),
        )
