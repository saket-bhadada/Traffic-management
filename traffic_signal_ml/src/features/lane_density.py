"""Polygon ROI filtering and per-lane vehicle / emergency vehicle counting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

from src.detection.detector import DetectionOutput
from src.feeds.base import LANES


@dataclass(frozen=True)
class LaneConfig:
    """Configuration for polygon ROIs per lane and optional area normalization."""

    polygons: Dict[str, np.ndarray]  # lane -> shape (K, 2) float32 polygon vertices
    normalize_by_area: bool = False
    anchor_point: str = "bottom_center"
    feature_log_csv: Path = Path("artifacts/feature_vectors_xt.csv")

    @classmethod
    def from_yaml(cls, yaml_path: Path, base_dir: Path | None = None) -> "LaneConfig":
        """Load LaneConfig from configs/lanes.yaml."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        root = base_dir if base_dir is not None else yaml_path.parent.parent
        lanes_raw = raw.get("lanes", {})
        polygons: Dict[str, np.ndarray] = {}
        for lane in LANES:
            pts = lanes_raw.get(lane, {}).get("polygon", [[0, 0], [640, 0], [640, 480], [0, 480]])
            polygons[lane] = np.asarray(pts, dtype=np.float32)

        return cls(
            polygons=polygons,
            normalize_by_area=bool(raw.get("normalize_by_area", False)),
            anchor_point=str(raw.get("anchor_point", "bottom_center")),
            feature_log_csv=(root / str(raw.get("feature_log_csv", "artifacts/feature_vectors_xt.csv"))).resolve(),
        )


@dataclass(frozen=True)
class LaneCounts:
    """Counts (or area-normalized densities) for a single lane at one tick."""

    standard_count: float
    emergency_count: float


def compute_box_anchor(box: np.ndarray, mode: str = "bottom_center") -> Tuple[float, float]:
    """Compute representative anchor point (x, y) for a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    cx = 0.5 * (x1 + x2)
    if mode == "center":
        cy = 0.5 * (y1 + y2)
    else:
        cy = y2
    return cx, cy


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Return True if point (x, y) lies inside or on the boundary of polygon."""
    cnt = polygon.reshape((-1, 1, 2)).astype(np.float32)
    dist = cv2.pointPolygonTest(cnt, (float(point[0]), float(point[1])), measureDist=False)
    return bool(dist >= 0.0)


def polygon_area(polygon: np.ndarray) -> float:
    """Compute polygon area in pixels^2 using OpenCV contourArea."""
    cnt = polygon.reshape((-1, 1, 2)).astype(np.float32)
    area = float(cv2.contourArea(cnt))
    return max(area, 1.0)


def count_vehicles_in_lane(
    detection: DetectionOutput,
    polygon: np.ndarray,
    normalize_by_area: bool = False,
    anchor_mode: str = "bottom_center",
) -> LaneCounts:
    """Count standard (class 0) and emergency (class 1) vehicles inside a lane's polygon ROI."""
    c_std = 0.0
    c_emg = 0.0

    for i in range(len(detection.boxes)):
        anchor = compute_box_anchor(detection.boxes[i], mode=anchor_mode)
        if point_in_polygon(anchor, polygon):
            cls_id = int(detection.class_ids[i])
            if cls_id == 0:
                c_std += 1.0
            elif cls_id == 1:
                c_emg += 1.0

    if normalize_by_area:
        area = polygon_area(polygon)
        c_std = c_std / area
        c_emg = c_emg / area

    return LaneCounts(standard_count=c_std, emergency_count=c_emg)
