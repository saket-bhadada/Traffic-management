"""Construct X_t = [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W] and log every tick to CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.detection.detector import DetectionOutput
from src.features.lane_density import LaneConfig, LaneCounts, count_vehicles_in_lane
from src.feeds.base import LANES

XT_COLUMNS: List[str] = ["C_N", "C_E", "C_S", "C_W", "E_N", "E_E", "E_S", "E_W"]


@dataclass(frozen=True)
class FeatureVectorRecord:
    """Single-tick feature vector X_t with timestamp."""

    timestamp: float
    x_t: np.ndarray  # shape (8,) float32: [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W]

    @property
    def standard_counts(self) -> Dict[str, float]:
        """Return {lane: C_lane} for N, E, S, W."""
        return {lane: float(self.x_t[i]) for i, lane in enumerate(LANES)}

    @property
    def emergency_counts(self) -> Dict[str, float]:
        """Return {lane: E_lane} for N, E, S, W."""
        return {lane: float(self.x_t[i + 4]) for i, lane in enumerate(LANES)}


def build_feature_vector(
    lane_counts: Dict[str, LaneCounts],
) -> np.ndarray:
    """Build the 8-element vector X_t = [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W]."""
    c_vals = [float(lane_counts[lane].standard_count) for lane in LANES]
    e_vals = [float(lane_counts[lane].emergency_count) for lane in LANES]
    return np.asarray(c_vals + e_vals, dtype=np.float32)


def extract_x_t_from_detections(
    detections_by_lane: Dict[str, DetectionOutput],
    lane_config: LaneConfig,
    timestamp: float,
) -> FeatureVectorRecord:
    """Compute per-lane ROI counts from 4-way detections and assemble FeatureVectorRecord."""
    counts_map: Dict[str, LaneCounts] = {}
    for lane in LANES:
        det = detections_by_lane[lane]
        poly = lane_config.polygons[lane]
        counts_map[lane] = count_vehicles_in_lane(
            detection=det,
            polygon=poly,
            normalize_by_area=lane_config.normalize_by_area,
            anchor_mode=lane_config.anchor_point,
        )
    vec = build_feature_vector(counts_map)
    return FeatureVectorRecord(timestamp=float(timestamp), x_t=vec)


class FeatureCSVLogger:
    """Logs every tick's timestamp and X_t = [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W] to CSV."""

    def __init__(self, csv_path: Path, overwrite: bool = True) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", *XT_COLUMNS])

    def log(self, record: FeatureVectorRecord) -> None:
        """Append a single FeatureVectorRecord row to the CSV."""
        if record.x_t.shape != (8,):
            raise ValueError(f"Expected X_t shape (8,), got {record.x_t.shape}")
        row = [f"{record.timestamp:.3f}"] + [f"{float(v):.4f}" for v in record.x_t]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
