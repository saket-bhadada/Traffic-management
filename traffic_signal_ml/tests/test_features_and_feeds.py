"""Pytest suite for X_t shape/ordering and 4-lane shared timestamp lockstep synchronization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.detection.detector import DetectionOutput
from src.features.feature_vector import (
    XT_COLUMNS,
    FeatureCSVLogger,
    build_feature_vector,
    extract_x_t_from_detections,
)
from src.features.lane_density import LaneConfig, LaneCounts, count_vehicles_in_lane
from src.feeds.base import LANES
from src.feeds.video_file_feed import (
    VideoFeedConfig,
    VideoFileFeed,
    generate_synthetic_four_way_videos,
)


def test_x_t_shape_and_ordering(tmp_path: Path) -> None:
    """Verify X_t has shape (8,) and exact ordering [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W]."""
    counts = {
        "N": LaneCounts(standard_count=10.0, emergency_count=1.0),
        "E": LaneCounts(standard_count=20.0, emergency_count=0.0),
        "S": LaneCounts(standard_count=30.0, emergency_count=2.0),
        "W": LaneCounts(standard_count=40.0, emergency_count=0.0),
    }
    x_t = build_feature_vector(counts)
    assert x_t.shape == (8,)
    assert x_t.dtype == np.float32
    expected = np.array([10.0, 20.0, 30.0, 40.0, 1.0, 0.0, 2.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(x_t, expected)

    # Also verify polygon ROI extraction and CSV logging preserve ordering
    poly = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype=np.float32)
    lane_cfg = LaneConfig(
        polygons={lane: poly for lane in LANES},
        normalize_by_area=False,
        anchor_point="bottom_center",
        feature_log_csv=tmp_path / "xt.csv",
    )

    dets = {
        "N": DetectionOutput(
            boxes=np.array([[100, 100, 140, 140], [200, 200, 240, 240]], dtype=np.float32),
            class_ids=np.array([0, 1], dtype=np.int64),
            confidences=np.array([0.9, 0.95], dtype=np.float32),
        ),
        "E": DetectionOutput(
            boxes=np.array([[100, 100, 140, 140]], dtype=np.float32),
            class_ids=np.array([0], dtype=np.int64),
            confidences=np.array([0.85], dtype=np.float32),
        ),
        "S": DetectionOutput(
            boxes=np.zeros((0, 4), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int64),
            confidences=np.zeros((0,), dtype=np.float32),
        ),
        "W": DetectionOutput(
            boxes=np.array([[150, 150, 190, 190]], dtype=np.float32),
            class_ids=np.array([1], dtype=np.int64),
            confidences=np.array([0.99], dtype=np.float32),
        ),
    }

    rec = extract_x_t_from_detections(dets, lane_cfg, timestamp=5.0)
    assert rec.x_t.shape == (8,)
    np.testing.assert_allclose(
        rec.x_t,
        np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    logger = FeatureCSVLogger(lane_cfg.feature_log_csv, overwrite=True)
    logger.log(rec)
    lines = lane_cfg.feature_log_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0].split(",") == ["timestamp", *XT_COLUMNS]


def test_area_normalization_flag() -> None:
    """Verify normalize_by_area config flag divides counts by polygon area."""
    poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)  # area = 10,000
    det = DetectionOutput(
        boxes=np.array([[20, 20, 40, 40], [50, 50, 70, 70]], dtype=np.float32),
        class_ids=np.array([0, 0], dtype=np.int64),
        confidences=np.array([0.9, 0.9], dtype=np.float32),
    )
    raw_counts = count_vehicles_in_lane(det, poly, normalize_by_area=False)
    norm_counts = count_vehicles_in_lane(det, poly, normalize_by_area=True)
    assert raw_counts.standard_count == pytest.approx(2.0)
    assert norm_counts.standard_count == pytest.approx(2.0 / 10000.0)


def test_four_lanes_share_timestamp_per_tick(tmp_path: Path) -> None:
    """Verify VideoFileFeed plays 4 video files in lockstep at 1 FPS with one shared timestamp per tick."""
    video_paths = generate_synthetic_four_way_videos(
        output_dir=tmp_path / "videos",
        duration_seconds=6,
        fps=5,
        seed=42,
    )
    feed = VideoFileFeed(
        VideoFeedConfig(video_paths=video_paths, target_fps=1.0, max_ticks=5)
    )
    timestamps = []
    try:
        for ts, frames_by_lane in feed:
            assert isinstance(ts, float)
            assert set(frames_by_lane.keys()) == {"N", "E", "S", "W"}
            for lane in LANES:
                assert frames_by_lane[lane].shape == (480, 640, 3)
            timestamps.append(ts)
    finally:
        feed.close()

    assert timestamps == [0.0, 1.0, 2.0, 3.0, 4.0]
