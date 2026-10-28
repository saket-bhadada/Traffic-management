"""Lockstep 4-way VideoFileFeed sampled at 1 FPS with a single shared timestamp per tick."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from src.feeds.base import LANES, FeedSource


@dataclass(frozen=True)
class VideoFeedConfig:
    """Configuration for synchronized four-way video file playback."""

    video_paths: Dict[str, Path]
    target_fps: float = 1.0
    max_ticks: Optional[int] = None


class VideoFileFeed(FeedSource):
    """Plays four video files (N, E, S, W) in lockstep at 1 FPS with one shared timestamp per tick."""

    def __init__(self, config: VideoFeedConfig) -> None:
        missing = [lane for lane in LANES if lane not in config.video_paths]
        if missing:
            raise ValueError(f"VideoFeedConfig is missing video paths for lanes: {missing}")

        self.config = config
        self._caps: Dict[str, cv2.VideoCapture] = {}
        self._source_fps: Dict[str, float] = {}

        for lane in LANES:
            path = config.video_paths[lane]
            if not path.exists():
                self.close()
                raise FileNotFoundError(f"Video file for lane {lane} not found at: {path}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                self.close()
                raise RuntimeError(f"Failed to open video file for lane {lane}: {path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            self._source_fps[lane] = fps if fps > 0 else 1.0
            self._caps[lane] = cap

    def __iter__(self) -> Iterator[Tuple[float, Dict[str, np.ndarray]]]:
        """Yield (timestamp, frames_by_lane) in lockstep at 1 FPS."""
        tick_idx = 0
        dt = 1.0 / self.config.target_fps

        while True:
            if self.config.max_ticks is not None and tick_idx >= self.config.max_ticks:
                break

            shared_timestamp = round(tick_idx * dt, 6)
            frames: Dict[str, np.ndarray] = {}
            all_ok = True

            for lane in LANES:
                cap = self._caps[lane]
                src_fps = self._source_fps[lane]
                target_frame_idx = int(round(shared_timestamp * src_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    all_ok = False
                    break
                frames[lane] = frame

            if not all_ok or len(frames) != len(LANES):
                break

            yield shared_timestamp, frames
            tick_idx += 1

    def close(self) -> None:
        """Release all four OpenCV VideoCapture handles."""
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()


def generate_synthetic_four_way_videos(
    output_dir: Path,
    duration_seconds: int = 30,
    fps: int = 5,
    frame_size: Tuple[int, int] = (640, 480),
    seed: int = 42,
) -> Dict[str, Path]:
    """Generate 4 synchronized synthetic approach videos (N, E, S, W) for testing and demo mode.

    Standard vehicles are drawn as cyan-blue rectangles `(220, 180, 40)` inside the lane ROI,
    and emergency vehicles are drawn as bright red rectangles `(30, 30, 240)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    paths: Dict[str, Path] = {}
    writers: Dict[str, cv2.VideoWriter] = {}
    for lane in LANES:
        p = output_dir / f"approach_{lane}.mp4"
        paths[lane] = p
        writers[lane] = cv2.VideoWriter(str(p), fourcc, float(fps), (width, height))

    roi_pts = np.array([[120, 80], [520, 80], [600, 460], [40, 460]], dtype=np.int32)

    for sec in range(duration_seconds):
        for sub_f in range(fps):
            for lane_idx, lane in enumerate(LANES):
                canvas = np.full((height, width, 3), 45, dtype=np.uint8)
                # Draw road polygon
                cv2.fillPoly(canvas, [roi_pts], (70, 70, 70))
                cv2.polylines(canvas, [roi_pts], isClosed=True, color=(180, 180, 180), thickness=2)

                # Deterministic per-second count per lane
                local_rng = np.random.default_rng(seed + sec * 100 + lane_idx)
                num_std = int(local_rng.integers(1, 7))
                has_emg = (sec in (8, 9, 10, 11) and lane == "S") or (
                    sec in (20, 21, 22) and lane == "W"
                )

                # Place standard vehicles inside ROI grid
                for v_i in range(num_std):
                    col = v_i % 3
                    row = v_i // 3
                    cx = 200 + col * 110
                    cy = 160 + row * 95
                    cv2.rectangle(
                        canvas,
                        (cx - 20, cy - 25),
                        (cx + 20, cy + 25),
                        (220, 180, 40),
                        thickness=-1,
                    )

                if has_emg:
                    ex, ey = 320, 360
                    cv2.rectangle(
                        canvas,
                        (ex - 24, ey - 30),
                        (ex + 24, ey + 30),
                        (30, 30, 240),
                        thickness=-1,
                    )

                writers[lane].write(canvas)

    for w in writers.values():
        w.release()

    return paths
