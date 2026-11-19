"""Synchronized live RTSP/HTTP video feeds for a four-approach intersection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple
from urllib.parse import urlsplit

import cv2
import numpy as np

from src.feeds.base import LANES, FeedSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RTSPFeedConfig:
    """Direct stream URLs for the N, E, S, and W approaches."""

    rtsp_urls: Dict[str, str]
    target_fps: float = 1.0
    reconnect_timeout_sec: float = 5.0


class RTSPFeed(FeedSource):
    """Sample four live network video streams in lockstep."""

    def __init__(self, config: RTSPFeedConfig) -> None:
        if config.target_fps <= 0:
            raise ValueError("target_fps must be greater than zero.")
        if config.reconnect_timeout_sec <= 0:
            raise ValueError("reconnect_timeout_sec must be greater than zero.")

        missing = [lane for lane in LANES if lane not in config.rtsp_urls]
        if missing:
            raise ValueError(f"RTSPFeedConfig is missing URLs for lanes: {missing}")

        self.config = config
        self._closed = False
        self._captures: Dict[str, cv2.VideoCapture] = {}
        self._urls: Dict[str, str] = {}

        for lane in LANES:
            url = config.rtsp_urls[lane].strip()
            self._validate_url(lane, url)
            self._urls[lane] = url

        try:
            for lane in LANES:
                self._captures[lane] = self._open_lane(lane)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_url(lane: str, url: str) -> None:
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme.lower() in {"rtsp", "rtsps", "http", "https"} and bool(parsed.hostname)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                f"Lane {lane} needs a direct rtsp://, rtsps://, http://, or https:// "
                "video-stream URL. A CCTV web page URL is not a stream URL."
            )

    def _open_lane(self, lane: str) -> cv2.VideoCapture:
        timeout_ms = max(1, int(self.config.reconnect_timeout_sec * 1000))
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            timeout_ms,
        ]
        try:
            capture = cv2.VideoCapture(self._urls[lane], cv2.CAP_FFMPEG, params)
        except cv2.error as exc:
            raise RuntimeError(
                f"OpenCV could not initialize the video stream configured for lane {lane}."
            ) from exc
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open the video stream configured for lane {lane}. "
                "Check that it is a direct, reachable stream URL supported by this "
                "OpenCV FFmpeg build."
            )

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _read_lane(self, lane: str) -> np.ndarray:
        capture = self._captures[lane]
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame

        deadline = time.monotonic() + self.config.reconnect_timeout_sec
        last_error = "stream returned no frame"
        capture.release()

        while not self._closed and time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            try:
                capture = self._open_lane(lane)
                self._captures[lane] = capture
            except RuntimeError as exc:
                last_error = str(exc)
                continue

            ok, frame = capture.read()
            if ok and frame is not None:
                logger.info("Reconnected to the configured video stream for lane %s.", lane)
                return frame
            capture.release()

        raise RuntimeError(
            f"Unable to read lane {lane} after {self.config.reconnect_timeout_sec:g} "
            f"seconds of reconnect attempts: {last_error}"
        )

    def __iter__(self) -> Iterator[Tuple[float, Dict[str, np.ndarray]]]:
        """Yield one synchronized set of frames at target_fps until closed."""
        interval = 1.0 / self.config.target_fps
        start = time.monotonic()
        next_tick = start

        while not self._closed:
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            frames = {lane: self._read_lane(lane) for lane in LANES}
            yield round(time.monotonic() - start, 6), frames

            next_tick += interval
            now = time.monotonic()
            if next_tick < now:
                next_tick = now

    def close(self) -> None:
        """Release all network video handles."""
        self._closed = True
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
