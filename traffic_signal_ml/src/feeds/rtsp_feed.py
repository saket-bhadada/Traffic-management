"""RTSP live camera stream feed stub implementing FeedSource."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np

from src.feeds.base import FeedSource


@dataclass(frozen=True)
class RTSPFeedConfig:
    """Configuration for four RTSP camera streams (N, E, S, W)."""

    rtsp_urls: Dict[str, str]
    target_fps: float = 1.0
    reconnect_timeout_sec: float = 5.0


class RTSPFeed(FeedSource):
    """Stub implementation of FeedSource for live RTSP intersection cameras."""

    def __init__(self, config: RTSPFeedConfig) -> None:
        self.config = config

    def __iter__(self) -> Iterator[Tuple[float, Dict[str, np.ndarray]]]:
        """Stub iterator; raises NotImplementedError until live RTSP cameras are provisioned."""
        raise NotImplementedError(
            "RTSPFeed is a stub for live camera deployment. Use VideoFileFeed or simulation mode."
        )

    def close(self) -> None:
        """No-op resource cleanup for RTSP stub."""
        return None
