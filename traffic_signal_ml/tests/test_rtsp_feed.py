from __future__ import annotations

import numpy as np
import pytest

from src.feeds.base import LANES
from src.feeds.rtsp_feed import RTSPFeed, RTSPFeedConfig


class FakeCapture:
    def __init__(self, url: str, _backend: int, _params: list[int]) -> None:
        self.url = url
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, _property: int, _value: int) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.full((2, 3, 3), len(self.url), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_rtsp_feed_yields_four_lanes_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = {lane: f"rtsp://camera.example/{lane}" for lane in LANES}
    captures: list[FakeCapture] = []

    def open_capture(url: str, backend: int, params: list[int]) -> FakeCapture:
        capture = FakeCapture(url, backend, params)
        captures.append(capture)
        return capture

    monkeypatch.setattr("src.feeds.rtsp_feed.cv2.VideoCapture", open_capture)
    feed = RTSPFeed(
        RTSPFeedConfig(rtsp_urls=urls, target_fps=100_000, reconnect_timeout_sec=1.0)
    )
    iterator = iter(feed)
    timestamp, frames = next(iterator)

    assert timestamp >= 0
    assert tuple(frames) == LANES
    assert all(frame.shape == (2, 3, 3) for frame in frames.values())

    feed.close()
    assert all(capture.released for capture in captures)


def test_rtsp_feed_rejects_unsupported_url_schemes() -> None:
    urls = {lane: f"rtsp://camera.example/{lane}" for lane in LANES}
    urls["N"] = "file:///camera.mp4"

    with pytest.raises(ValueError, match="direct rtsp://"):
        RTSPFeed(RTSPFeedConfig(rtsp_urls=urls))


def test_rtsp_feed_requires_all_lanes() -> None:
    with pytest.raises(ValueError, match="missing URLs for lanes"):
        RTSPFeed(RTSPFeedConfig(rtsp_urls={"N": "rtsp://camera.example/n"}))
