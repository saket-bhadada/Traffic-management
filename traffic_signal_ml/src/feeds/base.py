"""Abstract FeedSource interface yielding synchronized 4-approach frames at 1 FPS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Tuple

import numpy as np

LANES: Tuple[str, str, str, str] = ("N", "E", "S", "W")


class FeedSource(ABC):
    """Interface for synchronized camera feeds across the 4 approaches (N, E, S, W)."""

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[float, Dict[str, np.ndarray]]]:
        """Yield (shared_timestamp_sec, {'N': frame_N, 'E': frame_E, 'S': frame_S, 'W': frame_W})."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying video/network resources."""
