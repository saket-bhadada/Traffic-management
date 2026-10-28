"""Sliding window dataset for predicting per-lane queue growth rate dC/dt from X_t history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.feature_vector import XT_COLUMNS


@dataclass(frozen=True)
class WindowDatasetConfig:
    """Configuration for sliding-window queue growth dataset construction."""

    window_size: int = 10
    horizon_seconds: int = 1


def compute_queue_growth_targets(x_series: np.ndarray) -> np.ndarray:
    """Compute per-lane queue growth rate dC/dt = C_{t+1} - C_t from an (N, 8) array of X_t.

    For step t, returns target y_t = C_{t+1} - C_t (or C_t - C_{t-1} as causal finite difference).
    Here we compute forward 1-second finite difference dC/dt at the end of each window [t-W+1 .. t]:
    target at window ending at t is (C_{t+1} - C_t) / 1.0.
    """
    if x_series.ndim != 2 or x_series.shape[1] != 8:
        raise ValueError(f"Expected x_series shape (T, 8), got {x_series.shape}")
    c_only = x_series[:, :4].astype(np.float32)
    diffs = np.zeros_like(c_only, dtype=np.float32)
    if len(c_only) > 1:
        diffs[:-1] = c_only[1:] - c_only[:-1]
        diffs[-1] = diffs[-2]
    return diffs


def build_sliding_windows(
    x_series: np.ndarray,
    window_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (windows, targets) arrays from an (N, 8) trajectory of X_t.

    Args:
        x_series: Array of shape (N, 8).
        window_size: Number of historical seconds W (default 10).

    Returns:
        windows: shape (M, W, 8) float32
        targets: shape (M, 4) float32 representing dC/dt per lane
    """
    if x_series.ndim != 2 or x_series.shape[1] != 8:
        raise ValueError(f"Expected (N, 8) X_t series, got {x_series.shape}")
    n = len(x_series)
    if n <= window_size:
        return (
            np.zeros((0, window_size, 8), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )

    dc_dt_all = compute_queue_growth_targets(x_series)
    windows = []
    targets = []
    for end_idx in range(window_size, n):
        win = x_series[end_idx - window_size : end_idx]
        target = dc_dt_all[end_idx - 1]
        windows.append(win)
        targets.append(target)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


class QueueGrowthDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset yielding (X_window [W, 8], dC_dt [4])."""

    def __init__(self, x_series: np.ndarray, window_size: int = 10) -> None:
        windows, targets = build_sliding_windows(x_series, window_size=window_size)
        self.windows = torch.from_numpy(windows).float()
        self.targets = torch.from_numpy(targets).float()

    @classmethod
    def from_csv(cls, csv_path: Path, window_size: int = 10) -> "QueueGrowthDataset":
        """Load logged X_t CSV and build sliding window dataset."""
        df = pd.read_csv(csv_path)
        x_arr = df[XT_COLUMNS].to_numpy(dtype=np.float32)
        return cls(x_series=x_arr, window_size=window_size)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets[idx]
