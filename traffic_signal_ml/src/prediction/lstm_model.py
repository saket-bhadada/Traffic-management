"""2-layer LSTM with a linear head to predict per-lane queue growth rate dC/dt."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class LSTMConfig:
    """Hyperparameters and architecture settings for the queue growth LSTM."""

    window_size: int = 10
    input_size: int = 8
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 4
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 15
    weights_path: Path = Path("artifacts/lstm_predictor.pt")
    seed: int = 42


class QueueGrowthLSTM(nn.Module):
    """2-layer LSTM followed by a linear head predicting dC/dt for (N, E, S, W)."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, window_size, 8).

        Returns:
            Tensor of shape (batch_size, 4) representing predicted dC/dt per lane.
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class OnlineQueuePredictor:
    """Maintains a sliding window of the last W seconds of X_t and predicts dC/dt per lane."""

    def __init__(self, config: LSTMConfig, model: Optional[QueueGrowthLSTM] = None) -> None:
        self.config = config
        self._buffer: Deque[np.ndarray] = deque(maxlen=config.window_size)
        if model is not None:
            self.model = model
        else:
            self.model = QueueGrowthLSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                output_size=config.output_size,
            )
            if config.weights_path.exists():
                state_dict = torch.load(config.weights_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict)
        self.model.eval()

    def push_and_predict(self, x_t: Sequence[float] | np.ndarray) -> np.ndarray:
        """Append current X_t (shape 8) to the sliding window and return predicted dC/dt (shape 4)."""
        arr = np.asarray(x_t, dtype=np.float32).reshape(8)
        self._buffer.append(arr)

        # Left-pad with the earliest observation if buffer has fewer than W steps
        if len(self._buffer) < self.config.window_size:
            pad_count = self.config.window_size - len(self._buffer)
            padded = [self._buffer[0]] * pad_count + list(self._buffer)
        else:
            padded = list(self._buffer)

        window_np = np.stack(padded, axis=0)  # (W, 8)
        tensor_in = torch.from_numpy(window_np).unsqueeze(0)  # (1, W, 8)
        with torch.no_grad():
            pred = self.model(tensor_in).squeeze(0).cpu().numpy().astype(np.float32)
        return pred
