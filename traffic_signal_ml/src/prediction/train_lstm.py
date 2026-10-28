"""Train the 2-layer LSTM queue growth predictor on real CSV logs or synthetic traffic data."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.prediction.dataset import QueueGrowthDataset, build_sliding_windows
from src.prediction.lstm_model import LSTMConfig, QueueGrowthLSTM
from src.sim.synthetic_traffic import SyntheticTrafficConfig, generate_synthetic_xt_series

logger = logging.getLogger(__name__)


def load_lstm_config(yaml_path: Path, base_dir: Optional[Path] = None) -> LSTMConfig:
    """Load LSTMConfig from configs/controller.yaml."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    root = base_dir if base_dir is not None else yaml_path.parent.parent
    l_cfg = raw.get("lstm", {})
    return LSTMConfig(
        window_size=int(l_cfg.get("window_size", 10)),
        input_size=int(l_cfg.get("input_size", 8)),
        hidden_size=int(l_cfg.get("hidden_size", 64)),
        num_layers=int(l_cfg.get("num_layers", 2)),
        output_size=int(l_cfg.get("output_size", 4)),
        learning_rate=float(l_cfg.get("learning_rate", 0.001)),
        batch_size=int(l_cfg.get("batch_size", 32)),
        epochs=int(l_cfg.get("epochs", 15)),
        weights_path=(root / str(l_cfg.get("weights_path", "artifacts/lstm_predictor.pt"))).resolve(),
        seed=int(raw.get("seed", 42)),
    )


def train_lstm_model(
    config: LSTMConfig,
    real_log_csv: Optional[Path] = None,
    synthetic_steps: int = 3000,
) -> Tuple[QueueGrowthLSTM, Dict[str, float]]:
    """Train the 2-layer LSTM on real logs if available, otherwise on synthetic_traffic.py traces.

    Per ML best practices for time-series forecasting, performs a strict chronological split
    (70% train, 15% val, 15% test) prior to training and evaluates MSE/MAE on val and test splits.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    used_synthetic = True
    if real_log_csv is not None and real_log_csv.exists():
        ds_check = QueueGrowthDataset.from_csv(real_log_csv, window_size=config.window_size)
        if len(ds_check) >= 100:
            logger.info("Training LSTM on real feature log CSV: %s", real_log_csv)
            import pandas as pd
            from src.features.feature_vector import XT_COLUMNS

            df = pd.read_csv(real_log_csv)
            x_series = df[XT_COLUMNS].to_numpy(dtype=np.float32)
            used_synthetic = False
        else:
            x_series = generate_synthetic_xt_series(
                SyntheticTrafficConfig(duration_seconds=synthetic_steps, seed=config.seed)
            )
    else:
        logger.info(
            "No real logs found; generating %d steps of synthetic traffic data for LSTM training.",
            synthetic_steps,
        )
        x_series = generate_synthetic_xt_series(
            SyntheticTrafficConfig(duration_seconds=synthetic_steps, seed=config.seed)
        )

    # Chronological split: 70% train, 15% val, 15% test
    n_total = len(x_series)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_series = x_series[:n_train]
    val_series = x_series[n_train : n_train + n_val]
    test_series = x_series[n_train + n_val :]

    train_ds = QueueGrowthDataset(train_series, window_size=config.window_size)
    val_ds = QueueGrowthDataset(val_series, window_size=config.window_size)
    test_ds = QueueGrowthDataset(test_series, window_size=config.window_size)

    g = torch.Generator()
    g.manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=g,
    )

    model = QueueGrowthLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)

        train_mse = running_loss / max(1, len(train_ds))

        model.eval()
        with torch.no_grad():
            val_preds = model(val_ds.windows)
            val_mse = float(criterion(val_preds, val_ds.targets).item())

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
            logger.info(
                "LSTM Epoch %02d/%02d - train_mse: %.4f - val_mse: %.4f",
                epoch,
                config.epochs,
                train_mse,
                val_mse,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_preds = model(test_ds.windows)
        test_mse = float(criterion(test_preds, test_ds.targets).item())
        test_mae = float(torch.mean(torch.abs(test_preds - test_ds.targets)).item())

    config.weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.weights_path)

    metrics = {
        "trained_on_synthetic": used_synthetic,
        "window_size": config.window_size,
        "best_val_mse": round(best_val_loss, 5),
        "test_mse": round(test_mse, 5),
        "test_mae": round(test_mae, 5),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }
    metrics_path = config.weights_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved trained LSTM weights to %s (metrics: %s)", config.weights_path, metrics)
    return model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg_file = Path(__file__).resolve().parents[2] / "configs" / "controller.yaml"
    lstm_cfg = load_lstm_config(cfg_file)
    train_lstm_model(lstm_cfg)
