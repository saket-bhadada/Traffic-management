"""Train YOLOv8n (with config switch for YOLOv8s) and evaluate per-class Precision, Recall, and mAP50 on test split."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YoloTrainConfig:
    """Training configuration for YOLOv8 detector."""

    model_variant: str = "yolov8n"  # "yolov8n" or "yolov8s"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    seed: int = 42
    data_yaml_path: Path = Path("data/splits/data.yaml")
    weights_output_path: Path = Path("artifacts/detector_best.pt")
    metrics_output_path: Path = Path("artifacts/detector_test_metrics.json")

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path,
        model_override: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ) -> "YoloTrainConfig":
        """Load YoloTrainConfig from configs/datasets.yaml."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        root = base_dir if base_dir is not None else yaml_path.parent.parent
        det = raw.get("detector", {})
        variant = model_override or str(det.get("model_variant", "yolov8n"))
        if variant not in ("yolov8n", "yolov8s"):
            raise ValueError(f"Unsupported model_variant '{variant}'. Expected 'yolov8n' or 'yolov8s'.")

        splits_dir = root / str(raw.get("splits_dir", "data/splits"))
        return cls(
            model_variant=variant,
            epochs=int(det.get("epochs", 50)),
            imgsz=int(det.get("imgsz", 640)),
            batch=int(det.get("batch", 16)),
            seed=int(raw.get("seed", 42)),
            data_yaml_path=(splits_dir / "data.yaml").resolve(),
            weights_output_path=(root / str(det.get("weights_path", "artifacts/detector_best.pt"))).resolve(),
            metrics_output_path=(root / str(det.get("metrics_output_path", "artifacts/detector_test_metrics.json"))).resolve(),
        )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate_yolo(config: YoloTrainConfig) -> Dict[str, Any]:
    """Train YOLOv8 (n or s) on unified splits and report per-class test Precision, Recall, and mAP50."""
    seed_everything(config.seed)
    if not config.data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset split YAML not found at {config.data_yaml_path}. "
            "Run download.py -> unify_labels.py -> build_splits.py first."
        )

    from ultralytics import YOLO

    pretrained_name = f"{config.model_variant}.pt"
    logger.info("Initializing %s for training (%d epochs)...", pretrained_name, config.epochs)
    model = YOLO(pretrained_name)

    train_results = model.train(
        data=str(config.data_yaml_path),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        seed=config.seed,
        deterministic=True,
        project=str(config.weights_output_path.parent / "yolo_runs"),
        name=f"train_{config.model_variant}",
        exist_ok=True,
    )

    best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
    config.weights_output_path.parent.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy2(best_pt, config.weights_output_path)
        model = YOLO(str(config.weights_output_path))

    logger.info("Evaluating trained model on test split...")
    val_results = model.val(
        data=str(config.data_yaml_path),
        split="test",
        imgsz=config.imgsz,
        batch=config.batch,
    )

    class_names = {0: "standard_vehicle", 1: "emergency_vehicle"}
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for cls_idx, cls_name in class_names.items():
        p_i, r_i, ap50_i, ap_i = val_results.box.class_result(cls_idx)
        per_class_metrics[cls_name] = {
            "precision": float(p_i),
            "recall": float(r_i),
            "mAP50": float(ap50_i),
            "mAP50_95": float(ap_i),
        }

    report: Dict[str, Any] = {
        "model_variant": config.model_variant,
        "overall": {
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
        },
        "per_class": per_class_metrics,
    }

    config.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved detector test split evaluation metrics to %s", config.metrics_output_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector (yolov8n or yolov8s).")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["yolov8n", "yolov8s"],
        help="Override model variant from configs/datasets.yaml",
    )
    args = parser.parse_args()
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"
    train_cfg = YoloTrainConfig.from_yaml(cfg_path, model_override=args.model)
    train_and_evaluate_yolo(train_cfg)
