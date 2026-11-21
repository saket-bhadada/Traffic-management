"""Remap labels of every downloaded dataset to the unified 2-class scheme.

    0: standard_vehicle
    1: emergency_vehicle

Mapping comes from configs/class_map.yaml. Labels that are not in the map are
dropped (and reported). Images left with no boxes are skipped.

Usage (from the traffic_signal_ml folder):
    python src/data/unify_labels.py
"""

from __future__ import annotations

import logging
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = ("train", "valid", "val", "test")


def _canon_spaced(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", name.strip().lower())


def _canon_joined(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def build_lookup(label_to_class: Dict[str, int]) -> Dict[str, int]:
    """Lookup that ignores case, hyphens, underscores and spaces."""
    lookup: Dict[str, int] = {}
    for key, cls in label_to_class.items():
        lookup[_canon_spaced(str(key))] = int(cls)
        lookup[_canon_joined(str(key))] = int(cls)
    return lookup


def map_name(name: str, lookup: Dict[str, int]) -> Optional[int]:
    for variant in (_canon_spaced(name), _canon_joined(name)):
        if variant in lookup:
            return lookup[variant]
    return None


def read_class_names(data_yaml: Path) -> List[str]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    names = raw.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """Return (class_id, xc, yc, w, h). Handles box lines and polygon lines."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if len(vals) == 4:
        xc, yc, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:  # segmentation polygon -> bounding box
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    else:
        return None

    # clip to image and drop degenerate boxes
    x1, x2 = max(0.0, xc - w / 2), min(1.0, xc + w / 2)
    y1, y2 = max(0.0, yc - h / 2), min(1.0, yc + h / 2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return cls, (x1 + x2) / 2, (y1 + y2) / 2, w, h


def _box_iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0.0 else 0.0


def _load_pseudo_label_model() -> Any:
    """Load an existing vehicle detector to pseudo-label unlabeled surrounding standard vehicles."""
    from ultralytics import YOLO

    candidates = [
        ROOT / "artifacts" / "yolo_runs" / "train_yolov8n" / "weights" / "best.pt",
        ROOT / "artifacts" / "detector_best.pt",
        ROOT / "yolov8n.pt",
    ]
    for c in candidates:
        if c.exists():
            logger.info("Loading pseudo-labeler weights from %s", c)
            return YOLO(str(c))
    logger.info("Falling back to pretrained yolov8n.pt for pseudo-labeling")
    return YOLO("yolov8n.pt")


def _pseudo_label_standard_vehicles(
    model: Any,
    img_paths: List[Path],
    gt_boxes_by_img: Dict[Path, List[Tuple[int, float, float, float, float]]],
    conf: float = 0.35,
    iou_exclude: float = 0.25,
    batch_size: int = 32,
) -> Dict[Path, List[Tuple[int, float, float, float, float]]]:
    """Add class-0 (standard_vehicle) pseudo-labels for surrounding vehicles in ambulance-only frames."""
    import torch

    device = "0" if torch.cuda.is_available() else "cpu"
    is_coco_model = len(model.names) > 2
    coco_vehicle_ids = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    added_boxes = 0
    out: Dict[Path, List[Tuple[int, float, float, float, float]]] = {}

    for start in range(0, len(img_paths), batch_size):
        batch = img_paths[start : start + batch_size]
        results = model.predict(
            source=[str(p) for p in batch],
            conf=conf,
            iou=0.45,
            agnostic_nms=True,
            device=device,
            verbose=False,
        )
        for img_p, res in zip(batch, results):
            gt = list(gt_boxes_by_img.get(img_p, []))
            emg_boxes = [(xc, yc, w, h) for (cls, xc, yc, w, h) in gt if cls == 1]
            if res.boxes is not None and len(res.boxes) > 0:
                xywhn = res.boxes.xywhn.detach().cpu().numpy()
                cls_ids = res.boxes.cls.detach().cpu().numpy().astype(int)
                for box, cid in zip(xywhn, cls_ids):
                    if is_coco_model and cid not in coco_vehicle_ids:
                        continue
                    xc, yc, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    if w <= 0.01 or h <= 0.01:
                        continue
                    # Never overwrite or overlap a ground-truth emergency vehicle
                    overlaps_emg = False
                    for eb in emg_boxes:
                        if _box_iou_xywh((xc, yc, w, h), eb) >= iou_exclude:
                            overlaps_emg = True
                            break
                        # Also skip if predicted center sits inside the GT emergency box
                        if (
                            abs(xc - eb[0]) <= eb[2] * 0.45
                            and abs(yc - eb[1]) <= eb[3] * 0.45
                        ):
                            overlaps_emg = True
                            break
                    if overlaps_emg:
                        continue
                    gt.append((0, xc, yc, w, h))
                    added_boxes += 1
            out[img_p] = gt
    logger.info("Pseudo-labeled %d surrounding standard_vehicle boxes across %d images", added_boxes, len(img_paths))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(ROOT / "configs" / "datasets.yaml", "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "class_map.yaml", "r", encoding="utf-8") as f:
        map_cfg = yaml.safe_load(f)

    seed = int(ds_cfg.get("seed", 42))
    rng = random.Random(seed)
    lookup = build_lookup(map_cfg["label_to_class"])
    raw_dir = ROOT / str(ds_cfg.get("raw_data_dir", "data/raw"))
    out_dir = ROOT / str(ds_cfg.get("unified_data_dir", "data/unified"))

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise SystemExit(f"No datasets found in {raw_dir}. Run download.py first.")

    if out_dir.exists():
        shutil.rmtree(out_dir)  # derived data; always rebuilt from raw
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)

    # Index configured dataset options (max_images, pseudo_label_standard)
    configured_specs: Dict[str, Dict[str, Any]] = {}
    for spec in ds_cfg.get("datasets", []):
        key = f"{spec['workspace']}__{spec['project']}"
        configured_specs[key] = spec

    pseudo_model: Any = None
    total_counts = Counter()
    unmapped_total: Dict[str, Counter] = defaultdict(Counter)

    for ds_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if configured_specs and ds_dir.name not in configured_specs:
            logger.info("Skipping %s (not listed in configs/datasets.yaml)", ds_dir.name)
            continue
        spec = configured_specs.get(ds_dir.name, {})
        max_images: Optional[int] = spec.get("max_images")
        do_pseudo: bool = bool(spec.get("pseudo_label_standard", False))

        data_yaml = ds_dir / "data.yaml"
        if not data_yaml.exists():
            logger.warning("Skipping %s (no data.yaml)", ds_dir.name)
            continue

        names = read_class_names(data_yaml)
        id_to_new: Dict[int, Optional[int]] = {i: map_name(n, lookup) for i, n in enumerate(names)}
        logger.info("%s source classes -> %s", ds_dir.name,
                    {n: id_to_new[i] for i, n in enumerate(names)})

        kept_images, skipped_empty, skipped_dup = 0, 0, 0
        ds_counts = Counter()
        candidates: List[Tuple[Path, List[Tuple[int, float, float, float, float]]]] = []
        seen_names = set()

        for split in SPLIT_DIRS:
            img_dir, lbl_dir = ds_dir / split / "images", ds_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                if img.name in seen_names:
                    skipped_dup += 1
                    continue
                lbl = lbl_dir / f"{img.stem}.txt"
                if not lbl.exists():
                    skipped_empty += 1
                    continue

                parsed_boxes: List[Tuple[int, float, float, float, float]] = []
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parsed = parse_label_line(line)
                    if parsed is None:
                        continue
                    cls, xc, yc, w, h = parsed
                    if cls < 0 or cls >= len(names):
                        continue
                    new_cls = id_to_new.get(cls)
                    if new_cls is None:
                        unmapped_total[ds_dir.name][names[cls]] += 1
                        continue
                    parsed_boxes.append((new_cls, xc, yc, w, h))

                if not parsed_boxes:
                    skipped_empty += 1
                    continue

                seen_names.add(img.name)
                candidates.append((img, parsed_boxes))

        if max_images is not None and len(candidates) > int(max_images):
            logger.info("Downsampling %s from %d to %d images (seed=%d)",
                        ds_dir.name, len(candidates), int(max_images), seed)
            rng.shuffle(candidates)
            candidates = candidates[: int(max_images)]

        if do_pseudo and candidates:
            if pseudo_model is None:
                pseudo_model = _load_pseudo_label_model()
            gt_map = {img: boxes for img, boxes in candidates}
            updated_map = _pseudo_label_standard_vehicles(
                pseudo_model, [img for img, _ in candidates], gt_map
            )
            candidates = [(img, updated_map[img]) for img, _ in candidates]

        for img, boxes in candidates:
            out_name = f"{ds_dir.name}__{img.name}"
            out_img = out_dir / "images" / out_name
            shutil.copy2(img, out_img)
            new_lines = []
            for new_cls, xc, yc, w, h in boxes:
                new_lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                ds_counts[new_cls] += 1
            (out_dir / "labels" / f"{Path(out_name).stem}.txt").write_text(
                "\n".join(new_lines) + "\n", encoding="utf-8"
            )
            kept_images += 1

        logger.info("  kept %d images | skipped (no usable boxes) %d | duplicates %d | boxes: standard=%d emergency=%d",
                    kept_images, skipped_empty, skipped_dup, ds_counts[0], ds_counts[1])
        total_counts.update(ds_counts)

    logger.info("TOTAL boxes -> standard_vehicle: %d | emergency_vehicle: %d",
                total_counts[0], total_counts[1])

    if unmapped_total:
        logger.warning("Dropped labels that are not in class_map.yaml:")
        for ds_name, counter in unmapped_total.items():
            for label, n in counter.most_common():
                logger.warning("  %s: '%s' x%d", ds_name, label, n)
        logger.warning("If any of these should be kept, add them to configs/class_map.yaml and re-run.")

    if total_counts[1] == 0:
        logger.error("No emergency_vehicle boxes found! Check that the ambulance datasets downloaded "
                     "and that their class names are in class_map.yaml.")


if __name__ == "__main__":
    main()
