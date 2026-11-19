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
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(ROOT / "configs" / "datasets.yaml", "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "class_map.yaml", "r", encoding="utf-8") as f:
        map_cfg = yaml.safe_load(f)

    lookup = build_lookup(map_cfg["label_to_class"])
    raw_dir = ROOT / str(ds_cfg.get("raw_data_dir", "data/raw"))
    out_dir = ROOT / str(ds_cfg.get("unified_data_dir", "data/unified"))

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise SystemExit(f"No datasets found in {raw_dir}. Run download.py first.")

    if out_dir.exists():
        shutil.rmtree(out_dir)  # derived data; always rebuilt from raw
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)

    total_counts = Counter()
    unmapped_total: Dict[str, Counter] = defaultdict(Counter)

    for ds_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
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

        for split in SPLIT_DIRS:
            img_dir, lbl_dir = ds_dir / split / "images", ds_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl = lbl_dir / f"{img.stem}.txt"
                if not lbl.exists():
                    skipped_empty += 1
                    continue

                new_lines: List[str] = []
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
                    new_lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    ds_counts[new_cls] += 1

                if not new_lines:
                    skipped_empty += 1
                    continue

                out_name = f"{ds_dir.name}__{img.name}"
                out_img = out_dir / "images" / out_name
                if out_img.exists():  # same image appearing in two original splits
                    skipped_dup += 1
                    for line in new_lines:
                        ds_counts[int(line.split()[0])] -= 1
                    continue
                shutil.copy2(img, out_img)
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
