"""Build leak-free train/val/test splits from data/unified and write data/splits/data.yaml.

* Roboflow augmentations (name_jpg.rf.<hash>.jpg, *_aug*) are grouped under their
  source image so all variants land in the same split (no leakage).
* Groups containing an emergency_vehicle are split separately so every split
  gets emergency examples.

Usage (from the traffic_signal_ml folder):
    python src/data/build_splits.py
"""

from __future__ import annotations

import logging
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def group_key(stem: str) -> str:
    """Canonical source-image name: strip '.rf.<hash>' and '_aug...' suffixes."""
    stem = re.sub(r"\.rf\.[0-9a-zA-Z]+$", "", stem)
    stem = re.sub(r"_aug\w*$", "", stem, flags=re.IGNORECASE)
    return stem


def split_keys(keys: List[str], ratios: Dict[str, float], rng: random.Random) -> Dict[str, List[str]]:
    keys = sorted(keys)
    rng.shuffle(keys)
    n = len(keys)
    if n < 3:
        return {"train": keys, "val": [], "test": []}
    n_val = max(1, round(n * ratios["val"]))
    n_test = max(1, round(n * ratios["test"]))
    n_train = n - n_val - n_test
    return {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(ROOT / "configs" / "datasets.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "class_map.yaml", "r", encoding="utf-8") as f:
        class_cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    ratios = cfg["split_ratios"]
    unified = ROOT / str(cfg.get("unified_data_dir", "data/unified"))
    splits_dir = ROOT / str(cfg.get("splits_dir", "data/splits"))
    img_dir, lbl_dir = unified / "images", unified / "labels"

    if not img_dir.exists() or not any(img_dir.iterdir()):
        raise SystemExit(f"Nothing found in {img_dir}. Run unify_labels.py first.")

    # ---- group images by source image ----
    groups: Dict[str, List[Path]] = defaultdict(list)
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in IMAGE_EXTS and (lbl_dir / f"{img.stem}.txt").exists():
            groups[group_key(img.stem)].append(img)

    def has_emergency(files: List[Path]) -> bool:
        for img in files:
            for line in (lbl_dir / f"{img.stem}.txt").read_text(encoding="utf-8").splitlines():
                if line.split() and line.split()[0] == "1":
                    return True
        return False

    emergency_keys = [k for k, files in groups.items() if has_emergency(files)]
    normal_keys = [k for k in groups if k not in set(emergency_keys)]
    logger.info("%d source images (%d groups with emergency vehicles, %d without)",
                len(groups), len(emergency_keys), len(normal_keys))

    rng = random.Random(seed)
    em_split = split_keys(emergency_keys, ratios, rng)
    nm_split = split_keys(normal_keys, ratios, rng)

    # ---- write out ----
    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    stats = {s: Counter() for s in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        (splits_dir / split / "images").mkdir(parents=True)
        (splits_dir / split / "labels").mkdir(parents=True)
        for key in em_split[split] + nm_split[split]:
            for img in groups[key]:
                lbl = lbl_dir / f"{img.stem}.txt"
                shutil.copy2(img, splits_dir / split / "images" / img.name)
                shutil.copy2(lbl, splits_dir / split / "labels" / lbl.name)
                stats[split]["images"] += 1
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        stats[split][f"class_{line.split()[0]}"] += 1

    names = {int(k): v for k, v in class_cfg["classes"].items()}
    data_yaml = {
        "path": splits_dir.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    with open(splits_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    logger.info("Wrote %s", splits_dir / "data.yaml")
    for split in ("train", "val", "test"):
        s = stats[split]
        logger.info("%-5s images=%d | standard boxes=%d | emergency boxes=%d",
                    split, s["images"], s["class_0"], s["class_1"])
        if s["class_1"] < 20:
            logger.warning("  %s split has very few emergency boxes (%d); "
                           "emergency metrics will be unreliable.", split, s["class_1"])


if __name__ == "__main__":
    main()
