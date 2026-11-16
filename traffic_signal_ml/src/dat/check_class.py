"""Draw boxes of one class on a few sample images from a raw dataset, to see what the class really is.

Usage (from the traffic_signal_ml folder):
    python src/data/check_class.py "ph-ambulance-dataset-with-augmentations__ph-ambulances" "Misc Vehicle"
    python src/data/check_class.py "himank-vpetc__ambulance-4bova" "My First Project - v1 2025-02-17 5-45am"

Red   = the class you asked about
Green = classes named 'ambulance' (for comparison)
Output: check_out/<class>_0.jpg ... in the current folder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", help="Folder name inside data/raw")
    parser.add_argument("class_name", help="Class name exactly as in the log")
    parser.add_argument("--count", type=int, default=6, help="How many sample images to save")
    args = parser.parse_args()

    ds = ROOT / "data" / "raw" / args.dataset_folder
    with open(ds / "data.yaml", "r", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]

    if args.class_name not in names:
        raise SystemExit(f"Class '{args.class_name}' not found. Classes are: {names}")
    target = names.index(args.class_name)
    ambulance_ids = {i for i, n in enumerate(names) if "ambulance" in n.lower()}
    print(f"Class '{args.class_name}' has id {target}")

    out_dir = Path("check_out")
    out_dir.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", args.class_name)[:30]

    saved = 0
    seen_sources = set()
    for lbl in sorted(ds.glob("*/labels/*.txt")):
        rows = [l.split() for l in lbl.read_text(encoding="utf-8").splitlines() if len(l.split()) >= 5]
        if not any(int(float(r[0])) == target for r in rows):
            continue
        source = re.sub(r"\.rf\.[0-9a-zA-Z]+$", "", lbl.stem)  # skip augmented copies of same image
        if source in seen_sources:
            continue
        img_path = next((p for p in (lbl.parent.parent / "images").glob(lbl.stem + ".*")), None)
        if img_path is None:
            continue
        seen_sources.add(source)

        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for r in rows:
            cid = int(float(r[0]))
            if cid != target and cid not in ambulance_ids:
                continue
            vals = [float(v) for v in r[1:]]
            if len(vals) == 4:
                xc, yc, w, h = vals
            else:  # polygon -> bounding box
                xs, ys = vals[0::2], vals[1::2]
                xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            color = "red" if cid == target else "lime"
            d.rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H],
                        outline=color, width=max(3, W // 200))
        im.thumbnail((900, 900))
        out = out_dir / f"{safe}_{saved}.jpg"
        im.save(out)
        saved += 1
        if saved >= args.count:
            break

    print(f"Saved {saved} images to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
