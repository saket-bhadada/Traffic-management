"""Download the Roboflow Universe datasets listed in configs/datasets.yaml.

Usage (from the traffic_signal_ml folder):
    python src/data/download.py
    python src/data/download.py --force      # re-download even if already present

Requires ROBOFLOW_API_KEY in a .env file (project root or traffic_signal_ml/).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# .../traffic_signal_ml/src/data/download.py -> parents[2] = traffic_signal_ml
ROOT = Path(__file__).resolve().parents[2]


def load_api_key() -> str:
    """Read ROBOFLOW_API_KEY from the environment or a .env file."""
    try:
        from dotenv import load_dotenv

        for candidate in (ROOT / ".env", ROOT.parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
    except ImportError:
        logger.warning("python-dotenv not installed; only real environment variables will be used.")

    key = os.getenv("ROBOFLOW_API_KEY", "").strip().strip('"').strip("'")
    if not key or key == "your_roboflow_api_key_here":
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set.\n"
            "Create a .env file in the project root containing:\n"
            "    ROBOFLOW_API_KEY=your_real_key\n"
        )
    return key


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Download Roboflow datasets.")
    parser.add_argument("--force", action="store_true", help="Re-download datasets that already exist.")
    parser.add_argument("--retries", type=int, default=5, help="Attempts per dataset (default 5).")
    args = parser.parse_args()

    with open(ROOT / "configs" / "datasets.yaml", "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    raw_dir = ROOT / str(cfg.get("raw_data_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())

    ok, failed = [], []
    for ds in cfg["datasets"]:
        ws, proj, ver = ds["workspace"], ds["project"], int(ds["version"])
        fmt = ds.get("format", "yolov8")
        dest = raw_dir / f"{ws}__{proj}"

        if (dest / "data.yaml").exists() and not args.force:
            logger.info("Skipping %s/%s (already downloaded at %s)", ws, proj, dest)
            ok.append(f"{ws}/{proj}")
            continue

        logger.info("Downloading %s/%s (version %d) ...", ws, proj, ver)
        success = False
        for attempt in range(1, args.retries + 1):
            try:
                if dest.exists():  # remove partial/corrupt earlier attempt
                    shutil.rmtree(dest, ignore_errors=True)
                version = rf.workspace(ws).project(proj).version(ver)
                version.download(fmt, location=str(dest))
                success = True
                break
            except Exception as exc:  # network drops are common on big files
                logger.error("Attempt %d/%d failed for %s/%s: %s", attempt, args.retries, ws, proj, exc)
        (ok if success else failed).append(f"{ws}/{proj}")

    logger.info("Downloaded/available: %d | Failed: %d", len(ok), len(failed))
    for name in failed:
        logger.info("  failed: %s  (check the workspace/project/version on its Roboflow Universe page)", name)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
