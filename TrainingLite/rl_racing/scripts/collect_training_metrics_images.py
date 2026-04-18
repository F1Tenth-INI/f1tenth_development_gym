#!/usr/bin/env python3
"""
Collect training_metrics.png from all models matching a prefix.

For each model in TrainingLite/rl_racing/models that starts with the given prefix,
this script copies that model's training metrics image into a single batch folder and
renames it to <model_name>.png.

Usage examples:
    python -u TrainingLite/rl_racing/scripts/collect_training_metrics_images.py --prefix Sweep_rank_Ex1_A0.0
    python -u TrainingLite/rl_racing/scripts/collect_training_metrics_images.py --prefix 2602 --output-base-dir batch_training_metrics
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List

# Add root dir to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_dir)


class TrainingMetricsCollector:
    """Collect and rename training metrics images for matching model folders."""

    def __init__(
        self,
        prefix: str,
        output_base_dir: str = "batch_training_metrics",
        source_filename: str = "training_metrics.png",
        verbose: bool = True,
    ):
        self.prefix = prefix
        self.source_filename = source_filename
        self.verbose = verbose

        self.models_dir = Path(root_dir) / "TrainingLite" / "rl_racing" / "models"

        safe_prefix = self._sanitize_for_path(prefix)
        self.output_dir = Path(root_dir) / output_base_dir / f"batch_{safe_prefix}_training_metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_for_path(text: str) -> str:
        """Make a folder-safe name while preserving readability."""
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
        return sanitized or "prefix"

    def find_models_by_prefix(self) -> List[str]:
        """Return sorted model names that start with the prefix."""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        models = []
        for item in sorted(self.models_dir.iterdir()):
            if item.is_dir() and item.name.startswith(self.prefix):
                models.append(item.name)

        return models

    def collect(self) -> int:
        """Copy and rename matching images into the batch output folder."""
        models = self.find_models_by_prefix()
        if not models:
            print(f"No model directories found with prefix: {self.prefix}")
            print(f"Searched in: {self.models_dir}")
            return 0

        print(f"Found {len(models)} model(s) with prefix '{self.prefix}'")
        print(f"Source filename: {self.source_filename}")
        print(f"Output directory: {self.output_dir}")

        copied = 0
        missing = 0

        for model_name in models:
            src = self.models_dir / model_name / self.source_filename
            dst = self.output_dir / f"{model_name}.png"

            if not src.is_file():
                missing += 1
                if self.verbose:
                    print(f"  [missing] {model_name}: {src}")
                continue

            shutil.copy2(src, dst)
            copied += 1
            if self.verbose:
                print(f"  [copied]  {src.name} -> {dst.name}")

        print("\nCollection summary")
        print(f"  Models matched: {len(models)}")
        print(f"  Images copied:  {copied}")
        print(f"  Missing images: {missing}")
        print(f"  Output folder:  {self.output_dir}")

        return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and rename training metrics images for all models matching a prefix."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Model prefix to match (e.g., Sweep_rank_Ex1_A0.0)",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="batch_training_metrics",
        help="Base output folder relative to repo root (default: batch_training_metrics)",
    )
    parser.add_argument(
        "--source-filename",
        type=str,
        default="training_metrics.png",
        help="Image filename to collect from each model folder (default: training_metrics.png)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-model logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    collector = TrainingMetricsCollector(
        prefix=args.prefix,
        output_base_dir=args.output_base_dir,
        source_filename=args.source_filename,
        verbose=not args.quiet,
    )
    collector.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
