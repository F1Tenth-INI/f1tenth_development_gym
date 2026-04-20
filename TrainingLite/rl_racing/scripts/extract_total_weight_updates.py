#!/usr/bin/env python3
"""Extract final total_weight_updates values from model learning_metrics.csv files.

This script mirrors the shell pipeline used previously:
- Scan model dirs matching a prefix (default: RCA2-1*).
- Read learning_metrics.csv from each model dir.
- Locate the total_weight_updates column by header name.
- Take the value from the last non-empty row.
- Keep numeric values, sort ascending, print values and summary stats.

Usage examples:
  python TrainingLite/rl_racing/scripts/extract_total_weight_updates.py
  python TrainingLite/rl_racing/scripts/extract_total_weight_updates.py --prefix RCA2-1_finetune
  python TrainingLite/rl_racing/scripts/extract_total_weight_updates.py --integers-only
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("TrainingLite/rl_racing/models"),
        help="Root directory containing model folders.",
    )
    parser.add_argument(
        "--prefix",
        default="RCA2-1",
        help="Model folder prefix to include.",
    )
    parser.add_argument(
        "--column",
        default="total_weight_updates",
        help="CSV column name to extract.",
    )
    parser.add_argument(
        "--integers-only",
        action="store_true",
        help="Keep only integer-valued entries (drop fractional values).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path for model|value lines.",
    )
    return parser.parse_args()


def is_numeric(value: str) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def is_integer_value(x: float, tol: float = 1e-12) -> bool:
    return abs(x - round(x)) <= tol


def read_last_non_empty_row(csv_path: Path) -> list[str] | None:
    last_row: list[str] | None = None
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if any(cell.strip() for cell in row):
                last_row = row
    return last_row


def extract_values(
    models_root: Path,
    prefix: str,
    column_name: str,
    integers_only: bool,
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []

    if not models_root.exists():
        raise FileNotFoundError(f"models root not found: {models_root}")

    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith(prefix):
            continue

        metrics = model_dir / "learning_metrics.csv"
        if not metrics.is_file():
            continue

        with metrics.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue

        if column_name not in header:
            continue

        idx = header.index(column_name)
        print(idx)
        last_row = read_last_non_empty_row(metrics)
        if last_row is None or last_row == header or idx >= len(last_row):
            continue

        raw = last_row[idx].strip().strip('"')
        if not is_numeric(raw):
            continue

        value = float(raw)
        if integers_only and not is_integer_value(value):
            continue

        results.append((model_dir.name, value))

    results.sort(key=lambda x: x[1])
    return results


def print_results(results: list[tuple[str, float]]) -> None:
    for model, value in results:
        print(f"{model}|{value:g}")

    print("---STATS---")
    if not results:
        print("count=0")
        return

    values = [v for _, v in results]
    print(f"count={len(values)}")
    print(f"min={min(values):.6f}")
    print(f"max={max(values):.6f}")
    print(f"range={max(values) - min(values):.6f}")
    print(f"mean={mean(values):.6f}")


def maybe_write_output(output_path: Path | None, results: list[tuple[str, float]]) -> None:
    if output_path is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        for model, value in results:
            f.write(f"{model}|{value:g}\n")


if __name__ == "__main__":
    args = parse_args()
    extracted = extract_values(
        models_root=args.models_root,
        prefix=args.prefix,
        column_name=args.column,
        integers_only=args.integers_only,
    )
    print_results(extracted)
    maybe_write_output(args.output, extracted)
