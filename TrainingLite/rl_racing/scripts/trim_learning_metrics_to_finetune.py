#!/usr/bin/env python3
"""Trim combined learning_metrics.csv files to finetune-only sections.

For each matching model directory:
1. Read learning_metrics.csv.
2. Find the first row where replay_buffer_size reaches the threshold (default 100000).
3. Find the first subsequent row where replay_buffer_size drops below the threshold.
4. Keep rows from that drop onward (finetune section), remove earlier rows.
5. Zero-base cumulative counters in the trimmed section so timesteps and
    weight updates reflect only the finetune run.
6. Backup original CSV as learning_metrics_full.csv.
7. Write trimmed data back to learning_metrics.csv.

By default this runs in dry-run mode. Use --apply to perform file changes.

python TrainingLite/rl_racing/scripts/trim_learning_metrics_to_finetune.py --prefix RCA2-Final
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelResult:
    model: str
    status: str
    detail: str


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
        default="",
        help="Only process model directories starting with this prefix.",
    )
    parser.add_argument(
        "--name-contains",
        default="",
        help="Only process model directories containing this substring.",
    )
    parser.add_argument(
        "--metrics-name",
        default="learning_metrics.csv",
        help="Learning metrics CSV filename.",
    )
    parser.add_argument(
        "--backup-name",
        default="learning_metrics_full.csv",
        help="Backup filename used for the full, untrimmed CSV.",
    )
    parser.add_argument(
        "--replay-column",
        default="replay_buffer_size",
        help="CSV column used to detect finetune start.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100000.0,
        help="Threshold at which replay buffer is considered full.",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Deprecated: backup files are now preserved and never overwritten.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, script runs as dry-run.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Rebuild learning_metrics.csv from the preserved backup if it already exists.",
    )
    return parser.parse_args()


def to_float(cell: str) -> float | None:
    raw = cell.strip().strip('"')
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def format_float_cell(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.6f}"
    return text.rstrip("0").rstrip(".")


def renormalize_counter_columns(header: list[str], rows: list[list[str]]) -> list[list[str]]:
    """Shift cumulative counters so the trimmed finetune section starts at zero."""
    if not rows:
        return rows

    column_to_index = {name: idx for idx, name in enumerate(header)}
    timesteps_idx = column_to_index.get("total_timesteps")
    updates_idx = column_to_index.get("total_weight_updates")
    udt_idx = column_to_index.get("UDT")

    if timesteps_idx is None and updates_idx is None:
        return rows

    base_timesteps = to_float(rows[0][timesteps_idx]) if timesteps_idx is not None and timesteps_idx < len(rows[0]) else None
    base_updates = to_float(rows[0][updates_idx]) if updates_idx is not None and updates_idx < len(rows[0]) else None

    if base_timesteps is None and base_updates is None:
        return rows

    renormalized_rows: list[list[str]] = []
    for row in rows:
        new_row = list(row)

        delta_timesteps: float | None = None
        delta_updates: float | None = None

        if timesteps_idx is not None and timesteps_idx < len(new_row):
            current_timesteps = to_float(new_row[timesteps_idx])
            if current_timesteps is not None:
                base_value = base_timesteps if base_timesteps is not None else 0.0
                delta_timesteps = max(0.0, current_timesteps - base_value)
                new_row[timesteps_idx] = format_float_cell(delta_timesteps)

        if updates_idx is not None and updates_idx < len(new_row):
            current_updates = to_float(new_row[updates_idx])
            if current_updates is not None:
                base_value = base_updates if base_updates is not None else 0.0
                delta_updates = max(0.0, current_updates - base_value)
                new_row[updates_idx] = format_float_cell(delta_updates)

        if udt_idx is not None and udt_idx < len(new_row):
            if delta_timesteps is None and timesteps_idx is not None and timesteps_idx < len(new_row):
                delta_timesteps = to_float(new_row[timesteps_idx])
            if delta_updates is None and updates_idx is not None and updates_idx < len(new_row):
                delta_updates = to_float(new_row[updates_idx])

            if delta_timesteps is not None and delta_updates is not None and delta_timesteps > 0.0:
                new_row[udt_idx] = format_float_cell(delta_updates / delta_timesteps)
            else:
                new_row[udt_idx] = "0"

        renormalized_rows.append(new_row)

    return renormalized_rows


def find_finetune_start_index(values: list[float | None], threshold: float) -> tuple[int | None, str]:
    first_reach = None
    for idx, value in enumerate(values):
        if value is not None and value >= threshold:
            first_reach = idx
            break

    if first_reach is None:
        return None, f"replay_buffer_size never reached threshold {threshold:g}"

    for idx in range(first_reach + 1, len(values)):
        value = values[idx]
        if value is not None and value < threshold:
            return idx, "ok"

    return None, "no drop below threshold found after reaching threshold"


def process_model_dir(model_dir: Path, args: argparse.Namespace) -> ModelResult:
    metrics_path = model_dir / args.metrics_name
    backup_path = model_dir / args.backup_name

    # Safeguard: if both files are present, treat the model as already processed
    # unless the caller explicitly wants to rebuild the trimmed CSV from backup.
    if metrics_path.is_file() and backup_path.is_file() and not args.reprocess:
        return ModelResult(
            model_dir.name,
            "skipped",
            f"both {args.metrics_name} and {args.backup_name} exist (already processed)",
        )

    source_path = backup_path if args.reprocess and backup_path.is_file() else metrics_path

    if not source_path.is_file():
        return ModelResult(model_dir.name, "skipped", f"missing {args.metrics_name}")

    with source_path.open("r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return ModelResult(model_dir.name, "skipped", "empty CSV")

    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        return ModelResult(model_dir.name, "skipped", "no data rows")

    if args.replay_column not in header:
        return ModelResult(model_dir.name, "skipped", f"missing column {args.replay_column}")

    replay_idx = header.index(args.replay_column)
    replay_values = [to_float(r[replay_idx]) if replay_idx < len(r) else None for r in data_rows]

    start_idx, reason = find_finetune_start_index(replay_values, args.threshold)
    if start_idx is None:
        return ModelResult(model_dir.name, "skipped", reason)

    trimmed_rows = data_rows[start_idx:]
    if not trimmed_rows:
        return ModelResult(model_dir.name, "skipped", "trim would produce empty CSV")

    trimmed_rows = renormalize_counter_columns(header, trimmed_rows)

    removed = len(data_rows) - len(trimmed_rows)
    if not args.apply:
        return ModelResult(
            model_dir.name,
            "dry-run",
            f"would keep {len(trimmed_rows)} rows, remove {removed}, backup -> {args.backup_name}",
        )

    if source_path == metrics_path and not backup_path.exists():
        shutil.copy2(str(metrics_path), str(backup_path))

    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(trimmed_rows)

    return ModelResult(
        model_dir.name,
        "updated",
        f"kept {len(trimmed_rows)} rows, removed {removed}, backup saved as {args.backup_name}",
    )


def model_matches(model_name: str, prefix: str, contains: str) -> bool:
    if prefix and not model_name.startswith(prefix):
        return False
    if contains and contains not in model_name:
        return False
    return True


def main() -> None:
    args = parse_args()

    if not args.models_root.exists():
        raise SystemExit(f"models root not found: {args.models_root}")

    results: list[ModelResult] = []
    for model_dir in sorted(args.models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if not model_matches(model_dir.name, args.prefix, args.name_contains):
            continue
        results.append(process_model_dir(model_dir, args))

    if not results:
        print("No matching model directories found.")
        return

    updated = sum(r.status == "updated" for r in results)
    dry_run = sum(r.status == "dry-run" for r in results)
    skipped = sum(r.status == "skipped" for r in results)

    for result in results:
        print(f"{result.model}: {result.status} - {result.detail}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print("---SUMMARY---")
    print(f"mode={mode}")
    print(f"total={len(results)}")
    print(f"updated={updated}")
    print(f"dry_run={dry_run}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
