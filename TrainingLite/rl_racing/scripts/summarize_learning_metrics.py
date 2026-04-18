#!/usr/bin/env python3
"""
Summarize milestone timings from model learning_metrics.csv files.

This script scans one model or a prefix batch and extracts the first row where
key training milestones are reached. To avoid early planner fallback artifacts,
the first N rows can be skipped (default: 1).
"""

import argparse
import ast
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = ROOT_DIR / "TrainingLite" / "rl_racing" / "models"


def _abbreviate_column_name(name: str) -> str:
    return name.replace("first", "1st").replace("episode", "ep").replace("reward", "rew")


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_number_list(value: Optional[str]) -> List[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []

    out: List[float] = []
    for item in parsed:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _set_first_event(
    events: Dict[str, Dict[str, Optional[float]]],
    event_name: str,
    row_index: int,
    row: Dict[str, str],
    matched_value: Optional[float] = None,
) -> None:
    if event_name in events:
        return
    events[event_name] = {
        "row_index": float(row_index),
        "time": _safe_float(row.get("time")),
        "total_timesteps": _safe_float(row.get("total_timesteps")),
        "total_weight_updates": _safe_float(row.get("total_weight_updates")),
        "matched_value": matched_value,
    }


def summarize_model_csv(csv_path: Path, skip_initial_rows: int, max_episode_length: int) -> Dict[str, object]:
    events: Dict[str, Dict[str, Optional[float]]] = {}
    rows_seen = 0
    rows_used = 0
    last_row: Optional[Dict[str, str]] = None
    fastest_recorded_laptime: Optional[float] = None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows_seen += 1
            all_lap_times = _parse_number_list(row.get("lap_times"))
            if all_lap_times:
                row_min_lap = min(all_lap_times)
                if fastest_recorded_laptime is None or row_min_lap < fastest_recorded_laptime:
                    fastest_recorded_laptime = row_min_lap

            if idx < skip_initial_rows:
                continue
            rows_used += 1
            last_row = row

            lap_times = _parse_number_list(row.get("lap_times"))
            mean_step_rewards = _parse_number_list(row.get("episode_mean_step_rewards"))
            episode_rewards = _parse_number_list(row.get("episode_rewards"))
            episode_lengths = _parse_number_list(row.get("episode_lengths"))

            if lap_times:
                _set_first_event(events, "first_lap", idx, row, matched_value=min(lap_times))

                laps_under_25 = [v for v in lap_times if v < 25.0]
                if laps_under_25:
                    _set_first_event(
                        events,
                        "first_lap_lt_25s",
                        idx,
                        row,
                        matched_value=min(laps_under_25),
                    )

                laps_under_20 = [v for v in lap_times if v < 20.0]
                if laps_under_20:
                    _set_first_event(
                        events,
                        "first_lap_lt_20s",
                        idx,
                        row,
                        matched_value=min(laps_under_20),
                    )

            rewards_gt_0 = [v for v in mean_step_rewards if v > 0.0]
            if rewards_gt_0:
                _set_first_event(
                    events,
                    "first_episode_mean_step_reward_gt_0",
                    idx,
                    row,
                    matched_value=max(rewards_gt_0),
                )

            episode_rewards_gt_100 = [v for v in episode_rewards if v > 100.0]
            if episode_rewards_gt_100:
                _set_first_event(
                    events,
                    "first_episode_reward_gt_100",
                    idx,
                    row,
                    matched_value=max(episode_rewards_gt_100),
                )

            max_len_hits = [v for v in episode_lengths if int(v) >= max_episode_length]
            if max_len_hits:
                _set_first_event(
                    events,
                    f"first_episode_len_ge_{max_episode_length}",
                    idx,
                    row,
                    matched_value=max(max_len_hits),
                )

    summary: Dict[str, object] = {
        "rows_seen": rows_seen,
        "rows_used": rows_used,
        "fastest_recorded_laptime": fastest_recorded_laptime,
    }

    if last_row is not None:
        summary["final_time"] = _safe_float(last_row.get("time"))
        summary["final_total_timesteps"] = _safe_float(last_row.get("total_timesteps"))
        summary["final_total_weight_updates"] = _safe_float(last_row.get("total_weight_updates"))
    else:
        summary["final_time"] = None
        summary["final_total_timesteps"] = None
        summary["final_total_weight_updates"] = None

    for event_name, event_data in events.items():
        if event_name == f"first_episode_len_ge_{max_episode_length}" and "matched_value" in event_data:
            event_data = {k: v for k, v in event_data.items() if k != "matched_value"}
        for field_name, field_value in event_data.items():
            summary[_abbreviate_column_name(f"{event_name}_{field_name}")] = field_value

    return summary


def find_models(models_dir: Path, name: Optional[str], prefix: Optional[str]) -> List[str]:
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    if name:
        model_dir = models_dir / name
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return [name]

    if not prefix:
        raise ValueError("Provide either --name or --prefix")

    matches: List[str] = []
    for item in sorted(models_dir.iterdir()):
        if item.is_dir() and item.name.startswith(prefix):
            matches.append(item.name)
    return matches


def build_output_path(out_file: Optional[str], name: Optional[str], prefix: Optional[str]) -> Path:
    if out_file:
        output = Path(out_file)
        if not output.is_absolute():
            output = ROOT_DIR / output
        return output

    if name:
        tag = name
    elif prefix:
        tag = prefix
    else:
        tag = "selection"

    safe_tag = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in tag)
    return ROOT_DIR / "batch_learning_metrics_summary" / safe_tag / f"learning_metrics_summary_{safe_tag}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize milestone timings from learning_metrics.csv files")
    parser.add_argument("--name", help="Exact model directory name to process")
    parser.add_argument("--prefix", help="Process all model directories with this prefix")
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Base models directory (default: TrainingLite/rl_racing/models)",
    )
    parser.add_argument(
        "--csv-name",
        default="learning_metrics.csv",
        help="CSV filename inside each model directory (default: learning_metrics.csv)",
    )
    parser.add_argument(
        "--skip-initial-rows",
        type=int,
        default=1,
        help="Rows to skip at top of each CSV to avoid startup artifacts (default: 1)",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=2000,
        help="Max episode length threshold for milestone detection (default: 2000)",
    )
    parser.add_argument(
        "--out-file",
        help="Output CSV path (default: batch_learning_metrics_summary/<tag>/learning_metrics_summary_<tag>.csv)",
    )

    args = parser.parse_args()
    if not args.name and not args.prefix:
        parser.error("Provide either --name or --prefix")

    models_dir = Path(args.models_dir)
    models = find_models(models_dir=models_dir, name=args.name, prefix=args.prefix)
    if not models:
        target = args.name if args.name else args.prefix
        print(f"No matching model directories found for: {target}")
        return

    output_path = build_output_path(args.out_file, args.name, args.prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for model in models:
        csv_path = models_dir / model / args.csv_name
        if not csv_path.is_file():
            print(f"[WARN] Missing CSV, skipping: {csv_path}")
            continue

        summary = summarize_model_csv(
            csv_path=csv_path,
            skip_initial_rows=max(0, args.skip_initial_rows),
            max_episode_length=args.max_episode_length,
        )
        summary["name"] = model
        summary["csv_path"] = str(csv_path)
        rows.append(summary)

    if not rows:
        print("No model summaries were generated.")
        return

    all_fields = sorted({key for row in rows for key in row.keys()})
    front_fields = ["name"]
    end_fields = [
        "csv_path",
        "rows_used",
        "final_time",
        "final_total_timesteps",
        "final_total_weight_updates",
    ]
    middle_fields = [f for f in all_fields if f not in set(front_fields + end_fields)]
    ordered_fields = front_fields + middle_fields + [f for f in end_fields if f in all_fields]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Processed models: {len(rows)}")
    print(f"Summary CSV: {output_path}")


if __name__ == "__main__":
    main()
