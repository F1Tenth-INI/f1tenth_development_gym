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
import math
import os
from statistics import median
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = ROOT_DIR / "TrainingLite" / "rl_racing" / "models"


def _abbreviate_column_name(name: str) -> str:
    return name.replace("first", "1st").replace("episode", "ep").replace("reward", "rew")


METRIC_DEFINITIONS = [
    {"event_name": "first_lap_lt_30s", "label": "first lap under 30s", "kind": "lap", "threshold": 30.0},
    {"event_name": "first_lap_lt_27_5s", "label": "first lap under 27.5s", "kind": "lap", "threshold": 27.5},
    {"event_name": "first_lap_lt_25s", "label": "first lap under 25s", "kind": "lap", "threshold": 25.0},
    {
        "event_name": "first_episode_len_ge_2000",
        "label": "first episode length >= 2000",
        "kind": "episode_length",
        "threshold": 2000.0,
    },
    {
        "event_name": "first_episode_len_gt_500_mean_step_reward_gt_0_00",
        "label": "first episode length > 500 and mean step reward > 0.00",
        "kind": "episode_length_and_reward",
        "length_threshold": 500.0,
        "reward_threshold": 0.0,
    },
    {
        "event_name": "first_episode_len_gt_500_mean_step_reward_gt_0_01",
        "label": "first episode length > 500 and mean step reward > 0.01",
        "kind": "episode_length_and_reward",
        "length_threshold": 500.0,
        "reward_threshold": 0.01,
    },
    {
        "event_name": "first_episode_len_gt_500_mean_step_reward_gt_0_02",
        "label": "first episode length > 500 and mean step reward > 0.02",
        "kind": "episode_length_and_reward",
        "length_threshold": 500.0,
        "reward_threshold": 0.02,
    },
]

REFERENCE_METRICS = [{"event_name": "first_lap", "label": "first lap"}] + METRIC_DEFINITIONS
RANKED_METRICS = METRIC_DEFINITIONS
SCORE_ALPHA = 0.5
MISSING_METRIC_PENALTY_FACTOR = 1.5
DEFAULT_OVERALL_TOP_N = 30

RANKING_MODES = {
    "all": METRIC_DEFINITIONS,
    "reward_only": [metric for metric in METRIC_DEFINITIONS if metric["kind"] == "episode_length_and_reward"],
}


def _metric_label(event_name: str) -> str:
    label = event_name.replace("first_", "First ").replace("_", " ")
    label = label.replace(" lt ", " < ").replace(" gt ", " > ").replace(" ge ", " >= ")
    return label


def _first_qualifying_episode_pair(
    episode_lengths: List[float],
    mean_step_rewards: List[float],
    min_length: float,
    reward_threshold: float,
) -> Optional[Dict[str, float]]:
    for length, reward in zip(episode_lengths, mean_step_rewards):
        if length > min_length and reward > reward_threshold:
            return {
                "matched_value": reward,
                "matched_episode_length": length,
                "matched_episode_mean_step_reward": reward,
            }
    return None


def _collect_reference_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    reference_rows: List[Dict[str, object]] = []
    for metric in REFERENCE_METRICS:
        prefix = _abbreviate_column_name(metric["event_name"])
        timesteps_key = f"{prefix}_total_timesteps"
        updates_key = f"{prefix}_total_weight_updates"

        timesteps: List[float] = []
        updates: List[float] = []
        for row in rows:
            timestep = row.get(timesteps_key)
            update = row.get(updates_key)
            if isinstance(timestep, (int, float)) and isinstance(update, (int, float)):
                timesteps.append(float(timestep))
                updates.append(float(update))

        reference_rows.append(
            {
                "metric_id": metric["event_name"],
                "metric_label": metric.get("label", _metric_label(metric["event_name"])),
                "sample_count": len(timesteps),
                "Tref": median(timesteps) if timesteps else None,
                "Uref": median(updates) if updates else None,
            }
        )

    return reference_rows


def _apply_metric_rankings(rows: List[Dict[str, object]], alpha: float) -> None:
    """Add per-metric normalized score and rank columns to each summary row.

    Score formula (lower is better):
      score = (T / Tref)^alpha * (U / Uref)^(1 - alpha)
    where Tref/Uref are medians across available models for that metric.
    """
    for metric in RANKED_METRICS:
        prefix = _abbreviate_column_name(metric["event_name"])
        timesteps_key = f"{prefix}_total_timesteps"
        updates_key = f"{prefix}_total_weight_updates"
        score_key = f"{prefix}_speed_score_a0_5"
        rank_key = f"{prefix}_speed_rank_a0_5"

        valid_timesteps: List[float] = []
        valid_updates: List[float] = []
        for row in rows:
            timestep = row.get(timesteps_key)
            update = row.get(updates_key)
            if isinstance(timestep, (int, float)) and isinstance(update, (int, float)):
                valid_timesteps.append(float(timestep))
                valid_updates.append(float(update))

        if not valid_timesteps or not valid_updates:
            for row in rows:
                row[score_key] = None
                row[rank_key] = None
            continue

        tref = median(valid_timesteps)
        uref = median(valid_updates)
        if tref <= 0.0 or uref <= 0.0:
            for row in rows:
                row[score_key] = None
                row[rank_key] = None
            continue

        scored_rows: List[tuple[float, Dict[str, object]]] = []
        for row in rows:
            timestep = row.get(timesteps_key)
            update = row.get(updates_key)
            if isinstance(timestep, (int, float)) and isinstance(update, (int, float)):
                t = float(timestep)
                u = float(update)
                if t > 0.0 and u > 0.0:
                    score = math.pow(t / tref, alpha) * math.pow(u / uref, 1.0 - alpha)
                    row[score_key] = score
                    scored_rows.append((score, row))
                    continue
            row[score_key] = None
            row[rank_key] = None

        scored_rows.sort(key=lambda item: item[0])
        for rank, (_, row) in enumerate(scored_rows, start=1):
            row[rank_key] = rank


def _compute_overall_rankings(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Compute overall ranking from per-metric scores.

    The final ranking uses the geometric mean of the per-metric scores, where
    lower is better. Missing metrics receive a score worse than the worst
    achieved score for that metric so they still affect the overall score.
    """
    metric_specs: List[Dict[str, object]] = []
    active_metric_count = 0

    for metric in RANKED_METRICS:
        prefix = _abbreviate_column_name(metric["event_name"])
        rank_key = f"{prefix}_speed_rank_a0_5"
        score_key = f"{prefix}_speed_score_a0_5"
        timesteps_key = f"{prefix}_total_timesteps"
        updates_key = f"{prefix}_total_weight_updates"
        time_key = f"{prefix}_time"
        matched_value_key = f"{prefix}_matched_value"

        rank_by_name: Dict[str, int] = {}
        score_by_name: Dict[str, float] = {}
        max_rank = 0
        valid_scores: List[float] = []
        for row in rows:
            name = str(row.get("name"))
            rank = row.get(rank_key)
            if isinstance(rank, (int, float)):
                as_int = int(rank)
                rank_by_name[name] = as_int
                if as_int > max_rank:
                    max_rank = as_int

            score = row.get(score_key)
            if isinstance(score, (int, float)) and float(score) > 0.0:
                score_value = float(score)
                score_by_name[name] = score_value
                valid_scores.append(score_value)

        penalty_score: Optional[float] = None
        if valid_scores:
            active_metric_count += 1
            penalty_score = max(valid_scores) * MISSING_METRIC_PENALTY_FACTOR

        metric_specs.append(
            {
                "metric_id": metric["event_name"],
                "metric_label": metric.get("label", metric["event_name"]),
                "rank_key": rank_key,
                "score_key": score_key,
                "timesteps_key": timesteps_key,
                "updates_key": updates_key,
                "time_key": time_key,
                "matched_value_key": matched_value_key,
                "rank_by_name": rank_by_name,
                "score_by_name": score_by_name,
                "max_rank": max_rank,
                "penalty_score": penalty_score,
            }
        )

    overall_rows: List[Dict[str, object]] = []
    for row in rows:
        name = str(row.get("name"))
        points = 0
        covered = 0
        score_values: List[float] = []
        per_metric: List[Dict[str, object]] = []

        for spec in metric_specs:
            rank_by_name = spec["rank_by_name"]
            score_by_name = spec["score_by_name"]
            max_rank = int(spec["max_rank"])
            penalty_score = spec["penalty_score"]
            rank_value = rank_by_name.get(name)
            score_value = score_by_name.get(name)
            penalty = (max_rank + 1) if max_rank > 0 else None

            if rank_value is not None:
                points += rank_value
                covered += 1
            elif penalty is not None:
                points += penalty

            effective_score = score_value if score_value is not None else penalty_score
            if effective_score is not None and effective_score > 0.0:
                score_values.append(effective_score)

            per_metric.append(
                {
                    "metric_id": spec["metric_id"],
                    "metric_label": spec["metric_label"],
                    "rank": rank_value,
                    "penalty": penalty,
                    "score": score_value,
                    "effective_score": effective_score,
                    "penalty_score": penalty_score,
                    "total_timesteps": row.get(spec["timesteps_key"]),
                    "total_weight_updates": row.get(spec["updates_key"]),
                    "time": row.get(spec["time_key"]),
                    "matched_value": row.get(spec["matched_value_key"]),
                }
            )

        overall_score = (
            math.exp(sum(math.log(score) for score in score_values) / len(score_values))
            if score_values
            else None
        )

        points_value: Optional[int] = points if active_metric_count > 0 else None
        row["overall_score"] = overall_score
        row["overall_rank_points_a0_5"] = points_value
        row["overall_score_metrics_covered"] = covered
        row["overall_score_metrics_total"] = active_metric_count

        overall_rows.append(
            {
                "name": name,
                "score": overall_score,
                "points": points_value,
                "metrics_covered": covered,
                "metrics_total": active_metric_count,
                "final_total_weight_updates": row.get("final_total_weight_updates"),
                "per_metric": per_metric,
            }
        )

    sortable = [r for r in overall_rows if isinstance(r.get("score"), (int, float))]
    sortable.sort(
        key=lambda r: (
            float(r["score"]),
            -int(r["metrics_covered"]),
            int(r["points"]) if isinstance(r.get("points"), int) else math.inf,
            str(r["name"]),
        )
    )

    position_by_name: Dict[str, int] = {}
    for idx, entry in enumerate(sortable, start=1):
        position_by_name[str(entry["name"])] = idx
        entry["position"] = idx

    for row in rows:
        row["overall_score_position"] = position_by_name.get(str(row.get("name")))

    for entry in overall_rows:
        if "position" not in entry:
            entry["position"] = None

    overall_rows.sort(
        key=lambda r: (
            math.inf if r.get("position") is None else int(r["position"]),
            str(r["name"]),
        )
    )
    return overall_rows


def average_same_model_rows(rows: Optional[List[Dict[str, object]]] = None) -> List[Dict[str, object]]:
    if not rows:
        return []

    grouped_rows: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        raw_name = str(row.get("name", ""))
        group_name = raw_name.rsplit("_", 1)[0] if "_" in raw_name else raw_name
        grouped_rows.setdefault(group_name, []).append(row)

    averaged_rows: List[Dict[str, object]] = []
    for group_name, entries in grouped_rows.items():
        averaged: Dict[str, object] = {
            "name": group_name,
            "csv_path": None,
            "group_count": len(entries),
        }

        all_keys = {k for entry in entries for k in entry.keys()}
        for key in all_keys:
            if key in {"name", "csv_path"}:
                continue

            values = [entry.get(key) for entry in entries]
            numeric_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]

            if numeric_values:
                averaged[key] = sum(numeric_values) / len(numeric_values)
                continue

            non_null_values = [value for value in values if value not in (None, "")]
            if non_null_values and all(value == non_null_values[0] for value in non_null_values):
                averaged[key] = non_null_values[0]
            else:
                averaged[key] = None

        averaged_rows.append(averaged)

    return averaged_rows


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
    extra_fields: Optional[Dict[str, Optional[float]]] = None,
) -> None:
    if event_name in events:
        return
    event_data: Dict[str, Optional[float]] = {
        "row_index": float(row_index),
        "time": _safe_float(row.get("time")),
        "total_timesteps": _safe_float(row.get("total_timesteps")),
        "total_weight_updates": _safe_float(row.get("total_weight_updates")),
        "matched_value": matched_value,
    }
    if extra_fields:
        event_data.update(extra_fields)
    events[event_name] = event_data


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
            episode_lengths = _parse_number_list(row.get("episode_lengths"))

            if lap_times:
                _set_first_event(events, "first_lap", idx, row, matched_value=min(lap_times))

            for metric in METRIC_DEFINITIONS:
                if metric["kind"] == "lap" and lap_times:
                    threshold = float(metric["threshold"])
                    laps_under_threshold = [v for v in lap_times if v < threshold]
                    if laps_under_threshold:
                        _set_first_event(
                            events,
                            metric["event_name"],
                            idx,
                            row,
                            matched_value=min(laps_under_threshold),
                        )
                elif metric["kind"] == "episode_length" and episode_lengths:
                    threshold = float(metric["threshold"])
                    hits = [v for v in episode_lengths if int(v) >= int(threshold)]
                    if hits:
                        _set_first_event(
                            events,
                            metric["event_name"],
                            idx,
                            row,
                            matched_value=max(hits),
                        )
                elif metric["kind"] == "episode_length_and_reward" and episode_lengths and mean_step_rewards:
                    hit = _first_qualifying_episode_pair(
                        episode_lengths=episode_lengths,
                        mean_step_rewards=mean_step_rewards,
                        min_length=float(metric["length_threshold"]),
                        reward_threshold=float(metric["reward_threshold"]),
                    )
                    if hit:
                        _set_first_event(
                            events,
                            metric["event_name"],
                            idx,
                            row,
                            matched_value=hit["matched_value"],
                            extra_fields={
                                "matched_episode_length": hit["matched_episode_length"],
                                "matched_episode_mean_step_reward": hit["matched_episode_mean_step_reward"],
                            },
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


def apply_ranking_mode(output_path: Path, ranking_mode: str) -> Path:
    if ranking_mode == "all":
        return output_path
    return output_path.with_name(f"{output_path.stem}_{ranking_mode}{output_path.suffix}")


def build_reference_output_path(summary_output_path: Path) -> Path:
    return summary_output_path.with_name(
        summary_output_path.stem.replace("learning_metrics_summary", "learning_metric_references")
        + summary_output_path.suffix
    )


def build_leaderboard_csv_output_path(summary_output_path: Path) -> Path:
    return summary_output_path.with_name(
        summary_output_path.stem.replace("learning_metrics_summary", "learning_metric_leaderboards")
        + summary_output_path.suffix
    )


def build_leaderboard_txt_output_path(summary_output_path: Path) -> Path:
    return summary_output_path.with_name(
        summary_output_path.stem.replace("learning_metrics_summary", "learning_metric_leaderboards")
        + ".txt"
    )


def _collect_leaderboard_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    leaderboard_rows: List[Dict[str, object]] = []
    for metric in RANKED_METRICS:
        prefix = _abbreviate_column_name(metric["event_name"])
        rank_key = f"{prefix}_speed_rank_a0_5"
        score_key = f"{prefix}_speed_score_a0_5"
        timesteps_key = f"{prefix}_total_timesteps"
        updates_key = f"{prefix}_total_weight_updates"
        time_key = f"{prefix}_time"
        matched_value_key = f"{prefix}_matched_value"

        metric_rows: List[Dict[str, object]] = []
        for row in rows:
            rank = row.get(rank_key)
            score = row.get(score_key)
            if not isinstance(rank, (int, float)) or not isinstance(score, (int, float)):
                continue
            metric_rows.append(
                {
                    "metric_id": metric["event_name"],
                    "metric_label": metric.get("label", metric["event_name"]),
                    "name": row.get("name"),
                    "rank": int(rank),
                    "speed_score_a0_5": float(score),
                    "total_timesteps": row.get(timesteps_key),
                    "total_weight_updates": row.get(updates_key),
                    "time": row.get(time_key),
                    "matched_value": row.get(matched_value_key),
                }
            )

        metric_rows.sort(key=lambda entry: (entry["rank"], entry["name"]))
        leaderboard_rows.extend(metric_rows)

    return leaderboard_rows


def _write_leaderboard_txt(
    leaderboard_rows: List[Dict[str, object]],
    overall_rows: List[Dict[str, object]],
    output_path: Path,
    overall_top_n: int,
) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    labels: Dict[str, str] = {}
    for row in leaderboard_rows:
        metric_id = str(row["metric_id"])
        grouped.setdefault(metric_id, []).append(row)
        labels[metric_id] = str(row["metric_label"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"alpha={SCORE_ALPHA}\n")
        f.write("score=(T/Tref)^alpha * (U/Uref)^(1-alpha), lower is better\n\n")
        f.write("overall_score=geometric_mean(per-metric score), lower is better\n")
        f.write(
            f"missing_metric_penalty_factor={MISSING_METRIC_PENALTY_FACTOR} "
            "(applied as worse-than-worst achieved score)\n\n"
        )

        f.write("=== overall_ranking ===\n")
        if not overall_rows:
            f.write("no ranked models\n\n")
        else:
            ranked_rows = [r for r in overall_rows if r.get("position") is not None]
            top_rows = ranked_rows[: max(0, overall_top_n)]
            if not ranked_rows:
                f.write("no ranked models\n\n")
            else:
                for entry in ranked_rows:
                    f.write(
                        "#{pos:>3} {name} | overall_score={score:.6f} | rank_sum={points} | covered={covered}/{total} | total_weight_updates={updates}\n".format(
                            pos=entry.get("position"),
                            name=entry.get("name"),
                            score=float(entry.get("score")) if isinstance(entry.get("score"), (int, float)) else float("nan"),
                            points=entry.get("points"),
                            covered=entry.get("metrics_covered"),
                            total=entry.get("metrics_total"),
                            updates=entry.get("final_total_weight_updates"),
                        )
                    )

                f.write("\n")
                f.write("--- detailed_overall_top_models ---\n")
                if not top_rows:
                    f.write("(no models in top selection)\n")
                else:
                    for entry in top_rows:
                        f.write(
                            "\n#{pos:>3} {name}\noverall_score={score:.6f}, rank_sum={points}, covered={covered}/{total}, total_weight_updates={updates}\n".format(
                                pos=entry.get("position"),
                                name=entry.get("name"),
                                score=float(entry.get("score")) if isinstance(entry.get("score"), (int, float)) else float("nan"),
                                points=entry.get("points"),
                                covered=entry.get("metrics_covered"),
                                total=entry.get("metrics_total"),
                                updates=entry.get("final_total_weight_updates"),
                            )
                        )
                        for metric_result in entry.get("per_metric", []):
                            rank_value = metric_result.get("rank")
                            penalty_value = metric_result.get("penalty")
                            score_value = metric_result.get("score")
                            effective_score_value = metric_result.get("effective_score")
                            rank_text = str(rank_value) if rank_value is not None else f"NA(penalty={penalty_value})"
                            if score_value is not None:
                                score_text = f"{score_value}"
                            elif effective_score_value is not None:
                                score_text = f"NA(penalty={effective_score_value})"
                            else:
                                score_text = "NA"
                            f.write(
                                "  - {metric_id}: rank={rank_text}, score={score}, T={t}, U={u}, time={time}, matched={matched}\n".format(
                                    metric_id=metric_result.get("metric_id"),
                                    rank_text=rank_text,
                                    score=score_text,
                                    t=metric_result.get("total_timesteps"),
                                    u=metric_result.get("total_weight_updates"),
                                    time=metric_result.get("time"),
                                    matched=metric_result.get("matched_value"),
                                )
                            )

                f.write("\n")

        for metric in RANKED_METRICS:
            metric_id = metric["event_name"]
            label = labels.get(metric_id, metric.get("label", metric_id))
            f.write(f"=== {metric_id} ({label}) ===\n")
            rows_for_metric = grouped.get(metric_id, [])
            if not rows_for_metric:
                f.write("no ranked models\n\n")
                continue
            for entry in rows_for_metric:
                f.write(
                    "#{rank:>3} {name} | score={score:.6f} | T={t} | U={u} | time={time} | matched={matched}\n".format(
                        rank=entry["rank"],
                        name=entry.get("name"),
                        score=entry["speed_score_a0_5"],
                        t=entry.get("total_timesteps"),
                        u=entry.get("total_weight_updates"),
                        time=entry.get("time"),
                        matched=entry.get("matched_value"),
                    )
                )
            f.write("\n")


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
    parser.add_argument(
        "--overall-top-n",
        type=int,
        default=DEFAULT_OVERALL_TOP_N,
        help="How many models to show in detailed overall ranking section of TXT (default: 30)",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=sorted(RANKING_MODES.keys()),
        default="all",
        help="Which milestone set to rank in the overall leaderboard (default: all)",
    )

    args = parser.parse_args()
    if not args.name and not args.prefix:
        parser.error("Provide either --name or --prefix")

    global RANKED_METRICS
    RANKED_METRICS = RANKING_MODES[args.ranking_mode]

    models_dir = Path(args.models_dir)
    models = find_models(models_dir=models_dir, name=args.name, prefix=args.prefix)
    if not models:
        target = args.name if args.name else args.prefix
        print(f"No matching model directories found for: {target}")
        return

    output_path = apply_ranking_mode(build_output_path(args.out_file, args.name, args.prefix), args.ranking_mode)
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

    grouped_rows = average_same_model_rows(rows)

    _apply_metric_rankings(rows, alpha=SCORE_ALPHA)
    overall_rows = _compute_overall_rankings(rows)

    _apply_metric_rankings(grouped_rows, alpha=SCORE_ALPHA)
    grouped_overall_rows = _compute_overall_rankings(grouped_rows)

    base_output_path = output_path
    outputs = [
        ("", rows, overall_rows),
        ("_grouped", grouped_rows, grouped_overall_rows),
    ]

    for output_suffix, output_rows, output_overall_rows in outputs:
        current_output_path = base_output_path.with_name(
            base_output_path.stem + output_suffix + base_output_path.suffix
        )

        all_fields = sorted({key for row in output_rows for key in row.keys()})
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

        with open(current_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fields)
            writer.writeheader()
            for row in output_rows:
                writer.writerow(row)

        reference_rows = _collect_reference_rows(output_rows)
        reference_output_path = build_reference_output_path(current_output_path)
        with open(reference_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric_id", "metric_label", "sample_count", "Tref", "Uref"])
            writer.writeheader()
            for row in reference_rows:
                writer.writerow(row)

        leaderboard_rows = _collect_leaderboard_rows(output_rows)
        leaderboard_csv_output_path = build_leaderboard_csv_output_path(current_output_path)
        with open(leaderboard_csv_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "metric_id",
                    "metric_label",
                    "name",
                    "rank",
                    "speed_score_a0_5",
                    "total_timesteps",
                    "total_weight_updates",
                    "time",
                    "matched_value",
                ],
            )
            writer.writeheader()
            for row in leaderboard_rows:
                writer.writerow(row)

        leaderboard_txt_output_path = build_leaderboard_txt_output_path(current_output_path)
        _write_leaderboard_txt(
            leaderboard_rows=leaderboard_rows,
            overall_rows=output_overall_rows,
            output_path=leaderboard_txt_output_path,
            overall_top_n=max(0, int(args.overall_top_n)),
        )

        print(f"Processed models: {len(output_rows)}")
        print(f"Summary CSV: {current_output_path}")
        print(f"Metric references CSV: {reference_output_path}")
        print(f"Metric leaderboards CSV: {leaderboard_csv_output_path}")
        print(f"Metric leaderboards TXT: {leaderboard_txt_output_path}")

    


if __name__ == "__main__":
    main()
