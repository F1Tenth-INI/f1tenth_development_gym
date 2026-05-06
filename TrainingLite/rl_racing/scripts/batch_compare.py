#!/usr/bin/env python3
"""
Compare one artifact per model in a batch using canonical model zip only.

This script reuses the same simulation/evaluation logic as run_sweep_experiments.py,
and evaluates only the canonical artifact at:
models/<model_name>/<model_name>.zip
Then it ranks all models against each other.

Usage examples:
    python TrainingLite/rl_racing/scripts/batch_compare.py --model-name 0603_checkpoint_test
    python TrainingLite/rl_racing/scripts/batch_compare.py --model-prefix Sweep_rank_Ex1_A0.0 --repeats 3 --max-length 10000

    python -u TrainingLite/rl_racing/scripts/batch_compare.py --model-name 0603_checkpoint_test
    python -u TrainingLite/rl_racing/scripts/batch_compare.py --model-prefix Sweep_rank_Ex1_A0.0 --repeats 3 --max-length 10000

    FOR FINETUNE-ONLY COMPARISON (after trimming learning_metrics.csv with trim_learning_metrics_to_finetune.py):
    python TrainingLite/rl_racing/scripts/batch_compare.py --repeats 2 --max-length 2000 --map-name RCA2 --env-car-parameter-file gym_car_parameters_finetune.yml --model-prefix RCA2-Final
"""

import argparse
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add root dir to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_dir)



class CheckpointSelector:
    """Compare one canonical artifact per model across one or many models."""

    def __init__(
        self,
        max_length: int = 8000,
        repeats: int = 3,
        verbose: bool = False,
        results_dir: str = "batch_compare_models",
        best_suffix: str = "_best",
        no_overwrite_best: bool = False,
        keep_eval_models: bool = False,
    ):
        self.max_length = max_length
        self.repeats = repeats
        self.verbose = verbose
        self.best_suffix = best_suffix
        self.no_overwrite_best = no_overwrite_best
        self.keep_eval_models = keep_eval_models

        self.models_dir = Path(root_dir) / "TrainingLite" / "rl_racing" / "models"
        self.results_dir = Path(root_dir) / results_dir
        self.results_dir.mkdir(exist_ok=True)

        # Import lazily so Settings overrides can be applied before loading runtime modules.
        from TrainingLite.rl_racing.scripts.run_sweep_experiments import SweepExperimentRunner

        # Prefix is unused because we only call run_experiment_on_model directly.
        self.runner = SweepExperimentRunner(prefix="batch_compare", max_length=max_length, verbose=verbose)

    def find_models(self, model_name: Optional[str], model_prefix: Optional[str]) -> List[str]:
        """Resolve target model list from name or prefix."""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        if model_name:
            model_dir = self.models_dir / model_name
            if not model_dir.is_dir():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            return [model_name]

        if not model_prefix:
            raise ValueError("Either model_name or model_prefix must be provided")

        models = []
        for item in sorted(self.models_dir.iterdir()):
            if item.is_dir() and item.name.startswith(model_prefix):
                models.append(item.name)
        return models

    @staticmethod
    def _checkpoint_step(checkpoint_name: str) -> int:
        """Extract timestep from a checkpoint filename like *_ckpt_50000 or *_ckpt_50000.zip."""
        match = re.search(r"_ckpt_(\d+)(?:\.zip)?$", checkpoint_name)
        if match:
            return int(match.group(1))
        return -1

    @staticmethod
    def _checkpoint_candidate_name(checkpoint_path: Path) -> str:
        """Return a stable candidate name for checkpoint artifacts."""
        name = checkpoint_path.name
        if name.lower().endswith(".zip"):
            return name[:-4]
        return name

    def find_eval_model_path(self, model_name: str) -> Optional[Path]:
        """Resolve the evaluation artifact without modifying model files.

        Prefer a promoted best model artifact when present, then fall back to the
        canonical model zip.
        """
        model_dir = self.models_dir / model_name
        best_zip = model_dir / f"{model_name}{self.best_suffix}.zip"
        if best_zip.is_file():
            return best_zip

        model_zip = model_dir / f"{model_name}.zip"
        if model_zip.is_file():
            return model_zip
        print(
            f"Evaluation artifact missing for {model_name}: {best_zip} or {model_zip}. "
            "Skipping to avoid any file mutation."
        )
        return None

    @staticmethod
    def _group_model_name(model_name: str) -> str:
        """Strip a trailing numeric run index so repeated seeds collapse into one group."""
        match = re.match(r"^(.*)_\d+$", model_name)
        return match.group(1) if match else model_name

    @staticmethod
    def _csv_value(value: Optional[str]) -> object:
        """Convert CSV text back into a lightweight Python value when possible."""
        if value is None:
            return None

        text = str(value).strip()
        if text == "" or text.lower() == "none":
            return None

        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False

        try:
            integer_value = int(text)
            return integer_value
        except ValueError:
            pass

        try:
            float_value = float(text)
            return float_value
        except ValueError:
            return text

    def _load_saved_model_rows(self) -> List[Dict[str, object]]:
        """Read per-model CSV outputs written by batch compare."""
        rows: List[Dict[str, object]] = []

        for csv_path in sorted(self.results_dir.glob("batch_compare_*.csv")):
            if csv_path.name.startswith("batch_compare_overall_summary_"):
                continue
            if csv_path.name.startswith("batch_compare_batch_best_"):
                continue

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    converted = {key: self._csv_value(value) for key, value in row.items()}
                    converted["csv_path"] = str(csv_path)
                    rows.append(converted)

        return rows

    def _load_saved_model_summaries(self) -> List[Dict[str, object]]:
        """Rebuild per-model summary rows from saved batch_compare CSV outputs."""
        summaries: List[Dict[str, object]] = []

        for csv_path in sorted(self.results_dir.glob("batch_compare_*.csv")):
            if csv_path.name.startswith("batch_compare_overall_summary_"):
                continue
            if csv_path.name.startswith("batch_compare_batch_best_"):
                continue

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [{key: self._csv_value(value) for key, value in row.items()} for row in reader]

            if not rows:
                continue

            best_row = next((row for row in rows if row.get("is_best") is True), None)
            if best_row is None:
                best_row = max(rows, key=self._ranking_key)

            model_name = str(best_row.get("model_name") or csv_path.stem.replace("batch_compare_", ""))
            summaries.append(
                {
                    "model_name": model_name,
                    "best_candidate_name": best_row.get("candidate_name", ""),
                    "best_checkpoint_path": best_row.get("candidate_path", ""),
                    "best_model_name": model_name,
                    "best_model_path": best_row.get("candidate_path", ""),
                    "ranking_mode": "avg-lap-time,fastest-lap-time",
                    "report_csv": str(csv_path),
                    "report_txt": str(csv_path.with_suffix(".txt")),
                    "evaluated_rows": rows,
                }
            )

        summaries.sort(key=lambda row: self._ranking_key_avg_lap(next((candidate for candidate in row.get("evaluated_rows", []) if candidate.get("is_best") is True), row.get("evaluated_rows", [{}])[0] if row.get("evaluated_rows") else {})))
        return summaries

    def _average_group_rows(self, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Average rows that only differ by trailing index suffix in the model name."""
        grouped_rows: Dict[str, List[Dict[str, object]]] = {}
        for row in rows:
            model_name = str(row.get("model_name", ""))
            group_name = self._group_model_name(model_name)
            grouped_rows.setdefault(group_name, []).append(row)

        averaged_rows: List[Dict[str, object]] = []
        for group_name, entries in grouped_rows.items():
            averaged: Dict[str, object] = {
                "model_name": group_name,
                "group_count": len(entries),
                "member_models": ", ".join(sorted(str(entry.get("model_name", "")) for entry in entries)),
            }

            all_keys = {key for entry in entries for key in entry.keys()}
            for key in all_keys:
                if key in {"model_name", "group_count", "member_models"}:
                    continue

                values = [entry.get(key) for entry in entries]
                numeric_values = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
                if numeric_values:
                    if key == "fastest_lap_time":
                        averaged[key] = min(numeric_values)
                        continue
                    averaged[key] = sum(numeric_values) / len(numeric_values)
                    continue

                non_null_values = [value for value in values if value not in (None, "")]
                if non_null_values and all(value == non_null_values[0] for value in non_null_values):
                    averaged[key] = non_null_values[0]
                else:
                    averaged[key] = None

            averaged_rows.append(averaged)

        averaged_rows.sort(key=lambda row: self._ranking_key_avg_lap(row), reverse=True)
        return averaged_rows

    def _write_grouped_summary_from_csvs(self, batch_tag: str) -> Optional[Path]:
        """Create a grouped txt summary by re-reading saved per-model CSV outputs."""
        saved_rows = self._load_saved_model_rows()
        if not saved_rows:
            return None

        grouped_rows = self._average_group_rows(saved_rows)
        grouped_txt = self.results_dir / f"batch_compare_grouped_summary_{batch_tag}.txt"

        with open(grouped_txt, "w", encoding="utf-8") as f:
            f.write("GROUPED BATCH COMPARISON SUMMARY (FROM SAVED CSV FILES)\n")
            f.write("=" * 80 + "\n")
            f.write(f"batch_tag: {batch_tag}\n")
            f.write(f"source_csv_files: {len(saved_rows)} rows across saved batch_compare_*.csv files\n")
            f.write("grouping_rule: strip trailing _<index> suffix from model_name\n")
            f.write("ranking_mode: avg-lap-time + fastest-lap-time\n\n")

            if not grouped_rows:
                f.write("No grouped rows found.\n")
                return grouped_txt

            for rank, row in enumerate(grouped_rows, 1):
                avg_lap_time = row.get("avg_lap_time")
                fastest_lap_time = row.get("fastest_lap_time")
                f.write(f"{rank}. {row.get('model_name')}\n")
                f.write(f"  Group Count: {int(row.get('group_count', 0))}\n")
                f.write(f"  Member Models: {row.get('member_models')}\n")
                if row.get("avg_lap_time") is not None:
                    f.write(f"  Avg Lap Time: {float(avg_lap_time):.4f}s\n")
                if row.get("fastest_lap_time") is not None:
                    f.write(f"  Fastest Lap Time: {float(fastest_lap_time):.4f}s\n")
                if row.get("lap_completion_rate") is not None:
                    f.write(f"  Lap Completion Rate: {float(row.get('lap_completion_rate')):.3f}\n")
                if row.get("avg_laps_completed") is not None:
                    f.write(f"  Avg Laps Completed: {float(row.get('avg_laps_completed')):.3f}\n")
                if row.get("avg_speed") is not None:
                    f.write(f"  Avg Speed: {float(row.get('avg_speed')):.4f}\n")
                if row.get("avg_reward") is not None:
                    f.write(f"  Avg Reward: {float(row.get('avg_reward')):.4f}\n")
                if row.get("avg_num_crashes") is not None:
                    f.write(f"  Avg Num Crashes: {float(row.get('avg_num_crashes')):.3f}\n")
                f.write("\n")

        return grouped_txt

    def _write_summaries_from_saved_csvs(self, batch_tag: str) -> Optional[Path]:
        """Rebuild overall summary artifacts from existing per-model CSV outputs."""
        all_summaries = self._load_saved_model_summaries()
        if not all_summaries:
            print(f"No saved batch_compare CSV files found in {self.results_dir}")
            return None

        overall_csv = self.results_dir / f"batch_compare_overall_summary_{batch_tag}.csv"
        with open(overall_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "model_name",
                "best_candidate_name",
                "best_checkpoint_path",
                "best_model_name",
                "best_model_path",
                "ranking_mode",
                "report_csv",
                "report_txt",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_summaries:
                out_row = {k: row.get(k, "") for k in fieldnames}
                out_row["ranking_mode"] = "avg-lap-time,fastest-lap-time"
                writer.writerow(out_row)

        global_candidates: List[Dict] = []
        for summary in all_summaries:
            for row in summary.get("evaluated_rows", []):
                global_candidates.append(dict(row))

        if global_candidates:
            global_best_avg = max(global_candidates, key=self._ranking_key_avg_lap)
            global_best_fastest = max(global_candidates, key=self._ranking_key_fastest_lap)
            top_candidates_avg = sorted(global_candidates, key=self._ranking_key_avg_lap, reverse=True)
            top_candidates_fastest = sorted(global_candidates, key=self._ranking_key_fastest_lap, reverse=True)
            global_best_model = str(global_best_avg["model_name"])
            global_best_source = Path(global_best_avg["candidate_path"])
            global_txt = self.results_dir / f"batch_compare_batch_best_{batch_tag}.txt"
            global_best_target = (
                self.models_dir
                / global_best_model
                / f"{global_best_model}{self.best_suffix}_batch_{batch_tag}.zip"
            )

            with open(global_txt, "w", encoding="utf-8") as f:
                f.write("BATCH BEST CANDIDATE (FROM SAVED CSV FILES)\n")
                f.write("=" * 80 + "\n")
                f.write(f"batch_tag: {batch_tag}\n")
                f.write("ranking_mode_selected: avg-lap-time + fastest-lap-time\n")

                f.write("\nBEST CANDIDATE BY avg-lap-time\n")
                f.write("-" * 80 + "\n")
                for key in [
                    "model_name",
                    "candidate_name",
                    "candidate_path",
                    "num_runs",
                    "num_completed_runs",
                    "num_lap_runs",
                    "lap_completion_rate",
                    "avg_lap_time",
                    "fastest_lap_time",
                    "avg_speed",
                    "avg_reward",
                    "avg_num_crashes",
                ]:
                    f.write(f"{key}: {global_best_avg.get(key)}\n")
                f.write(f"promoted_global_artifact: {global_best_target}\n\n")

                f.write("BEST CANDIDATE BY fastest-lap-time\n")
                f.write("-" * 80 + "\n")
                for key in [
                    "model_name",
                    "candidate_name",
                    "candidate_path",
                    "num_runs",
                    "num_completed_runs",
                    "num_lap_runs",
                    "lap_completion_rate",
                    "avg_lap_time",
                    "fastest_lap_time",
                    "avg_speed",
                    "avg_reward",
                    "avg_num_crashes",
                ]:
                    f.write(f"{key}: {global_best_fastest.get(key)}\n")
                f.write(f"promoted_global_artifact: {global_best_target}\n\n")

                f.write("COMPACT RANKING (BY avg-lap-time)\n")
                f.write("-" * 80 + "\n")
                for rank, candidate in enumerate(top_candidates_avg, 1):
                    f.write(
                        f"{rank}) name: \"{candidate.get('model_name')}\" | "
                        f"avg laptime: \"{self._format_laptime(candidate.get('avg_lap_time'))}\"\n"
                    )
                f.write("\n")

                f.write("TOP 30 CANDIDATES (RANKED BY avg-lap-time)\n")
                f.write("-" * 80 + "\n")
                for rank, candidate in enumerate(top_candidates_avg, 1):
                    f.write(f"rank: {rank}\n")
                    for key in [
                        "model_name",
                        "candidate_name",
                        "candidate_type",
                        "candidate_path",
                        "num_runs",
                        "num_completed_runs",
                        "num_lap_runs",
                        "lap_completion_rate",
                        "avg_lap_time",
                        "fastest_lap_time",
                        "avg_speed",
                        "avg_reward",
                        "avg_num_crashes",
                    ]:
                        f.write(f"{key}: {candidate.get(key)}\n")
                    f.write("\n")

                f.write("COMPACT RANKING (BY fastest-lap-time)\n")
                f.write("-" * 80 + "\n")
                for rank, candidate in enumerate(top_candidates_fastest, 1):
                    f.write(
                        f"{rank}) name: \"{candidate.get('model_name')}\" | "
                        f"fastest laptime: \"{self._format_laptime(candidate.get('fastest_lap_time'))}\"\n"
                    )
                f.write("\n")

                f.write("TOP 30 CANDIDATES (RANKED BY fastest-lap-time)\n")
                f.write("-" * 80 + "\n")
                for rank, candidate in enumerate(top_candidates_fastest, 1):
                    f.write(f"rank: {rank}\n")
                    for key in [
                        "model_name",
                        "candidate_name",
                        "candidate_type",
                        "candidate_path",
                        "num_runs",
                        "num_completed_runs",
                        "num_lap_runs",
                        "lap_completion_rate",
                        "avg_lap_time",
                        "fastest_lap_time",
                        "avg_speed",
                        "avg_reward",
                        "avg_num_crashes",
                    ]:
                        f.write(f"{key}: {candidate.get(key)}\n")
                    f.write("\n")

            print(f"Global best candidate (avg-lap-time): {global_best_avg['candidate_name']}")
            print(f"Global best candidate (fastest-lap-time): {global_best_fastest['candidate_name']}")
            print(f"Global promoted artifact: {global_best_target}")
            print(f"Global summary: {global_txt}")

        grouped_txt = self._write_grouped_summary_from_csvs(batch_tag)
        if grouped_txt is not None:
            print(f"Grouped summary saved to: {grouped_txt}")

        print(f"\nOverall summary saved to: {overall_csv}")
        return overall_csv

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        filtered = [float(v) for v in values if v is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    def _aggregate_candidate_runs(
        self,
        model_name: str,
        candidate_name: str,
        candidate_type: str,
        candidate_path: Path,
        runs: List[Dict],
    ) -> Dict:
        """Aggregate repeated inference runs for one evaluated candidate."""
        completed = [r for r in runs if r.get("status") == "completed"]
        lap_runs = [r for r in completed if (r.get("num_laps_completed") or 0) > 0]

        avg_lap_time = self._mean([r.get("avg_lap_time") for r in lap_runs])
        fastest_lap_time = min(
            [r.get("min_lap_time") for r in lap_runs if r.get("min_lap_time") is not None]
            or [r.get("avg_lap_time") for r in lap_runs if r.get("avg_lap_time") is not None],
            default=None,
        )
        avg_speed = self._mean([r.get("avg_speed") for r in completed])
        avg_laps_completed = self._mean([r.get("num_laps_completed") for r in completed])
        avg_rewards = self._mean([r.get("total_reward") for r in completed if "total_reward" in r])
        avg_crashes = self._mean([r.get("num_crashes") for r in completed])

        success_rate = len(completed) / len(runs) if runs else 0.0
        lap_completion_rate = len(lap_runs) / len(runs) if runs else 0.0

        return {
            "model_name": model_name,
            "candidate_name": candidate_name,
            "candidate_type": candidate_type,
            "candidate_path": str(candidate_path),
            "num_runs": len(runs),
            "num_completed_runs": len(completed),
            "num_lap_runs": len(lap_runs),
            "success_rate": success_rate,
            "lap_completion_rate": lap_completion_rate,
            "avg_laps_completed": avg_laps_completed,
            "avg_lap_time": avg_lap_time,
            "fastest_lap_time": fastest_lap_time,
            "avg_speed": avg_speed,
            "avg_reward": avg_rewards,
            "avg_num_crashes": avg_crashes,
        }

    @staticmethod
    def _ranking_key_avg_lap(aggregated: Dict) -> Tuple:
        """
        Sort key where larger tuple is better.

        Priority:
        1) Runs that completed and produced at least one lap.
        2) Lower average lap time.
        3) Higher lap completion rate.
        4) Higher average speed.
        5) Fewer crashes.
        """
        avg_lap_time = aggregated.get("avg_lap_time")
        avg_speed = aggregated.get("avg_speed")
        avg_num_crashes = aggregated.get("avg_num_crashes")

        return (
            aggregated.get("num_lap_runs", 0) > 0,
            -avg_lap_time if avg_lap_time is not None else float("-inf"),
            aggregated.get("lap_completion_rate", 0.0),
            avg_speed if avg_speed is not None else float("-inf"),
            -(avg_num_crashes if avg_num_crashes is not None else float("inf")),
        )

    @staticmethod
    def _ranking_key_fastest_lap(aggregated: Dict) -> Tuple:
        """
        Sort key where larger tuple is better.

        Priority:
        1) Runs that completed and produced at least one lap.
        2) Lower fastest lap time.
        3) Higher lap completion rate.
        4) Lower average lap time.
        5) Higher average speed.
        6) Fewer crashes.
        """
        fastest_lap_time = aggregated.get("fastest_lap_time")
        avg_lap_time = aggregated.get("avg_lap_time")
        avg_speed = aggregated.get("avg_speed")
        avg_num_crashes = aggregated.get("avg_num_crashes")

        return (
            aggregated.get("num_lap_runs", 0) > 0,
            -fastest_lap_time if fastest_lap_time is not None else float("-inf"),
            aggregated.get("lap_completion_rate", 0.0),
            -avg_lap_time if avg_lap_time is not None else float("-inf"),
            avg_speed if avg_speed is not None else float("-inf"),
            -(avg_num_crashes if avg_num_crashes is not None else float("inf")),
        )

    def _ranking_key(self, aggregated: Dict) -> Tuple:
        """Return ranking key prioritizing lowest average lap time."""
        return self._ranking_key_avg_lap(aggregated)

    @staticmethod
    def _format_laptime(value: Optional[float]) -> str:
        """Render lap time consistently for compact ranking lines."""
        if value is None:
            return "N/A"
        return f"{float(value):.4f}s"

    def _write_model_results(self, model_name: str, rows: List[Dict], best_row: Dict, sweep_name: str) -> Tuple[Path, Path]:
        """Write per-model comparison CSV and text summary."""
        csv_path = self.results_dir / f"batch_compare_{model_name}.csv"
        txt_path = self.results_dir / f"batch_compare_{model_name}.txt"

        headers = [
            "model_name",
            "candidate_name",
            "candidate_type",
            "candidate_path",
            "num_runs",
            "num_completed_runs",
            "num_lap_runs",
            "success_rate",
            "lap_completion_rate",
            "avg_laps_completed",
            "avg_lap_time",
            "fastest_lap_time",
            "avg_speed",
            "avg_reward",
            "avg_num_crashes",
            "is_best",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["is_best"] = row["candidate_name"] == best_row["candidate_name"]
                writer.writerow(out)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Batch model compare for model: {model_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Evaluated candidates: {len(rows)}\n")
            f.write(f"Runs per candidate: {self.repeats}\n")
            f.write(f"Max simulation length: {self.max_length}\n\n")

            f.write("BEST CANDIDATE\n")
            f.write("-" * 80 + "\n")
            for key in [
                "candidate_name",
                "candidate_type",
                "candidate_path",
                "num_runs",
                "num_completed_runs",
                "num_lap_runs",
                "lap_completion_rate",
                "avg_laps_completed",
                "avg_lap_time",
                "fastest_lap_time",
                "avg_speed",
                "avg_reward",
                "avg_num_crashes",
            ]:
                f.write(f"{key}: {best_row.get(key)}\n")

        return csv_path, txt_path

    def evaluate_model(self, model_name: str, model_prefix: str) -> Optional[Dict]:
        """Evaluate the selected artifact for a model and return summary."""
        eval_model_path = self.find_eval_model_path(model_name)
        if not eval_model_path:
            print(f"Evaluation artifact not found for model: {model_name}")
            return None

        candidates: List[Tuple[str, str, Path]] = [(model_name, "final", eval_model_path)]

        if not candidates:
            print(f"No candidates to evaluate for model: {model_name}")
            return None

        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"Canonical model present: {'yes' if eval_model_path is not None else 'no'}")
        print(f"{'=' * 80}")

        aggregated_rows: List[Dict] = []
        all_run_durations: List[float] = []
        eval_model_name = model_name

        for idx, (candidate_name, candidate_type, candidate_path) in enumerate(candidates, 1):
            print(f"[{idx}/{len(candidates)}] Evaluating {candidate_name} ({candidate_type})")

            runs: List[Dict] = []
            for run_idx in range(self.repeats):
                print(f"  run {run_idx + 1}/{self.repeats} ", end="", flush=True)
                run_start = time.perf_counter()
                result = self.runner.run_experiment_on_model(eval_model_name)
                run_duration = time.perf_counter() - run_start
                all_run_durations.append(run_duration)
                print(f" [{run_duration:.2f}s]", end="", flush=True)
                runs.append(result)
            print()

            aggregated = self._aggregate_candidate_runs(
                model_name=model_name,
                candidate_name=candidate_name,
                candidate_type=candidate_type,
                candidate_path=candidate_path,
                runs=runs,
            )
            aggregated_rows.append(aggregated)

            lap_time_text = f"{aggregated['avg_lap_time']:.4f}s" if aggregated["avg_lap_time"] is not None else "N/A"
            fastest_lap_text = (
                f"{aggregated['fastest_lap_time']:.4f}s" if aggregated["fastest_lap_time"] is not None else "N/A"
            )
            print(
                f"  -> completed_runs={aggregated['num_completed_runs']}/{aggregated['num_runs']}, "
                f"lap_runs={aggregated['num_lap_runs']}/{aggregated['num_runs']}, "
                f"avg_lap_time={lap_time_text}, fastest_lap={fastest_lap_text}"
            )

        if all_run_durations:
            avg_run_time = sum(all_run_durations) / len(all_run_durations)
            print(
                f"Timing summary ({model_name}): runs={len(all_run_durations)}, "
                f"avg={avg_run_time:.2f}s, min={min(all_run_durations):.2f}s, max={max(all_run_durations):.2f}s"
            )

            trend_window = min(10, len(all_run_durations) // 2)
            if trend_window >= 1:
                first_avg = sum(all_run_durations[:trend_window]) / trend_window
                last_avg = sum(all_run_durations[-trend_window:]) / trend_window
                delta = last_avg - first_avg
                print(
                    f"Timing trend ({model_name}): first_{trend_window}_avg={first_avg:.2f}s, "
                    f"last_{trend_window}_avg={last_avg:.2f}s, delta={delta:+.2f}s"
                )

        best_row = max(aggregated_rows, key=lambda row: self._ranking_key(row))
        best_checkpoint_path = Path(best_row["candidate_path"])
        best_model_path = best_checkpoint_path
        csv_path, txt_path = self._write_model_results(model_name, aggregated_rows, best_row, sweep_name = model_prefix if model_prefix else model_name)

        # print(f"\nBest candidate for {model_name}: {best_row['candidate_name']} ({best_row['candidate_type']})")
        print(f"Selected artifact: {best_model_path}")
        print(f"CSV report: {csv_path}")
        print(f"Text report: {txt_path}")

        return {
            "model_name": model_name,
            "best_candidate_name": best_row["candidate_name"],
            "best_checkpoint_path": str(best_checkpoint_path),
            "best_model_name": model_name,
            "best_model_path": str(best_model_path),
            "report_csv": str(csv_path),
            "report_txt": str(txt_path),
            "evaluated_rows": aggregated_rows,
        }

    def run(self, model_name: Optional[str], model_prefix: Optional[str], summarize_only: bool = False) -> None:
        """Execute batch model comparison for the requested models."""
        from utilities.Settings import Settings

        # This workload repeatedly constructs CarSystem/WaypointUtils during evaluation.
        # Disable background waypoint reload threads for this script run to avoid
        # thread creation overhead accumulating over long checkpoint batches.
        previous_reload_setting = Settings.RELOAD_WP_IN_BACKGROUND
        Settings.RELOAD_WP_IN_BACKGROUND = False

        try:
            if model_prefix:
                batch_tag = re.sub(r"[^A-Za-z0-9_\-]", "_", model_prefix)
            elif model_name:
                batch_tag = re.sub(r"[^A-Za-z0-9_\-]", "_", model_name)
            else:
                batch_tag = "selection"

            run_results_dir = self.results_dir / batch_tag
            run_results_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir = run_results_dir

            if summarize_only:
                print(f"Summary-only mode enabled; rebuilding artifacts from: {self.results_dir}")
                self._write_summaries_from_saved_csvs(batch_tag)
                return

            models = self.find_models(model_name=model_name, model_prefix=model_prefix)
            if not models:
                target = model_name if model_name else model_prefix
                print(f"No matching model directories found for: {target}")
                return

            print(f"Target models: {len(models)}")
            print(f"Batch tag: {batch_tag}")
            print(f"Results directory: {self.results_dir}")
            print("Ranking mode: avg-lap-time + fastest-lap-time")
            print("Waypoint background reload: disabled for this run")
            all_summaries: List[Dict] = []

            for model in models:
                summary = self.evaluate_model(model, model_prefix)
                if summary is not None:
                    all_summaries.append(summary)

            if not all_summaries:
                print("No models were processed successfully.")
                return

            overall_csv = self.results_dir / f"batch_compare_overall_summary_{batch_tag}.csv"
            with open(overall_csv, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "model_name",
                    "best_candidate_name",
                    "best_checkpoint_path",
                    "best_model_name",
                    "best_model_path",
                    "ranking_mode",
                    "report_csv",
                    "report_txt",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_summaries:
                    out_row = {k: row.get(k, "") for k in fieldnames}
                    out_row["ranking_mode"] = "avg-lap-time,fastest-lap-time"
                    writer.writerow(out_row)

            # Global best across all evaluated model candidates (best-or-final per model).
            global_candidates: List[Dict] = []
            for summary in all_summaries:
                for row in summary.get("evaluated_rows", []):
                    global_candidates.append(dict(row))

            if global_candidates:
                global_best_avg = max(global_candidates, key=self._ranking_key_avg_lap)
                global_best_fastest = max(global_candidates, key=self._ranking_key_fastest_lap)
                top_candidates_avg = sorted(global_candidates, key=self._ranking_key_avg_lap, reverse=True)
                top_candidates_fastest = sorted(global_candidates, key=self._ranking_key_fastest_lap, reverse=True)
                global_best_model = str(global_best_avg["model_name"])
                global_best_source = Path(global_best_avg["candidate_path"])
                global_txt = self.results_dir / f"batch_compare_batch_best_{batch_tag}.txt"
                global_best_target = (
                    self.models_dir
                    / global_best_model
                    / f"{global_best_model}{self.best_suffix}_batch_{batch_tag}.zip"
                )

                if global_best_target.exists() and self.no_overwrite_best:
                    print(f"[INFO] Skipping global best promotion (already exists): {global_best_target.name}")
                else:
                    shutil.copy2(global_best_source, global_best_target)

                with open(global_txt, "w", encoding="utf-8") as f:
                    f.write("BATCH BEST CANDIDATE (THIS SCRIPT RUN ONLY)\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"batch_tag: {batch_tag}\n")
                    f.write("ranking_mode_selected: avg-lap-time + fastest-lap-time\n")

                    f.write("\nBEST CANDIDATE BY avg-lap-time\n")
                    f.write("-" * 80 + "\n")
                    for key in [
                        "model_name",
                        "candidate_name",

                        "candidate_path",
                        "num_runs",
                        "num_completed_runs",
                        "num_lap_runs",
                        "lap_completion_rate",
                        "avg_laps_completed",
                        "avg_lap_time",
                        "fastest_lap_time",
                        "avg_speed",
                        "avg_reward",
                        "avg_num_crashes",
                    ]:
                        f.write(f"{key}: {global_best_avg.get(key)}\n")
                    f.write(f"promoted_global_artifact: {global_best_target}\n\n")

                    f.write("BEST CANDIDATE BY fastest-lap-time\n")
                    f.write("-" * 80 + "\n")
                    for key in [
                        "model_name",
                        "candidate_name",
                        "candidate_path",
                        "num_runs",
                        "num_completed_runs",
                        "num_lap_runs",
                        "lap_completion_rate",
                        "avg_laps_completed",
                        "avg_lap_time",
                        "fastest_lap_time",
                        "avg_speed",
                        "avg_reward",
                        "avg_num_crashes",
                    ]:
                        f.write(f"{key}: {global_best_fastest.get(key)}\n")
                    f.write(f"promoted_global_artifact: {global_best_target}\n\n")

                    f.write("COMPACT RANKING (BY avg-lap-time)\n")
                    f.write("-" * 80 + "\n")
                    for rank, candidate in enumerate(top_candidates_avg, 1):
                        f.write(
                            f"{rank}) name: \"{candidate.get('model_name')}\" | "
                            f"avg laptime: \"{self._format_laptime(candidate.get('avg_lap_time'))}\"\n"
                        )
                    f.write("\n")

                    f.write("TOP 30 CANDIDATES (RANKED BY avg-lap-time)\n")
                    f.write("-" * 80 + "\n")
                    for rank, candidate in enumerate(top_candidates_avg, 1):
                        f.write(f"rank: {rank}\n")
                        for key in [
                            "model_name",
                            "candidate_name",
                            "candidate_type",
                            "candidate_path",
                            "num_runs",
                            "num_completed_runs",
                            "num_lap_runs",
                            "lap_completion_rate",
                            "avg_laps_completed",
                            "avg_lap_time",
                            "fastest_lap_time",
                            "avg_speed",
                            "avg_reward",
                            "avg_num_crashes",
                        ]:
                            f.write(f"{key}: {candidate.get(key)}\n")
                        f.write("\n")

                    f.write("COMPACT RANKING (BY fastest-lap-time)\n")
                    f.write("-" * 80 + "\n")
                    for rank, candidate in enumerate(top_candidates_fastest, 1):
                        f.write(
                            f"{rank}) name: \"{candidate.get('model_name')}\" | "
                            f"fastest laptime: \"{self._format_laptime(candidate.get('fastest_lap_time'))}\"\n"
                        )
                    f.write("\n")

                    f.write("TOP 30 CANDIDATES (RANKED BY fastest-lap-time)\n")
                    f.write("-" * 80 + "\n")
                    for rank, candidate in enumerate(top_candidates_fastest, 1):
                        f.write(f"rank: {rank}\n")
                        for key in [
                            "model_name",
                            "candidate_name",
                            "candidate_type",
                            "candidate_path",
                            "num_runs",
                            "num_completed_runs",
                            "num_lap_runs",
                            "lap_completion_rate",
                            "avg_laps_completed",
                            "avg_lap_time",
                            "fastest_lap_time",
                            "avg_speed",
                            "avg_reward",
                            "avg_num_crashes",
                        ]:
                            f.write(f"{key}: {candidate.get(key)}\n")
                        f.write("\n")

                print(
                    f"\nGlobal best candidate (avg-lap-time): {global_best_avg['candidate_name']} "
                )
                print(f"Global best candidate (fastest-lap-time): {global_best_fastest['candidate_name']}")
                print(f"Global promoted artifact: {global_best_target}")
                print(f"Global summary: {global_txt}")

            grouped_txt = self._write_grouped_summary_from_csvs(batch_tag)
            if grouped_txt is not None:
                print(f"Grouped summary saved to: {grouped_txt}")

            print(f"\nOverall summary saved to: {overall_csv}")
        finally:
            Settings.RELOAD_WP_IN_BACKGROUND = previous_reload_setting


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare best/final model artifacts across a batch")
    parser.add_argument("--model-name", help="Exact model directory name to process")
    parser.add_argument("--model-prefix", help="Process all model directories with this prefix")
    parser.add_argument(
        "--MAP_NAME",
        "--map-name",
        dest="MAP_NAME",
        default=None,
        help="Override Settings.MAP_NAME for evaluation runs",
    )
    parser.add_argument(
        "--ENV_CAR_PARAMETER_FILE",
        "--env-car-parameter-file",
        dest="ENV_CAR_PARAMETER_FILE",
        default=None,
        help="Override Settings.ENV_CAR_PARAMETER_FILE for evaluation runs",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Maximum simulation length per inference run (default: 8000)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated runs per checkpoint (default: 3)",
    )
    parser.add_argument(
        "--results-dir",
        default="batch_compare_models",
        help="Directory for batch compare outputs (default: batch_compare_models)",
    )
    parser.add_argument(
        "--best-suffix",
        default="_best",
        help="Suffix for promoted best artifact filename inside each model folder (default: _best)",
    )
    parser.add_argument(
        "--no-overwrite-best",
        action="store_true",
        help="Skip promotion if best artifact already exists (default: overwrite)",
    )
    parser.add_argument(
        "--keep-eval-models",
        action="store_true",
        help="Keep temporary evaluation model folders for debugging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-run output",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild overall/grouped/global summaries from existing batch_compare CSVs without rerunning experiments",
    )

    args, settings_passthrough = parser.parse_known_args()

    if not args.model_name and not args.model_prefix:
        parser.error("Provide either --model-name or --model-prefix")

    settings_args: List[str] = list(settings_passthrough)
    if args.MAP_NAME is not None:
        settings_args.extend(["--MAP_NAME", args.MAP_NAME])
    if args.ENV_CAR_PARAMETER_FILE is not None:
        settings_args.extend(["--ENV_CAR_PARAMETER_FILE", args.ENV_CAR_PARAMETER_FILE])

    if settings_args:
        from utilities.parser_utilities import parse_settings_args

        original_argv = sys.argv
        try:
            sys.argv = ["batch_compare.py"] + settings_args
            parse_settings_args(
                description="Applying batch_compare Settings overrides",
                save_snapshot=False,
                verbose=True,
            )
        finally:
            sys.argv = original_argv

    selector = CheckpointSelector(
        max_length=args.max_length,
        repeats=args.repeats,
        verbose=args.verbose,
        results_dir=args.results_dir,
        best_suffix=args.best_suffix,
        no_overwrite_best=args.no_overwrite_best,
        keep_eval_models=args.keep_eval_models,
    )
    selector.run(model_name=args.model_name, model_prefix=args.model_prefix, summarize_only=args.summarize_only)


if __name__ == "__main__":
    main()
