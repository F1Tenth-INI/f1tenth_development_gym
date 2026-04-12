#!/usr/bin/env python3
"""
Compare one artifact per model in a batch: prefer *_best.zip, fallback to final *.zip.

This script reuses the same simulation/evaluation logic as run_sweep_experiments.py,
but evaluates only the promoted best artifact per model (or final model fallback),
then ranks all models against each other.

Usage examples:
    python TrainingLite/rl_racing/scripts/batch_compare.py --model-name 0603_checkpoint_test
    python TrainingLite/rl_racing/scripts/batch_compare.py --model-prefix Sweep_rank_Ex1_A0.0 --repeats 3 --max-length 10000

    python -u TrainingLite/rl_racing/scripts/batch_compare.py --model-name 0603_checkpoint_test
    python -u TrainingLite/rl_racing/scripts/batch_compare.py --model-prefix Sweep_rank_Ex1_A0.0 --repeats 3 --max-length 10000
"""

import argparse
import csv
import hashlib
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

from TrainingLite.rl_racing.scripts.run_sweep_experiments import SweepExperimentRunner
from utilities.Settings import Settings


class CheckpointSelector:
    """Compare one selected artifact per model across one or many models."""

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

    def find_best_model(self, model_name: str) -> Optional[Path]:
        """Resolve a single candidate artifact: prefer *_best.zip, else final *.zip."""
        return self.find_best_model_path(model_name)

    def find_best_model_path(self, model_name: str) -> Optional[Path]:
        """Find model saved as best, if it doesnt exist take final base."""
        final_path = self.models_dir / model_name / f"{model_name}_best.zip"
        if final_path.is_file():
            return final_path
        print(f"No best model found for {model_name}, falling back to final model if it exists.")
        final_path = self.models_dir / model_name / f"{model_name}.zip"
        if final_path.is_file():
            return final_path
        print(f"No final model found for {model_name}. Skipping model.")
        return None

    def _build_eval_alias(self, model_name: str) -> Tuple[str, Path, Path]:
        """Create/reuse one temporary model alias folder for a model's candidate evaluations."""
        safe_model = re.sub(r"[^A-Za-z0-9_\-]", "_", model_name)[:24]
        # Keep temp alias short to avoid MAX_PATH failures on Windows.
        # Stable per model so we can reuse it across all candidate checkpoints.
        fingerprint = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:10]
        eval_model_name = f"eval_{safe_model}_{fingerprint}"
        # Must live under models/<name>/<name>.zip because SacUtilities.resolve_model_paths
        # always resolves inference artifacts from that layout.
        eval_model_dir = self.models_dir / eval_model_name
        eval_model_dir.mkdir(parents=True, exist_ok=True)
        eval_zip = eval_model_dir / f"{eval_model_name}.zip"
        return eval_model_name, eval_model_dir, eval_zip

    @staticmethod
    def _stage_candidate_artifact(candidate_path: Path, eval_zip: Path) -> None:
        """Swap the staged candidate artifact used by the shared evaluation alias."""
        if eval_zip.exists():
            eval_zip.unlink()
        shutil.copy2(candidate_path, eval_zip)

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
            "avg_speed": avg_speed,
            "avg_reward": avg_rewards,
            "avg_num_crashes": avg_crashes,
        }

    @staticmethod
    def _ranking_key(aggregated: Dict) -> Tuple:
        """
        Sort key where larger tuple is better.

        Priority:
        1) Runs that completed and produced at least one lap.
        2) Higher lap completion rate.
        3) More average laps completed.
        4) Lower average lap time.
        5) Higher average speed.
        6) Fewer crashes.
        """
        avg_lap_time = aggregated.get("avg_lap_time")
        avg_speed = aggregated.get("avg_speed")
        avg_num_crashes = aggregated.get("avg_num_crashes")

        return (
            aggregated.get("num_lap_runs", 0) > 0,
            aggregated.get("lap_completion_rate", 0.0),
            aggregated.get("avg_laps_completed") or -1.0,
            -avg_lap_time if avg_lap_time is not None else float("-inf"),
            avg_speed if avg_speed is not None else float("-inf"),
            -(avg_num_crashes if avg_num_crashes is not None else float("inf")),
        )

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
                "avg_speed",
                "avg_reward",
                "avg_num_crashes",
            ]:
                f.write(f"{key}: {best_row.get(key)}\n")

        return csv_path, txt_path

    def evaluate_model(self, model_name: str, model_prefix: str) -> Optional[Dict]:
        """Evaluate the selected artifact for a model and return summary."""
        # final_model_path = self.find_best_model_path(model_name)
        best_model = self.find_best_model(model_name)
        if not best_model:
            print(f"Best model not found for model: {model_name}")
            return None

        candidates: List[Tuple[str, str, Path]] = []
        if best_model is not None:
            candidate_type = "best" if best_model.name.endswith(f"{self.best_suffix}.zip") else "final"
            candidates.append((model_name, candidate_type, best_model))
        # for checkpoint in checkpoints_to_eval:
        #     candidates.append((self._checkpoint_candidate_name(checkpoint), "checkpoint", checkpoint))

        if not candidates:
            print(f"No candidates to evaluate for model: {model_name}")
            return None

        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"Best model present: {'yes' if best_model is not None else 'no'}")
        # print(f"Found {len(checkpoints)} total checkpoints")
        # print(f"Skipping first {skip_count} checkpoints, evaluating {len(checkpoints_to_eval)}")
        # print(f"Total evaluated candidates (final + selected checkpoints): {len(candidates)}")
        print(f"{'=' * 80}")

        aggregated_rows: List[Dict] = []
        all_run_durations: List[float] = []
        eval_model_name, eval_model_dir, eval_zip = self._build_eval_alias(model_name)

        try:
            for idx, (candidate_name, candidate_type, candidate_path) in enumerate(candidates, 1):
                print(f"[{idx}/{len(candidates)}] Evaluating {candidate_name} ({candidate_type})")
                self._stage_candidate_artifact(candidate_path, eval_zip)

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
                print(
                    f"  -> completed_runs={aggregated['num_completed_runs']}/{aggregated['num_runs']}, "
                    f"lap_runs={aggregated['num_lap_runs']}/{aggregated['num_runs']}, avg_lap_time={lap_time_text}"
                )
        finally:
            if not self.keep_eval_models and eval_model_dir.exists():
                shutil.rmtree(eval_model_dir)

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

        best_row = max(aggregated_rows, key=self._ranking_key)
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

    def run(self, model_name: Optional[str], model_prefix: Optional[str]) -> None:
        """Execute batch model comparison for the requested models."""
        # This workload repeatedly constructs CarSystem/WaypointUtils during evaluation.
        # Disable background waypoint reload threads for this script run to avoid
        # thread creation overhead accumulating over long checkpoint batches.
        previous_reload_setting = Settings.RELOAD_WP_IN_BACKGROUND
        Settings.RELOAD_WP_IN_BACKGROUND = False

        try:
            models = self.find_models(model_name=model_name, model_prefix=model_prefix)
            if not models:
                target = model_name if model_name else model_prefix
                print(f"No matching model directories found for: {target}")
                return

            if model_prefix:
                batch_tag = re.sub(r"[^A-Za-z0-9_\-]", "_", model_prefix)
            elif model_name:
                batch_tag = re.sub(r"[^A-Za-z0-9_\-]", "_", model_name)
            else:
                batch_tag = "selection"

            run_results_dir = self.results_dir / batch_tag
            run_results_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir = run_results_dir

            print(f"Target models: {len(models)}")
            print(f"Batch tag: {batch_tag}")
            print(f"Results directory: {self.results_dir}")
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
                    "report_csv",
                    "report_txt",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_summaries:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

            # Global best across all evaluated model candidates (best-or-final per model).
            global_candidates: List[Dict] = []
            for summary in all_summaries:
                for row in summary.get("evaluated_rows", []):
                    global_candidates.append(dict(row))

            if global_candidates:
                global_best = max(global_candidates, key=self._ranking_key)
                top_candidates = sorted(global_candidates, key=self._ranking_key, reverse=True)[:30]
                global_best_model = str(global_best["model_name"])
                global_best_source = Path(global_best["candidate_path"])
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
                        "avg_speed",
                        "avg_reward",
                        "avg_num_crashes",
                    ]:
                        f.write(f"{key}: {global_best.get(key)}\n")
                    f.write(f"promoted_global_artifact: {global_best_target}\n\n")

                    f.write("TOP 30 CANDIDATES (RANKED)\n")
                    f.write("-" * 80 + "\n")
                    for rank, candidate in enumerate(top_candidates, 1):
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
                            "avg_speed",
                            "avg_reward",
                            "avg_num_crashes",
                        ]:
                            f.write(f"{key}: {candidate.get(key)}\n")
                        f.write("\n")

                print(
                    f"\nGlobal best candidate: {global_best['candidate_name']} "
                )
                print(f"Global promoted artifact: {global_best_target}")
                print(f"Global summary: {global_txt}")

            print(f"\nOverall summary saved to: {overall_csv}")
        finally:
            Settings.RELOAD_WP_IN_BACKGROUND = previous_reload_setting


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare best/final model artifacts across a batch")
    parser.add_argument("--model-name", help="Exact model directory name to process")
    parser.add_argument("--model-prefix", help="Process all model directories with this prefix")
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

    args = parser.parse_args()

    if not args.model_name and not args.model_prefix:
        parser.error("Provide either --model-name or --model-prefix")

    selector = CheckpointSelector(
        max_length=args.max_length,
        repeats=args.repeats,
        verbose=args.verbose,
        results_dir=args.results_dir,
        best_suffix=args.best_suffix,
        no_overwrite_best=args.no_overwrite_best,
        keep_eval_models=args.keep_eval_models,
    )
    selector.run(model_name=args.model_name, model_prefix=args.model_prefix)


if __name__ == "__main__":
    main()
