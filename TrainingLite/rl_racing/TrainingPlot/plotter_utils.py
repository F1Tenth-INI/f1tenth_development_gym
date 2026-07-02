"""Shared helpers for TrainingPlot (model discovery and static file install)."""

from __future__ import annotations

import shutil
from pathlib import Path

TRAINING_PLOT_DIR = Path(__file__).resolve().parent
MODELS_ROOT = TRAINING_PLOT_DIR.parent / "models"
STATIC_FILES = ("index.html", "styles.css", "main.js", "metrics_tab.js", "reward_tab.js")
EPISODES_CSV = "episodes.csv"
LEARNING_METRICS_CSV = "learning_metrics.csv"


def resolve_model_dir(model_name: str) -> Path:
    model_dir = MODELS_ROOT / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    return model_dir


def _file_mtime(path: Path) -> float:
    if not path.is_file():
        return 0.0
    return path.stat().st_mtime


def _model_activity_mtime(model_dir: Path) -> float:
    return max(
        _file_mtime(model_dir / EPISODES_CSV),
        _file_mtime(model_dir / LEARNING_METRICS_CSV),
    )


def list_model_dirs() -> list[Path]:
    if not MODELS_ROOT.is_dir():
        return []
    model_dirs = [
        entry
        for entry in MODELS_ROOT.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    model_dirs.sort(key=lambda path: (_model_activity_mtime(path), path.name.lower()), reverse=True)
    return model_dirs


def list_models() -> list[dict[str, str | float | bool]]:
    models = []
    for model_dir in list_model_dirs():
        episodes_mtime = _file_mtime(model_dir / EPISODES_CSV)
        metrics_mtime = _file_mtime(model_dir / LEARNING_METRICS_CSV)
        models.append(
            {
                "name": model_dir.name,
                "mtime": max(episodes_mtime, metrics_mtime),
                "has_episodes": episodes_mtime > 0,
                "has_metrics": metrics_mtime > 0,
            }
        )
    return models


def get_latest_model_name() -> str | None:
    for model in list_models():
        if model["has_episodes"] or model["has_metrics"]:
            return str(model["name"])
    models = list_models()
    return str(models[0]["name"]) if models else None


def install_training_plot(model_dir: str | Path) -> None:
    """Copy viewer static files into a model directory (legacy convenience)."""
    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in STATIC_FILES:
        shutil.copy2(TRAINING_PLOT_DIR / filename, target_dir / filename)
