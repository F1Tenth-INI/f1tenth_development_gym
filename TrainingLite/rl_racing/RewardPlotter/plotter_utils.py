"""Install and serve the static reward-components viewer."""

from __future__ import annotations

import shutil
from pathlib import Path

REWARD_PLOTTER_DIR = Path(__file__).resolve().parent
MODELS_ROOT = REWARD_PLOTTER_DIR.parent / "models"
STATIC_FILES = ("index.html", "app.js", "styles.css")
EPISODES_CSV = "episodes.csv"


def resolve_model_dir(model_name: str) -> Path:
    model_dir = MODELS_ROOT / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    return model_dir


def _episodes_csv_mtime(model_dir: Path) -> float:
    episodes_path = model_dir / EPISODES_CSV
    if not episodes_path.is_file():
        return 0.0
    return episodes_path.stat().st_mtime


def list_model_dirs() -> list[Path]:
    if not MODELS_ROOT.is_dir():
        return []
    model_dirs = [
        entry
        for entry in MODELS_ROOT.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    model_dirs.sort(key=lambda path: (_episodes_csv_mtime(path), path.name.lower()), reverse=True)
    return model_dirs


def list_models() -> list[dict[str, str | float]]:
    models = []
    for model_dir in list_model_dirs():
        mtime = _episodes_csv_mtime(model_dir)
        models.append(
            {
                "name": model_dir.name,
                "mtime": mtime,
                "has_episodes": mtime > 0,
            }
        )
    return models


def get_latest_model_name() -> str | None:
    for model in list_models():
        if model["has_episodes"]:
            return str(model["name"])
    models = list_models()
    return str(models[0]["name"]) if models else None


def install_reward_plotter(model_dir: str | Path) -> None:
    """Copy viewer static files next to episodes.csv in a model directory."""
    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in STATIC_FILES:
        shutil.copy2(REWARD_PLOTTER_DIR / filename, target_dir / filename)
