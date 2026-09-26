"""Checkpoint directories for the hierarchical world-model agent.

Model sizes and training flags live in ``utilities/Settings.py``. This module
only locates the weight files under ``TrainingLite/hwm/models/<name>/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "TrainingLite" / "hwm" / "models"


def resolve_model_dir(model_name: str, create: bool = False) -> Path:
    model_dir = MODELS_ROOT / str(model_name)
    if create:
        os.makedirs(model_dir, exist_ok=True)
    return model_dir


def model_exists(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    model_dir = resolve_model_dir(model_name)
    return model_dir.is_dir() and any(model_dir.glob("*.pt"))
