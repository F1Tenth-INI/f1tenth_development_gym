from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenameOp:
    old_path: Path
    new_path: Path


def matches_target_model_name(name: str, old_prefix: str) -> bool:
    if not name.startswith(old_prefix):
        return False

    suffix = name[len(old_prefix) :]
    return re.fullmatch(r".+_idx[123]", suffix) is not None


def build_new_model_name(name: str, old_prefix: str, new_prefix: str) -> str:
    suffix = name[len(old_prefix) :]
    return f"{new_prefix}{suffix}"


def plan_model_renames(model_dir: Path, old_prefix: str, new_prefix: str) -> list[RenameOp]:
    old_model_name = model_dir.name
    new_model_name = build_new_model_name(old_model_name, old_prefix, new_prefix)

    operations: list[RenameOp] = []

    # Rename descendants first so parent path changes do not invalidate child paths.
    descendants = sorted(
        model_dir.rglob("*"),
        key=lambda p: (len(p.parts), str(p)),
        reverse=True,
    )

    for entry in descendants:
        if old_model_name not in entry.name:
            continue

        new_name = entry.name.replace(old_model_name, new_model_name)
        if new_name == entry.name:
            continue

        operations.append(RenameOp(old_path=entry, new_path=entry.with_name(new_name)))

    # Rename the model directory itself last.
    operations.append(
        RenameOp(old_path=model_dir, new_path=model_dir.with_name(new_model_name))
    )

    return operations


def has_conflicts(operations: list[RenameOp]) -> tuple[bool, str]:
    old_paths = {op.old_path.resolve() for op in operations}
    seen_targets: set[Path] = set()

    for op in operations:
        target = op.new_path.resolve()

        if target in seen_targets:
            return True, f"duplicate target in plan: {target}"
        seen_targets.add(target)

        # Allow targets that are old paths because they will be moved away first.
        if target not in old_paths and target.exists():
            return True, f"target already exists: {target}"

    return False, ""


def find_target_model_dirs(models_root: Path, old_prefix: str) -> list[Path]:
    if not models_root.exists() or not models_root.is_dir():
        return []

    candidates = [p for p in models_root.iterdir() if p.is_dir()]
    return sorted(
        [p for p in candidates if matches_target_model_name(p.name, old_prefix)],
        key=lambda p: p.name,
    )


def execute_plan(operations: list[RenameOp], apply: bool) -> int:
    changed = 0
    for op in operations:
        if op.old_path == op.new_path:
            continue

        print(f"{op.old_path} -> {op.new_path}")
        if apply:
            op.old_path.rename(op.new_path)
        changed += 1

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename sweep model prefixes while preserving A/B/R/CritU/idx suffixes, "
            "including model folder, model zip, and checkpoint names."
        )
    )
    parser.add_argument(
        "--models-root",
        default="TrainingLite/rl_racing/models",
        help="Root directory containing model folders",
    )
    parser.add_argument(
        "--old-prefix",
        default="Sweep_BETTER_fresh_cur_",
        help="Old model name prefix",
    )
    parser.add_argument(
        "--new-prefix",
        default="0404_sweep_",
        help="New model name prefix",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, the script runs in dry-run mode.",
    )

    args = parser.parse_args()
    models_root = Path(args.models_root).expanduser().resolve()

    model_dirs = find_target_model_dirs(models_root, args.old_prefix)
    if not model_dirs:
        print(f"No matching model directories found in: {models_root}")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Found {len(model_dirs)} matching model directories")

    total_changed = 0
    skipped_conflicts = 0

    for model_dir in model_dirs:
        print(f"\nModel: {model_dir.name}")
        operations = plan_model_renames(
            model_dir=model_dir,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
        )

        conflict, reason = has_conflicts(operations)
        if conflict:
            print(f"SKIP (conflict): {reason}")
            skipped_conflicts += 1
            continue

        changed = execute_plan(operations, apply=args.apply)
        total_changed += changed

    print(
        f"\n[{mode}] changed_paths={total_changed}, "
        f"skipped_conflict_models={skipped_conflicts}, "
        f"total_models={len(model_dirs)}"
    )


if __name__ == "__main__":
    main()
