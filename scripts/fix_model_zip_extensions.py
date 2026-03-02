from __future__ import annotations

import argparse
from pathlib import Path


def rename_missing_zip(model_dir: Path, dry_run: bool = False) -> tuple[bool, str]:
    bare_file = model_dir / model_dir.name
    zip_file = model_dir / f"{model_dir.name}.zip"

    if zip_file.exists():
        return False, f"SKIP (already has zip): {zip_file}"

    if not bare_file.exists() or not bare_file.is_file():
        return False, f"SKIP (missing expected file): {bare_file}"

    if dry_run:
        return True, f"DRY-RUN rename: {bare_file.name} -> {zip_file.name}"

    bare_file.rename(zip_file)
    return True, f"RENAMED: {bare_file.name} -> {zip_file.name}"


def find_model_dirs(path: Path, recursive: bool = False) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []

    expected_file = path / path.name
    expected_zip = path / f"{path.name}.zip"
    if expected_file.is_file() or expected_zip.is_file():
        return [path]

    pattern = "**/*" if recursive else "*"
    dirs = [entry for entry in path.glob(pattern) if entry.is_dir()]
    return sorted(dirs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add .zip extension to model files named like their folder."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="TrainingLite/rl_racing/models",
        help="Model folder or models root directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively when path is a root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended renames without changing files",
    )
    args = parser.parse_args()

    target_path = Path(args.path).expanduser().resolve()
    model_dirs = find_model_dirs(target_path, recursive=args.recursive)

    if not model_dirs:
        print(f"No model directories found at: {target_path}")
        return

    changed = 0
    skipped = 0

    for model_dir in model_dirs:
        was_changed, message = rename_missing_zip(model_dir, dry_run=args.dry_run)
        print(message)
        if was_changed:
            changed += 1
        else:
            skipped += 1

    mode = "DRY-RUN" if args.dry_run else "DONE"
    print(f"[{mode}] changed={changed}, skipped={skipped}, total={len(model_dirs)}")


if __name__ == "__main__":
    main()