#!/usr/bin/env python3
"""
Copy model folders to a new location, excluding stats_log.csv files.

This script recursively copies directory structures and files while skipping
any stats_log.csv files to save transfer time and storage space.

Usage:
    python -u copy_models_exclude_stats.py --source <src_dir> --dest <dst_dir> --prefix <prefix>
    python -u copy_models_exclude_stats.py --source TrainingLite/rl_racing/models --dest /mnt/external/models --prefix 0411
    python -u copy_models_exclude_stats.py --source TrainingLite/rl_racing/models --dest models_no_stat_csv --prefix 0412
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def should_exclude(file_path: Path) -> bool:
    """Return True if file should be excluded from copy."""
    return file_path.name == "stats_log.csv"


def copy_tree_exclude(src: Path, dst: Path, prefix: str = None, verbose: bool = False) -> dict:
    """
    Recursively copy directory tree, excluding stats_log.csv files.
    
    Args:
        src: Source directory to copy from
        dst: Destination directory root
        prefix: Optional prefix filter; only copy dirs starting with this
        verbose: Print detailed copy info
        
    Returns:
        Dict with copy statistics
    """
    stats = {
        "dirs_copied": 0,
        "files_copied": 0,
        "files_skipped": 0,
        "bytes_copied": 0,
        "models_processed": 0,
    }
    
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")
    
    dst.mkdir(parents=True, exist_ok=True)
    
    for item in sorted(src.iterdir()):
        # Filter by prefix (required)
        if not item.name.startswith(prefix):
            continue
        
        if item.is_dir():
            stats["models_processed"] += 1
            dst_subdir = dst / item.name
            dst_subdir.mkdir(parents=True, exist_ok=True)
            stats["dirs_copied"] += 1
            
            if verbose:
                print(f"[DIR]  {item.name}")
            
            # Recursively copy contents
            for root, dirs, files in os.walk(item):
                root_path = Path(root)
                # Create all subdirectories
                for dir_name in dirs:
                    src_subdir = root_path / dir_name
                    rel_path = src_subdir.relative_to(item)
                    dst_full_subdir = dst_subdir / rel_path
                    dst_full_subdir.mkdir(parents=True, exist_ok=True)
                    stats["dirs_copied"] += 1
                
                # Copy all files except stats_log.csv
                for file_name in files:
                    src_file = root_path / file_name
                    if should_exclude(src_file):
                        stats["files_skipped"] += 1
                        if verbose:
                            print(f"  [SKIP] {src_file.name}")
                        continue
                    
                    rel_path = src_file.relative_to(item)
                    dst_file = dst_subdir / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        shutil.copy2(src_file, dst_file)
                        file_size = src_file.stat().st_size
                        stats["files_copied"] += 1
                        stats["bytes_copied"] += file_size
                        
                        if verbose:
                            size_mb = file_size / (1024 * 1024)
                            print(f"  [COPY] {rel_path} ({size_mb:.2f} MB)")
                    except Exception as e:
                        print(f"[ERROR] Failed to copy {src_file}: {e}", file=sys.stderr)
    
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy model folders while excluding stats_log.csv files"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source directory containing model folders",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory where copies will be placed",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Only copy model folders matching this prefix",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed copy information",
    )
    
    args = parser.parse_args()
    
    src_dir = Path(args.source)
    dst_dir = Path(args.dest)
    
    if not src_dir.is_dir():
        print(f"[ERROR] Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dst_dir}")
    if args.prefix:
        print(f"Filter prefix: {args.prefix}")
    print()
    
    try:
        stats = copy_tree_exclude(
            src=src_dir,
            dst=dst_dir,
            prefix=args.prefix,
            verbose=args.verbose,
        )
        
        print("\n" + "=" * 80)
        print("COPY COMPLETE")
        print("=" * 80)
        print(f"Models processed: {stats['models_processed']}")
        print(f"Directories created: {stats['dirs_copied']}")
        print(f"Files copied: {stats['files_copied']}")
        print(f"Files skipped (stats_log.csv): {stats['files_skipped']}")
        print(f"Data transferred: {stats['bytes_copied'] / (1024 * 1024):.2f} MB")
        
    except Exception as e:
        print(f"[ERROR] Copy failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
