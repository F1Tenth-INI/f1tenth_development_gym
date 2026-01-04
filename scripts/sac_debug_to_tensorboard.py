#!/usr/bin/env python3
"""
Tail a learner_server log file, parse lines printed by the `[SAC DEBUG]` block,
and write scalars/histograms to TensorBoard. Does not modify the learner_server.

Usage:
  python scripts/sac_debug_to_tensorboard.py --log-file path/to/server.log --tb-dir path/to/tb_logs --follow

If --follow is passed the script will tail the file and write metrics live.
Otherwise it will parse the existing file and exit.

It also supports `--csv path/to/out.csv` to dump parsed rows.
"""
import re
import argparse
import time
import os
from collections import defaultdict

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

LINE_RE_1 = re.compile(r"\[SAC DEBUG\]\[step=(?P<step>\d+)\] batch_size=(?P<batch>\d+) \| steps mean=(?P<steps_mean>[-0-9.]+), min=(?P<steps_min>[-0-9.]+), max=(?P<steps_max>[-0-9.]+) \| reward mean=(?P<reward_mean>[-0-9.]+), std=(?P<reward_std>[-0-9.]+)")
LINE_RE_2 = re.compile(r"\[SAC DEBUG\]\[step=(?P<step>\d+)\] discounts mean=(?P<discounts_mean>[-0-9.]+) \| next_logprob mean=(?P<next_logprob_mean>[-0-9.]+), ent_coef=(?P<ent_coef>[-0-9.eE+-]+)")
LINE_RE_3 = re.compile(r"\[SAC DEBUG\]\[step=(?P<step>\d+)\] target_q mean=(?P<target_q_mean>[-0-9.]+), min=(?P<target_q_min>[-0-9.]+), max=(?P<target_q_max>[-0-9.]+) \| is_weights mean=(?P<is_weights_mean>[-0-9.]+)")

# Fallback: sometimes debug lines are printed as one-line combined; try to catch that
LINE_RE_COMBINED = re.compile(
    r"\[SAC DEBUG\]\[step=(?P<step>\d+)\] batch_size=(?P<batch>\d+) .*? steps mean=(?P<steps_mean>[-0-9.]+).*? reward mean=(?P<reward_mean>[-0-9.]+), std=(?P<reward_std>[-0-9.]+).*?discounts mean=(?P<discounts_mean>[-0-9.]+).*?next_logprob mean=(?P<next_logprob_mean>[-0-9.]+), ent_coef=(?P<ent_coef>[-0-9.eE+-]+).*?target_q mean=(?P<target_q_mean>[-0-9.]+), min=(?P<target_q_min>[-0-9.]+), max=(?P<target_q_max>[-0-9.]+) .*?is_weights mean=(?P<is_weights_mean>[-0-9.]+)",
    re.IGNORECASE | re.DOTALL,
)


def write_csv_row(csv_path, row, header=None):
    import csv
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if (not exists) and header:
            writer.writerow(header)
        writer.writerow(row)


def process_line(line, writer=None, csv_path=None):
    m = LINE_RE_COMBINED.search(line)
    if m:
        d = m.groupdict()
    else:
        d = {}
        m1 = LINE_RE_1.search(line)
        if m1:
            d.update(m1.groupdict())
        m2 = LINE_RE_2.search(line)
        if m2:
            d.update(m2.groupdict())
        m3 = LINE_RE_3.search(line)
        if m3:
            d.update(m3.groupdict())

    if not d:
        return False

    # Convert numeric fields
    numeric = {}
    for k, v in d.items():
        try:
            if "." in v or "e" in v.lower():
                numeric[k] = float(v)
            else:
                numeric[k] = int(v)
        except Exception:
            numeric[k] = v

    step = int(numeric.get("step", int(time.time())))

    # Write to TensorBoard
    if writer is not None:
        tags = {
            "sac/batch_size": numeric.get("batch"),
            "sac/steps_mean": numeric.get("steps_mean"),
            "sac/steps_min": numeric.get("steps_min"),
            "sac/steps_max": numeric.get("steps_max"),
            "sac/reward_mean": numeric.get("reward_mean"),
            "sac/reward_std": numeric.get("reward_std"),
            "sac/discounts_mean": numeric.get("discounts_mean"),
            "sac/next_logprob_mean": numeric.get("next_logprob_mean"),
            "sac/ent_coef": numeric.get("ent_coef"),
            "sac/target_q_mean": numeric.get("target_q_mean"),
            "sac/target_q_min": numeric.get("target_q_min"),
            "sac/target_q_max": numeric.get("target_q_max"),
            "sac/is_weights_mean": numeric.get("is_weights_mean"),
        }
        for tag, val in tags.items():
            if val is not None:
                try:
                    writer.add_scalar(tag, float(val), step)
                except Exception:
                    pass

    # Append CSV
    if csv_path is not None:
        header = [
            "step",
            "batch_size",
            "steps_mean",
            "steps_min",
            "steps_max",
            "reward_mean",
            "reward_std",
            "discounts_mean",
            "next_logprob_mean",
            "ent_coef",
            "target_q_mean",
            "target_q_min",
            "target_q_max",
            "is_weights_mean",
        ]
        row = [
            step,
            numeric.get("batch"),
            numeric.get("steps_mean"),
            numeric.get("steps_min"),
            numeric.get("steps_max"),
            numeric.get("reward_mean"),
            numeric.get("reward_std"),
            numeric.get("discounts_mean"),
            numeric.get("next_logprob_mean"),
            numeric.get("ent_coef"),
            numeric.get("target_q_mean"),
            numeric.get("target_q_min"),
            numeric.get("target_q_max"),
            numeric.get("is_weights_mean"),
        ]
        write_csv_row(csv_path, row, header=header)

    return True


def tail_and_parse(log_file, tb_dir, csv_path=None, follow=False):
    writer = None
    if tb_dir is not None:
        if SummaryWriter is None:
            print("TensorBoard SummaryWriter not available. Install torch and tensorboard.")
        else:
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"Writing TensorBoard logs to: {tb_dir}")

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        if follow:
            # Seek to end and wait for new lines
            f.seek(0, os.SEEK_END)
            print(f"Tailing {log_file} (CTRL+C to stop)")
            try:
                while True:
                    where = f.tell()
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        f.seek(where)
                    else:
                        process_line(line, writer=writer, csv_path=csv_path)
            except KeyboardInterrupt:
                print("Stopping tail.")
        else:
            # Read entire file
            for line in f:
                process_line(line, writer=writer, csv_path=csv_path)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--log-file', required=True, help='Path to learner_server stdout log file')
    p.add_argument('--tb-dir', required=False, help='TensorBoard log directory (if omitted, TB not used)')
    p.add_argument('--csv', required=False, help='Optional CSV output file to append parsed rows to')
    p.add_argument('--follow', action='store_true', help='Tail the log file (live)')
    args = p.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Log file does not exist: {args.log_file}")
        raise SystemExit(1)

    tail_and_parse(args.log_file, tb_dir=args.tb_dir, csv_path=args.csv, follow=args.follow)
