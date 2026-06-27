#!/usr/bin/env python3
"""
Normalize timestamp columns in CSV files so each file's timestamps
start at 0 (relative seconds from the first timestamp).
"""

import os
import csv
import argparse


def normalize_timestamps(input_dir, output_dir=None, inplace=False):
    if not inplace and output_dir is None:
        output_dir = input_dir + "_normalized"

    if not inplace:
        os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for filename in sorted(csv_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename) if not inplace else input_path

        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if "timestamp" not in fieldnames:
                print(f"  SKIP {filename} — no 'timestamp' column")
                continue
            rows = list(reader)

        if not rows:
            print(f"  SKIP {filename} — empty file")
            continue

        t0 = float(rows[0]["timestamp"])
        for row in rows:
            row["timestamp"] = f"{float(row['timestamp']) - t0:.6f}"

        dest = output_path if not inplace else input_path
        with open(dest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  OK  {filename}  (t0={t0}  →  0.000000)")

    print(f"\nDone. {'Modified in place' if inplace else f'Output written to {output_dir}'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize CSV timestamp columns to start at 0."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "csv"),
        help="Directory containing CSV files (default: ./csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write normalized files (default: <input_dir>_normalized)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original files instead of writing to a new directory",
    )
    args = parser.parse_args()

    normalize_timestamps(args.input_dir, args.output_dir, args.inplace)
