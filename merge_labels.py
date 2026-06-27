#!/usr/bin/env python3
"""
Merge label files into normalized CSVs.
Adds a 'Status' column: filled for rows within a labeled time range, blank otherwise.
Output goes to csv_labelled/.
"""

import os
import csv

NORM_DIR   = os.path.join(os.path.dirname(__file__), "csv_normalized")
LABEL_DIR  = os.path.join(os.path.dirname(__file__), "label")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "csv_labelled")


def load_label_ranges(label_path):
    """Return list of (start, end, status) tuples, skipping blank/malformed rows."""
    ranges = []
    with open(label_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s, e, st = row.get("start_time", "").strip(), row.get("end_time", "").strip(), row.get("Status", "").strip()
            if s and e and st:
                try:
                    ranges.append((float(s), float(e), st))
                except ValueError:
                    print(f"    WARN  bad row in {os.path.basename(label_path)}: {row}")
    return ranges


def label_for(t, ranges):
    for start, end, status in ranges:
        if start <= t <= end:
            return status
    return ""


def merge_file(csv_path, label_path, out_path):
    ranges = load_label_ranges(label_path)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["Status"]
        rows = list(reader)

    for row in rows:
        row["Status"] = label_for(float(row["timestamp"]), ranges)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labeled = sum(1 for r in rows if r["Status"])
    print(f"  OK  {os.path.basename(csv_path):35s}  {labeled}/{len(rows)} rows labeled")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(NORM_DIR) if f.endswith(".csv"))
    missing_labels = []

    for filename in csv_files:
        stem = os.path.splitext(filename)[0]
        label_filename = f"{stem}_label.csv"
        label_path = os.path.join(LABEL_DIR, label_filename)

        if not os.path.exists(label_path):
            missing_labels.append(filename)
            print(f"  SKIP {filename:35s}  (no label file found)")
            continue

        merge_file(
            os.path.join(NORM_DIR, filename),
            label_path,
            os.path.join(OUTPUT_DIR, filename),
        )

    print(f"\nDone. Merged files written to {OUTPUT_DIR}/")
    if missing_labels:
        print(f"Missing labels for: {', '.join(missing_labels)}")


if __name__ == "__main__":
    main()
