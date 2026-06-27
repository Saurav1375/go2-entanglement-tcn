"""CSV discovery, loading, validation for csv_labelled/."""
from __future__ import annotations

import glob
import os

import pandas as pd

from . import config as C

# Expected raw signal columns (everything except timestamp + Status). Use the FIXED
# full column set (independent of ablation flags) for validation.
_EXPECTED_SIGNAL_COLS = C.raw_signal_columns()
_REQUIRED_COLS = ["timestamp"] + _EXPECTED_SIGNAL_COLS + [C.STATUS_COL]


def list_recordings(directory: str | None = None) -> dict[str, str]:
    """Map recording stem -> csv path for every file in csv_labelled/."""
    directory = directory or C.CSV_LABELLED_DIR
    out: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        out[stem] = path
    return out


def load_recording(path: str) -> pd.DataFrame:
    """Load one CSV, validate columns, return DataFrame with a clean Status column.

    Status is normalized: blanks/NaN -> "" (empty string).
    """
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing[:8]}"
                         f"{'...' if len(missing) > 8 else ''}")
    # Normalize Status: blank cells read as NaN -> ""
    df[C.STATUS_COL] = df[C.STATUS_COL].fillna("").astype(str).str.strip()
    return df


def validate_filename_table(directory: str | None = None) -> list[tuple[str, set[str], str]]:
    """Return (stem, affected_legs, category) for every recording; for inspection/tests."""
    rows = []
    for stem in list_recordings(directory):
        legs = C.parse_legs(stem)
        if legs:
            category = "entangled"
        elif stem.startswith("walking"):
            category = "walking"
        elif stem.startswith("stop"):
            category = "stop"
        else:
            category = "unknown"
        rows.append((stem, legs, category))
    return rows


if __name__ == "__main__":
    recs = list_recordings()
    print(f"Found {len(recs)} recordings in {C.CSV_LABELLED_DIR}\n")
    print(f"{'recording':24s} {'legs':12s} {'category':10s} {'rows':>7s} {'statuses'}")
    for stem, legs, cat in validate_filename_table():
        df = load_recording(recs[stem])
        statuses = sorted(s for s in df[C.STATUS_COL].unique() if s)
        legs_str = ",".join(sorted(legs)) if legs else "-"
        print(f"{stem:24s} {legs_str:12s} {cat:10s} {len(df):7d} {statuses}")

    # sanity: every split file exists; positives are the 12 known files
    all_stems = set(recs)
    for split, files in C.SPLIT.items():
        for f in files:
            assert f in all_stems, f"split file {f} not found"
    print("\nSplit files all present. OK.")
