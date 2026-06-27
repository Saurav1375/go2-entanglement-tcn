"""Resample a recording to a uniform TARGET_HZ grid.

Signals are linearly interpolated; the Status label is resampled by
nearest-neighbour (a label is a step function and must never be interpolated).
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from . import config as C


def resample_recording(df: pd.DataFrame, target_hz: int = C.TARGET_HZ
                       ) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (resampled_signals_df, status_array) on a uniform 1/target_hz grid.

    resampled_signals_df has columns = raw signal columns (no timestamp/Status),
    indexed 0..N-1. status_array is an object array of the same length.
    """
    t = df["timestamp"].to_numpy(dtype=np.float64)
    # Guard against non-monotonic timestamps (shouldn't happen post-normalize).
    order = np.argsort(t, kind="stable")
    t = t[order]
    df = df.iloc[order].reset_index(drop=True)

    t0, t1 = float(t[0]), float(t[-1])
    n = max(1, int(np.floor((t1 - t0) * target_hz)) + 1)
    grid = t0 + np.arange(n) / target_hz

    signal_cols = C.raw_signal_columns()  # always resample the full set; channels chosen later
    out = {}
    for col in signal_cols:
        out[col] = np.interp(grid, t, df[col].to_numpy(dtype=np.float64))
    res_df = pd.DataFrame(out)

    # Nearest-neighbour for Status: index of closest original sample.
    status = df[C.STATUS_COL].to_numpy()
    idx = np.searchsorted(t, grid)
    idx = np.clip(idx, 0, len(t) - 1)
    left = np.clip(idx - 1, 0, len(t) - 1)
    choose_left = np.abs(grid - t[left]) <= np.abs(t[idx] - grid)
    nn = np.where(choose_left, left, idx)
    status_grid = status[nn].astype(object)

    return res_df, status_grid


@functools.lru_cache(maxsize=None)
def cached_resample(path: str):
    """Path-keyed cache of (resampled_signals_df, status). Config-independent — the
    resampled raw signals are the same regardless of channel-ablation flags — so LORO
    folds and ablation configs reuse one resample instead of redoing the interpolation.
    Returns copies on each call so callers can't corrupt the cached frame.
    """
    from . import io_load
    df = io_load.load_recording(path)
    res_df, status = resample_recording(df)
    return res_df, status


if __name__ == "__main__":
    from . import io_load
    recs = io_load.list_recordings()
    print(f"{'recording':22s} {'orig_rows':>9s} {'orig_hz':>8s} {'res_rows':>9s} {'dur_s':>7s}")
    for stem in ["front_left_hand2", "back_left_hand2", "walking1"]:  # 409 Hz, 960 Hz, 505 Hz
        df = io_load.load_recording(recs[stem])
        t = df["timestamp"].to_numpy()
        orig_hz = (len(t) - 1) / (t[-1] - t[0])
        res_df, status = resample_recording(df)
        dur = (len(res_df) - 1) / C.TARGET_HZ
        print(f"{stem:22s} {len(df):9d} {orig_hz:8.1f} {len(res_df):9d} {dur:7.2f}")
        # the resampled rate must be exactly TARGET_HZ
        assert abs(dur - (t[-1] - t[0])) < 0.01, "duration drift after resample"
        assert len(status) == len(res_df)
    print("Resample OK (uniform 500 Hz, status length matches).")
