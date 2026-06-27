"""Build the channel matrix [T, C] from a resampled signal DataFrame.

Raw channels come straight from the resampled signals (in the canonical order
from config.raw_channel_names()). Engineered channels reproduce the exact
heuristic formulas from the reuse-source live_entanglement_detector.py:

    thigh_tau_down   = max(0, -thigh_tau)
    thigh_down_effort= thigh_tau_down / (|thigh_dq| + EFFORT_DQ_EPSILON)
    tau_sum          = |hip_tau| + |thigh_tau| + |calf_tau|
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def engineered_channels(res_df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for leg in C.LEG_ORDER:
        thigh_tau = res_df[f"{leg}_thigh_tau"].to_numpy(dtype=np.float64)
        thigh_dq = res_df[f"{leg}_thigh_dq"].to_numpy(dtype=np.float64)
        hip_tau = res_df[f"{leg}_hip_tau"].to_numpy(dtype=np.float64)
        calf_tau = res_df[f"{leg}_calf_tau"].to_numpy(dtype=np.float64)

        thigh_tau_down = np.maximum(0.0, -thigh_tau)
        out[f"{leg}_thigh_tau_down"] = thigh_tau_down
        out[f"{leg}_thigh_down_effort"] = thigh_tau_down / (np.abs(thigh_dq) + C.EFFORT_DQ_EPSILON)
        out[f"{leg}_tau_sum"] = np.abs(hip_tau) + np.abs(thigh_tau) + np.abs(calf_tau)
    return out


def build_channel_matrix(res_df: pd.DataFrame) -> np.ndarray:
    """Return float32 array of shape [T, C] in config.channel_names() order."""
    names = C.channel_names()
    eng = engineered_channels(res_df) if (C.USE_ENGINEERED or C.ENGINEERED_ONLY) else {}
    cols = []
    for name in names:
        if name in res_df.columns:
            cols.append(res_df[name].to_numpy(dtype=np.float32))
        else:
            cols.append(eng[name].astype(np.float32))
    return np.stack(cols, axis=1)  # [T, C]


if __name__ == "__main__":
    from . import io_load, resample

    recs = io_load.list_recordings()
    df = io_load.load_recording(recs["back_left_hand2"])  # RL entangled
    res_df, _ = resample.resample_recording(df)
    X = build_channel_matrix(res_df)
    names = C.channel_names()
    print(f"channel matrix shape {X.shape}  (expected C={C.n_channels()})")
    assert X.shape[1] == C.n_channels()

    # Cross-check engineered formula against a manual computation on row 100.
    i = 100
    rl_thigh_tau = res_df["RL_thigh_tau"].iloc[i]
    rl_thigh_dq = res_df["RL_thigh_dq"].iloc[i]
    expect_down = max(0.0, -rl_thigh_tau)
    expect_eff = expect_down / (abs(rl_thigh_dq) + C.EFFORT_DQ_EPSILON)
    j_down = names.index("RL_thigh_tau_down")
    j_eff = names.index("RL_thigh_down_effort")
    assert abs(X[i, j_down] - expect_down) < 1e-4, (X[i, j_down], expect_down)
    assert abs(X[i, j_eff] - expect_eff) < 1e-4, (X[i, j_eff], expect_eff)
    print("Engineered channels match heuristic formulas. OK.")
    print("first 3 channel names:", names[:3], "| last 3:", names[-3:])
