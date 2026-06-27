"""Slice a resampled recording into fixed-length windows with derived labels.

A window is the last WINDOW_SAMPLES rows ending at index i (causal: the window's
label/decision is attributed to its final timestep, matching online use).

Per-window labels (over the window's Status slice):
  entangled_fraction = fraction of rows == "Entangled"
  y_bin = 1 if entangled_fraction >= POS_FRACTION else 0
  y_legs = file's affected legs (multi-hot [FR,FL,RR,RL]) if y_bin else zeros
A window is DROPPED entirely if any row in it is blank ("") -- this removes
the warm-up/cool-down at file ends (blanks are contiguous there).

`train_mask` is False for windows in the transition band
(TRANSITION_LOW < entangled_fraction < POS_FRACTION); those are excluded from
training but kept for evaluation (honest onset/latency).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class WindowSet:
    X: np.ndarray            # [N, C, WINDOW_SAMPLES]  float32
    y_bin: np.ndarray        # [N]                     float32
    y_legs: np.ndarray       # [N, 4]                  float32
    end_idx: np.ndarray      # [N] int   resampled-row index of each window's last sample
    ent_frac: np.ndarray     # [N] float entangled fraction in the window
    train_mask: np.ndarray   # [N] bool  False inside the transition band
    stem: str
    onset_idx: int           # first resampled Entangled row (-1 if none)


def _legs_multihot(legs: set[str]) -> np.ndarray:
    return np.array([1.0 if leg in legs else 0.0 for leg in C.LEG_ORDER], dtype=np.float32)


def make_windows(channel_matrix: np.ndarray, status: np.ndarray, stem: str,
                 affected_legs: set[str], hop: int) -> WindowSet:
    T, Cc = channel_matrix.shape
    W = C.WINDOW_SAMPLES
    legs_vec = _legs_multihot(affected_legs)

    ent_mask = (status == "Entangled")
    blank_mask = (status == "")
    onset_idx = int(np.argmax(ent_mask)) if ent_mask.any() else -1

    Xs, ybin, ylegs, ends, fracs, tmask = [], [], [], [], [], []
    if T >= W:
        # transpose once: [T, C] -> [C, T] so a window slice is [C, W]
        Xt = channel_matrix.T  # [C, T]
        for end in range(W - 1, T, hop):
            start = end - W + 1
            sl = slice(start, end + 1)
            if blank_mask[sl].any():
                continue  # drop windows touching unlabeled warm-up/cool-down
            frac = float(ent_mask[sl].mean())
            is_pos = frac >= C.POS_FRACTION
            Xs.append(Xt[:, sl])
            ybin.append(1.0 if is_pos else 0.0)
            ylegs.append(legs_vec if is_pos else np.zeros(C.N_LEGS, dtype=np.float32))
            ends.append(end)
            fracs.append(frac)
            # exclude transition band from training
            in_transition = (C.TRANSITION_LOW < frac < C.POS_FRACTION)
            tmask.append(not in_transition)

    if Xs:
        X = np.stack(Xs).astype(np.float32)
        y_bin = np.array(ybin, dtype=np.float32)
        y_legs = np.stack(ylegs).astype(np.float32)
    else:
        X = np.zeros((0, Cc, W), dtype=np.float32)
        y_bin = np.zeros((0,), dtype=np.float32)
        y_legs = np.zeros((0, C.N_LEGS), dtype=np.float32)

    return WindowSet(
        X=X, y_bin=y_bin, y_legs=y_legs,
        end_idx=np.array(ends, dtype=np.int64),
        ent_frac=np.array(fracs, dtype=np.float32),
        train_mask=np.array(tmask, dtype=bool),
        stem=stem, onset_idx=onset_idx,
    )


def windows_for_recording(stem: str, path: str, hop: int) -> WindowSet:
    """Convenience: load -> resample -> features -> windows for one recording."""
    from . import resample, features
    res_df, status = resample.cached_resample(path)
    X = features.build_channel_matrix(res_df)
    return make_windows(X, status, stem, C.parse_legs(stem), hop)


if __name__ == "__main__":
    from . import io_load
    recs = io_load.list_recordings()
    print(f"{'recording':22s} {'n_win':>6s} {'pos':>5s} {'transition':>10s} {'onset_s':>8s}")
    for stem in ["back_left_hand2", "front_both_wire2", "walking1", "stop1", "walking5"]:
        ws = windows_for_recording(stem, recs[stem], hop=C.HOP_TRAIN)
        npos = int(ws.y_bin.sum())
        ntrans = int((~ws.train_mask).sum())
        onset_s = ws.onset_idx / C.TARGET_HZ if ws.onset_idx >= 0 else -1
        print(f"{stem:22s} {len(ws.y_bin):6d} {npos:5d} {ntrans:10d} {onset_s:8.2f}")
        if npos:
            # positive windows must carry exactly the file's affected legs
            pos_rows = ws.y_legs[ws.y_bin == 1]
            expected = sorted(C.parse_legs(stem))
            got = sorted(np.array(C.LEG_ORDER)[pos_rows[0] == 1].tolist())
            assert got == expected, (stem, got, expected)
        assert ws.X.shape[1:] == (C.n_channels(), C.WINDOW_SAMPLES)
    print("Windowing OK (shapes, labels, transition band).")
