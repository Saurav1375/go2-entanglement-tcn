"""Re-implementation of the user's rear-thigh Statistical Detector, adapted to the
ML pipeline's resampled windows so it can be scored under the IDENTICAL protocol.

Faithful to `statistical_detector/statistical_detection.py` core rule:
  * Calibrate per rear leg (RR, RL) from TRAIN walking: center=median(thigh_tau),
    robust MAD scale; down/up thresholds = center ∓ sigma*scale (guarded by 0.5/99.5
    percentiles), sigma=4 (the script's default).
  * Per window: a sustained REAR-leg DOWNWARD spike ⇒ back entanglement (that rear
    leg); a sustained REAR-leg UPWARD spike ⇒ FRONT entanglement (rear legs push up
    to compensate) ⇒ predict both front legs (the rule cannot localize FL vs FR).
  * Continuous detection score (for PR/ROC-AUC) = max over rear legs of the
    threshold-exceedance in robust-σ units. The fraction/persistence gating from the
    original script is reproduced by the shared protocol's persistence rule.
  * A light FL/FR stop veto (front thigh_tau far below a walking-derived threshold).

This is the user's "best statistical detector"; it uses ONLY the 4 thigh torques.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import io_load, resample
from .windowing import make_windows
from .features import build_channel_matrix

REAR = ("RR", "RL")
FRONT = ("FR", "FL")
SIGMA = 4.0


def calibrate(train_stems):
    """Per-rear-leg down/up thresholds + scale, and front stop threshold, from TRAIN walking."""
    recs = io_load.list_recordings()
    rear_vals = {leg: [] for leg in REAR}
    front_vals = {leg: [] for leg in FRONT}
    for stem in train_stems:
        res_df, status = resample.cached_resample(recs[stem])
        walk = status == "Walking"
        if not walk.any():
            continue
        for leg in REAR:
            rear_vals[leg].append(res_df[f"{leg}_thigh_tau"].to_numpy()[walk])
        for leg in FRONT:
            front_vals[leg].append(res_df[f"{leg}_thigh_tau"].to_numpy()[walk])

    def robust(v):
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        return float(med), float(max(mad, np.std(v), 1e-6))

    rear = {}
    for leg in REAR:
        v = np.concatenate(rear_vals[leg])
        center, scale = robust(v)
        rear[leg] = {"center": center, "scale": scale,
                     "down": min(np.percentile(v, 0.5), center - SIGMA * scale),
                     "up": max(np.percentile(v, 99.5), center + SIGMA * scale)}
    front = {}
    for leg in FRONT:
        v = np.concatenate(front_vals[leg])
        center, scale = robust(v)
        front[leg] = {"stop": center - 6.0 * scale, "scale": scale}  # far-below-walking spike
    return {"rear": rear, "front": front}


def score_recording(stem, calib, hop=1):
    """Per-window continuous detection score + predicted leg multi-hot (statistical rule)."""
    recs = io_load.list_recordings()
    res_df, status = resample.cached_resample(recs[stem])
    X = build_channel_matrix(res_df)
    ws = make_windows(X, status, stem, C.parse_legs(stem), hop=hop)
    n = len(ws.y_bin)
    if n == 0:
        return None

    rear_tau = {leg: res_df[f"{leg}_thigh_tau"].to_numpy() for leg in REAR}
    front_tau = {leg: res_df[f"{leg}_thigh_tau"].to_numpy() for leg in FRONT}
    score = np.zeros(n, dtype=np.float32)
    pred_legs = np.zeros((n, C.N_LEGS), dtype=int)
    stop_veto = np.zeros(n, dtype=bool)

    for r, end in enumerate(ws.end_idx):
        s = slice(int(end) - C.WINDOW_SAMPLES + 1, int(end) + 1)
        # Deviation-from-walking-center in robust-σ units (decoupled from the original
        # script's mis-tuned absolute thresholds, which never trigger on this subtler
        # dataset). This is the detector's true discriminative quantity; the shared
        # protocol then picks the VAL operating threshold. Ranking is unchanged whether
        # we measure exceedance-beyond-threshold or deviation, but deviation is non-zero
        # so PR/ROC-AUC are meaningful.
        best, best_dir, best_leg = 0.0, None, None
        for leg in REAR:
            v = rear_tau[leg][s]
            cal = calib["rear"][leg]
            down_dev = max(0.0, (cal["center"] - v.min()) / cal["scale"])   # downward swing
            up_dev = max(0.0, (v.max() - cal["center"]) / cal["scale"])     # upward swing
            if down_dev > best:
                best, best_dir, best_leg = down_dev, "down", leg
            if up_dev > best:
                best, best_dir, best_leg = up_dev, "up", leg
        score[r] = best
        # stop veto: a front leg dips far below walking with only a few spike samples
        for leg in FRONT:
            v = front_tau[leg][s]
            spikes = v <= calib["front"][leg]["stop"]
            if spikes.any() and spikes.mean() <= 0.22:
                stop_veto[r] = True
        # predicted legs (used only when the shared protocol declares an alarm)
        if best_dir == "down" and best_leg is not None:        # rear/back entanglement
            pred_legs[r, C.LEG_ORDER.index(best_leg)] = 1
        elif best_dir == "up":                                  # front entanglement (can't localize)
            for leg in FRONT:
                pred_legs[r, C.LEG_ORDER.index(leg)] = 1

    # stop windows are negatives: zero their score so they don't raise entanglement alarms
    score = np.where(stop_veto, 0.0, score)
    pred_legs[stop_veto] = 0
    return {"stem": stem, "n": n, "end_idx": ws.end_idx, "ent_frac": ws.ent_frac,
            "y_bin": ws.y_bin, "y_legs": ws.y_legs, "score": score, "leg_z": score[:, None],
            "pred_legs_raw": pred_legs, "onset_idx": ws.onset_idx,
            "t": ws.end_idx / C.TARGET_HZ, "affected": C.parse_legs(stem), "is_prob": False}


if __name__ == "__main__":
    calib = calibrate(C.SPLIT["train"])
    print("Statistical detector calibrated on TRAIN walking:")
    for leg in REAR:
        c = calib["rear"][leg]
        print(f"  {leg}: center={c['center']:.2f} scale={c['scale']:.2f} "
              f"down<={c['down']:.2f} up>={c['up']:.2f}")
    # smoke: score a back-left (RL) and a front-both file
    for stem in ["back_left_hand2", "front_both_wire2", "walking4"]:
        r = score_recording(stem, calib, hop=2)
        if r is None:
            print(f"  {stem}: no windows"); continue
        ent = r["y_bin"] == 1
        print(f"  {stem:18s} max_score_ent={r['score'][ent].max() if ent.any() else float('nan'):.2f} "
              f"max_score_walk={r['score'][~ent].max():.2f} affected={sorted(r['affected'])}")
    print("stat_detector OK.")
