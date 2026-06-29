"""Feature ablation study, identical protocol, no architecture change.

Configs (channel set fed to the SAME TCN):
  raw_only         : 48  (motors+foot+IMU, no engineered)
  engineered_only  : 12  (per-leg thigh_tau_down, thigh_down_effort, tau_sum)
  raw+engineered   : 60  (shipped default)
  no_imu           : 52  (drop 8 IMU channels)
  no_foot          : 56  (drop 4 foot channels)

For each: fit normalizer+calibrator on TRAIN, train the fixed split, score TEST
(P/R/F1 @ VAL-1%FAR threshold, PR-AUC, ROC-AUC, leg-macro-F1, exact-match), then
run LORO (mean±std detection F1). Results appended to artifacts/ablation.json.

Run:
    python -m dataset.ml.ablation            # all 5 configs (use a background run; ~20 min)
    python -m dataset.ml.ablation raw_only   # a single config
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import config as C
from . import io_load, evaluate
from .report import select_threshold, binary_metrics, aucs, leg_metrics, TARGET_FAR, LEG_THR
from .train import train_one
from .normalize import fit_from_train
from .intensity import IntensityCalibrator

CONFIGS = {
    "raw_only":        dict(INCLUDE_FOOT=True,  INCLUDE_IMU=True,  USE_ENGINEERED=False, ENGINEERED_ONLY=False),
    "engineered_only": dict(INCLUDE_FOOT=True,  INCLUDE_IMU=True,  USE_ENGINEERED=False, ENGINEERED_ONLY=True),
    "raw+engineered":  dict(INCLUDE_FOOT=True,  INCLUDE_IMU=True,  USE_ENGINEERED=True,  ENGINEERED_ONLY=False),
    "no_imu":          dict(INCLUDE_FOOT=True,  INCLUDE_IMU=False, USE_ENGINEERED=True,  ENGINEERED_ONLY=False),
    "no_foot":         dict(INCLUDE_FOOT=False, INCLUDE_IMU=True,  USE_ENGINEERED=True,  ENGINEERED_ONLY=False),
}
EPOCHS = 20


def _apply(flags):
    C.INCLUDE_FOOT = flags["INCLUDE_FOOT"]
    C.INCLUDE_IMU = flags["INCLUDE_IMU"]
    C.USE_ENGINEERED = flags["USE_ENGINEERED"]
    C.ENGINEERED_ONLY = flags["ENGINEERED_ONLY"]


def _restore():
    _apply(CONFIGS["raw+engineered"])


def eval_fixed(model, normalizer, calibrator, device):
    recs = io_load.list_recordings()
    val = [evaluate.dense_infer(s, recs[s], model, normalizer, calibrator, device, hop=5)
           for s in C.SPLIT["val"]]
    val = [r for r in val if r["n"]]
    vneg = np.concatenate([r["p_bin"][r["y_bin"] == 0] for r in val])
    thr = select_threshold(vneg, TARGET_FAR, is_prob=True)
    test = [evaluate.dense_infer(s, recs[s], model, normalizer, calibrator, device, hop=1)
            for s in C.SPLIT["test"]]
    y = np.concatenate([r["y_bin"] for r in test]); p = np.concatenate([r["p_bin"] for r in test])
    yl = np.concatenate([r["y_legs"] for r in test])
    pl = np.concatenate([(r["p_legs"] >= LEG_THR).astype(int) for r in test])
    bm = binary_metrics(y, p, thr); ap, roc = aucs(y, p)
    lm = leg_metrics(yl, pl, y)
    leg_macro = float(np.mean([lm[l]["F1"] for l in C.LEG_ORDER]))
    return {"thr": thr, "F1": bm["F1"], "P": bm["P"], "R": bm["R"],
            "PR_AUC": ap, "ROC_AUC": roc, "leg_macro_F1": leg_macro,
            "exact_match": lm["exact_match"]}


def eval_loro(device, epochs=EPOCHS):
    recs_all = list(C.SPLIT["train"]) + list(C.SPLIT["val"]) + list(C.SPLIT["test"])
    f1s = []
    for held in C.POSITIVE_FILES:
        train_stems = [s for s in recs_all if s != held]
        walk = [s for s in train_stems if not C.parse_legs(s)] or train_stems
        norm = fit_from_train(train_stems)
        calib = IntensityCalibrator.fit(walk)
        model, _, _ = train_one(train_stems, None, norm, calib, device, epochs=epochs, verbose=False)
        r = evaluate.dense_infer(held, io_load.list_recordings()[held], model, norm, calib, device, hop=1)
        neg = r["p_bin"][r["y_bin"] == 0]
        thr = select_threshold(neg, TARGET_FAR, is_prob=True)
        f1s.append(binary_metrics(r["y_bin"], r["p_bin"], thr)["F1"])
    return float(np.mean(f1s)), float(np.std(f1s)), f1s


def run_config(name, with_loro=True):
    device = evaluate._device()
    _apply(CONFIGS[name])
    nch = C.n_channels()
    print(f"\n=== ablation: {name}  (C={nch}) ===")
    normalizer = fit_from_train(C.SPLIT["train"])
    calibrator = IntensityCalibrator.fit([s for s in C.SPLIT["train"] if not C.parse_legs(s)])
    model, best, _ = train_one(C.SPLIT["train"], C.SPLIT["val"], normalizer, calibrator,
                               device, epochs=EPOCHS, verbose=False)
    fixed = eval_fixed(model, normalizer, calibrator, device)
    print(f"  fixed: F1={fixed['F1']:.3f} PR-AUC={fixed['PR_AUC']:.3f} ROC-AUC={fixed['ROC_AUC']:.3f} "
          f"leg_macroF1={fixed['leg_macro_F1']:.3f}")
    rec = {"n_channels": nch, "fixed": fixed}
    if with_loro:
        m, s, f1s = eval_loro(device)
        rec["loro_mean_F1"], rec["loro_std_F1"], rec["loro_folds"] = m, s, f1s
        print(f"  LORO: F1={m:.3f} ± {s:.3f}")
    _restore()

    # persist incrementally
    path = os.path.join(C.ARTIFACTS_DIR, "ablation.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[name] = rec
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return rec


if __name__ == "__main__":
    names = [sys.argv[1]] if len(sys.argv) > 1 else list(CONFIGS.keys())
    for nm in names:
        run_config(nm, with_loro=True)
    print("\nablation.json updated.")
