"""Retraining experiments to separate GENUINE GENERALIZATION from MEMORIZATION.

(1) ABLATION — retrain with the new data REMOVED (v1 split, current code). Isolates
    the effect of the new DATA from any code/seed change. Expectation: the lock/stop
    false alarms come BACK (proving the fix is the data, not code).

(2) LEAVE-ONE-NEW-RECORDING-OUT (LONRO) — for each new recording that is in TRAIN,
    retrain with it HELD OUT and evaluate on it. Gives an UNBIASED estimate of whether
    the fix generalizes to that recording (vs memorizing it). `lock_stop` is already in
    TEST (never trained on) so the shipped model's result on it is already unbiased.

Run:  python -m ml.validate_v2_experiments
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C
from . import io_load, evaluate
from . import report as R
from .normalize import fit_from_train
from .intensity import IntensityCalibrator
from .train import train_one
from .validate_v2 import phase_fire, window_status, ORIG_TEST, RAW_THR

V1_TRAIN = ["back_both_wire1", "back_left_hand1", "back_left_wire1", "back_right_hand1",
            "front_both_wire1", "front_left_hand1", "stop1", "stop2", "stop3",
            "walking1", "walking2", "walking3", "walking5"]
V1_VAL = ["back_both_wire2", "back_right_hand2", "walking4"]
EPOCHS = 30


def retrain(train_stems, val_stems, device, epochs=EPOCHS):
    norm = fit_from_train(train_stems)
    calib = IntensityCalibrator.fit([s for s in train_stems if not C.parse_legs(s)] or train_stems)
    model, _, _ = train_one(train_stems, val_stems, norm, calib, device, epochs=epochs, verbose=False)
    return model, norm, calib


def orig4(model, norm, calib, device):
    recs = io_load.list_recordings()
    y = np.concatenate([evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=1)["y_bin"] for s in ORIG_TEST])
    p = np.concatenate([evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=1)["p_bin"] for s in ORIG_TEST])
    bm = R.binary_metrics(y, p, RAW_THR); ap, roc = R.aucs(y, p)
    return {"F1": bm["F1"], "PR_AUC": ap, "ROC_AUC": roc}


def main():
    device = evaluate._device()
    C.SEED = 1234
    out = {}

    # ---------------- (1) ABLATION: remove all new data ----------------
    print("=== (1) ABLATION: retrain WITHOUT the 5 new recordings (v1 split, current code) ===")
    ab_model, ab_norm, ab_calib = retrain(V1_TRAIN, V1_VAL, device)
    ab_o4 = orig4(ab_model, ab_norm, ab_calib, device)
    print("  orig-4: F1={:.3f} PR-AUC={:.3f} ROC-AUC={:.3f}".format(ab_o4["F1"], ab_o4["PR_AUC"], ab_o4["ROC_AUC"]))
    ab_new = {}
    for stem, aff, _ in [("go2_lowstate_lock_stop", "none", ""),
                         ("go2_lowstate_walk_stop_back", "none", ""),
                         ("go2_lowstate_entaglement_front_right_wire", "FR", ""),
                         ("go2_lowstate_entaglement_back_right_intensity", "RR", "")]:
        pf = phase_fire(ab_model, ab_norm, ab_calib, device, stem, aff)
        ab_new[stem] = pf
        key = "Entangled" if aff != "none" else "Lock"
        cell = pf.get(key, {})
        print("  no-new-data on {:42s} {} fire%={}  {}".format(
            stem, key, cell.get("fire_pct", "-"),
            ("FR/RRleg%=" + str(cell.get("aff_leg_pct", "-"))) if aff != "none" else ""))
    out["ablation_no_new_data"] = {"orig4": ab_o4, "new_files": ab_new}

    # ---------------- (2) LONRO: hold out each TRAIN new recording ----------------
    print("\n=== (2) Leave-one-new-recording-out (retrain held out; UNBIASED on that file) ===")
    lonro = {}
    cur_train = list(C.SPLIT["train"]); cur_val = list(C.SPLIT["val"])
    targets = [("go2_lowstate_walk_stop_back", "none", ["Walking", "Lock"]),
               ("go2_lowstate_entaglement_front_right_wire", "FR", ["Entangled"]),
               ("go2_lowstate_entaglement_back_right_intensity", "RR", ["Lock", "Entangled"])]
    for held, aff, phases in targets:
        tr = [s for s in cur_train if s != held]
        m, n, c = retrain(tr, cur_val, device)
        pf = phase_fire(m, n, c, device, held, aff)
        o4 = orig4(m, n, c, device)
        lonro[held] = {"affected": aff, "phase_fire": pf, "orig4_F1": o4["F1"]}
        cells = {ph: pf.get(ph, {}) for ph in phases}
        print("  held-out {:42s} (orig4 F1={:.3f}):".format(held, o4["F1"]))
        for ph in phases:
            cc = cells[ph]
            extra = ("  {}leg%={}".format(aff, cc.get("aff_leg_pct", "-"))) if aff != "none" else ""
            print("       {:9s} fire%={}{}".format(ph, cc.get("fire_pct", "-"), extra))
    out["lonro"] = lonro
    # lock_stop is already TEST (held-out) in the shipped model — note it
    out["lock_stop_note"] = "lock_stop is in SPLIT['test']; shipped v2 result on it is already held-out/unbiased."

    with open(os.path.join(C.ARTIFACTS_DIR, "validation_v2_experiments.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved validation_v2_experiments.json")


if __name__ == "__main__":
    main()
