"""Apples-to-apples BEFORE vs AFTER comparison on the ORIGINAL 4 test files.

The fixed-split pooled metric in report.py is confounded because the v2 test set
added `lock_stop` (a pure-negative file). This script evaluates BOTH models on the
EXACT same original 4 entanglement test recordings with the same protocol, so the
regression check is clean. Threshold-free metrics (PR-AUC, ROC-AUC) are the primary
read; F1 is at the shared raw operating threshold 0.9999.

Run:  python -m ml.compare_before_after
"""
from __future__ import annotations

import os

import numpy as np
import torch

from . import config as C
from . import io_load, evaluate
from .model import EntanglementTCN
from .normalize import Normalizer
from .intensity import IntensityCalibrator

ORIG_TEST = ["back_left_hand2", "back_right_wire1", "front_left_hand2", "front_both_wire2"]
RAW_THR = 0.9999
BEFORE_DIR = os.path.join(C.ARTIFACTS_DIR, "before_v1")
AFTER_DIR = C.ARTIFACTS_DIR


def load_from(d, device):
    ckpt = torch.load(os.path.join(d, "model.pt"), map_location=device, weights_only=False)
    model = EntanglementTCN(in_channels=ckpt["config"]["n_channels"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    norm = Normalizer.load(os.path.join(d, "normalize.json"))
    calib = IntensityCalibrator.load(os.path.join(d, "intensity_calib.json"))
    return model, norm, calib


def prf(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def aucs(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def eval_model(tag, d, device):
    model, norm, calib = load_from(d, device)
    recs = io_load.list_recordings()
    Y, P, YL, PL = [], [], [], []
    for s in ORIG_TEST:
        r = evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=1)
        Y.append(r["y_bin"]); P.append(r["p_bin"]); YL.append(r["y_legs"]); PL.append(r["p_legs"])
    y = np.concatenate(Y); p = np.concatenate(P)
    yl = np.concatenate(YL); pl = np.concatenate(PL)
    Pp, Rr, F1 = prf(y, p, RAW_THR)
    ap, roc = aucs(y, p)
    leg_thr = {"FR": 0.5, "FL": 0.5, "RR": 0.9, "RL": 0.5}
    legf = {}
    for j, leg in enumerate(C.LEG_ORDER):
        lp, lr, lf = prf(yl[:, j], pl[:, j], leg_thr[leg])
        legf[leg] = (lp, lr, lf)
    return {"P": Pp, "R": Rr, "F1": F1, "PR_AUC": ap, "ROC_AUC": roc, "legf": legf}


def main():
    device = evaluate._device()
    print("BEFORE vs AFTER on the ORIGINAL 4 test files (identical protocol, thr={:.4f})\n".format(RAW_THR))
    b = eval_model("before", BEFORE_DIR, device)
    a = eval_model("after", AFTER_DIR, device)
    print("{:10s} {:>7s} {:>7s} {:>7s} {:>8s} {:>9s}".format("model", "P", "R", "F1", "PR-AUC", "ROC-AUC"))
    for tag, m in [("BEFORE", b), ("AFTER", a)]:
        print("{:10s} {:7.3f} {:7.3f} {:7.3f} {:8.3f} {:9.3f}".format(
            tag, m["P"], m["R"], m["F1"], m["PR_AUC"], m["ROC_AUC"]))
    print("\nper-leg F1 (P/R/F1):")
    for leg in C.LEG_ORDER:
        bl = b["legf"][leg]; al = a["legf"][leg]
        print("  {}: BEFORE F1={:.3f}(P{:.2f}/R{:.2f})  AFTER F1={:.3f}(P{:.2f}/R{:.2f})".format(
            leg, bl[2], bl[0], bl[1], al[2], al[0], al[1]))


if __name__ == "__main__":
    main()
