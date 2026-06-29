"""Complete v2 validation (no retraining here — see validate_v2_experiments.py).

Compares the v1 model (ml/artifacts/before_v1/) and the v2 model (ml/artifacts/) on:
  - the ORIGINAL 4 held-out test files: P/R/F1, PR-AUC, ROC-AUC, per-leg P/R/F1,
    exact-match, false-alarm rate, alarm latency  (clean apples-to-apples)
  - confusion matrices (saved as plots)
  - calibration: Brier, ECE, raw + calibrated thresholds
  - the 5 NEW recordings, phase-grouped firing, each annotated train/val/test
  - before/after replay plots for original-4 + 5 new -> artifacts/plots/validation_v2/

Writes artifacts/validation_v2_results.json and prints a summary.
Run:  python -m ml.validate_v2
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import config as C
from . import io_load, resample, evaluate, calibration as calib_mod
from .model import EntanglementTCN
from .normalize import Normalizer
from .intensity import IntensityCalibrator
from .windowing import make_windows
from .features import build_channel_matrix
from . import report as R

ORIG_TEST = ["back_left_hand2", "back_right_wire1", "front_left_hand2", "front_both_wire2"]
RAW_THR = 0.9999
LEG_THR = {"FR": 0.5, "FL": 0.5, "RR": 0.9, "RL": 0.5}
NEW_FILES = [
    ("go2_lowstate_lock_stop", "none", "TEST (held-out — UNBIASED)"),
    ("go2_lowstate_walk_stop_back", "none", "TRAIN (biased — used in training)"),
    ("go2_lowstate_entaglement_front_right_wire", "FR", "TRAIN (biased)"),
    ("go2_lowstate_entaglement_back_right_stop", "RR", "VAL (threshold/early-stop only)"),
    ("go2_lowstate_entaglement_back_right_intensity", "RR", "TRAIN (biased)"),
]
BEFORE_DIR = os.path.join(C.ARTIFACTS_DIR, "before_v1")
VAL_DIR = os.path.join(C.PLOTS_DIR, "validation_v2")


def load_from(d, device):
    ckpt = torch.load(os.path.join(d, "model.pt"), map_location=device, weights_only=False)
    model = EntanglementTCN(in_channels=ckpt["config"]["n_channels"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return (model, Normalizer.load(os.path.join(d, "normalize.json")),
            IntensityCalibrator.load(os.path.join(d, "intensity_calib.json")))


def window_status(stem):
    res_df, status = resample.cached_resample(io_load.list_recordings()[stem])
    X = build_channel_matrix(res_df)
    ws = make_windows(X, status, stem, C.parse_legs(stem), hop=1)
    doms = []
    for end in ws.end_idx:
        sl = slice(int(end) - C.WINDOW_SAMPLES + 1, int(end) + 1)
        u, c = np.unique(status[sl].astype(str), return_counts=True)
        doms.append(u[int(np.argmax(c))])
    return np.array(doms, dtype=object)


def orig4_metrics(model, norm, calib, device):
    recs = io_load.list_recordings()
    res = {s: evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=1) for s in ORIG_TEST}
    y = np.concatenate([res[s]["y_bin"] for s in ORIG_TEST])
    p = np.concatenate([res[s]["p_bin"] for s in ORIG_TEST])
    yl = np.concatenate([res[s]["y_legs"] for s in ORIG_TEST])
    pl = np.concatenate([(res[s]["p_legs"] >= 0.5).astype(int) for s in ORIG_TEST])
    bm = R.binary_metrics(y, p, RAW_THR); ap, roc = R.aucs(y, p)
    lm = R.leg_metrics(yl, pl, y)
    # latency per file (persistence-gated) + FAR on negative (walking-prefix) windows
    lat = []
    for s in ORIG_TEST:
        l = R.latency_persist(res[s]["onset_idx"], res[s]["end_idx"], res[s]["p_bin"], RAW_THR)
        if l is not None:
            lat.append(l)
    neg = y == 0
    far = float((p[neg] >= RAW_THR).mean())
    return {"bm": bm, "PR_AUC": ap, "ROC_AUC": roc, "leg": lm,
            "median_latency_ms": float(np.median(lat) * 1000) if lat else None,
            "far_prefix": far}, res


def calibration_metrics(d, device):
    model, norm, calib = load_from(d, device)
    recs = io_load.list_recordings()
    val_files = [s for s in C.SPLIT["val"]]  # current split for v2; for v1 use its own? use orig val
    # Use each model's TEST-set logits for Brier/ECE on the ORIGINAL 4 (comparable set)
    tl = np.concatenate([evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=1)["bin_logit"]
                         for s in ORIG_TEST])
    ty = np.concatenate([evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=1)["y_bin"]
                         for s in ORIG_TEST])
    # temperature fit on the model's VAL logits
    vlog = np.concatenate([evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=5)["bin_logit"]
                           for s in C.SPLIT["val"] if evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=5)["n"]])
    vy = np.concatenate([evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=5)["y_bin"]
                         for s in C.SPLIT["val"] if evaluate.dense_infer(s, recs[s], model, norm, calib, device, hop=5)["n"]])
    T = calib_mod.fit_temperature(vlog, vy)
    p_raw = 1 / (1 + np.exp(-tl)); p_cal = calib_mod.apply_temperature(tl, T)
    return {"T": float(T), "brier_raw": calib_mod.brier(p_raw, ty), "ece_raw": calib_mod.ece(p_raw, ty),
            "brier_cal": calib_mod.brier(p_cal, ty), "ece_cal": calib_mod.ece(p_cal, ty),
            "sat_raw_pct": float((p_raw >= 0.999999).mean() * 100)}


def confusion_plot(bm_v1, bm_v2, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (tag, bm) in zip(axes, [("v1", bm_v1), ("v2", bm_v2)]):
        M = np.array([[bm["TP"], bm["FN"]], [bm["FP"], bm["TN"]]])
        ax.imshow(M, cmap="Blues")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                    color="white" if v > M.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred +", "pred −"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["actual +", "actual −"])
        ax.set_title("{}  (F1={:.3f})".format(tag, bm["F1"]))
    fig.suptitle("Confusion matrix — original-4 test (window-level @ {:.4f})".format(RAW_THR))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def phase_fire(model, norm, calib, device, stem, aff):
    r = evaluate.dense_infer(stem, io_load.list_recordings()[stem], model, norm, calib, device, hop=1)
    if r["n"] == 0:
        return {}
    doms = window_status(stem)[:r["n"]]
    out = {}
    for ph in ["Walking", "Stop", "Lock", "Entangled"]:
        m = doms == ph
        if not m.any():
            continue
        fire = float((r["p_bin"][:len(doms)][m] >= RAW_THR).mean()) * 100
        d = {"n": int(m.sum()), "fire_pct": round(fire, 1)}
        if aff != "none":
            j = C.LEG_ORDER.index(aff)
            d["aff_leg_pct"] = round(float((r["p_legs"][:len(doms)][m][:, j] >= LEG_THR[aff]).mean()) * 100, 1)
        out[ph] = d
    return out


def main():
    os.makedirs(VAL_DIR, exist_ok=True)
    device = evaluate._device()
    v1 = load_from(BEFORE_DIR, device)
    v2 = load_from(C.ARTIFACTS_DIR, device)

    print("=== ORIGINAL-4 held-out test: v1 vs v2 ===")
    m1, res1 = orig4_metrics(*v1, device)
    m2, res2 = orig4_metrics(*v2, device)
    def line(tag, m):
        b = m["bm"]
        print("  {:3s} P={:.3f} R={:.3f} F1={:.3f} PR-AUC={:.3f} ROC-AUC={:.3f} exact={:.3f} "
              "FARpre={:.1f}% lat={}".format(tag, b["P"], b["R"], b["F1"], m["PR_AUC"], m["ROC_AUC"],
              m["leg"]["exact_match"], m["far_prefix"]*100,
              ("%.0fms" % m["median_latency_ms"]) if m["median_latency_ms"] is not None else "n/a"))
    line("v1", m1); line("v2", m2)
    print("  per-leg F1:  " + "  ".join("{}: v1={:.2f}/v2={:.2f}".format(
        leg, m1["leg"][leg]["F1"], m2["leg"][leg]["F1"]) for leg in C.LEG_ORDER))
    print("  RR precision v1={:.2f} v2={:.2f} | FR precision v1={:.2f} v2={:.2f}".format(
        m1["leg"]["RR"]["P"], m2["leg"]["RR"]["P"], m1["leg"]["FR"]["P"], m2["leg"]["FR"]["P"]))

    confusion_plot(m1["bm"], m2["bm"], os.path.join(VAL_DIR, "confusion_v1_v2_orig4.png"))
    print("  saved confusion_v1_v2_orig4.png")

    print("\n=== Calibration (orig-4 logits; temperature fit on each model's VAL) ===")
    cal1 = calibration_metrics(BEFORE_DIR, device)
    cal2 = calibration_metrics(C.ARTIFACTS_DIR, device)
    for tag, c in [("v1", cal1), ("v2", cal2)]:
        print("  {}: T={:.2f} Brier raw={:.4f}->cal={:.4f}  ECE raw={:.4f}->cal={:.4f}  sat@1={:.0f}%".format(
            tag, c["T"], c["brier_raw"], c["brier_cal"], c["ece_raw"], c["ece_cal"], c["sat_raw_pct"]))

    print("\n=== NEW recordings phase-firing: v1 vs v2 (split annotated) ===")
    new_results = {}
    for stem, aff, split in NEW_FILES:
        f1 = phase_fire(*v1, device, stem, aff)
        f2 = phase_fire(*v2, device, stem, aff)
        new_results[stem] = {"split": split, "affected": aff, "v1": f1, "v2": f2}
        print("  {}  [{}]".format(stem, split))
        for ph in ["Walking", "Stop", "Lock", "Entangled"]:
            if ph in f2:
                a1 = f1.get(ph, {}); a2 = f2.get(ph, {})
                extra = ""
                if "aff_leg_pct" in a2:
                    extra = "  {}leg v1={}%/v2={}%".format(aff, a1.get("aff_leg_pct", "-"), a2.get("aff_leg_pct"))
                print("     {:9s} fire% v1={:>5}/v2={:>5}{}".format(
                    ph, a1.get("fire_pct", "-"), a2.get("fire_pct"), extra))

    print("\n=== Replay plots (before/after) -> {} ===".format(os.path.relpath(VAL_DIR, C.PROJECT_DIR)))
    plot_stems = ORIG_TEST + [s for s, _, _ in NEW_FILES]
    for stem in plot_stems:
        for tag, (model, norm, calib) in [("v1", v1), ("v2", v2)]:
            rr = R.tcn_scores(stem, model, norm, calib, device, hop=1)
            if rr is not None:
                R.plot_replay_full(rr, RAW_THR, VAL_DIR, kind=tag)
    print("  wrote {} replay plots".format(len(plot_stems) * 2))

    # persist
    out = {"orig4": {"v1": {"P": m1["bm"]["P"], "R": m1["bm"]["R"], "F1": m1["bm"]["F1"],
                            "PR_AUC": m1["PR_AUC"], "ROC_AUC": m1["ROC_AUC"],
                            "exact_match": m1["leg"]["exact_match"], "far_prefix": m1["far_prefix"],
                            "latency_ms": m1["median_latency_ms"],
                            "leg": {l: m1["leg"][l] for l in C.LEG_ORDER}, "confusion": m1["bm"]},
                     "v2": {"P": m2["bm"]["P"], "R": m2["bm"]["R"], "F1": m2["bm"]["F1"],
                            "PR_AUC": m2["PR_AUC"], "ROC_AUC": m2["ROC_AUC"],
                            "exact_match": m2["leg"]["exact_match"], "far_prefix": m2["far_prefix"],
                            "latency_ms": m2["median_latency_ms"],
                            "leg": {l: m2["leg"][l] for l in C.LEG_ORDER}, "confusion": m2["bm"]}},
           "calibration": {"v1": cal1, "v2": cal2}, "new_files": new_results}
    with open(os.path.join(C.ARTIFACTS_DIR, "validation_v2_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved validation_v2_results.json")


if __name__ == "__main__":
    main()
