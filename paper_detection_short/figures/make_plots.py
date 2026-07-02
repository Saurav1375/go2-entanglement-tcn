#!/usr/bin/env python3
"""Generate all publication figures for the paper from REAL repository data.

Sources (no fabricated numbers):
  - ml/artifacts/report_metrics.json      (v2 test-split detection + per-leg + 15-fold LORO)
  - ml/artifacts/validation_v2_results.json (v1 vs v2 before/after, calibration, field issues)
  - ml/artifacts/ablation.json            (feature-set ablation; v1 study)
  - ml/artifacts/operating_point.json     (temperature, thresholds)
  - a read-only model run (ml.evaluate)   (ROC / PR / reliability curves on the test split)
  - a read-only ONNX engine run           (per-window latency on the robot runtime)

Run from the repository root:  python paper/figures/make_figures.py
Outputs vector PDFs into paper/figures/.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ART = os.path.join(REPO, "ml", "artifacts")
OUT = HERE

# ------------------------------------------------------------------ style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL = {"tcn": "#3C7A3C", "heur": "#B5701A", "v1": "#9AA7BD", "v2": "#3B5B92",
       "data": "#3B5B92", "accent": "#5B4B9E", "rec": "#B5701A", "neg": "#C0632A"}
W1, W2 = 3.45, 7.16   # single- / double-column widths (inches)


def load(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)


def save(fig, stem):
    p = os.path.join(OUT, stem + ".pdf")
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, REPO))


# ================================================================== 1. dataset
def fig_dataset():
    # positive recordings per affected leg (parse_legs semantics) + negatives
    per_leg = {"FR": 3, "FL": 4, "RR": 7, "RL": 5}   # traceable from split.json + filename table
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    legs = list(per_leg.keys())
    ax[0].bar(legs, [per_leg[l] for l in legs], color=COL["v2"], width=0.62)
    ax[0].axhline(np.mean(list(per_leg.values())), ls="--", lw=0.8, color="#888",
                  label="mean = %.1f" % np.mean(list(per_leg.values())))
    for i, l in enumerate(legs):
        ax[0].text(i, per_leg[l] + 0.08, str(per_leg[l]), ha="center", va="bottom", fontsize=8)
    ax[0].set_ylabel("positive recordings")
    ax[0].set_title("(a) Recordings per affected leg")
    ax[0].set_ylim(0, 8)
    ax[0].legend(frameon=False, loc="upper left")

    # split composition (recordings), pos vs neg
    splits = ["train (16)", "val (4)", "test (5)"]
    pos = [8, 3, 4]
    neg = [8, 1, 1]
    x = np.arange(len(splits))
    ax[1].bar(x, pos, width=0.55, color=COL["tcn"], label="positive")
    ax[1].bar(x, neg, width=0.55, bottom=pos, color=COL["heur"], label="negative")
    for i in range(len(splits)):
        ax[1].text(i, pos[i] / 2, str(pos[i]), ha="center", va="center", color="white", fontsize=8)
        ax[1].text(i, pos[i] + neg[i] / 2, str(neg[i]), ha="center", va="center", color="white", fontsize=8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(splits)
    ax[1].set_ylabel("recordings")
    ax[1].set_title("(b) Leakage-safe split (grouped by recording)")
    ax[1].legend(frameon=False, loc="upper right")
    save(fig, "dataset_distribution")


# ================================================================== 2. LORO folds
def fig_loro(rm):
    folds = rm["loro"]["folds"]
    names = [f["held"].replace("go2_lowstate_", "").replace("entaglement_", "")
             for f in folds]
    f1 = [f["F1"] for f in folds]
    order = np.argsort(f1)
    names = [names[i] for i in order]
    f1 = [f1[i] for i in order]
    m, s = rm["loro"]["mean_F1"], rm["loro"]["std_F1"]
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    colors = [COL["neg"] if v < m - s else COL["v2"] for v in f1]
    ax.bar(range(len(f1)), f1, color=colors, width=0.7)
    ax.axhline(m, color="#333", lw=1.1, label="mean F1 = %.3f" % m)
    ax.axhspan(m - s, m + s, color="#333", alpha=0.10, label="$\\pm$1 s.d. (%.3f)" % s)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6.6)
    ax.set_ylabel("detection F1 (held-out fold)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Leave-one-recording-out cross-validation (15 folds)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "loro_folds")


# ================================================================== 3. per-leg
def fig_perleg(rm):
    legs = ["FR", "FL", "RR", "RL"]
    P = [rm["leg"][l]["P"] for l in legs] if False else [rm["fixed_split"]["leg"][l]["P"] for l in legs]
    R = [rm["fixed_split"]["leg"][l]["R"] for l in legs]
    F = [rm["fixed_split"]["leg"][l]["F1"] for l in legs]
    x = np.arange(len(legs)); w = 0.26
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.bar(x - w, P, w, label="Precision", color=COL["accent"])
    ax.bar(x, R, w, label="Recall", color=COL["tcn"])
    ax.bar(x + w, F, w, label="F1", color=COL["v2"])
    ax.set_xticks(x); ax.set_xticklabels(legs)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-leg attribution (test split)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    save(fig, "per_leg_metrics")


# ================================================================== 4. detector vs baseline
def fig_compare(rm):
    d = rm["fixed_split"]["detection"]; h = rm["heuristic"]
    metrics = ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
    tcn = [d["P"], d["R"], d["F1"], d["PR_AUC"], d["ROC_AUC"]]
    heu = [h["P"], h["R"], h["F1"], h["PR_AUC"], h["ROC_AUC"]]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.bar(x - w / 2, tcn, w, label="Causal TCN", color=COL["tcn"])
    ax.bar(x + w / 2, heu, w, label="Rule-based (thigh torque)", color=COL["heur"])
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=18, ha="right")
    ax.set_ylabel("score"); ax.set_ylim(0, 1.05)
    ax.set_title("Detection: TCN vs. rule-based baseline")
    ax.legend(frameon=False, loc="upper center")
    save(fig, "detection_comparison")


# ================================================================== 5. confusion
def fig_confusion(rm):
    d = rm["fixed_split"]["detection"]
    cm = np.array([[d["TP"], d["FN"]], [d["FP"], d["TN"]]], dtype=float)
    row = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    im = ax.imshow(row, cmap="Greens", vmin=0, vmax=1)
    labels = ["Entangled", "Not entangled"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Detection confusion matrix (test split)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "%d\n(%.1f%%)" % (cm[i, j], 100 * row[i, j]),
                    ha="center", va="center",
                    color="white" if row[i, j] > 0.5 else "#222", fontsize=8.5)
    ax.grid(False)
    save(fig, "confusion_matrix")


# ================================================================== 6. ablation
def fig_ablation(ab):
    order = ["engineered_only", "raw_only", "no_foot", "no_imu", "raw+engineered"]
    labels = {"engineered_only": "engineered\nonly (12)", "raw_only": "raw only (48)",
              "no_foot": "no foot (56)", "no_imu": "no IMU (52)",
              "raw+engineered": "raw+eng (60)\n[shipped]"}
    m = [ab[k]["loro_mean_F1"] for k in order]
    s = [ab[k]["loro_std_F1"] for k in order]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    colors = [COL["accent"]] * (len(order) - 1) + [COL["v2"]]
    ax.bar(range(len(order)), m, yerr=s, capsize=3, color=colors, width=0.66,
           error_kw=dict(lw=0.8, ecolor="#555"))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[k] for k in order], fontsize=6.8)
    ax.set_ylabel("LORO F1 (mean $\\pm$ s.d.)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Feature-set ablation")
    save(fig, "ablation")


# ================================================================== 7. field issues (v1 vs v2)
def fig_fieldissues(vr):
    nf = vr["new_files"]
    # false-firing (lower better)
    fp_labels = ["Lock\n(lock_stop)", "Lock\n(walk_stop_back)", "RR while\nstopped*", "backward\nwalking"]
    v1 = [nf["go2_lowstate_lock_stop"]["v1"]["Lock"]["fire_pct"],
          nf["go2_lowstate_walk_stop_back"]["v1"]["Lock"]["fire_pct"],
          nf["go2_lowstate_entaglement_back_right_intensity"]["v1"]["Lock"]["fire_pct"],
          nf["go2_lowstate_walk_stop_back"]["v1"]["Walking"]["fire_pct"]]
    v2 = [nf["go2_lowstate_lock_stop"]["v2"]["Lock"]["fire_pct"],
          nf["go2_lowstate_walk_stop_back"]["v2"]["Lock"]["fire_pct"],
          nf["go2_lowstate_entaglement_back_right_intensity"]["v2"]["Lock"]["fire_pct"],
          nf["go2_lowstate_walk_stop_back"]["v2"]["Walking"]["fire_pct"]]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    x = np.arange(len(fp_labels)); w = 0.38
    ax[0].bar(x - w / 2, v1, w, label="v1", color=COL["v1"])
    ax[0].bar(x + w / 2, v2, w, label="v2 (retrained)", color=COL["v2"])
    for i in range(len(x)):
        ax[0].text(x[i] - w / 2, v1[i] + 1.5, "%.1f" % v1[i], ha="center", fontsize=6.6)
        ax[0].text(x[i] + w / 2, v2[i] + 1.5, "%.1f" % v2[i], ha="center", fontsize=6.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels(fp_labels, fontsize=6.8)
    ax[0].set_ylabel("false-fire rate (%)")
    ax[0].set_title("(a) False firing (lower is better)")
    ax[0].set_ylim(0, 108)
    ax[0].legend(frameon=False, loc="upper right")

    # FR detection (higher better): affected-leg fire on the dedicated FR file
    fr = nf["go2_lowstate_entaglement_front_right_wire"]
    a1 = fr["v1"]["Entangled"]["aff_leg_pct"]; a2 = fr["v2"]["Entangled"]["aff_leg_pct"]
    b1 = fr["v1"]["Entangled"]["fire_pct"]; b2 = fr["v2"]["Entangled"]["fire_pct"]
    x2 = np.arange(2)
    ax[1].bar(x2 - w / 2, [b1, a1], w, label="v1", color=COL["v1"])
    ax[1].bar(x2 + w / 2, [b2, a2], w, label="v2 (retrained)", color=COL["tcn"])
    for i, (u, v) in enumerate([(b1, b2), (a1, a2)]):
        ax[1].text(x2[i] - w / 2, u + 1.5, "%.0f" % u, ha="center", fontsize=6.8)
        ax[1].text(x2[i] + w / 2, v + 1.5, "%.0f" % v, ha="center", fontsize=6.8)
    ax[1].set_xticks(x2); ax[1].set_xticklabels(["binary\nfire", "FR-leg\nfire"], fontsize=7.2)
    ax[1].set_ylabel("detection rate (%)")
    ax[1].set_title("(b) Front-right detection (higher is better)")
    ax[1].set_ylim(0, 112)
    ax[1].legend(frameon=False, loc="upper left")
    save(fig, "field_issues")


# ================================================================== 8. model-run curves
def fig_model_curves(op):
    """ROC, PR, reliability curves on the held-out test split (read-only model run)."""
    try:
        sys.path.insert(0, REPO)
        from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
        import ml.config as C
        from ml.evaluate import load_artifacts, dense_infer
        dev = "cpu"
        model, norm, calib = load_artifacts(dev)
        ys, ps, logits = [], [], []
        for stem in ["back_left_hand2", "back_right_wire1", "front_left_hand2",
                     "front_both_wire2", "go2_lowstate_lock_stop"]:
            path = os.path.join(C.CSV_LABELLED_DIR, stem + ".csv")
            res = dense_infer(stem, path, model, norm, calib, dev)
            if not res.get("n"):
                continue
            ys.append(res["y_bin"]); ps.append(res["p_bin"]); logits.append(res["bin_logit"])
        y = np.concatenate(ys); p = np.concatenate(ps); lg = np.concatenate(logits)
    except Exception as exc:  # pragma: no cover
        print("  [skip] model-run curves:", exc)
        return

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    ax.plot(fpr, tpr, color=COL["tcn"], lw=1.6, label="TCN (AUC = %.3f)" % auc)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999", label="chance")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("ROC (test split)"); ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    save(fig, "roc_curve")

    # PR
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    base = y.mean()
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    ax.plot(rec, prec, color=COL["v2"], lw=1.6, label="TCN (AP = %.3f)" % ap)
    ax.axhline(base, ls="--", lw=0.8, color="#999", label="prevalence = %.2f" % base)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-Recall (test split)"); ax.legend(frameon=False, loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    save(fig, "pr_curve")

    # reliability (raw vs temperature-calibrated)
    T = op["temperature"]
    p_cal = 1.0 / (1.0 + np.exp(-lg / T))

    def rel(prob, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        xs, ys_ = [], []
        for i in range(bins):
            m = (prob >= edges[i]) & (prob < edges[i + 1] if i < bins - 1 else prob <= edges[i + 1])
            if m.sum() > 0:
                xs.append(prob[m].mean()); ys_.append(y[m].mean())
        return np.array(xs), np.array(ys_)
    xr, yr = rel(p)
    xc, yc = rel(p_cal)
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999", label="perfect")
    ax.plot(xr, yr, "o-", ms=3, lw=1.3, color=COL["heur"], label="raw")
    ax.plot(xc, yc, "s-", ms=3, lw=1.3, color=COL["v2"], label="calibrated (T=%.1f)" % T)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical frequency")
    ax.set_title("Reliability diagram (test split)"); ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    save(fig, "reliability_curve")


# ================================================================== 9. latency (ONNX runtime)
def fig_latency():
    try:
        import time
        import pandas as pd
        sys.path.insert(0, os.path.join(REPO, "robot_package", "src", "entanglement_detector"))
        from entanglement_detector.engine import EntanglementEngine
        import entanglement_detector.constants as K
        mdir = os.path.join(REPO, "robot_package", "src", "entanglement_detector", "models")
        eng = EntanglementEngine(
            model_path=os.path.join(mdir, "entanglement_tcn.onnx"),
            normalize_path=os.path.join(mdir, "normalize.json"),
            intensity_calib_path=os.path.join(mdir, "intensity_calib.json"),
            temperature=5.918, num_threads=1)
        df = pd.read_csv(os.path.join(REPO, "csv_labelled", "back_left_hand2.csv"))
        names = K.raw_channel_names()
        rows = df[names].to_dict("records")
        times = []
        for r in rows:
            t0 = time.perf_counter()
            out = eng.push(r)
            dt = (time.perf_counter() - t0) * 1e3
            if out is not None:
                times.append(dt)
        times = np.array(times)
    except Exception as exc:  # pragma: no cover
        print("  [skip] latency:", exc)
        return
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.hist(times, bins=45, color=COL["tcn"], alpha=0.85, edgecolor="white", linewidth=0.3)
    for q, ls, lab in [(np.mean(times), "-", "mean %.2f ms"),
                       (np.percentile(times, 95), "--", "p95 %.2f ms")]:
        ax.axvline(q, color="#333", ls=ls, lw=1.0, label=lab % q)
    ax.axvline(2.0, color=COL["neg"], ls=":", lw=1.1, label="budget 2.0 ms @ 500 Hz")
    ax.set_xlabel("per-window inference latency (ms)")
    ax.set_ylabel("windows")
    ax.set_title("Runtime latency (ONNX, 1 thread, n=%d)" % len(times))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "latency_hist")
    print("  latency: mean=%.2f median=%.2f p95=%.2f max=%.2f ms" %
          (times.mean(), np.median(times), np.percentile(times, 95), times.max()))


def main():
    rm = load("report_metrics.json")
    vr = load("validation_v2_results.json")
    ab = load("ablation.json")
    op = load("operating_point.json")
    print("generating figures ->", os.path.relpath(OUT, REPO))
    fig_dataset()
    fig_loro(rm)
    fig_perleg(rm)
    fig_compare(rm)
    fig_confusion(rm)
    fig_ablation(ab)
    fig_fieldissues(vr)
    fig_model_curves(op)
    fig_latency()
    print("done.")


if __name__ == "__main__":
    main()
