"""Evaluate the trained multi-task TCN on the test split + produce replay plots.

Metrics:
  detection   : precision / recall / F1, PR-AUC, ROC-AUC, alarm latency,
                false-alarm rate on walking/stop
  leg attrib. : per-leg P/R/F1, confusion, exact-match (set equality)
  intensity   : qualitative (median on walking/stop, monotonic-from-onset,
                affected>non-affected) + Spearman vs thigh_tau_down proxy
Also writes per-recording replay PNGs and prints a GBM baseline comparison.

Run:
    python -m dataset.ml.evaluate
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import config as C
from . import io_load, resample, features
from .intensity import IntensityCalibrator
from .model import EntanglementTCN
from .normalize import Normalizer
from .windowing import make_windows


def _device() -> str:
    return "cuda" if (C.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"


def load_artifacts(device: str):
    ckpt = torch.load(os.path.join(C.ARTIFACTS_DIR, "model.pt"), map_location=device,
                      weights_only=False)
    model = EntanglementTCN(in_channels=ckpt["config"]["n_channels"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalizer = Normalizer.load()
    calibrator = IntensityCalibrator.load()
    return model, normalizer, calibrator


@torch.no_grad()
def dense_infer(stem: str, path: str, model, normalizer, calibrator, device,
                hop: int = C.HOP_EVAL, intensity_blend: float = 0.5) -> dict:
    """Run the model densely over one recording. Returns per-window arrays."""
    res_df, status = resample.cached_resample(path)
    X = features.build_channel_matrix(res_df)
    ws = make_windows(X, status, stem, C.parse_legs(stem), hop=hop)
    n = len(ws.y_bin)
    if n == 0:
        return {"stem": stem, "n": 0}

    p_bin = np.zeros(n, dtype=np.float32)
    bin_logit = np.zeros(n, dtype=np.float32)
    p_legs = np.zeros((n, C.N_LEGS), dtype=np.float32)
    head_int = np.zeros((n, C.N_LEGS), dtype=np.float32)
    B = 512
    for i in range(0, n, B):
        # normalize per-batch to avoid a full-recording copy (big walking files)
        xb_np = normalizer.apply(ws.X[i:i + B])
        xb = torch.from_numpy(np.ascontiguousarray(xb_np)).to(device)
        out = model(xb)
        bin_logit[i:i + B] = out["bin_logit"].cpu().numpy()
        p_bin[i:i + B] = torch.sigmoid(out["bin_logit"]).cpu().numpy()
        p_legs[i:i + B] = torch.sigmoid(out["legs_logit"]).cpu().numpy()
        head_int[i:i + B] = torch.sigmoid(out["intensity_logit"]).cpu().numpy()

    # physics intensity: gate (per-leg prob) * magnitude; reported = blend(head, physics)
    phys_mag = np.zeros((n, C.N_LEGS), dtype=np.float32)
    proxy = np.zeros((n, C.N_LEGS), dtype=np.float32)  # thigh_tau_down mean per leg (GT-free proxy)
    for r, end in enumerate(ws.end_idx):
        start = int(end) - C.WINDOW_SAMPLES + 1
        phys_mag[r] = calibrator.magnitudes_all_legs(res_df, start, int(end))
        for j, leg in enumerate(C.LEG_ORDER):
            tt = res_df[f"{leg}_thigh_tau"].to_numpy()[start:int(end) + 1]
            proxy[r, j] = np.maximum(0.0, -tt).mean()
    phys_I = p_legs * phys_mag
    intensity = intensity_blend * head_int + (1.0 - intensity_blend) * phys_I

    return {
        "stem": stem, "n": n,
        "end_idx": ws.end_idx, "ent_frac": ws.ent_frac,
        "y_bin": ws.y_bin, "y_legs": ws.y_legs,
        "p_bin": p_bin, "bin_logit": bin_logit, "p_legs": p_legs,
        "intensity": intensity, "phys_mag": phys_mag, "proxy": proxy,
        "onset_idx": ws.onset_idx, "t": ws.end_idx / C.TARGET_HZ,
        "affected": C.parse_legs(stem),
    }


# --------------------------------------------------------------------- metrics
def prf(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def choose_threshold(val_results, target_far=0.01):
    """Pick the lowest detection threshold whose false-alarm rate on negative
    windows is <= target_far (maximizes recall subject to the FAR budget).

    Negative scores can cluster very close to 1.0, so the upper cap is 0.9999;
    capping lower would silently admit a higher-than-target FAR.
    """
    neg_p = np.concatenate([r["p_bin"][r["y_bin"] == 0] for r in val_results if r["n"]])
    if len(neg_p) == 0:
        return 0.5
    # threshold = (1-target_far) quantile of negative scores
    thr = float(np.quantile(neg_p, 1.0 - target_far))
    return min(max(thr, 0.05), 0.9999)


def alarm_latency(res, thr, persistence=C.PERSISTENCE_WINDOWS):
    """Seconds from entanglement onset to first `persistence` consecutive fires."""
    if res["onset_idx"] < 0:
        return None
    fire = res["p_bin"] >= thr
    run = 0
    for i in range(res["n"]):
        run = run + 1 if fire[i] else 0
        if run >= persistence and res["end_idx"][i] >= res["onset_idx"]:
            return (res["end_idx"][i] - res["onset_idx"]) / C.TARGET_HZ
    return None  # never sustained-fired after onset


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


# --------------------------------------------------------------------- plotting
def plot_replay(res, thr, leg_thr, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = res["t"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Replay — {res['stem']}  (affected: "
                 f"{','.join(sorted(res['affected'])) or 'none'})",
                 fontsize=13, fontweight="bold")

    # GT ribbon + p_bin
    ax = axes[0]
    ax.fill_between(t, 0, 1, where=res["y_bin"] == 1, color="#f1948a", alpha=0.4,
                    step="mid", label="GT Entangled")
    ax.plot(t, res["p_bin"], color="#1f77b4", lw=1.0, label="p(entangled)")
    ax.axhline(thr, color="gray", ls="--", lw=0.8, label=f"thr={thr:.2f}")
    if res["onset_idx"] >= 0:
        ax.axvline(res["onset_idx"] / C.TARGET_HZ, color="red", ls=":", lw=1.2, label="onset")
    ax.set_ylabel("detection"); ax.set_ylim(-0.05, 1.05); ax.legend(loc="upper right", fontsize=7)

    # per-leg probability
    ax = axes[1]
    colors = {"FR": "#e74c3c", "FL": "#2ecc71", "RR": "#3498db", "RL": "#f39c12"}
    for j, leg in enumerate(C.LEG_ORDER):
        ax.plot(t, res["p_legs"][:, j], color=colors[leg], lw=0.9,
                label=f"{leg}{'*' if leg in res['affected'] else ''}")
    ax.axhline(leg_thr, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("per-leg p"); ax.set_ylim(-0.05, 1.05); ax.legend(loc="upper right", fontsize=7, ncol=4)

    # per-leg intensity
    ax = axes[2]
    for j, leg in enumerate(C.LEG_ORDER):
        ax.plot(t, res["intensity"][:, j], color=colors[leg], lw=0.9,
                label=f"{leg}{'*' if leg in res['affected'] else ''}")
    ax.set_ylabel("intensity"); ax.set_ylim(-0.05, 1.05); ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=7, ncol=4)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"replay_{res['stem']}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# --------------------------------------------------------------------- main
def main():
    device = _device()
    model, normalizer, calibrator = load_artifacts(device)
    print(f"Loaded artifacts on {device}\n")

    # threshold selection on VAL
    val_results = [dense_infer(s, io_load.list_recordings()[s], model, normalizer,
                               calibrator, device, hop=5) for s in C.SPLIT["val"]]
    thr = choose_threshold([r for r in val_results if r["n"]], target_far=0.01)
    leg_thr = 0.5
    print(f"Detection threshold (VAL, FAR<=1%): {thr:.3f}\n")

    recs = io_load.list_recordings()

    # ---- detection + leg + intensity on TEST ----
    print("=== TEST per-recording ===")
    all_y, all_p = [], []
    all_yl, all_pl = [], []
    exact_hits = exact_total = 0
    latencies = []
    for stem in C.SPLIT["test"]:
        res = dense_infer(stem, recs[stem], model, normalizer, calibrator, device, hop=1)
        all_y.append(res["y_bin"]); all_p.append(res["p_bin"])
        all_yl.append(res["y_legs"]); all_pl.append(res["p_legs"])
        prec, rec, f1 = prf(res["y_bin"], res["p_bin"], thr)
        lat = alarm_latency(res, thr)
        if lat is not None:
            latencies.append(lat)
        # exact-match leg set on truly-entangled windows
        pos = res["y_bin"] == 1
        if pos.any():
            pred_sets = (res["p_legs"][pos] >= leg_thr)
            true_sets = (res["y_legs"][pos] >= 0.5)
            exact_hits += int((pred_sets == true_sets).all(axis=1).sum())
            exact_total += int(pos.sum())
        plot_replay(res, thr, leg_thr, C.PLOTS_DIR)
        print(f"  {stem:20s} P={prec:.2f} R={rec:.2f} F1={f1:.2f}  "
              f"latency={'%.3fs' % lat if lat is not None else 'n/a':>8}")

    y = np.concatenate(all_y); p = np.concatenate(all_p)
    yl = np.concatenate(all_yl); pl = np.concatenate(all_pl)
    P, R, F1 = prf(y, p, thr)
    ap, roc = pr_auc(y, p)
    print(f"\n[detection · pooled test] P={P:.3f} R={R:.3f} F1={F1:.3f} "
          f"PR-AUC={ap:.3f} ROC-AUC={roc:.3f}")
    print(f"[detection] median alarm latency = "
          f"{np.median(latencies)*1000:.0f} ms" if latencies else "[detection] no sustained alarms")

    # leg attribution
    print("\n=== leg attribution (pooled test) ===")
    for j, leg in enumerate(C.LEG_ORDER):
        lp, lr, lf = prf(yl[:, j], pl[:, j], leg_thr)
        print(f"  {leg}: P={lp:.2f} R={lr:.2f} F1={lf:.2f}")
    print(f"  exact-match (set equality on entangled windows): "
          f"{exact_hits}/{exact_total} = {exact_hits/max(exact_total,1):.2f}")

    # ---- false-alarm rate on negatives (val walking + stop) ----
    # NOTE: Stop episodes are shorter than the 0.40 s window, so stop files may
    # yield no full non-blank windows; those are skipped and reported.
    print("\n=== false-alarm rate (negatives) ===")
    for stem in ["walking4", "stop1", "stop2", "stop3"]:
        res = dense_infer(stem, recs[stem], model, normalizer, calibrator, device, hop=5)
        if res["n"] == 0:
            print(f"  {stem:12s} (no full 0.40s windows — short labeled episodes)")
            continue
        far = float((res["p_bin"] >= thr).mean())
        print(f"  {stem:12s} FAR={far*100:.2f}% of windows  ({res['n']} windows)")

    # ---- intensity validation ----
    print("\n=== intensity validation ===")
    # near-zero on negatives
    neg_int = []
    for stem in ["walking4", "stop1", "stop2"]:
        res = dense_infer(stem, recs[stem], model, normalizer, calibrator, device, hop=5)
        if res["n"] == 0:
            continue
        neg_int.append(res["intensity"].max(axis=1))
    neg_int = np.concatenate(neg_int)
    print(f"  negatives: median max-leg intensity = {np.median(neg_int):.3f} "
          f"(target < 0.05)")
    # affected>non-affected + Spearman proxy on entangled windows
    sp_list = []
    for stem in C.SPLIT["test"]:
        res = dense_infer(stem, recs[stem], model, normalizer, calibrator, device, hop=2)
        pos = res["y_bin"] == 1
        if not pos.any():
            continue
        aff_idx = [C.LEG_ORDER.index(l) for l in res["affected"]]
        non_idx = [i for i in range(C.N_LEGS) if i not in aff_idx]
        aff_mean = res["intensity"][pos][:, aff_idx].mean()
        non_mean = res["intensity"][pos][:, non_idx].mean() if non_idx else float("nan")
        # spearman of affected-leg intensity vs thigh_tau_down proxy
        for ai in aff_idx:
            sp = spearman(res["intensity"][pos][:, ai], res["proxy"][pos][:, ai])
            if not np.isnan(sp):
                sp_list.append(sp)
        print(f"  {stem:20s} affected_I={aff_mean:.2f} non_affected_I={non_mean:.2f}")
    if sp_list:
        print(f"  mean Spearman(intensity, thigh_tau_down proxy) on affected legs = "
              f"{np.mean(sp_list):.2f} (target >= 0.6)")

    # save metrics json
    metrics = {"threshold": thr, "detection": {"P": P, "R": R, "F1": F1,
               "PR_AUC": ap, "ROC_AUC": roc,
               "median_latency_ms": float(np.median(latencies) * 1000) if latencies else None},
               "leg_exact_match": exact_hits / max(exact_total, 1)}
    with open(os.path.join(C.ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics.json and replay PNGs -> {C.PLOTS_DIR}")
    return metrics


if __name__ == "__main__":
    main()
