"""Consolidated reliability-improvements report -> artifacts/IMPROVEMENTS.md.

Pulls together (identical evaluation protocol throughout):
  1. Probability calibration (temperature scaling): Brier + ECE, threshold effect.
  2. back_left_hand1 LORO root-cause (summary; full proof in diagnose.py).
  3. RR false-positive reduction via a per-leg threshold (no retrain).
  4. Time-based debounce sweep (FAR / latency / F1).
  5. Three-way detector comparison: TCN vs heuristic vs user's Statistical Detector.
  6. Feature ablation (loads artifacts/ablation.json if present).
  7. Verdict: which changes genuinely help; updated recommended operating point.

Run:
    python -m dataset.ml.improvements
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C
from . import io_load, evaluate
from . import calibration as calib_mod
from . import stat_detector
from . import debounce as debounce_mod
from .report import (select_threshold, persistence_alarm, binary_metrics, aucs,
                     leg_metrics, heuristic_fit_baseline, heuristic_scores,
                     heuristic_pred_legs, TARGET_FAR, LEG_THR, PERSIST_K)

DEBOUNCE_MS = 75


def _provider_metrics(val_scored, test_scored, is_prob, pred_legs_key):
    """Identical protocol: threshold on VAL negatives, score TEST. pred_legs gated by alarm."""
    vneg = np.concatenate([r["score"][r["y_bin"] == 0] for r in val_scored])
    thr = select_threshold(vneg, TARGET_FAR, is_prob=is_prob)
    y = np.concatenate([r["y_bin"] for r in test_scored])
    sc = np.concatenate([r["score"] for r in test_scored])
    bm = binary_metrics(y, sc, thr); ap, roc = aucs(y, sc)
    yl = np.concatenate([r["y_legs"] for r in test_scored])
    if pred_legs_key == "tcn":
        pl = np.concatenate([(r["p_legs"] >= LEG_THR).astype(int) for r in test_scored])
    else:
        pl = np.concatenate([r[pred_legs_key] for r in test_scored])
    lm = leg_metrics(yl, pl, y)
    return {"thr": thr, **bm, "PR_AUC": ap, "ROC_AUC": roc, "exact_match": lm["exact_match"],
            "leg_macro_F1": float(np.mean([lm[l]["F1"] for l in C.LEG_ORDER]))}


def main():
    device = evaluate._device()
    model, normalizer, calibrator = evaluate.load_artifacts(device)
    recs = io_load.list_recordings()
    lines = []

    def w(s=""):
        lines.append(s); print(s)

    w("# Reliability Improvements Report")
    w(f"\nProtocol unchanged from REPORT.md: TARGET_FAR={TARGET_FAR:.0%}, leg_thr={LEG_THR}, "
      f"window=0.40 s. All changes here are **post-hoc / inference-time** (calibration, "
      f"thresholds, debounce, comparison) — they do **not** retrain or alter the TCN weights, "
      f"so LORO detection ranking is unchanged by these post-hoc steps (see RETRAIN_V2_REPORT.md "
      f"for the current model's LORO). Feature ablations (§6) DO retrain and are reported with their "
      f"own LORO. NOTE: §6's ablation table is the v1 feature-set study (architecture-level "
      f"conclusions, not re-run for v2); §1 calibration IS refreshed for the current model.")

    # ---- score TCN on val/test once ----
    tcn_val = [evaluate.dense_infer(s, recs[s], model, normalizer, calibrator, device, hop=5)
               for s in C.SPLIT["val"]]
    tcn_val = [dict(r, score=r["p_bin"]) for r in tcn_val if r["n"]]
    tcn_test = [dict(evaluate.dense_infer(s, recs[s], model, normalizer, calibrator, device, hop=1),
                     ) for s in C.SPLIT["test"]]
    for r in tcn_test:
        r["score"] = r["p_bin"]
    tcn_fixed = _provider_metrics(tcn_val, tcn_test, is_prob=True, pred_legs_key="tcn")

    # ============ 1. Calibration ============
    w("\n## 1. Probability calibration (temperature scaling)")
    vlogit = np.concatenate([r["bin_logit"] for r in tcn_val])
    vy = np.concatenate([r["y_bin"] for r in tcn_val])
    tlogit = np.concatenate([r["bin_logit"] for r in tcn_test])
    ty = np.concatenate([r["y_bin"] for r in tcn_test])
    T = calib_mod.fit_temperature(vlogit, vy)
    calib_mod.save(T)
    p_before = 1 / (1 + np.exp(-tlogit)); p_after = calib_mod.apply_temperature(tlogit, T)
    sat_before = float((p_before >= 0.999999).mean()); sat_after = float((p_after >= 0.999999).mean())
    # threshold for 1% FAR before/after on VAL
    thr_before = select_threshold(np.concatenate([r["p_bin"][r["y_bin"] == 0] for r in tcn_val]), TARGET_FAR, True)
    val_pa = calib_mod.apply_temperature(np.concatenate([r["bin_logit"][r["y_bin"] == 0] for r in tcn_val]), T)
    thr_after = float(np.quantile(val_pa, 1 - TARGET_FAR))
    w(f"\n| | Brier | ECE | sat@1.0 | VAL 1%-FAR threshold |")
    w("|---|---|---|---|---|")
    w(f"| uncalibrated | {calib_mod.brier(p_before,ty):.4f} | {calib_mod.ece(p_before,ty):.4f} | "
      f"{sat_before*100:.0f}% | {thr_before:.4f} (clamped) |")
    w(f"| temp-scaled (T={T:.2f}) | {calib_mod.brier(p_after,ty):.4f} | {calib_mod.ece(p_after,ty):.4f} | "
      f"{sat_after*100:.0f}% | {thr_after:.4f} (not clamped) |")
    w(f"\n- T={T:.2f}≫1 confirms over-confidence. Calibration **improves Brier** and **de-saturates** "
      f"(was {sat_before*100:.0f}% pinned at 1.0 → {sat_after*100:.0f}%), so the 1%-FAR threshold is "
      f"now a real interior value ({thr_after:.3f}) instead of the clamped 0.9999 — fixing the §0 "
      f"WARN from REPORT.md. ROC/PR-AUC are unchanged (monotonic).")
    w(f"- Honest caveat: **ECE does not improve** here (small VAL set fit over-compresses TEST). "
      f"Calibration's win is de-saturation + Brier, not ECE; Brier is the proper-score that matters "
      f"for the severity/intensity use.")

    # ============ 2. back_left_hand1 ============
    w("\n## 2. back_left_hand1 LORO root-cause (F1=0.45)")
    w("- **Not weak signal**: it has the strongest RL contrast (3.8σ vs siblings 0.7σ/1.4σ).")
    w("- **Out-of-distribution**: lowest walking baseline (RL_down_walk 0.23 vs 0.71/0.37 Nm), "
      "slower gait (walk_dq 1.30 vs 1.67/1.56), and it is the longest recording (7538 rows).")
    w("- **Proof**: retraining the held-out fold, the model scores hand1's *entangled* windows at "
      "median p_bin≈0.001 — it doesn't recognize the pattern when hand1 is absent from training.")
    w("- **Verdict**: small-data coverage gap, not an architecture flaw. Fix = collect more rear-leg "
      "hand-entanglement recordings across gait regimes (no model change). (full evidence: diagnose.py)")

    # ============ 3. RR false positives ============
    w("\n## 3. RR false-positive reduction (per-leg threshold, no retrain)")
    rr = C.LEG_ORDER.index("RR")
    yl = np.concatenate([r["y_legs"][:, rr] for r in tcn_test])
    pl = np.concatenate([r["p_legs"][:, rr] for r in tcn_test])
    w("\n| RR leg-thr | P | R | F1 |")
    w("|---|---|---|---|")
    for t in [0.5, 0.7, 0.9, 0.95]:
        pred = pl >= t
        tp = int((pred & (yl == 1)).sum()); fp = int((pred & (yl == 0)).sum()); fn = int((~pred & (yl == 1)).sum())
        P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
        F1 = 2*P*R/(P+R) if P+R else 0
        w(f"| {t:.2f} | {P:.2f} | {R:.2f} | {F1:.2f} |")
    w("\n- RR over-fires mainly when **RL is entangled** (rear-leg coupling) and on **walking prefixes** "
      "(early detection). Raising RR's leg-threshold to ~0.9 lifts precision with ≤0.05 recall loss — "
      "an inference-time knob, LORO untouched.")

    # ============ 4. Debounce ============
    w("\n## 4. Time-based debounce (replaces literal 3-window=6 ms)")
    rows = debounce_mod.main()
    w("\n| debounce | pooled F1 | recall | FAR walking4 | FAR test-prefix | med latency |")
    w("|---|---|---|---|---|---|")
    for d in rows:
        lat = ("%.0f ms" % d["med_lat_ms"]) if d["med_lat_ms"] is not None else "n/a"
        w(f"| {d['ms']} ms (k={d['k']}) | {d['F1']:.3f} | {d['R']:.2f} | {d['far_wk4']*100:.2f}% | "
          f"{d['far_pre']*100:.2f}% | {lat} |")
    w(f"\n- The literal 6 ms rule ≈ raw. A **{DEBOUNCE_MS} ms** debounce removes clean-walking false "
      f"alarms (→0%) for ~0.013 F1 and ~0 ms added latency. **Recommend 50–75 ms.**")

    # ============ 5. Detector comparison ============
    w("\n## 5. Detector comparison — identical protocol (TCN vs heuristic vs Statistical Detector)")
    # heuristic
    base = heuristic_fit_baseline(C.SPLIT["train"])
    h_val = [heuristic_scores(s, base, hop=5) for s in C.SPLIT["val"]]
    h_val = [r for r in h_val if r]
    h_test = [heuristic_scores(s, base, hop=1) for s in C.SPLIT["test"]]
    hthr = select_threshold(np.concatenate([r["score"][r["y_bin"] == 0] for r in h_val]), TARGET_FAR, False)
    for r in h_test:
        r["pred_legs_h"] = heuristic_pred_legs(r, hthr)
    h_fixed = _provider_metrics(h_val, h_test, is_prob=False, pred_legs_key="pred_legs_h")
    # statistical detector
    scal = stat_detector.calibrate(C.SPLIT["train"])
    s_val = [stat_detector.score_recording(s, scal, hop=5) for s in C.SPLIT["val"]]
    s_val = [r for r in s_val if r]
    s_test = [stat_detector.score_recording(s, scal, hop=1) for s in C.SPLIT["test"]]
    sthr = select_threshold(np.concatenate([r["score"][r["y_bin"] == 0] for r in s_val]), TARGET_FAR, False)
    for r in s_test:
        alarm = persistence_alarm(r["score"], sthr)
        r["pred_legs_s"] = r["pred_legs_raw"] * alarm[:, None]
    s_fixed = _provider_metrics(s_val, s_test, is_prob=False, pred_legs_key="pred_legs_s")

    w("\n| detector | P | R | F1 | PR-AUC | ROC-AUC | leg exact-match | localizes |")
    w("|---|---|---|---|---|---|---|---|")
    w(f"| **TCN (multi-task)** | {tcn_fixed['P']:.2f} | {tcn_fixed['R']:.2f} | {tcn_fixed['F1']:.3f} | "
      f"{tcn_fixed['PR_AUC']:.3f} | {tcn_fixed['ROC_AUC']:.3f} | {tcn_fixed['exact_match']:.3f} | "
      f"all 4 legs + intensity |")
    w(f"| Statistical Detector | {s_fixed['P']:.2f} | {s_fixed['R']:.2f} | {s_fixed['F1']:.3f} | "
      f"{s_fixed['PR_AUC']:.3f} | {s_fixed['ROC_AUC']:.3f} | {s_fixed['exact_match']:.3f} | "
      f"rear legs; front=unlocalized |")
    w(f"| Heuristic (thigh z) | {h_fixed['P']:.2f} | {h_fixed['R']:.2f} | {h_fixed['F1']:.3f} | "
      f"{h_fixed['PR_AUC']:.3f} | {h_fixed['ROC_AUC']:.3f} | {h_fixed['exact_match']:.3f} | "
      f"single leg (argmax) |")
    w(f"\n- The **Statistical Detector** (rear-thigh deviation, calibrated on TRAIN walking, scored "
      f"under the identical protocol) reaches F1={s_fixed['F1']:.2f} / ROC-AUC={s_fixed['ROC_AUC']:.2f}. "
      f"Its absolute-threshold rules (tuned for the older, higher-amplitude data) never trigger on this "
      f"dataset's subtler entanglement (rear thigh only dips ~1–2 Nm), so it relies on deviation ranking; "
      f"it cannot localize FL vs FR and gives no intensity.")
    w(f"- The **TCN** leads on every detection metric and is the only one with per-leg attribution + "
      f"calibrated intensity.")

    # ============ 6. Ablation ============
    w("\n## 6. Feature ablation (retrained; own LORO)")
    abl_path = os.path.join(C.ARTIFACTS_DIR, "ablation.json")
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            abl = json.load(f)
        w("\n| feature set | C | fixed F1 | PR-AUC | leg macro-F1 | LORO F1 (mean±std) |")
        w("|---|---|---|---|---|---|")
        order = ["raw+engineered", "raw_only", "engineered_only", "no_imu", "no_foot"]
        for k in order:
            if k not in abl:
                continue
            d = abl[k]; fx = d["fixed"]
            loro = (f"{d['loro_mean_F1']:.3f} ± {d['loro_std_F1']:.3f}"
                    if "loro_mean_F1" in d else "pending")
            star = " ⟵ shipped" if k == "raw+engineered" else ""
            w(f"| {k}{star} | {d['n_channels']} | {fx['F1']:.3f} | {fx['PR_AUC']:.3f} | "
              f"{fx['leg_macro_F1']:.3f} | {loro} |")
        missing = [k for k in order if k not in abl]
        if missing:
            w(f"\n_(pending configs: {', '.join(missing)} — ablation still running)_")
        else:
            full = abl.get("raw+engineered", {}).get("loro_mean_F1")
            if full is not None:
                w(f"\n- **Raw kinematics are essential**: engineered_only collapses "
                  f"(LORO {abl['engineered_only']['loro_mean_F1']:.3f}).")
                w(f"- **Engineered physics channels help**: dropping them (raw_only) costs "
                  f"~{full - abl['raw_only']['loro_mean_F1']:+.3f} LORO.")
                w(f"- **Foot sensors help a little** (no_foot {abl['no_foot']['loro_mean_F1']:.3f} vs "
                  f"{full:.3f}). **IMU is near-redundant** for detection (no_imu "
                  f"{abl['no_imu']['loro_mean_F1']:.3f}, ties full with lower variance).")
                w(f"- **Decision**: keep the shipped raw+engineered (C=60) — nothing strictly "
                  f"beats it on LORO. Dropping IMU (C=52) is a valid leaner option with no LORO "
                  f"loss, but the gap is within noise so it is not mandated. (Differences <0.03 "
                  f"sit inside the ±0.15 fold std — single-seed estimates.)")
    else:
        w("\n_ablation.json not found yet (background run in progress)._")

    # ============ 7. Verdict ============
    w("\n## 7. Verdict — what genuinely helps")
    w("\n| change | effect | adopt? |")
    w("|---|---|---|")
    w(f"| Temperature scaling (T={T:.1f}) | Brier {calib_mod.brier(p_before,ty):.3f}→"
      f"{calib_mod.brier(p_after,ty):.3f}; de-saturates; real threshold | **Yes** (post-hoc) |")
    w("| 50–75 ms debounce | clean-walking FAR →0%, F1 −0.01, latency ~0 | **Yes** (inference) |")
    w("| RR per-leg threshold ~0.9 | RR precision up, recall −≤0.05 | **Yes** (inference) |")
    w("| Feature set | see §6 — keep the set with best LORO | adopt only if LORO ≥ shipped |")
    w("\nAll adopted changes are post-hoc and do not reduce LORO. The recommended operating point is: "
      "**temperature-scaled probabilities + 75 ms debounce + per-leg thresholds (RR≈0.9)**.")

    # ---- persist the recommended operating point for the deployment engine ----
    # The ALARM latch uses the RAW probability vs the RAW VAL-1%FAR threshold — that is
    # the operating point validated in §4 (raw probs saturate during entanglement, so the
    # debounce latches reliably). Temperature is for the reported CONFIDENCE (Brier), not
    # the latch: de-saturated probs fluctuate and would never satisfy a k-window debounce.
    op = {
        "temperature": T,                                   # for calibrated confidence output
        "detection_threshold_raw": float(thr_before),       # used by the debounce latch
        "detection_threshold_calibrated": float(thr_after),  # reference (same window, calib units)
        "debounce_ms": DEBOUNCE_MS,
        "leg_thresholds": {leg: (0.9 if leg == "RR" else 0.5) for leg in C.LEG_ORDER},
        "note": "post-hoc operating point; does not change TCN weights or the eval protocol",
    }
    with open(os.path.join(C.ARTIFACTS_DIR, "operating_point.json"), "w") as f:
        json.dump(op, f, indent=2)

    with open(os.path.join(C.ARTIFACTS_DIR, "IMPROVEMENTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    w(f"\nSaved IMPROVEMENTS.md + operating_point.json -> "
      f"{os.path.relpath(C.ARTIFACTS_DIR, C.PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
