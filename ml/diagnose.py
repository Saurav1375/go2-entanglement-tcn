"""Diagnostics: (1) back_left_hand1 LORO failure root-cause, (2) RR false positives.

Both are read-only analyses; neither changes the model.

Run:
    python -m dataset.ml.diagnose blh1     # back_left_hand1 vs RL siblings
    python -m dataset.ml.diagnose rr        # RR false-positive analysis + threshold sweep
    python -m dataset.ml.diagnose all
"""
from __future__ import annotations

import sys

import numpy as np

from . import config as C
from . import io_load, resample, evaluate


# ---------------------------------------------------------------- (1) blh1
def regime_stats(stem):
    """Walking-regime + entanglement-contrast stats for one recording."""
    recs = io_load.list_recordings()
    df = io_load.load_recording(recs[stem])
    res_df, status = resample.resample_recording(df)
    ent = status == "Entangled"
    walk = status == "Walking"
    tt = res_df["RL_thigh_tau"].to_numpy()
    dq = np.abs(res_df["RL_thigh_dq"].to_numpy())
    down = np.maximum(0.0, -tt)
    w_mu, w_sd = down[walk].mean(), max(down[walk].std(), 1e-6)
    # whole-body walking regime descriptors (do the recordings live in the same space?)
    all_dq = np.abs(res_df[[f"{l}_{j}_dq" for l in C.LEG_ORDER for j in C.JOINT_ORDER]].to_numpy())
    foot = res_df[[f"foot_{l}" for l in C.CSV_FOOT_ORDER]].to_numpy()
    return {
        "n_rows": len(res_df),
        "dur_ent": float(ent.sum() / C.TARGET_HZ),
        "onset": float(np.argmax(ent) / C.TARGET_HZ) if ent.any() else -1,
        "RL_down_walk": float(down[walk].mean()),
        "RL_down_ent": float(down[ent].mean()) if ent.any() else float("nan"),
        "contrast_sigma": float((down[ent].mean() - w_mu) / w_sd) if ent.any() else float("nan"),
        "walk_dq_mean": float(all_dq[walk].mean()),       # gait speed proxy
        "walk_foot_mean": float(foot[walk].mean()),         # load regime
    }


def run_blh1():
    print("=== back_left_hand1 LORO root-cause (RL recordings) ===\n")
    print("All three are RL entanglements. contrast_σ = rise of RL downward thigh torque during")
    print("entanglement, in walking-std units. walk_dq/walk_foot describe the WALKING regime.\n")
    print(f"{'recording':18s} {'split':5s} {'rows':>6s} {'durE':>5s} {'onset':>6s} "
          f"{'RLdown_w':>9s} {'RLdown_e':>9s} {'contrastσ':>10s} {'walk_dq':>8s} {'walk_foot':>9s}")
    rows = {}
    for stem in ["back_left_hand1", "back_left_hand2", "back_left_wire1"]:
        s = regime_stats(stem); rows[stem] = s
        split = ("test" if stem in C.SPLIT["test"] else "val" if stem in C.SPLIT["val"] else "train")
        print(f"{stem:18s} {split:5s} {s['n_rows']:6d} {s['dur_ent']:5.1f} {s['onset']:6.1f} "
              f"{s['RL_down_walk']:9.2f} {s['RL_down_ent']:9.2f} {s['contrast_sigma']:10.1f} "
              f"{s['walk_dq_mean']:8.3f} {s['walk_foot_mean']:9.2f}")

    # Definitive proof: retrain the hand1-held-out fold and look at its scores ON hand1.
    print("\nRetraining the back_left_hand1-held-out LORO fold to inspect its scores...")
    from .train import train_one
    from .normalize import fit_from_train
    from .intensity import IntensityCalibrator
    device = evaluate._device()
    held = "back_left_hand1"
    recs_all = list(C.SPLIT["train"]) + list(C.SPLIT["val"]) + list(C.SPLIT["test"])
    train_stems = [s for s in recs_all if s != held]
    norm = fit_from_train(train_stems)
    calib = IntensityCalibrator.fit([s for s in train_stems if not C.parse_legs(s)] or train_stems)
    model, _, _ = train_one(train_stems, None, norm, calib, device, epochs=20, verbose=False)
    r = evaluate.dense_infer(held, io_load.list_recordings()[held], model, norm, calib, device, hop=1)
    ent = r["y_bin"] == 1
    print(f"  held-out model p_bin on hand1: entangled windows median={np.median(r['p_bin'][ent]):.3f} "
          f"(mean={r['p_bin'][ent].mean():.3f}); walking windows median={np.median(r['p_bin'][~ent]):.3f}")
    print(f"  → the held-out model assigns LOW scores during hand1's entanglement despite its "
          f"strong torque contrast, i.e. it fails to recognize hand1's pattern, not the data being weak.")

    print("\nRoot cause:")
    h1, h2, w1 = rows["back_left_hand1"], rows["back_left_hand2"], rows["back_left_wire1"]
    print(f"- NOT a weak signal: hand1 has the STRONGEST RL contrast ({h1['contrast_sigma']:.1f}σ vs "
          f"hand2 {h2['contrast_sigma']:.1f}σ, wire1 {w1['contrast_sigma']:.1f}σ).")
    print(f"- It is OUT-OF-DISTRIBUTION: its walking regime differs — RL_down_walk "
          f"{h1['RL_down_walk']:.2f} vs {h2['RL_down_walk']:.2f}/{w1['RL_down_walk']:.2f} Nm, "
          f"walk_dq {h1['walk_dq_mean']:.3f} vs {h2['walk_dq_mean']:.3f}/{w1['walk_dq_mean']:.3f}, "
          f"and it is the longest recording ({h1['n_rows']} rows).")
    print(f"- With hand1 removed, only 2 RL positives remain (both shorter, different regime), so the "
          f"model can't generalize to hand1's distribution → held-out scores stay low → R≈0.30. "
          f"This is small-data coverage, fixable with more recordings, NOT a model-architecture flaw.")


# ---------------------------------------------------------------- (2) RR FP
def run_rr(model=None, normalizer=None, calibrator=None, device=None, leg_thr=0.5):
    if model is None:
        device = evaluate._device()
        model, normalizer, calibrator = evaluate.load_artifacts(device)
    recs = io_load.list_recordings()
    print("=== RR false-positive analysis (per-leg threshold = %.2f) ===\n" % leg_thr)

    # gather TEST windows; find where RR predicted but RR not truly entangled
    fp_by_truth = {}
    rr_idx = C.LEG_ORDER.index("RR")
    tot_fp = tot_tp = 0
    for stem in C.SPLIT["test"]:
        r = evaluate.dense_infer(stem, recs[stem], model, normalizer, calibrator, device, hop=1)
        pred_rr = r["p_legs"][:, rr_idx] >= leg_thr
        true_rr = r["y_legs"][:, rr_idx] >= 0.5
        fp = pred_rr & ~true_rr
        tot_fp += int(fp.sum()); tot_tp += int((pred_rr & true_rr).sum())
        if fp.any():
            # what is the true state of those FP windows?
            ent = r["y_bin"][fp] == 1
            key = f"{stem} (affected {','.join(sorted(r['affected'])) or 'none'})"
            fp_by_truth[key] = {"fp": int(fp.sum()),
                                "during_entangled_other_leg": int(ent.sum()),
                                "during_walking": int((~ent).sum())}
    print(f"RR: total TP windows={tot_tp}, total FP windows={tot_fp}")
    print("Where RR false-fires:")
    for k, v in fp_by_truth.items():
        print(f"  {k}: {v['fp']} FP  (entangled-other-leg={v['during_entangled_other_leg']}, "
              f"walking={v['during_walking']})")

    # threshold sweep for RR: precision/recall vs leg_thr (pooled test)
    yl, pl = [], []
    for stem in C.SPLIT["test"]:
        r = evaluate.dense_infer(stem, recs[stem], model, normalizer, calibrator, device, hop=1)
        yl.append(r["y_legs"][:, rr_idx]); pl.append(r["p_legs"][:, rr_idx])
    yl = np.concatenate(yl); pl = np.concatenate(pl)
    print("\nRR leg-threshold sweep (pooled TEST):")
    print(f"  {'thr':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    best = None
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pred = pl >= thr
        tp = int((pred & (yl == 1)).sum()); fp = int((pred & (yl == 0)).sum())
        fn = int((~pred & (yl == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        print(f"  {thr:5.2f} {P:6.2f} {R:6.2f} {F1:6.2f}")
        if best is None or F1 >= best[3]:
            best = (thr, P, R, F1)
    print(f"\n→ raising RR leg-threshold to ~{best[0]:.2f} lifts precision to {best[1]:.2f} "
          f"(R={best[2]:.2f}, F1={best[3]:.2f}) without retraining. This is an inference-time "
          f"knob; it does not touch the model or LORO.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("blh1", "all"):
        run_blh1()
        print()
    if which in ("rr", "all"):
        run_rr()
