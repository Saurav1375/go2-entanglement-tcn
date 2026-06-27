"""Configurable time-based debounce vs the literal 3-window persistence.

A debounce of D ms requires the detection score to stay >= threshold for
round(D/1000 * TARGET_HZ) consecutive dense windows before an alarm latches.
We compare, under the IDENTICAL protocol (same VAL-1%FAR threshold), how FAR,
alarm latency, and persistence-gated F1 trade off across debounce settings.

Run:
    python -m dataset.ml.debounce
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import io_load, evaluate
from .report import (select_threshold, persistence_alarm, binary_metrics,
                     latency_persist, TARGET_FAR)

SETTINGS_MS = [0, 6, 25, 50, 75, 100, 150]   # 0 == raw per-window (k=1)


def k_for(ms):
    return max(1, round(ms / 1000 * C.TARGET_HZ))


def main():
    device = evaluate._device()
    model, normalizer, calibrator = evaluate.load_artifacts(device)
    recs = io_load.list_recordings()

    # VAL threshold (shared protocol)
    val = [evaluate.dense_infer(s, recs[s], model, normalizer, calibrator, device, hop=5)
           for s in C.SPLIT["val"]]
    val = [r for r in val if r["n"]]
    thr = select_threshold(np.concatenate([r["p_bin"][r["y_bin"] == 0] for r in val]),
                           TARGET_FAR, is_prob=True)

    test = {s: evaluate.dense_infer(s, recs[s], model, normalizer, calibrator, device, hop=1)
            for s in C.SPLIT["test"]}
    wk4 = evaluate.dense_infer("walking4", recs["walking4"], model, normalizer, calibrator, device, hop=5)

    print(f"threshold={thr:.4f}  (persistence applied per-recording; runs don't cross files)\n")
    print(f"{'debounce':>9s} {'k':>4s} {'pooled F1':>10s} {'recall':>7s} "
          f"{'FAR walk4':>10s} {'FAR test-pre':>13s} {'med latency':>12s}")
    rows = []
    for ms in SETTINGS_MS:
        k = k_for(ms)
        # persistence-gated predictions per recording
        yb, pred, lats = [], [], []
        for s, r in test.items():
            alarm = persistence_alarm(r["p_bin"], thr, k)
            yb.append(r["y_bin"]); pred.append(alarm.astype(int))
            lat = latency_persist(r["onset_idx"], r["end_idx"], r["p_bin"], thr, k)
            if lat is not None:
                lats.append(lat)
        yb = np.concatenate(yb); pred = np.concatenate(pred)
        tp = int(((pred == 1) & (yb == 1)).sum()); fp = int(((pred == 1) & (yb == 0)).sum())
        fn = int(((pred == 0) & (yb == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        far_wk4 = persistence_alarm(wk4["p_bin"], thr, k)[wk4["y_bin"] == 0].mean()
        # FAR on test walking prefixes (per recording)
        far_pre = np.concatenate([persistence_alarm(r["p_bin"], thr, k)[r["y_bin"] == 0]
                                  for r in test.values()]).mean()
        med_lat = (np.median(lats) * 1000) if lats else None
        rows.append({"ms": ms, "k": k, "F1": F1, "R": R, "far_wk4": float(far_wk4),
                     "far_pre": float(far_pre), "med_lat_ms": med_lat})
        print(f"{ms:7d}ms {k:4d} {F1:10.3f} {R:7.2f} {far_wk4*100:9.2f}% "
              f"{far_pre*100:12.2f}% {('%.0f ms' % med_lat) if med_lat is not None else 'n/a':>12}")

    print("\nReading: 0 ms == raw per-window. The literal 3-window rule (6 ms) is ~identical to raw. "
          "A 75-100 ms debounce removes clean-walking false alarms at a small latency cost.")
    return rows


if __name__ == "__main__":
    main()
