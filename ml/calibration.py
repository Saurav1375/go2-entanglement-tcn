"""Post-hoc probability calibration (temperature scaling) + reliability metrics.

Temperature scaling fits a single scalar T>0 on VAL logits by minimizing NLL;
calibrated prob = sigmoid(logit / T). It is monotonic, so it does NOT change
ROC-AUC / PR-AUC or the ranking of windows — it only rescales probabilities,
which is exactly what fixes the over-confidence flagged in the report (and makes
the chosen threshold meaningful) WITHOUT touching the model.

Metrics:
  Brier score = mean( (p - y)^2 )           (lower is better)
  ECE (Expected Calibration Error) = sum_b (n_b/N) * |acc_b - conf_b|  over bins

Run:
    python -m dataset.ml.calibration
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C


def fit_temperature(logits: np.ndarray, labels: np.ndarray, iters: int = 200) -> float:
    """Minimize binary NLL over T>0 by 1-D Newton/grid refine (no torch needed)."""
    logits = logits.astype(np.float64)
    labels = labels.astype(np.float64)

    def nll(T):
        z = logits / T
        # stable log-sigmoid
        return np.mean(np.logaddexp(0.0, z) - labels * z)

    # coarse grid then local refine
    grid = np.linspace(0.25, 8.0, 80)
    T = float(grid[np.argmin([nll(t) for t in grid])])
    lo, hi = max(0.05, T - 0.2), T + 0.2
    for _ in range(iters):
        mid1, mid2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(mid1) < nll(mid2):
            hi = mid2
        else:
            lo = mid1
    return float((lo + hi) / 2)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / T))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    N = len(p)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.sum() / N) * abs(acc - conf)
    return float(e)


def save(T: float, path: str | None = None) -> str:
    path = path or os.path.join(C.ARTIFACTS_DIR, "calibration.json")
    with open(path, "w") as f:
        json.dump({"temperature": T}, f, indent=2)
    return path


def load(path: str | None = None) -> float:
    path = path or os.path.join(C.ARTIFACTS_DIR, "calibration.json")
    with open(path) as f:
        return float(json.load(f)["temperature"])


if __name__ == "__main__":
    from . import io_load, evaluate
    device = evaluate._device()
    model, normalizer, calibrator = evaluate.load_artifacts(device)
    recs = io_load.list_recordings()

    # VAL logits/labels (fit T), TEST logits/labels (report)
    val = [evaluate.dense_infer(s, recs[s], model, normalizer, calibrator, device, hop=5)
           for s in C.SPLIT["val"]]
    val = [r for r in val if r["n"]]
    vlogit = np.concatenate([r["bin_logit"] for r in val])
    vy = np.concatenate([r["y_bin"] for r in val])
    test = [evaluate.dense_infer(s, recs[s], model, normalizer, calibrator, device, hop=1)
            for s in C.SPLIT["test"]]
    tlogit = np.concatenate([r["bin_logit"] for r in test])
    ty = np.concatenate([r["y_bin"] for r in test])

    T = fit_temperature(vlogit, vy)
    save(T)
    p_before = 1.0 / (1.0 + np.exp(-tlogit))
    p_after = apply_temperature(tlogit, T)
    print(f"Fitted temperature T = {T:.3f}  (T>1 ⇒ was over-confident)")
    print(f"{'':12s}{'Brier':>10s}{'ECE':>10s}")
    print(f"{'uncalib':12s}{brier(p_before,ty):>10.4f}{ece(p_before,ty):>10.4f}")
    print(f"{'temp-scaled':12s}{brier(p_after,ty):>10.4f}{ece(p_after,ty):>10.4f}")
    # Ranking invariance: temperature scaling is strictly monotonic in the logit,
    # so AUC on logits is exactly preserved. (AUC on the *probabilities* can wiggle
    # at the 1e-3 level only because the raw sigmoid saturates to 1.0 for large
    # logits -> ties; de-saturating via T removes those ties. That is a feature.)
    from sklearn.metrics import roc_auc_score
    auc_logit = roc_auc_score(ty, tlogit)
    auc_scaled_logit = roc_auc_score(ty, tlogit / T)
    sat = float((p_before >= 0.999999).mean())
    print(f"ROC-AUC on logits (rank-true): {auc_logit:.4f} == {auc_scaled_logit:.4f} "
          f"(Δ={abs(auc_logit-auc_scaled_logit):.2e})  ⇒ ranking preserved")
    print(f"raw probs saturated at ~1.0: {sat*100:.1f}% of TEST windows "
          f"(why thresholding was brittle)")
    assert abs(auc_logit - auc_scaled_logit) < 1e-12, "monotone scaling preserves logit-AUC"
    print(f"saved calibration.json (T={T:.3f})")
