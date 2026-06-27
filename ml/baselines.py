"""Baselines the temporal model must beat, on the SAME windows/split/metrics.

1. GBM  : sklearn HistGradientBoostingClassifier on window-averaged channels
          (each window -> mean over time of every channel -> [C] feature vector).
2. Heuristic: a reproducible re-implementation of the existing detector's core
          signal -- per-leg thigh_tau_down z-scored against the TRAIN-walking
          baseline (thigh_tau_down has weight 5.0 in live_entanglement_detector.py,
          the dominant entanglement cue). Detection = max-leg z-score >= threshold,
          threshold chosen on VAL for ~1% false-alarm rate. This avoids depending
          on the original script's venv while mirroring its decision rule.

Run:
    python -m dataset.ml.baselines
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import io_load, resample, features
from .windowing import make_windows


def _window_mean_features(stems, hop, training):
    """Return (Xmean[N,C], y_bin[N]) -- per-window channel means."""
    recs = io_load.list_recordings()
    Xs, ys = [], []
    for stem in stems:
        df = io_load.load_recording(recs[stem])
        res_df, status = resample.resample_recording(df)
        X = features.build_channel_matrix(res_df)
        ws = make_windows(X, status, stem, C.parse_legs(stem), hop=hop)
        if ws.X.shape[0] == 0:
            continue
        keep = ws.train_mask if training else np.ones(len(ws.y_bin), dtype=bool)
        Xs.append(ws.X[keep].mean(axis=2))   # mean over time -> [n, C]
        ys.append(ws.y_bin[keep])
    if not Xs:
        return np.zeros((0, C.n_channels())), np.zeros((0,))
    return np.concatenate(Xs), np.concatenate(ys)


def _f1(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def gbm_baseline():
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr = _window_mean_features(C.SPLIT["train"], C.HOP_TRAIN, training=True)
    Xte, yte = _window_mean_features(C.SPLIT["test"], C.HOP_EVAL, training=False)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        class_weight="balanced", random_state=C.SEED)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    # threshold maximizing F1 on test (generous to the baseline)
    best = max((_f1(yte, p, t)[2], t) for t in np.linspace(0.1, 0.9, 33))
    prec, rec, f1 = _f1(yte, p, best[1])
    return {"P": prec, "R": rec, "F1": f1, "n_train": len(ytr), "n_test": len(yte)}


def heuristic_baseline(target_far=0.01):
    """Per-leg thigh_tau_down z-score vs TRAIN-walking baseline; max-leg score detects."""
    recs = io_load.list_recordings()

    # 1) fit per-leg baseline mean/std of thigh_tau_down on TRAIN walking windows
    vals = {leg: [] for leg in C.LEG_ORDER}
    for stem in C.SPLIT["train"]:
        df = io_load.load_recording(recs[stem])
        res_df, status = resample.resample_recording(df)
        walk = (status == "Walking")
        for end in range(C.WINDOW_SAMPLES - 1, len(res_df), C.HOP_TRAIN):
            start = end - C.WINDOW_SAMPLES + 1
            if not walk[start:end + 1].all():
                continue
            for leg in C.LEG_ORDER:
                tt = res_df[f"{leg}_thigh_tau"].to_numpy()[start:end + 1]
                vals[leg].append(np.maximum(0.0, -tt).mean())
    base = {leg: (np.mean(v), max(np.std(v), 1e-6)) for leg, v in vals.items()}

    def score_windows(stem, hop, training=False):
        df = io_load.load_recording(recs[stem])
        res_df, status = resample.resample_recording(df)
        X = features.build_channel_matrix(res_df)
        ws = make_windows(X, status, stem, C.parse_legs(stem), hop=hop)
        scores, ys = [], []
        for r, end in enumerate(ws.end_idx):
            start = int(end) - C.WINDOW_SAMPLES + 1
            z = []
            for leg in C.LEG_ORDER:
                tt = res_df[f"{leg}_thigh_tau"].to_numpy()[start:int(end) + 1]
                m = np.maximum(0.0, -tt).mean()
                mu, sd = base[leg]
                z.append((m - mu) / sd)
            scores.append(max(z))
            ys.append(ws.y_bin[r])
        return np.array(scores), np.array(ys)

    # 2) threshold on VAL for target FAR
    val_s, val_y = [], []
    for stem in C.SPLIT["val"]:
        s, y = score_windows(stem, hop=5)
        val_s.append(s); val_y.append(y)
    val_s = np.concatenate(val_s); val_y = np.concatenate(val_y)
    neg = val_s[val_y == 0]
    thr = float(np.quantile(neg, 1.0 - target_far)) if len(neg) else 3.0

    # 3) evaluate on TEST
    te_s, te_y = [], []
    for stem in C.SPLIT["test"]:
        s, y = score_windows(stem, hop=1)
        te_s.append(s); te_y.append(y)
    te_s = np.concatenate(te_s); te_y = np.concatenate(te_y)
    prec, rec, f1 = _f1(te_y, te_s, thr)
    return {"P": prec, "R": rec, "F1": f1, "threshold": thr}


if __name__ == "__main__":
    print("Training GBM baseline (window-mean features)...")
    gbm = gbm_baseline()
    print(f"  GBM        P={gbm['P']:.3f} R={gbm['R']:.3f} F1={gbm['F1']:.3f}  "
          f"(train {gbm['n_train']} / test {gbm['n_test']} windows)")
    print("Computing heuristic-style baseline (thigh-torque z-score)...")
    heu = heuristic_baseline()
    print(f"  Heuristic  P={heu['P']:.3f} R={heu['R']:.3f} F1={heu['F1']:.3f}  "
          f"(thr={heu['threshold']:.2f})")
    print("\nCompare against the TCN's pooled-test detection F1 from evaluate.py (~0.85).")
