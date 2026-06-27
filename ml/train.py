"""Train the multi-task TCN.

Default: train on the fixed leakage-safe split, validate on VAL, and ship the
artifact (model.pt + normalize.json + intensity_calib.json + split.json).
With --loro: grouped Leave-One-Recording-Out CV over the 12 positive events
(reported as mean±std; the honest generalization estimate).

Usage:
    python -m dataset.ml.train            # fixed split, ship artifacts
    python -m dataset.ml.train --loro     # LORO-CV report
    python -m dataset.ml.train --epochs 5 # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import config as C
from .dataset import WindowDataset, make_loader
from .intensity import IntensityCalibrator
from .losses import MultiTaskLoss
from .model import EntanglementTCN
from .normalize import Normalizer, fit_from_train


def _device() -> str:
    return "cuda" if (C.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"


def set_seed(seed: int = C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_split(model, loader, device) -> dict:
    """Window-level detection F1 + per-leg F1 on a loader (no sampler)."""
    model.eval()
    yb_all, pb_all, yl_all, pl_all = [], [], [], []
    for xb, yb, ylb, itb, _ in loader:
        out = model(xb.to(device))
        pb_all.append(torch.sigmoid(out["bin_logit"]).cpu().numpy())
        pl_all.append(torch.sigmoid(out["legs_logit"]).cpu().numpy())
        yb_all.append(yb.numpy())
        yl_all.append(ylb.numpy())
    if not yb_all:
        return {"f1": 0.0, "leg_f1": 0.0}
    yb = np.concatenate(yb_all); pb = np.concatenate(pb_all)
    yl = np.concatenate(yl_all); pl = np.concatenate(pl_all)

    def f1(y, p, thr=0.5):
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    leg_f1 = float(np.mean([f1(yl[:, j], pl[:, j]) for j in range(C.N_LEGS)]))
    return {"f1": f1(yb, pb), "leg_f1": leg_f1}


def train_one(train_stems, val_stems, normalizer, calibrator, device,
              epochs=C.EPOCHS, verbose=True):
    set_seed()
    train_ds = WindowDataset(train_stems, normalizer, hop=C.HOP_TRAIN,
                             training=True, augment=True, calibrator=calibrator)
    val_ds = WindowDataset(val_stems, normalizer, hop=C.HOP_TRAIN, training=False) \
        if val_stems else None
    train_loader = make_loader(train_ds, training=True)
    val_loader = make_loader(val_ds, training=False) if val_ds and len(val_ds) else None

    bin_pw, leg_pw = train_ds.pos_weights()
    model = EntanglementTCN().to(device)
    loss_fn = MultiTaskLoss(bin_pw, torch.tensor(leg_pw, device=device)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = {"score": -1.0, "f1": 0.0, "leg_f1": 0.0, "epoch": -1}
    best_state = None
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb, ylb, itb, _ in train_loader:
            xb, yb, ylb, itb = (xb.to(device), yb.to(device),
                                ylb.to(device), itb.to(device))
            opt.zero_grad()
            out = model(xb)
            loss, _ = loss_fn(out, yb, ylb, itb)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        if val_loader:
            m = evaluate_split(model, val_loader, device)
            score = 0.5 * m["f1"] + 0.5 * m["leg_f1"]
            if score > best["score"]:
                best = {"score": score, "f1": m["f1"], "leg_f1": m["leg_f1"], "epoch": ep}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep{ep:02d} loss={running/max(len(train_loader),1):.4f} "
                      f"val_f1={m['f1']:.3f} val_leg_f1={m['leg_f1']:.3f}")
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, (bin_pw, leg_pw)


def run_fixed_split(epochs=C.EPOCHS):
    device = _device()
    print(f"[fixed split] device={device}  train={len(C.SPLIT['train'])} "
          f"val={len(C.SPLIT['val'])} test={len(C.SPLIT['test'])}")
    normalizer = fit_from_train(C.SPLIT["train"])
    normalizer.save()
    calibrator = IntensityCalibrator.fit(C.SPLIT["train"])
    calibrator.save()

    model, best, pw = train_one(C.SPLIT["train"], C.SPLIT["val"],
                                normalizer, calibrator, device, epochs=epochs)
    print(f"[fixed split] best val score={best['score']:.3f} "
          f"(f1={best['f1']:.3f} leg_f1={best['leg_f1']:.3f}) @ep{best['epoch']}")

    os.makedirs(C.ARTIFACTS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"n_channels": C.n_channels(),
                           "window_samples": C.WINDOW_SAMPLES,
                           "use_engineered": C.USE_ENGINEERED},
                "bin_pos_weight": float(pw[0]),
                "leg_pos_weight": pw[1].tolist()},
               os.path.join(C.ARTIFACTS_DIR, "model.pt"))
    with open(os.path.join(C.ARTIFACTS_DIR, "split.json"), "w") as f:
        json.dump(C.SPLIT, f, indent=2)
    print(f"[fixed split] saved model.pt, normalize.json, intensity_calib.json, split.json "
          f"-> {C.ARTIFACTS_DIR}")


def run_loro(epochs=C.EPOCHS):
    """Leave-One-Recording-Out CV over the 12 positive events."""
    device = _device()
    recs_all = list(C.SPLIT["train"]) + list(C.SPLIT["val"]) + list(C.SPLIT["test"])
    print(f"[LORO] {len(C.POSITIVE_FILES)} positive events, device={device}")
    scores = []
    for held in C.POSITIVE_FILES:
        train_stems = [s for s in recs_all if s != held]
        normalizer = fit_from_train([s for s in train_stems if not C.parse_legs(s)] or train_stems)
        calibrator = IntensityCalibrator.fit(
            [s for s in train_stems if not C.parse_legs(s)] or train_stems)
        model, _, _ = train_one(train_stems, None, normalizer, calibrator,
                                device, epochs=epochs, verbose=False)
        # evaluate detection on the held-out positive recording
        test_ds = WindowDataset([held], normalizer, hop=C.HOP_EVAL, training=False)
        loader = make_loader(test_ds, training=False)
        m = evaluate_split(model, loader, device)
        scores.append(m["f1"])
        print(f"  held={held:20s} detect_f1={m['f1']:.3f} leg_f1={m['leg_f1']:.3f}")
    scores = np.array(scores)
    print(f"[LORO] detection F1 mean={scores.mean():.3f} std={scores.std():.3f}")
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loro", action="store_true", help="run LORO-CV instead of fixed split")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()
    if args.loro:
        run_loro(epochs=args.epochs)
    else:
        run_fixed_split(epochs=args.epochs)
