"""Torch Dataset / DataLoader over windowed recordings.

A WindowDataset holds the concatenated, normalized windows for a list of
recordings (one split). It also computes per-window sampling weights so a
WeightedRandomSampler can balance positives/negatives and up-weight rare legs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import config as C
from . import io_load, resample, features
from .normalize import Normalizer
from .windowing import make_windows


class WindowDataset(Dataset):
    def __init__(self, stems: list[str], normalizer: Normalizer, hop: int,
                 training: bool, augment: bool = False, calibrator=None):
        recs = io_load.list_recordings()
        self.training = training
        self.augment = augment

        Xs, ybin, ylegs, src, itgt = [], [], [], [], []
        self.recording_meta: list[dict] = []
        for stem in stems:
            res_df, status = resample.cached_resample(recs[stem])
            X = features.build_channel_matrix(res_df)
            ws = make_windows(X, status, stem, C.parse_legs(stem), hop)
            if len(ws.y_bin) == 0:
                continue
            keep = ws.train_mask if training else np.ones(len(ws.y_bin), dtype=bool)
            if keep.sum() == 0:
                continue
            Xs.append(ws.X[keep])
            ybin.append(ws.y_bin[keep])
            ylegs.append(ws.y_legs[keep])
            src.append(np.full(keep.sum(), len(self.recording_meta), dtype=np.int64))
            self.recording_meta.append({"stem": stem, "onset_idx": ws.onset_idx})

            # intensity pseudo-target: physics magnitude on AFFECTED legs only, 0 elsewhere
            if calibrator is not None:
                ends = ws.end_idx[keep]
                legs = ws.y_legs[keep]
                tg = np.zeros((len(ends), C.N_LEGS), dtype=np.float32)
                for r, end in enumerate(ends):
                    if legs[r].sum() > 0:  # positive window
                        start = int(end) - C.WINDOW_SAMPLES + 1
                        mags = calibrator.magnitudes_all_legs(res_df, start, int(end))
                        tg[r] = mags * legs[r]
                itgt.append(tg)

        if Xs:
            self.X = np.concatenate(Xs).astype(np.float32)
            self.y_bin = np.concatenate(ybin).astype(np.float32)
            self.y_legs = np.concatenate(ylegs).astype(np.float32)
            self.src = np.concatenate(src)
            self.itgt = np.concatenate(itgt).astype(np.float32) if itgt else None
        else:
            cc = C.n_channels()
            self.X = np.zeros((0, cc, C.WINDOW_SAMPLES), np.float32)
            self.y_bin = np.zeros((0,), np.float32)
            self.y_legs = np.zeros((0, C.N_LEGS), np.float32)
            self.src = np.zeros((0,), np.int64)
            self.itgt = None

        self.X = normalizer.apply(self.X).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y_bin)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.training and self.augment:
            x = self._augment(x)
        itgt = self.itgt[i] if self.itgt is not None else np.zeros(C.N_LEGS, dtype=np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.tensor(self.y_bin[i]),
                torch.from_numpy(self.y_legs[i]),
                torch.from_numpy(itgt),
                int(self.src[i]))

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # light gaussian jitter (already normalized scale) -- robustness to sensor noise
        return x + np.random.normal(0.0, 0.02, size=x.shape).astype(np.float32)

    # ---- balanced sampling ----
    def sample_weights(self) -> np.ndarray:
        """50/50 pos/neg, with rare-leg (esp. FR) up-weighting inside positives."""
        n = len(self.y_bin)
        if n == 0:
            return np.zeros(0)
        w = np.ones(n, dtype=np.float64)
        pos = self.y_bin == 1
        neg = ~pos
        npos, nneg = int(pos.sum()), int(neg.sum())
        if npos > 0:
            w[pos] = 0.5 / npos
        if nneg > 0:
            w[neg] = 0.5 / nneg
        # up-weight rare legs among positives (inverse per-leg window frequency)
        if npos > 0:
            leg_counts = self.y_legs[pos].sum(axis=0)  # [4]
            leg_counts = np.maximum(leg_counts, 1.0)
            inv = leg_counts.max() / leg_counts          # FR (rarest) gets largest factor
            factor = (self.y_legs * inv[None, :]).max(axis=1)
            factor[factor == 0] = 1.0
            w[pos] *= factor[pos]
        return w

    def pos_weights(self) -> tuple[float, np.ndarray]:
        """BCE pos_weight for the binary head and per-leg heads (capped)."""
        n = len(self.y_bin)
        npos = float(self.y_bin.sum())
        bin_pw = min(C.POS_WEIGHT_CAP, (n - npos) / max(npos, 1.0)) if npos > 0 else 1.0
        leg_pos = self.y_legs.sum(axis=0)
        leg_pw = np.minimum(C.POS_WEIGHT_CAP,
                            (n - leg_pos) / np.maximum(leg_pos, 1.0))
        return bin_pw, leg_pw.astype(np.float32)


def make_loader(ds: WindowDataset, training: bool) -> DataLoader:
    if len(ds) == 0:
        return DataLoader(ds, batch_size=C.BATCH_SIZE)
    if training:
        weights = ds.sample_weights()
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=C.BATCH_SIZE, sampler=sampler, drop_last=True)
    return DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False)


if __name__ == "__main__":
    norm = Normalizer.load()
    train_ds = WindowDataset(C.SPLIT["train"], norm, hop=C.HOP_TRAIN, training=True, augment=True)
    val_ds = WindowDataset(C.SPLIT["val"], norm, hop=C.HOP_TRAIN, training=False)
    print(f"train windows={len(train_ds)}  pos={int(train_ds.y_bin.sum())}  "
          f"val windows={len(val_ds)}  pos={int(val_ds.y_bin.sum())}")
    bin_pw, leg_pw = train_ds.pos_weights()
    print(f"bin pos_weight={bin_pw:.2f}  per-leg pos_weight={dict(zip(C.LEG_ORDER, leg_pw.round(2)))}")
    loader = make_loader(train_ds, training=True)
    xb, yb, ylb, itb, srcb = next(iter(loader))
    print(f"batch x={tuple(xb.shape)}  y_bin={tuple(yb.shape)}  y_legs={tuple(ylb.shape)}  itgt={tuple(itb.shape)}")
    print(f"batch pos fraction (should be ~0.5 via sampler): {yb.mean().item():.2f}")
    assert xb.shape == (C.BATCH_SIZE, C.n_channels(), C.WINDOW_SAMPLES)
    print("Dataset OK ([B,60,200], balanced sampler).")
