"""Per-channel z-score normalization, fit on TRAIN rows only.

Stats are computed over all resampled rows of the training recordings (row-level,
not window-level, so heavy window overlap doesn't bias the stats). Persisted to
artifacts/normalize.json so evaluate.py and infer.py use identical parameters.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C

STD_FLOOR = 1e-6


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, channels: list[str]):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), STD_FLOOR)
        self.channels = channels

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Normalize X with shape [..., C, T] (channel axis = -2)."""
        m = self.mean[:, None]
        s = self.std[:, None]
        return (X - m) / s

    # ---- persistence ----
    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(C.ARTIFACTS_DIR, "normalize.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "channels": self.channels,
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "target_hz": C.TARGET_HZ,
                "window_samples": C.WINDOW_SAMPLES,
                "use_engineered": C.USE_ENGINEERED,
            }, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "Normalizer":
        path = path or os.path.join(C.ARTIFACTS_DIR, "normalize.json")
        with open(path) as f:
            d = json.load(f)
        return cls(np.array(d["mean"]), np.array(d["std"]), d["channels"])


def fit_from_train(train_stems: list[str]) -> Normalizer:
    """Streaming mean/std over all resampled rows of the train recordings."""
    from . import io_load, resample, features
    recs = io_load.list_recordings()
    n = 0
    s1 = None  # sum
    s2 = None  # sum of squares
    for stem in train_stems:
        res_df, _ = resample.cached_resample(recs[stem])
        X = features.build_channel_matrix(res_df)  # [T, C]
        if s1 is None:
            s1 = np.zeros(X.shape[1], dtype=np.float64)
            s2 = np.zeros(X.shape[1], dtype=np.float64)
        s1 += X.sum(axis=0)
        s2 += (X.astype(np.float64) ** 2).sum(axis=0)
        n += X.shape[0]
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    return Normalizer(mean, np.sqrt(var), C.channel_names())


if __name__ == "__main__":
    norm = fit_from_train(C.SPLIT["train"])
    p = norm.save()
    print(f"Fit normalizer on {len(C.SPLIT['train'])} train recordings -> {p}")
    print(f"channels={len(norm.channels)}  mean[:3]={norm.mean[:3]}  std[:3]={norm.std[:3]}")
    # round-trip
    norm2 = Normalizer.load()
    assert np.allclose(norm.mean, norm2.mean) and np.allclose(norm.std, norm2.std)
    # a normalized train-ish sample should be roughly zero-mean / unit-std overall
    from . import io_load, resample, features
    recs = io_load.list_recordings()
    df = io_load.load_recording(recs["walking1"])
    res_df, _ = resample.resample_recording(df)
    X = features.build_channel_matrix(res_df).T[None]  # [1, C, T]
    Z = norm.apply(X)
    print(f"normalized walking1: mean={Z.mean():.3f} std={Z.std():.3f}")
    print("Normalize OK (save/load round-trip).")
