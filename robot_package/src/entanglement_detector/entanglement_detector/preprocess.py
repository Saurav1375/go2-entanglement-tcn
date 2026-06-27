"""Window -> normalized 60-channel tensor. Pure numpy, Python 3.8 compatible.

Reproduces the research pipeline's `features.build_channel_matrix` +
`normalize.Normalizer.apply` exactly, but operating on a raw [W, 48] window
array (no pandas).
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from . import constants as K

STD_FLOOR = 1e-6


def _raw_index():
    # type: () -> Dict[str, int]
    return {name: i for i, name in enumerate(K.raw_channel_names())}


_RAW_IDX = _raw_index()


def build_channel_matrix(raw_window):
    # type: (np.ndarray) -> np.ndarray
    """raw_window: [W, 48] in raw_channel_names() order -> [60, W] float32.

    Raw channels pass through; engineered channels reproduce the exact formulas:
      thigh_tau_down    = max(0, -thigh_tau)
      thigh_down_effort = thigh_tau_down / (|thigh_dq| + EFFORT_DQ_EPSILON)
      tau_sum           = |hip_tau| + |thigh_tau| + |calf_tau|
    """
    raw = np.asarray(raw_window, dtype=np.float32)          # [W, 48]
    cols = [raw[:, i] for i in range(K.N_RAW)]              # raw channels (48)
    for leg in K.LEG_ORDER:
        thigh_tau = raw[:, _RAW_IDX["{}_thigh_tau".format(leg)]]
        thigh_dq = raw[:, _RAW_IDX["{}_thigh_dq".format(leg)]]
        hip_tau = raw[:, _RAW_IDX["{}_hip_tau".format(leg)]]
        calf_tau = raw[:, _RAW_IDX["{}_calf_tau".format(leg)]]
        down = np.maximum(0.0, -thigh_tau)
        cols.append(down)
        cols.append(down / (np.abs(thigh_dq) + K.EFFORT_DQ_EPSILON))
        cols.append(np.abs(hip_tau) + np.abs(thigh_tau) + np.abs(calf_tau))
    mat = np.stack(cols, axis=1).astype(np.float32)        # [W, 60]
    return mat.T                                            # [60, W]


class Normalizer(object):
    """Per-channel z-score with stats loaded from normalize.json."""

    def __init__(self, mean, std, channels):
        # type: (np.ndarray, np.ndarray, List[str]) -> None
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), STD_FLOOR).reshape(-1, 1)
        self.channels = channels

    @classmethod
    def load(cls, path):
        # type: (str) -> "Normalizer"
        with open(path) as f:
            d = json.load(f)
        norm = cls(np.array(d["mean"]), np.array(d["std"]), d["channels"])
        if len(d["channels"]) != K.N_CHANNELS:
            raise ValueError("normalize.json has {} channels, expected {}".format(
                len(d["channels"]), K.N_CHANNELS))
        return norm

    def apply(self, mat):
        # type: (np.ndarray) -> np.ndarray
        """mat: [60, W] -> normalized [60, W] float32."""
        return ((mat - self.mean) / self.std).astype(np.float32)
