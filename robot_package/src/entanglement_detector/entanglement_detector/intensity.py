"""Physics-grounded 0-1 per-leg intensity magnitude. Pure numpy, Python 3.8.

Reproduces the research `intensity.IntensityCalibrator` magnitude (the ungated
sqrt(d*r)); the engine multiplies by the per-leg probability gate.

Per leg, over the window:
  f = [thigh_tau_down.mean, thigh_down_effort.mean, tau_sum.mean,
       foot_force.mean, |thigh_dq|.mean]
  D = sqrt((f-mean) . cov_inv . (f-mean));  d = sigmoid(a_D*(D-q95_D)/iqr_D)
  R = thigh_down_effort.mean;               r = sigmoid(a_R*(R-q95_R)/iqr_R)
  magnitude = sqrt(d * r)
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np

from . import constants as K
from .preprocess import _RAW_IDX


def _sigmoid(z):
    # type: (np.ndarray) -> np.ndarray
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class IntensityCalibrator(object):
    def __init__(self, params):
        # type: (dict) -> None
        self.a_D = float(params["a_D"])
        self.a_R = float(params["a_R"])
        self.legs = {}  # type: Dict[str, dict]
        for leg in K.LEG_ORDER:
            p = params["legs"][leg]
            self.legs[leg] = {
                "mean": np.asarray(p["mean"], dtype=np.float64),
                "cov_inv": np.asarray(p["cov_inv"], dtype=np.float64),
                "q95_D": float(p["q95_D"]), "iqr_D": float(p["iqr_D"]),
                "q95_R": float(p["q95_R"]), "iqr_R": float(p["iqr_R"]),
            }

    @classmethod
    def load(cls, path):
        # type: (str) -> "IntensityCalibrator"
        with open(path) as f:
            return cls(json.load(f))

    def _leg_feature(self, raw_window, leg):
        # type: (np.ndarray, str) -> tuple
        thigh_tau = raw_window[:, _RAW_IDX["{}_thigh_tau".format(leg)]]
        thigh_dq = raw_window[:, _RAW_IDX["{}_thigh_dq".format(leg)]]
        hip_tau = raw_window[:, _RAW_IDX["{}_hip_tau".format(leg)]]
        calf_tau = raw_window[:, _RAW_IDX["{}_calf_tau".format(leg)]]
        foot = raw_window[:, _RAW_IDX["foot_{}".format(leg)]]
        down = np.maximum(0.0, -thigh_tau)
        effort = down / (np.abs(thigh_dq) + K.EFFORT_DQ_EPSILON)
        tau_sum = np.abs(hip_tau) + np.abs(thigh_tau) + np.abs(calf_tau)
        f = np.array([down.mean(), effort.mean(), tau_sum.mean(),
                      foot.mean(), np.abs(thigh_dq).mean()], dtype=np.float64)
        return f, float(effort.mean())

    def magnitudes(self, raw_window):
        # type: (np.ndarray) -> np.ndarray
        """raw_window: [W, 48] -> per-leg magnitude [4] in LEG_ORDER, each in [0,1]."""
        raw = np.asarray(raw_window, dtype=np.float64)
        out = np.zeros(K.N_LEGS, dtype=np.float32)
        for j, leg in enumerate(K.LEG_ORDER):
            f, R = self._leg_feature(raw, leg)
            p = self.legs[leg]
            diff = f - p["mean"]
            D = float(np.sqrt(max(diff.dot(p["cov_inv"]).dot(diff), 0.0)))
            d = _sigmoid(np.array(self.a_D * (D - p["q95_D"]) / p["iqr_D"]))
            r = _sigmoid(np.array(self.a_R * (R - p["q95_R"]) / p["iqr_R"]))
            out[j] = float(np.sqrt(d * r))
        return out
