"""Streaming entanglement inference engine. Pure numpy + a model backend.

Mirrors the research `ml/infer.py::InferenceEngine` exactly:
  ring buffer (200 samples) -> 60-channel matrix -> normalize -> backend ->
  sigmoids -> temperature-scaled confidence -> physics intensity (gated) ->
  raw-threshold debounce -> per-leg thresholds.

No pandas / sklearn / scipy / torch (torch only if the TorchScript backend is
chosen). Python 3.8 compatible.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from . import constants as K
from .preprocess import Normalizer, build_channel_matrix
from .intensity import IntensityCalibrator
from .model_backend import load_backend


def _sigmoid(x):
    # type: (float) -> float
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    e = np.exp(x)
    return float(e / (1.0 + e))


class EntanglementEngine(object):
    def __init__(self, model_path, normalize_path, intensity_calib_path,
                 temperature=1.0, detection_threshold=0.9999, debounce_ms=75,
                 leg_thresholds=None, intensity_blend=0.5, num_threads=1):
        # type: (str, str, str, float, float, float, Optional[Dict[str, float]], float, int) -> None
        self.backend = load_backend(model_path, num_threads=num_threads)
        self.normalizer = Normalizer.load(normalize_path)
        self.calibrator = IntensityCalibrator.load(intensity_calib_path)
        self.temperature = float(temperature)
        self.det_threshold = float(detection_threshold)     # RAW probability threshold
        self.debounce_k = max(1, int(round(debounce_ms / 1000.0 * K.TARGET_HZ)))
        self.leg_thresholds = dict(leg_thresholds or {})
        self.intensity_blend = float(intensity_blend)
        self.raw_names = K.raw_channel_names()
        self.buf = deque(maxlen=K.WINDOW_SAMPLES)            # type: deque
        self._run = 0

    def reset(self):
        # type: () -> None
        self.buf.clear()
        self._run = 0

    def push(self, sample):
        # type: (Dict[str, float]) -> Optional[dict]
        """Feed one /lowstate-shaped sample. Returns a state dict, or None until
        the 200-sample buffer fills."""
        self.buf.append([float(sample[name]) for name in self.raw_names])
        if len(self.buf) < K.WINDOW_SAMPLES:
            return None

        raw = np.asarray(self.buf, dtype=np.float32)            # [200, 48]
        mat = build_channel_matrix(raw)                         # [60, 200]
        x = self.normalizer.apply(mat)[None]                    # [1, 60, 200]

        bin_logit, legs_logit, inten_logit = self.backend.forward(x)
        logit = float(bin_logit[0])
        p_bin = _sigmoid(logit)                                 # RAW probability
        p_bin_cal = _sigmoid(logit / self.temperature)          # calibrated confidence
        p_legs = 1.0 / (1.0 + np.exp(-np.clip(legs_logit[0], -60, 60)))
        head_int = 1.0 / (1.0 + np.exp(-np.clip(inten_logit[0], -60, 60)))

        mags = self.calibrator.magnitudes(raw)                  # [4]
        phys_I = p_legs * mags
        intensity = self.intensity_blend * head_int + (1.0 - self.intensity_blend) * phys_I

        above = p_bin >= self.det_threshold
        self._run = self._run + 1 if above else 0
        alarm = self._run >= self.debounce_k

        alarm_leg = None  # type: Optional[str]
        if alarm:
            best_j, best_p = -1, -1.0
            for j, leg in enumerate(K.LEG_ORDER):
                thr = float(self.leg_thresholds.get(leg, 0.5))
                if p_legs[j] >= thr and p_legs[j] > best_p:
                    best_j, best_p = j, float(p_legs[j])
            alarm_leg = K.LEG_ORDER[best_j] if best_j >= 0 else None

        return {
            "entangled": bool(alarm),
            "confidence": float(p_bin_cal),
            "p_bin_raw": float(p_bin),
            "alarm_leg": alarm_leg,
            "p_legs": {leg: float(p_legs[j]) for j, leg in enumerate(K.LEG_ORDER)},
            "intensity": {leg: float(intensity[j]) for j, leg in enumerate(K.LEG_ORDER)},
        }
