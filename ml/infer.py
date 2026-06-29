"""Streaming inference engine — the clean path to a live /lowstate node.

`InferenceEngine.push(sample)` takes ONE timestep as a dict keyed by the CSV /
lowstate field names (e.g. "FR_hip_q", ..., "foot_FL", "roll", "gyro_x", ...),
maintains a ring buffer of the last WINDOW_SAMPLES samples, and once full returns
per-step predictions. A future ROS2 node only has to map unitree_go/msg/LowState
-> this dict (using config.MOTOR_INDEX / CSV_FOOT_ORDER); no model logic lives in
the node. All ordering constants come from config.py, mirroring the reuse-source.

The engine assumes samples arrive already on the TARGET_HZ grid (a live node
resamples-by-latest before pushing). Fed the resampled rows of a recording, it
reproduces evaluate.dense_infer's per-window outputs exactly (see __main__).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import torch

from . import config as C
from .intensity import IntensityCalibrator
from .model import EntanglementTCN
from .normalize import Normalizer


def _load_op():
    """Load artifacts/operating_point.json (recommended post-hoc settings) or None."""
    import json
    import os
    path = os.path.join(C.ARTIFACTS_DIR, "operating_point.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class InferenceEngine:
    def __init__(self, model, normalizer: Normalizer, calibrator: IntensityCalibrator,
                 device: str = "cpu", intensity_blend: float = 0.5):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.calibrator = calibrator
        self.device = device
        self.intensity_blend = intensity_blend
        self.raw_names = C.raw_channel_names()
        self._dq_idx = [self.raw_names.index(f"{leg}_{j}_dq")
                        for leg in C.LEG_ORDER for j in C.JOINT_ORDER]
        self.buf: deque = deque(maxlen=C.WINDOW_SAMPLES)

        # Recommended operating point (post-hoc; written by improvements.py). Defaults are
        # the raw, uncalibrated behaviour if no operating_point.json exists.
        op = _load_op() or {}
        self.temperature = float(op.get("temperature", 1.0))            # calibrated confidence
        # alarm latch uses the RAW probability vs the RAW VAL-1%FAR threshold (§4-validated)
        self.det_threshold = float(op.get("detection_threshold_raw", 0.5))
        self.debounce_k = max(1, round(op.get("debounce_ms", 0) / 1000 * C.TARGET_HZ))
        self.leg_thresholds = op.get("leg_thresholds", {})
        # ---- stationarity (Stop/Lock) gate; disabled by default (thresh 0 -> never fires) ----
        self.stationary_dq_thresh = float(op.get("stationary_dq_thresh", 0.0))
        self.stationary_min_k = max(1, round(op.get("stationary_min_ms", 100) / 1000 * C.TARGET_HZ))
        self._stabilize_k = max(0, round(op.get("stabilize_ms", 0) / 1000 * C.TARGET_HZ))
        self._run = 0  # consecutive above-threshold windows (debounce state)
        self._stationary_run = 0
        self._reset_armed = False
        self._stabilize_left = 0

    @classmethod
    def from_artifacts(cls, device: str = "cpu", intensity_blend: float = 0.5):
        import os
        ckpt = torch.load(os.path.join(C.ARTIFACTS_DIR, "model.pt"),
                          map_location=device, weights_only=False)
        model = EntanglementTCN(in_channels=ckpt["config"]["n_channels"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, Normalizer.load(), IntensityCalibrator.load(),
                   device=device, intensity_blend=intensity_blend)

    def reset(self):
        self.buf.clear()
        self._run = 0
        self._stationary_run = 0
        self._reset_armed = False
        self._stabilize_left = 0

    @torch.no_grad()
    def push(self, sample: dict) -> dict | None:
        """Feed one timestep. Returns a prediction dict, or None until the buffer fills."""
        row = [float(sample[name]) for name in self.raw_names]
        # stationarity (Stop/Lock) gate — edge-triggered reset of stale temporal context;
        # disabled when stationary_dq_thresh == 0 (default) so equivalence/validation hold.
        if self.stationary_dq_thresh > 0.0:
            mean_abs_dq = sum(abs(row[i]) for i in self._dq_idx) / len(self._dq_idx)
            if mean_abs_dq < self.stationary_dq_thresh:
                self._stationary_run += 1
                if self._stationary_run == self.stationary_min_k:
                    self.buf.clear(); self._run = 0; self._reset_armed = True
            else:
                if self._stationary_run >= self.stationary_min_k:
                    self._stabilize_left = self._stabilize_k
                self._stationary_run = 0
                self._reset_armed = False
                if self._stabilize_left > 0:
                    self._stabilize_left -= 1
        self.buf.append(row)
        if len(self.buf) < C.WINDOW_SAMPLES:
            return None

        raw = np.asarray(self.buf, dtype=np.float64)          # [W, n_raw]
        win_df = pd.DataFrame(raw, columns=self.raw_names)     # window-local frame

        # channel matrix [C, W] (raw + engineered), then normalize
        from .features import build_channel_matrix
        X = build_channel_matrix(win_df).T                     # [C, W]
        Xn = self.normalizer.apply(X[None]).astype(np.float32)  # [1, C, W]

        out = self.model(torch.from_numpy(Xn).to(self.device))
        logit = float(out["bin_logit"][0])
        p_bin = float(torch.sigmoid(out["bin_logit"])[0])          # RAW (unchanged)
        p_bin_cal = 1.0 / (1.0 + np.exp(-logit / self.temperature))  # calibrated
        p_legs = torch.sigmoid(out["legs_logit"])[0].cpu().numpy()
        head_int = torch.sigmoid(out["intensity_logit"])[0].cpu().numpy()

        # physics intensity: gate (per-leg prob) * magnitude on the window
        mags = self.calibrator.magnitudes_all_legs(win_df, 0, C.WINDOW_SAMPLES - 1)
        phys_I = p_legs * mags
        intensity = self.intensity_blend * head_int + (1.0 - self.intensity_blend) * phys_I

        # debounced alarm on RAW prob vs raw detection threshold (saturates during
        # entanglement -> latches reliably); p_bin_cal is the reported confidence only
        above = p_bin >= self.det_threshold
        self._run = self._run + 1 if above else 0
        alarm = (self._run >= self.debounce_k) and (self._stabilize_left == 0)

        # alarm leg using per-leg thresholds (RR raised to curb its false positives)
        alarm_leg = None
        if alarm:
            best_j, best_p = None, -1.0
            for j, leg in enumerate(C.LEG_ORDER):
                thr = float(self.leg_thresholds.get(leg, 0.5))
                if p_legs[j] >= thr and p_legs[j] > best_p:
                    best_j, best_p = j, p_legs[j]
            alarm_leg = C.LEG_ORDER[best_j] if best_j is not None else None

        return {
            "p_bin": p_bin,                 # raw probability (reproduces evaluate)
            "p_bin_cal": float(p_bin_cal),  # temperature-calibrated probability
            "alarm": bool(alarm),           # debounced detection decision
            "p_legs": {leg: float(p_legs[j]) for j, leg in enumerate(C.LEG_ORDER)},
            "intensity": {leg: float(intensity[j]) for j, leg in enumerate(C.LEG_ORDER)},
            "alarm_leg": alarm_leg,
        }


if __name__ == "__main__":
    # Verification: streaming the resampled rows of a test recording reproduces
    # evaluate.dense_infer's per-window outputs (same windowing, same model).
    from . import io_load, resample, evaluate

    # CPU for the equivalence check: deterministic and batch-size-independent, so
    # streaming (batch=1) and dense_infer (batched) must match to float precision.
    # (On GPU, cuDNN may select different conv kernels per batch size -> ~1e-3 drift,
    #  which is acceptable for deployment but would make an exact assert flaky.)
    device = "cpu"
    eng = InferenceEngine.from_artifacts(device=device)
    model, normalizer, calibrator = evaluate.load_artifacts(device)

    stem = "back_left_hand2"
    path = io_load.list_recordings()[stem]
    ref = evaluate.dense_infer(stem, path, model, normalizer, calibrator, device, hop=1)

    # feed resampled raw rows one-by-one
    df = io_load.load_recording(path)
    res_df, _ = resample.resample_recording(df)
    raw_names = C.raw_channel_names()
    stream_pbin = {}
    for i in range(len(res_df)):
        sample = {name: res_df[name].iloc[i] for name in raw_names}
        r = eng.push(sample)
        if r is not None:
            stream_pbin[i] = r  # i is the window's last-sample index == end_idx

    # compare on the dense windows (end_idx == row index since hop=1)
    diffs = []
    for k, end in enumerate(ref["end_idx"]):
        rr = stream_pbin.get(int(end))
        if rr is None:
            continue
        diffs.append(abs(rr["p_bin"] - ref["p_bin"][k]))
        leg_d = max(abs(rr["intensity"][leg] - ref["intensity"][k, j])
                    for j, leg in enumerate(C.LEG_ORDER))
        diffs.append(leg_d)
    diffs = np.array(diffs)
    print(f"streamed {len(stream_pbin)} windows vs dense_infer; "
          f"max |p_bin/intensity diff| = {diffs.max():.2e}")
    assert diffs.max() < 1e-4, "streaming must reproduce batched dense inference"

    # show the operating point and the first DEBOUNCED alarm (calibrated + 75 ms debounce)
    print(f"operating point: T={eng.temperature:.2f}  det_thr={eng.det_threshold:.3f}  "
          f"debounce={eng.debounce_k} windows  leg_thr(RR)={eng.leg_thresholds.get('RR')}")
    onset_t = ref["onset_idx"] / C.TARGET_HZ
    first = next((i for i in sorted(stream_pbin) if stream_pbin[i]["alarm"]), None)
    if first is not None:
        r = stream_pbin[first]
        kind = "PRE-ONSET (false/early)" if first / C.TARGET_HZ < onset_t else "true"
        print(f"  GT onset t={onset_t:.2f}s; first debounced ALARM t={first/C.TARGET_HZ:.2f}s "
              f"[{kind}] leg={r['alarm_leg']}")
        # behavior DURING the true entanglement (affected leg = RL here)
        ent_idx = [i for i in stream_pbin if i >= ref["onset_idx"] and stream_pbin[i]["alarm"]]
        from collections import Counter
        legs = Counter(stream_pbin[i]["alarm_leg"] for i in ent_idx)
        print(f"  during GT entanglement: {len(ent_idx)} alarmed windows, alarm-leg vote={dict(legs)} "
              f"(affected={sorted(ref['affected'])})")
        n_alarm = sum(1 for v in stream_pbin.values() if v["alarm"])
        print(f"  total alarmed windows: {n_alarm}/{len(stream_pbin)} "
              f"(includes pre-onset early fires — see FAR analysis)")
    else:
        print("  (no debounced alarm latched on this file at the recommended operating point)")
    print("Infer OK (streaming reproduces evaluate; live contract works).")
