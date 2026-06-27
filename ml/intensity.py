"""Physics-grounded 0-1 per-leg intensity (no ground-truth labels).

Per leg, from one window:
  D_leg : Mahalanobis distance of the window-mean per-leg feature vector
          [thigh_tau_down, thigh_down_effort, tau_sum, foot_force, |thigh_dq|]
          to the WALKING distribution (mean/cov fit on TRAIN Walking only).
  R_leg : resistive effort = mean( max(0,-thigh_tau)/(|thigh_dq|+eps) ) in the window.
  g_leg : detection confidence gate (calibrated per-leg probability from the model).

Calibration (fit on TRAIN Walking windows, so normal walking -> ~0):
  d_leg = sigmoid(a_D * (D_leg - q95_D) / iqr_D)
  r_leg = sigmoid(a_R * (R_leg - q95_R) / iqr_R)
  m_leg = sqrt(d_leg * r_leg)              # BOTH deviation and effort must be high
  I_leg = g_leg * m_leg                    # ~0 when not detected

Reported intensity blends the model head with this physics score (config default
0.5/0.5); physics-only is the explainable fallback.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C

# per-leg feature vector used for the Mahalanobis distance
_FEATS = ("thigh_tau_down", "thigh_down_effort", "tau_sum", "foot_force", "abs_thigh_dq")
A_D = 2.0
A_R = 2.0


def per_leg_feature_window(res_df, leg: str, start: int, end: int) -> np.ndarray:
    """Window-mean per-leg feature vector (the 5 _FEATS), inclusive [start, end]."""
    sl = slice(start, end + 1)
    thigh_tau = res_df[f"{leg}_thigh_tau"].to_numpy()[sl]
    thigh_dq = res_df[f"{leg}_thigh_dq"].to_numpy()[sl]
    hip_tau = res_df[f"{leg}_hip_tau"].to_numpy()[sl]
    calf_tau = res_df[f"{leg}_calf_tau"].to_numpy()[sl]
    foot = res_df[f"foot_{leg}"].to_numpy()[sl]
    thigh_tau_down = np.maximum(0.0, -thigh_tau)
    effort = thigh_tau_down / (np.abs(thigh_dq) + C.EFFORT_DQ_EPSILON)
    tau_sum = np.abs(hip_tau) + np.abs(thigh_tau) + np.abs(calf_tau)
    return np.array([thigh_tau_down.mean(), effort.mean(), tau_sum.mean(),
                     foot.mean(), np.abs(thigh_dq).mean()], dtype=np.float64)


def resistive_effort_window(res_df, leg: str, start: int, end: int) -> float:
    sl = slice(start, end + 1)
    thigh_tau = res_df[f"{leg}_thigh_tau"].to_numpy()[sl]
    thigh_dq = res_df[f"{leg}_thigh_dq"].to_numpy()[sl]
    down = np.maximum(0.0, -thigh_tau)
    return float((down / (np.abs(thigh_dq) + C.EFFORT_DQ_EPSILON)).mean())


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class IntensityCalibrator:
    """Holds per-leg Walking statistics + calibration constants."""

    def __init__(self, params: dict):
        self.params = params

    # ---- fit ----
    @classmethod
    def fit(cls, train_stems: list[str]) -> "IntensityCalibrator":
        from . import io_load, resample
        recs = io_load.list_recordings()
        W = C.WINDOW_SAMPLES

        # collect Walking-window features/effort per leg
        feats = {leg: [] for leg in C.LEG_ORDER}
        efforts = {leg: [] for leg in C.LEG_ORDER}
        for stem in train_stems:
            res_df, status = resample.cached_resample(recs[stem])
            walk = (status == "Walking")
            T = len(res_df)
            for end in range(W - 1, T, C.HOP_TRAIN):
                start = end - W + 1
                if not walk[start:end + 1].all():
                    continue  # only pure-Walking windows define "normal"
                for leg in C.LEG_ORDER:
                    feats[leg].append(per_leg_feature_window(res_df, leg, start, end))
                    efforts[leg].append(resistive_effort_window(res_df, leg, start, end))

        params = {"feats": list(_FEATS), "a_D": A_D, "a_R": A_R, "legs": {}}
        for leg in C.LEG_ORDER:
            F = np.array(feats[leg])              # [n, 5]
            E = np.array(efforts[leg])            # [n]
            mean = F.mean(axis=0)
            cov = np.cov(F, rowvar=False) + 1e-6 * np.eye(F.shape[1])
            cov_inv = np.linalg.inv(cov)
            D = cls._maha(F, mean, cov_inv)       # distances of normal walking
            q95_D, iqr_D = np.percentile(D, 95), max(np.subtract(*np.percentile(D, [75, 25])), 1e-6)
            q95_R, iqr_R = np.percentile(E, 95), max(np.subtract(*np.percentile(E, [75, 25])), 1e-6)
            params["legs"][leg] = {
                "mean": mean.tolist(), "cov_inv": cov_inv.tolist(),
                "q95_D": float(q95_D), "iqr_D": float(iqr_D),
                "q95_R": float(q95_R), "iqr_R": float(iqr_R),
                "n_walk_windows": int(len(D)),
            }
        return cls(params)

    @staticmethod
    def _maha(F: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
        d = F - mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, cov_inv, d), 0.0))

    # ---- score ----
    def magnitude(self, res_df, leg: str, start: int, end: int) -> float:
        """Physics magnitude m_leg = sqrt(d_leg * r_leg) in [0,1] (ungated)."""
        p = self.params["legs"][leg]
        f = per_leg_feature_window(res_df, leg, start, end)
        D = self._maha(f[None], np.array(p["mean"]), np.array(p["cov_inv"]))[0]
        R = resistive_effort_window(res_df, leg, start, end)
        d = _sigmoid(self.params["a_D"] * (D - p["q95_D"]) / p["iqr_D"])
        r = _sigmoid(self.params["a_R"] * (R - p["q95_R"]) / p["iqr_R"])
        return float(np.sqrt(d * r))

    def magnitudes_all_legs(self, res_df, start: int, end: int) -> np.ndarray:
        return np.array([self.magnitude(res_df, leg, start, end) for leg in C.LEG_ORDER],
                        dtype=np.float32)

    def intensity(self, res_df, start: int, end: int, gate: np.ndarray) -> np.ndarray:
        """I_leg = gate_leg * m_leg, shape [4]."""
        return self.magnitudes_all_legs(res_df, start, end) * np.asarray(gate, dtype=np.float32)

    # ---- persistence ----
    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(C.ARTIFACTS_DIR, "intensity_calib.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "IntensityCalibrator":
        path = path or os.path.join(C.ARTIFACTS_DIR, "intensity_calib.json")
        with open(path) as f:
            return cls(json.load(f))


if __name__ == "__main__":
    from . import io_load, resample
    calib = IntensityCalibrator.fit(C.SPLIT["train"])
    p = calib.save()
    print(f"Fit intensity calibrator on TRAIN walking windows -> {p}")
    for leg in C.LEG_ORDER:
        lp = calib.params["legs"][leg]
        print(f"  {leg}: n_walk={lp['n_walk_windows']:5d}  q95_D={lp['q95_D']:.2f}  q95_R={lp['q95_R']:.2f}")

    # Sanity: magnitude ~0 on a walking file, higher on the affected leg of an entangled file.
    recs = io_load.list_recordings()
    W = C.WINDOW_SAMPLES

    def mean_mag_on(stem, status_filter):
        df = io_load.load_recording(recs[stem])
        res_df, status = resample.resample_recording(df)
        vals = {leg: [] for leg in C.LEG_ORDER}
        for end in range(W - 1, len(res_df), C.HOP_TRAIN):
            start = end - W + 1
            if status_filter == "Walking" and not (status[start:end+1] == "Walking").all():
                continue
            if status_filter == "Entangled" and not (status[start:end+1] == "Entangled").all():
                continue
            for j, leg in enumerate(C.LEG_ORDER):
                vals[leg].append(calib.magnitude(res_df, leg, start, end))
        return {leg: (np.mean(v) if v else float("nan")) for leg, v in vals.items()}

    walk_mag = mean_mag_on("walking4", "Walking")
    print(f"\nwalking4 mean magnitude per leg: { {k: round(v,3) for k,v in walk_mag.items()} }")
    ent_mag = mean_mag_on("back_left_hand2", "Entangled")  # RL affected
    print(f"back_left_hand2 (RL) entangled mean magnitude: { {k: round(v,3) for k,v in ent_mag.items()} }")
    assert max(walk_mag.values()) < 0.2, "walking magnitude should be near zero"
    assert ent_mag["RL"] > walk_mag["RL"] + 0.1, "affected leg should rise on entanglement"
    print("Intensity OK (near-zero on walking, rises on affected leg).")
