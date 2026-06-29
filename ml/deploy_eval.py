"""Phase-grouped evaluation on the 5 new GO2 recordings — quantifies the 4
deployment issues, comparable BEFORE vs AFTER retraining.

For each new recording it groups dense (hop=1) windows by their dominant true
Status (Walking/Stop/Lock/Entangled) and reports, per phase:
  - n windows
  - FIRE% : fraction with raw p_bin >= detection threshold (0.9999) -> false alarms
            on Lock/Stop/Walking phases; true detections on Entangled phases.
  - RRfire% / per-affected-leg detection at the per-leg thresholds.

Maps to the reported issues:
  1. standup->Lock RR false alarm    -> lock_stop / walk_stop_back  Lock FIRE%
  2. RR false alarm after entanglement when stopped -> *_stop / *_intensity Stop/Lock FIRE%
  3. false alarm during backward walking + stops    -> walk_stop_back Walking/Stop FIRE%
  4. front-right detection            -> front_right_wire Entangled detection of FR

Run (uses whatever model is currently in ml/artifacts/):
    python -m ml.deploy_eval
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import io_load, resample, evaluate
from .windowing import make_windows
from .features import build_channel_matrix

NEW_FILES = [
    ("go2_lowstate_lock_stop", "none"),
    ("go2_lowstate_walk_stop_back", "none"),
    ("go2_lowstate_entaglement_front_right_wire", "FR"),
    ("go2_lowstate_entaglement_back_right_stop", "RR"),
    ("go2_lowstate_entaglement_back_right_intensity", "RR"),
]
RAW_THR = 0.9999
LEG_THR = {"FR": 0.5, "FL": 0.5, "RR": 0.9, "RL": 0.5}


def window_statuses(stem, path):
    """Dominant Status string per dense window (aligned to make_windows end_idx)."""
    res_df, status = resample.cached_resample(path)
    X = build_channel_matrix(res_df)
    ws = make_windows(X, status, stem, C.parse_legs(stem), hop=1)
    doms = []
    for end in ws.end_idx:
        sl = slice(int(end) - C.WINDOW_SAMPLES + 1, int(end) + 1)
        vals = status[sl]
        # dominant non-handling: pick the most common label in the window
        uniq, cnt = np.unique(vals.astype(str), return_counts=True)
        doms.append(uniq[int(np.argmax(cnt))])
    return ws.end_idx, np.array(doms, dtype=object)


def main():
    device = evaluate._device()
    model, norm, calib = evaluate.load_artifacts(device)
    recs = io_load.list_recordings()
    print("Phase-grouped detection on new recordings (model = current ml/artifacts/model.pt)")
    print("FIRE% = windows with raw p_bin >= {:.4f}\n".format(RAW_THR))

    for stem, aff in NEW_FILES:
        res = evaluate.dense_infer(stem, recs[stem], model, norm, calib, device, hop=1)
        if res["n"] == 0:
            print("{}: no windows".format(stem)); continue
        ends, doms = window_statuses(stem, recs[stem])
        # align (same hop=1 windowing -> same order/length)
        n = min(len(doms), res["n"])
        p_bin = res["p_bin"][:n]; p_legs = res["p_legs"][:n]; doms = doms[:n]
        print("== {}  (affected: {}) ==".format(stem, aff))
        for phase in ["Walking", "Stop", "Lock", "Entangled"]:
            m = doms == phase
            if not m.any():
                continue
            fire = float((p_bin[m] >= RAW_THR).mean())
            line = "   {:9s} n={:5d}  FIRE%={:5.1f}".format(phase, int(m.sum()), fire * 100)
            # per-leg fire at thresholds
            legbits = []
            for j, leg in enumerate(C.LEG_ORDER):
                lf = float((p_legs[m][:, j] >= LEG_THR[leg]).mean())
                if lf > 0.02:
                    legbits.append("{}={:.0f}%".format(leg, lf * 100))
            if legbits:
                line += "  legfire(" + ",".join(legbits) + ")"
            print(line)
        print()


if __name__ == "__main__":
    main()
