#!/usr/bin/env python3
"""Validate the robot runtime against the research evaluation (DEV-SIDE).

For a held-out test recording it feeds the research-resampled 500 Hz rows through
the standalone runtime EntanglementEngine (ONNX backend) and compares the
per-window outputs to `ml.evaluate.dense_infer` (the same code path as
evaluate.py). Also benchmarks CPU inference latency.

Usage (from the research repo root):
    python robot_package/tools/validate_runtime.py
    python robot_package/tools/validate_runtime.py --backend ts --stem front_both_wire2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
PKG_DIR = os.path.join(REPO_ROOT, "robot_package", "src", "entanglement_detector")
MODELS = os.path.join(PKG_DIR, "models")
sys.path.insert(0, REPO_ROOT)        # research `ml` package (reference only)
sys.path.insert(0, PKG_DIR)          # runtime `entanglement_detector` package

from entanglement_detector.engine import EntanglementEngine  # noqa: E402
from entanglement_detector import constants as K             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="back_left_hand2")
    ap.add_argument("--backend", choices=["onnx", "ts"], default="onnx")
    args = ap.parse_args()

    model_file = ("entanglement_tcn.onnx" if args.backend == "onnx"
                  else "entanglement_tcn_ts.pt")

    # ---- research reference on CPU (matches evaluate.py) ----
    from ml import io_load, resample, evaluate, config as C
    device = "cpu"
    ref_model, ref_norm, ref_calib = evaluate.load_artifacts(device)
    path = io_load.list_recordings()[args.stem]
    ref = evaluate.dense_infer(args.stem, path, ref_model, ref_norm, ref_calib, device, hop=1)

    # ---- runtime engine: detection latch off so we compare raw per-window math ----
    eng = EntanglementEngine(
        model_path=os.path.join(MODELS, model_file),
        normalize_path=os.path.join(MODELS, "normalize.json"),
        intensity_calib_path=os.path.join(MODELS, "intensity_calib.json"),
        temperature=1.0, detection_threshold=2.0, debounce_ms=0,
        leg_thresholds={}, num_threads=1,
    )

    res_df, _ = resample.cached_resample(path)
    raw_names = K.raw_channel_names()
    cols = {n: res_df[n].to_numpy() for n in raw_names}

    out_by_end = {}
    latencies = []
    for i in range(len(res_df)):
        sample = {n: cols[n][i] for n in raw_names}
        t0 = time.perf_counter()
        r = eng.push(sample)
        if r is not None:
            latencies.append((time.perf_counter() - t0) * 1000.0)
            out_by_end[i] = r

    # ---- compare on dense windows (hop=1 -> end_idx == row index) ----
    dbin, dleg, dint = [], [], []
    for k, end in enumerate(ref["end_idx"]):
        r = out_by_end.get(int(end))
        if r is None:
            continue
        dbin.append(abs(r["p_bin_raw"] - float(ref["p_bin"][k])))
        for j, leg in enumerate(K.LEG_ORDER):
            dleg.append(abs(r["p_legs"][leg] - float(ref["p_legs"][k, j])))
            dint.append(abs(r["intensity"][leg] - float(ref["intensity"][k, j])))
    dbin, dleg, dint = np.array(dbin), np.array(dleg), np.array(dint)

    print("Validation: runtime ({}) vs ml.evaluate.dense_infer on '{}'".format(args.backend, args.stem))
    print("  windows compared : {}".format(len(dbin)))
    print("  max |p_bin diff|     = {:.2e}".format(dbin.max()))
    print("  max |p_leg diff|     = {:.2e}".format(dleg.max()))
    print("  max |intensity diff| = {:.2e}".format(dint.max()))
    tol = 2e-3
    ok = dbin.max() < tol and dleg.max() < tol and dint.max() < tol
    print("  MATCH (<{:.0e}): {}".format(tol, ok))

    lat = np.array(latencies)
    print("\nBenchmark (CPU, 1 thread, {} backend, per-window forward+post):".format(args.backend))
    print("  windows={}  mean={:.2f} ms  median={:.2f} ms  p95={:.2f} ms  max={:.2f} ms".format(
        len(lat), lat.mean(), np.median(lat), np.percentile(lat, 95), lat.max()))
    print("  budget @ {} Hz = {:.2f} ms/sample -> {}".format(
        K.TARGET_HZ, 1000.0 / K.TARGET_HZ,
        "OK (real-time)" if np.percentile(lat, 95) < 1000.0 / K.TARGET_HZ else
        "exceeds per-sample budget; decimate or use ONNX"))
    assert ok, "runtime diverges from evaluate.py beyond tolerance"
    print("\nVALIDATION PASSED.")


if __name__ == "__main__":
    main()
