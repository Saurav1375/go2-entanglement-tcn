#!/usr/bin/env python3
"""Export the trained TCN for on-robot deployment (DEV-SIDE script).

Run on the development machine (where the research `ml/` package and a trained
`ml/artifacts/model.pt` exist). Produces, under
`robot_package/src/entanglement_detector/models/`:
  - entanglement_tcn.onnx        (ONNX, recommended runtime: onnxruntime, no torch)
  - entanglement_tcn_ts.pt       (TorchScript, fallback runtime: torch.jit)
  - normalize.json, intensity_calib.json, calibration.json   (copied artifacts)

The exported model is self-contained: the robot loads it WITHOUT the research
training code or the EntanglementTCN class.

Usage (from the research repo root):
    python robot_package/export_model.py
"""
from __future__ import annotations

import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ml.model import EntanglementTCN  # noqa: E402
from ml import config as C  # noqa: E402

ARTIFACTS = os.path.join(REPO_ROOT, "ml", "artifacts")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "src", "entanglement_detector", "models")


class ExportWrapper(nn.Module):
    """Returns a fixed tuple (no dict) so TorchScript/ONNX export is clean."""

    def __init__(self, model):
        super(ExportWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["bin_logit"], out["legs_logit"], out["intensity_logit"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = torch.load(os.path.join(ARTIFACTS, "model.pt"), map_location="cpu",
                      weights_only=False)
    n_ch = ckpt["config"]["n_channels"]
    model = EntanglementTCN(in_channels=n_ch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    wrapper = ExportWrapper(model).eval()

    example = torch.randn(1, n_ch, C.WINDOW_SAMPLES)
    with torch.no_grad():
        ref = wrapper(example)

    # ---- TorchScript (traced; forward is static) ----
    ts_path = os.path.join(OUT_DIR, "entanglement_tcn_ts.pt")
    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, example)
    scripted.save(ts_path)

    # ---- ONNX ----
    onnx_path = os.path.join(OUT_DIR, "entanglement_tcn.onnx")
    with torch.no_grad():
        try:
            torch.onnx.export(
                wrapper, example, onnx_path,
                input_names=["window"],
                output_names=["bin_logit", "legs_logit", "intensity_logit"],
                dynamic_axes={"window": {0: "batch"}},
                opset_version=13, dynamo=False,
            )
        except TypeError:
            # older torch without the `dynamo` kwarg
            torch.onnx.export(
                wrapper, example, onnx_path,
                input_names=["window"],
                output_names=["bin_logit", "legs_logit", "intensity_logit"],
                dynamic_axes={"window": {0: "batch"}},
                opset_version=13,
            )

    # ---- copy calibration artifacts ----
    for name in ("normalize.json", "intensity_calib.json", "calibration.json"):
        shutil.copyfile(os.path.join(ARTIFACTS, name), os.path.join(OUT_DIR, name))

    # ---- verify both backends load WITHOUT the training code and match ----
    # reload in a fresh torch.jit context
    ts = torch.jit.load(ts_path, map_location="cpu").eval()
    with torch.no_grad():
        ts_out = ts(example)
    ts_err = max(float((a - b).abs().max()) for a, b in zip(ts_out, ref))

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onx = sess.run(None, {"window": example.numpy()})
    onnx_err = max(float(abs(o - r.numpy()).max()) for o, r in zip(onx, ref))

    print("Exported to {}".format(OUT_DIR))
    print("  entanglement_tcn.onnx       (channels={}, window={})".format(n_ch, C.WINDOW_SAMPLES))
    print("  entanglement_tcn_ts.pt")
    print("  normalize.json / intensity_calib.json / calibration.json")
    print("Verification vs eager model:  TorchScript max|err|={:.2e}  ONNX max|err|={:.2e}".format(
        ts_err, onnx_err))
    assert ts_err < 1e-4 and onnx_err < 1e-4, "exported model diverges from eager model"
    print("OK: exported models reproduce the eager model and load without training code.")


if __name__ == "__main__":
    main()
