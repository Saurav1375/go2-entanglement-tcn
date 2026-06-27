"""Pluggable inference backend: ONNX Runtime (default, lightweight) or TorchScript.

Both take a [1, 60, 200] float32 input and return three numpy arrays:
  bin_logit [1], legs_logit [1, 4], intensity_logit [1, 4].

ONNX Runtime is recommended on the GO2 (CPU-only, no full PyTorch install).
TorchScript is a fallback if onnxruntime is unavailable. Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class OnnxBackend(object):
    def __init__(self, path, num_threads=1):
        # type: (str, int) -> None
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(num_threads)
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def forward(self, x):
        # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        outs = self.sess.run(self.output_names, {self.input_name: x.astype(np.float32)})
        return outs[0].reshape(-1), outs[1].reshape(1, -1), outs[2].reshape(1, -1)


class TorchScriptBackend(object):
    def __init__(self, path, num_threads=1):
        # type: (str, int) -> None
        import torch
        torch.set_num_threads(int(num_threads))
        self.torch = torch
        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()

    def forward(self, x):
        # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        torch = self.torch
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32))
            b, legs, inten = self.model(t)
        return (b.cpu().numpy().reshape(-1),
                legs.cpu().numpy().reshape(1, -1),
                inten.cpu().numpy().reshape(1, -1))


def load_backend(path, num_threads=1):
    # type: (str, int) -> object
    if path.endswith(".onnx"):
        return OnnxBackend(path, num_threads)
    if path.endswith(".pt") or path.endswith(".pth") or path.endswith(".ts"):
        return TorchScriptBackend(path, num_threads)
    raise ValueError("Unknown model format: {} (use .onnx or .pt/.ts)".format(path))
