"""Causal dilated TCN with a shared encoder and three heads.

Encoder: Conv1d stem -> N residual TCN blocks (dilations 1,2,4,8,16).
All convolutions are CAUSAL (left-padded only), so the embedding at the final
timestep depends only on past samples -> directly usable online.
The window embedding is the encoder output at the LAST timestep.

Heads (from the 64-d embedding):
  (a) binary detection      -> 1 logit
  (b) per-leg multilabel     -> 4 logits  [FR, FL, RR, RL]
  (c) intensity             -> 4 logits  (weakly supervised by physics pseudo-target)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from . import config as C


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation       # left padding only
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))   # pad left
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel, dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.conv1(x)))
        y = self.drop(self.act(self.conv2(y)))
        return x + y                              # residual


class EntanglementTCN(nn.Module):
    def __init__(self, in_channels: int | None = None,
                 hidden: int = C.ENCODER_CHANNELS,
                 dilations: tuple[int, ...] = C.TCN_DILATIONS,
                 kernel: int = C.TCN_KERNEL, dropout: float = C.TCN_DROPOUT,
                 n_legs: int = C.N_LEGS):
        super().__init__()
        in_channels = in_channels or C.n_channels()
        self.stem = CausalConv1d(in_channels, hidden, kernel, dilation=1)
        self.stem_act = nn.GELU()
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden, kernel, d, dropout) for d in dilations])
        self.head_bin = nn.Linear(hidden, 1)
        self.head_legs = nn.Linear(hidden, n_legs)
        self.head_intensity = nn.Linear(hidden, n_legs)

    @property
    def receptive_field(self) -> int:
        rf = 1
        rf += (C.TCN_KERNEL - 1) * 1               # stem
        for d in C.TCN_DILATIONS:
            rf += 2 * (C.TCN_KERNEL - 1) * d        # two convs per block
        return rf

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> embedding [B, hidden] (last timestep)."""
        h = self.stem_act(self.stem(x))
        for blk in self.blocks:
            h = blk(h)
        return h[:, :, -1]                          # last timestep

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {
            "bin_logit": self.head_bin(h).squeeze(-1),    # [B]
            "legs_logit": self.head_legs(h),              # [B, 4]
            "intensity_logit": self.head_intensity(h),    # [B, 4]
            "embedding": h,
        }


if __name__ == "__main__":
    torch.manual_seed(C.SEED)
    model = EntanglementTCN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}  receptive_field={model.receptive_field} samples "
          f"({model.receptive_field / C.TARGET_HZ * 1000:.0f} ms)  window={C.WINDOW_SAMPLES}")
    assert model.receptive_field < C.WINDOW_SAMPLES, "RF must fit inside window"
    x = torch.randn(8, C.n_channels(), C.WINDOW_SAMPLES)
    out = model(x)
    assert out["bin_logit"].shape == (8,)
    assert out["legs_logit"].shape == (8, 4)
    assert out["intensity_logit"].shape == (8, 4)

    # causality check (eval mode -> deterministic, no dropout): the last-timestep
    # embedding must be unaffected by zeroing inputs earlier than the receptive field.
    model.eval()
    x2 = x.clone()
    x2[:, :, : C.WINDOW_SAMPLES - model.receptive_field] = 0.0
    with torch.no_grad():
        h1 = model.encode(x)
        h2 = model.encode(x2)
    print(f"max embedding diff when masking pre-RF inputs: {(h1 - h2).abs().max().item():.2e}")
    assert torch.allclose(h1, h2, atol=1e-4), "last-timestep embedding must depend only on RF window"
    print("Model OK (shapes, RF < window, causal RF verified).")
