"""Multi-task loss: binary detection + per-leg multilabel + intensity regression.

  L = W_BIN * BCE(bin) + W_LEGS * BCE_multilabel(legs)
        + W_INTENSITY * MaskedHuber(intensity, physics_pseudo_target)

The intensity term is a weak auxiliary: the head learns to track the physics
pseudo-target (computed in intensity.py) so its output is temporally smooth.
The physics formula remains authoritative at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class MultiTaskLoss(nn.Module):
    def __init__(self, bin_pos_weight: float, leg_pos_weight: torch.Tensor):
        super().__init__()
        self.bce_bin = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bin_pos_weight))
        self.bce_legs = nn.BCEWithLogitsLoss(pos_weight=leg_pos_weight)
        self.huber = nn.HuberLoss(reduction="none", delta=0.1)

    def forward(self, out: dict[str, torch.Tensor], y_bin: torch.Tensor,
                y_legs: torch.Tensor, intensity_target: torch.Tensor | None):
        l_bin = self.bce_bin(out["bin_logit"], y_bin)
        l_legs = self.bce_legs(out["legs_logit"], y_legs)

        if intensity_target is not None:
            pred = torch.sigmoid(out["intensity_logit"])
            per = self.huber(pred, intensity_target)            # [B, 4]
            # weight positives (affected legs) more so "0 on normal" doesn't dominate
            w = torch.where(y_legs > 0.5, 3.0, 1.0)
            l_int = (per * w).mean()
        else:
            l_int = torch.zeros((), device=y_bin.device)

        total = C.W_BIN * l_bin + C.W_LEGS * l_legs + C.W_INTENSITY * l_int
        return total, {"bin": l_bin.detach().item(), "legs": l_legs.detach().item(),
                       "intensity": l_int.detach().item()}


if __name__ == "__main__":
    # Overfit a tiny synthetic batch to confirm the wiring trains.
    from .model import EntanglementTCN
    torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EntanglementTCN().to(device)

    B = 16
    x = torch.randn(B, C.n_channels(), C.WINDOW_SAMPLES, device=device)
    y_bin = (torch.arange(B, device=device) % 2).float()
    y_legs = torch.zeros(B, 4, device=device)
    y_legs[y_bin == 1, 0] = 1.0  # positives -> FR
    int_tgt = y_legs.clone()

    loss_fn = MultiTaskLoss(1.0, torch.ones(4, device=device)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for step in range(150):
        opt.zero_grad()
        out = model(x)
        loss, parts = loss_fn(out, y_bin, y_legs, int_tgt)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    model.eval()
    with torch.no_grad():
        out = model(x)
        bin_acc = ((torch.sigmoid(out["bin_logit"]) > 0.5).float() == y_bin).float().mean().item()
    print(f"loss {first:.3f} -> {loss.item():.4f}  parts={ {k: round(v,3) for k,v in parts.items()} }")
    print(f"train-batch binary accuracy after overfit: {bin_acc:.2f}")
    assert loss.item() < first * 0.3, "loss should drop sharply when overfitting"
    assert bin_acc == 1.0, "should perfectly fit a tiny batch"
    print("Losses + model wiring OK (overfits tiny batch).")
