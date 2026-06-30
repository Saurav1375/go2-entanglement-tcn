"""Detector-aware Recovery Strategy Manager (PURE Python, unit-testable).

Given a RecoveryContext (alarm_leg, confidence, intensity, fallen) and config, produce the
ORDERED list of strategy names to try in the closed loop. Every decision is config-driven and
explained — no magic numbers. Detector-awareness:

  - leg-aware     : the chosen strategies' DIRECTIONS are computed from alarm_leg (strategies.py);
                    e.g. RR vs FR produce opposite side-step / weight-shift directions.
  - confidence-aware: active locomotion strategies (Move-based reverse/sidestep/rotate) are only
                    attempted when confidence >= active_confidence_min (don't drive on a guess).
  - intensity-aware: low intensity -> gentle full ladder with normal magnitudes; HIGH intensity
                    (>= high_intensity_thresh) -> CONSERVATIVE: skip active locomotion entirely
                    (avoid thrashing a strongly-stuck leg / risking damage) and escalate. Motion
                    magnitudes also scale down with intensity inside each strategy.

Escalation policy (evidence-based): RecoveryStand is the FALLEN-body primitive — it is used ONLY
when posture is fallen, never on an upright snag (forcing it against a hard snag is unsafe). For
an upright snag that won't clear, the terminal action is EmergencyStop (Damp) -> hand to operator.
"""
from __future__ import annotations

from typing import List

from .states import RecoveryContext


class StrategyManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def order(self, ctx):
        # type: (RecoveryContext) -> List[str]
        cfg = self.cfg
        if ctx.fallen:
            seq = []  # type: List[str]
            if cfg.escalate_to_recovery_stand:
                seq.append("recovery_stand")
            seq.append("emergency_stop")
            return seq

        seq = []
        if cfg.enable_balance_stand:
            seq.append("balance_stand")
        if cfg.enable_weight_shift:
            seq.append("weight_shift")

        conservative = ctx.intensity >= cfg.high_intensity_thresh
        active_ok = (ctx.confidence >= cfg.active_confidence_min) and not conservative
        if active_ok:
            if cfg.enable_small_reverse:
                seq.append("small_reverse")
            if cfg.enable_small_sidestep:
                seq.append("small_sidestep")
            if cfg.enable_rotate:
                seq.append("rotate")

        seq.append("emergency_stop")   # terminal safe action for an unresolved upright snag
        return seq
