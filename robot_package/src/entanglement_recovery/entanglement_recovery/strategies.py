"""Pluggable recovery strategies (PURE Python, no ROS, unit-testable).

Each Strategy maps to a MotionPlan built from VERIFIED Sport-API actions (api_id + params
confirmed against unitree_ros2/unitree_sdk2 — see docs/INTELLIGENT_RECOVERY.md §"Verified vs
Assumed"). IMPORTANT: every command's *kinematic* effect is verified, but its *disentanglement*
effect (actually freeing a snagged leg) is a DESIGN ASSUMPTION grounded in robotics literature,
not a Go2-verified behavior. Directions/magnitudes are config-driven (no magic numbers) and are
likewise design assumptions to validate on hardware.

Leg sets (body frame, ROS REP-103: +x forward, +y left):
  FRONT={FR,FL} REAR={RR,RL} LEFT={FL,RL} RIGHT={FR,RR}
"""
from __future__ import annotations

from typing import List

from .states import Command, MotionStep, MotionPlan, RecoveryContext

FRONT = {"FR", "FL"}
REAR = {"RR", "RL"}
LEFT = {"FL", "RL"}
RIGHT = {"FR", "RR"}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _intensity_scale(intensity, floor):
    """Scale a motion magnitude DOWN as intensity rises (gentler when more stuck).
    intensity 0 -> 1.0x ; intensity 1 -> `floor`x. floor in (0,1]."""
    i = _clamp(intensity, 0.0, 1.0)
    return 1.0 - (1.0 - floor) * i


class Strategy:
    name = "base"
    command = Command.NONE
    uses_locomotion = False   # True => streams Move; gated by confidence (active motion)
    is_escalation = False     # RecoveryStand / EmergencyStop

    def plan(self, ctx, cfg):
        # type: (RecoveryContext, object) -> MotionPlan
        raise NotImplementedError


class BalanceStandStrategy(Strategy):
    name = "balance_stand"
    command = Command.BALANCE_STAND

    def plan(self, ctx, cfg):
        # gentlest: re-engage actively-balanced stance (may reseat a lightly-pinched foot).
        return MotionPlan(self.name, (MotionStep("BALANCE_STAND"),))


class WeightShiftStrategy(Strategy):
    name = "weight_shift"
    command = Command.WEIGHT_SHIFT

    def plan(self, ctx, cfg):
        s = _intensity_scale(ctx.intensity, cfg.intensity_min_scale)
        roll = cfg.weightshift_roll * s
        pitch = cfg.weightshift_pitch * s
        # tilt AWAY from the snagged corner to unload it (signs are an UNVERIFIED assumption):
        #   left snag  -> roll right (negative) ; right snag -> roll left (positive)
        #   front snag -> pitch back (positive) ; rear snag  -> pitch forward (negative)
        rx = (-roll if ctx.alarm_leg in LEFT else roll) if ctx.alarm_leg else 0.0
        py = (pitch if ctx.alarm_leg in FRONT else -pitch) if ctx.alarm_leg else 0.0
        return MotionPlan(self.name, (
            MotionStep("EULER", {"x": rx, "y": py, "z": 0.0}, "ONESHOT"),
            MotionStep("EULER", {"x": rx, "y": py, "z": 0.0}, "HOLD", cfg.weightshift_hold_s),
            MotionStep("EULER", {"x": 0.0, "y": 0.0, "z": 0.0}, "ONESHOT"),  # return to neutral
            MotionStep("BALANCE_STAND"),
        ))


class SmallReverseStrategy(Strategy):
    name = "small_reverse"
    command = Command.SMALL_REVERSE
    uses_locomotion = True

    def plan(self, ctx, cfg):
        s = _intensity_scale(ctx.intensity, cfg.intensity_min_scale)
        mag = cfg.reverse_speed * s
        # front-leg snag -> back up (vx<0) to draw the foot out; rear-leg snag -> small forward.
        vx = -mag if (not ctx.alarm_leg or ctx.alarm_leg in FRONT) else mag
        return _move_plan(self.name, vx, 0.0, 0.0, cfg.reverse_duration_s)


class SmallSideStepStrategy(Strategy):
    name = "small_sidestep"
    command = Command.SMALL_SIDESTEP
    uses_locomotion = True

    def plan(self, ctx, cfg):
        s = _intensity_scale(ctx.intensity, cfg.intensity_min_scale)
        mag = cfg.sidestep_speed * s
        # step AWAY from the snagged side: right-side snag -> step left (+y); left-side -> right (-y)
        vy = mag if ctx.alarm_leg in RIGHT else (-mag if ctx.alarm_leg in LEFT else mag)
        return _move_plan(self.name, 0.0, vy, 0.0, cfg.sidestep_duration_s)


class RotateStrategy(Strategy):
    name = "rotate"
    command = Command.ROTATE
    uses_locomotion = True

    def plan(self, ctx, cfg):
        s = _intensity_scale(ctx.intensity, cfg.intensity_min_scale)
        mag = cfg.rotate_speed * s
        # yaw direction is a GUESS (wrap chirality unobservable): turn the snagged side backward.
        vyaw = mag if ctx.alarm_leg in RIGHT else -mag
        return _move_plan(self.name, 0.0, 0.0, vyaw, cfg.rotate_duration_s)


class RecoveryStandStrategy(Strategy):
    name = "recovery_stand"
    command = Command.RECOVERY_STAND
    is_escalation = True

    def plan(self, ctx, cfg):
        # escalation: right the body from a fall (large motion). Use only when fallen.
        return MotionPlan(self.name, (MotionStep("RECOVERY_STAND"),))


class EmergencyStopStrategy(Strategy):
    name = "emergency_stop"
    command = Command.DAMP
    is_escalation = True

    def plan(self, ctx, cfg):
        # terminal safe action: stop + damp (limp). Does NOT free a leg; hands off to operator.
        return MotionPlan(self.name, (MotionStep("STOP_MOVE"), MotionStep("DAMP"),))


def _move_plan(name, vx, vy, vyaw, duration_s):
    # type: (str, float, float, float, float) -> MotionPlan
    """Stream Move(vx,vy,vyaw) for duration, then zero-velocity + StopMove, then re-balance."""
    return MotionPlan(name, (
        MotionStep("MOVE", {"x": vx, "y": vy, "z": vyaw}, "STREAM", duration_s),
        MotionStep("MOVE", {"x": 0.0, "y": 0.0, "z": 0.0}, "ONESHOT"),
        MotionStep("STOP_MOVE"),
        MotionStep("BALANCE_STAND"),
    ))


# registry: Command handle -> Strategy instance
ALL_STRATEGIES = [
    BalanceStandStrategy(), WeightShiftStrategy(), SmallReverseStrategy(),
    SmallSideStepStrategy(), RotateStrategy(), RecoveryStandStrategy(), EmergencyStopStrategy(),
]
BY_NAME = {s.name: s for s in ALL_STRATEGIES}
BY_COMMAND = {s.command: s for s in ALL_STRATEGIES}
