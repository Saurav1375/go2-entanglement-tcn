"""Executes a strategy MotionPlan step-by-step over node ticks (ROS-agnostic logic).

Handles the three MotionStep kinds:
  ONESHOT : publish once; advance next tick (any non-zero Sport response -> failed).
  STREAM  : re-publish every tick for `duration_s` (required for Move), then advance.
  HOLD    : publish nothing; just wait `duration_s` (e.g. let an Euler pose settle).
On completion it reports success/failure so the node can call fsm.on_command_result.

It calls a `send(name, params)` callable (SportClient.send_api) and a `pop_error()` callable;
both are injected so this class needs no ROS import (kept simple/testable). Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Callable, Optional


class PlanRunner:
    def __init__(self, plan, command, send, pop_error, now):
        # type: (object, object, Callable, Callable, float) -> None
        self.plan = plan
        self.command = command          # the FSM Command this plan fulfills (for the ack)
        self._send = send
        self._pop_error = pop_error
        self._idx = 0
        self._step_started = None        # type: Optional[float]
        self._failed = False

    def tick(self, now):
        # type: (float) -> str
        """Advance the plan. Returns 'running' | 'done' | 'failed'."""
        if self._failed:
            return "failed"
        steps = self.plan.steps
        if self._idx >= len(steps):
            return "done"
        step = steps[self._idx]
        if self._step_started is None:                 # first visit to this step
            self._step_started = now
            if step.kind in ("ONESHOT", "STREAM"):
                self._send(step.api, step.params or None)
        elif step.kind == "STREAM":                    # keep streaming (e.g. Move)
            self._send(step.api, step.params or None)

        if self._pop_error():                          # any non-zero Sport response -> abort
            self._failed = True
            return "failed"

        elapsed = now - self._step_started
        if step.kind == "ONESHOT":
            self._advance()
        elif elapsed >= step.duration_s:               # STREAM / HOLD finished
            self._advance()
        return "done" if self._idx >= len(steps) else "running"

    def _advance(self):
        self._idx += 1
        self._step_started = None
