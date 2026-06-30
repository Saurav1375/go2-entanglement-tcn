"""Unit tests for the PlanRunner motion executor (pure; injected send/pop_error)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from entanglement_recovery.plan_runner import PlanRunner
from entanglement_recovery.states import MotionPlan, MotionStep, Command


class FakeSport:
    def __init__(self, fail_after=None):
        self.sent = []
        self._fail_after = fail_after  # raise error flag after N sends

    def send_api(self, name, params=None):
        self.sent.append((name, params))

    def pop_error(self):
        if self._fail_after is not None and len(self.sent) >= self._fail_after:
            return True
        return False


def _drive(runner, n=200, dt=0.1):
    now = 0.0
    for _ in range(n):
        now = round(now + dt, 6)
        st = runner.tick(now)
        if st in ("done", "failed"):
            return st, now
    return "running", now


def test_oneshot_plan_done():
    sp = FakeSport()
    plan = MotionPlan("bs", (MotionStep("BALANCE_STAND"),))
    r = PlanRunner(plan, Command.BALANCE_STAND, sp.send_api, sp.pop_error, 0.0)
    st, _ = _drive(r)
    assert st == "done"
    assert sp.sent == [("BALANCE_STAND", None)]


def test_stream_plan_streams_for_duration_then_stops():
    sp = FakeSport()
    # SmallReverse-like: stream Move for 0.6 s, zero, StopMove, BalanceStand
    plan = MotionPlan("rev", (
        MotionStep("MOVE", {"x": -0.15, "y": 0.0, "z": 0.0}, "STREAM", 0.6),
        MotionStep("MOVE", {"x": 0.0, "y": 0.0, "z": 0.0}, "ONESHOT"),
        MotionStep("STOP_MOVE"),
        MotionStep("BALANCE_STAND"),
    ))
    r = PlanRunner(plan, Command.SMALL_REVERSE, sp.send_api, sp.pop_error, 0.0)
    st, _ = _drive(r, dt=0.1)
    assert st == "done"
    move_sends = [s for s in sp.sent if s[0] == "MOVE" and s[1]["x"] == -0.15]
    assert len(move_sends) >= 5            # streamed repeatedly (~0.6s / 0.1s)
    names = [s[0] for s in sp.sent]
    assert names[-2:] == ["STOP_MOVE", "BALANCE_STAND"]   # terminated + re-balanced


def test_plan_failure_on_api_error():
    sp = FakeSport(fail_after=1)           # error after the first send
    plan = MotionPlan("bs", (MotionStep("BALANCE_STAND"), MotionStep("STOP_MOVE")))
    r = PlanRunner(plan, Command.BALANCE_STAND, sp.send_api, sp.pop_error, 0.0)
    st, _ = _drive(r)
    assert st == "failed"


def _all_tests():
    return [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for t in _all_tests():
        t(); print("PASS", t.__name__)
    print("\n{}/{} PlanRunner tests passed".format(len(_all_tests()), len(_all_tests())))
