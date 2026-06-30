"""Unit tests for the pure RecoveryFSM (no ROS, no hardware).

Run with pytest, or standalone:
    PYTHONPATH=robot_package/src/entanglement_recovery python3 -m pytest -q
    PYTHONPATH=robot_package/src/entanglement_recovery python3 robot_package/src/entanglement_recovery/test/test_recovery_fsm.py
Covers the docs/RECOVERY_TESTING.md scenarios.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from entanglement_recovery.recovery_fsm import RecoveryFSM, RecoveryConfig
from entanglement_recovery.states import State, Command, Posture, Detection, RobotState

CMDS = {Command.STOP_MOVE, Command.BALANCE_STAND, Command.RECOVERY_STAND, Command.DAMP,
        Command.WEIGHT_SHIFT, Command.SMALL_REVERSE, Command.SMALL_SIDESTEP, Command.ROTATE}


def cfg(**kw):
    base = dict(confirmation_time_s=0.5, command_timeout_s=1.0, retry_limit=2,
                stop_settle_s=0.5, recovery_settle_s=2.0, verification_duration_s=1.5,
                verify_timeout_s=4.0, resume_delay_s=1.0, cooldown_s=5.0,
                robot_state_timeout_s=0.5, max_state_time_s=10.0)
    base.update(kw)
    return RecoveryConfig(**base)


def rs(posture=Posture.LOCOMOTION, now=0.0, mode=3, roll=0.0, pitch=0.0, soc=100.0):
    return RobotState(posture=posture, mode=mode, roll=roll, pitch=pitch, soc=soc,
                      stamp=now, fresh=True)


def detok(ent, now=0.0, conf=0.9, leg="RR"):
    return Detection(entangled=ent, confidence=conf, alarm_leg=leg, stamp=now)


def drive(fsm, t, det=None, robot=None, ack=True, fail=False, dt=0.1):
    """Advance to time t in dt steps, feeding det/robot each tick and acking commands."""
    now = fsm._since  # start near current
    while now < t - 1e-9:
        now = round(now + dt, 6)
        if det is not None:
            fsm.on_detection(det(now) if callable(det) else det)
        if robot is not None:
            fsm.on_robot_state(robot(now) if callable(robot) else robot)
        cmd = fsm.update(now)
        if cmd in CMDS and cmd != Command.DAMP and ack:
            fsm.on_command_result(cmd, success=not fail)
    return now


# ---------------------------------------------------------------- scenarios
def test_normal_walking_no_action():
    f = RecoveryFSM(cfg()); f.arm(0.0)
    drive(f, 3.0, det=lambda n: detok(False, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    assert f.state == State.MONITORING


def test_false_alarm_returns_to_monitoring():
    f = RecoveryFSM(cfg()); f.arm(0.0)
    # entangled for only 0.3 s (< confirmation 0.5) then clears
    f.on_detection(detok(True, 0.1)); f.update(0.1)
    drive(f, 0.4, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    assert f.state == State.CONFIRMING
    drive(f, 1.0, det=lambda n: detok(False, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    assert f.state == State.MONITORING


def test_happy_path_full_cycle():
    f = RecoveryFSM(cfg()); f.arm(0.0)
    # sustained entanglement -> confirm -> stop
    drive(f, 0.8, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    assert f.state == State.STOPPING
    # robot reports stopped (upright) -> recovering
    drive(f, 2.0, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.UPRIGHT, n))
    assert f.state in (State.RECOVERING, State.VERIFYING)
    # detection clears, robot upright -> verify -> resume -> cooldown -> monitoring
    drive(f, 12.0, det=lambda n: detok(False, n), robot=lambda n: rs(Posture.UPRIGHT, n))
    assert f.state == State.MONITORING


def test_robot_already_stopped_skips_stopmove():
    f = RecoveryFSM(cfg()); f.arm(0.0)
    emitted = []
    # already upright/stopped at confirmation
    now = 0.0
    for _ in range(8):
        now = round(now + 0.1, 6)
        f.on_detection(detok(True, now)); f.on_robot_state(rs(Posture.UPRIGHT, now))
        c = f.update(now)
        if c in CMDS:
            emitted.append(c)
            if c != Command.DAMP:
                f.on_command_result(c, True)
    assert Command.STOP_MOVE not in emitted   # skipped because already stopped
    assert f.state in (State.RECOVERING, State.VERIFYING)


def test_fallen_uses_recovery_stand():
    f = RecoveryFSM(cfg()); f.arm(0.0)
    seen = []
    now = 0.0
    for _ in range(40):
        now = round(now + 0.1, 6)
        f.on_detection(detok(True, now))
        f.on_robot_state(rs(Posture.FALLEN, now, mode=5))   # fallen
        c = f.update(now)
        if c in CMDS:
            seen.append(c)
            if c != Command.DAMP:
                f.on_command_result(c, True)
    assert Command.RECOVERY_STAND in seen
    assert Command.BALANCE_STAND not in seen


def test_api_failure_goes_to_fault_and_damps():
    f = RecoveryFSM(cfg(retry_limit=1)); f.arm(0.0)
    # confirm -> stopping, then never ack StopMove (commands fail) -> retries -> FAULT + DAMP
    drive(f, 0.8, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    assert f.state == State.STOPPING
    damped = []
    now = f._since
    for _ in range(60):
        now = round(now + 0.5, 6)   # advance past command_timeout repeatedly, NEVER ack
        f.on_detection(detok(True, now)); f.on_robot_state(rs(Posture.LOCOMOTION, now))
        c = f.update(now)
        if c == Command.DAMP:
            damped.append(now)
    assert f.state == State.FAULT
    assert damped, "FAULT should emit DAMP"


def test_estop_from_any_state():
    f = RecoveryFSM(cfg()); f.arm(0.0)
    drive(f, 0.8, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    f.request_estop()
    c = f.update(f._since + 0.1)
    assert f.state == State.ESTOP and c == Command.DAMP


def test_recovery_not_reentered_while_active():
    f = RecoveryFSM(cfg()); f.arm(0.0)
    drive(f, 0.8, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    assert f.state == State.STOPPING
    # flood with new detections; must NOT restart/leave the active cycle back to CONFIRMING
    states = []
    now = f._since
    for _ in range(10):
        now = round(now + 0.1, 6)
        f.on_detection(detok(True, now)); f.on_robot_state(rs(Posture.UPRIGHT, now))
        f.update(now); states.append(f.state)
        # ack to progress
        if f._inflight in CMDS:
            f.on_command_result(f._inflight, True)
    assert State.CONFIRMING not in states  # never re-entered confirmation while active


def test_verify_failure_reattempts_then_faults():
    f = RecoveryFSM(cfg(retry_limit=1, verify_timeout_s=1.0)); f.arm(0.0)
    # get to VERIFYING with entanglement that never clears
    drive(f, 0.8, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    drive(f, 6.0, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.UPRIGHT, n))
    # still entangled -> re-attempts bounded -> FAULT
    drive(f, 20.0, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.UPRIGHT, n))
    assert f.state == State.FAULT


def test_reset_recovers_from_fault():
    f = RecoveryFSM(cfg(retry_limit=0)); f.arm(0.0)
    drive(f, 0.8, det=lambda n: detok(True, n), robot=lambda n: rs(Posture.LOCOMOTION, n))
    now = f._since
    for _ in range(20):
        now = round(now + 0.5, 6)
        f.on_robot_state(rs(Posture.LOCOMOTION, now)); f.update(now)  # no ack -> fault
    assert f.state == State.FAULT
    f.reset(now); f.update(now + 0.1)
    assert f.state == State.MONITORING


# ---------------------------------------------------------------- strategy-system scenarios
def _run(f, t, det, robot, ack=True, dt=0.1):
    """Drive to time t, feeding det(now)/robot(now), acking every emitted command; record
    the ordered list of distinct commands emitted."""
    seen = []
    now = f._since
    while now < t - 1e-9:
        now = round(now + dt, 6)
        f.on_detection(det(now)); f.on_robot_state(robot(now))
        c = f.update(now)
        if c in CMDS:
            if not seen or seen[-1] != c:
                seen.append(c)
            if ack:
                f.on_command_result(c, True)
    return seen


def test_strategy_first_success():
    # detector clears immediately after the first (gentlest) strategy -> resume, only BalanceStand
    f = RecoveryFSM(cfg()); f.arm(0.0)
    _run(f, 0.8, lambda n: detok(True, n), lambda n: rs(Posture.LOCOMOTION, n))
    seen = _run(f, 12.0, lambda n: detok(False, n), lambda n: rs(Posture.UPRIGHT, n))
    assert f.state == State.MONITORING
    assert Command.SMALL_REVERSE not in seen and Command.WEIGHT_SHIFT not in seen


def test_strategy_sequencing_then_escalate_to_fault():
    # detector NEVER clears -> strategies tried in ORDER, ending emergency-stop(Damp) -> FAULT
    f = RecoveryFSM(cfg(verify_timeout_s=0.4, recovery_settle_s=0.2)); f.arm(0.0)
    seen = _run(f, 40.0, lambda n: detok(True, n), lambda n: rs(Posture.UPRIGHT, n))
    # order: BalanceStand, WeightShift, SmallReverse, SmallSideStep, Rotate, (Damp)
    order = [Command.BALANCE_STAND, Command.WEIGHT_SHIFT, Command.SMALL_REVERSE,
             Command.SMALL_SIDESTEP, Command.ROTATE, Command.DAMP]
    filtered = [c for c in seen if c in order]
    assert filtered == order, filtered          # exact sequence, no re-ordering / repeats
    assert f.state == State.FAULT


def test_strategy_clears_after_second_strategy():
    # entangled through BalanceStand+WeightShift; clears once WeightShift reached -> resume
    f = RecoveryFSM(cfg(verify_timeout_s=0.4, recovery_settle_s=0.2, verification_duration_s=0.3,
                        resume_delay_s=0.2, cooldown_s=0.5)); f.arm(0.0)
    # phase 1: drive (entangled) only until the 2nd strategy (WeightShift) has been emitted
    seen1 = []
    now = 0.0
    while now < 5.0 and Command.WEIGHT_SHIFT not in seen1:
        now = round(now + 0.1, 6)
        f.on_detection(detok(True, now)); f.on_robot_state(rs(Posture.UPRIGHT, now))
        c = f.update(now)
        if c in CMDS:
            seen1.append(c); f.on_command_result(c, True)
    assert Command.WEIGHT_SHIFT in seen1 and Command.SMALL_REVERSE not in seen1
    # phase 2: now it clears -> resume WITHOUT needing the Move strategies
    seen2 = _run(f, now + 8.0, lambda n: detok(False, n), lambda n: rs(Posture.UPRIGHT, n))
    assert f.state == State.MONITORING
    assert Command.SMALL_REVERSE not in (seen1 + seen2)


def test_low_confidence_skips_active_motions():
    # low confidence -> only gentle strategies (no Move), then escalate
    f = RecoveryFSM(cfg(verify_timeout_s=0.4, recovery_settle_s=0.2, confidence_min=0.0,
                        active_confidence_min=0.6)); f.arm(0.0)
    seen = _run(f, 30.0, lambda n: detok(True, n, conf=0.4),
                lambda n: rs(Posture.UPRIGHT, n))
    assert Command.SMALL_REVERSE not in seen and Command.ROTATE not in seen
    assert Command.BALANCE_STAND in seen and f.state == State.FAULT


def test_high_intensity_conservative():
    # high intensity -> conservative: gentle only, no active Move, escalate
    f = RecoveryFSM(cfg(verify_timeout_s=0.4, recovery_settle_s=0.2, high_intensity_thresh=0.8))
    f.arm(0.0)
    det = lambda n: Detection(entangled=True, confidence=0.95, alarm_leg="RR", intensity=0.9, stamp=n)
    seen = _run(f, 30.0, det, lambda n: rs(Posture.UPRIGHT, n))
    assert Command.SMALL_REVERSE not in seen and Command.SMALL_SIDESTEP not in seen
    assert f.state == State.FAULT


def test_strategy_command_timeout_faults():
    # a strategy command never acked (API/comm failure) -> retries -> FAULT
    f = RecoveryFSM(cfg(retry_limit=1, recovery_settle_s=0.2)); f.arm(0.0)
    _run(f, 0.8, lambda n: detok(True, n), lambda n: rs(Posture.LOCOMOTION, n))  # -> STOPPING
    now = f._since
    for _ in range(40):
        now = round(now + 0.5, 6)   # advance past command_timeout, NEVER ack
        f.on_detection(detok(True, now)); f.on_robot_state(rs(Posture.UPRIGHT, now))
        f.update(now)
    assert f.state == State.FAULT


def _all_tests():
    return [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    passed = 0
    for t in _all_tests():
        t(); print("PASS", t.__name__); passed += 1
    print("\n{}/{} FSM tests passed".format(passed, len(_all_tests())))
