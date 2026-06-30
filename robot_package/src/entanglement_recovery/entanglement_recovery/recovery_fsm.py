"""Recovery finite-state machine — PURE Python, no ROS, fully deterministic & unit-testable.

Time is supplied by the caller (monotonic seconds) so tests can drive it exactly. The node
feeds events (on_detection / on_robot_state / on_command_result / request_estop / arm / reset)
and calls update(now) every tick; update returns the abstract Command to EXECUTE this tick
(Command.NONE if none). The node executes it and reports the outcome via on_command_result.

Design: see docs/RECOVERY_DESIGN.md. Recovery for an UPRIGHT entanglement uses
StopMove -> BalanceStand; escalates to RecoveryStand only on a confirmed FALLEN posture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .states import State, Command, Posture, Detection, RobotState, Diagnostics


@dataclass
class RecoveryConfig:
    # gating / timing (seconds) — all tunable; no magic numbers in the logic below
    confirmation_time_s: float = 0.5      # sustained detection before acting (rejects noise)
    command_timeout_s: float = 1.0        # per-command response wait before retry
    retry_limit: int = 2                  # retries per command before FAULT
    stop_settle_s: float = 0.5            # settle after StopMove ack
    recovery_settle_s: float = 2.0        # settle after BalanceStand/RecoveryStand ack
    verification_duration_s: float = 1.5  # detector must read clear this long
    verify_timeout_s: float = 4.0         # if still entangled this long in VERIFYING -> re-attempt
    resume_delay_s: float = 1.0           # hold before handing back
    cooldown_s: float = 5.0               # ignore triggers after a cycle
    robot_state_timeout_s: float = 0.5    # telemetry older than this is "stale"
    max_state_time_s: float = 10.0        # per-active-state watchdog -> FAULT
    auto_reset_s: float = 0.0             # 0 = manual reset only; >0 auto-leaves FAULT
    # behavior
    confidence_min: float = 0.0           # ignore detections below this confidence
    escalate_to_recovery_stand: bool = True
    use_damp_on_fault: bool = True
    tip_roll_deg: float = 50.0            # |roll| beyond -> FALLEN
    tip_pitch_deg: float = 50.0
    min_soc_pct: float = 10.0             # gate RecoveryStand on battery


_ACTIVE = {State.CONFIRMING, State.STOPPING, State.RECOVERING, State.VERIFYING, State.RESUMING}


class RecoveryFSM:
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.cfg = config or RecoveryConfig()
        self.state = State.IDLE
        self._since = 0.0                 # time current state was entered
        self._det = Detection(entangled=False)
        self._rs = RobotState()
        self._estop_req = False
        self._reset_req = False
        self._armed = False
        # command sub-protocol
        self._inflight: Optional[Command] = None
        self._sent_at = 0.0
        self._attempts = 0
        self._result: Optional[bool] = None
        self._ack_at: Optional[float] = None   # when current command succeeded
        # verification / re-attempt bookkeeping
        self._clear_since: Optional[float] = None
        self._recovery_attempts = 0
        self._fault_detail = ""
        self.last_command = Command.NONE

    # ---------------- event inputs ----------------
    def arm(self, now: float) -> None:
        self._armed = True
        if self.state == State.IDLE:
            self._goto(State.MONITORING, now)

    def on_detection(self, det: Detection) -> None:
        self._det = det

    def on_robot_state(self, rs: RobotState) -> None:
        self._rs = rs

    def on_command_result(self, command: Command, success: bool) -> None:
        if command == self._inflight:
            self._result = success

    def request_estop(self) -> None:
        self._estop_req = True

    def reset(self, now: float) -> None:
        self._reset_req = True

    # ---------------- helpers ----------------
    def _goto(self, state: State, now: float) -> None:
        self.state = state
        self._since = now
        self._reset_cmd()
        if state == State.VERIFYING:
            self._clear_since = None

    def _reset_cmd(self) -> None:
        self._inflight = None
        self._attempts = 0
        self._result = None
        self._ack_at = None

    def _entangled_now(self) -> bool:
        return self._det.entangled and self._det.confidence >= self.cfg.confidence_min

    def _telemetry_fresh(self, now: float) -> bool:
        return self._rs.fresh and (now - self._rs.stamp) <= self.cfg.robot_state_timeout_s

    def _posture(self, now: float) -> Posture:
        return self._rs.posture if self._telemetry_fresh(now) else Posture.UNKNOWN

    def _is_fallen(self, now: float) -> bool:
        if not self._telemetry_fresh(now):
            return False
        if self._rs.posture == Posture.FALLEN:
            return True
        return (abs(self._rs.roll) > self.cfg.tip_roll_deg or
                abs(self._rs.pitch) > self.cfg.tip_pitch_deg)

    def _run_command(self, cmd: Command, now: float) -> Tuple[Command, bool, bool]:
        """Drive one command with retry/timeout. Returns (emit, acked, failed_out)."""
        if self._result is not None:                 # a result arrived
            res, self._result = self._result, None
            if res:
                self._ack_at = now
                return (Command.NONE, True, False)
            self._attempts += 1
            if self._attempts > self.cfg.retry_limit:
                return (Command.NONE, False, True)
            self._sent_at = now
            return (cmd, False, False)               # retry
        if self._inflight is None:                   # first send
            self._inflight = cmd
            self._sent_at = now
            self._attempts = 0
            return (cmd, False, False)
        if now - self._sent_at > self.cfg.command_timeout_s:  # no response = failure
            self._attempts += 1
            if self._attempts > self.cfg.retry_limit:
                return (Command.NONE, False, True)
            self._sent_at = now
            return (cmd, False, False)
        return (Command.NONE, False, False)          # still awaiting

    def _watchdog_tripped(self, now: float) -> bool:
        return (self.state in _ACTIVE and
                (now - self._since) > self.cfg.max_state_time_s)

    # ---------------- main tick ----------------
    def update(self, now: float) -> Command:
        # global overrides first
        if self._reset_req:
            self._reset_req = False
            self._estop_req = False
            self._recovery_attempts = 0
            self._goto(State.MONITORING if self._armed else State.IDLE, now)
            return Command.NONE
        if self._estop_req and self.state != State.ESTOP:
            self._goto(State.ESTOP, now)
            return Command.DAMP
        if self._watchdog_tripped(now):
            self._fault_detail = "watchdog timeout in {}".format(self.state.value)
            self._goto(State.FAULT, now)
            return Command.DAMP if self.cfg.use_damp_on_fault else Command.NONE

        s = self.state
        if s == State.IDLE:
            return Command.NONE
        if s == State.MONITORING:
            return self._monitoring(now)
        if s == State.CONFIRMING:
            return self._confirming(now)
        if s == State.STOPPING:
            return self._stopping(now)
        if s == State.RECOVERING:
            return self._recovering(now)
        if s == State.VERIFYING:
            return self._verifying(now)
        if s == State.RESUMING:
            return self._resuming(now)
        if s == State.COOLDOWN:
            return self._cooldown(now)
        if s == State.FAULT:
            return self._fault(now)
        if s == State.ESTOP:
            return Command.NONE
        return Command.NONE

    # ---------------- per-state ----------------
    def _monitoring(self, now: float) -> Command:
        if self._entangled_now():
            self._goto(State.CONFIRMING, now)
        return Command.NONE

    def _confirming(self, now: float) -> Command:
        if not self._entangled_now():
            self._goto(State.MONITORING, now)      # false alarm
            return Command.NONE
        if (now - self._since) >= self.cfg.confirmation_time_s:
            self._recovery_attempts = 0
            # robot already stopped (fresh telemetry, not locomotion) -> skip StopMove
            if self._posture(now) in (Posture.UPRIGHT, Posture.FALLEN, Posture.OTHER):
                self._goto(State.RECOVERING, now)
            else:
                self._goto(State.STOPPING, now)
        return Command.NONE

    def _stopping(self, now: float) -> Command:
        emit, acked, failed = self._run_command(Command.STOP_MOVE, now)
        if emit != Command.NONE:
            self.last_command = emit
        if failed:
            self._fault_detail = "StopMove failed after retries"
            self._goto(State.FAULT, now)
            return Command.DAMP if self.cfg.use_damp_on_fault else Command.NONE
        # complete when acked AND (not locomotion | settle elapsed)
        if self._ack_at is not None:
            settled = (now - self._ack_at) >= self.cfg.stop_settle_s
            stopped = self._posture(now) != Posture.LOCOMOTION and self._telemetry_fresh(now)
            if stopped or settled:
                self._goto(State.RECOVERING, now)
        return emit

    def _recovering(self, now: float) -> Command:
        fallen = self._is_fallen(now)
        want = Command.BALANCE_STAND
        if fallen and self.cfg.escalate_to_recovery_stand and self._rs.soc >= self.cfg.min_soc_pct:
            want = Command.RECOVERY_STAND
        # if posture flipped to fallen after we already sent BALANCE_STAND, re-issue as recovery
        if (self._inflight == Command.BALANCE_STAND and want == Command.RECOVERY_STAND):
            self._reset_cmd()
        emit, acked, failed = self._run_command(want, now)
        if emit != Command.NONE:
            self.last_command = emit
        if failed:
            self._fault_detail = "{} failed after retries".format(want.value)
            self._goto(State.FAULT, now)
            return Command.DAMP if self.cfg.use_damp_on_fault else Command.NONE
        if self._ack_at is not None:
            settled = (now - self._ack_at) >= self.cfg.recovery_settle_s
            upright = self._posture(now) == Posture.UPRIGHT and self._telemetry_fresh(now)
            if upright or settled:
                self._goto(State.VERIFYING, now)
        return emit

    def _verifying(self, now: float) -> Command:
        if self._entangled_now():
            self._clear_since = None
            if (now - self._since) >= self.cfg.verify_timeout_s:
                self._recovery_attempts += 1
                if self._recovery_attempts > self.cfg.retry_limit:
                    self._fault_detail = "still entangled after recovery attempts"
                    self._goto(State.FAULT, now)
                    return Command.DAMP if self.cfg.use_damp_on_fault else Command.NONE
                self._goto(State.RECOVERING, now)  # re-attempt
            return Command.NONE
        # detector reads clear
        if self._clear_since is None:
            self._clear_since = now
        if (now - self._clear_since) >= self.cfg.verification_duration_s:
            self._goto(State.RESUMING, now)
        return Command.NONE

    def _resuming(self, now: float) -> Command:
        if (now - self._since) >= self.cfg.resume_delay_s:
            self._goto(State.COOLDOWN, now)
        return Command.NONE

    def _cooldown(self, now: float) -> Command:
        if (now - self._since) >= self.cfg.cooldown_s:
            self._goto(State.MONITORING, now)
        return Command.NONE

    def _fault(self, now: float) -> Command:
        if self.cfg.auto_reset_s > 0 and (now - self._since) >= self.cfg.auto_reset_s:
            self._recovery_attempts = 0
            self._goto(State.MONITORING if self._armed else State.IDLE, now)
        return Command.NONE

    # ---------------- diagnostics ----------------
    def diagnostics(self, actuation_enabled: bool) -> Diagnostics:
        return Diagnostics(
            state=self.state.value,
            last_command=self.last_command.value,
            detail=self._fault_detail if self.state == State.FAULT else self._det.alarm_leg,
            retries=self._recovery_attempts,
            actuation_enabled=actuation_enabled,
        )
