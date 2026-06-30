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

from .states import State, Command, Posture, Detection, RobotState, RecoveryContext, Diagnostics
from .strategy_manager import StrategyManager
from . import strategies as STRAT


@dataclass
class RecoveryConfig:
    # gating / timing (seconds) — all tunable; no magic numbers in the logic below
    confirmation_time_s: float = 0.5      # sustained detection before acting (rejects noise)
    command_timeout_s: float = 1.0        # per-command response wait before retry
    retry_limit: int = 2                  # retries per command before FAULT
    stop_settle_s: float = 0.5            # settle after StopMove ack
    recovery_settle_s: float = 2.0        # settle after a strategy's motion ack
    verification_duration_s: float = 1.5  # detector must read clear this long
    verify_timeout_s: float = 4.0         # still entangled this long -> try the NEXT strategy
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
    # ---- strategy policy (intelligent recovery) ----
    active_confidence_min: float = 0.6    # min confidence to attempt Move-based strategies
    high_intensity_thresh: float = 0.8    # intensity >= this -> conservative (no active Move)
    intensity_min_scale: float = 0.5      # motion-magnitude floor at intensity=1 (gentler)
    enable_balance_stand: bool = True
    enable_weight_shift: bool = True
    enable_small_reverse: bool = True
    enable_small_sidestep: bool = True
    enable_rotate: bool = True
    # ---- strategy motion magnitudes (TUNABLE design assumptions; below the 0.3 m/s demo) ----
    reverse_speed: float = 0.15           # m/s
    reverse_duration_s: float = 0.6
    sidestep_speed: float = 0.15          # m/s
    sidestep_duration_s: float = 0.6
    rotate_speed: float = 0.4             # rad/s
    rotate_duration_s: float = 0.5
    weightshift_roll: float = 0.2         # rad (Euler), within +/-0.75 limit
    weightshift_pitch: float = 0.2        # rad
    weightshift_hold_s: float = 1.0


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
        # ---- strategy state (intelligent recovery; only the RECOVERING/VERIFYING loop) ----
        self.manager = StrategyManager(self.cfg)
        self._order = []                       # type: list  ordered strategy names
        self._strategy_idx = 0
        self._current_plan_for: Optional[str] = None
        self.current_plan = None               # MotionPlan for the node to execute
        self._active_strategy = ""

    def _context(self, now: float) -> RecoveryContext:
        return RecoveryContext(alarm_leg=self._det.alarm_leg, confidence=self._det.confidence,
                               intensity=self._det.intensity, fallen=self._is_fallen(now))

    def _begin_recovery_cycle(self, now: float) -> None:
        self._order = self.manager.order(self._context(now))
        self._strategy_idx = 0
        self._current_plan_for = None
        self.current_plan = None

    def current_motion_plan(self):
        """The node reads this to execute the active strategy's verified MotionPlan."""
        return self.current_plan

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
            self._begin_recovery_cycle(now)    # compute the detector-aware strategy order
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
        # mid-cycle fall escalation: if posture became fallen, switch to the fallen order.
        if self._is_fallen(now) and self._order and self._order[0] != "recovery_stand":
            self._begin_recovery_cycle(now)
        if not self._order:                       # defensive
            self._order = ["emergency_stop"]; self._strategy_idx = 0
        name = self._order[min(self._strategy_idx, len(self._order) - 1)]
        strat = STRAT.BY_NAME[name]
        self._active_strategy = name
        if self._current_plan_for != name:        # build this strategy's verified MotionPlan once
            self.current_plan = strat.plan(self._context(now), self.cfg)
            self._current_plan_for = name
        emit, acked, failed = self._run_command(strat.command, now)
        if emit != Command.NONE:
            self.last_command = emit
        if failed:
            self._fault_detail = "strategy '{}' command failed after retries".format(name)
            self._goto(State.FAULT, now)
            return Command.DAMP if self.cfg.use_damp_on_fault else Command.NONE
        if self._ack_at is not None:              # plan executed by the node
            settled = (now - self._ack_at) >= self.cfg.recovery_settle_s
            done = settled or (self._posture(now) == Posture.UPRIGHT and self._telemetry_fresh(now))
            if done:
                if strat.command == Command.DAMP:  # terminal safe action -> fault / operator handoff
                    self._fault_detail = "all strategies exhausted; emergency stop (damp) issued"
                    self._goto(State.FAULT, now)
                    return Command.NONE
                self._goto(State.VERIFYING, now)   # closed loop: let the detector judge success
        return emit

    def _verifying(self, now: float) -> Command:
        if self._entangled_now():
            self._clear_since = None
            if (now - self._since) >= self.cfg.verify_timeout_s:
                self._strategy_idx += 1            # not freed -> try the NEXT strategy
                self._recovery_attempts = self._strategy_idx
                if self._strategy_idx >= len(self._order):
                    self._fault_detail = "all {} strategies exhausted; still entangled".format(
                        len(self._order))
                    self._goto(State.FAULT, now)
                    return Command.DAMP if self.cfg.use_damp_on_fault else Command.NONE
                self._goto(State.RECOVERING, now)
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
        if self.state == State.FAULT:
            detail = self._fault_detail
        elif self.state in (State.RECOVERING, State.VERIFYING):
            detail = "strategy {}/{}: {} (leg={})".format(
                self._strategy_idx + 1, len(self._order), self._active_strategy, self._det.alarm_leg)
        else:
            detail = self._det.alarm_leg
        return Diagnostics(
            state=self.state.value,
            last_command=self.last_command.value,
            detail=detail,
            retries=self._strategy_idx,
            actuation_enabled=actuation_enabled,
        )
