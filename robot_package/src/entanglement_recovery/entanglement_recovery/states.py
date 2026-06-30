"""Pure enums + dataclasses for the recovery FSM (no ROS imports). Python 3.8 compatible."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(Enum):
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    CONFIRMING = "CONFIRMING"          # verify the detection is real (sustained)
    STOPPING = "STOPPING"              # stop the robot
    RECOVERING = "RECOVERING"          # balance-stand (or recovery-stand if fallen)
    VERIFYING = "VERIFYING"            # verify the detection cleared
    RESUMING = "RESUMING"              # hold, hand control back
    COOLDOWN = "COOLDOWN"              # ignore triggers briefly
    FAULT = "FAULT"                    # command failure / unverifiable -> safe hold
    ESTOP = "ESTOP"                    # soft emergency stop (damp)


class Command(Enum):
    """Abstract command emitted by the FSM = the active recovery STRATEGY's handle.

    The node maps each to a verified MotionPlan (see strategies.py / sport_api.py).
    NONE/STOP_MOVE/BALANCE_STAND/RECOVERY_STAND/DAMP are unchanged for backward
    compatibility; WEIGHT_SHIFT/SMALL_REVERSE/SMALL_SIDESTEP/ROTATE are the new
    strategy handles (all implemented with verified Sport APIs: Euler 1007 / Move 1008).
    """
    NONE = "NONE"
    STOP_MOVE = "STOP_MOVE"
    BALANCE_STAND = "BALANCE_STAND"
    RECOVERY_STAND = "RECOVERY_STAND"
    DAMP = "DAMP"
    WEIGHT_SHIFT = "WEIGHT_SHIFT"
    SMALL_REVERSE = "SMALL_REVERSE"
    SMALL_SIDESTEP = "SMALL_SIDESTEP"
    ROTATE = "ROTATE"


class Posture(Enum):
    """Decoded robot posture from SportModeState (telemetry-driven guards)."""
    UNKNOWN = "UNKNOWN"        # no/stale telemetry
    LOCOMOTION = "LOCOMOTION"  # walking (mode 3)
    UPRIGHT = "UPRIGHT"        # idle / balanceStand (mode 0/1)
    FALLEN = "FALLEN"          # lieDown or tipped past threshold
    RECOVERING = "RECOVERING"  # recoveryStand in progress (mode 8)
    OTHER = "OTHER"            # pose/sit/damping/etc.


@dataclass(frozen=True)
class Detection:
    """One detector event (mirrors entanglement_interfaces/EntanglementState)."""
    entangled: bool
    confidence: float = 0.0
    alarm_leg: str = ""
    intensity: float = 0.0   # severity of the alarmed leg (0..1); drives recovery aggressiveness
    stamp: float = 0.0       # monotonic seconds when received


@dataclass(frozen=True)
class RecoveryContext:
    """Snapshot of the detector output that the strategy policy reasons about."""
    alarm_leg: str = ""      # "FR"/"FL"/"RR"/"RL" or ""
    confidence: float = 0.0
    intensity: float = 0.0
    fallen: bool = False     # robot posture is fallen/abnormal (escalate)
    soc: float = 100.0       # battery % (gate energy-intensive RecoveryStand)


@dataclass(frozen=True)
class MotionStep:
    """One verified Sport-API action within a strategy's MotionPlan.

    api: one of sport_api.SPORT_API_ID keys (e.g. MOVE, EULER, STOP_MOVE, BALANCE_STAND...).
    kind: ONESHOT (publish once, await response) | STREAM (re-publish each tick for duration,
          required for MOVE) | HOLD (no publish; just wait duration, e.g. let a pose settle).
    """
    api: str
    params: dict = field(default_factory=dict)
    kind: str = "ONESHOT"
    duration_s: float = 0.0


@dataclass(frozen=True)
class MotionPlan:
    """Ordered, verified actions implementing one recovery strategy."""
    name: str
    steps: tuple = ()        # tuple[MotionStep, ...]


@dataclass(frozen=True)
class RobotState:
    """Decoded robot telemetry for FSM guards."""
    posture: Posture = Posture.UNKNOWN
    mode: int = -1
    roll: float = 0.0
    pitch: float = 0.0
    soc: float = 100.0       # battery %, from LowState (100 if unknown)
    stamp: float = 0.0       # monotonic seconds of last telemetry
    fresh: bool = False      # telemetry not stale


@dataclass
class Diagnostics:
    """What the node publishes / logs each tick."""
    state: str = "IDLE"
    last_command: str = "NONE"
    detail: str = ""
    retries: int = 0
    actuation_enabled: bool = False
