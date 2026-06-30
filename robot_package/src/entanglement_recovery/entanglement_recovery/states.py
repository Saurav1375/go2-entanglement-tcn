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
    """Abstract command emitted by the FSM; SportClient maps it to a Sport api_id."""
    NONE = "NONE"
    STOP_MOVE = "STOP_MOVE"
    BALANCE_STAND = "BALANCE_STAND"
    RECOVERY_STAND = "RECOVERY_STAND"
    DAMP = "DAMP"


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
    stamp: float = 0.0  # monotonic seconds when received


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
