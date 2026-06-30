"""ROS 2 adapter: decode /sportmodestate (+ optional /lowstate) into a clean RobotState.

Maps SportModeState.mode + imu_state.rpy into a Posture enum used by the FSM guards, instead
of relying on timers alone. Python 3.8 compatible.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from unitree_go.msg import SportModeState  # provided by the robot's ROS 2 env

from .states import RobotState, Posture
from . import sport_api as API

# The Unitree Go2 publishes /sportmodestate and /lowstate as BEST_EFFORT; a default (RELIABLE)
# subscription would receive nothing. Match the publisher QoS (same as the detector node).
_GO2_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST)


class RobotStateMonitor:
    def __init__(self, node, sportmode_topic, lowstate_topic=None, logger=None):
        # type: (object, str, Optional[str], object) -> None
        self.node = node
        self.log = logger
        self._mode = -1
        self._roll = 0.0
        self._pitch = 0.0
        self._soc = 100.0
        self._last_stamp = 0.0   # monotonic seconds of last SportModeState
        self.sub = node.create_subscription(SportModeState, sportmode_topic, self._on_state, _GO2_QOS)
        self._low_sub = None
        if lowstate_topic:
            try:
                from unitree_go.msg import LowState
                self._low_sub = node.create_subscription(LowState, lowstate_topic, self._on_low, _GO2_QOS)
            except Exception as exc:
                if self.log:
                    self.log.warn("LowState unavailable ({}); SOC gating disabled".format(exc))

    @staticmethod
    def _posture_from_mode(mode, roll_deg, pitch_deg, tip_roll=50.0, tip_pitch=50.0):
        # type: (int, float, float, float, float) -> Posture
        if mode in API.MODE_FALLEN:
            return Posture.FALLEN
        if abs(roll_deg) > tip_roll or abs(pitch_deg) > tip_pitch:
            return Posture.FALLEN
        if mode == API.MODE_LOCOMOTION:
            return Posture.LOCOMOTION
        if mode in API.MODE_UPRIGHT_STABLE:
            return Posture.UPRIGHT
        if mode in API.MODE_RECOVERY_IN_PROGRESS:
            return Posture.RECOVERING
        return Posture.OTHER

    def _on_state(self, msg):
        # type: (SportModeState) -> None
        try:
            self._mode = int(msg.mode)
            rpy = list(msg.imu_state.rpy)
            self._roll = math.degrees(float(rpy[0]))
            self._pitch = math.degrees(float(rpy[1]))
        except Exception as exc:
            if self.log:
                self.log.warn("bad SportModeState: {}".format(exc))
            return
        self._last_stamp = time.monotonic()

    def _on_low(self, msg):
        try:
            self._soc = float(msg.bms_state.soc)
        except Exception:
            pass

    def get(self, now, timeout_s, tip_roll=50.0, tip_pitch=50.0):
        # type: (float, float, float, float) -> RobotState
        fresh = (self._last_stamp > 0.0) and ((now - self._last_stamp) <= timeout_s)
        posture = (self._posture_from_mode(self._mode, self._roll, self._pitch, tip_roll, tip_pitch)
                   if fresh else Posture.UNKNOWN)
        return RobotState(posture=posture, mode=self._mode, roll=self._roll, pitch=self._pitch,
                          soc=self._soc, stamp=self._last_stamp, fresh=fresh)
