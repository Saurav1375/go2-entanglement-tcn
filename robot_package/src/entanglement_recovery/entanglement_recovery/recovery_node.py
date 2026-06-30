#!/usr/bin/env python3
"""ROS 2 recovery orchestrator node.

Subscribes the detector's /entanglement_state, ticks the pure RecoveryFSM on a timer using
robot telemetry (/sportmodestate, /lowstate) and Sport API responses, and executes the FSM's
commands via SportClient. Publishes /recovery_status. The detector is untouched.

SAFETY: actuation is OFF by default (`enable_actuation:=false`) — the FSM runs and logs the
commands it WOULD send (dry-run), enabling safe validation. Set true to actuate the robot.
Python 3.8 compatible.
"""
from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Empty

from entanglement_interfaces.msg import EntanglementState

from .recovery_fsm import RecoveryFSM, RecoveryConfig
from .states import Command, Detection
from .robot_state import RobotStateMonitor
from .sport_client import SportClient
from . import sport_api as API

_AWAIT_CMDS = (Command.STOP_MOVE, Command.BALANCE_STAND, Command.RECOVERY_STAND)


class RecoveryNode(Node):
    def __init__(self):
        super().__init__("entanglement_recovery")
        p = self.declare_parameter

        # ---- config -> RecoveryConfig (no magic numbers; all from params/yaml) ----
        cfg = RecoveryConfig(
            confirmation_time_s=float(p("confirmation_time_s", 0.5).value),
            command_timeout_s=float(p("command_timeout_s", 1.0).value),
            retry_limit=int(p("retry_limit", 2).value),
            stop_settle_s=float(p("stop_settle_s", 0.5).value),
            recovery_settle_s=float(p("recovery_settle_s", 2.0).value),
            verification_duration_s=float(p("verification_duration_s", 1.5).value),
            verify_timeout_s=float(p("verify_timeout_s", 4.0).value),
            resume_delay_s=float(p("resume_delay_s", 1.0).value),
            cooldown_s=float(p("cooldown_s", 5.0).value),
            robot_state_timeout_s=float(p("robot_state_timeout_s", 0.5).value),
            max_state_time_s=float(p("max_state_time_s", 10.0).value),
            auto_reset_s=float(p("auto_reset_s", 0.0).value),
            confidence_min=float(p("confidence_min", 0.0).value),
            escalate_to_recovery_stand=bool(p("escalate_to_recovery_stand", True).value),
            use_damp_on_fault=bool(p("use_damp_on_fault", True).value),
            tip_roll_deg=float(p("tip_roll_deg", 50.0).value),
            tip_pitch_deg=float(p("tip_pitch_deg", 50.0).value),
            min_soc_pct=float(p("min_soc_pct", 10.0).value),
        )
        self.cfg = cfg
        self.enable_actuation = bool(p("enable_actuation", False).value)
        self.control_rate_hz = float(p("control_rate_hz", 50.0).value)
        auto_arm = bool(p("auto_arm", True).value)

        # ---- topics ----
        ent_topic = p("entanglement_topic", "/entanglement_state").value
        status_topic = p("status_topic", "/recovery_status").value
        sport_req = p("sport_request_topic", API.TOPIC_SPORT_REQUEST).value
        sport_resp = p("sport_response_topic", API.TOPIC_SPORT_RESPONSE).value
        sportmode = p("sportmode_topic", API.TOPIC_SPORT_MODE_STATE).value
        lowstate = p("lowstate_topic", API.TOPIC_LOW_STATE).value
        estop_topic = p("estop_topic", "/recovery_estop").value
        reset_topic = p("reset_topic", "/recovery_reset").value

        # ---- wiring ----
        self.fsm = RecoveryFSM(cfg)
        self.robot = RobotStateMonitor(self, sportmode, lowstate, self.get_logger())
        self.sport = SportClient(self, sport_req, sport_resp, self.enable_actuation, self.get_logger())
        self.sport.set_result_callback(self.fsm.on_command_result)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(EntanglementState, ent_topic, self._on_detection, qos)
        self.create_subscription(Empty, estop_topic, lambda _m: self.fsm.request_estop(), 10)
        self.create_subscription(Empty, reset_topic, lambda _m: self.fsm.reset(time.monotonic()), 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)

        if auto_arm:
            self.fsm.arm(time.monotonic())
        self._last_state = None
        self.timer = self.create_timer(1.0 / max(self.control_rate_hz, 1.0), self._tick)
        self.get_logger().warn(
            "entanglement_recovery up | actuation={} | strategy=StopMove->BalanceStand "
            "(escalate RecoveryStand on fall) | sub={} | sport_req={}".format(
                "ENABLED" if self.enable_actuation else "DRY-RUN (no commands sent)",
                ent_topic, sport_req))

    # ---- callbacks ----
    def _on_detection(self, msg):
        self.fsm.on_detection(Detection(
            entangled=bool(msg.entangled), confidence=float(msg.confidence),
            alarm_leg=str(msg.alarm_leg), stamp=time.monotonic()))

    def _tick(self):
        now = time.monotonic()
        self.fsm.on_robot_state(self.robot.get(
            now, self.cfg.robot_state_timeout_s, self.cfg.tip_roll_deg, self.cfg.tip_pitch_deg))
        cmd = self.fsm.update(now)
        if cmd != Command.NONE:
            self.sport.send(cmd)
            # In DRY-RUN the robot never replies, so simulate success to let the FSM
            # progress and reveal the full intended sequence (logged, not actuated).
            if not self.enable_actuation and cmd in _AWAIT_CMDS:
                self.fsm.on_command_result(cmd, True)
        self._publish_status()

    def _publish_status(self):
        d = self.fsm.diagnostics(self.enable_actuation)
        msg = String()
        msg.data = json.dumps({
            "state": d.state, "last_command": d.last_command, "detail": d.detail,
            "retries": d.retries, "actuation_enabled": d.actuation_enabled})
        self.status_pub.publish(msg)
        if d.state != self._last_state:
            self.get_logger().info("recovery state -> {} ({})".format(d.state, d.detail))
            self._last_state = d.state


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
