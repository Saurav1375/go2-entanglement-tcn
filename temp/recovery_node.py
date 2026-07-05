#!/usr/bin/env python3
"""Emergency-stop recovery supervisor for the leg-entanglement detector.

State machine:
    NORMAL  ->  (entanglement alarm)  ->  EMERGENCY_STOP
    EMERGENCY_STOP  ->  (ros2 service call /recovery/reset)  ->  NORMAL

A sport_worker subprocess is pre-started at node init so the Unitree SDK2
CycloneDDS stack is fully initialised before the first alarm.  On alarm the
worker receives "start" on stdin and begins sending StopMove at 100 Hz,
overriding remote controller input.  On reset the worker receives "stop" and
returns to standby — it stays alive so the next alarm has zero startup delay.
"""
from __future__ import annotations

import enum
import os
import select as _select
import subprocess
import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from entanglement_interfaces.msg import EntanglementState


class _State(enum.Enum):
    NORMAL = "NORMAL"
    EMERGENCY_STOP = "EMERGENCY_STOP"


_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "sport_worker.py")


def _clean_env():
    """Strip ROS 2 / DDS env vars that block the SDK's CycloneDDS init."""
    env = os.environ.copy()
    for key in ("CYCLONEDDS_URI", "RMW_IMPLEMENTATION",
                "FASTRTPS_DEFAULT_PROFILES_FILE",
                "RMW_FASTRTPS_USE_QOS_FROM_XML"):
        env.pop(key, None)
    return env


class EntanglementRecoveryNode(Node):
    """Latches an emergency stop on any entanglement alarm.

    Parameters (see recovery_config.yaml):
        network_interface   str    Interface for Unitree SDK2 DDS (e.g. "eth0").
        min_intensity       float  Intensity gate (0.0 = any alarm triggers stop).
        entanglement_topic  str    Topic published by the detector node.
    """

    def __init__(self):
        super().__init__("entanglement_recovery")

        p = self.declare_parameter
        self._network_interface = p("network_interface", "eth0").value
        self._min_intensity = float(p("min_intensity", 0.0).value)
        entanglement_topic = p("entanglement_topic", "/entanglement_state").value

        self._state = _State.NORMAL
        self._worker: subprocess.Popen | None = None
        self._worker_ready = False   # True once worker prints READY
        self._start_sent = False     # True while "start" is active in worker

        self.sub = self.create_subscription(
            EntanglementState, entanglement_topic, self._on_entanglement, 10)
        self.reset_srv = self.create_service(
            Trigger, "/recovery/reset", self._handle_reset)

        # Watchdog: polls READY handshake, restarts dead worker, resends
        # "start" after a worker restart while still latched.
        self._watchdog = self.create_timer(0.5, self._watchdog_tick)

        # Pre-start so SDK initialises before the first alarm.
        self._launch_worker()

        self.get_logger().info(
            "entanglement_recovery ready — "
            "topic={} min_intensity={:.2f} iface={} "
            "reset: ros2 service call /recovery/reset std_srvs/srv/Trigger".format(
                entanglement_topic, self._min_intensity,
                self._network_interface or "auto"))

    # ------------------------------------------------------------------ #
    # Worker management                                                    #
    # ------------------------------------------------------------------ #

    def _launch_worker(self):
        if self._worker is not None and self._worker.poll() is None:
            return  # already running
        cmd = [sys.executable, _WORKER_SCRIPT, self._network_interface]
        try:
            self._worker = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=_clean_env(),
            )
            self._worker_ready = False
            self._start_sent = False
            self.get_logger().info(
                "[RECOVERY] sport_worker launching (pid={})...".format(
                    self._worker.pid))
        except Exception as exc:
            self.get_logger().error(
                "[RECOVERY] Failed to launch sport_worker: {}".format(exc))
            self._worker = None

    def _send(self, cmd):
        """Write a command to the worker's stdin. Returns False on failure."""
        if self._worker is None or self._worker.poll() is not None:
            return False
        try:
            self._worker.stdin.write(cmd + "\n")
            self._worker.stdin.flush()
            return True
        except Exception:
            return False

    def _kill_worker(self):
        if self._worker is None:
            return
        self._send("quit")
        try:
            self._worker.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._worker.kill()
        self._worker = None
        self._worker_ready = False
        self._start_sent = False
        self.get_logger().info("[RECOVERY] sport_worker stopped.")

    # ------------------------------------------------------------------ #
    # Watchdog timer (0.5 s)                                               #
    # ------------------------------------------------------------------ #

    def _watchdog_tick(self):
        # Poll for READY / ERROR from the worker (non-blocking).
        if self._worker is not None and not self._worker_ready:
            if self._worker.poll() is not None:
                self.get_logger().error(
                    "[RECOVERY] sport_worker died during init — will retry.")
                self._worker = None
            else:
                r, _, _ = _select.select([self._worker.stdout], [], [], 0)
                if r:
                    line = self._worker.stdout.readline().strip()
                    if line == "READY":
                        self._worker_ready = True
                        self.get_logger().info("[RECOVERY] sport_worker ready.")
                    elif line.startswith("ERROR"):
                        self.get_logger().error(
                            "[RECOVERY] sport_worker: {}".format(line))
                        self._worker.wait()
                        self._worker = None

        # While latched: ensure worker is running and "start" is sent.
        if self._state is _State.EMERGENCY_STOP:
            if self._worker is None or self._worker.poll() is not None:
                self.get_logger().warn(
                    "[RECOVERY] sport_worker died while latched — restarting.")
                self._launch_worker()
            elif self._worker_ready and not self._start_sent:
                # Worker (re)became ready while already latched — send start.
                self._send("start")
                self._start_sent = True

        # In standby: restart worker if it died unexpectedly.
        if self._state is _State.NORMAL and self._worker is None:
            self._launch_worker()

    # ------------------------------------------------------------------ #
    # Subscriber callback                                                  #
    # ------------------------------------------------------------------ #

    def _on_entanglement(self, msg):
        if not msg.entangled:
            return
        max_intensity = max(
            msg.fr_intensity, msg.fl_intensity,
            msg.rr_intensity, msg.rl_intensity)
        if max_intensity < self._min_intensity:
            return

        if self._state is _State.NORMAL:
            self.get_logger().error(
                "[RECOVERY] EMERGENCY STOP — leg={} conf={:.3f} intensity={:.3f}".format(
                    msg.alarm_leg or "?", msg.confidence, max_intensity))
            if self._worker_ready:
                self._send("start")
                self._start_sent = True
            else:
                # SDK still initialising — watchdog sends "start" once ready.
                self.get_logger().warn(
                    "[RECOVERY] sport_worker still initialising; "
                    "StopMove begins once SDK is ready (~1 s).")

        self._state = _State.EMERGENCY_STOP

    # ------------------------------------------------------------------ #
    # Reset service                                                        #
    # ------------------------------------------------------------------ #

    def _handle_reset(self, _request, response):
        if self._state is _State.EMERGENCY_STOP:
            self._send("stop")
            self._start_sent = False
            self._state = _State.NORMAL
            self.get_logger().info("[RECOVERY] latch cleared — returning to NORMAL.")
            response.success = True
            response.message = "Entanglement latch cleared. Robot may now move."
        else:
            response.success = False
            response.message = "No active latch — nothing to clear."
        return response

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def destroy_node(self):
        self._kill_worker()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EntanglementRecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
