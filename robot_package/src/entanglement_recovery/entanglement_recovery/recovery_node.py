#!/usr/bin/env python3
"""One-shot front-jump recovery for the leg-entanglement detector.

Behaviour
---------
    NORMAL  --(sustained entanglement alarm)-->  perform ONE front jump  -->  LATCHED
    LATCHED --(ros2 service call /recovery/reset)-->  NORMAL

On the first *sustained* entanglement alarm the robot performs a single Unitree
front jump and then latches: every subsequent alarm is ignored until an operator
clears the latch with

    ros2 service call /recovery/reset std_srvs/srv/Trigger

so the robot never jumps in a loop for a continuous entanglement.

Why a subprocess?
-----------------
Actuation goes through the Unitree SDK2 (``unitree_sdk2py``) running in a child
process launched with a CLEANED DDS environment. The SDK needs its own
CycloneDDS participant on the robot's internal interface (``eth0``); the ROS 2
node's DDS env vars (``CYCLONEDDS_URI`` etc.) would otherwise block the SDK's
``ChannelFactoryInitialize``. The worker is pre-started at node init so the SDK
handshake is complete before the first alarm (zero added latency), and it is
kept alive so a post-reset re-arm is instant.

SAFETY
------
A front jump is a dynamic maneuver. Run only on flat, high-friction ground with
clear space ahead, the robot already standing, and adequate battery. This node
issues *only* the front jump: it never streams commands and never sends Damp,
StopMove, or pose changes, so it cannot make the robot suddenly collapse. The
jump fires exactly once per latch.
"""
from __future__ import annotations

import enum
import os
import select as _select
import subprocess
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger

from entanglement_interfaces.msg import EntanglementState


class _State(enum.Enum):
    NORMAL = "NORMAL"
    LATCHED = "LATCHED"


_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "sport_worker.py")


def _clean_env():
    """Strip ROS 2 / DDS env vars that block the Unitree SDK's CycloneDDS init."""
    env = os.environ.copy()
    for key in ("CYCLONEDDS_URI", "RMW_IMPLEMENTATION",
                "FASTRTPS_DEFAULT_PROFILES_FILE",
                "RMW_FASTRTPS_USE_QOS_FROM_XML"):
        env.pop(key, None)
    return env


class FrontJumpRecoveryNode(Node):
    """Triggers a single front jump on a sustained entanglement alarm.

    Parameters (see config/recovery.yaml):
        network_interface   str    Interface for the Unitree SDK2 DDS (e.g. "eth0").
        entanglement_topic  str    Topic published by the detector node.
        min_intensity       float  Ignore alarms whose max per-leg intensity is below this
                                    (0.0 = trigger on any debounced alarm).
        confirm_count       int    Consecutive qualifying alarm messages required before
                                    committing to the jump (guards against a lone stray frame;
                                    the detector is already debounced, so this is small).
    """

    def __init__(self):
        super().__init__("entanglement_recovery")

        p = self.declare_parameter
        self._iface = p("network_interface", "eth0").value
        topic = p("entanglement_topic", "/entanglement_state").value
        self._min_intensity = float(p("min_intensity", 0.0).value)
        self._confirm_count = max(1, int(p("confirm_count", 1).value))

        self._state = _State.NORMAL
        self._consec = 0                 # consecutive qualifying alarms (pre-latch)
        self._require_clear = False      # after a jump/reset, wait for one non-entangled frame
        self._worker: subprocess.Popen | None = None
        self._worker_ready = False

        # Match the detector's /entanglement_state publisher (RELIABLE/KEEP_LAST); pin it
        # explicitly so the contract can't silently drift.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(EntanglementState, topic, self._on_entanglement, qos)
        self.create_service(Trigger, "/recovery/reset", self._handle_reset)
        self._watchdog = self.create_timer(0.5, self._watchdog_tick)

        self._launch_worker()   # pre-start so the SDK is ready before the first alarm

        self.get_logger().warn(
            "front-jump recovery ready | topic={} iface={} min_intensity={:.2f} "
            "confirm={} | jumps ONCE per alarm; reset: "
            "ros2 service call /recovery/reset std_srvs/srv/Trigger".format(
                topic, self._iface or "auto", self._min_intensity, self._confirm_count))

    # ------------------------------------------------------------------ worker
    def _launch_worker(self):
        if self._worker is not None and self._worker.poll() is None:
            return
        try:
            self._worker = subprocess.Popen(
                [sys.executable, _WORKER_SCRIPT, self._iface],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                # Discard the SDK's own (possibly chatty) stderr rather than merging it into
                # the handshake pipe, so verbose CycloneDDS logging can't back up the pipe and
                # block the worker mid-command. The worker reports failures on stdout (ERROR:).
                stderr=subprocess.DEVNULL, text=True, env=_clean_env())
            self._worker_ready = False
            self.get_logger().info(
                "[JUMP] sport_worker launching (pid={})...".format(self._worker.pid))
        except Exception as exc:
            self.get_logger().error("[JUMP] failed to launch sport_worker: {}".format(exc))
            self._worker = None

    def _send(self, cmd):
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

    # ------------------------------------------------------------------ watchdog
    def _watchdog_tick(self):
        w = self._worker
        if w is None:
            self._launch_worker()
            return
        if w.poll() is not None:                       # worker died
            self.get_logger().warn("[JUMP] sport_worker exited — relaunching.")
            self._worker = None
            self._launch_worker()
            return
        # Drain the worker's stdout each tick (bounded loop) so the pipe never backs up.
        # Lines are the short handshake tokens READY / ERROR: / DONE <code>.
        for _ in range(20):
            r, _, _ = _select.select([w.stdout], [], [], 0)
            if not r:
                break
            raw = w.stdout.readline()
            if raw == "":                              # pipe closed
                break
            line = raw.strip()
            if not line:
                continue
            if line == "READY":
                if not self._worker_ready:
                    self._worker_ready = True
                    self.get_logger().info("[JUMP] sport_worker ready (SDK initialised).")
            elif line.startswith("ERROR"):
                self.get_logger().error("[JUMP] sport_worker {}".format(line))
            else:
                self.get_logger().info("[JUMP] sport_worker: {}".format(line))

    # ------------------------------------------------------------------ alarm
    def _on_entanglement(self, msg):
        if self._state is _State.LATCHED:
            return
        if not msg.entangled:
            self._consec = 0
            self._require_clear = False   # a clear frame re-arms after a jump/reset
            return
        # After a jump or a reset we require at least one non-entangled frame before counting
        # toward the next jump, so resetting while the leg is still entangled cannot immediately
        # re-jump (belt-and-suspenders on top of the LATCHED guard).
        if self._require_clear:
            return
        max_intensity = max(msg.fr_intensity, msg.fl_intensity,
                            msg.rr_intensity, msg.rl_intensity)
        if max_intensity < self._min_intensity:
            self._consec = 0
            return

        self._consec += 1
        if self._consec < self._confirm_count:
            return

        if not self._worker_ready:
            # SDK still initialising; do not latch, so the jump still fires once
            # the worker is ready and the alarm is still present.
            self.get_logger().warn("[JUMP] alarm confirmed but SDK not ready yet — waiting.")
            return

        self.get_logger().error(
            "[JUMP] FRONT JUMP — leg={} conf={:.3f} intensity={:.3f}".format(
                msg.alarm_leg or "?", msg.confidence, max_intensity))
        self._send("jump")
        self._state = _State.LATCHED     # latch immediately: jump exactly once
        self._require_clear = True       # after reset, require a clear frame before any re-jump

    # ------------------------------------------------------------------ reset
    def _handle_reset(self, _request, response):
        if self._state is _State.LATCHED:
            self._state = _State.NORMAL
            self._consec = 0
            self._require_clear = True   # wait for a non-entangled frame before re-arming
            response.success = True
            response.message = ("Front-jump latch cleared; will re-arm after a non-entangled "
                                "frame, then act on the next sustained alarm.")
            self.get_logger().info("[JUMP] latch cleared — will re-arm after a clear frame.")
        else:
            response.success = False
            response.message = "No active latch — nothing to clear."
        return response

    def destroy_node(self):
        self._kill_worker()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FrontJumpRecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
