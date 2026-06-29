#!/usr/bin/env python3
"""ROS 2 node: subscribe Unitree GO2 /lowstate -> publish /entanglement_state.

CPU-only, Python 3.8 compatible. Inference runs in the subscription callback via
the standalone EntanglementEngine (numpy + ONNX Runtime by default).
"""
from __future__ import annotations

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from entanglement_interfaces.msg import EntanglementState

from .engine import EntanglementEngine
from .lowstate_adapter import lowstate_to_sample
from . import constants as K


def _load_lowstate_msg():
    """Import the Unitree LowState message type (provided by the robot's env)."""
    from unitree_go.msg import LowState
    return LowState


def _resolve(path, share_dir):
    # type: (str, str) -> str
    if os.path.isabs(path):
        return path
    return os.path.join(share_dir, path)


def _read_temperature(calibration_path, default=1.0):
    # type: (str, float) -> float
    try:
        import json
        with open(calibration_path) as f:
            return float(json.load(f).get("temperature", default))
    except Exception:
        return default


class EntanglementNode(Node):
    def __init__(self):
        super().__init__("entanglement_detector")

        # ---- parameters ----
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("entanglement_detector")

        p = self.declare_parameter
        model_path = p("model_path", "models/entanglement_tcn.onnx").value
        normalize_path = p("normalize_path", "models/normalize.json").value
        intensity_calib_path = p("intensity_calib_path", "models/intensity_calib.json").value
        calibration_path = p("calibration_path", "models/calibration.json").value
        det_thr = float(p("detection_threshold", 0.9999).value)
        debounce_ms = float(p("debounce_ms", 75.0).value)
        intensity_blend = float(p("intensity_blend", 0.5).value)
        num_threads = int(p("num_threads", 1).value)
        stationary_dq_thresh = float(p("stationary_dq_thresh", 0.0).value)
        stationary_min_ms = float(p("stationary_min_ms", 100.0).value)
        stabilize_ms = float(p("stabilize_ms", 0.0).value)
        self.publish_every_n = max(1, int(p("publish_every_n", 1).value))
        self.log_every_n = int(p("log_every_n", 250).value)
        lowstate_topic = p("lowstate_topic", "/lowstate").value
        output_topic = p("output_topic", "/entanglement_state").value
        leg_thresholds = {
            "FR": float(p("leg_thresholds.FR", 0.5).value),
            "FL": float(p("leg_thresholds.FL", 0.5).value),
            "RR": float(p("leg_thresholds.RR", 0.9).value),
            "RL": float(p("leg_thresholds.RL", 0.5).value),
        }

        temperature = _read_temperature(_resolve(calibration_path, share))
        self.engine = EntanglementEngine(
            model_path=_resolve(model_path, share),
            normalize_path=_resolve(normalize_path, share),
            intensity_calib_path=_resolve(intensity_calib_path, share),
            temperature=temperature,
            detection_threshold=det_thr,
            debounce_ms=debounce_ms,
            leg_thresholds=leg_thresholds,
            intensity_blend=intensity_blend,
            num_threads=num_threads,
            stationary_dq_thresh=stationary_dq_thresh,
            stationary_min_ms=stationary_min_ms,
            stabilize_ms=stabilize_ms,
        )

        # ---- pub / sub ----
        self.pub = self.create_publisher(EntanglementState, output_topic, 10)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        LowState = _load_lowstate_msg()
        self.sub = self.create_subscription(LowState, lowstate_topic, self._on_lowstate, qos)

        self._count = 0
        self.get_logger().info(
            "entanglement_detector up: backend={} temp={:.2f} debounce={}ms "
            "thr={} sub={} pub={}".format(
                os.path.basename(_resolve(model_path, share)), temperature,
                int(debounce_ms), det_thr, lowstate_topic, output_topic))

    def _on_lowstate(self, msg):
        try:
            sample = lowstate_to_sample(msg)
        except Exception as exc:  # malformed message -> skip this sample
            self.get_logger().warn("bad /lowstate sample: {}".format(exc))
            return

        result = self.engine.push(sample)
        if result is None:
            return  # buffer still filling (< 0.40 s of data)

        self._count += 1
        if self._count % self.publish_every_n != 0:
            return

        out = EntanglementState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "base"
        out.entangled = bool(result["entangled"])
        out.confidence = float(result["confidence"])
        pl = result["p_legs"]; it = result["intensity"]
        out.fr_probability = pl["FR"]; out.fl_probability = pl["FL"]
        out.rr_probability = pl["RR"]; out.rl_probability = pl["RL"]
        out.fr_intensity = it["FR"]; out.fl_intensity = it["FL"]
        out.rr_intensity = it["RR"]; out.rl_intensity = it["RL"]
        out.alarm_leg = result["alarm_leg"] or ""
        self.pub.publish(out)

        if self.log_every_n and result["entangled"]:
            self.get_logger().warn(
                "ENTANGLED leg={} conf={:.2f} intensity={:.2f}".format(
                    out.alarm_leg, out.confidence,
                    max(it.values())))
        elif self.log_every_n and self._count % self.log_every_n == 0:
            self.get_logger().info("ok conf={:.2f}".format(out.confidence))


def main(args=None):
    rclpy.init(args=args)
    node = EntanglementNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
