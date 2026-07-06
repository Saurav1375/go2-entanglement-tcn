#!/usr/bin/env python3
"""Verify recovery_node decision logic (ROS + subprocess mocked): one-shot latch, confirm_count,
min_intensity gate, non-entangled reset, worker-not-ready deferral, reset re-arm. No ROS/SDK needed.

Run:  python3 robot_package/src/entanglement_recovery/test/test_recovery_node.py
"""
import os
import sys
import types

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "entanglement_recovery")
sys.path.insert(0, PKG)


class _Logger:
    def warn(self, *a): pass
    def info(self, *a): pass
    def error(self, *a): pass


class _Param:
    def __init__(self, v): self.value = v


class _Node:
    def __init__(self, name): self._log = _Logger()
    def declare_parameter(self, name, default): return _Param(default)
    def create_subscription(self, *a, **k): return None
    def create_service(self, *a, **k): return None
    def create_timer(self, *a, **k): return None
    def get_logger(self): return self._log
    def destroy_node(self): pass


rclpy = types.ModuleType("rclpy")
rclpy.init = rclpy.shutdown = rclpy.spin = lambda *a, **k: None
rclpy_node = types.ModuleType("rclpy.node"); rclpy_node.Node = _Node
rclpy_qos = types.ModuleType("rclpy.qos")
rclpy_qos.QoSProfile = lambda **k: None


class _RP:
    RELIABLE = 1; BEST_EFFORT = 2


class _HP:
    KEEP_LAST = 1


rclpy_qos.ReliabilityPolicy = _RP; rclpy_qos.HistoryPolicy = _HP
std_srvs = types.ModuleType("std_srvs"); std_srvs_srv = types.ModuleType("std_srvs.srv")


class Trigger:
    class Response:
        def __init__(self): self.success = None; self.message = ""


std_srvs_srv.Trigger = Trigger
ei = types.ModuleType("entanglement_interfaces"); ei_msg = types.ModuleType("entanglement_interfaces.msg")


class EntanglementState:
    pass


ei_msg.EntanglementState = EntanglementState
for n, m in {"rclpy": rclpy, "rclpy.node": rclpy_node, "rclpy.qos": rclpy_qos,
             "std_srvs": std_srvs, "std_srvs.srv": std_srvs_srv,
             "entanglement_interfaces": ei, "entanglement_interfaces.msg": ei_msg}.items():
    sys.modules[n] = m

sys.modules.pop("recovery_node", None)
import recovery_node


class _FakeProc:
    pid = 4321
    def poll(self): return None


recovery_node.subprocess.Popen = lambda *a, **k: _FakeProc()


def make_node(confirm=1, min_int=0.0):
    n = recovery_node.RecoveryNode()
    n._confirm_count = confirm; n._min_intensity = min_int; n._worker_ready = True
    n._sent = []; n._send = lambda cmd: (n._sent.append(cmd) or True)
    return n


def msg(entangled=True, inten=1.0, conf=0.99, leg="RR"):
    m = EntanglementState()
    m.entangled = entangled; m.confidence = conf; m.alarm_leg = leg
    m.fr_intensity = m.fl_intensity = m.rr_intensity = m.rl_intensity = 0.0
    m.rr_intensity = inten
    return m


S = recovery_node._State
fails = []


def check(cond, label, extra=""):
    print(("PASS " if cond else "FAIL ") + label + ("" if cond else "  -> " + str(extra)))
    if not cond:
        fails.append(label)


# 1) confirm_count gating + one-shot latch + no loop over continuous alarms
n = make_node(confirm=3)
n._on_entanglement(msg()); n._on_entanglement(msg())
check(n._sent == [] and n._state is S.NORMAL, "below confirm_count: no action")
n._on_entanglement(msg())
check(n._sent == ["recover"] and n._state is S.LATCHED, "confirm reached: one 'recover', LATCHED")
for _ in range(20):
    n._on_entanglement(msg())
check(n._sent == ["recover"], "continuous alarms after latch -> still exactly ONE sequence (no loop)")

# 2) reset requires a clear frame before re-arming
r = n._handle_reset(None, Trigger.Response())
check(r.success and n._state is S.NORMAL, "reset clears latch")
for _ in range(5):
    n._on_entanglement(msg())
check(n._sent == ["recover"], "reset while still entangled -> no immediate re-trigger")
n._on_entanglement(msg(entangled=False))
n._on_entanglement(msg()); n._on_entanglement(msg()); n._on_entanglement(msg())
check(n._sent == ["recover", "recover"], "after a clear frame, a new alarm triggers once more")

# 3) min_intensity gate
n = make_node(confirm=1, min_int=0.5)
for _ in range(5):
    n._on_entanglement(msg(inten=0.3))
check(n._sent == [], "below min_intensity: no action")
n._on_entanglement(msg(inten=0.9))
check(n._sent == ["recover"], "at/above min_intensity: acts")

# 4) worker-not-ready deferral
n = make_node(confirm=1); n._worker_ready = False
n._on_entanglement(msg())
check(n._sent == [] and n._state is S.NORMAL, "SDK not ready: defers, no latch")
n._worker_ready = True; n._on_entanglement(msg())
check(n._sent == ["recover"] and n._state is S.LATCHED, "once ready, acts exactly once")

# 5) default confirm_count is 1 (act on first alarm, no latency)
nd = recovery_node.RecoveryNode()
check(nd._confirm_count == 1, "default confirm_count is 1 (no added latency)", nd._confirm_count)

print("\n%s (%d failed)" % ("ALL NODE CHECKS PASS" if not fails else "NODE FAILURES", len(fails)))
sys.exit(1 if fails else 0)
