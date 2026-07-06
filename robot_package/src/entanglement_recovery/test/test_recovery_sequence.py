#!/usr/bin/env python3
"""Verify sport_worker.py runs the sequence stop -> back -> stop -> jump -> stop exactly once,
with the Unitree SDK mocked (no robot / no SDK install needed).

Run:  python3 robot_package/src/entanglement_recovery/test/test_recovery_sequence.py
"""
import os
import sys
import io
import types

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "entanglement_recovery")
sys.path.insert(0, PKG)


def install_sdk_mock(calls):
    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelFactoryInitialize = lambda d=0, i="": calls.append(("init", d, i))
    sc = types.ModuleType("unitree_sdk2py.go2.sport.sport_client")

    class SportClient:
        def SetTimeout(self, t): calls.append(("SetTimeout", t))
        def Init(self): calls.append(("Init",))
        def StopMove(self): calls.append(("StopMove",)); return 0
        def Move(self, vx, vy, vyaw): calls.append(("Move", round(vx, 3), vy, vyaw)); return 0
        def FrontJump(self): calls.append(("FrontJump",)); return 0
    sc.SportClient = SportClient
    for n, m in {
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": channel,
        "unitree_sdk2py.go2": types.ModuleType("unitree_sdk2py.go2"),
        "unitree_sdk2py.go2.sport": types.ModuleType("unitree_sdk2py.go2.sport"),
        "unitree_sdk2py.go2.sport.sport_client": sc,
    }.items():
        sys.modules[n] = m


def run(stdin_text, back_speed="0.3", back_duration="4"):
    calls = []
    install_sdk_mock(calls)
    sys.modules.pop("sport_worker", None)
    import sport_worker
    # make time deterministic + instant: monotonic counts up, sleep is a no-op
    counter = {"t": 0}

    def fake_monotonic():
        counter["t"] += 1
        return counter["t"]
    sport_worker.time.sleep = lambda *a: None
    sport_worker.time.monotonic = fake_monotonic
    old_in, old_out, old_argv = sys.stdin, sys.stdout, sys.argv
    sys.stdin = io.StringIO(stdin_text); out = io.StringIO(); sys.stdout = out
    sys.argv = ["sport_worker", "eth-test", back_speed, back_duration]
    try:
        sport_worker.main()
    finally:
        sys.stdin, sys.stdout, sys.argv = old_in, old_out, old_argv
    return calls, out.getvalue()


fails = []


def check(cond, label, extra=""):
    print(("PASS " if cond else "FAIL ") + label + ("" if cond else "  -> " + str(extra)))
    if not cond:
        fails.append(label)


calls, out = run("recover\nquit\n")
seq = [c for c in calls if c[0] in ("StopMove", "Move", "FrontJump")]
kinds = [c[0] for c in seq]

check("READY" in out, "prints READY")
check(out.count("DONE") == 1, "reports DONE once", out)
check(kinds and kinds[0] == "StopMove", "sequence starts with StopMove", kinds[:1])
check("Move" in kinds, "includes a backward Move phase")
moves = [c for c in seq if c[0] == "Move"]
check(all(m[1] < 0 for m in moves), "all Move calls are backward (vx<0)", moves[:3])
check(kinds.count("FrontJump") == 1, "FrontJump called exactly once", kinds.count("FrontJump"))
check(kinds[-1] == "StopMove", "sequence ends with StopMove", kinds[-1:])
order = [k for i, k in enumerate(kinds) if i == 0 or kinds[i] != kinds[i - 1]]
check(order == ["StopMove", "Move", "StopMove", "FrontJump", "StopMove"],
      "order is stop -> back -> stop -> jump -> stop", order)

# a single 'recover' must not loop the jump
calls2, _ = run("recover\nquit\n")
check([c[0] for c in calls2].count("FrontJump") == 1, "one 'recover' -> exactly one FrontJump (no loop)")

# no command -> no motion at all
calls3, _ = run("quit\n")
check(not any(c[0] in ("StopMove", "Move", "FrontJump") for c in calls3), "no 'recover' -> no motion")

print("\n%s (%d failed)" % ("ALL WORKER CHECKS PASS" if not fails else "WORKER FAILURES", len(fails)))
sys.exit(1 if fails else 0)
