# Entanglement Recovery — one-shot sequence

When the detector raises a leg-entanglement alarm, the Unitree Go2 runs **one recovery sequence**
and then **latches** until an operator resets it:

```
stop  ->  move back for back_duration s  ->  stop  ->  front jump  ->  stop
```

The sequence runs **exactly once per latch**, so a continuous entanglement can never drive a
recovery loop. Because `confirm_count` defaults to **1** and the detector output is already
debounced, the sequence starts on the **first** alarm — no added latency.

```
NORMAL ──(entanglement alarm)──▶ stop → back → stop → jump → stop ──▶ LATCHED
LATCHED ──(ros2 service call /recovery/reset)──▶ NORMAL
```

The ROS package that implements this is `robot_package/src/entanglement_recovery/`. This document is
the single source of truth for the recovery design, configuration, and safety; the package README is
a short operational quick-start.

## The sequence, step by step
1. **stop** — `StopMove()` halts whatever gait is running. A discrete call, not streamed (spamming
   `StopMove` caused jitter).
2. **move back** — `Move(-back_speed, 0, 0)` streamed at ~20 Hz for `back_duration` seconds to back
   away from the snag. Velocity commands must be refreshed to keep the robot moving, so this step
   *is* streamed.
3. **stop** — `StopMove()` stops the backward motion and lets the robot settle.
4. **jump** — `FrontJump()`, a single front jump to clear the entangled leg.
5. **stop** — `StopMove()` final stop; the robot is left standing.

Settle pauses between steps (0.3 s after the first stop, 0.4 s after the back-up, ~3 s for the jump
to physically complete) are fixed in `sport_worker.py`.

## Why the Unitree SDK2 (and not ROS `/api/sport/request`)
Actuation goes through the **Unitree SDK2** (`unitree_sdk2py` → `SportClient`) running in a
**subprocess with a cleaned DDS environment** — `recovery_node.py` strips `CYCLONEDDS_URI`,
`RMW_IMPLEMENTATION`, and `FASTRTPS_*` before launching it. The SDK brings up its own CycloneDDS
participant on the robot's internal interface (`eth0`); the ROS 2 middleware env vars would otherwise
block `ChannelFactoryInitialize`. The worker is pre-started at node init so the SDK handshake is done
before the first alarm.

The earlier FSM/strategy recovery did nothing on hardware for two reasons, both fixed here:
- it defaulted to a **dry-run** gate (logged intended commands, sent nothing), and
- even when enabled it published `unitree_api/Request` on `/api/sport/request`, which depends on the
  ROS 2 sport bridge and is overridden by the remote controller.

Driving the SDK2 directly is the proven path.

## Configuration (`config/recovery.yaml`)
| key | meaning | default |
|---|---|---|
| `network_interface` | interface for the SDK2 DDS (Go2 internal ethernet) | `eth0` |
| `entanglement_topic` | detector output topic | `/entanglement_state` |
| `min_intensity` | ignore alarms below this max per-leg intensity (0 = any) | `0.0` |
| `confirm_count` | consecutive alarm messages before acting (1 = first alarm, no latency) | `1` |
| `back_speed` | backward speed (m/s) for the "move back" step | `0.3` |
| `back_duration` | duration (s) of the "move back" step | `0.5` |

## Run
```bash
# detector + recovery together:
ros2 launch entanglement_recovery detector_and_recovery.launch.py network_interface:=eth0
# recovery alone (detector already running):
ros2 launch entanglement_recovery recovery.launch.py network_interface:=eth0

# re-arm after a recovery:
ros2 service call /recovery/reset std_srvs/srv/Trigger
```

## One-shot latch and re-arming
- On the first qualifying alarm the node sends `recover` to the worker once and moves to `LATCHED`.
- While `LATCHED`, every alarm is ignored — the sequence cannot repeat for one entanglement.
- `/recovery/reset` (std_srvs/Trigger) clears the latch. To avoid an immediate re-trigger while the
  leg is still snagged, the node then requires **one non-entangled frame** before it will act again.

## Safety
The sequence ends in a front jump, a dynamic maneuver. Before enabling on hardware:
- flat, high-friction ground with **clear space behind and ahead of** the robot (it backs up, then
  jumps forward/up);
- robot already **standing/balanced**, adequate **battery**;
- keep an **e-stop / remote** in hand for the first trials.

Each "stop" is a discrete `StopMove` (never streamed), so the node cannot make the robot suddenly
collapse, and the sequence runs once per latch. The logic (step order, one-shot latch, no loop) is
verified off-robot; the motions themselves are **not yet validated on hardware** — test carefully.
