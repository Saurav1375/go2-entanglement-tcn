# entanglement_recovery — one-shot entanglement recovery

On a leg-entanglement alarm from the detector, the Unitree Go2 runs **one recovery sequence**, then
**latches** (ignores all further alarms) until an operator resets it. The sequence is:

```
stop  ->  move back for back_duration s  ->  stop  ->  front jump  ->  stop
```

It runs **exactly once per latch**, so a continuous entanglement can never drive a recovery loop.

```
NORMAL ──(entanglement alarm)──▶ stop → back → stop → jump → stop ──▶ LATCHED
LATCHED ──(ros2 service call /recovery/reset)──▶ NORMAL
```

Because `confirm_count` defaults to **1** and the detector output is already debounced, the sequence
starts on the **first** alarm with no added latency.

## How it actuates (and why the previous version didn't work)

Actuation goes through the **Unitree SDK2** (`unitree_sdk2py` → `SportClient`), **not** the ROS 2
`/api/sport/request` topic. The SDK runs in a **subprocess launched with a cleaned DDS environment**
(`recovery_node.py` strips `CYCLONEDDS_URI`, `RMW_IMPLEMENTATION`, `FASTRTPS_*`): the SDK2 brings up
its own CycloneDDS participant on the robot's internal interface (`eth0`), and the ROS 2 middleware
env vars would otherwise block `ChannelFactoryInitialize`. The worker is pre-started at node init so
the SDK handshake finishes before the first alarm.

The earlier FSM/strategy-based recovery failed for two reasons, both fixed here: it defaulted to a
dry-run gate (it logged intended commands but sent nothing), and even when enabled it published
`unitree_api/Request` on `/api/sport/request`, which depends on the ROS 2 sport bridge and is
overridden by the remote controller. Driving the SDK2 directly is the proven path.

## The sequence, step by step
1. `StopMove()` — halt whatever gait is running (discrete call, not streamed → no jitter).
2. `Move(-back_speed, 0, 0)` streamed at ~20 Hz for `back_duration` s — back away from the snag
   (velocity commands must be refreshed to keep the robot moving).
3. `StopMove()` — stop the backward motion and settle.
4. `FrontJump()` — a single front jump to clear the leg.
5. `StopMove()` — final stop; the robot is left standing.

`StopMove` is issued as discrete calls (never streamed) because spamming it caused jitter; only the
backward `Move` is streamed, since it sets a velocity that must be refreshed.

## Files
- `entanglement_recovery/recovery_node.py` — ROS 2 node: subscribes `/entanglement_state`, confirms
  the alarm, triggers one sequence, latches, and exposes `/recovery/reset`.
- `entanglement_recovery/sport_worker.py` — SDK2 subprocess: inits `SportClient`, runs the sequence
  once per `recover` command. Never loops.

## Build & run (on the robot)
```bash
# unitree_sdk2py must be installed into the ROS 2 interpreter:
pip3 install unitree_sdk2py            # or the Unitree-provided build
cd ~/ros2_ws && colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery
source install/setup.bash

# detector + recovery together:
ros2 launch entanglement_recovery detector_and_recovery.launch.py network_interface:=eth0
# or recovery alone (detector already running):
ros2 launch entanglement_recovery recovery.launch.py network_interface:=eth0

# re-arm after a recovery:
ros2 service call /recovery/reset std_srvs/srv/Trigger
```

## Configuration (`config/recovery.yaml`)
| key | meaning | default |
|---|---|---|
| `network_interface` | interface for the SDK2 DDS (Go2 internal ethernet) | `eth0` |
| `entanglement_topic` | detector output topic | `/entanglement_state` |
| `min_intensity` | ignore alarms below this max per-leg intensity (0 = any) | `0.0` |
| `confirm_count` | consecutive alarm messages before acting (1 = first alarm, no latency) | `1` |
| `back_speed` | backward speed (m/s) for the "move back" step | `0.3` |
| `back_duration` | duration (s) of the "move back" step | `0.5` |

## ⚠️ Safety
The sequence ends in a front jump, a dynamic maneuver. Before enabling on hardware:
- flat, high-friction ground with **clear space behind and ahead of** the robot (it backs up, then
  jumps forward/up);
- robot already **standing/balanced**, adequate **battery**;
- keep an **e-stop / remote** in hand for the first trials.

Each "stop" is a discrete `StopMove` (never streamed), so the node cannot make the robot suddenly
collapse, and the sequence runs **once per latch**. It has been verified for logic/structure
off-robot but the motions themselves are **not yet validated on hardware**; test carefully.
