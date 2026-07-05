# entanglement_recovery — one-shot front-jump recovery

On a sustained leg-entanglement alarm from the detector, the Unitree Go2 performs **a single
front jump**, then **latches** (ignores all further alarms) until an operator resets it. That is the
whole mechanism — deliberately simple, and it jumps **exactly once** so a continuous entanglement
can never drive a jump loop.

```
NORMAL ──(sustained entanglement alarm)──▶ one FrontJump() ──▶ LATCHED
LATCHED ──(ros2 service call /recovery/reset)──▶ NORMAL
```

## How it actuates (and why the previous version didn't work)

Actuation goes through the **Unitree SDK2** (`unitree_sdk2py` → `SportClient.FrontJump()`), **not**
the ROS 2 `/api/sport/request` topic. The SDK runs in a **subprocess launched with a cleaned DDS
environment** (`recovery_node.py` strips `CYCLONEDDS_URI`, `RMW_IMPLEMENTATION`, `FASTRTPS_*`): the
SDK2 brings up its own CycloneDDS participant on the robot's internal interface (`eth0`), and the
ROS 2 middleware env vars would otherwise block `ChannelFactoryInitialize`. The worker is
pre-started at node init so the SDK handshake finishes before the first alarm.

The earlier FSM-based recovery failed for two reasons, both fixed here: it defaulted to a dry-run
gate (it logged intended commands but sent nothing), and even when enabled it published
`unitree_api/Request` on `/api/sport/request`, which depends on the ROS 2 sport bridge and is
overridden by the remote controller. Driving the SDK2 directly is the proven path.

## Files
- `entanglement_recovery/recovery_node.py` — ROS 2 node: subscribes `/entanglement_state`, confirms
  the alarm, triggers one jump, latches, and exposes `/recovery/reset`.
- `entanglement_recovery/sport_worker.py` — SDK2 subprocess: inits `SportClient`, calls
  `FrontJump()` once per `jump` command. Never streams, so it cannot repeat the jump.

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

# re-arm after a jump:
ros2 service call /recovery/reset std_srvs/srv/Trigger
```

## Configuration (`config/recovery.yaml`)
| key | meaning | default |
|---|---|---|
| `network_interface` | interface for the SDK2 DDS (Go2 internal ethernet) | `eth0` |
| `entanglement_topic` | detector output topic | `/entanglement_state` |
| `min_intensity` | ignore alarms below this max per-leg intensity (0 = any) | `0.0` |
| `confirm_count` | consecutive alarm messages required before jumping | `3` |

## ⚠️ Safety
A front jump is a dynamic maneuver. Before enabling on hardware:
- flat, high-friction ground with **clear space ahead and above**;
- robot already **standing/balanced**, adequate **battery**;
- keep an **e-stop / remote** in hand for the first trials.

This node issues **only** the front jump — it never sends `Damp`, `StopMove`, or pose changes, and
never streams commands, so it cannot make the robot suddenly collapse. It has been verified for
logic/structure off-robot but the jump itself is **not yet validated on hardware**; test carefully.
