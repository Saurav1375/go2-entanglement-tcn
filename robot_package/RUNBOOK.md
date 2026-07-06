# GO2 Hardware Runbook — Detector + Recovery (single source)

End-to-end deploy + test of the **full pipeline** on a Unitree Go2: the entanglement **detector**
→ a **one-shot recovery** that, on the first alarm, runs the sequence
**stop → move back → stop → front jump → stop** once, then latches until reset.

The recovery ends in a **front jump**. Bring the detector up first (no motion), confirm it alarms
correctly, and only launch the recovery node on safe ground with an e-stop in hand.

## 0. Prerequisites (on the robot)
- ROS 2 (e.g. Foxy/Humble, Python 3.8) with the Unitree stack providing **`unitree_go`** messages
  and publishing `/lowstate` (verify: `ros2 topic list | grep lowstate`).
- Detector python deps: `numpy`, `onnxruntime`, `PyYAML`.
- Recovery: the **Unitree SDK2** Python bindings (`unitree_sdk2py`) installed into the ROS 2
  interpreter (`pip3 install unitree_sdk2py`, or the Unitree-provided build). Recovery drives the
  SDK's `SportClient` directly on the robot's internal ethernet (`eth0`) — not the ROS
  `/api/sport/request` topic — from a subprocess with a cleaned DDS environment (handled internally).

## 1. Copy all three packages into a colcon workspace
```bash
scp -r robot_package/src/entanglement_interfaces  unitree@<go2-ip>:~/ros2_ws/src/
scp -r robot_package/src/entanglement_detector     unitree@<go2-ip>:~/ros2_ws/src/
scp -r robot_package/src/entanglement_recovery     unitree@<go2-ip>:~/ros2_ws/src/
scp robot_package/requirements_robot.txt           unitree@<go2-ip>:~/ros2_ws/
```

## 2. Install deps + build
```bash
ssh unitree@<go2-ip>
cd ~/ros2_ws && pip3 install -r requirements_robot.txt
pip3 install unitree_sdk2py                          # recovery only
source /opt/ros/<distro>/setup.bash
colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery
source install/setup.bash
ros2 interface show entanglement_interfaces/msg/EntanglementState   # sanity
```

## 3. Network / DDS (often required to see the Unitree topics)
The Unitree stack publishes telemetry as **BEST_EFFORT** (the detector matches this QoS). You
typically also need the middleware/network aligned with the robot:
```bash
export ROS_DOMAIN_ID=<robot's domain id>            # match the robot
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp         # if the Go2 stack uses CycloneDDS
export CYCLONEDDS_URI=file:///path/to/cyclonedds.xml # bind to the robot-facing interface
ros2 topic hz /lowstate                              # verify you actually receive data
```
Note: the recovery worker deliberately runs with these DDS env vars **stripped** so the SDK2 can
bring up its own CycloneDDS participant on `eth0`; that is handled inside `recovery_node.py` and does
not affect the ROS environment above.

## 4. Detector only (no motion)
```bash
ros2 launch entanglement_detector entanglement.launch.py
ros2 topic echo /entanglement_state        # entangled / confidence / per-leg prob+intensity / alarm_leg
ros2 topic hz   /entanglement_state        # ~ /lowstate rate
```
Induce a snag and confirm the detector alarms on the right leg with no false alarms while
walking/standing before enabling recovery.

## 5. Recovery bring-up (motion — clearance required)
Pre-flight: flat high-friction floor, **clear space behind and ahead of** the robot (it backs up,
then jumps), robot standing/balanced, adequate battery, **e-stop in hand**.
```bash
ros2 launch entanglement_recovery detector_and_recovery.launch.py network_interface:=eth0
```
- Watch the node log. On startup it prints the config and `sport_worker ready (SDK initialised)` once
  the SDK2 handshake completes. If instead you see `sport_worker ERROR: ...`, the SDK isn't installed
  or `eth0` is wrong — fix that before proceeding (the node will not actuate until the worker is ready).
- On the **first** alarm the node logs `entanglement — leg=… : stop -> back -> stop -> jump -> stop`,
  sends `recover` **once**, and latches. The worker runs the sequence and reports `DONE <codes>`.
- Every further alarm is ignored until you re-arm:
  ```bash
  ros2 service call /recovery/reset std_srvs/srv/Trigger
  ```
  After a reset the node waits for one non-entangled frame before it will act again.

Tune `back_speed` / `back_duration` in `entanglement_recovery/config/recovery.yaml` from what you
observe. Full design + config + safety: [`../docs/recovery/RECOVERY.md`](../docs/recovery/RECOVERY.md).

## Safety recap
- The sequence ends in a front jump — a dynamic maneuver. Only launch the recovery node with
  clearance and an e-stop in hand.
- Each "stop" is a discrete `StopMove` (never streamed), so the robot cannot suddenly collapse, and
  the sequence runs **once per latch** — a continuous entanglement can never loop the jump.
- **Not yet hardware-validated**: whether stop→back→jump actually frees a snagged leg, and the chosen
  `back_speed` / `back_duration`, are design assumptions. Treat the first actuated runs as experimental.

## Before the robot: pure logic tests (no ROS / no SDK needed)
```bash
python3 robot_package/src/entanglement_recovery/test/test_recovery_sequence.py   # sequence order, one-shot
python3 robot_package/src/entanglement_recovery/test/test_recovery_node.py       # latch, confirm, re-arm
python3 robot_package/tools/validate_runtime.py --backend onnx                   # detector == evaluate.py
```
