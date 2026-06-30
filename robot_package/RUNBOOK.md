# GO2 Hardware Runbook — Detector + Recovery (single source)

End-to-end deploy + test of the **full pipeline** (entanglement detector → intelligent recovery)
on a Unitree Go2. Consolidates `SETUP_GO2.md` (detector), `docs/RECOVERY_DELIVERABLES.md`
(recovery), and `docs/RECOVERY_TESTING.md` (test tiers). **Start every session in dry-run.**

## 0. Prerequisites (on the robot)
- ROS 2 (e.g. Foxy/Humble, Python 3.8) with the Unitree stack providing **`unitree_api`** and
  **`unitree_go`** messages, publishing `/lowstate`, `/sportmodestate`, and accepting
  `/api/sport/request` (verify: `ros2 topic list | grep -E "lowstate|sportmodestate|sport/request"`).
- Python runtime deps (detector): `numpy`, `onnxruntime`, `PyYAML` (recovery needs nothing extra).

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
source /opt/ros/<distro>/setup.bash
colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery
source install/setup.bash
ros2 interface show entanglement_interfaces/msg/EntanglementState   # sanity
```

## 3. Network / DDS (often required to see the Unitree topics)
The Unitree stack publishes telemetry as **BEST_EFFORT** (both nodes already match this QoS). You
typically also need the middleware/network aligned with the robot:
```bash
export ROS_DOMAIN_ID=<robot's domain id>            # match the robot
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp         # if the Go2 stack uses CycloneDDS
export CYCLONEDDS_URI=file:///path/to/cyclonedds.xml # bind to the robot-facing interface
# verify you actually receive data:
ros2 topic hz /lowstate ; ros2 topic hz /sportmodestate
```
If `ros2 topic hz /sportmodestate` shows nothing, recovery telemetry guards fall back to timeouts
(degraded) — fix DDS before trusting posture-aware behavior.

## 4. Tier A — detector only (no motion)
```bash
ros2 launch entanglement_detector entanglement.launch.py
ros2 topic echo /entanglement_state        # entangled / confidence / per-leg prob+intensity / alarm_leg
ros2 topic hz   /entanglement_state        # ~ /lowstate rate
```

## 5. Tier B — recovery in DRY-RUN (still no motion)
```bash
ros2 launch entanglement_recovery detector_and_recovery.launch.py     # enable_actuation:=false (default)
ros2 topic echo /recovery_status           # JSON: state, strategy i/N, last_command, actuation_enabled
# induce a synthetic detection (no robot needed):
ros2 topic pub --once /entanglement_state entanglement_interfaces/msg/EntanglementState \
  "{entangled: true, confidence: 0.95, alarm_leg: 'RR', rr_intensity: 0.4}"
```
Expect the log to show `[DRY-RUN] would send StopMove (1003) … BalanceStand (1002) … Move …` and
the status to walk the strategy ladder. **Verify the sequence/leg-direction/timing before actuating.**
E-stop / reset triggers: `ros2 topic pub --once /recovery_estop std_msgs/msg/Empty "{}"` /
`/recovery_reset`.

## 6. Tier C — ACTUATED bring-up (motion — clearance required)
Pre-flight: ≥2 m clearance, flat high-friction floor, battery > `min_soc_pct`, **e-stop in hand**,
sport service active, start with a gentle induced snag.
```bash
ros2 launch entanglement_recovery detector_and_recovery.launch.py enable_actuation:=true
```
Watch `/recovery_status`; confirm StopMove→BalanceStand happens without violent motion; confirm
`/recovery_estop` damps instantly. Tune `recovery.yaml` magnitudes/directions (all flagged as
design assumptions — esp. the Euler weight-shift **sign convention** and Move directions).

## Safety recap
- `enable_actuation` defaults **false**; the combined launch arg must be set true to move.
- `Damp` is the soft e-stop (`/recovery_estop`) and the terminal action on FAULT.
- Recovery never re-enters while active; cooldown after each cycle; per-state watchdog → FAULT(Damp).
- **Not yet hardware-validated**: leg-freeing efficacy + leg directions + magnitudes are unproven
  (see `docs/INTELLIGENT_RECOVERY.md` §6 Verified-vs-Assumption). Treat Tier C as experimental.

## Tier-0 (before the robot): pure logic tests
```bash
PYTHONPATH=robot_package/src/entanglement_recovery python3 \
  robot_package/src/entanglement_recovery/test/test_recovery_fsm.py     # 16/16
PYTHONPATH=robot_package/src/entanglement_recovery python3 \
  robot_package/src/entanglement_recovery/test/test_plan_runner.py      # 3/3
python3 robot_package/tools/validate_runtime.py --backend onnx          # detector == evaluate.py
```
