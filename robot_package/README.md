# Leg-Entanglement Detector + Recovery — Unitree GO2 Deployment Package

Self-contained ROS 2 workspace `src/` with **three packages** that run the full pipeline **on the
robot** (CPU-only):

- **`entanglement_interfaces`** — the `EntanglementState` message (ament_cmake).
- **`entanglement_detector`** — subscribes to `/lowstate`, runs the trained multi-task TCN every
  sample, publishes `/entanglement_state` (entangled flag, calibrated confidence, per-leg
  probabilities, per-leg intensities). Inference uses **numpy + ONNX Runtime** — no training code,
  dataset, pandas, scikit-learn, scipy, or matplotlib.
- **`entanglement_recovery`** — consumes `/entanglement_state` and, on the first alarm, runs **one
  recovery sequence** (stop → move back → stop → front jump → stop) via the Unitree SDK2, then
  latches until reset. A thin ROS node drives an SDK2 subprocess; it fires **once per latch**.

> **Deploying on hardware? Follow [`RUNBOOK.md`](RUNBOOK.md)** — the single authoritative procedure
> (network/DDS, staged bring-up). [`SETUP_GO2.md`](SETUP_GO2.md) covers the detector alone. Recovery
> design/config/safety: [`../docs/recovery/RECOVERY.md`](../docs/recovery/RECOVERY.md).

> The research/training pipeline lives in the parent repository. The model here was
> exported from it unchanged; performance is identical (validated to ~1e-6, see below).

---

## Contents

```
robot_package/
├── README.md                     # this file
├── RUNBOOK.md                    # ★ single hardware procedure (detector + recovery)
├── SETUP_GO2.md                  # detector-only step-by-step setup
├── requirements_robot.txt        # runtime python deps (numpy, onnxruntime, pyyaml)
├── export_model.py               # DEV-SIDE: export ONNX/TorchScript from the trained model
├── tools/validate_runtime.py     # DEV-SIDE: confirm runtime == evaluate.py + benchmark
└── src/
    ├── entanglement_interfaces/  # ament_cmake pkg: EntanglementState.msg
    ├── entanglement_detector/    # ament_python pkg: the detector node + runtime
    │   ├── config/config.yaml    # model paths, threshold, debounce, per-leg thresholds, gate
    │   ├── launch/entanglement.launch.py
    │   ├── models/               # entanglement_tcn.onnx + *_ts.pt + *.json (from export)
    │   └── entanglement_detector/
    │       ├── constants.py       # channel layout / GO2 ordering
    │       ├── preprocess.py      # 60-channel window + normalization (numpy)
    │       ├── intensity.py       # physics intensity (numpy)
    │       ├── model_backend.py   # ONNX Runtime / TorchScript backend
    │       ├── engine.py          # ring buffer + inference + debounce + thresholds + gate
    │       ├── lowstate_adapter.py# unitree_go/LowState -> model input
    │       └── node.py            # ROS 2 node
    └── entanglement_recovery/    # ament_python pkg: recovery node + SDK2 worker
        ├── config/recovery.yaml  # recovery params (iface, confirm_count, back_speed, back_duration)
        ├── launch/recovery.launch.py
        ├── launch/detector_and_recovery.launch.py   # launches detector + recovery together
        └── entanglement_recovery/
            ├── recovery_node.py   # ROS node: /entanglement_state -> run sequence once -> latch;
            │                      #           exposes /recovery/reset; launches the SDK2 subprocess
            └── sport_worker.py    # Unitree SDK2 subprocess (cleaned DDS env): runs the sequence
                                   #   StopMove -> Move(back) -> StopMove -> FrontJump -> StopMove
```

---

## How it works

```
/lowstate (unitree_go/msg/LowState, ~500 Hz)
    │  lowstate_adapter: motor q/dq/tau + foot_force + IMU -> 48 raw channels
    ▼
EntanglementEngine
    ├─ 200-sample ring buffer (0.40 s)
    ├─ build 60-channel window (48 raw + 12 engineered) + z-score normalize
    ├─ ONNX TCN forward  -> bin / per-leg / intensity logits
    ├─ temperature-scaled confidence, physics-gated intensity
    └─ raw-threshold + 75 ms debounce, per-leg thresholds (RR=0.9)
    ▼
/entanglement_state (entanglement_interfaces/msg/EntanglementState)
```

The engine is causal (decision uses only current + past samples), matching how the model
was trained and evaluated.

---

## `/entanglement_state` message

```
std_msgs/Header header        # header.stamp = timestamp
bool    entangled             # debounced alarm
float32 confidence            # calibrated P(entangled)
float32 fr_probability        # per-leg probabilities
float32 fl_probability
float32 rr_probability
float32 rl_probability
float32 fr_intensity          # per-leg 0..1 severity
float32 fl_intensity
float32 rr_intensity
float32 rl_intensity
string  alarm_leg             # e.g. "RL" while entangled, "" otherwise
```

---

## Quick start (on the GO2)

See **[RUNBOOK.md](RUNBOOK.md)** for the full hardware procedure (detector + recovery). In short:

```bash
# 1. python runtime deps (into the ROS 2 interpreter)
pip3 install -r requirements_robot.txt

# 2. build (from a colcon workspace whose src/ contains the three packages)
cd ~/ros2_ws
colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery
source install/setup.bash

# 3a. detector only (no motion)
ros2 launch entanglement_detector entanglement.launch.py
ros2 topic echo /entanglement_state

# 3b. or the full pipeline (detector + recovery) — on an alarm the robot runs the sequence
#     stop -> move back -> stop -> front jump -> stop (once). Clear space + e-stop in hand.
ros2 launch entanglement_recovery detector_and_recovery.launch.py network_interface:=eth0
ros2 service call /recovery/reset std_srvs/srv/Trigger   # re-arm after a recovery
```

---

## Configuration

### Detector (`entanglement_detector/config/config.yaml`)

| key | meaning | default |
|---|---|---|
| `model_path` | ONNX (`.onnx`) or TorchScript (`_ts.pt`) model | `models/entanglement_tcn.onnx` |
| `normalize_path` | channel z-score stats | `models/normalize.json` |
| `intensity_calib_path` | physics-intensity calibration | `models/intensity_calib.json` |
| `calibration_path` | temperature for calibrated confidence | `models/calibration.json` |
| `detection_threshold` | raw-probability alarm threshold | `0.9999` |
| `debounce_ms` | sustained detection before latching | `75` |
| `leg_thresholds` | per-leg prob to name the alarm leg | `FR/FL/RL=0.5, RR=0.9` |
| `intensity_blend` | `0.5*model_head + 0.5*physics` | `0.5` |
| `stationary_dq_thresh` | mean \|joint dq\| (rad/s) below which the robot is "stationary" — entering a sustained Stop/Lock resets the ring buffer + debounce once (0.0 disables the gate → pre-v2 behavior) | `0.30` |
| `stationary_min_ms` | sustained stationary time before that reset | `100` |
| `stabilize_ms` | suppress alarms this long after motion resumes (gait re-acquisition) | `300` |
| `lowstate_topic` / `output_topic` | ROS topics | `/lowstate`, `/entanglement_state` |
| `num_threads` / `publish_every_n` | CPU threads / output decimation | `1` / `1` |
| `log_every_n` | throttle info logging (0 = silent) | `250` |

Relative paths resolve against the installed package share directory.

### Recovery (`entanglement_recovery/config/recovery.yaml`)

| key | meaning | default |
|---|---|---|
| `network_interface` | interface for the SDK2 DDS (Go2 internal ethernet) | `eth0` |
| `entanglement_topic` | detector output topic | `/entanglement_state` |
| `min_intensity` | ignore alarms below this max per-leg intensity (0 = any) | `0.0` |
| `confirm_count` | consecutive alarm messages before acting (1 = first alarm, no latency) | `1` |
| `back_speed` | backward speed (m/s) for the "move back" step | `0.3` |
| `back_duration` | duration (s) of the "move back" step | `0.5` |

The `back_speed` / `back_duration` values and the front jump itself are design assumptions to
validate on hardware. Full design + safety: **[`../docs/recovery/RECOVERY.md`](../docs/recovery/RECOVERY.md)**.

---

## Performance (measured on the dev CPU, single thread)

| backend | match vs `evaluate.py` | mean | p95 | per-sample budget @ 500 Hz |
|---|---|---|---|---|
| **ONNX Runtime** | 2.3e-6 | **1.0 ms** | **1.2 ms** | 2.0 ms → real-time ✅ |
| TorchScript | 4.9e-7 | 1.9 ms | 2.1 ms | 2.0 ms → borderline |

ONNX is the recommended backend (faster, no full PyTorch). If a target CPU is slower than
the 2 ms/sample budget, set `publish_every_n` > 1 to decimate decisions without changing the
window. Memory footprint is small (~0.5 MB model + numpy buffers).

Re-run the check yourself (dev machine, research repo present):
```bash
python robot_package/tools/validate_runtime.py --backend onnx
```

---

## Re-exporting the model (dev side)

If the model is retrained, regenerate the deployable files from the research repo root:
```bash
python robot_package/export_model.py
```
This writes `entanglement_tcn.onnx`, `entanglement_tcn_ts.pt`, and copies
`normalize.json` / `intensity_calib.json` / `calibration.json` into the package `models/`,
and verifies the exports reproduce the eager model.

---

## Notes / constraints

- **CPU-only**: no GPU required; ONNX Runtime CPU provider is used.
- **Python 3.8 / ROS 2**: runtime code is 3.8-compatible (no 3.9+ syntax).
- **Unitree messages / SDK**: the detector needs `unitree_go/msg/LowState` (reads
  `motor_state[].q/dq/tau_est`, `foot_force[]`, `imu_state.{rpy,gyroscope,accelerometer}`). Recovery
  needs the **Unitree SDK2** (`unitree_sdk2py`) installed into the ROS 2 interpreter; it drives
  `SportClient` directly (not the ROS `/api/sport/request` topic) from a subprocess on `eth0`.
- **Detector is observe-only**: it only publishes a detection topic; it sends no motor commands.
- **Recovery commands motion on an alarm**: it runs the sequence stop → back → stop → front jump →
  stop **once per latch**. The sequence ends in a front jump — only run the recovery node with clear
  space behind/ahead, the robot standing, adequate battery, and an e-stop in hand. The motions are
  unproven design assumptions until hardware-validated.

License: MIT (see parent repository).
