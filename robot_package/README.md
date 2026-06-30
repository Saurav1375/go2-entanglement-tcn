# Leg-Entanglement Detector + Recovery — Unitree GO2 Deployment Package

Self-contained ROS 2 workspace `src/` with **three packages** that run the full pipeline **on the
robot** (CPU-only):

- **`entanglement_interfaces`** — the `EntanglementState` message (ament_cmake).
- **`entanglement_detector`** — subscribes to `/lowstate`, runs the trained multi-task TCN every
  sample, publishes `/entanglement_state` (entangled flag, calibrated confidence, per-leg
  probabilities, per-leg intensities). Inference uses **numpy + ONNX Runtime** — no training code,
  dataset, pandas, scikit-learn, scipy, or matplotlib.
- **`entanglement_recovery`** — consumes `/entanglement_state`, runs the **recovery FSM + pluggable
  StrategyManager**, and (when actuation is enabled) commands verified Sport-API recovery motions;
  publishes `/recovery_status`. Pure-Python FSM behind thin ROS adapters; **dry-run by default**.

> **Deploying on hardware? Follow [`RUNBOOK.md`](RUNBOOK.md)** — the single authoritative procedure
> (network/DDS, staged dry-run → actuated bring-up). [`SETUP_GO2.md`](SETUP_GO2.md) covers the
> detector alone. Recovery design/config: parent [`docs/`](../docs) (`RECOVERY_DESIGN.md`,
> `INTELLIGENT_RECOVERY.md`, `RECOVERY_CONFIG.md`).

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
    └── entanglement_recovery/    # ament_python pkg: recovery FSM + strategies + adapters
        ├── config/recovery.yaml  # all recovery params (actuation gate, timings, strategies)
        ├── launch/recovery.launch.py
        ├── launch/detector_and_recovery.launch.py   # launches detector + recovery together
        ├── test/{test_recovery_fsm.py, test_plan_runner.py}   # 16 + 3 pure tests
        └── entanglement_recovery/
            ├── states.py          # enums + Detection/RobotState/RecoveryContext/MotionPlan (pure)
            ├── recovery_fsm.py    # safety-critical FSM (pure); RECOVERING -> StrategyManager
            ├── strategies.py      # 7 strategies + MotionPlan builders (pure)
            ├── strategy_manager.py# detector-aware ordering policy (pure)
            ├── plan_runner.py     # ONESHOT/STREAM/HOLD step execution over ticks (pure)
            ├── sport_api.py       # verified Sport-API ids + topics + mode map (pure)
            ├── sport_client.py    # ROS: Command -> /api/sport/request + dry-run gate
            ├── robot_state.py     # ROS: /sportmodestate(+/lowstate) -> RobotState/Posture
            └── recovery_node.py   # ROS orchestrator -> /recovery_status
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

# 3b. or the full pipeline (detector + recovery; dry-run by default — no motion)
ros2 launch entanglement_recovery detector_and_recovery.launch.py
ros2 topic echo /recovery_status
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

`enable_actuation` (default **`false`** — dry-run), confirmation/cooldown/retry/watchdog timings,
posture/SOC gates (`tip_*_deg`, `min_soc_pct`), and the per-strategy enables/magnitudes
(`active_confidence_min`, `high_intensity_thresh`, `reverse_*`, `sidestep_*`, `rotate_*`,
`weightshift_*`). Every key is documented in **[`../docs/RECOVERY_CONFIG.md`](../docs/RECOVERY_CONFIG.md)**.
The motion directions/magnitudes and the Euler weight-shift **sign convention** are design
assumptions to validate on hardware (RUNBOOK Tier C).

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
- **Unitree messages**: `unitree_go/msg/LowState` (detector) and `unitree_api/msg/Request` +
  `unitree_go/msg/SportModeState` (recovery) must be available on the robot (they are, on the GO2
  ROS 2 stack). The detector adapter reads `motor_state[].q/dq/tau_est`, `foot_force[]`, and
  `imu_state.{rpy,gyroscope,accelerometer}`.
- **Detector is observe-only**: it only publishes a detection topic; it sends no motor commands.
- **Recovery can command motion but ships disabled**: `enable_actuation` defaults `false` (dry-run —
  logs intended commands, sends nothing). Set it true only after the dry-run validation in
  `RUNBOOK.md`, with clearance and an e-stop in hand. Motion directions/magnitudes are unproven
  design assumptions until hardware-validated.

License: MIT (see parent repository).
