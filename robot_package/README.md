# Leg-Entanglement Detector — Unitree GO2 Deployment Package

Self-contained ROS 2 package that runs the trained multi-task TCN **on the robot**:
it subscribes to `/lowstate`, runs CPU-only inference every sample, and publishes
`/entanglement_state` (entangled flag, calibrated confidence, per-leg probabilities,
per-leg intensities, timestamp).

This package contains **only what the GO2 needs** — no training code, dataset, pandas,
scikit-learn, scipy, or matplotlib. Inference uses **numpy + ONNX Runtime**.

> The research/training pipeline lives in the parent repository. The model here was
> exported from it unchanged; performance is identical (validated to ~1e-6, see below).

---

## Contents

```
robot_package/
├── README.md                     # this file
├── SETUP_GO2.md                  # step-by-step robot setup
├── requirements_robot.txt        # runtime python deps (numpy, onnxruntime, pyyaml)
├── export_model.py               # DEV-SIDE: export ONNX/TorchScript from the trained model
├── tools/validate_runtime.py     # DEV-SIDE: confirm runtime == evaluate.py + benchmark
└── src/
    ├── entanglement_interfaces/  # ament_cmake pkg: EntanglementState.msg
    └── entanglement_detector/    # ament_python pkg: the node + runtime
        ├── config/config.yaml    # model paths, threshold, debounce, per-leg thresholds
        ├── launch/entanglement.launch.py
        ├── models/               # entanglement_tcn.onnx + *_ts.pt + *.json (from export)
        └── entanglement_detector/
            ├── constants.py       # channel layout / GO2 ordering
            ├── preprocess.py      # 60-channel window + normalization (numpy)
            ├── intensity.py       # physics intensity (numpy)
            ├── model_backend.py   # ONNX Runtime / TorchScript backend
            ├── engine.py          # ring buffer + inference + debounce + thresholds
            ├── lowstate_adapter.py# unitree_go/LowState -> model input
            └── node.py            # ROS 2 node
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

See **[SETUP_GO2.md](SETUP_GO2.md)** for full detail. In short:

```bash
# 1. python runtime deps (into the ROS 2 interpreter)
pip3 install -r requirements_robot.txt

# 2. build (from a colcon workspace whose src/ contains the two packages)
cd ~/ros2_ws
colcon build --packages-select entanglement_interfaces entanglement_detector
source install/setup.bash

# 3. run
ros2 launch entanglement_detector entanglement.launch.py

# 4. watch output
ros2 topic echo /entanglement_state
```

---

## Configuration (`config/config.yaml`)

| key | meaning | default |
|---|---|---|
| `model_path` | ONNX (`.onnx`) or TorchScript (`_ts.pt`) model | `models/entanglement_tcn.onnx` |
| `normalize_path` | channel z-score stats | `models/normalize.json` |
| `intensity_calib_path` | physics-intensity calibration | `models/intensity_calib.json` |
| `calibration_path` | temperature for calibrated confidence | `models/calibration.json` |
| `detection_threshold` | raw-probability alarm threshold | `0.9999` |
| `debounce_ms` | sustained detection before latching | `75` |
| `leg_thresholds` | per-leg prob to name the alarm leg | `FR/FL/RL=0.5, RR=0.9` |
| `lowstate_topic` / `output_topic` | ROS topics | `/lowstate`, `/entanglement_state` |
| `num_threads` / `publish_every_n` | CPU threads / output decimation | `1` / `1` |

Relative paths resolve against the installed package share directory.

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
- **Unitree messages**: `unitree_go/msg/LowState` must be available on the robot (it is, on
  the GO2 ROS 2 stack). The adapter reads `motor_state[].q/dq/tau_est`, `foot_force[]`,
  and `imu_state.{rpy,gyroscope,accelerometer}`.
- **Observe-only**: this node only publishes a detection topic; it sends no motor commands.

License: MIT (see parent repository).
