# Leg-Entanglement Detection for a Unitree GO2 Quadruped

Real-time detection of **leg entanglement** (a leg caught in a wire, net, or grabbed by hand)
on a Unitree GO2 quadruped, from proprioceptive `/lowstate` data only. A multi-task
**causal Temporal Convolutional Network (TCN)** predicts, at every timestep:

1. **Whether** a leg is entangled (binary detection),
2. **Which** leg (per-leg attribution: FR / FL / RR / RL),
3. **How severe** the entanglement is (a physics-grounded 0–1 intensity per leg).

The model is causal and runs on a sliding window, so the same code path used offline can be
wrapped in a live ROS 2 `/lowstate` node with no change to the model logic.

---

## Table of Contents
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Future Work](#future-work)
- [License](#license)

---

## Motivation

A walking quadruped that snags a leg on a wire or net can stumble or fall. Detecting the snag
**early, per-leg, and with a severity estimate** lets a controller react (slow down, lift the
leg, trigger recovery). Cameras are unreliable for thin wires and self-occlusion, so this project
uses only **proprioception** — joint torques/velocities/positions, foot forces, and IMU — which
the robot already streams at ~500 Hz on `/lowstate`.

A prior rule-based detector (thigh-torque thresholds) worked on earlier, higher-amplitude data
but is brittle on subtler entanglements and cannot estimate severity. This project replaces it
with a learned, multi-task temporal model that is rigorously evaluated (leave-one-recording-out
cross-validation) and ships a calibrated, debounced real-time operating point.

---

## Dataset

20 recordings collected from a Unitree GO2 via ROS 2 `/lowstate`, normalized so each file's
timestamp starts at 0. **~129,726 rows**, variable sample rate **409–960 Hz** (median ~536 Hz),
no missing values.

**Columns (51):** `timestamp`, then per leg `{LEG}_{hip|thigh|calf}_{q|dq|tau}` for
`LEG ∈ {FR, FL, RR, RL}` (36 motor channels), four foot forces `foot_{FL,FR,RL,RR}`,
IMU `roll, pitch, yaw, gyro_{x,y,z}, acc_{x,y,z}`, and a `Status` label
(`Walking`, `Entangled`, `Stop`, or blank).

**Which leg is entangled is encoded in the filename**, not the labels:

| filename pattern | affected leg(s) | | filename pattern | affected leg(s) |
|---|---|---|---|---|
| `back_both_wire*`  | RR, RL | | `front_both_wire*` | FR, FL |
| `back_left_*`      | RL     | | `front_left_*`     | FL     |
| `back_right_*`     | RR     | | `walking*`, `stop*`| none   |

There are **12 positive recordings** (one contiguous entanglement event each) and 8
negatives (`walking1–5`, `stop1–3`). Entanglement type (`wire` vs `hand`) is in the name but
not used as a target.

Directory layout for the data:
- `csv/` — raw recordings **(committed)**
- `label/` — per-recording `start_time,end_time,Status` annotations **(committed)**
- `csv_normalized/`, `csv_labelled/` — **derived** (git-ignored; regenerate with the prep scripts)

---

## Model Architecture

A **causal dilated TCN** with a shared encoder and three task heads.

```
input  [B, 60, 200]          # 60 channels x 200 samples (0.40 s @ 500 Hz)
  │
  ├─ CausalConv1d stem (60→64, k=3)               # left-padded only (no future leakage)
  ├─ 5 × residual TCN block (64 ch, k=3,
  │     dilations 1,2,4,8,16, GELU, dropout 0.1)  # receptive field = 127 samples (~254 ms)
  │
  └─ embedding = encoder output at the LAST timestep  [B, 64]   # the online decision feature
        ├─ head_bin       Linear(64→1)   → P(entangled)
        ├─ head_legs      Linear(64→4)   → P(entangled | leg) for FR,FL,RR,RL
        └─ head_intensity Linear(64→4)   → per-leg severity (auxiliary)
```

- **Channels (60):** 48 raw (36 motor + 4 foot + roll, pitch, gyro xyz, acc xyz; `yaw` dropped)
  plus 12 engineered physics channels per leg (`thigh_tau_down`, `thigh_down_effort`, `tau_sum`).
- **Causal:** because only the **last** timestep's embedding is used and all convolutions are
  left-padded, the prediction depends only on the current and past samples — directly usable in a
  live stream.
- **Loss:** `1.0·BCE(detection) + 1.0·BCE(per-leg) + 0.3·HuberMasked(intensity)`, with
  class-balanced sampling and per-task positive weighting.
- **Intensity (no ground truth):** a physics-grounded, calibrated 0–1 score per leg =
  `confidence_gate × √(manifold_deviation × resistive_effort)`, where deviation is the Mahalanobis
  distance of the window's per-leg features to the *Walking* distribution and resistive effort is
  `mean(max(0,−thigh_tau)/(|thigh_dq|+ε))`. It is ~0 on normal walking and rises monotonically with
  entanglement; reported as a *calibrated severity index*, not a measured force.

~136k parameters; sub-millisecond GPU inference per window.

---

## Pipeline

```
csv/ ──normalize_timestamps.py──▶ csv_normalized/ ──merge_labels.py──▶ csv_labelled/
                                                                            │
                              ┌─────────────────────────────────────────────┘
                              ▼
   resample → 500 Hz  →  channel features (raw + engineered)  →  sliding windows (0.40 s)
                              │
                              ▼
            multi-task TCN  →  detection / per-leg / intensity  →  calibrate + debounce
```

Key design choices: resample every file to a common **500 Hz** (1 step = 2 ms) so temporal
filters mean the same thing across recordings; windows of **200 samples (0.40 s)**, hop 25 for
training / 1 for dense evaluation; **leakage-safe splits grouped by recording** (no window spans
train and test).

---

## Installation

```bash
git clone <your-fork-url> leg-entanglement-detection
cd leg-entanglement-detection
python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.0+ are required. A CUDA GPU is optional (used automatically if present;
CPU works for inference and small training runs).

---

## Usage

All `ml/` modules are run as a package **from the project root**:

```bash
# 1. Prepare data (regenerates the git-ignored derived directories)
python normalize_timestamps.py          # csv/            -> csv_normalized/
python merge_labels.py                  # csv_normalized/ + label/ -> csv_labelled/

# 2. Train (fixed split → writes ml/artifacts/model.pt + normalize/intensity calib + split.json)
python -m ml.train
python -m ml.train --loro               # leave-one-recording-out CV (headline metric)

# 3. Evaluate (metrics + per-recording replay plots in ml/artifacts/plots/)
python -m ml.evaluate
python -m ml.report                     # full verification report -> ml/artifacts/REPORT.md

# 4. Reliability analysis (calibration, debounce, ablation, detector comparison)
python -m ml.improvements               # -> ml/artifacts/IMPROVEMENTS.md + operating_point.json

# 5. Streaming inference (reproduces offline outputs; live /lowstate contract)
python -m ml.infer
```

Every module under `ml/` is independently runnable as `python -m ml.<module>` and prints a
self-test, e.g. `python -m ml.model`, `python -m ml.windowing`, `python -m ml.calibration`.

Optional exploratory plots of thigh torque per recording:
```bash
python plot_thigh_torque.py             # -> plots/
```

### Live inference (sketch)

`ml/infer.py` exposes `InferenceEngine`, a ring-buffered streaming detector whose input is one
`/lowstate`-shaped sample dict per call (CSV column names). A future ROS 2 node only has to map
`unitree_go/msg/LowState` → that dict; no model logic lives in the node.

```python
from ml.infer import InferenceEngine
engine = InferenceEngine.from_artifacts(device="cpu")
for sample in lowstate_stream:           # dict: {"FR_hip_q": ..., "foot_FL": ..., "roll": ...}
    out = engine.push(sample)            # None until the 0.40 s buffer fills, then:
    if out and out["alarm"]:
        print(out["alarm_leg"], out["intensity"])
```

---

## Training

- **Fixed split (leakage-safe, by recording):** 13 train / 3 val / 4 test, covering every leg and
  both entanglement types in each split.
- **LORO-CV** over the 12 positive events is the headline generalization metric (reports mean ± std).
- **Imbalance** handled by a class-balanced `WeightedRandomSampler` (~50/50 per batch, rare legs
  up-weighted) plus per-task BCE positive weights.
- Defaults (in `ml/config.py`): 30 epochs, AdamW lr 1e-3, cosine schedule, batch 256, seed 1234.

Artifacts written to `ml/artifacts/`: `model.pt`, `normalize.json`, `intensity_calib.json`,
`split.json` (all git-ignored — regenerate by training).

---

## Evaluation

One shared protocol is used everywhere so results stay comparable:

- **Threshold** chosen on validation negatives for a 1% per-window false-alarm rate.
- **Detection:** Precision/Recall/F1, PR-AUC, ROC-AUC, confusion matrix, alarm latency, false-alarm rate.
- **Leg attribution:** per-leg P/R/F1, confusion, **exact-match** (set equality for `_both` files).
- **Intensity (no GT):** qualitative checks + Spearman correlation with a physical proxy.
- **Reliability:** probability calibration (Brier, ECE), time-based debounce sweep, feature ablation,
  and a head-to-head comparison against the rule-based baselines.

See [`docs/REPORT.md`](docs/REPORT.md) (baseline verification) and
[`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md) (reliability follow-ups) for the full writeups.

---

## Results

**Detection (fixed split, pooled test):**

| Metric | TCN | Statistical detector¹ | Heuristic (thigh-z)¹ |
|---|---|---|---|
| Precision | 0.83 | 0.86 | 0.68 |
| Recall | 0.87 | 0.22 | 0.08 |
| **F1** | **0.851** | 0.349 | 0.150 |
| PR-AUC | **0.821** | 0.616 | 0.639 |
| ROC-AUC | **0.938** | 0.671 | 0.735 |
| Leg exact-match | **0.858** | 0.069 | 0.079 |

¹ Evaluated under the identical protocol. The rule-based detectors were tuned for higher-amplitude
data; this dataset's entanglement is subtler (rear thigh dips only ~1–2 Nm), which limits them.

**Generalization (LORO-CV, 12 folds):** detection **F1 = 0.832 ± 0.166** — consistent with the
fixed-split number, confirming the split is representative.

**Per-leg F1 (test):** FR 0.87 · FL 0.84 · RR 0.71 · RL 0.83.

**Intensity:** median ≈ 0.00 on walking/stop windows; affected leg ≫ non-affected during
entanglement; Spearman(intensity, thigh-torque proxy) ≈ 0.79.

**Reliability improvements (post-hoc; do not change weights or LORO):**
- Temperature scaling (T ≈ 6.7) improves Brier 0.110 → 0.099 and de-saturates probabilities
  (29% → 0% pinned at 1.0), making the operating threshold meaningful.
- A 50–75 ms time-based debounce removes clean-walking false alarms (→ 0%) at ~0.01 F1 cost.
- A per-leg threshold (RR ≈ 0.9) raises RR precision with ≤0.05 recall loss.

**Feature ablation (LORO F1):** raw+engineered **0.832** ≈ no_imu 0.833 > no_foot 0.819 >
raw_only 0.799 ≫ engineered_only 0.699 — raw kinematics are essential, engineered channels help,
and IMU is near-redundant for detection.

---

## Repository Structure

```
.
├── README.md  LICENSE  requirements.txt  .gitignore
├── normalize_timestamps.py     # csv/ -> csv_normalized/ (timestamps rebased to 0)
├── merge_labels.py             # csv_normalized/ + label/ -> csv_labelled/ (adds Status)
├── plot_thigh_torque.py        # optional: per-recording thigh-torque plots
├── csv/                        # raw GO2 /lowstate recordings  (committed)
├── label/                      # per-recording time-range labels (committed)
├── docs/
│   ├── REPORT.md               # baseline evaluation / verification report
│   ├── IMPROVEMENTS.md         # reliability improvements report
│   └── PLAN.md                 # design plan
├── statistical_detector/
│   └── statistical_detection.py# reference rule-based detector (rear-thigh torque)
└── ml/                         # ML pipeline — run as `python -m ml.<module>`
    ├── config.py               # constants, channel sets, splits, paths (single source of truth)
    ├── io_load.py  resample.py  features.py  windowing.py  normalize.py  dataset.py
    ├── model.py  losses.py  intensity.py
    ├── train.py  evaluate.py  baselines.py  report.py
    ├── diagnose.py  debounce.py  ablation.py  calibration.py  improvements.py
    ├── infer.py                # streaming InferenceEngine (live /lowstate contract)
    └── artifacts/              # generated model + metrics + plots (git-ignored)
```

Derived data (`csv_normalized/`, `csv_labelled/`, `plots/`) and generated artifacts
(`ml/artifacts/`) are git-ignored; regenerate them with the prep scripts and training commands.

---

## Future Work

- Wrap `InferenceEngine` in a ROS 2 `/lowstate` node for on-robot deployment.
- Collect more rear-leg "hand" recordings across gait regimes (the one out-of-distribution
  recording drives most of the LORO variance).
- Add explicit severity labels to enable supervised intensity regression.
- Reduce pre-onset false alarms on the walking-to-entanglement transition.
- Explore a leaner channel set (dropping IMU is LORO-neutral) for tighter on-robot latency.

---

## License

Released under the [MIT License](LICENSE).

---

*Data collected from a Unitree GO2. This repository contains a research prototype; it observes and
predicts only — it does not send motor commands to the robot.*
