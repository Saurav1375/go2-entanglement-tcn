# Leg-Entanglement Detection & Recovery for the Unitree GO2

Real-time detection of **leg entanglement** (a leg caught in a wire/net or grabbed by hand) on a
Unitree GO2 quadruped — from proprioceptive `/lowstate` data only — and an **intelligent recovery**
system that tries to free the snagged leg using verified Unitree Sport-API motions.

The repository ships two cooperating halves:

| | what | where | runtime |
|---|---|---|---|
| **1. Detector** | a multi-task **causal TCN** that predicts, at every timestep, *whether* a leg is entangled, *which* leg (FR/FL/RR/RL), and *how severe* (0–1 intensity) | research in [`ml/`](ml/), deploy in [`robot_package/`](robot_package/) | PyTorch (research) / numpy + ONNX (robot) |
| **2. Recovery** | a safety-gated **recovery FSM + pluggable strategy manager** that consumes the detector output and commands recovery motions in a closed loop | [`robot_package/src/entanglement_recovery/`](robot_package/src/entanglement_recovery/) | pure-Python FSM + thin ROS 2 adapters |

The detector is causal and runs on a sliding window, so the exact code path validated offline is
the one wrapped in the live ROS 2 node — no model logic changes between research and robot.

> **New here?** Jump to [Quick start](#quick-start) · [Repository structure](#repository-structure) ·
> [Documentation index](#documentation-index) · [Deploy on the GO2](robot_package/RUNBOOK.md).

---

## Table of Contents
- [System architecture](#system-architecture)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [The detector](#the-detector)
  - [Dataset](#dataset)
  - [Model architecture](#model-architecture)
  - [Pipeline](#pipeline)
- [The recovery system](#the-recovery-system)
- [Installation & build](#installation--build)
- [Usage](#usage)
- [Testing](#testing)
- [Results](#results)
- [Documentation index](#documentation-index)
- [Limitations & hardware-validation status](#limitations--hardware-validation-status)
- [Future work](#future-work)
- [License](#license)

---

## System architecture

End to end, proprioception flows into a detection topic, which drives a recovery FSM that emits
verified Sport-API motion commands — every motion gated behind a dry-run flag by default.

```
                       Unitree GO2 (ROS 2 / Unitree stack)
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │  /lowstate (unitree_go/LowState, ~500 Hz)   /sportmodestate    /api/sport/*   │
   └───────┬──────────────────────────────────────────▲───────────────────▲───────┘
           │ motor q/dq/tau + foot force + IMU         │ posture / SOC      │ Request
           ▼                                           │                    │
   ┌──────────────────────┐                            │                    │
   │ entanglement_detector │  200-sample ring buffer (0.40 s)               │
   │  ────────────────────│  → 60-ch window → ONNX TCN                      │
   │ causal multi-task TCN │  → calibrated confidence + per-leg + intensity │
   └───────┬──────────────┘  → threshold + debounce + stationarity gate     │
           │ /entanglement_state                                            │
           │ (entangled, confidence, per-leg prob+intensity, alarm_leg)     │
           ▼                                                                │
   ┌──────────────────────┐     ┌───────────────────────┐                  │
   │ entanglement_recovery │────▶│   StrategyManager      │  detector-aware │
   │  ───────────────────  │     │  orders strategies by  │  ordering       │
   │  Recovery FSM:        │◀────│  alarm_leg/confidence/ │                 │
   │  IDLE→MONITORING→     │     │  intensity/posture/SOC │                 │
   │  CONFIRMING→STOPPING→ │     └───────────────────────┘                  │
   │  RECOVERING→VERIFYING→│     closed loop: try motion → re-check detector │
   │  RESUMING→COOLDOWN    │     → recovered? next snag : escalate           │
   │  (+ FAULT / ESTOP)    │─────────────────────────────────────────────────┘
   └───────┬──────────────┘     SportClient (dry-run gate: enable_actuation=false default)
           │ /recovery_status
           ▼
       operator / logs
```

A fuller component diagram lives in [`docs/diagrams/`](docs/diagrams/) (see
[`intelligent_recovery_architecture.md`](docs/diagrams/intelligent_recovery_architecture.md) and
[`recovery_flow.md`](docs/diagrams/recovery_flow.md)).

---

## Quick start

There are two independent entry points; pick the one that matches your goal.

**A. Research / retrain / evaluate the detector (dev machine, GPU optional):**
```bash
pip install -r requirements.txt
python normalize_timestamps.py        # csv/ -> csv_normalized/  (rebase timestamps to 0)
python merge_labels.py                # + label/ -> csv_labelled/ (attach Status)
python -m ml.train                    # writes ml/artifacts/{model.pt,normalize.json,...}
python -m ml.evaluate                 # metrics + per-recording replay plots
```

**B. Deploy detector + recovery on the GO2 (robot, CPU-only, Python 3.8):**
```bash
# follow robot_package/RUNBOOK.md for network/DDS + staged dry-run→actuated bring-up
cd ~/ros2_ws
colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery
source install/setup.bash
ros2 launch entanglement_recovery detector_and_recovery.launch.py   # dry-run by default (no motion)
```

The single authoritative hardware procedure is **[`robot_package/RUNBOOK.md`](robot_package/RUNBOOK.md)**
(deploy → Tier A detector-only → Tier B recovery dry-run → Tier C actuated bring-up).

---

## Repository structure

```
.
├── README.md  LICENSE  requirements.txt  .gitignore
│
│   ── research / data prep (run from repo root) ──
├── normalize_timestamps.py        # csv/ -> csv_normalized/ (timestamps rebased to 0)
├── merge_labels.py                # csv_normalized/ + label/ -> csv_labelled/ (adds Status)
├── plot_thigh_torque.py           # optional: per-recording thigh-torque plots
├── csv/                           # 25 raw GO2 /lowstate recordings  (committed)
├── label/                         # per-recording start/end/Status annotations (committed)
├── statistical_detector/          # reference rule-based detector (rear-thigh torque)
├── ml/                            # ML pipeline — run as `python -m ml.<module>`
│   ├── config.py                  # constants, channel sets, split, paths (single source of truth)
│   ├── io_load · resample · features · windowing · normalize · dataset
│   ├── model · losses · intensity
│   ├── train · evaluate · baselines · report
│   ├── diagnose · debounce · ablation · calibration · improvements
│   ├── infer.py                   # streaming InferenceEngine (live /lowstate contract)
│   └── artifacts/                 # generated model + metrics + plots (git-ignored)
│
│   ── robot deployment (colcon workspace src/) ──
├── robot_package/
│   ├── README.md                  # deployment-package overview
│   ├── RUNBOOK.md                 # ★ single hardware runbook (detector + recovery)
│   ├── SETUP_GO2.md               # detector-only setup detail
│   ├── requirements_robot.txt     # runtime deps (numpy, onnxruntime, pyyaml)
│   ├── export_model.py            # DEV: export ONNX/TorchScript from the trained model
│   ├── tools/validate_runtime.py  # DEV: confirm robot runtime == evaluate.py
│   └── src/
│       ├── entanglement_interfaces/   # ament_cmake: EntanglementState.msg
│       ├── entanglement_detector/     # ament_python: TCN node + numpy/ONNX runtime
│       └── entanglement_recovery/     # ament_python: recovery FSM + strategies + ROS adapters
│
└── docs/                          # architecture, reports, design, testing (see doc index)
```

Derived data (`csv_normalized/`, `csv_labelled/`, `plots/`) and generated artifacts
(`ml/artifacts/`) are git-ignored; regenerate them with the prep scripts and training commands.

### Two pipelines, one contract

`ml/` (research) and `robot_package/` (deploy) are an **intentional mirror**, not duplication:
research uses pandas/torch and is the source of truth for training/evaluation; the robot runtime is
numpy-only + ONNX for Python-3.8 / CPU constraints. The two are **equivalence-guarded** — the
deployment constants, preprocessing, normalizer, intensity, and engine reproduce the research
outputs to ~1e-6 (`robot_package/tools/validate_runtime.py`). Do not collapse them.

---

## The detector

### Dataset

**25 recordings** collected from a Unitree GO2 via ROS 2 `/lowstate`, normalized so each file's
timestamp starts at 0. **~159,001 rows**, variable sample rate **409–960 Hz** (median ~536 Hz),
no missing values.

**Columns (51):** `timestamp`, then per leg `{LEG}_{hip|thigh|calf}_{q|dq|tau}` for
`LEG ∈ {FR, FL, RR, RL}` (36 motor channels), four foot forces `foot_{FL,FR,RL,RR}`,
IMU `roll, pitch, yaw, gyro_{x,y,z}, acc_{x,y,z}`, and a `Status` label
(`Walking`, `Entangled`, `Stop`, `Lock`, or blank).

**Which leg is entangled is encoded in the filename**, not the labels:

| filename pattern | affected leg(s) | | filename pattern | affected leg(s) |
|---|---|---|---|---|
| `back_both_*`  | RR, RL | | `front_both_*` | FR, FL |
| `back_left_*`  | RL     | | `front_left_*` | FL     |
| `back_right_*` / `*back_right*` | RR | | `*front_right*` | FR |
| `walking*`, `stop*`, `lock_stop`, `walk_stop_back` | none | | | |

There are **15 positive recordings** (one contiguous entanglement event each) and **10 negatives**
(`walking1–5`, `stop1–3`, plus two GO2 deployment captures `lock_stop` and `walk_stop_back`).

**The "Lock" state** (rigid stand-up / held stance) is a **non-entangled negative** — same class as
Stop, no new output head. It was added in the v2 retrain because the field detector falsely fired
during the rigid post-stand-up hold; training on Lock data removed that (see
[`docs/RETRAIN_V2_REPORT.md`](docs/RETRAIN_V2_REPORT.md)).

Directory layout for the data:
- `csv/` — raw recordings **(committed)**
- `label/` — per-recording `start_time,end_time,Status` annotations **(committed)**
- `csv_normalized/`, `csv_labelled/` — **derived** (git-ignored; regenerate with the prep scripts)

### Model architecture

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
  `mean(max(0,−thigh_tau)/(|thigh_dq|+ε))`. It is ~0 on normal walking/stop and rises with
  entanglement; reported as a *calibrated severity index*, not a measured force.

~136k parameters; sub-millisecond GPU inference, ~1.0 ms/window on CPU via ONNX.

### Pipeline

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

Key design choices: resample every file to a common **500 Hz** (1 step = 2 ms) so temporal filters
mean the same thing across recordings; windows of **200 samples (0.40 s)**, hop 25 for training / 1
for dense evaluation; **leakage-safe splits grouped by recording** (no window spans train and test).

---

## The recovery system

`entanglement_recovery` consumes `/entanglement_state` and runs a **pure-Python FSM** behind thin
ROS 2 adapters (so the safety logic is unit-testable without a robot). The intelligent layer
redesigns only the `RECOVERING` state into a closed loop driven by a pluggable **StrategyManager**.

```
IDLE ─▶ MONITORING ─▶ CONFIRMING ─▶ STOPPING ─▶ RECOVERING ─▶ VERIFYING ─▶ RESUMING ─▶ COOLDOWN
                          │                          ▲   │                     │
                          │              recovered?  │   │ not recovered →     │
                          └──────────────────────────┘   └─ next strategy / escalate
                                       FAULT / ESTOP (Damp) reachable from any state
```

- **Detector-aware ordering** (no hardcoded leg/values): the manager orders strategies from the
  `alarm_leg`, `confidence`, `intensity`, posture, and battery SOC in `RecoveryContext`. Fallen →
  `[recovery_stand (SOC-gated), emergency_stop]`; otherwise `[balance_stand, weight_shift,
  (if confidence ≥ active_confidence_min and intensity < high_intensity_thresh) small_reverse /
  sidestep / rotate, emergency_stop]`.
- **Verified motions only**: every strategy maps to a `MotionPlan` built from Sport-API actions
  whose `api_id`+params are verified against unitree_ros2/unitree_sdk2 (BalanceStand 1002, StopMove
  1003, RecoveryStand 1006, Euler 1007, Move 1008, Damp 1001, …). Their **kinematic** effect is
  verified; their **disentanglement** effect (and the directions/magnitudes) are design assumptions
  to validate on hardware — see [`docs/INTELLIGENT_RECOVERY.md`](docs/INTELLIGENT_RECOVERY.md) §
  "Verified vs Assumed".
- **Safety**: `enable_actuation` defaults **false** (dry-run — logs would-be commands, sends
  nothing); `Damp` is the soft e-stop and the terminal FAULT action; recovery never re-enters while
  active; per-state watchdog → FAULT; cooldown after each cycle.

Design and rationale: [`docs/RECOVERY_DESIGN.md`](docs/RECOVERY_DESIGN.md),
[`docs/INTELLIGENT_RECOVERY.md`](docs/INTELLIGENT_RECOVERY.md). Config:
[`docs/RECOVERY_CONFIG.md`](docs/RECOVERY_CONFIG.md).

---

## Installation & build

**Research (`ml/`):** Python 3.10+ and PyTorch 2.0+. A CUDA GPU is optional (used automatically if
present; CPU works for inference and small training runs).
```bash
git clone <your-fork-url> leg-entanglement
cd leg-entanglement
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

**Robot (`robot_package/`):** Python 3.8, ROS 2 (e.g. Foxy/Humble), CPU-only. The robot runtime
needs only numpy + onnxruntime + PyYAML — no torch/pandas/sklearn. Build all three packages in a
colcon workspace:
```bash
# copy robot_package/src/* into ~/ros2_ws/src/ and robot_package/requirements_robot.txt to ~/ros2_ws/
cd ~/ros2_ws && pip3 install -r requirements_robot.txt
source /opt/ros/<distro>/setup.bash
colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery
source install/setup.bash
```
`entanglement_interfaces` (the `EntanglementState` message) builds first; the single `colcon build`
above resolves ordering automatically. Full procedure incl. network/DDS:
[`robot_package/RUNBOOK.md`](robot_package/RUNBOOK.md).

---

## Usage

All `ml/` modules run as a package **from the project root**:

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
self-test (e.g. `python -m ml.model`, `python -m ml.windowing`, `python -m ml.calibration`).

**On the robot**, launch detector-only or the full pipeline:
```bash
ros2 launch entanglement_detector entanglement.launch.py                 # detector only
ros2 launch entanglement_recovery detector_and_recovery.launch.py        # detector + recovery (dry-run)
ros2 topic echo /entanglement_state      # per-leg prob + intensity + alarm_leg
ros2 topic echo /recovery_status         # FSM state, strategy i/N, last command, actuation flag
```

### Live inference (research sketch)

`ml/infer.py` exposes `InferenceEngine`, a ring-buffered streaming detector whose input is one
`/lowstate`-shaped sample dict per call (CSV column names) — the same contract the robot node maps
`unitree_go/msg/LowState` onto:

```python
from ml.infer import InferenceEngine
engine = InferenceEngine.from_artifacts(device="cpu")
for sample in lowstate_stream:           # dict: {"FR_hip_q": ..., "foot_FL": ..., "roll": ...}
    out = engine.push(sample)            # None until the 0.40 s buffer fills, then:
    if out and out["alarm"]:
        print(out["alarm_leg"], out["intensity"])
```

---

## Testing

| suite | command | count |
|---|---|---|
| Recovery FSM (pure, no ROS) | `PYTHONPATH=robot_package/src/entanglement_recovery python3 robot_package/src/entanglement_recovery/test/test_recovery_fsm.py` | 16 |
| Plan runner (pure) | `PYTHONPATH=robot_package/src/entanglement_recovery python3 robot_package/src/entanglement_recovery/test/test_plan_runner.py` | 3 |
| ML module self-tests | `python -m ml.model` · `python -m ml.windowing` · `python -m ml.calibration` (etc.) | per-module |
| Runtime equivalence (robot == research) | `python robot_package/tools/validate_runtime.py --backend onnx` | ~1e-6 match |

Run the recovery suites before any hardware session (Tier-0 in the runbook).

---

## Results

The **shipped model is v2** (after adding 5 GO2 deployment recordings incl. the Lock state and the
first dedicated front-right entanglement). Full before/after: [`docs/RETRAIN_V2_REPORT.md`](docs/RETRAIN_V2_REPORT.md);
independent re-validation: [`docs/VALIDATION_REPORT_V2.md`](docs/VALIDATION_REPORT_V2.md).

**Detection (fixed split, clean original-4 test protocol, raw thr 0.9999):**

| Metric | TCN v2 | Statistical detector¹ | Heuristic (thigh-z)¹ |
|---|---|---|---|
| Precision | 0.80 | 0.86 | 0.68 |
| Recall | 0.82 | 0.22 | 0.08 |
| **F1** | **0.807** | 0.349 | 0.150 |
| PR-AUC | **0.808** | 0.616 | 0.639 |
| ROC-AUC | **0.929** | 0.671 | 0.735 |
| Leg exact-match | **0.908** | 0.069 | 0.079 |

¹ Evaluated under the identical protocol. The rule-based detectors were tuned for higher-amplitude
data; this dataset's entanglement is subtler, which limits them.

**Generalization (LORO-CV):** 15-fold **F1 = 0.807 ± 0.142**; on the *same 12 folds* as v1 the model
**improved to 0.843** with lower variance — the 15-fold headline is lower only because 3 genuinely
harder new scenarios were added.

**The four GO2 field issues — all fixed by the v2 retrain** (firing %, see RETRAIN_V2_REPORT §2):

| field issue | before | after |
|---|---|---|
| False entanglement on stand-up / Lock | 0.8–4.7% | **0.0%** |
| False RR after entanglement while stopped | 100% | **0.0%** |
| False detection during backward walking | 9.7% | **0.0%** |
| Front-right detection (dedicated FR file) | 4% | **100%** |

**Intensity:** median ≈ 0.00 on walking/stop/Lock windows; affected leg ≫ non-affected during
entanglement; Spearman(intensity, thigh-torque proxy) ≈ 0.79.

**Reliability (post-hoc; do not change weights or LORO):** temperature scaling de-saturates
probabilities and improves Brier 0.110 → 0.091 (operating threshold de-clamps 0.9999 → 0.963);
a 50–75 ms time-based debounce removes clean-walking false alarms; a per-leg RR threshold (≈0.9)
raises RR precision. Feature ablation (v1 study, architecture-level): raw kinematics essential,
engineered channels help, IMU near-redundant for detection.

**Trade-offs (honest):** FR per-leg *precision* regressed on the original-4 protocol after learning
the dedicated FR signature (recall stays 1.0, and it is gated by the binary head in deployment);
one front-both LORO fold weakened. Both are documented for the next data-collection round.

---

## Documentation index

| doc | what it covers |
|---|---|
| **[robot_package/RUNBOOK.md](robot_package/RUNBOOK.md)** | ★ single hardware procedure: deploy → detector-only → recovery dry-run → actuated bring-up |
| [docs/CONSOLIDATION_REPORT.md](docs/CONSOLIDATION_REPORT.md) | verification results, **deployment checklist**, change summary, assumptions needing real-robot validation |
| [robot_package/README.md](robot_package/README.md) | deployment-package overview (node, message, config, performance) |
| [robot_package/SETUP_GO2.md](robot_package/SETUP_GO2.md) | detector-only setup detail + systemd autostart |
| [docs/RETRAIN_V2_REPORT.md](docs/RETRAIN_V2_REPORT.md) | **current** model: v2 retrain, the four field fixes, before/after metrics |
| [docs/VALIDATION_REPORT_V2.md](docs/VALIDATION_REPORT_V2.md) | independent re-validation of v2 (generalization vs memorization) |
| [docs/RECOVERY_DESIGN.md](docs/RECOVERY_DESIGN.md) | recovery FSM design, Sport-API rationale, safety |
| [docs/INTELLIGENT_RECOVERY.md](docs/INTELLIGENT_RECOVERY.md) | strategy manager, closed loop, verified-vs-assumed |
| [docs/RECOVERY_CONFIG.md](docs/RECOVERY_CONFIG.md) | every recovery parameter |
| [docs/RECOVERY_TESTING.md](docs/RECOVERY_TESTING.md) | scenario matrix → tests, test tiers |
| [docs/RECOVERY_DELIVERABLES.md](docs/RECOVERY_DELIVERABLES.md) | recovery package file map + deliverables |
| [docs/diagrams/](docs/diagrams/) | architecture / state-machine / strategy-flow diagrams |
| [docs/REPORT.md](docs/REPORT.md) | *(v1, superseded by RETRAIN_V2_REPORT)* baseline detector verification |
| [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) | *(v1; calibration refreshed for v2, ablation is v1)* reliability study |
| [docs/PLAN.md](docs/PLAN.md) | original ML design plan |
| [docs/PUBLISH_REVIEW.md](docs/PUBLISH_REVIEW.md) | *(pre-recovery)* GitHub-publication review of the ML repo |

---

## Limitations & hardware-validation status

- **Detector — validated offline, not yet on the live robot loop.** Metrics above are
  leakage-safe offline (fixed split + LORO-CV); the v2 runtime equivalence to research is verified
  to ~1e-6, but real-time behavior on the robot's actual CPU/DDS still needs a field session.
- **Recovery — motion *kinematics* verified, *disentanglement efficacy* NOT.** Every Sport-API
  command's effect is verified against the SDK, but whether a given motion actually frees a snagged
  leg — and the chosen directions, magnitudes, and the Euler weight-shift **sign convention** — are
  **design assumptions**. They must be validated on hardware (RUNBOOK Tier C). Recovery ships
  **dry-run by default**.
- **Data scale:** 15 positive events; FR is positive in only 3 files, which drives most LORO
  variance and the FR precision trade-off. More front-right / front-both captures are the top
  data need.
- **Telemetry QoS:** posture/SOC-aware behavior needs `/sportmodestate` over BEST_EFFORT QoS; if DDS
  isn't aligned the FSM degrades to timeouts (see RUNBOOK §3).

---

## Future work

- Field-validate the recovery strategies on the GO2 (efficacy, directions, magnitudes) and feed
  results back into `recovery.yaml` and the strategy ordering policy.
- Collect more front-right / front-both and rear-leg "hand" recordings across gait regimes to
  rebalance FR precision and reduce LORO variance.
- Add explicit severity labels to enable supervised intensity regression.
- Reduce pre-onset false alarms on the walking-to-entanglement transition.
- Explore a leaner channel set (dropping IMU is LORO-neutral) for tighter on-robot latency.

---

## License

Released under the [MIT License](LICENSE).

---

*Data collected from a Unitree GO2. The detector observes and predicts only. The recovery system can
command motion, but ships disabled (`enable_actuation: false`) and must be validated on hardware
before actuated use — see [`robot_package/RUNBOOK.md`](robot_package/RUNBOOK.md).*
