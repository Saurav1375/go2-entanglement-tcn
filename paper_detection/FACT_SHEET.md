# Verified Fact Sheet — Entanglement Detection Paper (source of truth)

All values verified against code/config/artifacts on 2026-07-01. Where `docs/detection/summary.txt`
disagrees with the implementation, the implementation value is used and the disagreement is
noted. **Do not state anything outside this sheet as a measured fact.**

## Platform & problem
- Unitree Go2 quadruped; proprioception streamed on ROS 2 topic `/lowstate`.
- Task: at every timestep predict (i) whether a leg is entangled, (ii) which leg (FR/FL/RR/RL),
  (iii) per-leg severity in [0,1]. Formulated as time-series anomaly/event detection.

## Dataset (verified)
- 25 recordings; 159,001 labeled rows (labeled set `csv_labelled/`).
- **51 columns** in the labeled set (raw `csv/` has 50; the pipeline uses `csv_labelled/`).
- Per-file effective sample rate **~410–502 Hz, median ~483 Hz** (non-uniform logging). [CORRECTION: summary.txt said "409–960 Hz"; 960 is unsupported by the data. Use ~410–500 Hz.]
- Signals per row: 36 motor = 4 legs × {hip,thigh,calf} × {q (position), dq (velocity), **tau (torque)**}; 4 foot-contact forces; IMU = roll,pitch,yaw,gyro(x,y,z),acc(x,y,z); plus a Status label. [CORRECTION: summary said motor triple is "q,dq,ddq"; code uses **tau**, not ddq.]
- Phase labels: Walking, Entangled, Stop, Lock, or blank. Counts: Walking 67,044; blank 50,889; Entangled 19,254; Lock 12,224; Stop 9,590. "Lock" = rigid stance held right after standing up.
- 15 positive recordings (one contiguous entangled event each) / 10 negative. Affected leg encoded in filename.
- Per-affected-leg positive appearances: FR=3, FL=4, RR=7, RL=5 (FR under-represented).
- Labeling: manual, from recorded timestamps + per-joint torque plots.
- Split (leakage-safe, grouped by whole recording): train 16 / val 4 / test 5.
  - test: back_left_hand2, back_right_wire1, front_left_hand2, front_both_wire2, go2_lowstate_lock_stop (4 pos + 1 neg).

## Feature engineering (verified)
- Resample every recording to common **500 Hz** (2 ms step): linear interpolation for continuous signals, nearest-neighbor for the Status label.
- Model input = 60 channels: 48 raw (36 motor + 4 foot + **8 IMU**, yaw dropped as unbounded integrator) + 12 engineered (3 per leg).
- Engineered (per leg), ε = 0.05:
  - `thigh_tau_down = max(0, −tau_thigh)` (resistive/downward thigh torque only).
  - `thigh_down_effort = thigh_tau_down / (|dq_thigh| + ε)` (high torque, little motion = snag signature).
  - `tau_sum = Σ_j |tau_j|` over j∈{hip,thigh,calf} (total per-leg effort; aids attribution).
- Z-score normalization; statistics fit on **training rows only** (no leakage).
- Windowing: **200 samples = 0.40 s** (≈ one trot stride). Hop 25 (train) / 1 (dense eval).
- Window label: positive if ≥50% of rows are Entangled; ambiguous fraction (0.2,0.5) excluded from **training**, kept for evaluation. Stop/Lock windows are negatives.

## Model (verified)
- Causal dilated TCN. Causal = strictly left-padded → prediction uses past+present only.
- Causal conv stem 60→64; 5 residual TCN blocks, dilations 1,2,4,8,16; kernel 3; GELU; dropout 0.1; weight-norm; residual (identity) connection per block.
- Residual block: CausalConv → GELU → CausalConv → GELU → Dropout → + input.
- Receptive field = **127 samples = 254 ms** at 500 Hz (< 0.40 s window).
- Read out the **last-timestep** 64-D embedding → 3 heads: detection (1 logit), leg (4 logits), intensity (4).
- Parameters: **136,393 (~136k)**.

## Training (verified)
- Loss: L = 1.0·BCE(detection) + 1.0·BCE_multilabel(leg) + 0.3·Huber(intensity).
  - Huber: reduction none, δ=0.1, on sigmoid(intensity_logit) vs a physics pseudo-target; affected-leg elements up-weighted **3:1** (weight 3.0 vs 1.0). [CORRECTION: not a hard "mask"; it is a 3:1 affected-leg up-weight so "0 on normal" does not dominate.]
  - Intensity head is a weak auxiliary that tracks the physics pseudo-target; the physics formula is authoritative at inference.
- Class imbalance: class-balanced weighted sampler + per-task BCE pos_weight (capped).
- Optimizer AdamW, lr 1e-3, weight decay 1e-4, cosine schedule, batch 256, 30 epochs, seed 1234.

## Severity / intensity (verified, inference)
- Per-leg physics index: I_leg = g · sqrt(d · r), calibrated to [0,1], where
  - g = per-leg entanglement probability (confidence gate; ≈0 when the leg is not flagged),
  - d = Mahalanobis deviation of the window's per-leg feature vector from the Walking distribution (mean/cov fit on training Walking only),
  - r = resistive effort = mean( max(0,−tau_thigh) / (|dq_thigh| + ε) ).
- Deployed intensity = 0.5·(network head) + 0.5·(physics index). No ground-truth severity labels exist → reported as a calibrated severity index, not a measured force.

## Runtime (verified)
- ROS 2 node `entanglement_detector`; ONNX Runtime, CPU; Python 3.8.
- Subscribes `/lowstate` (BEST_EFFORT QoS, to minimize latency); adapts to 48 raw signals.
- 200-sample circular ring buffer; build 60-ch matrix; z-score (saved train stats); ONNX forward; sigmoid; temperature-scaled confidence (T=5.918); intensity blend; raw-probability threshold + 75 ms debounce; per-leg thresholds; publish `/entanglement_state` (flag, confidence, 4 per-leg prob, 4 per-leg intensity, alarm_leg).
- Operating point: raw threshold 0.9999; calibrated threshold 0.963; debounce 75 ms; per-leg thr FR/FL/RL = 0.5, RR = 0.9 (RR raised to curb false positives).
- Stationarity gate: if mean |joint dq| < 0.30 rad/s for 100 ms → clear ring buffer + reset debounce + arm a 300 ms post-resume suppression (edge-triggered). Disabled in research so the evaluation pipeline is byte-identical to the runtime; enabled in deployment.

## Results — verified numbers
- Held-out test split (raw threshold 0.9999): P 0.799, R 0.818, **F1 0.809**, PR-AUC 0.808, **ROC-AUC 0.956**, exact-match (multi-label) 0.908; confusion TP 2833 / FP 712 / FN 630 / TN 9126.
- Per-leg F1 (test): FR 0.636 (P 0.467, R 1.000), FL 0.845 (P 0.731, R 1.000), RR 0.700 (P 0.551, R 0.959), RL 0.823 (P 0.748, R 0.916).
- Rule-based baseline (thigh-torque heuristic, identical protocol): P 0.735, R 0.131, F1 0.222, PR-AUC 0.639, ROC-AUC 0.830, exact-match 0.106.
- **LORO cross-validation: 15 folds, F1 = 0.807 ± 0.142.**
- Feature ablation (LORO F1, mean±std): engineered-only(12) 0.699±0.223; raw-only(48) 0.799±0.171; no-foot(56) 0.819±0.150; no-IMU(52) 0.833±0.146; raw+engineered(60, shipped) 0.832±0.166.
- Calibration (temperature scaling, T=5.918): raw probabilities are saturated (~27% pinned near 1.0), so the FAR threshold clamps at 0.9999; scaling de-saturates and lets the operating threshold move to 0.963. Brier 0.128→0.118 (slight improvement); ECE 0.134→0.156 (**worsened**). Report honestly: scaling de-saturates + slightly improves Brier but does NOT improve ECE.
- Deployment robustness (targeted data augmentation; offline per-phase firing % v1→v2): Lock false-fire lock_stop 0.8→0.0 and walk_stop_back-Lock 4.7→0.0; rear-right firing while stopped (back_right_intensity-Lock) 100→0; backward-walking (walk_stop_back-Walking) 9.7→0; front-right event (front_right_wire): binary fire 30.7→78.5, correct-leg fire 4→100.
- Runtime latency (measured, ONNX runtime, single CPU thread, dev machine): mean ~1.0 ms/window, median ~1.0 ms, p95 ~1.3 ms, well under the 2.0 ms budget at 500 Hz. Model ~0.5 MB. (Deployment-runtime benchmark, NOT measured on the Go2's onboard computer.)
- Runtime==research numerical equivalence: max |Δ| ≤ 3.8e-7 (detection), 6.3e-7 (per-leg), 2.1e-7 (intensity) over the test windows.

## GAPS — must NOT claim
- No GBM baseline numbers and no statistical-detector numbers persisted (source exists; not evaluated under this protocol). Do not report figures for them.
- No training/validation loss curves logged → no convergence-curve figure.
- All detection metrics are OFFLINE (recorded-CSV replay). No live on-robot detection metrics. Do not claim measured on-robot detection performance.
- Latency is a dev-CPU benchmark of the deployment runtime, not on-robot; state as such.
- Calibration: do NOT claim reliability improved (ECE worsened); frame as de-saturation + Brier.
