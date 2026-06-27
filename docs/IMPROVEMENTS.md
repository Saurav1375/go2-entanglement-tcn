# Reliability Improvements Report

Protocol unchanged from REPORT.md: TARGET_FAR=1%, leg_thr=0.5, window=0.40 s. All changes here are **post-hoc / inference-time** (calibration, thresholds, debounce, comparison) — they do **not** retrain or alter the TCN weights, so LORO detection ranking (F1=0.83±0.17) is unchanged. Feature ablations (§6) DO retrain and are reported with their own LORO.

## 1. Probability calibration (temperature scaling)

| | Brier | ECE | sat@1.0 | VAL 1%-FAR threshold |
|---|---|---|---|---|
| uncalibrated | 0.1099 | 0.1126 | 29% | 0.9999 (clamped) |
| temp-scaled (T=6.66) | 0.0991 | 0.1197 | 0% | 0.9857 (not clamped) |

- T=6.66≫1 confirms over-confidence. Calibration **improves Brier** and **de-saturates** (was 29% pinned at 1.0 → 0%), so the 1%-FAR threshold is now a real interior value (0.986) instead of the clamped 0.9999 — fixing the §0 WARN from REPORT.md. ROC/PR-AUC are unchanged (monotonic).
- Honest caveat: **ECE does not improve** here (small VAL set fit over-compresses TEST). Calibration's win is de-saturation + Brier, not ECE; Brier is the proper-score that matters for the severity/intensity use.

## 2. back_left_hand1 LORO root-cause (F1=0.45)
- **Not weak signal**: it has the strongest RL contrast (3.8σ vs siblings 0.7σ/1.4σ).
- **Out-of-distribution**: lowest walking baseline (RL_down_walk 0.23 vs 0.71/0.37 Nm), slower gait (walk_dq 1.30 vs 1.67/1.56), and it is the longest recording (7538 rows).
- **Proof**: retraining the held-out fold, the model scores hand1's *entangled* windows at median p_bin≈0.001 — it doesn't recognize the pattern when hand1 is absent from training.
- **Verdict**: small-data coverage gap, not an architecture flaw. Fix = collect more rear-leg hand-entanglement recordings across gait regimes (no model change). (full evidence: diagnose.py)

## 3. RR false-positive reduction (per-leg threshold, no retrain)

| RR leg-thr | P | R | F1 |
|---|---|---|---|
| 0.50 | 0.55 | 1.00 | 0.71 |
| 0.70 | 0.58 | 0.98 | 0.73 |
| 0.90 | 0.62 | 0.96 | 0.76 |
| 0.95 | 0.64 | 0.95 | 0.77 |

- RR over-fires mainly when **RL is entangled** (rear-leg coupling) and on **walking prefixes** (early detection). Raising RR's leg-threshold to ~0.9 lifts precision with ≤0.05 recall loss — an inference-time knob, LORO untouched.

## 4. Time-based debounce (replaces literal 3-window=6 ms)

| debounce | pooled F1 | recall | FAR walking4 | FAR test-prefix | med latency |
|---|---|---|---|---|---|
| 0 ms (k=1) | 0.851 | 0.87 | 2.75% | 9.86% | 0 ms |
| 6 ms (k=3) | 0.850 | 0.86 | 1.38% | 9.73% | 0 ms |
| 25 ms (k=12) | 0.847 | 0.85 | 0.00% | 9.14% | 0 ms |
| 50 ms (k=25) | 0.842 | 0.83 | 0.00% | 8.28% | 0 ms |
| 75 ms (k=38) | 0.838 | 0.81 | 0.00% | 7.42% | 0 ms |
| 100 ms (k=50) | 0.835 | 0.80 | 0.00% | 6.69% | 0 ms |
| 150 ms (k=75) | 0.826 | 0.77 | 0.00% | 5.45% | 5 ms |

- The literal 6 ms rule ≈ raw. A **75 ms** debounce removes clean-walking false alarms (→0%) for ~0.013 F1 and ~0 ms added latency. **Recommend 50–75 ms.**

## 5. Detector comparison — identical protocol (TCN vs heuristic vs Statistical Detector)

| detector | P | R | F1 | PR-AUC | ROC-AUC | leg exact-match | localizes |
|---|---|---|---|---|---|---|---|
| **TCN (multi-task)** | 0.83 | 0.87 | 0.851 | 0.821 | 0.938 | 0.858 | all 4 legs + intensity |
| Statistical Detector | 0.86 | 0.22 | 0.349 | 0.616 | 0.671 | 0.069 | rear legs; front=unlocalized |
| Heuristic (thigh z) | 0.68 | 0.08 | 0.150 | 0.639 | 0.735 | 0.079 | single leg (argmax) |

- The **Statistical Detector** (rear-thigh deviation, calibrated on TRAIN walking, scored under the identical protocol) reaches F1=0.35 / ROC-AUC=0.67. Its absolute-threshold rules (tuned for the older, higher-amplitude data) never trigger on this dataset's subtler entanglement (rear thigh only dips ~1–2 Nm), so it relies on deviation ranking; it cannot localize FL vs FR and gives no intensity.
- The **TCN** leads on every detection metric and is the only one with per-leg attribution + calibrated intensity.

## 6. Feature ablation (retrained; own LORO)

| feature set | C | fixed F1 | PR-AUC | leg macro-F1 | LORO F1 (mean±std) |
|---|---|---|---|---|---|
| raw+engineered ⟵ shipped | 60 | 0.788 | 0.827 | 0.739 | 0.832 ± 0.166 |
| raw_only | 48 | 0.800 | 0.864 | 0.755 | 0.799 ± 0.171 |
| engineered_only | 12 | 0.648 | 0.741 | 0.528 | 0.699 ± 0.223 |
| no_imu | 52 | 0.783 | 0.831 | 0.764 | 0.833 ± 0.146 |
| no_foot | 56 | 0.811 | 0.837 | 0.719 | 0.819 ± 0.150 |

- **Raw kinematics are essential**: engineered_only collapses (LORO 0.699).
- **Engineered physics channels help**: dropping them (raw_only) costs ~+0.033 LORO.
- **Foot sensors help a little** (no_foot 0.819 vs 0.832). **IMU is near-redundant** for detection (no_imu 0.833, ties full with lower variance).
- **Decision**: keep the shipped raw+engineered (C=60) — nothing strictly beats it on LORO. Dropping IMU (C=52) is a valid leaner option with no LORO loss, but the gap is within noise so it is not mandated. (Differences <0.03 sit inside the ±0.15 fold std — single-seed estimates.)

## 7. Verdict — what genuinely helps

| change | effect | adopt? |
|---|---|---|
| Temperature scaling (T=6.7) | Brier 0.110→0.099; de-saturates; real threshold | **Yes** (post-hoc) |
| 50–75 ms debounce | clean-walking FAR →0%, F1 −0.01, latency ~0 | **Yes** (inference) |
| RR per-leg threshold ~0.9 | RR precision up, recall −≤0.05 | **Yes** (inference) |
| Feature set | see §6 — keep the set with best LORO | adopt only if LORO ≥ shipped |

All adopted changes are post-hoc and do not reduce LORO. The recommended operating point is: **temperature-scaled probabilities + 75 ms debounce + per-leg thresholds (RR≈0.9)**.
