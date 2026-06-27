# Plan: Real-Time Leg-Entanglement Detection — Multi-Task Temporal Deep Net

## Context

The user collects proprioceptive data from a Unitree GO2 quadruped (streamed live via ROS2
`/lowstate`) and wants to predict, at any instant, **(1) whether a leg is entangled, (2) which
leg, and (3) an intensity/severity score** — accurately, with advanced ML/DL.

A prior heuristic system already exists at `(external) prior-detector/`
(hand-crafted features + a centroid/threshold prototype, `live_entanglement_detector.py` +
`entanglement_prototype.json`). It has no proper train/test split, no temporal model, and only
pass/fail evaluation. The new `./` is the clean, **labeled** successor
to that project's old data — identical 49-signal format. This plan builds a supervised,
temporal deep-learning replacement that is more accurate and rigorously evaluated, while keeping a
clean path back to the existing live `/lowstate` interface.

**Locked-in decisions (from user):**
1. **Intensity** = physics-grounded, calibrated **0–1 per-leg** score (no GT intensity labels exist).
2. **Model** = multi-task **temporal deep net (causal dilated TCN)**, shared encoder + 3 heads.
3. **Scope** = **offline train + eval pipeline first** (no ROS2 node this round), but inference code
   structured so a live node can wrap it later against the existing input contract.
4. **Location** = self-contained under **`./ml/`**, reading `csv_labelled/`.
   PyTorch 2.11 + CUDA, numpy/pandas/sklearn/scipy already installed.

**Verified data facts:** 20 files, ~129,726 rows, 51 cols (`timestamp` + 36 motor
`{LEG}_{hip|thigh|calf}_{q|dq|tau}` for LEG∈FR,FL,RR,RL + `foot_{FL,FR,RL,RR}` + IMU
`roll,pitch,yaw,gyro_x,y,z,acc_x,y,z` + `Status`). No NaNs. Per-file rate varies **409–960 Hz**
(median ~536). **No `front_right` files exist** (FR is positive only in the 2 `front_both` files).
Each positive file has **exactly one contiguous Entangled segment** → **12 positive events total**.
`Status` ∈ {Walking, Entangled, Stop, blank}; blanks only at file ends. Which leg is entangled
comes from the **filename**, not the labels.

## Module layout — `dataset/ml/`

| file | responsibility |
|---|---|
| `config.py` | Single source of truth: `LEG_ORDER=(FR,FL,RR,RL)`, joint/foot order, `TARGET_HZ=500`, `WINDOW_SAMPLES=200`, hops, channel list, head dims, loss weights, file→split, seed. Mirrors reuse-source ordering constants. |
| `io_load.py` | CSV glob, `parse_legs(filename)->set[str]` (table below), load to DataFrame, validate 51 cols. |
| `resample.py` | Per-file uniform resample to 500 Hz — linear-interp signals, nearest-neighbor `Status`. |
| `features.py` | Raw channel selection + engineered per-leg physics channels (exact reuse formulas). |
| `windowing.py` | Sliding-window tensor builder + per-window label derivation. |
| `normalize.py` | Per-channel z-score; fit on **train rows only**; save/load `normalize.json`. |
| `dataset.py` | torch Dataset/DataLoader: resample→features→window→normalize; weighted sampler; returns `(x[C,T], y_bin, y_legs[4], meta)`. |
| `model.py` | Causal dilated TCN shared encoder + 3 heads. |
| `losses.py` | Multi-task weighted loss + per-task class weights. |
| `intensity.py` | Physics-grounded 0–1 per-leg intensity + calibration fit/save/load. |
| `train.py` | Training loop, grouped fixed split + LORO-CV, checkpoint, export `normalize.json` + `intensity_calib.json`. |
| `evaluate.py` | All metrics, per-recording replay traces/plots, baseline comparison. |
| `infer.py` | `InferenceEngine`: ring buffer → window → model → intensity; `/lowstate`-shaped input dict; **no ROS dependency**. |
| `baselines.py` | sklearn GBM baseline + hook to replay the existing heuristic for comparison. |
| `artifacts/` | `model.pt`, `normalize.json`, `intensity_calib.json`, `split.json`, metrics, `plots/`. |

## Key design specs

**Resampling + windowing.** Resample every file to **500 Hz** (1 step = 2 ms everywhere; matches
live GO2 cadence) — linear interp for signals, nearest-neighbor for `Status` (never interpolate a
label). **Window = 0.40 s = 200 samples** (≈ one trot stride), **hop = 25 (train), 1 (eval/replay)**.
Tensor shape `[C, 200]`. Drop raw `yaw` (unbounded integrator).

**Channels C = 60:** 48 raw (36 motor + 4 foot + roll,pitch + gyro_x,y,z,acc_x,y,z) + 12 engineered
(per-leg `thigh_tau_down=max(0,−thigh_tau)`, `thigh_down_effort=thigh_tau_down/(|thigh_dq|+0.05)`,
`tau_sum=Σ|tau|`). `USE_ENGINEERED=False` → C=48 ablation. Normalization: per-channel z-score over
**train rows only**, persisted to `normalize.json` (std floored 1e-6).

**filename → affected legs:**

| file | legs | | file | legs |
|---|---|---|---|---|
| back_both_wire1/2 | RR, RL | | front_both_wire1/2 | FR, FL |
| back_left_hand1/2, back_left_wire1 | RL | | front_left_hand1/2 | FL |
| back_right_hand1/2, back_right_wire1 | RR | | stop1/2/3, walking1..5 | none |

**Per-window targets** (over resampled Status): keep window only if all 200 rows non-blank;
`y_bin=1` iff Entangled fraction ≥ 0.5; `y_legs[FR,FL,RR,RL]` = file's affected set when `y_bin=1`
else zeros; Stop = negative. Transition band (Entangled fraction 0.2–0.5) excluded from **training**,
kept in **eval**. Onset = first Entangled resampled row (for latency).

**Split (leakage-safe; group = recording, never split windows across files).**
- **TEST (4):** `back_left_hand2`, `back_right_wire1`, `front_left_hand2`, `front_both_wire2`
  (covers RL, RR, FL, FR+FL; wire+hand; 409–960 Hz mix).
- **VAL (3):** `back_both_wire2`, `back_right_hand2`, `walking4`.
- **TRAIN (13):** the remaining positives + `stop1/2/3`, `walking1,2,3,5`.
- **Headline metric = grouped Leave-One-Recording-Out CV** over the 12 positive events (mean±std);
  the fixed split ships the artifact. Every leg has a train positive (FR via `front_both_wire1`).
- **Imbalance:** `WeightedRandomSampler` (~50/50 pos/neg per batch, upweight rare-leg/FR windows) +
  BCE `pos_weight` per task (capped 10). FR recall reported separately.

**Model (causal dilated TCN).** Conv1d stem 60→64 (k=3, causal pad); 5 residual TCN blocks, 64 ch,
dilations 1,2,4,8,16, each `[Conv(k3,d)→WeightNorm→GELU→Dropout0.1→Conv(k3,d)→residual]`.
Receptive field 125 samples ≈ 0.25 s < window. **Left-pad only (causal)**; use **last-timestep**
64-d embedding `h` (= the online decision feature). Heads from `h`: (a) binary `Linear(64→1)→σ`;
(b) per-leg `Linear(64→4)→4σ` `[FR,FL,RR,RL]`; (c) intensity `Linear(64→4)→σ` weakly supervised by
the physics pseudo-target. **Loss** = `1.0·BCE(bin) + 1.0·BCE_multilabel(legs) + 0.3·MaskedHuber(intensity, pseudo)`.

**Physics-grounded intensity (0–1, no GT).** Per leg combine: (i) **Mahalanobis distance** `D_leg` of
the window's mean per-leg feature vector `[thigh_tau_down, thigh_down_effort, tau_sum, foot_force,
|thigh_dq|]` to the **Walking** distribution (mean/cov fit on **train Walking only**; promote to a
Walking-only autoencoder if under-separated); (ii) **resistive effort** `R_leg = mean(max(0,−thigh_tau)/(|thigh_dq|+0.05))`;
(iii) **confidence gate** `g_leg` = temperature-calibrated per-leg prob. Calibrate via train-Walking
percentiles: `d_leg=σ(a·(D−q95)/iqr)`, `r_leg=σ(a·(R−q95)/iqr)`; `m_leg=√(d_leg·r_leg)`;
**`I_leg = g_leg·m_leg`** (≈0 when not detected). Reported `I = 0.5·head + 0.5·physics` (physics-only
is the explainable fallback). Persist all params to `intensity_calib.json`. Validate qualitatively:
median <0.05 on walking/stop, monotonic rise from onset, affected>non-affected, Spearman(I,
`thigh_tau_down_mean`) ≥ 0.6 on entangled windows.

**Evaluation.** Detection: P/R/F1, PR-AUC, ROC-AUC (threshold chosen on VAL for a fixed false-alarm
rate); fixed-split + LORO-CV mean±std. **Alarm latency** (onset→first 3-window-persistent fire),
**false-alarm rate** on walking/stop (headline safety). Leg attribution: per-leg P/R/F1, confusion,
**exact-match** (set equality for `_both`). Intensity: qualitative + Spearman proxy. **Per-recording
replay PNGs** (GT ribbon + `p_bin` + per-leg `p_leg` + `I_leg` + heuristic overlay) in
`data_analysis/plots/`-comparable format. **Baselines to beat:** existing heuristic (via its
CSV-replay path) + sklearn `HistGradientBoostingClassifier` on window-averaged engineered features.

## Reuse (do NOT modify these)
- `(external) prior-detector/live_entanglement_detector.py` — exact feature
  formulas, `LEG_ORDER`, motor idx FR(0-2)…RL(9-11), `CSV_FOOT_ORDER=(FL,FR,RL,RR)`, `EFFORT_DQ_EPSILON=0.05`,
  0.30 s window + 3-window persistence (mirror in eval), and the `/lowstate` input contract `infer.py` targets.
- `entanglement_model.py` — prototype/centroid baseline reference.

## Build order
1. `config.py` + `io_load.py` (validate filename table on 20 files).
2. `resample.py` + `features.py` (verify 500 Hz grid + engineered channels match heuristic formulas).
3. `windowing.py` + `normalize.py` + `dataset.py` (check label fractions + `[B,60,200]`).
4. `model.py` + `losses.py` (overfit a tiny batch to confirm wiring).
5. `train.py` (fixed split, then LORO-CV).
6. `intensity.py` (Mahalanobis + resistive + gate; fit on train Walking).
7. `evaluate.py` + `baselines.py` (beat heuristic + GBM).
8. `infer.py` (ring buffer; confirm `/lowstate`-shaped input contract).

## Verification (end-to-end)
- `python ml/train.py` trains, prints LORO-CV mean±std, writes `model.pt` + `normalize.json` + `intensity_calib.json` + `split.json`.
- `python ml/evaluate.py` prints detection/leg/intensity metrics on the 4 test files, writes replay PNGs, and prints the heuristic + GBM baseline comparison (new model should beat both on F1 / false-alarm rate).
- Intensity sanity asserts: median `I` < 0.05 on walking/stop test windows; monotonic rise after onset; Spearman(I, `thigh_tau_down_mean`) ≥ 0.6.
- `python -c "from ml.infer import InferenceEngine"` + a smoke test feeding rows from one test CSV as `/lowstate`-shaped dicts reproduces `evaluate.py`'s per-window outputs (confirms the live path).

## Risks / edge cases
- **12 events, FR only in 2 files** → LORO-CV is the honest metric; augment (tau/dq jitter, ±10% time-warp doubling as Hz robustness, amplitude scaling), oversample `front_both`, report FR recall separately.
- **Variable Hz** → solved by 500 Hz resample; explicitly verify on 409 Hz and 960 Hz test files.
- **Transition label noise** → POS_FRACTION=0.5 + exclude 0.2–0.5 band from training.
- **Intensity has no GT** → report as a *calibrated severity index*, not a measured force; physics formula authoritative, head auxiliary.
- **Real-time latency** → causal TCN last-timestep embedding; forward on `[60,200]` ≪ 2 ms budget at 500 Hz; config decimation if CPU-bound.
- **Leakage** → file-level grouping; normalize/Mahalanobis/calibration fit on train Walking only.
