# Leg-Entanglement TCN — Evaluation Verification Report

> **Status:** this is the BASELINE verification (pre-improvement); numbers below are intentionally
> left unchanged so the improvements stay directly comparable. The reliability follow-ups live in
> **`IMPROVEMENTS.md`**. Items resolved there:
> - The §0 **WARN** (1%-FAR threshold clamped at 0.9999, realized 3.36%) is fixed by **temperature
>   scaling** (T=6.66; de-saturates 29%→0%, Brier 0.110→0.099). Detection ranking (PR-AUC 0.821,
>   ROC-AUC 0.938) and LORO F1 0.83±0.17 are unchanged (calibration is monotonic).
> - The **§6 "persistence ≈ 6 ms" caveat** is replaced by a configurable time-based debounce
>   (75 ms → clean-walking FAR 0%, F1 −0.01, latency ~0).
> - **RR precision** improved via a per-leg threshold (RR≈0.9); root-cause of the `back_left_hand1`
>   LORO drop identified (out-of-distribution recording). Detector comparison now also includes the
>   user's **Statistical Detector** and a **feature ablation**.
> - Recommended deployment operating point saved to `operating_point.json` (loaded by `infer.py`).

Protocol: TARGET_FAR=1%, persistence k=3 windows (6 ms @ 500 Hz), leg_thr=0.5, window=200 samples (0.40 s). device=cuda.

## 1. Metrics & how computed
- **Window-level binary (P/R/F1, confusion)**: at the VAL-selected threshold, over every 0.40 s window of the TEST recordings (dense, hop=1). A window is positive iff Entangled-fraction ≥ 0.5.
- **PR-AUC / ROC-AUC**: threshold-free, on the same TEST windows.
- **Threshold**: lowest value with ≤1% per-window FAR on VAL negatives = **0.9999** (clamped at 0.9999; VAL negatives cluster near 1.0).
- **Persistence-gated alarm**: requires 3 consecutive windows ≥ threshold.
- **Per-leg P/R/F1**: per-leg prob ≥ 0.5 vs filename-derived affected legs, over all TEST windows.
- **Exact-match**: predicted leg-set == true leg-set, on truly-Entangled windows only.
- **Fixed split** = train on 13 / val on 3 / test on 4 recordings (leakage-safe, by recording). **LORO** = leave-one-positive-recording-out CV over the 12 positive events; each held-out file is scored with its own VAL-style threshold from its walking prefix.

## 0. Internal consistency checks
- [WARN] **VAL FAR at chosen threshold ≤ 1%** — realized VAL FAR = 3.36% (threshold 0.9999; CLAMPED at 0.9999 → scores saturate, so realized FAR may exceed target — read PR-AUC, not just F1)
- [PASS] **model scores are not fully saturated** — 7.5% of VAL scores lie in (0.01,0.99) — low value ⇒ overconfident, brittle thresholding
- [PASS] **p_bin and p_legs are coupled** — 100.0% of detection-positive windows also have ≥1 leg ≥0.5
- [PASS] **intensity ≈ 0 on negative windows** — median max-leg intensity on TEST negatives = 0.000
- [PASS] **no window crosses recordings; normalizer/calibrator fit on TRAIN only** — by construction (grouping=recording; fit_from_train / IntensityCalibrator.fit use train stems)
- [PASS] **streaming infer reproduces batched eval** — infer.py CPU check: max |diff| ≈ 6.6e-7

## 2. Binary detection — TCN, fixed split (pooled TEST)

| P | R | F1 | PR-AUC | ROC-AUC |
|---|---|----|--------|---------|
| 0.834 | 0.867 | 0.851 | 0.821 | 0.938 |

**Confusion matrix** (window-level @ thr=0.9999):

| | pred + | pred − |
|---|---|---|
| **actual +** | TP=3004 | FN=459 |
| **actual −** | FP=597 | TN=5456 |

## 3. Per-leg attribution — TCN, fixed split (pooled TEST)

| leg | P | R | F1 |
|-----|---|---|----|
| FR | 0.769 | 1.000 | 0.869 |
| FL | 0.721 | 0.999 | 0.838 |
| RR | 0.550 | 1.000 | 0.710 |
| RL | 0.758 | 0.913 | 0.828 |

Exact-match (set equality, entangled windows): **0.858** (3463 windows).

### Per-recording (TEST)

| recording | affected | F1 | R | P | persist-latency |
|---|---|---|---|---|---|
| back_left_hand2 | RL | 0.830 | 0.85 | 0.81 | 0 ms |
| back_right_wire1 | RR | 0.902 | 1.00 | 0.82 | 0 ms |
| front_left_hand2 | FL | 0.783 | 0.75 | 0.82 | 0 ms |
| front_both_wire2 | FL,FR | 0.994 | 0.99 | 1.00 | 210 ms |

## 6. False-alarm rate — per-window vs persistence-gated

**Caveat (consistency finding):** the literal k=3 rule is only 6 ms at dense hop=1, so it is *not* an effective debounce — it barely changes FAR. A time-meaningful debounce (k=38 ≈ 75 ms) is shown alongside. Negatives: walking4 (VAL, clean) and the walking *prefixes* of TEST files (same recordings where entanglement later occurs — some 'false' fires there may be early true detections of the wire/hand approaching).

| source | windows | raw FAR | persist k=3 (6ms) | debounce k=38 (~75ms) |
|---|---|---|---|---|
| walking4 (VAL, clean) | 800 | 2.75% | 1.38% | 0.00% |
| TEST walking prefixes | 6053 | 9.86% | 9.73% | 7.42% |

## 8. TCN vs heuristic baseline — identical protocol

Heuristic = max-leg `thigh_tau_down` z-score vs TRAIN-walking baseline; same threshold rule (VAL 1% FAR), same persistence, same metrics. Leg attribution = argmax-z on alarm (single leg — so it structurally cannot exact-match a 2-leg `_both` file).

| model | thr | P | R | F1 | PR-AUC | ROC-AUC | leg exact-match |
|---|---|---|---|----|--------|---------|---|
| **TCN** | 0.9999 | 0.834 | 0.867 | 0.851 | 0.821 | 0.938 | 0.858 |
| Heuristic | 5.87 | 0.685 | 0.084 | 0.150 | 0.639 | 0.735 | 0.079 |

## 7. Replay plots (all 20 recordings)
- walking5: no full-window data (short labeled episodes) — skipped

Wrote TCN replay plots for 19 recordings + heuristic plots for the 4 TEST files -> `ml/artifacts/plots/`

## 9. Worst-performing recordings (TCN, all files with positives)

| recording | split | F1 | R | P | affected | note |
|---|---|---|---|---|---|---|
| front_left_hand2 | test | 0.783 | 0.75 | 0.82 | FL | held-out |
| back_left_hand2 | test | 0.830 | 0.85 | 0.81 | RL | held-out |
| back_left_hand1 | train | 0.871 | 0.77 | 1.00 | RL | train (optimistic) |
| back_left_wire1 | train | 0.877 | 0.82 | 0.94 | RL | train (optimistic) |
| back_right_wire1 | test | 0.902 | 1.00 | 0.82 | RR | held-out |

**Failure analysis** (cross-referenced with LORO, the honest per-recording view):
- *Threshold brittleness dominates recall.* The clamped threshold (≈1.000) means any window where p_bin briefly dips below 1.0 becomes a false negative. On `front_left_hand2` (test F1 0.78) recall is lost exactly at mid-entanglement dips, not because the leg is wrong. The model is overconfident (only ~7% of scores are non-saturated), so a small threshold change swings recall a lot — PR-AUC (threshold-free) is the more stable read.
- *Hand-type, single-rear-leg recordings generalize worst.* `back_left_hand1` is perfect with the shipped model (it trains on it) but collapses to R≈0.30 when held out in LORO — it is the longest recording and its RL-hand signature isn't covered by the remaining positives.
- *Transient leg confusion at onset.* Around the walking→entangled transition the model can briefly favour the wrong leg (e.g. FL→FR spike on `front_left_hand2`) before locking on, which lowers per-leg precision (RR precision 0.55 is the weakest: it over-fires as the compensating rear leg).
- *Some 'false alarms' are early detections.* TEST walking-prefix FAR (≈9.9%) is far above clean walking4 (2.75%) because those prefixes are the same recordings' pre-entanglement walking — the model often fires a few hundred ms before the human-labeled onset (visible in the replay plots).

## 4. LORO-CV — every fold (20 epochs/fold)

| held-out fold | F1 | P | R | PR-AUC | ROC-AUC | latency | thr |
|---|---|---|---|---|---|---|---|
| back_both_wire1 | 0.978 | 0.99 | 0.97 | 0.997 | 0.997 | 310 ms | 0.3700 |
| back_both_wire2 | 0.944 | 0.90 | 0.99 | 0.915 | 0.969 | 0 ms | 0.9999 |
| back_left_hand1 | 0.454 | 0.94 | 0.30 | 0.777 | 0.843 | 1456 ms | 0.3180 |
| back_left_hand2 | 0.733 | 0.82 | 0.66 | 0.787 | 0.896 | 0 ms | 0.9999 |
| back_left_wire1 | 0.612 | 0.71 | 0.54 | 0.688 | 0.879 | 0 ms | 0.9999 |
| back_right_hand1 | 0.784 | 0.99 | 0.65 | 0.936 | 0.906 | 908 ms | 0.0515 |
| back_right_hand2 | 0.992 | 0.99 | 1.00 | 1.000 | 1.000 | 200 ms | 0.1731 |
| back_right_wire1 | 0.891 | 0.84 | 0.95 | 0.821 | 0.935 | 0 ms | 0.9999 |
| front_both_wire1 | 0.936 | 0.98 | 0.89 | 0.975 | 0.967 | 396 ms | 0.9607 |
| front_both_wire2 | 0.977 | 0.95 | 1.00 | 1.000 | 1.000 | 170 ms | 0.9999 |
| front_left_hand1 | 0.970 | 0.98 | 0.96 | 0.990 | 0.991 | 318 ms | 0.1858 |
| front_left_hand2 | 0.713 | 0.78 | 0.66 | 0.787 | 0.911 | 0 ms | 0.9999 |

**LORO detection F1 = 0.832 ± 0.166** (min 0.454 @ back_left_hand1, max 0.992).

## 5. Why LORO differs from the fixed split
- Fixed-split pooled-test F1 = **0.851**; LORO mean F1 = **0.832** (± 0.166).
- They measure different things. The fixed split scores **one** train/test partition (4 specific test files); LORO averages **12** partitions, each holding out a different positive event, so it estimates generalization to an unseen recording and exposes between-recording variance (the ± std).
- The fixed-split number falls inside the LORO mean ± std, so the chosen split is representative (not cherry-picked). Folds where a distinctive recording is held out (e.g. the only same-type sibling) drop most — that variance is invisible in a single split.
