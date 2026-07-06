# VALIDATION_REPORT_V2 — Proving (not assuming) the v2 improvements

Independent validation of the v2 retrain. Every claim is tested against held-out data, an
ablation that removes the new recordings, and leave-one-new-recording-out (LONRO) retrains.
Models: **v1** = `ml/artifacts/before_v1/` (original 20 recordings), **v2** = shipped
(`ml/artifacts/`, 25 recordings), **no-new-data** = ablation retrain on the v1 split with the
current code. Protocol identical throughout (raw threshold 0.9999; window-level; LORO/LONRO
fit normalizer+calibrator on training only). Plots: `ml/artifacts/plots/validation_v2/`.

> **Headline:** v2 fixes all four field issues **on the shipped model**, and two of them are
> proven to **genuinely generalize** to held-out data. But LONRO shows the **front-right
> detection** fix and the **post-entanglement RR-lock** fix are largely **memorization** of the
> single example of each — they do **not** generalize to held-out data. v2 is safe to deploy as
> an increment (it is not worse than v1 in the field) but **more data is required** before FR
> detection and post-entanglement-lock suppression can be trusted on unseen situations.

---

## 1. Original held-out test set (the 4 original entanglement recordings) — v1 vs v2

Same 4 files are held out for both models (clean apples-to-apples).

| metric | v1 | v2 | Δ |
|---|---|---|---|
| Precision | 0.834 | 0.800 | −0.034 |
| Recall | 0.867 | 0.815 | −0.052 |
| **F1** | 0.851 | 0.807 | −0.044 |
| **PR-AUC** | 0.821 | 0.808 | −0.013 |
| **ROC-AUC** | 0.938 | 0.929 | −0.009 |
| Leg exact-match | 0.858 | **0.910** | +0.052 |
| False-alarm (walking-prefix) | 9.9% | 11.7% | +1.8 pts |
| Median alarm latency | 0 ms | 0 ms | — |
| **per-leg F1** FR / FL / RR / RL | 0.87 / 0.84 / 0.71 / 0.83 | **0.65** / 0.85 / 0.70 / 0.82 | FR −0.22 |
| RR precision | 0.55 | 0.55 | flat |
| **FR precision** | 0.77 | **0.48** | **−0.29** |

**Verdict:** threshold-free discrimination is **maintained within noise** (PR-AUC −0.013,
ROC-AUC −0.009); the F1 drop is threshold-saturation. Two real regressions: **FR per-leg
precision** (the model over-fires FR after learning the single FR recording) and a small
walking-prefix FAR rise. RR precision and the other legs are unchanged; leg exact-match
improved.

---

## 2. Confusion matrices (original-4, window-level @ 0.9999)

`plots/validation_v2/confusion_v1_v2_orig4.png`

| | v1 | v2 |
|---|---|---|
| TP | 3004 | 2823 |
| FN | 459 | 640 |
| FP | 597 | 707 |
| TN | 5456 | 5346 |

v2 has fewer TP / more FN (lower recall at the clamped threshold) and slightly more FP — the
fixed-threshold operating-point shift, consistent with §1.

---

## 3. Calibration (Brier, ECE, thresholds)

Measured on the original-4 logits; temperature fit on each model's own validation split.

| | T | Brier raw→cal | ECE raw→cal | sat@1.0 | raw thr | calibrated thr |
|---|---|---|---|---|---|---|
| v1 | 8.20 | 0.110 → 0.102 | 0.113 → 0.126 | 29% | 0.9999 | ~0.97 |
| v2 | 5.92 | 0.128 → 0.118 | 0.134 → 0.156 | 27% | 0.9999 | 0.963 |

On the original-4 set v2's Brier is **slightly worse** (it is marginally less accurate there,
matching §1). NOTE: on the **full v2 test set** (which includes the held-out `lock_stop`
negatives the v2 model handles well) v2's Brier is **0.091** — better than v1 — so calibration
quality is eval-set dependent. Both models stay over-confident (T≫1); ECE does not improve
under temperature scaling (small VAL). Deployment uses the de-clamped calibrated confidence.

---

## 4. The five new recordings — v1 vs v2, with train/held-out status

FIRE% = windows with raw p_bin ≥ 0.9999. **Bias flag is critical**: results on TRAIN files are
NOT unbiased generalization. The unbiased estimate for those is in §6 (LONRO).

| recording | split | phase | v1 | v2 (shipped) | unbiased (held-out) |
|---|---|---|---|---|---|
| `lock_stop` | **TEST (held-out ✓)** | Lock fire | 0.8% | **0.0%** | **0.0%** (already held-out) |
| `walk_stop_back` | TRAIN (biased) | Walking fire | 9.7% | 0.0% | **1.9%** (LONRO) |
| `walk_stop_back` | TRAIN (biased) | Lock fire | 4.7% | 0.0% | **37.7%** (LONRO) |
| `front_right_wire` | TRAIN (biased) | Entangled FR-leg | 4% | 100% | **1.8%** (LONRO) |
| `back_right_stop` | VAL | Entangled fire | 28.1% | 26.7% | n/a (VAL≈held from weights) |
| `back_right_stop` | VAL | Stop RR-leg | 32.8% | **0.0%** | — |
| `back_right_intensity` | TRAIN (biased) | Lock fire | 100% | 0.0% | **100%** (LONRO) |

Real entanglement applied **while standing still** (`back_right_stop`, VAL) is still detected by
v2 (26.7%), confirming the Lock handling does not blanket-suppress genuine at-rest entanglement.

---

## 5. Ablation — remove the new recordings (isolate the data effect)

Retrained on the **v1 split with the current code** ("no-new-data"). This separates the effect
of the new DATA from any code/seed change.

| | orig-4 F1 | lock_stop Lock | walk_stop_back Lock | front_right_wire Ent. (FR-leg) | back_right_intensity Lock |
|---|---|---|---|---|---|
| **no-new-data** | 0.838 | 1.8% | 14.0% | 31.2% (13%) | 100% |
| **v2 (with new data)** | 0.807 | 0.0% | 0.0% | 78.5% (100%) | 0.0% |

**Conclusion:** removing the new data brings the failures **back** (lock false alarms, no FR
detection, 100% post-entanglement-lock firing) — so the v2 fixes are caused by the **new data**,
not code changes. The same comparison shows the new data **costs ~0.031 F1 on the original-4**
(0.838 → 0.807); code/seed alone accounts for only 0.851 → 0.838.

---

## 6. Leave-one-new-recording-out (LONRO) — genuine generalization vs memorization

For each new recording **in TRAIN**, retrained with it **held out**, then evaluated on it
(unbiased). `lock_stop` is already in TEST, so the shipped result on it is already unbiased.

| held-out recording | what it tests | shipped (trained) | **LONRO (held-out)** | generalizes? |
|---|---|---|---|---|
| `lock_stop` (Lock) | rigid-stance lock FP (issue 1) | 0.0% | **0.0%** | **YES — genuine** |
| `walk_stop_back` (Walking) | backward-walk FP (issue 3) | 0.0% | **1.9%** | **YES — genuine** (v1 was 9.7%) |
| `walk_stop_back` (Lock) | stand-up/settle lock FP (issue 1) | 0.0% | **37.7%** | **NO — memorized** |
| `front_right_wire` (Entangled FR) | front-right detection (issue 4) | 100% | **1.8%** | **NO — memorized** |
| `back_right_intensity` (Lock) | post-entanglement RR-lock FP (issue 2) | 0.0% | **100%** | **NO — memorized** |

LONRO models keep orig-4 F1 healthy (0.81–0.85), so these are valid models, not degenerate.

**Interpretation:** with only **one example** of front-right entanglement and **one** of a
post-RR-entanglement lock, the model **memorizes** those specific recordings — held out, FR
detection collapses to baseline (1.8% ≈ v1's 4%) and the post-entanglement RR-lock fires 100%
again. By contrast, **rigid-stance lock** (lock_stop) and **backward-walking false alarms** are
covered by multiple/representative negatives and **genuinely generalize**.

---

## 7. Are the original issues fixed without degrading existing performance?

| issue | on the shipped model | generalizes to unseen? | existing perf |
|---|---|---|---|
| 1. stand-up / Lock false RR | FIXED (rigid lock 0%, held-out) | rigid YES; stand-up-settle NO (37.7%) | — |
| 2. post-entanglement RR-lock | FIXED on recording (100%→0%) | **NO (held-out 100%)** | — |
| 3. backward-walk / stop FP | FIXED (held-out 1.9%) | **YES** | — |
| 4. front-right detection | FIXED on recording (4%→100%) | **NO (held-out 1.8%)** | **FR per-leg F1 ↓ 0.87→0.65, precision ↓ 0.77→0.48** |
| RR precision | — | — | **unchanged (0.55)** ✓ |

So: issues 1(rigid)/3 are genuinely fixed; issues 2/4 are fixed only for the specific
recordings (memorized). The one **existing-performance degradation** is **front-leg (FR)
precision** — a direct side effect of training on a single FR recording. RR precision is
preserved.

---

## 8. Replay plots (before/after)

`ml/artifacts/plots/validation_v2/`:
- `confusion_v1_v2_orig4.png`
- `replay_v1_<rec>.png` / `replay_v2_<rec>.png` for the 4 original test files and all 5 new
  recordings (18 plots). Each shows GT ribbon, p(entangled), persistence alarm, per-leg
  probabilities, and intensity. The `back_right_intensity` and `front_right_wire` v1↔v2 pairs
  visualize the fixes (and, read with §6, their memorized nature).

---

## 9. Genuine generalization vs memorization — summary

- **Genuine generalization (held-out proven):** backward-walking false alarms (issue 3) and
  rigid-stance lock false alarms (issue 1, `lock_stop` 0% held-out). These had representative
  coverage (multiple walking/stop negatives + a clean rigid-lock recording).
- **Memorization (held-out fails):** front-right detection (issue 4) and post-RR-entanglement
  lock suppression (issue 2), and the stand-up/settle variant of lock — each backed by only a
  **single** recording. The shipped model fixes them only because it trains on those exact files.
- The deployment **stationarity gate** resets stale buffers on Stop/Lock but, by design, does not
  blanket-suppress while stationary, so it does **not** cover the memorized post-entanglement-lock
  gap on unseen situations.

---

## 10. Conclusion — deployment readiness

**v2 is safe to deploy as an incremental upgrade, but not a finished fix.**

- It is **not worse than v1 in the field**: the no-new-data ablation (≈ v1) and v2 both keep
  orig-4 discrimination within noise, and v2 strictly improves the **proven-generalizing**
  scenarios (backward-walk, rigid lock) and every scenario it trained on.
- It carries one real **regression to watch**: FR per-leg **precision** (0.77→0.48). Mitigate in
  deployment with the per-leg RR-style threshold raised for FR if false FR attributions are seen.
- It does **not** yet provide trustworthy **front-right detection** or **post-entanglement-lock
  suppression** on **unseen** situations (memorized with one example each).

**Recommendation — collect more data before relying on issues 2 & 4 in the field:**
1. **≥3–5 front-right entanglement** recordings (varied gait/speed/wire) — to convert issue 4
   from memorized to generalizing and recover FR precision.
2. **≥3–5 post-entanglement-lock** recordings across legs (entangle → release → hold) — for issue 2.
3. **A few more stand-up / settle** sequences — for the stand-up-settle lock variant.
Then retrain and re-run this validation; expect the LONRO held-out numbers (not the shipped-on-
training numbers) to be the acceptance criteria.

**Ship v2 now** for the genuinely-fixed behaviors (backward-walk + rigid-lock false alarms) and
the recorded scenarios, with the stationarity gate enabled; **gate field trust** of FR detection
and post-entanglement-lock on the additional data collection above.
