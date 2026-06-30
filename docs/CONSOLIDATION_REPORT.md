# Consolidation Report — Production-Ready Detector + Recovery

This documents the final consolidation of the leg-entanglement **detector** and **intelligent
recovery** work into a single production branch (`main`). It contains: (1) what was verified, (2) a
deployment checklist, (3) a summary of every change made in this pass, and (4) the remaining
assumptions that still require **real-robot validation**.

Consolidation date: 2026-06-30. Shipped model: **v2** (25 recordings, 16/4/5 split, 15 positives).

---

## 1. Verification performed (all green)

| check | command | result |
|---|---|---|
| Recovery FSM unit tests | `test/test_recovery_fsm.py` | **16/16 pass** |
| PlanRunner unit tests | `test/test_plan_runner.py` | **3/3 pass** |
| Pure recovery imports | all of `states/recovery_fsm/strategies/strategy_manager/plan_runner/sport_api` | import OK |
| ML module self-tests | `python -m ml.{config,io_load,resample,features,windowing,normalize,model,losses,intensity,calibration}` | all OK |
| Detector runtime == research | `tools/validate_runtime.py --backend onnx` | p_bin **3.8e-7**, p_leg **6.3e-7**, intensity **2.1e-7** → **MATCH** |
| Real-time budget (CPU, 1 thread) | (same) | mean **1.0 ms**/window, p95 1.2 ms < 2.0 ms @ 500 Hz |
| Research artifacts vs deployed | `diff` normalize/intensity_calib/calibration | **identical** (no deployment drift) |

> **No regression:** the detector model, preprocessing, normalization, intensity, and calibration
> are byte-for-byte the shipped v2 export; the recovery package is additive and only *subscribes* to
> the detector's existing `/entanglement_state`.

---

## 2. Deployment checklist (GO2)

Follow **[`../robot_package/RUNBOOK.md`](../robot_package/RUNBOOK.md)** for the full procedure; this
is the condensed go/no-go list.

**Pre-deploy (dev machine)**
- [ ] `git` on `main`, clean tree.
- [ ] Recovery tests pass: `test_recovery_fsm.py` (16) + `test_plan_runner.py` (3).
- [ ] Runtime equivalence passes: `python robot_package/tools/validate_runtime.py --backend onnx` (MATCH).
- [ ] If the model was retrained: re-run `python robot_package/export_model.py` and re-validate.

**On the robot — build**
- [ ] Copy `robot_package/src/{entanglement_interfaces,entanglement_detector,entanglement_recovery}`
      into `~/ros2_ws/src/` and `requirements_robot.txt` to `~/ros2_ws/`.
- [ ] `pip3 install -r requirements_robot.txt` into the ROS 2 interpreter.
- [ ] `colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery`.
- [ ] `ros2 interface show entanglement_interfaces/msg/EntanglementState` succeeds.

**On the robot — network / telemetry**
- [ ] `ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` / `CYCLONEDDS_URI` match the robot.
- [ ] `ros2 topic hz /lowstate` and `ros2 topic hz /sportmodestate` both show data (BEST_EFFORT QoS).
- [ ] `ros2 topic list | grep sport/request` confirms the sport service is reachable.

**Tier A — detector only (no motion)**
- [ ] `ros2 launch entanglement_detector entanglement.launch.py`.
- [ ] `/entanglement_state` publishes at ~`/lowstate` rate; CPU headroom OK (`top`).
- [ ] No false alarms during normal walking / stop / Lock.

**Tier B — recovery DRY-RUN (still no motion, `enable_actuation:=false` default)**
- [ ] `ros2 launch entanglement_recovery detector_and_recovery.launch.py`.
- [ ] Synthetic detection walks the FSM ladder; logs `[DRY-RUN] would send …` with correct api_ids.
- [ ] `/recovery_estop` and `/recovery_reset` behave as expected.
- [ ] **Verify the command sequence, leg-directions, and timing match expectations.**

**Tier C — ACTUATED bring-up (motion — clearance required)**
- [ ] ≥2 m clearance, flat high-friction floor, battery > `min_soc_pct`, **e-stop in hand**.
- [ ] `… enable_actuation:=true`; start with a gentle induced snag.
- [ ] StopMove → BalanceStand happens without violent motion; `/recovery_estop` damps instantly.
- [ ] Tune `recovery.yaml` magnitudes/directions (see §4).

---

## 3. Summary of every change in this consolidation pass

**Behavior-preserving cleanup (no functional change):**
- `ml/config.py`: removed dead constant `OPPOSITE_LEG_PAIR` (zero references; grep-verified).
- `entanglement_recovery/strategies.py`: removed dead registry `BY_COMMAND` (zero references; `BY_NAME` retained).
- **Deferred (intentionally not removed)** to avoid touching the equivalence-guarded runtime mirror:
  `_reset_armed` (write-only in both `engine.py` and `infer.py`) and `IntensityCalibrator.intensity()`
  (unused method in `ml/intensity.py`). Removing these would edit mirrored files for no behavioral
  gain; left for a dedicated, separately-validated change.

**Documentation (the bulk of the work):**
- **Root `README.md`** — rewritten as the project front door: system-architecture diagram
  (detector → recovery → robot), the two-pipeline structure, repo map, two quick-start paths,
  install + colcon build (all 3 packages), usage, a testing table, **v2 results** (replacing the
  stale v1 figures), a documentation index, limitations/hardware-validation status, future work.
  Stale facts corrected: 20→**25 recordings**, 12→**15 positives** / 8→**10 negatives**,
  13/3/4→**16/4/5 split**, added the **Lock** negative class, added the deployment + recovery story.
- **`robot_package/README.md`** — now documents all **three** packages (added `entanglement_recovery`
  to the tree + build), points to `RUNBOOK.md`, and completes the detector config table (added
  `intensity_blend`, `stationary_dq_thresh`, `stationary_min_ms`, `stabilize_ms`, `log_every_n`) plus
  a recovery-config section; updated the "observe-only" note to reflect recovery's dry-run actuation.
- **`docs/RECOVERY_TESTING.md`** — test counts `10/10` → **`16/16` + `3/3` PlanRunner**; added the
  PlanRunner command and acceptance criterion.
- **`docs/RECOVERY_DELIVERABLES.md`** — added a status banner (intelligent layer is now on `main`);
  file map now lists `strategies.py` / `strategy_manager.py` / `plan_runner.py` / `test_plan_runner.py`;
  test counts corrected to 16+3.
- **`docs/RECOVERY_DESIGN.md`** — top banner + §4 note: the fixed `StopMove→BalanceStand` /
  "does **not** command Move" policy is **superseded** in `RECOVERING` by the detector-aware strategy
  ladder (which *does* use Move for selected strategies); file map reconciled (strategies/manager/
  runner + new diagrams + INTELLIGENT_RECOVERY.md).
- **`docs/REPORT.md`** — v1-superseded banner pointing to the v2 reports for current metrics.
- **`docs/IMPROVEMENTS.md`** — v1-study banner (calibration refreshed for v2; ablation is the v1 study).
- **`docs/PUBLISH_REVIEW.md`** — historical (pre-recovery, pre-v2) banner.
- **`docs/CONSOLIDATION_REPORT.md`** — this file.

**Branch consolidation:**
- All work fast-forwarded into **`main`** (detector + v2 retrain + recovery framework + intelligent
  recovery + hardware-readiness fixes + this consolidation). No new feature branches created.

**Explicitly unchanged:** the TCN architecture, model weights, training/eval logic, the v2 dataset
and labels, all deployed model artifacts, and every `entanglement_detector` / `entanglement_interfaces`
runtime file.

---

## 4. Remaining assumptions requiring real-robot validation

These are **design assumptions**, not verified GO2 behaviors. The recovery package ships with
`enable_actuation: false` precisely because of them.

1. **Disentanglement efficacy.** Every Sport-API command's *kinematic* effect is verified against the
   SDK, but whether any given motion actually **frees a snagged leg** is unproven on hardware.
2. **Weight-shift sign convention.** The Euler roll/pitch directions used to "tilt away from the
   snag" (`weightshift_roll`/`weightshift_pitch` signs in `strategies.py`) are an assumption — confirm
   the tilt unloads the correct corner before trusting it.
3. **Move directions.** `small_reverse` (front-snag → back up), `small_sidestep` (step away from the
   snagged side), and especially `rotate` (yaw direction — wrap chirality is unobservable) are guesses
   to validate; magnitudes/durations in `recovery.yaml` are conservative starting points.
4. **Strategy ordering / thresholds.** `active_confidence_min` (0.6) and `high_intensity_thresh`
   (0.8) gating active Move strategies, and the overall ladder order, are heuristics tuned offline.
5. **Unitree timing facts.** Inter-command settle times and the necessity of `StopMove` before a mode
   switch (`*_settle_s`) are assumptions (RECOVERY_DESIGN §10) — tune on hardware.
6. **Detector on the live loop.** Offline metrics + ~1e-6 runtime equivalence are verified, but
   real-time behavior on the robot's actual CPU/DDS (latency, jitter, the stationarity gate under
   real Stop/Lock transitions) needs a field session.
7. **Battery/clearance gates.** `min_soc_pct` gating RecoveryStand and posture thresholds
   (`tip_*_deg`) need confirmation against real `bms_state.soc` / IMU readings.

**Top data need:** more **front-right** and **front-both** recordings — FR is positive in only 3
files, which drives the FR per-leg precision trade-off and most of the LORO variance.
