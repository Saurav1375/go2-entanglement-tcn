# Intelligent Recovery — Design & Evidence

Branch **`feature/intelligent-recovery`** (from `feature/recovery-framework`). Transforms the
recovery framework's single `BalanceStand` reaction into a **pluggable, detector-aware,
closed-loop recovery strategy system** whose goal is to *maximize the probability of physically
freeing an entangled leg* using **only official Unitree Sport APIs**.

> **Foundational rule honored:** the FSM, safety mechanisms, configuration system, tests, and
> detector integration are unchanged. **Only the `RECOVERING` (+ its `VERIFYING` re-attempt loop)
> internals were redesigned.** All original behavior and tests still pass (10/10), and the
> detector package is byte-for-byte untouched.

All Unitree API facts here were source-verified (unitree_ros2 `ros2_sport_client.{h,cpp}`,
unitree_sdk2, README). **Critical honesty:** every command's *kinematic* effect is verified, but
**no Sport-API command's *disentanglement* effect (actually freeing a snagged leg) is verified** —
those are design assumptions grounded in robotics literature. §6 separates verified from assumed.

---

## Phase 2 — Analysis of the existing design (what stays, what changes)

| Component | Verdict |
|---|---|
| FSM graph (IDLE→MONITORING→CONFIRMING→STOPPING→RECOVERING→VERIFYING→RESUMING→COOLDOWN, FAULT/ESTOP) | **KEEP** — sound. |
| Safety (dry-run gate, retry, watchdog, command timeout, cooldown, confirmation, e-stop, fault, Damp) | **KEEP** — unchanged. |
| YAML config system, detector integration (`/entanglement_state`), unit tests | **KEEP** — unchanged; only extended. |
| Recovery *strategy* (single `StopMove→BalanceStand`) | **REDESIGN** — was the only weakness: one motion, no per-detection adaptation, no closed loop beyond a same-command re-attempt, ignored `alarm_leg`/`confidence`/`intensity`. |

**Change made:** `RECOVERING` now delegates to a **Strategy Manager**; the existing
`RECOVERING↔VERIFYING` loop iterates a *detector-aware ordered list of strategies* (closed loop),
escalating to a safe stop on exhaustion. No new states; no change to the other states.

---

## Phase 3 — Strategy system (pluggable)

```
MONITOR → CONFIRM → STOP → ┌─────────── Recovery Strategy Manager ───────────┐
                           │  order = policy(alarm_leg, confidence, intensity)│
                           │  for strategy in order:                          │
                           │     RECOVERING(strategy)  →  VERIFYING(detector)  │
                           │        recovered? ── yes ──▶ RESUME               │
                           │           │ no                                    │
                           │           ▼ next strategy … exhausted ▶ escalate  │
                           └──────────────────────────────────────────────────┘
                                          ↓ (unresolved)         ↓ (fall)
                                     EmergencyStop(Damp)→FAULT   RecoveryStand
```

New pure modules (no ROS; unit-tested): `strategies.py` (Strategy base + 7 strategies + verified
`MotionPlan`s), `strategy_manager.py` (ordering policy), `plan_runner.py` (executes a MotionPlan's
one-shot/streamed/hold steps in the node). The FSM holds a `StrategyManager`; the node holds a
`PlanRunner`. Strategies are registered in `strategies.ALL_STRATEGIES` — adding one is a class +
a registry entry + an enable flag.

**Strategies** (each → verified Sport API):

| Strategy | FSM Command | Verified API(s) | What it does |
|---|---|---|---|
| `balance_stand` | BALANCE_STAND | BalanceStand 1002 | re-settle actively-balanced stance (gentlest) |
| `weight_shift` | WEIGHT_SHIFT | Euler 1007 (+ BalanceStand) | tilt body to unload the snagged corner (CoG-shift *approx*) |
| `small_reverse` | SMALL_REVERSE | Move 1008 (stream) + StopMove 1003 | back the foot out (front snag) / nudge forward (rear) |
| `small_sidestep` | SMALL_SIDESTEP | Move 1008 + StopMove | strafe away from the snagged side |
| `rotate` | ROTATE | Move 1008 + StopMove | yaw to unwind a wrapped tether |
| `recovery_stand` | RECOVERY_STAND | RecoveryStand 1006 | escalation: right a **fallen** body |
| `emergency_stop` | DAMP | StopMove 1003 + Damp 1001 | terminal safe limp; hand to operator |

---

## Phase 4 — Detector-aware policy (leg / confidence / intensity)

`StrategyManager.order(context)` builds the ordered list. Every threshold is a config parameter
(`recovery.yaml`); see `docs/diagrams/strategy_selection_flowchart.md`.

- **Leg-aware** (from `alarm_leg`): the *direction* of each motion is computed per leg
  (strategies.py). E.g. side-step `vy` is +0.15 for a right-side snag (FR/RR → step left) and
  −0.15 for a left-side snag (FL/RL → step right); reverse is `vx<0` for a front snag, `vx>0` for
  a rear snag; weight-shift tilts away from the snagged corner. → *"RR chooses a different motion
  than FR."* (Directions are design assumptions — §6.)
- **Confidence-aware**: Move-based strategies (which actually drive the robot) are attempted only
  when `confidence ≥ active_confidence_min` (default 0.6). On a low-confidence alarm the system
  stays gentle (BalanceStand + WeightShift) then escalates — it does not drive on a guess.
- **Intensity-aware**: motion magnitudes scale *down* as intensity rises
  (`magnitude = base·(1 − (1−intensity_min_scale)·intensity)`) — gentler when more stuck. Above
  `high_intensity_thresh` (default 0.8) the policy is **conservative**: skip active Move entirely
  (avoid thrashing a strongly-pinned leg / risking damage) and escalate. This realizes
  *"low intensity → gentle; high intensity → conservative."*
- **Posture-aware**: if telemetry shows a **fallen** body, the order is `[recovery_stand,
  emergency_stop]` — RecoveryStand is the fall primitive and is **never** used on an upright snag
  (forcing it against a hard snag is unsafe; evidence in §6).

No magic numbers: `active_confidence_min`, `high_intensity_thresh`, `intensity_min_scale`,
per-strategy speeds/durations, and enable flags are all in `recovery.yaml` with rationale.

---

## Phase 5 — Closed-loop (perception-driven)

Recovery is `Motion → Detector → Recovered?`. After **each** strategy's MotionPlan executes,
`VERIFYING` consults the live detector: if `entangled` clears for `verification_duration_s` →
`RESUMING` (success); if still entangled after `verify_timeout_s` → advance to the **next**
strategy and loop; when the ordered list is exhausted → `FAULT` (robot left safely stopped/damped).
The detector — not a timer — decides success, and guides which strategy runs next.

---

## Phase 6 — Verified vs. Design-Assumption (the honest core)

| Item | Status | Evidence |
|---|---|---|
| Move api_id 1008, params `{x:vx,y:vy,z:vyaw}` (body frame), must be streamed, StopMove 1003 ends it | **VERIFIED** | `ros2_sport_client.cpp`; unitree_sdk2 example (dt=0.005 recurrent); go2_ros2_sdk |
| Euler api_id 1007, params `{x:roll,y:pitch,z:yaw}`, ranges ±0.75/±0.75/±1.5 rad | **VERIFIED** | `ros2_sport_client.cpp`; go2_robot README |
| BalanceStand 1002 / RecoveryStand 1006 / Damp 1001 / StopMove 1003 ids + purpose | **VERIFIED** | source headers |
| RecoveryStand = right a **fallen** body (not an upright primitive) | **VERIFIED** (purpose) | Unitree docs / npaka mirror |
| **Any strategy actually frees a snagged leg** | **ASSUMPTION** | robotics literature (CoM-shift / back-out), *no Go2-specific evidence* |
| Leg-aware **directions** (which way to move/tilt per leg) | **ASSUMPTION** | rigid-body reasoning; Euler roll/pitch *sign convention undocumented* |
| Move/Euler **magnitudes & durations** (0.15 m/s, 0.4 rad/s, 0.6 s…) | **TUNABLE ASSUMPTION** | set below the one verified demo value (0.3 m/s); no extraction reference exists |
| Euler keeps feet planted | **INFERRED** (README "posture when standing/walking", not explicit) | README |

**Not achievable with high-level Sport APIs (verified by exhaustive method review):** lifting/
unloading a **single named leg**, **swing-leg force/torque** control (the literature-validated
reflex: hip-retraction + knee-torque, arXiv:2304.02129 / 2010.11251), step-in-place, and vertical
CoG/body-height change (`BodyHeight 1013` / `FootRaiseHeight 1014` are **removed** from the V2
interface). These require the **low-level `rt/lowcmd` interface** (`unitree_go/msg/LowCmd`,
per-motor `q/dq/kp/kd/tau` over the 12 motors; see unitree_sdk2 `go2_stand_example.cpp`), which
disables firmware balancing and is significantly riskier — proposed as future work, **not**
implemented here.

---

## Phase 7 — Safety preserved (nothing regressed)

✓ dry-run (`enable_actuation:false` default) ✓ retry ✓ watchdog ✓ command timeout ✓ cooldown
✓ confirmation ✓ e-stop (`/recovery_estop`→Damp) ✓ fault handling ✓ YAML config ✓ existing tests
(10/10) ✓ detector package untouched ✓ backward compatible (gentlest strategy is BalanceStand, so
mild cases behave as before). New active-motion strategies are gated (confidence + intensity) and
magnitude-limited; on any API failure or exhaustion the robot is stopped/damped.

---

## Architecture, assumptions, limitations, future

- **Architecture diagram:** `docs/diagrams/intelligent_recovery_architecture.md`
- **Strategy/closed-loop diagram:** `docs/diagrams/recovery_strategy_flow.md`
- **Selection flowchart:** `docs/diagrams/strategy_selection_flowchart.md`

**Assumptions:** (1) high-level commands can free a leg (unproven on Go2); (2) leg-aware
directions + Euler sign conventions; (3) Move/Euler magnitudes/durations; (4) StopMove-before-
switch and inter-command settle times are prudent defaults, not documented requirements;
(5) `SportModeState.mode` mapping is firmware-version-sensitive.

**Limitations:** observe-only by default; actuated behavior is **unvalidated on hardware** (no
robot in CI); high-level APIs cannot do precise per-leg extrication; rotate direction is a guess
(wrap chirality unobservable). Recovery may fail on tightly-snagged legs → terminal Damp + operator.

**Future (low-level extension, not implemented):** a `LowCmdStrategy` on `rt/lowcmd` implementing
the literature swing-leg reflex (hip-retraction velocity + knee feed-forward torque on the
`alarm_leg`) for true single-leg extrication — behind a separate, explicitly-armed safety gate,
since it bypasses firmware balancing.
