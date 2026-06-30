# Recovery Framework — Design Document

> **Note — the `RECOVERING` state was redesigned by the intelligent-recovery layer (now on `main`).**
> §4 below describes the *base* framework's fixed `StopMove → BalanceStand` sequence and its
> deliberate "does **not** command `Move`" policy. That `RECOVERING` behavior is now **superseded**
> by a pluggable, detector-aware **strategy ladder** (`balance_stand → weight_shift → optional
> Move-based small_reverse/sidestep/rotate → emergency_stop`) — i.e. the framework now *does*
> command `Move` for selected strategies, gated by confidence/intensity and still dry-run by
> default. Everything else here (STOPPING, escalation-to-RecoveryStand, safety rules, watchdog,
> Sport-API facts) is current. See **[`INTELLIGENT_RECOVERY.md`](INTELLIGENT_RECOVERY.md)** for the
> current `RECOVERING` design and the verified-vs-assumed list.

Recovery framework for the Unitree Go2 that reacts to the leg-entanglement detector and
brings the robot to a safe, stable state using **official Unitree high-level Sport APIs**.

- **Branch:** `feature/recovery-framework` (consolidated into **`main`**)
- **Principle:** robustness, safety, maintainability; **the detector is untouched** and keeps
  working exactly as before. Recovery only *subscribes* to detector events.
- **Default posture:** **observe-and-intent-only** (`enable_actuation: false`) — the framework
  runs the full state machine and logs the commands it *would* send, but sends nothing to the
  robot until an operator explicitly enables actuation. This makes it safe to deploy/validate
  without moving a robot.

All Unitree API facts below were verified against primary sources on 2026-06-30
(`unitree_ros2` `ros2_sport_client.h` + README, `unitree_sdk2` `sport_client.hpp`, the
`unitree_api` `.msg` files, and the npaka mirror of Unitree's official Sports Service doc).
Items that could **not** be verified are listed in §10 (Assumptions) and §11 (Limitations).

---

## 1. Overall architecture

```
┌────────────────────┐   /entanglement_state    ┌────────────────────────────────┐
│ entanglement_detector│ ───(EntanglementState)──▶│        recovery_node           │
│   (UNCHANGED)        │                          │  ┌──────────────────────────┐  │
└────────────────────┘                            │  │ RecoveryFSM (pure Python)│  │
                                                   │  │  states + guards + timers│  │
┌────────────────────┐   /sportmodestate          │  └───────────┬──────────────┘  │
│  Go2 sport service │ ──(SportModeState)────────▶│   robot_state│ actions          │
│                    │                            │  ┌───────────▼──────────────┐  │
│                    │ ◀─(unitree_api/Request)────│  │      SportClient          │  │
│                    │   /api/sport/request       │  │ api_id -> Request publish │  │
│                    │ ──(unitree_api/Response)──▶│  │ /api/sport/response codes │  │
└────────────────────┘   /api/sport/response      │  └───────────────────────────┘  │
                                                   │   publishes /recovery_status     │
                                                   └────────────────────────────────┘
```

Four cleanly-separated modules (see §12 file map):

1. **`recovery_fsm.py` — pure Python**, no ROS imports. Holds all decision logic (states,
   transitions, guards, timers, retry/cooldown). Deterministic and fully unit-testable. Emits
   abstract `Command`s (`STOP_MOVE`, `BALANCE_STAND`, `RECOVERY_STAND`, `DAMP`, `NONE`).
2. **`sport_client.py` — ROS wrapper**. Maps abstract commands to verified Sport API IDs,
   builds `unitree_api/msg/Request`, publishes on `/api/sport/request`, and tracks
   `/api/sport/response` return codes. Honors the `enable_actuation` safety gate (dry-run).
3. **`robot_state.py` — ROS wrapper**. Subscribes `/sportmodestate`, decodes `mode`/`gait_type`/
   `imu_state.rpy` into a clean `RobotState` (UPRIGHT / LOCOMOTION / STOPPED / FALLEN / …).
4. **`recovery_node.py` — orchestrator**. Subscribes detector events + robot state + command
   responses, ticks the FSM on a timer, executes emitted commands via SportClient, and publishes
   `/recovery_status` diagnostics.

**Why this split:** the FSM (the safety-critical logic) is testable with zero ROS/hardware; the
ROS/hardware coupling lives in thin adapters; the detector stays a standalone publisher.

---

## 2. State machine

States (superset of the requested pipeline, hardened):

| State | Meaning |
|---|---|
| `IDLE` | Framework up but not yet monitoring (or actuation disabled at boot until armed). |
| `MONITORING` | Normal operation; watching `/entanglement_state`. |
| `CONFIRMING` | A detection arrived; waiting for it to **persist** ≥ `confirmation_time_s` (rejects noise/oscillation/false alarms) — this is the "VERIFY DETECTION" step. |
| `STOPPING` | Issued `StopMove`; waiting for the robot to actually stop (robot-state or timeout) — "STOP ROBOT". |
| `RECOVERING` | Issued `BalanceStand` (or `RecoveryStand` if fallen); waiting for the stance to settle — "RECOVERY". |
| `VERIFYING` | Recovery done; require the detector to read **clear** for ≥ `verification_duration_s` — "VERIFY SUCCESS". |
| `RESUMING` | Hold a safe stance for `resume_delay_s`, then hand control back — "RESUME WALKING". |
| `COOLDOWN` | Ignore new triggers for `cooldown_s` (prevents thrash / repeated re-entry). |
| `FAULT` | Command failure / retries exhausted / cannot verify. Holds a safe stance (optionally `Damp`); needs reset. |
| `ESTOP` | Soft emergency stop (`Damp`, limp joints). Terminal until explicit reset. |

```
IDLE ──arm──▶ MONITORING
MONITORING ──entangled──▶ CONFIRMING
CONFIRMING ──cleared (false alarm)──▶ MONITORING
CONFIRMING ──sustained ≥ confirmation_time──▶ STOPPING        [Cmd: STOP_MOVE]
STOPPING   ──stopped (mode≠locomotion | resp.code==0 | settle)──▶ RECOVERING [Cmd: BALANCE_STAND*]
RECOVERING ──stance settled (mode∈{balanceStand,idle} | settle)──▶ VERIFYING
VERIFYING  ──detector clear ≥ verification_duration──▶ RESUMING
VERIFYING  ──still entangled past verify_timeout──▶ RECOVERING (retry, ≤ retry_limit) ─exhausted▶ FAULT
RESUMING   ──resume_delay elapsed──▶ COOLDOWN
COOLDOWN   ──cooldown_s elapsed──▶ MONITORING

* BALANCE_STAND normally; escalates to RECOVERY_STAND if a FALLEN posture is detected.

ANY active state ──command fail × retry_limit | state watchdog──▶ FAULT   [Cmd: DAMP if configured]
ANY state ──estop request | fall during non-recovery──▶ ESTOP            [Cmd: DAMP]
FAULT/ESTOP ──operator reset (or auto_reset_s)──▶ IDLE
```

A textual + Graphviz diagram is in `docs/diagrams/recovery_state_machine.dot` and the flow in
`docs/diagrams/recovery_flow.md`.

---

## 3. Transitions & guards (robustness requirements)

| Hazard | How the FSM handles it |
|---|---|
| **Duplicate detections** | Detections only *start* a cycle from `MONITORING`. In any active state they update the "latest detection" but never restart the cycle. |
| **Repeated / oscillating detections** | `CONFIRMING` requires the alarm to persist `confirmation_time_s`; a clear during confirmation returns to `MONITORING` as a false alarm. `COOLDOWN` after each cycle blocks rapid re-triggering. |
| **Noisy predictions** | Confirmation gate + the detector's own debounce; FSM acts on the boolean `entangled` (already debounced) plus `confidence`. |
| **Recovery re-entered while active** | Structurally impossible: the only entry to `STOPPING` is `CONFIRMING→STOPPING`, reachable only from `MONITORING`. Active states ignore new triggers. |
| **API failure / non-zero return code** | Each command has a `command_timeout_s`; on failure it retries up to `retry_limit`, then → `FAULT`. |
| **Communication failure** | If `/sportmodestate` is stale > `robot_state_timeout_s`, the FSM falls back to **timeouts** (never blocks forever) and, if it cannot verify recovery, conservatively → `FAULT` (safe hold) rather than falsely "resume". |
| **Robot already stopped** | `STOPPING` checks `mode`; if already not-locomotion, the stop is treated as satisfied immediately (no redundant command spam). |
| **Robot already sitting / fallen** | If `RobotState` is `FALLEN`/`SITTING` at recovery time, the FSM escalates to `RECOVERY_STAND` (the correct primitive for a non-upright body) instead of `BALANCE_STAND`. |
| **Robot already recovering** | If `mode == recoveryStand(8)` the FSM waits for it to finish rather than issuing another command. |
| **Per-state hang** | A `max_state_time_s` watchdog forces any active state to `FAULT` if it never completes. |

---

## 4. Recovery strategy (API selection rationale)

> **Superseded for `RECOVERING`:** this section's "Sequence A only / no `Move`" describes the base
> framework. The current package runs a detector-aware **strategy ladder** in `RECOVERING`
> (`balance_stand` first — i.e. Sequence A is still the gentlest rung — then `weight_shift`, then
> optional Move-based `small_reverse`/`small_sidestep`/`rotate` gated by confidence & intensity,
> then `emergency_stop`). The rationale below for *why BalanceStand is the safe first action* and
> *why RecoveryStand is escalation-only* still holds. See `INTELLIGENT_RECOVERY.md`.

For an **entanglement the robot is still upright** (not a fall), the gentlest first action is
**Sequence A**: `StopMove (1003)` → settle → `BalanceStand (1002)`.

| Option | Sequence | Decision |
|---|---|---|
| **A (chosen)** | StopMove → BalanceStand | Minimal, low-energy, stays upright. `StopMove` cancels the gait that is stepping into the snag; `BalanceStand` re-engages the actively-balanced upright stance (auto CoG) — the correct state for an already-upright robot. |
| B | StopMove → RecoveryStand → BalanceStand | **Avoided for upright.** `RecoveryStand (1006)` is documented as righting the body **from a fallen state (face-up/down)** — issuing it while upright risks a violent self-right against the snag. Used **only on escalation** when a fall is confirmed. |
| C | StopMove → Sit → StandUp → BalanceStand | Rejected: high-energy down/up cycle is pointless if posture was never lost, and the manual warns against strenuous posture changes on poor footing (likely during entanglement). |

**Escalation rule:** if `RobotState` becomes `FALLEN` (`mode == lieDown(5)`, or `|roll|`/`|pitch|`
beyond `tip_*_deg`), the upright assumption is void → use `RecoveryStand (1006)` (gated by
clearance/battery). Optional pre-step `Damp (1001)` releases torque if the gait is actively
dragging the snagged leg before re-engaging `BalanceStand`.

**Resume policy:** the framework does **not** command `Move` to auto-walk. After `BalanceStand`
it hands a stable, upright robot back to the operator/higher-level planner. This is deliberate
(safety): autonomous re-walking into the same hazard is out of scope. Configurable hook exists
for future strategies (§13).

---

## 5. Safety rules

- **Actuation gate (`enable_actuation`, default `false`):** in dry-run the SportClient publishes
  nothing — only logs intended `(api_id, name)`. Production must explicitly set it true.
- **`Damp (1001)` as soft e-stop:** available from any state via `/recovery_estop` (or on fatal
  fault) — releases joint torque to a safe limp state.
- **Return-code checking:** every command result is read from `/api/sport/response`
  (`header.status.code == 0` ⇒ success; matched by `header.identity.api_id`). Non-zero ⇒ retry/fault.
- **No command spamming:** exactly one command is emitted per transition; the node never re-sends
  while awaiting a response (within `command_timeout_s`). Recovery is never re-entered while active.
- **Environment/battery guards:** `RecoveryStand` escalation is gated on `bms_state.soc ≥ min_soc_pct`
  (from `LowState`) and on the configured assumption of clearance.
- **Conservative on uncertainty:** missing telemetry or unverifiable recovery → `FAULT` (hold),
  never an optimistic "resume".

---

## 6. Failure modes & handling

| Failure | Detection | Response |
|---|---|---|
| Command rejected (non-zero code) | `/api/sport/response` code≠0 within timeout | retry ≤ `retry_limit`, else `FAULT` |
| Command lost / no response | no matching response within `command_timeout_s` | retry, else `FAULT` |
| `/sportmodestate` lost | last-msg age > `robot_state_timeout_s` | timeout-driven transitions; if recovery unverifiable → `FAULT` |
| Detector silent | no `/entanglement_state` (handled implicitly) | stays `MONITORING` (no alarm = no action) |
| Robot tips / falls mid-recovery | `mode`/`rpy` → FALLEN | escalate to `RECOVERY_STAND` |
| Recovery doesn't clear detection | `VERIFYING` exceeds `verify_timeout` still entangled | re-attempt (≤ retry_limit) → `FAULT` |
| Operator e-stop | `/recovery_estop` | `ESTOP` + `Damp` |

---

## 7. Timeout handling

All timers are config-driven (no magic numbers). `monotonic` time; every active state has the
`max_state_time_s` watchdog as a backstop. Defaults (tunable, §8): `confirmation_time_s 0.5`,
`command_timeout_s 1.0`, `stop_settle_s 0.5`, `recovery_settle_s 2.0`, `verification_duration_s 1.5`,
`resume_delay_s 1.0`, `cooldown_s 5.0`, `retry_limit 2`, `robot_state_timeout_s 0.5`,
`max_state_time_s 10.0`.

---

## 8. Configuration

All parameters live in `config/recovery.yaml` (ROS 2 params). See `docs/RECOVERY_CONFIG.md` for
the annotated list. No magic numbers in code; every timer/threshold/topic is a parameter.

---

## 9. Expected robot behavior (nominal)

Walking → detector raises `entangled` → (≥0.5 s sustained) → robot **stops in place** → settles
into a **balanced stand** → detector reads clear for ≥1.5 s → robot **holds, ready** (control
handed back) → 5 s cooldown → resume monitoring. If a fall is detected at any point → **RecoveryStand**.
Any API failure → **safe hold (Damp)** and FAULT for operator attention. In dry-run, identical
state flow with commands **logged, not sent**.

---

## 10. Assumptions (unverified — treat as design assumptions)

1. No official rule mandates `StopMove` before a high-level mode switch; we do it as prudent
   practice (the only documented "stop first" rule is `ReleaseMode` before **low-level** control).
2. No documented universal inter-command delay between high-level sport commands; settle times are
   engineering defaults to tune on hardware.
3. `SportModeState.mode` mapping (0 idle,1 balanceStand,3 locomotion,5 lieDown,7 damping,8
   recoveryStand,10 sit) is from the README; treated as authoritative but firmware-version-sensitive.
4. DDS topic names (`/api/sport/request`, `/sportmodestate`, `rt/` prefixes) are standard but can
   vary with firmware/network config — all topics are configurable.
5. The Unitree support SPA pages could not be machine-read; command *semantics* come from the SDK
   headers + README + npaka mirror of the official doc.
6. `foot_force[4]` leg ordering and `progress` field are conventions, not guaranteed contracts.

## 11. Limitations

- Observe-only by default; real-robot actuation is unvalidated here (no hardware in CI). Hardware
  bring-up must follow `docs/RECOVERY_TESTING.md` starting in dry-run.
- High-level recovery only (Sport API). The low-level per-joint swing-leg "underbrush" recovery
  (paper/MBO) is intentionally **out of scope** — it needs `ReleaseMode` low-level control and is
  far riskier; this framework is the safe, supported layer.
- Does not auto-resume locomotion (by design).
- `RecoveryStand` escalation is best-effort; true fall handling on a snagged leg may still need
  manual intervention.

## 12. File map (deliverables)

```
robot_package/src/entanglement_recovery/
  package.xml  setup.py  setup.cfg  resource/entanglement_recovery
  config/recovery.yaml
  launch/recovery.launch.py
  launch/detector_and_recovery.launch.py
  entanglement_recovery/
    __init__.py
    states.py            # State/Command/Posture enums + Detection/RobotState/RecoveryContext/MotionStep/MotionPlan (pure)
    recovery_fsm.py      # pure-Python FSM (no ROS) — the safety-critical logic; RECOVERING -> StrategyManager
    strategies.py        # 7 pure strategies + MotionPlan builders (intelligent layer)
    strategy_manager.py  # detector-aware strategy ordering policy (intelligent layer)
    plan_runner.py       # executes ONESHOT/STREAM/HOLD MotionSteps over ticks (intelligent layer)
    sport_api.py         # verified Sport API ID constants + topic names + mode map (pure)
    sport_client.py      # ROS: command -> Request publish + response-code tracking + dry-run gate
    robot_state.py       # ROS: /sportmodestate(+/lowstate, BEST_EFFORT) -> RobotState/Posture
    recovery_node.py     # ROS orchestrator
  test/test_recovery_fsm.py    # 16 tests — pure FSM (all §3/§9 scenarios + strategy ladder)
  test/test_plan_runner.py     # 3 tests — PlanRunner step execution
docs/RECOVERY_DESIGN.md  docs/RECOVERY_TESTING.md  docs/RECOVERY_CONFIG.md  docs/INTELLIGENT_RECOVERY.md
docs/diagrams/recovery_state_machine.dot  docs/diagrams/recovery_flow.md
docs/diagrams/intelligent_recovery_architecture.md  docs/diagrams/recovery_strategy_flow.md  docs/diagrams/strategy_selection_flowchart.md
```

## 13. Future extensions

- Pluggable recovery strategies (Strategy pattern) — e.g., add a "gentle nudge-out" using a brief
  streamed `Move` against the snag, or the low-level MBO swing-leg recovery once validated.
- Per-leg recovery informed by `alarm_leg` / per-leg intensity (e.g., shift weight off the snagged leg).
- Use `motion_switcher` to confirm/select the `normal` sport service before commanding.
- Battery/clearance-aware escalation policies; ROS 2 lifecycle node; action-server interface.
