# Recovery Framework — Testing Guide

Three tiers: (1) **pure FSM unit tests** (no ROS/hardware, run anywhere), (2) **dry-run on the
robot** (`enable_actuation:=false`, no motion), (3) **actuated bring-up** (only after 1 & 2 pass).
**Always start every hardware session in dry-run.**

## Tier 1 — FSM unit tests (CI-able, no hardware)

```bash
PYTHONPATH=robot_package/src/entanglement_recovery \
  python3 robot_package/src/entanglement_recovery/test/test_recovery_fsm.py   # 16/16 PASS
PYTHONPATH=robot_package/src/entanglement_recovery \
  python3 robot_package/src/entanglement_recovery/test/test_plan_runner.py    # 3/3 PASS
# or: colcon test --packages-select entanglement_recovery
```

`test_recovery_fsm.py` drives the pure `RecoveryFSM` with synthetic detection + telemetry +
command results and asserts the transitions below (incl. the intelligent strategy ladder /
escalation in `RECOVERING`); `test_plan_runner.py` covers the `PlanRunner` that executes
ONESHOT/STREAM/HOLD `MotionStep`s over ticks. They cover every scenario in the table.

## Tier 2 — Dry-run on the robot (no motion)

```bash
ros2 launch entanglement_recovery detector_and_recovery.launch.py        # enable_actuation:=false (default)
ros2 topic echo /recovery_status      # watch the state machine
ros2 topic echo /entanglement_state   # detector events
# trigger a synthetic detection without the robot:
ros2 topic pub --once /entanglement_state entanglement_interfaces/msg/EntanglementState "{entangled: true, confidence: 0.95, alarm_leg: 'RR'}"
# e-stop / reset:
ros2 topic pub --once /recovery_estop std_msgs/msg/Empty "{}"
ros2 topic pub --once /recovery_reset std_msgs/msg/Empty "{}"
```
In dry-run the node logs `[DRY-RUN] would send StopMove (api_id=1003)` etc. and walks the full
state sequence (commands auto-succeed) — verify the sequence and timing match expectations.

## Tier 3 — Actuated bring-up (hardware, with clearance)

Pre-flight: ≥2 m clearance, flat high-friction floor, battery > `min_soc_pct`, e-stop in hand,
sport service active. Then `... enable_actuation:=true`. Start with a gentle induced snag.

## Scenario matrix (expected behavior)

| # | Scenario | Inputs | Expected behavior |
|---|---|---|---|
| 1 | **Normal walking** | `entangled=false`, mode=locomotion | Stays `MONITORING`; no command. |
| 2 | **False alarm** | `entangled=true` for < `confirmation_time_s` then clears | `MONITORING→CONFIRMING→MONITORING`; **no** StopMove. |
| 3 | **Genuine entanglement** | sustained `entangled=true` | `CONFIRMING→STOPPING` (StopMove) `→RECOVERING` (BalanceStand) `→VERIFYING`→(clear)→`RESUMING→COOLDOWN→MONITORING`. |
| 4 | **Repeated alarms** | many `entangled=true` while active | Cycle proceeds once; new detections never restart it. |
| 5 | **Multiple alarms (back-to-back)** | new alarm right after a cycle | Blocked during `COOLDOWN` (cooldown_s); handled after. |
| 6 | **Robot already stopped** | entangled & mode≠locomotion (UPRIGHT) | **Skips** StopMove; goes straight to `RECOVERING`. |
| 7 | **Robot sitting / fallen** | entangled & mode=lieDown / tipped | Recovery uses **RecoveryStand (1006)**, not BalanceStand. |
| 8 | **Recovery interrupted (fall mid-recovery)** | posture→FALLEN during `RECOVERING` | Re-issues as **RecoveryStand**. |
| 9 | **API failure** | command code≠0 / no response | Retries ≤ `retry_limit`, then `FAULT` + **Damp**. |
| 10 | **Network / response delay** | response slower than `command_timeout_s` | Treated as a failed attempt → retry; if telemetry shows the action happened, completes via settle. |
| 11 | **Detector timeout / telemetry stale** | `/sportmodestate` stops | Transitions fall back to timeouts; if recovery unverifiable → `FAULT` (safe hold), never false "resume". |
| 12 | **Verification fails** | still `entangled` after recovery | Re-attempts ≤ `retry_limit`, then `FAULT`. |
| 13 | **E-stop** | publish `/recovery_estop` | Immediately `ESTOP` + **Damp** from any state. |
| 14 | **Reset** | publish `/recovery_reset` | Leaves `FAULT`/`ESTOP` → `MONITORING`. |

Scenarios 1–14 map to the Tier-1 tests (`test_recovery_fsm.py`): normal(1), false-alarm(2),
happy-path(3), not-reentered(4), already-stopped(6), fallen(7), api-failure(9), verify-failure(12),
estop(13), reset(14). Scenarios 5/8/10/11 are exercised in dry-run/hardware.

## Acceptance criteria
- Tier 1: 16/16 FSM tests + 3/3 PlanRunner tests pass.
- Tier 2: dry-run shows the correct command **sequence + IDs** for scenarios 1–14; no command spam.
- Tier 3: on an induced snag the robot stops then balance-stands without violent motion; e-stop
  (Damp) works instantly; no false recovery during normal walking/stops.
