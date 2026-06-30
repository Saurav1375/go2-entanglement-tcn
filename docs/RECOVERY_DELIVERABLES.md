# Recovery Framework — Deliverables, Migration & Summary

> **Status:** this documents the **base recovery framework** (StopMove→BalanceStand→escalate). The
> **intelligent-recovery layer** (now consolidated into the production package) added a pluggable
> **StrategyManager** + 7 strategies (`strategies.py`, `strategy_manager.py`) and a `PlanRunner`
> (`plan_runner.py`), and implemented several items listed under *Recommendations* below
> (Strategy pattern, leg/intensity-aware selection, SOC gating). See
> [`INTELLIGENT_RECOVERY.md`](INTELLIGENT_RECOVERY.md) for the current `RECOVERING`-state design.
> Test count is now **16 FSM + 3 PlanRunner** (not 10). All work is on **`main`**.

## New files

**ROS 2 package `robot_package/src/entanglement_recovery/`** (ament_python):
| file | purpose |
|---|---|
| `entanglement_recovery/sport_api.py` | Source-verified Sport API id constants + topic names + `SportModeState.mode` map (pure). |
| `entanglement_recovery/states.py` | `State`/`Command`/`Posture` enums + `Detection`/`RobotState`/`Diagnostics` dataclasses (pure). |
| `entanglement_recovery/recovery_fsm.py` | **The safety-critical FSM** — pure Python, deterministic, unit-tested. `RECOVERING` delegates to the StrategyManager. |
| `entanglement_recovery/strategies.py` | **(intelligent layer)** 7 pure strategies + `MotionPlan`/`MotionStep`, each mapped to verified Sport-API actions. |
| `entanglement_recovery/strategy_manager.py` | **(intelligent layer)** detector-aware ordering policy (leg/confidence/intensity/posture/SOC) for the `RECOVERING` ladder. |
| `entanglement_recovery/plan_runner.py` | **(intelligent layer)** executes ONESHOT/STREAM/HOLD `MotionStep`s over ticks. |
| `entanglement_recovery/sport_client.py` | ROS adapter: Command→Request publish on `/api/sport/request`, response-code tracking, **dry-run gate**. |
| `entanglement_recovery/robot_state.py` | ROS adapter: `/sportmodestate`(+`/lowstate`)→`RobotState`/`Posture` (BEST_EFFORT QoS). |
| `entanglement_recovery/recovery_node.py` | Orchestrator node: wires events+telemetry+FSM+client, publishes `/recovery_status`. |
| `config/recovery.yaml` | All parameters (no magic numbers). |
| `launch/recovery.launch.py`, `launch/detector_and_recovery.launch.py` | Launch recovery alone / detector+recovery. |
| `test/test_recovery_fsm.py` | 16 FSM unit tests (no ROS/hardware) covering the scenario matrix + strategy ladder. |
| `test/test_plan_runner.py` | 3 PlanRunner unit tests (step execution over ticks). |
| `package.xml`, `setup.py`, `setup.cfg`, `resource/entanglement_recovery` | ament_python packaging. |

**Docs (`docs/`):** `RECOVERY_DESIGN.md`, `RECOVERY_TESTING.md`, `RECOVERY_CONFIG.md`,
`diagrams/recovery_state_machine.dot`, `diagrams/recovery_flow.md`, this file.

## Modified files
**None of the detector / ML pipeline.** The detector (`entanglement_detector`, `entanglement_interfaces`,
`ml/`) is unchanged — recovery is a new, independent package that only *subscribes* to the
detector's existing `/entanglement_state`. Verified: `git status` shows no changes to those paths.

## Why each change was necessary
- **Separate package, not edits to the detector** → satisfies "no regression / detector stays
  independent". Recovery depends on `entanglement_interfaces` only as a consumer.
- **Pure FSM split from ROS adapters** → the safety logic is testable without a robot (16/16 FSM +
  3/3 PlanRunner tests) and the hardware coupling is isolated/replaceable.
- **Verified Sport API ids + ROS contract** → commands use the official `/api/sport/request`
  interface with correct ids (StopMove 1003, BalanceStand 1002, RecoveryStand 1006, Damp 1001) and
  read return codes from `/api/sport/response`.
- **Dry-run actuation gate (default off)** → safe to deploy/validate without moving a robot; a
  hard, explicit opt-in is required to actuate.
- **Robot-state awareness (SportModeState.mode/rpy)** → transitions use telemetry (stopped /
  upright / fallen / recovering) instead of timers where possible, with timeout fallbacks.
- **Cooldown / confirmation / retry / watchdog / Damp e-stop** → robustness against the failure
  modes in the brief (duplicate/oscillating detections, API/comm failure, already-stopped/sitting,
  never re-enter recovery while active).

## Migration notes (deploying recovery on the GO2)
1. Copy `robot_package/src/entanglement_recovery` into the same colcon workspace `src/` as the
   detector (it needs `entanglement_interfaces` already there).
2. `colcon build --packages-select entanglement_interfaces entanglement_detector entanglement_recovery`
   then `source install/setup.bash`.
3. Ensure `unitree_api` and `unitree_go` messages are on the `AMENT_PREFIX_PATH` (they ship with
   the GO2 ROS 2 stack).
4. Run **dry-run first**: `ros2 launch entanglement_recovery detector_and_recovery.launch.py`
   and watch `/recovery_status`. Only then `... enable_actuation:=true` with clearance (see
   RECOVERY_TESTING.md).
5. No detector reconfiguration is required; recovery is purely additive.

## Remaining limitations
- Observe-only by default; **actuated behavior is unvalidated on hardware here** (no robot in CI).
- High-level Sport-API recovery only; low-level per-joint "underbrush" recovery is out of scope
  (needs `ReleaseMode`, far riskier).
- Does not auto-resume locomotion (by design — control is handed back upright/stable).
- Several Unitree timing facts (inter-command delay, StopMove-before-switch necessity) are
  unverified assumptions (RECOVERY_DESIGN.md §10) — tune `*_settle_s` on hardware.
- `RecoveryStand` escalation is best-effort; a badly snagged leg may still need manual help.

## Recommendations for future improvements
- Pluggable strategy interface (Strategy pattern): add a brief streamed-`Move` "nudge-out" or the
  validated MBO swing-leg recovery as alternative strategies.
- Use `alarm_leg` + per-leg intensity for leg-specific recovery (e.g. weight-shift off the snag).
- Confirm/select the `normal` sport service via `motion_switcher` before commanding; gate on
  `bms_state.soc` and obstacle range from `SportModeState.range_obstacle`.
- Convert to a ROS 2 lifecycle node and/or an action server; add hardware-in-the-loop tests.
