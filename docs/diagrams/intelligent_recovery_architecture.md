# Intelligent recovery — architecture

```
 entanglement_detector (UNCHANGED) ──/entanglement_state──┐
                                                          ▼
 /sportmodestate, /lowstate ─────▶ RobotStateMonitor ─┐  recovery_node (orchestrator)
                                                       │   ┌───────────────────────────────────┐
                                                       └──▶│ RecoveryFSM (pure)                 │
                                                           │   states unchanged; RECOVERING ──▶ │
                                                           │   StrategyManager.order(context)   │
                                                           │     context = {alarm_leg,          │
                                                           │       confidence, intensity, fallen}│
                                                           │   current_plan = Strategy.plan(...) │
                                                           └──────────────┬─────────────────────┘
                                                                          │ MotionPlan + Command
                                                          ┌───────────────▼───────────────┐
                                                          │ PlanRunner (executes steps)    │
                                                          │  ONESHOT / STREAM(Move) / HOLD │
                                                          └───────────────┬────────────────┘
                                                                          │ SportClient.send_api
 /api/sport/request  ◀── unitree_api/Request (api_id+param, DRY-RUN gated)─┘
 /api/sport/response ─── code==0 ──▶ PlanRunner.pop_error
 /recovery_status (std_msgs/String JSON)  ◀── diagnostics (state, strategy i/N, last cmd)
 /recovery_estop, /recovery_reset (std_msgs/Empty)
```

Pure / testable (no ROS): `recovery_fsm`, `strategies`, `strategy_manager`, `plan_runner`, `states`,
`sport_api`. ROS adapters: `sport_client`, `robot_state`, `recovery_node`. Detector + interfaces:
unchanged.
```
```
