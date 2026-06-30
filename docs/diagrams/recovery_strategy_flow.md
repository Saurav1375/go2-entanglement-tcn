# Recovery strategy closed-loop (the redesigned RECOVERING↔VERIFYING)

```
 CONFIRM ──sustained──▶ STOP (StopMove) ──▶ begin cycle: order = manager.order(context)
                                                  │  idx = 0
                                                  ▼
                        ┌────────────────── RECOVERING(order[idx]) ──────────────────┐
                        │  build verified MotionPlan(strategy, context, cfg)         │
                        │  PlanRunner executes it (one-shot / streamed Move / hold)  │
                        │  API error × retries ─────────────────────────▶ FAULT(Damp)│
                        └───────────────────────────┬────────────────────────────────┘
                                          plan executed (acked + settled)
                                                     ▼
                        ┌────────────────────── VERIFYING (detector) ────────────────┐
                        │  entangled clear ≥ verification_duration ─▶ RESUMING ─▶ ... │
                        │  still entangled > verify_timeout:                          │
                        │       idx += 1                                              │
                        │       idx < len(order) ──▶ RECOVERING(order[idx])  (loop)   │
                        │       else ──────────────▶ FAULT                            │
                        └─────────────────────────────────────────────────────────────┘

 Example order (upright, high-confidence, low-intensity, alarm_leg=FR):
   balance_stand → weight_shift → small_reverse(vx<0) → small_sidestep(vy>0) → rotate → [emergency_stop→FAULT]
 Example order (fallen):            recovery_stand → [emergency_stop→FAULT]
 Example order (high intensity):    balance_stand → weight_shift → [emergency_stop→FAULT]   (conservative)
```

The detector decides success after every motion (perception-driven); strategies escalate gentle →
active → safe-stop. `emergency_stop` (StopMove+Damp) is the terminal upright action → FAULT/operator.
