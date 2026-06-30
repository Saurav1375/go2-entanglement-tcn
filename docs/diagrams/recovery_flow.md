# Recovery flow (happy path + branches)

```
        ┌─────────────┐
        │  MONITORING │◀──────────────────────────────────────────────┐
        └──────┬──────┘                                                │
       entangled (conf ≥ min)                                          │
               ▼                                                       │
        ┌─────────────┐   cleared (false alarm)                        │
        │ CONFIRMING  │───────────────────────────────────────────────┤
        └──────┬──────┘                                                │
   sustained ≥ confirmation_time                                       │
        ├───────────────► already stopped? ──yes──┐                    │
        │ no (locomotion)                          │                    │
        ▼                                          │                    │
   [CMD StopMove] ┌─────────┐                      │                    │
        │         │STOPPING │                      │                    │
        └────────▶└────┬────┘                      │                    │
        StopMove ok & (not locomotion | settle)    │                    │
                       ▼                            ▼                    │
            [CMD BalanceStand]  ┌────────────┐  (skip StopMove)          │
                                │ RECOVERING │◀──────────────┐          │
                                └─────┬──────┘   fall ⇒ [CMD  │          │
            Stand ok & (upright|settle)│         RecoveryStand]          │
                                       ▼                                 │
                                ┌────────────┐  still entangled          │
                                │ VERIFYING  │──> verify_timeout ────────┘ (re-attempt ≤ limit)
                                └─────┬──────┘
            detector clear ≥ verification_duration
                                     ▼
                                ┌──────────┐   resume_delay   ┌──────────┐  cooldown
                                │ RESUMING │─────────────────▶│ COOLDOWN │──────────┐
                                └──────────┘                  └──────────┘          │
                                                                                    └─▶ MONITORING

  Anywhere:  command retries exhausted / watchdog ──▶ FAULT  ([CMD Damp], reset to recover)
             /recovery_estop ──▶ ESTOP ([CMD Damp])
```

Recovery sequence executed on the robot (Sequence A): **StopMove (1003) → BalanceStand (1002)**;
escalates to **RecoveryStand (1006)** only when a FALL is detected. **Damp (1001)** is the soft
e-stop used on FAULT/ESTOP. In dry-run (`enable_actuation:=false`) these are logged, not sent.
