# Strategy selection flowchart (StrategyManager.order)

```
                         ┌───────────────────────────┐
 detection context ─────▶│ posture == FALLEN ?        │
 {alarm_leg, confidence, │   (mode lieDown / tipped)  │
  intensity, fallen}     └─────┬───────────────┬──────┘
                            yes│               │no
                               ▼               ▼
                  [recovery_stand,    ┌──────────────────────────────┐
                   emergency_stop]    │ seq = [balance_stand,         │  (gentle, always)
                                      │        weight_shift]          │
                                      └──────────────┬────────────────┘
                                                     ▼
                          ┌──────────────────────────────────────────────┐
                          │ active_ok = confidence ≥ active_confidence_min│
                          │            AND intensity < high_intensity_thresh│
                          └───────────────┬───────────────┬────────────────┘
                                       yes│               │no  (low-conf OR high-intensity)
                                          ▼               │   = CONSERVATIVE: no active Move
              append leg-aware Move strategies:           │
                small_reverse (vx sign from front/rear)   │
                small_sidestep (vy sign from left/right)  │
                rotate (vyaw guess)                       │
                                          └───────┬───────┘
                                                  ▼
                                  append emergency_stop  (terminal → FAULT)

 Magnitudes scale by intensity:  magnitude = base · (1 − (1−intensity_min_scale)·intensity)
 Each strategy can be disabled via enable_<strategy> in recovery.yaml.
```

Leg → direction (design assumption): FRONT={FR,FL} REAR={RR,RL} LEFT={FL,RL} RIGHT={FR,RR}.
reverse vx<0 if front else >0; sidestep vy>0 if right-side else <0; weight-shift tilts away from
the snagged corner; rotate sign = right-side→+yaw else −yaw (chirality unobservable → a guess).
```
```
