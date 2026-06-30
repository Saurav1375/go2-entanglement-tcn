# Recovery Framework — Configuration Reference

All parameters live in `robot_package/src/entanglement_recovery/config/recovery.yaml` (ROS 2
params under node `entanglement_recovery`). No magic numbers exist in code. Override any value at
launch, e.g. `ros2 launch entanglement_recovery recovery.launch.py enable_actuation:=true`.

## Safety gate
| param | default | meaning |
|---|---|---|
| `enable_actuation` | **false** | `false` = DRY-RUN: FSM runs, commands are **logged not sent**. `true` = actuate the robot. **Validate in dry-run first.** |
| `auto_arm` | true | begin `MONITORING` at startup. |
| `control_rate_hz` | 50.0 | FSM tick rate. |

## Detection gating
| param | default | meaning |
|---|---|---|
| `confirmation_time_s` | 0.5 | sustained `entangled` required before acting (rejects noise/oscillation). |
| `confidence_min` | 0.0 | ignore detector events below this calibrated confidence. |

## Command execution
| param | default | meaning |
|---|---|---|
| `command_timeout_s` | 1.0 | wait for `/api/sport/response` before counting a failure. |
| `retry_limit` | 2 | retries per command (and re-attempts of recovery) before `FAULT`. |
| `stop_settle_s` | 0.5 | settle time after `StopMove` ack (used if telemetry doesn't confirm stop). |
| `recovery_settle_s` | 2.0 | settle time after `BalanceStand`/`RecoveryStand` ack. |

## Verification / resume
| param | default | meaning |
|---|---|---|
| `verification_duration_s` | 1.5 | detector must read **clear** this long to confirm success. |
| `verify_timeout_s` | 4.0 | still entangled this long in `VERIFYING` → re-attempt recovery. |
| `resume_delay_s` | 1.0 | hold a stable stance before handing control back. |
| `cooldown_s` | 5.0 | ignore new triggers after a cycle (anti-thrash). |

## Robustness / safety
| param | default | meaning |
|---|---|---|
| `robot_state_timeout_s` | 0.5 | `/sportmodestate` older than this is treated as stale (fall back to timeouts). |
| `max_state_time_s` | 10.0 | per-active-state watchdog → `FAULT`. |
| `auto_reset_s` | 0.0 | `0` = `FAULT` needs manual `/recovery_reset`; `>0` auto-clears after this long. |
| `escalate_to_recovery_stand` | true | use `RecoveryStand(1006)` when a FALL is detected (else stay with BalanceStand). |
| `use_damp_on_fault` | true | issue `Damp(1001)` on `FAULT`/`ESTOP` (soft e-stop). |
| `tip_roll_deg` / `tip_pitch_deg` | 50.0 | |roll|/|pitch| beyond → FALLEN (escalate). |
| `min_soc_pct` | 10.0 | gate `RecoveryStand` on battery % (from LowState). |

## Topics
| param | default | type |
|---|---|---|
| `entanglement_topic` | `/entanglement_state` | sub: `entanglement_interfaces/EntanglementState` |
| `status_topic` | `/recovery_status` | pub: `std_msgs/String` (JSON diagnostics) |
| `sport_request_topic` | `/api/sport/request` | pub: `unitree_api/Request` |
| `sport_response_topic` | `/api/sport/response` | sub: `unitree_api/Response` |
| `sportmode_topic` | `/sportmodestate` | sub: `unitree_go/SportModeState` |
| `lowstate_topic` | `/lowstate` | sub: `unitree_go/LowState` (battery) |
| `estop_topic` | `/recovery_estop` | sub: `std_msgs/Empty` → soft e-stop |
| `reset_topic` | `/recovery_reset` | sub: `std_msgs/Empty` → clear FAULT/ESTOP |

## Tuning notes
- `confirmation_time_s` ↑ → fewer false recoveries, slower reaction. Pair with the detector's own
  debounce (75 ms) and `confidence_min`.
- `recovery_settle_s` should exceed how long BalanceStand needs to settle on your firmware (tune in
  dry-run by watching `/sportmodestate.mode` reach `1` (balanceStand)).
- Inter-command delay / command rate are not officially specified (see RECOVERY_DESIGN.md §10);
  tune `*_settle_s` on hardware.
