# Paper Outline + Evidence Map (ARS Phase 2)

**Title (working):** Proprioceptive Leg-Entanglement Detection for a Quadruped: A Multi-Task
Causal TCN with Per-Leg Attribution and Physics-Grounded Severity.

**Venue:** IEEE conference (two-column). **Scope:** detection only (recovery excluded).

| § | Section | Key evidence (all in FACT_SHEET.md) | Figures / Tables |
|---|---------|-------------------------------------|------------------|
| — | Abstract + Keywords | contributions; F1 0.809; LORO 0.807±0.142; per-leg + severity; ~1 ms | — |
| I | Introduction | problem, why proprioception, why hard (subtle rear-thigh signature, Lock confusion), contributions | fig:system |
| II | Related Work | Yim et al. IROS 2023 (arXiv:2304.02129) anchor; proprioceptive legged learning; TCN; calibration | — |
| III-A | Methodology: Data Collection | 25 rec, 51 col, ~410–500 Hz, labels, 15/10, split 16/4/5, per-leg counts | fig:data, tab:dataset, fig:dataset |
| III-B | Methodology: Feature Engineering | resample 500 Hz, 60 ch, 3 engineered (formulas), z-score, window 200/0.40 s, hop 25/1, labeling rule | fig:feat, tab:features |
| III-C | Methodology: ML Pipeline | causal TCN, stem, 5 blocks, dilations, RF 127/254 ms, last-step, 3 heads, 136k, loss (3:1 Huber), sampler, AdamW, severity index g·√(d·r) | fig:tcn, tab:hparams |
| III-D | Methodology: Runtime | ROS2 node, BEST_EFFORT, ring buffer, ONNX, sigmoid+temperature, blend, threshold+debounce+per-leg, stationarity gate | fig:runtime, fig:e2e, tab:oppoint |
| IV | Experimental Setup | leakage-safe split + LORO protocol, metrics defs, baseline, ablation design, latency measurement, calibration protocol | — |
| V | Results | detection (TCN vs heuristic), confusion, ROC/PR, LORO, per-leg, calibration (honest), ablation, deployment-robustness, latency, runtime≡research | tab:detection, fig:confusion, fig:rocpr, fig:loro, fig:perleg (+tab:perleg), fig:reliability, tab:ablation (+fig:ablation), fig:field, fig:latency (+tab:runtime) |
| VI | Discussion | why TCN, causal readout intuition, FR precision trade-off, data-centric robustness, calibration nuance | — |
| VII | Limitations | offline-only, 15 events / FR under-represented, no severity GT, ECE not improved, no GBM/stat-detector numbers | — |
| VIII | Future Work | more FR/front data, on-robot field study, supervised severity, leaner channels | — |
| IX | Conclusion | recap contributions + honest scope | — |
| — | References | ~13–15 IEEE, all verified | — |

## Missing / uninferable details (cannot be claimed)
1. **Training/validation loss curves** — not logged; no convergence-curve figure.
2. **GBM & statistical-detector metrics** — code exists but no numbers persisted under this protocol; only the rule-based thigh-torque heuristic is reported as baseline.
3. **On-robot live detection metrics** — all metrics are offline CSV replay; latency is a dev-CPU benchmark of the deployment runtime, not the Go2 onboard computer.
4. **Ground-truth severity** — none exists; severity validated only qualitatively + as a calibrated index.
5. **Calibration** — temperature scaling de-saturates + slightly improves Brier but ECE worsened (0.134→0.156); cannot claim improved reliability.
6. **Sample-rate upper bound** — measured ~502 Hz max; the summary's "960 Hz" is unsupported and is not used.
