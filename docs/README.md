# Documentation

Documentation for the GO2 leg-entanglement **detector** and **recovery**. Start with the repo
[root README](../README.md) for the overview; this folder holds the detailed reports and designs.

## Recovery
- [recovery/RECOVERY.md](recovery/RECOVERY.md) — the recovery mechanism: the one-shot sequence
  (stop → move back → stop → front jump → stop), the Unitree SDK2 subprocess, configuration, and
  safety. The package quick-start is
  [`robot_package/src/entanglement_recovery/README.md`](../robot_package/src/entanglement_recovery/README.md).

## Detection
- [detection/RETRAIN_V2_REPORT.md](detection/RETRAIN_V2_REPORT.md) — **current** model: the v2
  retrain, the four GO2 field fixes, and before/after metrics.
- [detection/VALIDATION_REPORT_V2.md](detection/VALIDATION_REPORT_V2.md) — independent re-validation
  of v2 (generalization vs memorization).
- [detection/REPORT.md](detection/REPORT.md) — *(v1, superseded)* baseline detector verification.
- [detection/IMPROVEMENTS.md](detection/IMPROVEMENTS.md) — *(v1)* reliability/calibration study.
- [detection/PLAN.md](detection/PLAN.md) — the original ML design plan.
- [detection/summary.txt](detection/summary.txt) — short running notes.

## Overview & figures
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — plain-language project overview (also
  `project_overview.pdf`).
- [diagrams/architecture/](diagrams/architecture/) — publication-quality detector architecture
  figures (high-level + detailed; SVG/DOT/Mermaid + PNG/PDF).
- [diagrams/overview/](diagrams/overview/) — simple overview visuals (flow, results, system in
  action).

## Deployment (in `robot_package/`)
- [RUNBOOK.md](../robot_package/RUNBOOK.md) — the single hardware procedure (deploy → detector-only →
  recovery bring-up).
- [robot_package README](../robot_package/README.md) — deployment-package overview.
- [SETUP_GO2.md](../robot_package/SETUP_GO2.md) — detector-only setup detail + systemd autostart.
