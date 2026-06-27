# Pre-Publication Repository Review

Status after the publish-readiness pass. No ML model or results were changed — only
repository quality, structure, and documentation.

## ✅ Done
- **Cleaned**: removed `__pycache__/`, `*.pyc`, and `ml/artifacts/ablation.log`.
- **`.gitignore`** added — verified **0 generated/ignored files staged** (no `csv_normalized/`,
  `csv_labelled/`, `plots/`, `ml/artifacts/`, caches, `.claude/`).
- **`.gitattributes`** added — deterministic LF line endings; binaries marked.
- **`README.md`** — overview, motivation, dataset, architecture, pipeline, install, usage,
  training, evaluation, results, structure, future work, license, example commands.
- **`LICENSE`** — MIT (Copyright 2026 Saurav Gupta — *verify name/year*).
- **`requirements.txt`** — pinned minimums + tested versions.
- **Absolute paths removed** from code/docs (`config.py` docstring, `PLAN.md`); all paths are
  derived from `__file__` or are configurable in `ml/config.py`.
- **Renamed** `Statistical Detection/` → `statistical_detector/` (no spaces).
- **Runnable from project root**: every `ml/<module>` runs via `python -m ml.<module>`; root
  scripts (`normalize_timestamps.py`, `merge_labels.py`, `plot_thigh_torque.py`) run standalone.
  Verified the full fresh-clone path: prep scripts + 6 module self-tests all pass.
- **Docs** curated into `docs/` (`REPORT.md`, `IMPROVEMENTS.md`, `PLAN.md`).

## ⚠️ Decide before publishing (not blocking, but your call)
1. **Repository size ≈ 104 MB** — dominated by committed raw data in `csv/` (largest file 17 MB,
   under GitHub's 50/100 MB limits, so no LFS strictly required). Options:
   - Keep as-is (self-contained, reproducible). ✅ current choice.
   - Move `csv/` to **Git LFS** to keep the main repo light.
   - Host data externally and add a small download script; gitignore `csv/`.
2. **Data privacy** — confirm the GO2 recordings are cleared for public release.
3. **Trained model is git-ignored** (`ml/artifacts/`). A fresh clone must train before
   `evaluate`/`infer`. If you want **clone-and-infer** with no training, commit `model.pt` +
   the small `*.json` calib files (via LFS or a GitHub Release asset) and relax `.gitignore`.
4. **`docs/REPORT.md` & `docs/IMPROVEMENTS.md` are snapshots** of generated reports; the live
   copies regenerate into `ml/artifacts/` (ignored). They can drift if you re-run — treat the
   `docs/` copies as the published results of record.

## 🔧 Optional polish
- **Reference detector**: `statistical_detector/statistical_detection.py` (~1840 lines) contains
  3 concatenated iterations and expects the *old* `go2_lowstate_*` data layout, so it won't run
  in this repo as-is. It's included for provenance; consider trimming to the final version or
  adding a one-line note. The pipeline's faithful re-implementation is `ml/stat_detector.py`.
- **CI**: add a GitHub Actions workflow running the module self-tests
  (`python -m ml.model`, `ml.windowing`, `ml.calibration`, …) on push.
- **Packaging**: add a `pyproject.toml` if you want `pip install -e .` instead of `-m ml.*`.

## 📦 Suggested first commit
```bash
# from the project root (git already initialized; files staged)
git commit -m "Leg-entanglement detection for Unitree GO2 (multi-task causal TCN)"
git branch -M main
git remote add origin <your-github-url>
git push -u origin main
```
