# Paper: Proprioceptive Leg-Entanglement Detection (IEEE, detection-only)

Publication-quality IEEE two-column conference paper on the **entanglement detection** system in
this repository. The recovery framework is deliberately out of scope. Produced with the ARS pipeline;
every number is grounded in `FACT_SHEET.md`, which was verified against the code/config/artifacts.

## Contents
```
paper_detection/
├── paper.tex            # IEEE two-column source (IEEEtran, conference)
├── paper.pdf            # compiled paper
├── references.bib       # bibliography (web-verified)
├── IEEEtran.cls/.bst    # bundled (self-contained build)
├── Makefile             # build targets
├── FACT_SHEET.md        # verified ground truth (source of every number/claim)
├── OUTLINE.md           # outline + evidence map + gap list (pre-writing deliverable)
├── _floats.tex          # all figure/table float definitions (edited by assembly)
├── sections/            # per-section LaTeX drafts (assembled into paper.tex)
└── figures/
    ├── fig1_system.svg … fig6_endtoend.svg   # 6 editable architecture diagrams (Canva-ready)
    ├── *.pdf                                  # vector renders used by the paper
    ├── <result plots>.pdf                     # confusion/ROC/PR/reliability/LORO/…
    └── make_plots.py                          # regenerates result plots from ml/artifacts + a read-only model run
```

## Build
Self-contained via **tectonic** (auto-fetches packages, runs BibTeX):
```bash
tectonic -X compile paper.tex
```
Or standard TeX Live (`IEEEtran.cls/.bst` are bundled):
```bash
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```
`make` runs the pdflatex route; `make tectonic` uses tectonic; `make figures` re-renders the six
architecture SVGs to PDF and regenerates the result plots.

## Figures
The six architecture diagrams are **hand-authored SVGs** (white background, minimal color, simple
boxes/arrows, IEEE aesthetic) and are directly editable in Inkscape / draw.io / Canva. Result plots
are produced by `figures/make_plots.py` from `ml/artifacts/*.json` and a read-only run of the trained
model / ONNX runtime on the held-out test split.

## Scope of claims
All detection, calibration, ablation, and latency numbers are **offline** (recorded-log replay);
latency is a dev-CPU benchmark of the deployment runtime, not the Go2 onboard computer. Severity is a
calibrated physics index (no ground-truth severity labels exist). Temperature scaling de-saturates the
probabilities and slightly improves Brier but does not improve ECE. See `FACT_SHEET.md` §GAPS.
