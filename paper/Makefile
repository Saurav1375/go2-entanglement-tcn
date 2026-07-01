# Build the paper. Default target uses pdflatex+bibtex (standard TeX Live).
# `make tectonic` uses the self-contained tectonic engine.
PAPER = paper

.PHONY: all tectonic figures clean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	pdflatex -interaction=nonstopmode $(PAPER)
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER)

tectonic:
	tectonic -X compile $(PAPER).tex

figures:
	cd .. && python3 paper/figures/make_figures.py

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out
