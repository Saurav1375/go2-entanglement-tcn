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
	cd figures && for f in fig1_system fig2_data fig3_features fig4_tcn fig5_runtime fig6_endtoend; do \
	  rsvg-convert -f pdf -o $$f.pdf $$f.svg; done
	cd .. && python3 paper_detection/figures/make_plots.py

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out
