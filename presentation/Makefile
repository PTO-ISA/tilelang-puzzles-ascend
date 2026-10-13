# Build the deck from slides.md.
#
# slides.md is authored in MARP dialect (front matter + `---` separators +
# `##` slide titles) so that, once a Chrome binary is available,
#
#     marp slides.md --pdf --pptx
#
# produces the same deck with no edits. Today it is built with pandoc + Beamer,
# which needs no browser: strip the Marp front matter and separators, then let
# pandoc cut slides at `##`.

NAME    := quant_kernel_style_compare
PANDOC  := pandoc
LATEX   := pdflatex

.PHONY: all clean check
all: $(NAME).pdf

slides.md: ../tools/make_slides.py ../perf/quant_vf_cycles.json
	@python3 ../tools/make_slides.py

build/body.md: slides.md
	@mkdir -p build
	@python3 ../tools/slides_strip_marp.py $< $@

build/$(NAME).tex: build/body.md beamer-preamble.tex
	$(PANDOC) $< -t beamer --listings --slide-level=2 \
	    -V classoption:aspectratio=169 \
	    -V classoption:t \
	    -V fontsize=10pt \
	    -H beamer-preamble.tex \
	    -s -o $@

$(NAME).pdf: build/$(NAME).tex
	cd build && $(LATEX) -interaction=nonstopmode -halt-on-error $(NAME).tex >/dev/null
	cd build && $(LATEX) -interaction=nonstopmode -halt-on-error $(NAME).tex >/dev/null
	@cp build/$(NAME).pdf $@
	@echo "built $@ -- $$(pdfinfo $@ | awk '/^Pages/{print $$2}') pages"
	@! grep -q "Overfull" build/$(NAME).log || \
	    (echo "WARNING: overfull boxes:"; grep -c "Overfull" build/$(NAME).log)

check:
	@python3 ../tools/make_report_snippets.py --check
	@python3 ../tools/make_slides.py --check
	@python3 ../tools/check_report_numbers.py

clean:
	rm -rf build $(NAME).pdf
