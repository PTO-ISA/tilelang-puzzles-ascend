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

.PHONY: all clean check marp
all: $(NAME).pdf

# The Marp build, for comparing presentation styles. Needs node >= 18 and a
# Chrome binary; marp-cli ships puppeteer-CORE, so it supplies no browser
# itself. `--no-sandbox` is required running as root in a container.
# slides.md is the same file the Beamer build uses -- its Marp front matter
# (theme + CSS) is what tools/slides_strip_marp.py removes for pandoc.
MARP_ENV := PATH=/opt/node/bin:$$PATH CHROME_PATH=/usr/bin/google-chrome
marp: slides.md
	$(MARP_ENV) marp slides.md --pdf --allow-local-files --no-sandbox \
	    -o $(NAME)_marp.pdf
	@echo "built $(NAME)_marp.pdf -- $$(pdfinfo $(NAME)_marp.pdf | awk '/^Pages/{print $$2}') pages"

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
	rm -rf build $(NAME).pdf $(NAME)_marp.pdf
