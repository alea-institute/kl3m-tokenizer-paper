# Makefile for compiling the KL3M tokenizer paper
# Usage:
#   make        # compile the paper (quiet mode)
#   make VERBOSE=1 # compile with verbose output
#   make clean  # remove all temporary files
#   make rebuild # clean and rebuild the paper
#   make view   # open the compiled PDF
#   make figures # run the tokenizer analysis script to regenerate figures

# Configuration
PAPER = main
LATEXMK = latexmk
PDFLATEX = pdflatex
BIBTEX = bibtex
PYTHON = python3
VIEWER = xdg-open

# Set VERBOSE=1 to enable verbose output
VERBOSE ?= 0
ifeq ($(VERBOSE),1)
  SILENCE := 
  REDIRECT :=
  LATEXMK_OPTS := -pdf -use-make
  PDFLATEX_OPTS := -interaction=nonstopmode -shell-escape
  BIBTEX_OPTS := 
else
  SILENCE := @
  REDIRECT := > /dev/null 2>&1 || true
  LATEXMK_OPTS := -pdf -quiet -use-make
  PDFLATEX_OPTS := -interaction=nonstopmode -shell-escape -quiet
  BIBTEX_OPTS := -terse
endif

# Source directories
SRC_DIR = src
FIG_DIR = figures
SEC_DIR = sections
BIB_DIR = bibliography

# Source files
TEX_FILES = $(PAPER).tex $(wildcard $(SEC_DIR)/*.tex)
BIB_FILES = $(wildcard $(BIB_DIR)/*.bib)
FIG_FILES = $(wildcard $(FIG_DIR)/*.png)

# Output files
PDF_FILE = $(PAPER).pdf
AUX_FILE = $(PAPER).aux
BBL_FILE = $(PAPER).bbl

# Main targets
.PHONY: all clean view figures rebuild

all: $(PDF_FILE)

# Rebuild target that cleans and then builds
rebuild: clean all

# Generate the PDF using latexmk if available, otherwise use pdflatex+bibtex
$(PDF_FILE): $(TEX_FILES) $(BIB_FILES) $(FIG_FILES)
	$(SILENCE)echo "Compiling $(PAPER).tex..."
	$(SILENCE)if command -v $(LATEXMK) > /dev/null; then \
		$(LATEXMK) $(LATEXMK_OPTS) $(PAPER) $(REDIRECT); \
	else \
		$(PDFLATEX) $(PDFLATEX_OPTS) $(PAPER) $(REDIRECT) || { echo "pdflatex failed with errors:"; grep -A 2 -B 2 "error\|warning" $(PAPER).log || cat $(PAPER).log; exit 1; }; \
		$(BIBTEX) $(BIBTEX_OPTS) $(PAPER) $(REDIRECT) || { echo "bibtex failed with errors:"; cat $(PAPER).blg; exit 1; }; \
		$(PDFLATEX) $(PDFLATEX_OPTS) $(PAPER) $(REDIRECT) || { echo "pdflatex failed with errors:"; grep -A 2 -B 2 "error\|warning" $(PAPER).log || cat $(PAPER).log; exit 1; }; \
		$(PDFLATEX) $(PDFLATEX_OPTS) $(PAPER) $(REDIRECT) || { echo "pdflatex failed with errors:"; grep -A 2 -B 2 "error\|warning" $(PAPER).log || cat $(PAPER).log; exit 1; }; \
	fi
	$(SILENCE)echo "✓ Compilation complete: $(PDF_FILE)"

# Generate figures by running the tokenizer analysis script
figures:
	$(SILENCE)echo "Generating figures..."
	$(SILENCE)cd $(SRC_DIR) && $(PYTHON) tokenizer_analysis.py $(if $(filter 0,$(VERBOSE)),2>&1 | grep -i "error\|warning" || true)
	$(SILENCE)echo "✓ Figures generated in $(FIG_DIR)/"

# Open the PDF with the default viewer
view: $(PDF_FILE)
	$(SILENCE)$(VIEWER) $(PDF_FILE) 2>/dev/null || echo "Could not open PDF. Try installing a PDF viewer."

# Clean up temporary files
clean:
	$(SILENCE)echo "Cleaning..."
	$(SILENCE)if command -v $(LATEXMK) > /dev/null; then \
		$(LATEXMK) -quiet -C $(PAPER) $(REDIRECT); \
	else \
		rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).blg $(PAPER).bbl $(PAPER).toc $(REDIRECT); \
	fi
	$(SILENCE)rm -f $(PAPER).pdf $(REDIRECT)
	$(SILENCE)echo "✓ Cleanup complete"

# Show help
help:
	@echo "Usage:"
	@echo "  make                 | Compile the paper (shows only warnings/errors)"
	@echo "  make VERBOSE=1       | Compile with verbose output"
	@echo "  make clean           | Remove all temporary files"
	@echo "  make rebuild         | Clean and rebuild the paper (clean + build)"
	@echo "  make rebuild VERBOSE=1 | Clean and rebuild with verbose output"
	@echo "  make view            | Open the compiled PDF"
	@echo "  make figures         | Generate figures"
	@echo "  make figures VERBOSE=1 | Generate figures with verbose output"
	@echo "  make help            | Show this help"