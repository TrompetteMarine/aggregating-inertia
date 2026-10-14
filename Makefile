# Aggregating Inertia: build the engine, compute everything, draw the figures.
#   make engine     build the Rust engine (cargo)
#   make data       every computation of the paper -> output/ (about 70 minutes on two cores)
#   make figures    draw every figure from output/ -> figures/{light,dark}/ (Julia, Makie)
#   make all        data, then figures
#   make test       unit tests of the engine
#   make check      compare output/ with the reference pipeline's outputs in reference/
#   make crosscheck run the reference Python + C solvers and the Rust engine side by side (needs numpy, scipy, gcc)
#   make paper-figures  the six figures of the paper as printed, from output/ -> figures/paper/ (needs pdflatex, pgfplots)
ENGINE ?= rust/target/release/inertia
JULIA ?= julia
PYTHON ?= python3
OUT ?= output
FIGS ?= figures

.PHONY: all engine data figures test check crosscheck paper-figures

all: data figures

engine:
	cargo build --release --manifest-path rust/Cargo.toml

data:
	$(ENGINE) --out $(OUT) all

figures:
	$(JULIA) --project=julia julia/make_figures.jl $(OUT) $(FIGS)

test:
	cargo test --release --manifest-path rust/Cargo.toml --lib

check:
	$(PYTHON) tests_ref/compare_outputs.py $(OUT) reference

crosscheck: engine
	$(MAKE) -C reference
	$(PYTHON) tests_ref/compare_episode.py reference/code/solvers rust
	$(PYTHON) tests_ref/compare_stationary.py reference/code/solvers rust
	$(PYTHON) tests_ref/compare_ghf.py reference/code/solvers rust
	$(PYTHON) tests_ref/compare_mc.py reference/code/solvers rust

# The paper's figure sources read ../../data/*.csv and the macros in values.tex: compile them in a copy of the
# paper's layout, with data/ pointing at $(OUT)/data. Figures 2 and 3 are schematics (no data).
PAPER_FIGS = episode trajectory reinjection stability record magnitudes
paper-figures:
	rm -rf build/paper && mkdir -p build/paper/figures/tikz figures/paper
	cp paper/figures/*.tex $(OUT)/tex/values.tex $(OUT)/tex/tipping_values.tex build/paper/figures/tikz/
	ln -s "$(abspath $(OUT))/data" build/paper/data
	cd build/paper/figures/tikz && for f in $(PAPER_FIGS); do \
	  pdflatex -interaction=nonstopmode -halt-on-error $$f.tex > $$f.build.log || { tail -n 20 $$f.build.log; exit 1; }; \
	done
	for f in $(PAPER_FIGS); do cp build/paper/figures/tikz/$$f.pdf figures/paper/; done
