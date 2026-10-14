# Reference pipeline (Python and C)

This folder keeps the pipeline that produced every number in the paper, unchanged, with its outputs:

- `code/`: the replication code of the paper. `make_data.py` writes the figure data and the macros in
  `tex/values.tex` and `tex/tipping_values.tex`; `make_tables.py` writes `tex/variants.tex` (Table S1.1) and
  `tex/validation_values.tex`; `checks/` holds the identity, perturbation and refinement checks and the mean-lag
  grid; `solvers/` holds the episode solver (`c9core.py`, `c9run.py`, `kernel.c`), the stationary solver
  (`fvsolver.py`, `analysis.py`, `march.c`), the one-dimensional solver (`ghf1d.py`) and the Monte Carlo kernel
  (`mc.c`).
- `research/`: the scripts of the research runs of 8 to 11 October 2026 that produced the grids and validation files
  in `data/grid` and `data/validation` (episode-solver grids, history-dependent variants, controls, the discrete form
  of Theorem 3, the Monte Carlo benchmark; stationary-solver scans, unimodality, refinement, the Monte Carlo kernel).
  They are kept as they were run; they import the solver modules by name, so run them with
  `PYTHONPATH=reference/code/solvers` after compiling the C kernels (`make -C reference`).
- `data/` and `tex/`: the outputs used in the paper.

The Rust engine in `rust/` is a port of this pipeline. `tests_ref/` compares the two.
