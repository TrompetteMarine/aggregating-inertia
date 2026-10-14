#!/bin/bash
# Sequential driver (one process at a time; the machine is shared).  Logs go to results/*.log
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
set -x
python3 -B run_grid.py 0.0                              > results/grid_lam0.0.log 2>&1
python3 -B run_validation.py                            > results/validation.log 2>&1
python3 -B run_refine.py 1.0 0.7 0.6 0.7136347625525    > results/refine_benchmark.log 2>&1
python3 -B run_refine.py 2.0 0.35 5.0 1.0               > results/refine_lam2_th0.35_eta5.log 2>&1
python3 -B run_refine.py 0.25 1.0 5.0 1.0 quick         > results/refine_lam0.25_th1_eta5.log 2>&1
python3 -B run_refine.py 0.0 0.7 2.0 1.0 quick          > results/refine_lam0_th0.7_eta2.log 2>&1
python3 -B run_claim1.py                                > results/claim1.log 2>&1
python3 -B run_cells.py 2.0 0.35 5.0 1.0                > results/cells_lam2_th0.35_eta5.log 2>&1
python3 -B run_cells.py 1.0 0.7 0.6 0.7136347625525     > results/cells_benchmark.log 2>&1
python3 -B run_mc.py 2.0 0.35 5.0 1.0 200000 0.002,0.004 > results/mc_lam2_th0.35_eta5.log 2>&1
python3 -B run_mc.py 1.0 0.7 0.6 0.7136347625525 200000 0.002 > results/mc_benchmark.log 2>&1
echo ALLDONE
