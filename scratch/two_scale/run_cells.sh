#!/bin/bash
# The cell-by-cell runs of the two-scale check (full and 0.01x contrast, 4->8 and 6->12).
cd "$(dirname "$0")/../.."
python -u scripts/derive_two_scale_galerkin.py 4 6 > scratch/two_scale/graded_p1_cells.txt 2>&1
python -u scripts/derive_two_scale_galerkin.py --scale=0.01 4 6 > scratch/two_scale/graded_p1_cells_scale0.01.txt 2>&1
