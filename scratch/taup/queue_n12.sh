#!/bin/bash
# After the 8-across run, the same tau-p sweep at 12 cells across (convergence of the section).
cd "$(dirname "$0")/../.."
while pgrep -f "[t]aup_graded_sphere.py --p=1 --n=8" >/dev/null; do sleep 60; done
python -u scripts/taup_graded_sphere.py --p=1 --n=12 > scratch/taup/run_p1_n12.txt 2>&1
