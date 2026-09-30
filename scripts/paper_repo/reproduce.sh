#!/usr/bin/env bash
# Reproduce the papers' numbers. Run from the repository root inside the conda environment.
#   ./reproduce.sh quick   the Python cross-checks and both figures, from the saved data (minutes)
#   ./reproduce.sh full    also the long runs listed in README.md (hours; see the timings there)
set -euo pipefail
tier="${1:-quick}"
t0=$(date +%s)
run() { echo; echo "== $*"; python -u "$@"; }

run scripts/gate_cube_shear_split.py
run scripts/gate_sphere_closure_vs_mie.py
run scripts/gate_heterogeneous_reference_vs_kennett.py
run scripts/crosscheck_first_moment_voxel.py
run scripts/crosscheck_second_moment_voxel.py
run scripts/crosscheck_graded_contrast.py
run scripts/crosscheck_graded_sphere.py
run scripts/plot_convergence_orders.py
run scripts/plot_graded_voxel_orders.py

if [ "$tier" = "full" ]; then
  run scripts/pilot_sphere_voxel_vs_mie.py 4 6 8
  run scripts/pilot_graded_voxel_sphere.py --ka=0.5 --arms=g0fft 4 6 8 12 16 20 24
  run scripts/pilot_graded_voxel_sphere.py --ka=1.0 --arms=g0fft 4 6 8 12 16 20 24
  run scripts/pilot_graded_voxel_sphere.py --weak --ka=0.5 --arms=t9,g0,g1 4 6 8
  run scripts/pilot_graded_voxel_sphere.py --weak --ka=1.0 --arms=t9,g0,g1 4 6 8
  run scripts/pilot_graded_voxel_sphere.py --ka=0.5 --arms=t9,g0,g1fft 4 6 8 10 12 14 16
  run scripts/pilot_graded_voxel_sphere.py --ka=1.0 --arms=t9,g0,g1fft 4 6 8 10 12 14 16
  run scripts/measure_graded_voxel_t2.py
  run scripts/measure_graded_voxel_resolution.py 0.5 0.5 s5,sinf --full 8 10 12 14 16
  run scripts/measure_graded_voxel_resolution.py 0.5 0.25 s5,sinf --full 8 10 12 14 16
fi
echo; echo "done in $(( $(date +%s) - t0 )) s"
