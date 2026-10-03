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
# the two bases, the stratified background, the moment hierarchy (seconds to a minute or two each)
run scripts/measure_layer_bases.py
run scripts/measure_layer_stratified_background.py
run scripts/crosscheck_cube_moments_ball_shell.py
run scripts/measure_layer_taylor_hierarchy.py
run scripts/measure_layer_taylor_hierarchy_graded.py
run scripts/measure_ball_gradient_hierarchy.py

if [ "$tier" = "full" ]; then
  run scripts/pilot_sphere_voxel_vs_mie.py 4 6 8
  run scripts/pilot_graded_voxel_sphere.py --ka=0.5 --arms=g0fft 4 6 8 12 16 20 24
  run scripts/pilot_graded_voxel_sphere.py --ka=1.0 --arms=g0fft 4 6 8 12 16 20 24
  run scripts/pilot_graded_voxel_sphere.py --weak --ka=0.5 --arms=t9,g0,g1 4 6 8
  run scripts/pilot_graded_voxel_sphere.py --weak --ka=1.0 --arms=t9,g0,g1 4 6 8
  run scripts/pilot_graded_voxel_sphere.py --ka=0.5 --arms=t9,g0,g1fft 4 6 8 10 12 14 16
  run scripts/pilot_graded_voxel_sphere.py --ka=1.0 --arms=t9,g0,g1fft 4 6 8 10 12 14 16
  # the hierarchy as a voxel scheme on the graded sphere with a 9 m shell (Table tab:hiergraded)
  run scripts/measure_graded_sphere_gradient_hierarchy.py --core=1 --q=1,2 4 6 8 10
  run scripts/measure_graded_sphere_gradient_hierarchy_fft.py --core=1 --q=3 --check 4 6
  run scripts/measure_graded_sphere_gradient_hierarchy_fft.py --core=1 --q=3 4 6 8 10 12 14
  run scripts/measure_graded_sphere_gradient_hierarchy_fft.py --core=1 --q=3 --profile=sin2 4 6 8 10 12 14
  run scripts/measure_hierarchy_table_cost.py
  # the single site of a cube, and the cube cut into n^3 cells (tab:hiercube, tab:hiersub)
  run scripts/measure_cube_gradient_hierarchy.py --ref=2,3
  run scripts/measure_lattice_gradient_hierarchy.py
  # oblique incidence: degree two, the stratified backgrounds, the two terms of the error
  run scripts/measure_oblique_degree.py
  run scripts/measure_oblique_stratified_background.py
  run scripts/measure_oblique_two_term.py
  # the graded sphere: near field, and against the impedance march
  run scripts/measure_graded_sphere_near_field.py --ka=0.5 4 6 8 12 16
  run scripts/measure_graded_sphere_near_field.py --ka=1.0 4 6 8 12 16
  for grid in 12:16 16:32 20:32 28:48; do
    run scripts/measure_graded_sphere_march.py --ka=0.5 --n=${grid%%:*} --period=5 --steps=${grid##*:}
  done
  run scripts/measure_graded_sphere_planes.py --ka=0.5 --arms=g0 8 16
  run scripts/measure_graded_sphere_planes.py --ka=0.5 --arms=g1 6 8 16
  run scripts/measure_graded_sphere_planes.py --ka=0.5 --arms=g2 8 10
fi
echo; echo "done in $(( $(date +%s) - t0 )) s"
