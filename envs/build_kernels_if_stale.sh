#!/bin/bash
# Build the compiled kernels of cubic_scattering/fortran when they are missing for this interpreter, or older than
# their sources.  Used by the Docker image (at build and at start), the dev container and the cloud SessionStart hook.
set -euo pipefail
cd "$(dirname "$0")/.."

py=${PYTHON:-python3}
suffix=$("$py" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
dir=cubic_scattering/fortran
stale=0
for module in _point_kernel _green_derivatives _cell_average _kseries_moments; do  # as SOURCES in cubic_scattering/fortran/build.py
  so="$dir/$module$suffix"
  if [ ! -f "$so" ]; then
    stale=1
    break
  fi
  for src in "$dir"/*.f90 "$dir/build.py"; do
    if [ "$src" -nt "$so" ]; then
      stale=1
      break 2
    fi
  done
done
[ "$stale" = 0 ] && exit 0

log=$(mktemp)
"$py" -m cubic_scattering.fortran.build >"$log" 2>&1 || { cat "$log" >&2; exit 1; }
rm -f "$log"
echo "built the Fortran kernels ($dir/*$suffix)" >&2
