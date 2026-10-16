#!/bin/bash
# Set up Claude Code cloud sessions: Python dependencies and the compiled Fortran kernels.
# Locally, use the conda env (envs/seismic.yml) and `python -m cubic_scattering.fortran.build`.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# gfortran for the f2py extensions (Ninja is already in the image).
if ! command -v gfortran >/dev/null; then
  apt-get install -y -q gfortran >/dev/null 2>&1 || {
    apt-get update -q >/dev/null
    apt-get install -y -q gfortran >/dev/null
  }
fi

python3 -m pip install --quiet --root-user-action=ignore -r envs/requirements-cloud.txt

# The extensions are untracked (*.so); build them unless both are already there.
if [ "$(ls cubic_scattering/fortran/_point_kernel*.so cubic_scattering/fortran/_green_derivatives*.so 2>/dev/null | wc -l)" -lt 2 ]; then
  log=$(mktemp)
  python3 -m cubic_scattering.fortran.build >"$log" 2>&1 || { cat "$log" >&2; exit 1; }
fi

# Scripts are run as `PYTHONPATH=. python scripts/<name>.py`.
echo 'export PYTHONPATH="'"$CLAUDE_PROJECT_DIR"'"' >> "$CLAUDE_ENV_FILE"
