#!/bin/bash
# The image's entrypoint: rebuild the kernels if a mounted checkout's are missing or stale, then run the command.
set -euo pipefail
PYTHON=python /work/envs/build_kernels_if_stale.sh
exec "$@"
