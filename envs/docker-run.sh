#!/bin/bash
# Run a command in the project's Docker image, on this checkout (mounted at /work).  The image is rebuilt first,
# which is quick when nothing changed (every layer cached) and redoes only the layers whose inputs did.
#
#   envs/docker-run.sh                       a shell
#   envs/docker-run.sh python scripts/<name>.py
#
# MSC_IMAGE names the image (default msc); extra `docker build` arguments go in MSC_BUILD_ARGS.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
image=${MSC_IMAGE:-msc}
# shellcheck disable=SC2086  # MSC_BUILD_ARGS is split on purpose
docker build -q ${MSC_BUILD_ARGS:-} -t "$image" "$root" >/dev/null
tty=()
[ -t 0 ] && [ -t 1 ] && tty=(-it)
exec docker run --rm "${tty[@]}" -v "$root":/work "$image" "$@"
