# syntax=docker/dockerfile:1
# The Multiple Scattering project's environment: Python 3.12, the pinned packages of envs/requirements-cloud.txt
# (the cloud sessions' set, kept in step with envs/seismic.yml), gfortran, the compiled Fortran kernels, and
# optionally torch.
#
#   envs/docker-run.sh                       a shell on this checkout (rebuilds the image first if anything changed)
#   envs/docker-run.sh python -m pytest cubic_scattering/tests/test_interface_solver.py
#   envs/docker-run.sh python scripts/gate_interface_gmres.py
#
# Each layer is rebuilt only when its inputs change: the system packages with this file, the Python packages with
# envs/requirements-cloud.txt, the kernels with their Fortran sources.  A mounted checkout's kernels are checked again
# at start (envs/build_kernels_if_stale.sh), so editing a .f90 needs no image rebuild.
#
# With the PyTorch solvers:  MSC_BUILD_ARGS="--build-arg WITH_TORCH=1" envs/docker-run.sh
#
# Behind a TLS-intercepting proxy, pass its CA bundle as a build secret (and the proxy as build args):
#   docker build --secret id=ca,src=/path/to/ca-bundle.crt --build-arg https_proxy=... -t msc .
FROM ubuntu:24.04

# torch is optional: only the GPU solvers use it (imported lazily; their tests are skipped without it).
# WITH_TORCH=1 installs it, from the CPU-only index: the PyPI wheel for x86-64 pulls ~4 GB of CUDA libraries, and a
# container on a Mac has no MPS.  The version is envs/seismic.yml's.
ARG WITH_TORCH=0
ARG TORCH_VERSION=2.10.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/work \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN --mount=type=secret,id=ca,mode=0444 \
    if [ -f /run/secrets/ca ]; then \
        # a proxy that tunnels HTTPS only: apt over https, verified against the given CA
        echo 'Acquire::https::CAInfo "/run/secrets/ca";' > /etc/apt/apt.conf.d/99ca \
        && sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev gfortran ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/99ca \
    && python3.12 -m venv "$VIRTUAL_ENV"

COPY envs/requirements-cloud.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=secret,id=ca,mode=0444 \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi \
    && pip install -r /tmp/requirements.txt \
    && if [ "$WITH_TORCH" = 1 ]; then pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"; fi

WORKDIR /work
COPY . .
RUN PYTHON=python envs/build_kernels_if_stale.sh

ENTRYPOINT ["/work/envs/docker-entrypoint.sh"]
CMD ["bash"]
