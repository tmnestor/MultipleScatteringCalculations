# Running on AWS

Edit on the Mac; run on AWS; get the results back. You don't commit or push anything to run, and the image is
rebuilt only when the environment changes.

```
Mac: python3 aws/msc_aws.py run --on ec2 --gpu -- python scripts/x.py
  │  packs the checkout as it is (uncommitted edits included) → S3
  ▼
EC2 runner, or a Batch job:  docker run <image>  python /work/aws/job.py --code s3://… -- python scripts/x.py
  │  unpacks the code over the image's copy, rebuilds the Fortran kernels only if a .f90 changed,
  │  runs the command, returns every file it created or changed, and its log → S3
  ▼
Mac: python3 aws/msc_aws.py fetch RUN   → aws-runs/RUN/files/   (or --apply, into the checkout)
```

| Piece | What it is |
|---|---|
| `aws/infra.yml` | CloudFormation: S3 bucket, ECR registry, IAM roles, launch templates for the runners, Batch queues, optional budget alert |
| `aws/msc_aws.py` | The Mac-side command (standard library and the `aws` CLI only) |
| `aws/job.py` | Runs inside the container: unpack, run, return results |
| `Dockerfile` | The image, with `WITH_TORCH=1`; CPU build is arm64 (Graviton), GPU build is amd64 + CUDA 12.8 |
| `envs/seismic-linux.yml` | The conda env for Linux, for using an instance without Docker |

There are two ways to run:

- **EC2 runner**: one instance per kind (CPU, GPU), started on demand. Best for iterating: the image and the
  k-series store stay on its disk between runs, and you can open a shell on it. It **stops itself after 30 idle
  minutes** (no job running and no shell open).
- **Batch**: queued jobs on instances that start for the job and go away after (scale to zero). Best for sweeps
  and anything you'd otherwise babysit: `--array N` runs N children, each with `AWS_BATCH_JOB_ARRAY_INDEX`.
  CPU jobs use Spot instances (cheaper; Batch retries a job if AWS reclaims the instance).

## One-time setup

1. **AWS CLI v2 and the Session Manager plugin** (the plugin is only for `ec2 shell`):
   ```bash
   brew install awscli
   brew install --cask session-manager-plugin
   ```
2. **Credentials.** Don't use the root user's keys. The recommended route is IAM Identity Center: in the console,
   enable it, create a user for yourself with the `AdministratorAccess` permission set, then
   ```bash
   aws configure sso            # profile name e.g. msc; region ap-southeast-2 (Sydney)
   export AWS_PROFILE=msc       # put it in ~/.zshrc
   aws sso login                # when the session expires
   aws sts get-caller-identity  # check
   ```
   (An IAM user with access keys via `aws configure` also works, but the keys are long-lived.)

   Region: Sydney (`ap-southeast-2`) has the widest choice of instances near you. Melbourne (`ap-southeast-4`)
   is newer and offers fewer GPU types.
3. **Quotas.** New accounts often have a vCPU quota of **0 for GPU (G and VT) instances**, both on-demand and
   Spot. In the console: Service Quotas → Amazon EC2 → request increases for
   *Running On-Demand G and VT instances* and *All G and VT Spot Instance Requests* (8 vCPUs is enough for a
   g6e.xlarge or g6.2xlarge). A request can take a day or so. CPU quotas usually start nonzero.
4. **Deploy the stack and push the images:**
   ```bash
   python3 aws/msc_aws.py deploy --budget-email you@example.com    # about 3 minutes
   python3 aws/msc_aws.py push-image cpu                            # native on the Mac
   python3 aws/msc_aws.py push-image gpu                            # emulated amd64 build: 10–30 minutes
   ```
   `deploy` uses the region's default VPC and only the instance families offered in the region. The budget
   emails you at 80% (forecast) and 100% of `--budget-usd` (default 100) a month. It doesn't stop
   anything; it only warns.

Push the images again only after changing `envs/`, the `Dockerfile`, or the Fortran build. Code changes travel
with every run.

## Daily use

```bash
# A GPU runner: g6e.xlarge (L40S, 48 GB) where offered, else g6.xlarge (L4, 24 GB) or g5.xlarge (A10G, 24 GB);
# Sydney has no g6e. First start takes ~2 min, later starts ~1 min
python3 aws/msc_aws.py ec2 start --gpu
python3 aws/msc_aws.py run --on ec2 --gpu -- python -c "import torch; print(torch.cuda.get_device_name())"
python3 aws/msc_aws.py run --on ec2 --gpu -- python scripts/probe_fft_sphere_scaling.py --device cuda --dtype complex128 --warm-store
python3 aws/msc_aws.py ec2 stop --gpu          # or let it stop itself after 30 idle minutes

# A big-memory CPU runner (default r7g.2xlarge: 8 vCPU, 64 GB; e.g. --type r7g.8xlarge for 256 GB)
python3 aws/msc_aws.py ec2 start --type r7g.8xlarge
python3 aws/msc_aws.py run --on ec2 -- python -m pytest -n auto -m "not slow" cubic_scattering/tests

# Batch: submit and walk away
python3 aws/msc_aws.py run --on batch --vcpus 16 --memory 120000 -- python scripts/gate_x.py
python3 aws/msc_aws.py run --on batch --array 32 -- python scripts/my_sweep.py   # reads AWS_BATCH_JOB_ARRAY_INDEX
python3 aws/msc_aws.py run --on batch --gpu -- python scripts/y.py

python3 aws/msc_aws.py status                  # recent runs
python3 aws/msc_aws.py logs RUN --follow       # live output (updated every 20 s); --index K for an array child
python3 aws/msc_aws.py fetch RUN               # → aws-runs/RUN/files/ ; --apply writes into the checkout
python3 aws/msc_aws.py ec2 shell --gpu         # a shell on the runner (nvidia-smi, htop, docker ps)
python3 aws/msc_aws.py cancel RUN              # stop a run: on a runner its container, on Batch the job
python3 aws/msc_aws.py ec2 kill --gpu          # stop every job on the GPU runner
```

`RUN` can be any unique part of the run id. `run` on EC2 waits and fetches by default (`--no-wait` to return at
once); on Batch it returns at once (`--wait` to follow). A stopped runner is started by `run` itself;
`ec2 start --type …` on a stopped runner changes its type. A runner takes one run at a time: `run` refuses while
an earlier run of yours is still going there, since two would share its CPUs and memory and spoil each other's
timings (`--force` overrides). Closing the terminal or Ctrl-C stops only the waiting, never the run: use `cancel`.

What gets sent: everything `git ls-files -co --exclude-standard` lists (tracked files, and untracked files that
aren't ignored), less the `.dockerignore` patterns (LaTeX, Mathematica, the thesis Fortran, `scratch/`) and any
`--exclude GLOB`. That's about 16 MB compressed at present. What comes back: the files the command created or
changed under the checkout (not caches or compiled kernels), plus its log and exit status.

## Choosing hardware

Run `scripts/probe_fft_sphere_scaling.py` on the Mac first: its extrapolation shows what a target n_sub needs.

- **Memory is mostly the FFT product's grids.** `torch_gmres` grows its Krylov basis by one vector a step (since
  2026-10-31; before, it allocated all gmres_maxiter + 1 vectors up front, 32 GB at n_sub = 128 in complex128).
  What remains does not depend on maxiter: about 16.5 GB at n_sub = 96 in complex128 (measured on r7g.2xlarge),
  probably mostly the padded 9 × nP³ grids of the FFT product (inferred from their size, not profiled). The probe's
  extrapolation gives about 37 GB at 128 and 69 GB at 160.
- **GPUs and complex128.** The g-family GPUs (L4, L40S, A10G) are fast in single precision and slow in double.
  `select_dtype` picks complex128 on CUDA. If the FFT products in complex64 meet your tolerance, pass
  `dtype=torch.complex64` and the cheap GPUs are excellent. For complex128 at speed you need A100/H100 (p4d/p5),
  which AWS sells only in large, expensive multi-GPU sizes, and they need their own quota.
- **Big CPU memory** (r7g/r8g up to hundreds of GB; x2 families to terabytes) is the simple route for the dense
  solvers and for large FFT solves in complex128. The Fortran kernels use every core (OpenMP).

## Costs and cleanup

- A stopped runner costs only its 100 GB disk, a few dollars a month. `ec2 terminate --gpu` removes it;
  `ec2 start` makes a fresh one.
- Batch costs nothing when idle.
- S3: code bundles and results under `runs/` are deleted after 30 days; `fetch` what you want to keep.
- Delete everything: `python3 aws/msc_aws.py teardown` (asks you to type the stack name).

## When something goes wrong

- `run-instances failed … VcpuLimitExceeded`: the quota (setup step 3).
- `… is not offered in the stack's zones`: pick one of the families it lists, e.g. `--type g6.xlarge`.
- A Batch job stays `RUNNABLE`: no instance can be launched (quota, or no Spot capacity for those types).
  Check the compute environment in the Batch console; for GPUs, redeploy with `--gpu-purchase EC2`.
- A run on a runner fails before the command starts: `logs` shows the runner's stderr (image pull, login).
  `ec2 shell` to look around.
- `torch.cuda.is_available()` is False on the GPU runner: check `nvidia-smi` in `ec2 shell`. The ECS GPU AMI
  provides the driver and container toolkit; `docker run --gpus all` needs both.
