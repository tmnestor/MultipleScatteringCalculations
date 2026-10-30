#!/usr/bin/env python3
"""Run this checkout on AWS: edit on the Mac, run in the cloud, get the results back.  aws/README.md is the guide.

One time:
    python3 aws/msc_aws.py deploy [--budget-email you@example.com]     the bucket, image registry, runners and queues
    python3 aws/msc_aws.py push-image cpu                               the arm64 image, for Graviton (native on the Mac)
    python3 aws/msc_aws.py push-image gpu                               the amd64 CUDA image (emulated on the Mac: slow)
Push the image again only when the environment changes (envs/, the Dockerfile): code travels with every run.

On-demand runner (one per kind; stops itself after 30 idle minutes):
    python3 aws/msc_aws.py ec2 start --gpu [--type g6e.xlarge]
    python3 aws/msc_aws.py run --on ec2 --gpu -- python scripts/probe_fft_sphere_scaling.py --device cuda
    python3 aws/msc_aws.py ec2 stop --gpu

Batch (queued; instances start for the job and go away after):
    python3 aws/msc_aws.py run --on batch --vcpus 16 --memory 60000 -- python scripts/gate_x.py
    python3 aws/msc_aws.py run --on batch --array 32 -- python scripts/sweep.py     (AWS_BATCH_JOB_ARRAY_INDEX = 0..31)

Then:
    python3 aws/msc_aws.py status                  recent runs
    python3 aws/msc_aws.py logs RUN [--follow]     the command's output
    python3 aws/msc_aws.py fetch RUN [--apply]     the files it wrote, into aws-runs/RUN/files (--apply: into the checkout)

What runs is the checkout as it is now, uncommitted edits included: the files git tracks or would track
(git ls-files -co --exclude-standard), less the .dockerignore patterns, packed and sent to S3 with each run;
aws/job.py unpacks them over the image's copy, runs the command, and returns every file it created or changed.

AWS profile and region: --profile/--region, else AWS_PROFILE and AWS_REGION, else the aws CLI's defaults.
Stack name: --stack, else MSC_STACK, else "msc".  Needs only the standard library and the aws CLI (and docker
for push-image).
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "aws-runs"
TEMPLATE = Path(__file__).resolve().parent / "infra.yml"
DEFAULT_TYPE = {"cpu": "r7g.2xlarge", "gpu": "g6e.xlarge"}
CPU_FAMILIES = ["c8g", "c7g", "m8g", "m7g", "r8g", "r7g"]
GPU_FAMILIES = ["g6e", "g6", "g5"]
TERMINAL = {"SUCCEEDED", "FAILED", "Success", "Failed", "Cancelled", "TimedOut", "Cancelling"}

G: argparse.Namespace  # the global options, set in main


# ---------------------------------------------------------------------------------------------------- the aws CLI


def aws(*args: str, parse: bool = True, check: bool = True, stdin: str | None = None):
    """Run the aws CLI with the global profile and region; its JSON output parsed (None if empty)."""
    cmd = ["aws", *args]
    if G.profile:
        cmd += ["--profile", G.profile]
    if G.region:
        cmd += ["--region", G.region]
    if parse:
        cmd += ["--output", "json"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, input=stdin, check=False)
    except FileNotFoundError:
        sys.exit("the aws CLI is not installed: see aws/README.md, 'One-time setup'")
    if check and p.returncode != 0:
        sys.exit(f"aws {' '.join(args[:2])} failed:\n{p.stderr.strip()}")
    if not parse:
        return p
    return json.loads(p.stdout) if p.stdout.strip() else None


def region() -> str:
    if G.region:
        return G.region
    r = aws("configure", "get", "region", parse=False, check=False).stdout.strip()
    if not r:
        sys.exit("no AWS region: pass --region, set AWS_REGION, or run 'aws configure'")
    G.region = r
    return r


_outputs: dict[str, str] | None = None


def outputs() -> dict[str, str]:
    global _outputs
    if _outputs is None:
        p = aws("cloudformation", "describe-stacks", "--stack-name", G.stack, check=False, parse=False)
        if p.returncode != 0:
            sys.exit(f"no stack '{G.stack}' in {region()}: run 'python3 aws/msc_aws.py deploy' first")
        stack = json.loads(p.stdout)["Stacks"][0]
        _outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}
    return _outputs


def offered(families: list[str], location: str = "region") -> list[str]:
    """The families whose .xlarge (or .2xlarge) is offered in the region."""
    names = [f"{f}.{s}" for f in families for s in ("xlarge", "2xlarge")]
    got = aws(
        "ec2",
        "describe-instance-type-offerings",
        "--location-type",
        location,
        "--filters",
        f"Name=instance-type,Values={','.join(names)}",
    )
    have = {o["InstanceType"].split(".")[0] for o in got["InstanceTypeOfferings"]}
    return [f for f in families if f in have]


# ---------------------------------------------------------------------------------------------------- deploy


def cmd_deploy(a: argparse.Namespace) -> None:
    vpcs = aws("ec2", "describe-vpcs", "--filters", "Name=isDefault,Values=true")["Vpcs"]
    if not vpcs and not a.vpc:
        sys.exit("this region has no default VPC: pass --vpc and --subnets (public subnets)")
    vpc = a.vpc or vpcs[0]["VpcId"]
    if a.subnets:
        subnets = a.subnets.split(",")
    else:
        found = aws("ec2", "describe-subnets", "--filters", f"Name=vpc-id,Values={vpc}")["Subnets"]
        subnets = [s["SubnetId"] for s in found if s.get("MapPublicIpOnLaunch")]
    if not subnets:
        sys.exit(f"no subnets that assign public IPs in {vpc}: pass --subnets")
    cpu = a.cpu_types.split(",") if a.cpu_types else offered(CPU_FAMILIES)
    gpu = a.gpu_types.split(",") if a.gpu_types else offered(GPU_FAMILIES)
    if not cpu or not gpu:
        sys.exit(
            f"no {'CPU' if not cpu else 'GPU'} instance family offered in {region()}: pass --cpu-types/--gpu-types"
        )
    params = [
        f"VpcId={vpc}",
        f"SubnetIds={','.join(subnets)}",
        f"CpuInstanceTypes={','.join(cpu)}",
        f"GpuInstanceTypes={','.join(gpu)}",
        f"GpuPurchase={a.gpu_purchase}",
        f"CpuMaxVcpus={a.cpu_max_vcpus}",
        f"GpuMaxVcpus={a.gpu_max_vcpus}",
        f"BudgetEmail={a.budget_email or ''}",
        f"MonthlyBudgetUSD={a.budget_usd}",
    ]
    print(
        f"deploying stack '{G.stack}' in {region()}: VPC {vpc}, {len(subnets)} subnets, "
        f"CPU {','.join(cpu)}, GPU {','.join(gpu)} ({a.gpu_purchase})",
        flush=True,
    )
    p = aws(
        "cloudformation",
        "deploy",
        "--template-file",
        str(TEMPLATE),
        "--stack-name",
        G.stack,
        "--capabilities",
        "CAPABILITY_IAM",
        "--no-fail-on-empty-changeset",
        "--parameter-overrides",
        *params,
        parse=False,
        check=False,
    )
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.exit(
            f"{p.stderr.strip()}\nsee the stack's events: aws cloudformation describe-stack-events "
            f"--stack-name {G.stack} --max-items 20"
        )
    for k, v in outputs().items():
        print(f"  {k:20} {v}")
    print("next: python3 aws/msc_aws.py push-image cpu   (and gpu)")


# ---------------------------------------------------------------------------------------------------- images


def cmd_push_image(a: argparse.Namespace) -> None:
    uri = outputs()["RepositoryUri"]
    registry = uri.split("/")[0]
    password = aws("ecr", "get-login-password", parse=False).stdout
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=password,
        text=True,
        check=True,
        capture_output=True,
    )
    sha = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        or "nogit"
    )
    args = ["--build-arg", "WITH_TORCH=1"]
    if a.kind == "gpu":
        platform, args = "linux/amd64", [*args, "--build-arg", "TORCH_INDEX_URL=https://pypi.org/simple"]
        print("building the amd64 image: on an Apple Silicon Mac this runs under emulation (expect 10-30 min)")
    else:
        platform = "linux/arm64"
    tags = [f"{uri}:{a.kind}", f"{uri}:{a.kind}-{sha}"]
    build = ["docker", "build", "--platform", platform, *args, *[x for t in tags for x in ("-t", t)]]
    if a.no_cache:
        build.append("--no-cache")
    subprocess.run([*build, str(ROOT)], check=True)
    for t in tags:
        subprocess.run(["docker", "push", t], check=True)
    print(f"pushed {', '.join(tags)}")


# ---------------------------------------------------------------------------------------------------- bundles


def _ignore_patterns() -> list[str]:
    f = ROOT / ".dockerignore"
    if not f.exists():
        return []
    return [ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def _ignored(path: str, patterns: list[str]) -> bool:
    parts = path.split("/")
    for p in patterns:
        bare = p.removeprefix("**/")
        if fnmatch.fnmatch(path, p) or path.startswith(p.rstrip("/") + "/"):
            return True
        if p.startswith("**/") and any(fnmatch.fnmatch(x, bare) for x in parts):
            return True
    return False


def make_bundle(extra_excludes: list[str]) -> tuple[bytes, int]:
    """The checkout's files (tracked, and untracked unless git-ignored), less .dockerignore's and --exclude's."""
    p = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True, check=True
    )
    patterns = _ignore_patterns() + extra_excludes
    files = [f for f in p.stdout.decode().split("\0") if f and not _ignored(f, patterns) and (ROOT / f).is_file()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in files:
            tf.add(ROOT / f, arcname=f)
    return buf.getvalue(), len(files)


# ---------------------------------------------------------------------------------------------------- runs


def _record(run_id: str) -> dict:
    f = RUNS / run_id / "run.json"
    if not f.exists():
        matches = sorted(d.name for d in RUNS.glob(f"*{run_id}*")) if RUNS.exists() else []
        if len(matches) == 1:
            return _record(matches[0])
        sys.exit(f"no run '{run_id}' in {RUNS}" + (f" (matches: {', '.join(matches)})" if matches else ""))
    return json.loads(f.read_text())


def _save(rec: dict) -> None:
    d = RUNS / rec["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(rec, indent=1))


def _s3_put_bytes(data: bytes, url: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as t:
        t.write(data)
    try:
        aws("s3", "cp", "--only-show-errors", t.name, url, parse=False)
    finally:
        os.unlink(t.name)


def _s3_text(url: str) -> str | None:
    p = aws("s3", "cp", "--only-show-errors", url, "-", parse=False, check=False)
    return p.stdout if p.returncode == 0 else None


def runner(kind: str, states: str = "pending,running,stopping,stopped") -> dict | None:
    got = aws(
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=tag:msc:stack,Values={G.stack}",
        f"Name=tag:msc:runner,Values={kind}",
        f"Name=instance-state-name,Values={states}",
    )
    inst = [i for r in got["Reservations"] for i in r["Instances"]]
    return max(inst, key=lambda i: i["LaunchTime"]) if inst else None


def _wait_ssm(instance_id: str, timeout: float = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        info = aws("ssm", "describe-instance-information", "--filters", f"Key=InstanceIds,Values={instance_id}")
        if info["InstanceInformationList"] and info["InstanceInformationList"][0]["PingStatus"] == "Online":
            return
        time.sleep(10)
    sys.exit(f"{instance_id} did not register with SSM within {timeout:.0f} s")


def _ensure_runner(kind: str) -> str:
    inst = runner(kind)
    if inst is None:
        sys.exit(f"no {kind} runner: python3 aws/msc_aws.py ec2 start{' --gpu' if kind == 'gpu' else ''}")
    state = inst["State"]["Name"]
    if state in ("stopped", "stopping"):
        print(f"starting the stopped {kind} runner {inst['InstanceId']} ...", flush=True)
        if state == "stopping":
            aws("ec2", "wait", "instance-stopped", "--instance-ids", inst["InstanceId"], parse=False)
        aws("ec2", "start-instances", "--instance-ids", inst["InstanceId"])
    if state != "running":
        aws("ec2", "wait", "instance-running", "--instance-ids", inst["InstanceId"], parse=False)
    _wait_ssm(inst["InstanceId"])
    return inst["InstanceId"]


def job_command(rec: dict, cmd: list[str], cache: bool) -> list[str]:
    base = f"s3://{rec['bucket']}/runs/{rec['id']}"
    out = ["python", "/work/aws/job.py", "--code", f"{base}/code.tgz", "--out", base]
    if cache:
        out += ["--cache", f"s3://{rec['bucket']}/cache"]
    return [*out, "--", *cmd]


def cmd_run(a: argparse.Namespace) -> None:
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        sys.exit("give the command after --, e.g.  run --on ec2 -- python scripts/x.py")
    o = outputs()
    kind = "gpu" if a.gpu else "cpu"
    stem = a.name or next((Path(x).stem for x in cmd if x.endswith(".py")), cmd[0])
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^A-Za-z0-9_-]+", "-", stem)[:40]
    rec = {
        "id": run_id,
        "on": a.on,
        "kind": kind,
        "command": cmd,
        "bucket": o["Bucket"],
        "region": region(),
        "submitted": time.time(),
        "array": a.array,
    }

    data, n = make_bundle(a.exclude)
    print(f"{run_id}: {n} files, {len(data) / 1e6:.1f} MB -> s3://{o['Bucket']}/runs/{run_id}/", flush=True)
    _s3_put_bytes(data, f"s3://{o['Bucket']}/runs/{run_id}/code.tgz")

    if a.on == "ec2":
        if a.array:
            sys.exit("--array is for Batch; on a runner, loop inside the command")
        iid = _ensure_runner(kind)
        image = f"{o['RepositoryUri']}:{kind}"
        registry = image.split("/")[0]
        docker = [
            "docker",
            "run",
            "--rm",
            "--label",
            "msc.job=1",
            "--ipc=host",
            "-e",
            f"AWS_DEFAULT_REGION={region()}",
            "-v",
            "/var/lib/msc-cache:/root/.cache/cubic_scattering",
        ]
        if kind == "gpu":
            docker += ["--gpus", "all"]
        script = [
            "set -euo pipefail",
            "command -v aws >/dev/null || dnf install -y -q awscli-2",
            (
                f"aws ecr get-login-password --region {region()}"
                f" | docker login --username AWS --password-stdin {registry} >/dev/null"
            ),
            f"docker pull -q {image} >/dev/null",
            shlex.join([*docker, image, *job_command(rec, cmd, cache=False)]),
        ]
        params = json.dumps({"commands": script, "executionTimeout": [str(a.timeout_hours * 3600)]})
        sent = aws(
            "ssm",
            "send-command",
            "--instance-ids",
            iid,
            "--document-name",
            "AWS-RunShellScript",
            "--comment",
            run_id[:100],
            "--parameters",
            params,
            "--output-s3-bucket-name",
            o["Bucket"],
            "--output-s3-key-prefix",
            f"runs/{run_id}/ssm",
        )
        rec.update(instance=iid, command_id=sent["Command"]["CommandId"])
        print(f"  sent to {iid} (SSM command {rec['command_id']})")
    else:
        overrides: dict = {
            "command": job_command(rec, cmd, cache=True),
            "environment": [{"name": "MSC_RUN_ID", "value": run_id}],
        }
        req = [
            {"type": t, "value": str(v)}
            for t, v in (("VCPU", a.vcpus), ("MEMORY", a.memory), ("GPU", a.gpus if kind == "gpu" else None))
            if v
        ]
        if req:
            overrides["resourceRequirements"] = req
        args = [
            "batch",
            "submit-job",
            "--job-name",
            f"msc-{run_id}"[:128],
            "--job-queue",
            o["GpuQueue" if a.gpu else "CpuQueue"],
            "--job-definition",
            o["GpuJobDefinition" if a.gpu else "CpuJobDefinition"],
            "--container-overrides",
            json.dumps(overrides),
            "--timeout",
            f"attemptDurationSeconds={a.timeout_hours * 3600}",
        ]
        if a.array:
            args += ["--array-properties", f"size={a.array}"]
        rec["job_id"] = aws(*args)["jobId"]
        print(f"  Batch job {rec['job_id']} on {o['GpuQueue' if a.gpu else 'CpuQueue']}")
    _save(rec)
    if (a.wait if a.wait is not None else a.on == "ec2") and not a.array:
        _follow(rec)
        print(f"[{state(rec)[0]}]")
        cmd_fetch(argparse.Namespace(run=run_id, apply=False))
    else:
        print(f"  follow:  python3 aws/msc_aws.py logs {run_id} --follow")


def state(rec: dict) -> tuple[str, str]:
    """(state, detail) of a run."""
    if rec["on"] == "ec2":
        inv = aws(
            "ssm",
            "get-command-invocation",
            "--command-id",
            rec["command_id"],
            "--instance-id",
            rec["instance"],
            check=False,
            parse=False,
        )
        if inv.returncode != 0:
            return "Pending", ""
        j = json.loads(inv.stdout)
        return j["Status"], f"exit {j['ResponseCode']}" if j["ResponseCode"] >= 0 else ""
    job = aws("batch", "describe-jobs", "--jobs", rec["job_id"])["jobs"]
    if not job:
        return "unknown", "(expired from Batch)"
    job = job[0]
    detail = job.get("statusReason", "")
    if "arrayProperties" in job:
        detail = " ".join(f"{k.lower()} {v}" for k, v in job["arrayProperties"].get("statusSummary", {}).items() if v)
    elif "container" in job and "exitCode" in job["container"]:
        detail = f"exit {job['container']['exitCode']}"
    return job["status"], detail


def _log_url(rec: dict, index: int | None) -> str:
    base = f"s3://{rec['bucket']}/runs/{rec['id']}"
    return f"{base}/array-{index}/log.txt" if index is not None else f"{base}/log.txt"


def _follow(rec: dict, index: int | None = None) -> None:
    """Print the log as it grows (aws/job.py uploads it every 20 s) until the run ends."""
    shown = 0
    while True:
        st = state(rec)[0]
        text = _s3_text(_log_url(rec, index)) or ""
        if len(text) > shown:
            sys.stdout.write(text[shown:])
            sys.stdout.flush()
            shown = len(text)
        if st in TERMINAL:
            text = _s3_text(_log_url(rec, index)) or ""
            sys.stdout.write(text[shown:])
            if st not in ("Success", "SUCCEEDED") and not text:
                _print_runner_stderr(rec)
            return
        time.sleep(15)


def _print_runner_stderr(rec: dict) -> None:
    """On a runner, a failure before job.py started (login, image pull) is only in SSM's captured stderr."""
    if rec["on"] != "ec2":
        return
    out = _s3_text(
        f"s3://{rec['bucket']}/runs/{rec['id']}/ssm/{rec['command_id']}/{rec['instance']}"
        "/awsrunShellScript/0.awsrunShellScript/stderr"
    )
    if out:
        print(f"--- runner stderr ---\n{out}")


def cmd_logs(a: argparse.Namespace) -> None:
    rec = _record(a.run)
    if a.follow:
        _follow(rec, a.index)
        return
    text = _s3_text(_log_url(rec, a.index))
    if text is not None:
        print(text)
        return
    st = state(rec)
    print(f"no log yet ({' '.join(st)})")
    if st[0] in TERMINAL:
        _print_runner_stderr(rec)


def cmd_status(a: argparse.Namespace) -> None:
    if not RUNS.exists():
        print("no runs yet")
        return
    recs = [json.loads(f.read_text()) for f in sorted(RUNS.glob("*/run.json"))][-a.n :]
    for rec in recs:
        st, detail = state(rec)
        where = f"{rec['on']}/{rec['kind']}" + (f" x{rec['array']}" if rec.get("array") else "")
        print(f"{rec['id']:45} {where:13} {st:10} {detail:24} {shlex.join(rec['command'])[:60]}")


def cmd_fetch(a: argparse.Namespace) -> None:
    rec = _record(a.run)
    dest = RUNS / rec["id"]
    base = f"s3://{rec['bucket']}/runs/{rec['id']}/"
    aws(
        "s3",
        "sync",
        "--only-show-errors",
        base,
        str(dest / "s3"),
        "--exclude",
        "code.tgz",
        "--exclude",
        "ssm/*",
        parse=False,
    )
    archives = sorted((dest / "s3").rglob("out.tgz"))
    if not archives:
        print(f"no results yet ({' '.join(state(rec))})")
        return
    for tgz in archives:
        sub = tgz.parent.relative_to(dest / "s3")
        with tarfile.open(tgz) as tf:
            names = tf.getnames()
            target = ROOT if a.apply and str(sub) == "." else dest / "files" / sub
            tf.extractall(target, filter="data")
        status = json.loads((tgz.parent / "status.json").read_text()) if (tgz.parent / "status.json").exists() else {}
        label = "" if str(sub) == "." else f"{sub}: "
        print(f"{label}exit {status.get('exit_code', '?')}, {len(names)} file(s) -> {target}")
        for nm in names[:20]:
            print(f"    {nm}")
        if len(names) > 20:
            print(f"    ... and {len(names) - 20} more")
    if a.apply and any(str(t.parent.relative_to(dest / "s3")) != "." for t in archives):
        print("array results are kept under aws-runs/ (--apply takes a single run's only)")


# ---------------------------------------------------------------------------------------------------- runners


def _arch(itype: str) -> list[str]:
    got = aws("ec2", "describe-instance-types", "--instance-types", itype)
    return got["InstanceTypes"][0]["ProcessorInfo"]["SupportedArchitectures"]


def cmd_ec2(a: argparse.Namespace) -> None:
    kinds = ["cpu", "gpu"] if getattr(a, "all", False) else ["gpu" if a.gpu else "cpu"]
    if a.action == "list":
        for kind in ("cpu", "gpu"):
            inst = runner(kind)
            if inst:
                print(f"{kind}: {inst['InstanceId']} {inst['InstanceType']} {inst['State']['Name']}")
            else:
                print(f"{kind}: none")
        return
    if a.action == "start":
        kind = kinds[0]
        itype = a.type or DEFAULT_TYPE[kind]
        want = "arm64" if kind == "cpu" else "x86_64"
        if want not in _arch(itype):
            sys.exit(
                f"{itype} is not {want}: the {kind} image needs {want} ({'Graviton, e.g. r7g' if kind == 'cpu' else 'e.g. g6e'})"
            )
        inst = runner(kind)
        if inst and inst["State"]["Name"] in ("running", "pending"):
            print(f"{kind} runner {inst['InstanceId']} ({inst['InstanceType']}) is already {inst['State']['Name']}")
            _wait_ssm(inst["InstanceId"])
            return
        if inst:
            iid = inst["InstanceId"]
            if inst["State"]["Name"] == "stopping":
                aws("ec2", "wait", "instance-stopped", "--instance-ids", iid, parse=False)
            if inst["InstanceType"] != itype and a.type:
                aws("ec2", "modify-instance-attribute", "--instance-id", iid, "--instance-type", f"Value={itype}")
            aws("ec2", "start-instances", "--instance-ids", iid)
        else:
            o = outputs()
            azs = {
                x["Location"]
                for x in aws(
                    "ec2",
                    "describe-instance-type-offerings",
                    "--location-type",
                    "availability-zone",
                    "--filters",
                    f"Name=instance-type,Values={itype}",
                )["InstanceTypeOfferings"]
            }
            subs = aws("ec2", "describe-subnets", "--subnet-ids", *o["SubnetIds"].split(","))["Subnets"]
            subnet = next((s["SubnetId"] for s in subs if s["AvailabilityZone"] in azs), None)
            if subnet is None:
                alts = offered(GPU_FAMILIES if kind == "gpu" else CPU_FAMILIES)
                sys.exit(
                    f"{itype} is not offered in the stack's zones in {region()}; families offered there: "
                    f"{', '.join(alts) or 'none of ' + ', '.join(GPU_FAMILIES if kind == 'gpu' else CPU_FAMILIES)} "
                    f"(e.g. --type {alts[0] + '.xlarge' if alts else '...'})"
                )
            tmpl = o["RunnerTemplateGpu" if kind == "gpu" else "RunnerTemplateCpu"]
            got = aws(
                "ec2",
                "run-instances",
                "--launch-template",
                f"LaunchTemplateId={tmpl},Version=$Latest",
                "--instance-type",
                itype,
                "--subnet-id",
                subnet,
                "--count",
                "1",
                check=False,
                parse=False,
            )
            if got.returncode != 0:
                hint = ""
                if "VcpuLimitExceeded" in got.stderr:
                    hint = (
                        "\nyour account's vCPU quota for this instance family is too low (new accounts: 0 for "
                        "G and P): Service Quotas -> Amazon EC2; see aws/README.md, 'Quotas'"
                    )
                sys.exit(f"run-instances failed:\n{got.stderr.strip()}{hint}")
            iid = json.loads(got.stdout)["Instances"][0]["InstanceId"]
        print(f"waiting for {kind} runner {iid} ({itype}) ...", flush=True)
        aws("ec2", "wait", "instance-running", "--instance-ids", iid, parse=False)
        _wait_ssm(iid)
        print(f"{kind} runner {iid} is ready; it stops itself after 30 idle minutes")
        return
    for kind in kinds:
        inst = runner(kind, "pending,running,stopping,stopped")
        if inst is None:
            print(f"no {kind} runner")
            continue
        iid = inst["InstanceId"]
        if a.action == "stop":
            aws("ec2", "stop-instances", "--instance-ids", iid)
            print(f"stopping {kind} runner {iid} (its disk is kept: a few dollars a month for 100 GB)")
        elif a.action == "terminate":
            aws("ec2", "terminate-instances", "--instance-ids", iid)
            print(f"terminating {kind} runner {iid}")
        elif a.action == "shell":
            iid = _ensure_runner(kind)
            cmd = ["aws", "ssm", "start-session", "--target", iid, "--region", region()]
            if G.profile:
                cmd += ["--profile", G.profile]
            os.execvp("aws", cmd)


# ---------------------------------------------------------------------------------------------------- teardown


def cmd_teardown(a: argparse.Namespace) -> None:
    o = outputs()
    if not a.yes:
        ans = input(
            f"delete stack '{G.stack}' in {region()}: the runners, queues, image registry and EVERY object "
            f"in s3://{o['Bucket']}? type the stack name to confirm: "
        )
        if ans.strip() != G.stack:
            sys.exit("not deleted")
    for kind in ("cpu", "gpu"):
        inst = runner(kind)
        if inst:
            aws("ec2", "terminate-instances", "--instance-ids", inst["InstanceId"])
            aws("ec2", "wait", "instance-terminated", "--instance-ids", inst["InstanceId"], parse=False)
    aws("s3", "rm", "--recursive", "--only-show-errors", f"s3://{o['Bucket']}", parse=False)
    aws("cloudformation", "delete-stack", "--stack-name", G.stack, parse=False)
    print("deleting the stack ...", flush=True)
    aws("cloudformation", "wait", "stack-delete-complete", "--stack-name", G.stack, parse=False)
    print("deleted")


# ---------------------------------------------------------------------------------------------------- main


def main() -> None:
    global G
    ap = argparse.ArgumentParser(
        prog="msc_aws.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stack", default=os.environ.get("MSC_STACK", "msc"))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("deploy", help="create or update the stack")
    d.add_argument("--budget-email", help="email a monthly cost budget's alerts here")
    d.add_argument("--budget-usd", type=int, default=100)
    d.add_argument("--gpu-purchase", choices=["SPOT", "EC2"], default="SPOT", help="the GPU queue's instances")
    d.add_argument("--cpu-types", help="Batch CPU families/types (arm64), comma-separated")
    d.add_argument("--gpu-types", help="Batch GPU families/types, comma-separated")
    d.add_argument("--cpu-max-vcpus", type=int, default=256)
    d.add_argument("--gpu-max-vcpus", type=int, default=32)
    d.add_argument("--vpc")
    d.add_argument("--subnets")
    d.set_defaults(func=cmd_deploy)

    p = sub.add_parser("push-image", help="build the image and push it to the stack's registry")
    p.add_argument("kind", choices=["cpu", "gpu"])
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_push_image)

    r = sub.add_parser("run", help="run a command on this checkout in AWS")
    r.add_argument("--on", choices=["ec2", "batch"], default="ec2")
    r.add_argument("--gpu", action="store_true", help="the GPU runner or queue")
    r.add_argument("--vcpus", type=int, help="Batch: vCPUs for the job")
    r.add_argument("--memory", type=int, help="Batch: memory for the job, MiB")
    r.add_argument("--gpus", type=int, help="Batch: GPUs for the job (with --gpu; default 1)")
    r.add_argument("--array", type=int, help="Batch: an array job of this many children")
    r.add_argument("--name", help="a label for the run id (default: the script's name)")
    r.add_argument("--exclude", action="append", default=[], help="leave these paths out of the bundle (glob)")
    r.add_argument("--timeout-hours", type=int, default=48)
    w = r.add_mutually_exclusive_group()
    w.add_argument("--wait", dest="wait", action="store_true", default=None, help="follow the log (default on ec2)")
    w.add_argument("--no-wait", dest="wait", action="store_false", help="return once submitted (default on batch)")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="recent runs")
    s.add_argument("-n", type=int, default=15)
    s.set_defaults(func=cmd_status)

    lg = sub.add_parser("logs", help="a run's output")
    lg.add_argument("run")
    lg.add_argument("--follow", "-f", action="store_true")
    lg.add_argument("--index", type=int, help="an array job's child")
    lg.set_defaults(func=cmd_logs)

    f = sub.add_parser("fetch", help="a run's results, into aws-runs/RUN/files")
    f.add_argument("run")
    f.add_argument("--apply", action="store_true", help="extract into the checkout itself (overwrites)")
    f.set_defaults(func=cmd_fetch)

    e = sub.add_parser("ec2", help="the on-demand runners")
    e.add_argument("action", choices=["start", "stop", "list", "shell", "terminate"])
    e.add_argument("--gpu", action="store_true")
    e.add_argument("--all", action="store_true", help="stop/terminate: both runners")
    e.add_argument("--type", help=f"start: the instance type (default {DEFAULT_TYPE})")
    e.set_defaults(func=cmd_ec2)

    t = sub.add_parser("teardown", help="delete everything the stack made, and the bucket's contents")
    t.add_argument("--yes", action="store_true")
    t.set_defaults(func=cmd_teardown)

    G = ap.parse_args()
    G.func(G)


if __name__ == "__main__":
    main()
