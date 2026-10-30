"""Run one command in the image on a code bundle, and return what it wrote: the in-container half of aws/msc_aws.py.

    python /work/aws/job.py --code s3://BUCKET/runs/ID/code.tgz --out s3://BUCKET/runs/ID/ [--cache s3://BUCKET/cache/] \
        -- python scripts/probe_fft_sphere_scaling.py --device cuda

1. The bundle (the checkout's files as msc_aws.py packed them, uncommitted edits included) is unpacked over /work, the
   image's copy, so the image's compiled kernels are kept unless a Fortran source in the bundle is newer
   (envs/build_kernels_if_stale.sh then rebuilds them, as the entrypoint does for a mounted checkout).
2. With --cache, the k-series coefficient store (numerics.yml: kseries_store, ~/.cache/cubic_scattering/) is fetched
   first and put back afterwards if it grew, so that sweep workers on fresh instances do not each rebuild it.  The
   EC2 runner mounts a persistent directory there instead and passes no --cache.
3. The command runs in /work; its output goes to this process's stdout (CloudWatch, for Batch) and to log.txt,
   which is uploaded every 20 s while it runs (msc_aws.py logs --follow reads it).
4. Every file under /work the command created or changed (newer than the start, less caches and compiled files) is
   packed into out.tgz; that, log.txt and status.json go to --out.  In an array job each child writes under
   --out/array-<index>/, and AWS_BATCH_JOB_ARRAY_INDEX is in the command's environment for it to choose its share.

The exit status is the command's.  --code, --out and --cache may be local paths instead of s3:// URLs (for testing).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

WORK = Path(os.environ.get("MSC_WORK", "/work"))
CACHE = Path.home() / ".cache" / "cubic_scattering"
#: not returned: caches, compiled files, and the bundle's own unpacking
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git"}
SKIP_SUFFIXES = (".so", ".pyc", ".o", ".mod")
SKIP_NAMES = {".testmondata", ".testmondata-shm", ".testmondata-wal"}


def _s3():
    import boto3

    return boto3.client("s3")


def _split(url: str) -> tuple[str, str]:
    bucket, _, key = url.removeprefix("s3://").partition("/")
    return bucket, key


def get(src: str, dst: Path) -> bool:
    """Fetch src (s3:// or a local path) to dst; False if there is none."""
    if src.startswith("s3://"):
        import botocore.exceptions

        try:
            _s3().download_file(*_split(src), str(dst))
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise
        return True
    if not Path(src).exists():
        return False
    dst.write_bytes(Path(src).read_bytes())
    return True


def put(src: Path, dst: str) -> None:
    if dst.startswith("s3://"):
        _s3().upload_file(str(src), *_split(dst))
    else:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(src.read_bytes())


def size_of(url: str) -> int:
    """Size of the object at url, -1 if there is none."""
    if url.startswith("s3://"):
        import botocore.exceptions

        try:
            return _s3().head_object(Bucket=_split(url)[0], Key=_split(url)[1])["ContentLength"]
        except botocore.exceptions.ClientError:
            return -1
    return Path(url).stat().st_size if Path(url).exists() else -1


def join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def changed_files(root: Path, since: float) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            p = Path(dirpath, f)
            if f in SKIP_NAMES or f.endswith(SKIP_SUFFIXES) or p.is_symlink():
                continue
            if p.stat().st_mtime >= since:
                out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a command on a code bundle and return its results.")
    ap.add_argument("--code", required=True, help="the code bundle (.tgz)")
    ap.add_argument("--out", required=True, help="prefix for out.tgz, log.txt and status.json")
    ap.add_argument("--cache", default=None, help="prefix of the shared k-series store")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")

    index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    out = join(a.out, f"array-{index}") if index is not None else a.out.rstrip("/")
    tmp = Path(tempfile.mkdtemp(prefix="msc-job-"))
    started = time.time()

    bundle = tmp / "code.tgz"
    if not get(a.code, bundle):
        print(f"no code bundle at {a.code}", file=sys.stderr)
        return 2
    with tarfile.open(bundle) as tf:
        tf.extractall(WORK, filter="data")  # "data": no absolute paths, no links out of /work

    cache_files: dict[str, int] = {}
    if a.cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        cached = tmp / "cache.tgz"
        if get(join(a.cache, "cache.tgz"), cached):
            with tarfile.open(cached) as tf:
                tf.extractall(CACHE, filter="data")
        cache_files = {str(p): p.stat().st_size for p in CACHE.rglob("*") if p.is_file()}

    build = WORK / "envs" / "build_kernels_if_stale.sh"
    if build.exists():
        subprocess.run([str(build)], cwd=WORK, env=dict(os.environ, PYTHON=sys.executable), check=True)

    mark = time.time()
    log = tmp / "log.txt"
    env = dict(os.environ, PYTHONPATH=str(WORK), PYTHONUNBUFFERED="1")
    print(f"[msc-job] {socket.gethostname()}: {' '.join(cmd)}", flush=True)
    stop = threading.Event()
    lock = threading.Lock()

    def upload_log() -> None:  # the partial log, while the command runs
        sent = -1
        while not stop.wait(20):
            with lock:
                size = log.stat().st_size
            if size != sent:
                try:
                    put(log, join(out, "log.txt"))
                    sent = size
                except Exception as e:  # noqa: BLE001  (a failed upload of a partial log must not end the job)
                    print(f"[msc-job] log upload failed: {e}", file=sys.stderr, flush=True)

    with log.open("w") as f:
        uploader = threading.Thread(target=upload_log, daemon=True)
        uploader.start()
        proc = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            with lock:
                f.write(line)
                f.flush()
        code = proc.wait()
    stop.set()
    uploader.join()

    files = changed_files(WORK, mark)
    results = tmp / "out.tgz"
    with tarfile.open(results, "w:gz") as tf:
        for p in files:
            tf.add(p, arcname=str(p.relative_to(WORK)))
    put(results, join(out, "out.tgz"))
    put(log, join(out, "log.txt"))
    status = {
        "exit_code": code,
        "command": cmd,
        "started": started,
        "finished": time.time(),
        "host": socket.gethostname(),
        "array_index": index,
        "files": [str(p.relative_to(WORK)) for p in files],
    }
    (tmp / "status.json").write_text(json.dumps(status, indent=1))
    put(tmp / "status.json", join(out, "status.json"))

    if a.cache:
        grown = {str(p): p.stat().st_size for p in CACHE.rglob("*") if p.is_file()} != cache_files
        if grown:
            packed = tmp / "cache-new.tgz"
            with tarfile.open(packed, "w:gz") as tf:
                tf.add(CACHE, arcname=".")
            # the store only grows: never replace a larger one (another worker's) with a smaller
            if packed.stat().st_size > size_of(join(a.cache, "cache.tgz")):
                put(packed, join(a.cache, "cache.tgz"))

    print(f"[msc-job] exit {code}; {len(files)} file(s) returned to {out}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
