"""Build the store of the near series' universal moments (package data, read by ``blocks.universal_moment``).

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (15 October: the near series per frequency).

WHAT.  U(m, idx, offset) = int W(s) d^idx r^m (R + s) ds on the unit half-side, for the touching orbits
(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), every derivative multi-index of ``family_tables`` (TB from m = 1: its
weight vanishes at m = -1), odd m from -1 to M, at the largest basis (n_source 35, n_test 10; the smaller bases
are its leading slices).  Odd m only: those go through the closed master integrals (seconds to tens of seconds
each), while even m are polynomials that a Gauss rule integrates exactly in milliseconds.  The moments depend on
neither the medium, the cell size nor the frequency: computed once, never again.

COVERAGE.  ``near_block_series`` sums m until (k_S 4 sqrt(3) h)^(m - 1) / (m - 1)! < 1e-17: M = 31 covers
omega h / beta <= 1/2.  Higher orders, or other offsets, are computed on the fly as before.

Resumable: each moment is checkpointed (NEAR_CHECKPOINT, default ~/.cache/cubic_scattering/near_moments); a final run assembles
scripts/data/near_series_moments.npz.

Run:  PYTHONPATH=. python scripts/build_near_series_moments.py [M] [workers]
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "data" / "near_series_moments.npz"
CHECKPOINT = Path(
    os.environ.get(
        "NEAR_CHECKPOINT", Path.home() / ".cache" / "cubic_scattering" / "near_moments"
    )
)
ORBITS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
N_SOURCE, N_TEST = 35, 10


def key(m: int, idx: tuple, offset: tuple) -> str:
    return f"{m}|{','.join(map(str, idx))}|{','.join(map(str, offset))}"


def jobs(m_max: int) -> list:
    from cubic_scattering.graded_voxel.blocks import family_tables

    ta, tb = family_tables()
    out = []
    for m in range(-1, m_max + 1, 2):
        idxs = sorted(
            set(ta) | (set(tb) if m > -1 else set()), key=lambda i: (len(i), i)
        )
        out += [(m, idx, off) for off in ORBITS for idx in idxs]
    return out


def compute(job) -> str:
    from cubic_scattering.graded_voxel.blocks import universal_moment_direct

    m, idx, off = job
    path = CHECKPOINT / (key(m, idx, off).replace("|", "_") + ".npy")
    if not path.exists():
        u = universal_moment_direct(
            m, idx, off, N_SOURCE, N_TEST
        )  # not the store being built
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, u)
        tmp.replace(path)
    return str(path)


def main() -> None:
    m_max = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 1)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    todo = jobs(m_max)
    # the costly orders last, so a partial run still covers the low frequencies
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for i, _ in enumerate(ex.map(compute, todo, chunksize=1), 1):
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} moments", flush=True)
    data = {
        key(m, idx, off): np.load(
            CHECKPOINT / (key(m, idx, off).replace("|", "_") + ".npy")
        )
        for m, idx, off in todo
    }
    data["meta"] = np.array([m_max, N_SOURCE, N_TEST])
    np.savez_compressed(OUT, **data)
    print(
        f"written {OUT.relative_to(ROOT)}: {len(todo)} moments, odd m to {m_max}, {OUT.stat().st_size / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
