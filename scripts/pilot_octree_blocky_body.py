#!/usr/bin/env python3
"""Pilot: a medium that is piecewise constant on the leaves: Haar for the medium, Legendre for the field.

THE BODY.  A cube of half-side B with a weak uniform contrast (OUTER times the central one), holding at its
centre a cube of half-side B / 4 with the full contrast.  Both are unions of octree cells, so a tree whose
leaves follow them represents the medium EXACTLY with a constant in each leaf: the relative projection
error is zero, and what is left is the error of the field's basis alone.

THE TREE.  The outer cube as 64 leaves of half-width B / 4, the eight at the centre split once more: 56
leaves of half-width B / 4 and 64 of half-width B / 8, 120 in all.  A uniform grid that represents the
same medium exactly needs the small size everywhere: 512 cells.

MEASURED, with a constant contrast in every leaf:
  - the tree with the field degree 0, 1 and 2 in every leaf, and with the degree chosen leaf by leaf (the
    lowest whose first-order error, against the leaf's eight children with quadratic fields, is below tol);
  - the uniform grid of 512 cells of half-width B / 8, which represents the medium exactly, and that of 64
    cells of half-width B / 4, which does NOT (the inner cube's faces fall inside cells), each with field
    degree 0 and 1.
For each: cells, unknowns, the first-order error predicted with no solve, and the error against the
reference.  There is no exact solution for this body: the reference is the tree with quadratic fields,
whose own first-order error is printed.

Run:  conda run -n seismic python -u scripts/pilot_octree_blocky_body.py [k_S B] [--dry] [--tol=1e-3,1e-4]
          [--summary=path]
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    _descendants,
    born_octree,
    leaf_energies,
    octree_far_field,
    refine_leaves,
    solve_graded_octree,
)
from gate_sphere_cell_average_vs_mie import CONTRAST, REF  # noqa: E402

HALF_SIDE = 5.0
OUTER = 0.25
K_HAT = np.array([1.0, 0.0, 0.0])
THETA = np.linspace(0.2, np.pi - 0.2, 9)
N_FUN = (1, 4, 10)
DENSE_LIMIT = 20000
R_FAR = 5e8 * HALF_SIDE


def prof_vec(pos: np.ndarray) -> np.ndarray:
    m = np.abs(pos).max(axis=-1)
    return np.where(m < HALF_SIDE / 4, 1.0, np.where(m < HALF_SIDE, OUTER, 0.0))


def prof(pos: np.ndarray) -> float:
    return float(prof_vec(np.asarray(pos)))


def cube_leaves(n: int) -> tuple[np.ndarray, np.ndarray]:
    h = HALF_SIDE / n
    ticks = (np.arange(n) + 0.5) * 2 * h - HALF_SIDE
    return np.array(list(itertools.product(ticks, repeat=3))), np.full(n**3, h)


def body_tree() -> tuple[np.ndarray, np.ndarray]:
    """64 leaves of half-width B / 4, the eight at the centre split once."""
    c, h = cube_leaves(4)
    return refine_leaves(c, h, np.abs(c).max(axis=1) < HALF_SIDE / 2)


def directions() -> np.ndarray:
    out = np.zeros((len(THETA), 3))
    out[:, 0], out[:, 1] = np.cos(THETA), np.sin(THETA)
    return out


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    dry = "--dry" in sys.argv
    kb = float(args[0]) if args else 2.0
    tols = [float(v) for v in opts.get("tol", "1e-3,1e-4").split(",") if v]
    omega = kb * REF.beta / HALF_SIDE
    dirs = directions()
    t0 = time.perf_counter()
    cache: dict = {}

    def born(centres: np.ndarray, hs: np.ndarray, p) -> np.ndarray:
        res = born_octree(omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p, r=0)
        return sum(octree_far_field(res, dirs, R_FAR))

    def solve(centres: np.ndarray, hs: np.ndarray, p) -> np.ndarray:
        res = solve_graded_octree(
            omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p, r=0, block_cache=cache
        )
        return sum(octree_far_field(res, dirs, R_FAR))

    ct, ht = body_tree()
    defect, _, norm = leaf_energies(prof_vec, ct, ht, 0)
    print(
        f"cube in a cube: k_S B = {kb} (k_S h = {kb / 4:.3f} and {kb / 8:.3f} on the tree's leaves), outer "
        f"contrast {OUTER}; tree of {len(ht)} leaves, projection error {defect.sum() / norm.sum():.1e}",
        flush=True,
    )
    # the first-order far field of the exact medium: every leaf's eight children, quadratic fields
    kids = np.concatenate([_descendants(c, float(h), 1)[0] for c, h in zip(ct, ht, strict=True)])
    kid_h = np.repeat(ht / 2, 8)
    born_fine = born(kids, kid_h, 2)
    peak_born = float(np.abs(born_fine).max())
    # per leaf and degree: the leaf's own first-order error
    leaf_err = np.zeros((3, len(ht)))
    for i, (c, h) in enumerate(zip(ct, ht, strict=True)):
        fine = born(*_descendants(c, float(h), 1)[:1], np.full(8, h / 2), 2)
        for p in range(3):
            leaf_err[p, i] = np.abs(born(c[None, :], np.array([h]), p) - fine).max() / peak_born

    cases = [(f"tree, field {p}", ct, ht, np.full(len(ht), p)) for p in (0, 1, 2)]
    for tol in tols:
        chosen = np.array([next((p for p in range(3) if leaf_err[p, i] < tol), 2) for i in range(len(ht))])
        cases.append((f"tree, field chosen, tol {tol:g}", ct, ht, chosen))
    for n, label in ((8, "uniform 512"), (4, "uniform 64, inexact")):
        cu, hu = cube_leaves(n)
        cases += [(f"{label}, field {p}", cu, hu, np.full(len(hu), p)) for p in (0, 1)]

    reference = None
    if not dry:
        reference = solve(ct, ht, 2)
        print(
            f"   reference solved (tree, quadratic fields)   [{time.perf_counter() - t0:.0f} s]", flush=True
        )
    peak = None if dry else float(np.abs(reference).max())
    print(
        "   case: cells | unknowns | first-order error (no solve) | error against the reference | degrees"
    )
    rows = []
    for name, centres, hs, p_leaf in cases:
        unknowns = int(sum(N_FUN[int(p)] * 9 for p in p_leaf))
        first = float(np.abs(born(centres, hs, p_leaf) - born_fine).max() / peak_born)
        degrees = {int(p): int((p_leaf == p).sum()) for p in np.unique(p_leaf)}
        row = {
            "case": name,
            "cells": len(hs),
            "unknowns": unknowns,
            "first_order": first,
            "degrees": degrees,
        }
        if not dry and unknowns <= DENSE_LIMIT and name != "tree, field 2":
            row["error"] = float(np.abs(solve(centres, hs, p_leaf) - reference).max() / peak)
        rows.append(row)
        true = f"{row['error']:.3e}" if "error" in row else "   -     "
        print(
            f"   {name:30s}: {row['cells']:4d} | {unknowns:6d} | {first:.3e} | {true} | {degrees}"
            f"   [{time.perf_counter() - t0:.0f} s]",
            flush=True,
        )
    if "summary" in opts:
        path = Path(opts["summary"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ks_b": kb, "outer": OUTER, "rows": rows}, indent=2) + "\n")
        print(f"   wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
