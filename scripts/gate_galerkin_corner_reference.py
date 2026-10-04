#!/usr/bin/env python3
"""Galerkin blocks of cells touching at a corner or an edge, against a high-precision reference.

Three routes compute the touching blocks of ``graded_voxel.blocks`` and, for the static (Kelvin) part,
they disagree pairwise by the same amount (2e-13 for linear cells at the corner, 1.5e-11 for quadratic
cells), while quadrature agrees with itself between 14 and 20 points to 1e-15. This settles which is
right with a reference that shares none of their integration:

    K_ac = int_{[-1,1]^6} xi^a xi'^c P(R + xi - xi') dxi dxi' = int_{[-2,2]^3} W_ac(s) P(R + s) ds,

half-width 1, R = 2 o. For corner and edge contact every static term d^idx r^m, even d^4 r ~ r^-3, is
absolutely integrable: W_ac vanishes as rho^3 (corner) or rho^2 (edge) at the singular point s* = -R, a
vertex of the pieces of W. So no derivative is moved and no distribution arises; the integral is taken
directly (``Mathematica/GradedVoxel_CornerReference.wl``) at 40 digits, W_ac by exact integration from
its definition and Duffy pyramids at s*.

    python scripts/gate_galerkin_corner_reference.py export    # the spec the Mathematica script reads
    wolframscript -file Mathematica/GradedVoxel_CornerReference.wl <cell> <n>
    python scripts/gate_galerkin_corner_reference.py compare <cell>

The static part is compared at k_S h = 1e-12, where the dynamic part of the real part is below 1e-23.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel import blocks  # noqa: E402
from cubic_scattering.graded_voxel.basis import (  # noqa: E402
    SOURCE_EXPONENTS,
    field_in_monomials,
    source_exponents,
)

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
SPEC = ROOT / "Mathematica" / "GradedVoxel_CornerReference_spec.json"
OUT = ROOT / "Mathematica" / "GradedVoxel_corner_reference_{cell}.json"
OFFSETS = {"corner": (1, 1, 1), "edge": (1, 1, 0)}
CELLS = {"linear": (10, 4), "quadratic": (35, 10)}


def export() -> None:
    ta, tb = blocks.family_tables()
    spec = {
        "medium": {"alpha": REF.alpha, "beta": REF.beta, "rho": REF.rho},
        "offsets": OFFSETS,
        "test_exponents": [list(e) for e in SOURCE_EXPONENTS[:10]],
        "source_exponents": [list(e) for e in source_exponents(35)],
        "field_in_monomials": field_in_monomials(10).tolist(),
        # the 9 x 9 coefficient of d^idx r^m: m = -1 from TA (G = delta_ij / r), m = 1 from TB (d_i d_j r)
        "terms": [{"m": -1, "idx": list(k), "coef": ta[k].tolist()} for k in sorted(ta)]
        + [{"m": 1, "idx": list(k), "coef": tb[k].tolist()} for k in sorted(tb)],
    }
    SPEC.write_text(json.dumps(spec))
    print(f"wrote {SPEC.relative_to(ROOT)}: {len(spec['terms'])} static terms")


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare(cell: str) -> int:
    n_source, n_test = CELLS[cell]
    data = json.loads(Path(str(OUT).format(cell=cell)).read_text())
    h = 1.0
    omega = 1e-12 * REF.beta / h
    for name, off in OFFSETS.items():
        # field rows (the identity for the four rows of a linear field)
        ref_blk = np.array(data[name]["block"], dtype=float).reshape(n_test, n_source, 9, 9)
        series = blocks.near_block_series(off, h, omega, REF, n_source, n_test).real
        blocks._NEAR_CACHE.clear()
        quad = blocks.near_block(off, h, omega, REF, n_q=20, n_source=n_source, n_test=n_test).real
        blocks._NEAR_CACHE.clear()
        closed = blocks.near_block(
            off, h, omega, REF, n_q=20, n_source=n_source, n_test=n_test, static="closed"
        ).real
        print(f"{cell} {name}: |K| = {np.linalg.norm(ref_blk):.6e}", flush=True)
        for label, blk in (("series", series), ("quadrature", quad), ("closed", closed)):
            err = rel(blk, ref_blk)
            parts = {
                p: rel(blk[..., sl_r, sl_c], ref_blk[..., sl_r, sl_c])
                for p, (sl_r, sl_c) in {
                    "G": (slice(0, 3), slice(0, 3)),
                    "C": (slice(0, 3), slice(3, 9)),
                    "H": (slice(3, 9), slice(0, 3)),
                    "S": (slice(3, 9), slice(3, 9)),
                }.items()
            }
            print(
                f"  {label:10s} against the reference {err:.1e}   by block "
                + "  ".join(f"{p} {e:.1e}" for p, e in parts.items()),
                flush=True,
            )
        # per term, the series' universal moments against the reference's
        worst = []
        for term in data[name]["terms"]:
            u_ref = np.array(term["U"], dtype=float)[:n_test, :n_source]
            u_py = blocks.universal_moment(term["m"], tuple(term["idx"]), off, n_source, n_test)
            scale = np.abs(u_ref).max()
            if scale > 0:
                worst.append((float(np.abs(u_py - u_ref).max() / scale), term["m"], tuple(term["idx"])))
        worst.sort(reverse=True)
        print(
            "  universal moments, worst terms (max entry / max |U|): "
            + ", ".join(f"m={m} {idx}: {e:.1e}" for e, m, idx in worst[:5]),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
        sys.exit(0)
    sys.exit(compare(sys.argv[2]))
