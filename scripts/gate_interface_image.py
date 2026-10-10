"""Gate: the static interface image blocks of ``cubic_scattering.interface_image`` (route A).

Plan: ``docs/2026-10-10-stratified-reference-legendre-cells-3d.md``, option 1.

I1  THE CORNER.  On a piece holding the singular corner, Euler's reduction to the three far faces against
    mpmath's tanh-sinh quadrature of the volume integral itself (independent: tanh-sinh integrates the
    singular corner directly, the reduction never does).  Measured, 10 October 2026:
        Phi0 d(1,1,0) s(1,0,1)  2.0e-15     Phi1 d(2,0,0) s(0,1,1)  6.0e-16     Phi1 d(1,0,0) s(0,0,1)  7.2e-16
        Phi2 d(1,1,0) s(0,0,1)  2.0e-15     Phi2 d(4,0,0) s(0,0,2)  3.8e-14     Phi0 d(0,0,2) s(1,1,2)  1.3e-14
        Phi1 d(2,2,0) s(2,0,2)  4.8e-14
I2  THE BLOCK.  Whole blocks between non-touching cells against brute-force 6-D Gauss over both cells of the
    canonical kernel (independent of the s-form, the exact weights and the pieces).  Measured:
        (-1,0,0) <- (-3,0,0)  p<=1, 10 sources  7.6e-15        (-1,0,0) <- (-3,2,0)   p<=1, 10 sources  8.5e-15
        (-3,0,0) <- (-1,2,-2) p=2,  20 sources  4.9e-15        (-1,0,0) <- (-1,4,0)   p<=1, 10 sources  3.3e-15
    (brute force converged to 1e-13 between orders 10 and 12).

I3  THE TOUCHING BLOCK, END TO END (recorded, not re-run: an hour).  A cell touching the interface with its
    own image, entry (field 1, source 1), against the 2-D Fourier integral of the Mathematica static spectral
    kernel (StaticInterfaceImage.json) times the cells' plane-wave moments, which converges only
    algebraically.  Refining cutoff and rules together, the reference approaches the engine:
        Qh = 320: 7.9e-6,   1280: 1.6e-6,   5120: 4.5e-7,   20480: 2.3e-8,
    while the engine does not move (face rule of order 32 or 64).  An earlier 4e-4 was the reference's own
    64-point azimuth rule.

T1  THE TRANSMISSION, IDENTICAL MEDIA.  With A = B the transmitted static field is Kelvin's, so a block across
    the interface must be Paper 2's static block (``graded_voxel.blocks.coupling_block`` at omega = 1e-6, real
    part) -- the touching corner at the interface included.  Measured, 10 October 2026:
        face 1.9e-15, edge 3.1e-15, corner 6.7e-15 (receiver below, p<=1, 10 sources); receiver above (the
        mirror) 4.3e-15; mirror, corner, p=2, 20 sources 7.6e-15; face, p=2, 35 sources 3.9e-15.
    And the image vanishes identically for A = B, on both sides.
T2  THE TRANSMISSION, DIFFERENT MEDIA.  The canonical kernel against direct 2-D Fourier integration of the
    spectral one (scripts/data/static_interface_transmission.json): 7.8e-13 and 6.6e-13.  Non-touching blocks
    against brute-force 6-D Gauss (order 10): 2.1e-13, 1.7e-13, 5.0e-13.

By default one case of each is run; ``--full`` runs all of them (tanh-sinh at 15 digits takes up to an hour
for the fourth derivatives).

Run:  PYTHONPATH=. python scripts/gate_interface_image.py [--full]
"""

import sys

import mpmath as mp
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss, legval

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel.basis import SOURCE_EXPONENTS_QUARTIC
from cubic_scattering.interface_image import (
    _corner_moments,
    _terms,
    image_block,
    kernel_degree,
    kernel_function,
)

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05

CORNERS = [
    ((1, (1, 0, 0)), (0, 2), (0, 2), (0, 2), (0, 0, 1)),
    ((0, (1, 1, 0)), (-2, 0), (0, 2), (0, 2), (1, 0, 1)),
    ((1, (2, 0, 0)), (0, 2), (-2, 0), (0, 2), (0, 1, 1)),
    ((2, (1, 1, 0)), (-2, 0), (-2, 0), (0, 2), (0, 0, 1)),
    ((2, (4, 0, 0)), (0, 2), (0, 2), (0, 2), (0, 0, 2)),
    ((0, (0, 0, 2)), (0, 2), (0, 2), (0, 2), (1, 1, 2)),
    ((1, (2, 2, 0)), (0, 2), (-2, 0), (0, 2), (2, 0, 2)),
]
BLOCKS = [
    ((-1, 0, 0), (-3, 0, 0), 10, 4),
    ((-1, 0, 0), (-3, 2, 0), 10, 4),
    ((-3, 0, 0), (-1, 2, -2), 20, 10),
    ((-1, 0, 0), (-1, 4, 0), 10, 4),
]


def gate_corner(case: tuple) -> bool:
    """I1 for one kernel, box and monomial."""
    (j, alpha), bx, by, bz, n = case
    rx, ry, zeta = sp.symbols("rx ry zeta", real=True)
    r = sp.sqrt(rx**2 + ry**2 + zeta**2)
    phi = {
        0: 1 / (2 * sp.pi * r),
        1: -sp.log(r + zeta) / (2 * sp.pi),
        2: (zeta * sp.log(r + zeta) - r) / (2 * sp.pi),
    }[j]
    expr = (
        sp.diff(phi, rx, alpha[0], ry, alpha[1], zeta, alpha[2]) if sum(alpha) else phi
    )
    g = sp.lambdify(
        (rx, ry, zeta), expr * rx ** n[0] * ry ** n[1] * zeta ** n[2], "mpmath"
    )
    mp.mp.dps = 15
    ref = float(mp.quad(g, bx, by, bz))
    got = _corner_moments(
        kernel_function(j, alpha), kernel_degree(j, alpha), (bx, by, bz), n, 40
    )[n]
    err = abs(got - ref) / abs(ref)
    ok = err < 1e-12
    print(
        f"I1  Phi{j} d{alpha} s{n} box {bx, by, bz}: {err:.1e}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def brute_block(
    rc: tuple, sc: tuple, n_source: int, n_test: int, n: int, kind: str = "reflected"
) -> np.ndarray:
    """The block by a tensor Gauss rule of order n over both cells (non-touching cells only)."""
    x, w = leggauss(n)
    v = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    wt = np.einsum("i,j,k->ijk", w, w, w).ravel()
    xr = np.array(rc, float) * H + H * v
    xs = np.array(sc, float) * H + H * v
    z, zp = xr[:, 0][:, None], xs[:, 0][None, :]
    zeta = -(z + zp) if kind == "reflected" else z - zp
    rx = xr[:, 1][:, None] - xs[:, 1][None, :]
    ry = xr[:, 2][:, None] - xs[:, 2][None, :]
    fa = np.array(
        [
            np.prod([legval(v[:, i], [0] * e[i] + [1]) for i in range(3)], 0)
            for e in SOURCE_EXPONENTS_QUARTIC[:n_test]
        ]
    )
    mc = np.array(
        [
            np.prod([v[:, i] ** e[i] for i in range(3)], 0)
            for e in SOURCE_EXPONENTS_QUARTIC[:n_source]
        ]
    )
    mats = (A.lam, A.mu, B.lam, B.mu)
    out = np.zeros((n_test, n_source, 9, 9))
    cache = {}
    for r, c, j, alpha, ap, bp, fn in _terms(kind):
        key = (j, alpha, ap, bp)
        if key not in cache:
            k = kernel_function(j, alpha)(rx, ry, zeta) * z**ap * zp**bp
            cache[key] = np.einsum("m,am,mn,n,cn->ac", wt, fa, k, wt, mc) * H**6
        out[:, :, r, c] += fn(*mats) * cache[key]
    return out


def gate_block(case: tuple) -> bool:
    """I2 for one pair of cells."""
    rc, sc, ns, nt = case
    got = image_block(rc, sc, H, A, B, n_source=ns, n_test=nt)
    ref = brute_block(rc, sc, ns, nt, 10)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    ok = err < 1e-11
    print(
        f"I2  {rc} <- {sc}, {nt} field x {ns} source: {err:.1e}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def gate_transmission_identical(
    rc: tuple, sc: tuple, ns: int = 10, nt: int = 4
) -> bool:
    """T1 for one pair across the interface, with identical media."""
    from cubic_scattering.graded_voxel.blocks import coupling_block

    off = tuple((a - b) // 2 for a, b in zip(rc, sc, strict=True))
    got = image_block(rc, sc, H, A, A, n_source=ns, n_test=nt)
    ref = coupling_block(off, H, 1e-6, A, n_source=ns, n_test=nt).real
    err = np.abs(got - ref).max() / np.abs(ref).max()
    ok = err < 1e-13
    print(f"T1  identical media {rc} <- {sc}: {err:.1e}  {'PASS' if ok else 'FAIL'}")
    return ok


def gate_transmission_brute(rc: tuple, sc: tuple, ns: int = 10, nt: int = 4) -> bool:
    """T2 for one non-touching pair across the interface, different media."""
    got = image_block(rc, sc, H, A, B, n_source=ns, n_test=nt)
    ref = brute_block(rc, sc, ns, nt, 10, kind="transmitted")
    err = np.abs(got - ref).max() / np.abs(ref).max()
    ok = err < 1e-11
    print(
        f"T2  {rc} <- {sc}, {nt} field x {ns} source: {err:.1e}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


if __name__ == "__main__":
    full = "--full" in sys.argv
    results = [gate_corner(c) for c in (CORNERS if full else CORNERS[:1])]
    results += [gate_block(b) for b in (BLOCKS if full else BLOCKS[:1])]
    results += [gate_transmission_identical((1, 0, 0), (-1, 0, 0))]
    results += [gate_transmission_brute((1, 0, 0), (-3, 0, 0))]
    if full:
        results += [
            gate_transmission_identical((1, 0, 0), (-1, 2, -2)),
            gate_transmission_identical((-1, 0, 0), (1, 0, 0)),
        ]
        results += [gate_transmission_brute((3, 2, 0), (-1, 0, 0))]
    print("ALL PASS" if all(results) else "FAILURES")
