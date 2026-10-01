#!/usr/bin/env python3
"""The cube group block-diagonalises the single-site matrix M - K(0) E of a cell with uniform contrast.

The 36 unknowns psi_a (a over the test functions 1, xi_i; nine components each) carry the representation
D(Q) = T(Q) (x) S(Q) of the 48 signed permutations Q (``graded_voxel.fft.symmetry_reps``).  With a
contrast uniform in the cell, A = M - K(0) E commutes with every D(Q), so it reduces on the irreducible
representations of O_h.  Checked here:

  1. D is a representation, and A commutes with it;
  2. the multiplicities, by characters: 2 A1g + 2 Eg + T1g + 2 T2g (15, even) and
     4 T1u + 2 T2u + A2u + Eu (21, odd);
  3. on the subspace of an irrep of dimension d and multiplicity n, A has n distinct eigenvalues, each
     d-fold: A is an n x n block repeated d times, the largest 4 x 4;
  4. the inverse assembled from the blocks equals the 36 x 36 inverse, and so does T36;
  5. with a gradient along a cell axis only the eight operations that fix the axis commute with A;
  6. for a gradient g in a general direction, D(Q) A(g) D(Q)^T = A(Q g).

Characters of O_h from the signed permutation Q = det(Q) R, R a rotation with underlying permutation pi:
A1 = 1, A2 = sign(pi), E = fix(pi) - 1, T1 = tr R, T2 = sign(pi) tr R; g and u by the factor 1 and det Q.

Run:  conda run -n seismic python -u scripts/check_graded_voxel_site_symmetry.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel.basis import gram_test, gram_test_source, source_expansion  # noqa: E402
from cubic_scattering.graded_voxel.blocks import near_block  # noqa: E402
from cubic_scattering.graded_voxel.fft import signed_permutations, symmetry_reps  # noqa: E402
from cubic_scattering.graded_voxel.site import contrast_operator  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA, H = 150.0, 1.25
DIMS = {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}
EXPECTED = {
    "A1g": 2, "A2g": 0, "Eg": 2, "T1g": 1, "T2g": 2,
    "A1u": 0, "A2u": 1, "Eu": 1, "T1u": 4, "T2u": 2,
}  # fmt: skip


def characters(q: np.ndarray) -> dict[str, float]:
    det = round(float(np.linalg.det(q)))
    rot = q * det
    perm = np.abs(rot)
    sign = round(float(np.linalg.det(perm)))
    fix = float(np.trace(perm))
    base = {
        "A1": 1.0,
        "A2": sign,
        "E": fix - 1.0,
        "T1": float(np.trace(rot)),
        "T2": sign * float(np.trace(rot)),
    }
    out = {}
    for name, chi in base.items():
        out[name + "g"] = chi
        out[name + "u"] = chi * det
    return out


def site_matrices(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A, F, M) with T36 = F A^-1 M, A = M - K(0) E."""
    k0 = near_block((0, 0, 0), H, OMEGA, REF)
    e = source_expansion(delta)
    m = np.kron(gram_test(H), np.eye(9))
    ke = np.einsum("acij,cbjk->aibk", k0, e).reshape(36, 36)
    f = np.einsum("ac,cbij->aibj", gram_test_source(H), e).reshape(36, 36)
    return m - ke, f, m


def main() -> int:
    ok = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        ok.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

    qs = signed_permutations()
    reps = []
    for q in qs:
        t, _, s = symmetry_reps(q)
        reps.append(np.kron(t, s))
    base = contrast_operator(2.0e9, 1.0e9, 100.0, OMEGA)
    delta = np.zeros((4, 9, 9), dtype=complex)
    delta[0] = base
    a, f, m = site_matrices(delta)
    scale = np.abs(a).max()

    # 1. representation and commutation
    idx = {q.tobytes(): i for i, q in enumerate(qs)}
    hom = max(
        np.abs(reps[i] @ reps[j] - reps[idx[(qs[i] @ qs[j]).tobytes()]]).max()
        for i in (3, 17, 40)
        for j in (5, 22)
    )
    check("1a. D(Q1) D(Q2) = D(Q1 Q2)", hom < 1e-14, f"{hom:.1e}")
    comm = max(np.abs(d @ a - a @ d).max() for d in reps) / scale
    check("1b. A commutes with all 48 D(Q) (uniform contrast)", comm < 1e-9, f"{comm:.1e}")

    # 2. multiplicities
    chars = [characters(q) for q in qs]
    mult = {
        name: sum(c[name] * np.trace(d) for c, d in zip(chars, reps, strict=True)) / 48 for name in EXPECTED
    }
    got = {name: round(v) for name, v in mult.items()}
    check("2. multiplicities by characters", got == EXPECTED, str({k: v for k, v in got.items() if v}))

    # 3. and 4. blocks
    inv = np.zeros((36, 36), dtype=complex)
    largest, clustered = 0, []
    for name, n in EXPECTED.items():
        if n == 0:
            continue
        d = DIMS[name[:-1]]
        proj = sum(c[name] * dm for c, dm in zip(chars, reps, strict=True)) * d / 48
        u, sv, _ = np.linalg.svd(proj)
        basis = u[:, : n * d]  # orthonormal basis of the isotypic subspace
        assert sv[n * d - 1] > 0.5 > (sv[n * d] if n * d < 36 else 0.0)
        block = basis.conj().T @ a @ basis
        ev = np.sort_complex(np.linalg.eigvals(np.linalg.solve(basis.conj().T @ m @ basis, block)))
        # n distinct values, each d-fold
        order = np.argsort(np.round(ev.real, 9) + 1j * np.round(ev.imag, 9))
        ev = ev[order]
        clusters = [ev[0:1]]
        for value in ev[1:]:
            if abs(value - clusters[-1].mean()) < 1e-7 * max(1.0, abs(value)):
                clusters[-1] = np.append(clusters[-1], value)
            else:
                clusters.append(np.array([value]))
        sizes = sorted(len(c) for c in clusters)
        clustered.append(sizes == [d] * n)
        largest = max(largest, n)
        print(
            f"      {name:4s} dimension {d}, multiplicity {n}: block {n} x {n}, eigenvalue clusters {sizes}"
        )
        inv += basis @ np.linalg.solve(block, basis.conj().T)
    check(
        "3. every irrep gives n distinct eigenvalues, each d-fold",
        all(clustered),
        f"largest block {largest} x {largest}",
    )
    direct = np.linalg.inv(a)
    err = np.abs(inv - direct).max() / np.abs(direct).max()
    check("4a. the inverse assembled from the blocks equals the 36 x 36 inverse", err < 1e-9, f"{err:.1e}")
    t36 = f @ direct @ m
    err_t = np.abs(f @ inv @ m - t36).max() / np.abs(t36).max()
    check("4b. T36 from the blocks", err_t < 1e-9, f"{err_t:.1e}")

    # 5. a gradient along axis 0 leaves the eight operations that fix the axis
    delta[1] = 0.3 * base
    a_g, _, _ = site_matrices(delta)
    keep = [i for i, d in enumerate(reps) if np.abs(d @ a_g - a_g @ d).max() / scale < 1e-9]
    fixes = [i for i, q in enumerate(qs) if q[0, 0] == 1.0]
    check(
        "5. with a gradient along an axis, exactly the 8 operations fixing it commute",
        keep == fixes,
        f"{len(keep)}",
    )
    # 6. a gradient in a general direction: D(Q) carries the cell to the cell with the rotated gradient
    grad = np.array([0.3, -0.1, 0.2])

    def with_gradient(gv: np.ndarray) -> np.ndarray:
        dl = np.zeros((4, 9, 9), dtype=complex)
        dl[0] = base
        for i in range(3):
            dl[1 + i] = gv[i] * base
        return site_matrices(dl)[0]

    a_ref = with_gradient(grad)
    cov = max(
        np.abs(d @ a_ref @ d.T - with_gradient(q @ grad)).max() for q, d in zip(qs, reps, strict=True)
    )
    check(
        "6. covariance: D(Q) A(g) D(Q)^T = A(Q g) for a general gradient g",
        cov / scale < 1e-9,
        f"{cov / scale:.1e}",
    )
    print(f"{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
