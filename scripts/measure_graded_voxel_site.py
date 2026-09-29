#!/usr/bin/env python3
"""The graded single site's measured numbers, as quoted in the paper (section 3).

[1] The parity-breaking coupling of T36 against the gradient: the largest entry between the 15- and
    21-dimensional parity sets, relative to the largest entry, for gh = 0, 0.05, 0.1, 0.2.
[2] The Born error with a gradient (gh = 0.2 along axis 0) at kh = 0.025, 0.05, 0.1: the degree-1 input
    drops the incident field's quadratic moments, so the error should scale as (kh)^2.

Run:  conda run -n seismic python scripts/measure_graded_voxel_site.py
"""

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel.basis import TEST_EXPONENTS, gram_test, monomials  # noqa: E402
from cubic_scattering.graded_voxel.blocks import near_block  # noqa: E402
from cubic_scattering.graded_voxel.site import contrast_operator, single_site_t36  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
H = 1.25


def main() -> int:
    omega = 150.0
    base = contrast_operator(CON.Dlambda, CON.Dmu, CON.Drho, omega)
    par = np.array([(-1 if i < 3 else 1) * (1 if a == 0 else -1) for a in range(4) for i in range(9)])
    print("[1] parity-breaking coupling against gh")
    for g in (0.0, 0.05, 0.1, 0.2):
        t = single_site_t36(
            H, np.array([base, g * base, 0 * base, 0 * base]), near_block((0, 0, 0), H, omega, REF)
        )
        print(f"    gh = {g:4.2f}: {np.abs(t[np.ix_(par == 1, par == -1)]).max() / np.abs(t).max():.3e}")

    print("[2] Born error with a gradient (gh = 0.2) against kh")
    x, w = leggauss(12)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    lt = monomials(TEST_EXPONENTS, xi)
    amp = np.arange(1, 10) + 0.5j
    errs = []
    khs = (0.025, 0.05, 0.1)
    for kh in khs:
        om = kh / H * REF.beta
        b = 1e-9 * contrast_operator(CON.Dlambda, CON.Dmu, CON.Drho, om)
        delta = np.array([b, 0.2 * b, 0 * b, 0 * b])
        field = np.exp(1j * H * xi @ (np.array([1.0, 0, 0]) * kh / H))
        coeff = np.linalg.solve(gram_test(H), H**3 * (lt * ww) @ field)
        got = single_site_t36(H, delta, near_block((0, 0, 0), H, om, REF)) @ np.kron(coeff, amp)
        exact = (
            H**3
            * np.einsum("an,n,nij,j->ai", lt * ww, field, np.einsum("an,aij->nij", lt, delta), amp).ravel()
        )
        errs.append(np.linalg.norm(got - exact) / np.linalg.norm(exact))
        print(f"    kh = {kh:5.3f}: {errs[-1]:.3e}")
    for (k1, e1), (k2, e2) in zip(
        zip(khs, errs, strict=True), list(zip(khs, errs, strict=True))[1:], strict=False
    ):
        print(f"    slope {k1} -> {k2}: {np.log(e2 / e1) / np.log(k2 / k1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
