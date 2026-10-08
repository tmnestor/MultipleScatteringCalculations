#!/usr/bin/env python3
"""The cost of a frequency sweep of the Legendre cell's single-site T-matrix, and the range of the series.

The closed form of ``derive_legendre_cell_dynamic.py`` gives the self block as K = sum_n (k_S h)^n K_n with
coefficients that depend on neither the frequency, the cell size nor the medium. A sweep then costs, per
frequency, a matrix polynomial and a 36 x 36 solve (or eight solves of at most 4 x 4 on the irreducible
representations). Measured here, per frequency, against

  quadrature: ``blocks.near_block`` (Duffy and Gauss rules for the singular static terms, Gauss rules for
              the dynamic remainder), recomputed at each frequency, then ``site.single_site_t36``;
  series:     ``blocks.near_block_series`` (the package's series of universal moments, cached after the first
              frequency), then ``single_site_t36``;

and the accuracy of the closed form truncated at n = N against quadrature over k_S h, which sets the range of
the truncated series (every further power is available in closed form).

Run:  python -u scripts/measure_legendre_cell_sweep.py     (after derive_legendre_cell_dynamic.py)
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from derive_legendre_cell_dynamic import (  # noqa: E402
    OUT,
    REF,
    irrep_t,
    numeric_k,
    reduce_on_irreps,
)

from cubic_scattering.graded_voxel.blocks import (  # noqa: E402
    near_block,
    near_block_series,
)
from cubic_scattering.graded_voxel.site import (  # noqa: E402
    contrast_operator,
    single_site_t36,
)

H = 1.0
DLAM, DMU, DRHO = 2.0e9, 1.0e9, 100.0


def delta_at(omega: float) -> np.ndarray:
    d = np.zeros((4, 9, 9), dtype=complex)
    d[0] = contrast_operator(DLAM, DMU, DRHO, omega)
    return d


def main() -> int:
    data = pickle.loads((OUT / "series_blocks.pkl").read_bytes())
    blocks, consts, n_max = data["blocks"], data["constants"], data["n_max"]
    values = {c: float(sp.N(c, 40)) for c in consts}
    gamma = REF.beta / REF.alpha

    # one-off: the numerical coefficient matrices K_n (h = 1), and the reduced blocks
    t0 = time.perf_counter()
    k_n = []
    for n in range(n_max + 1):
        # K_n alone: numeric_k at k_S h = 1 on a series holding only the power n
        trunc = [(blocks[m][0], blocks[m][1]) if m == n else ([[{}] * 36 for _ in range(36)],) * 2
                 for m in range(n + 1)]
        k_n.append(numeric_k(trunc, values, 1.0, H, gamma, REF.mu, n).reshape(4, 4, 9, 9)
                   .transpose(0, 2, 1, 3).reshape(36, 36))
    t_coeff = time.perf_counter() - t0
    t0 = time.perf_counter()
    reduced = reduce_on_irreps(blocks)
    t_reduce = time.perf_counter() - t0
    print(f"one-off: numerical K_0..K_{n_max} {t_coeff:.1f} s; exact reduction on the irreps {t_reduce:.1f} s")

    gram = np.kron(np.diag([8.0, 8 / 3, 8 / 3, 8 / 3]) * H**3, np.eye(9))

    def t36_closed(kh: float) -> np.ndarray:
        k = sum(kh**n * k_n[n] for n in range(n_max + 1))
        dl = np.kron(np.eye(4), delta_at(kh / H * REF.beta)[0])
        return gram @ dl @ np.linalg.solve(gram - k @ dl, gram)

    # the reduced blocks as numbers, once: N_Gamma = sum_n (k_S h)^n [w_A(n) NA_n + w_B(n) NB_n] / (4 pi mu)
    from derive_legendre_cell_dynamic import lin_value

    num = []
    for r in reduced.values():
        k = len(r["M"])
        na = np.array([[[lin_value(r["NA"][n][i][j], values) for j in range(k)] for i in range(k)]
                       for n in range(n_max + 1)])
        nb = np.array([[[lin_value(r["NB"][n][i][j], values) for j in range(k)] for i in range(k)]
                       for n in range(n_max + 1)])
        v = r["V"]
        num.append((na, nb, np.diag([float(x) for x in r["M"]]), np.linalg.solve(v.T @ v, v.T), v))
    pw = np.arange(n_max + 1)
    fact = np.array([float(sp.factorial(n)) for n in pw])
    fact2 = np.array([float(sp.factorial(n + 2)) for n in pw])

    def t_irreps(kh: float) -> list[np.ndarray]:
        d36 = np.kron(np.eye(4), delta_at(kh / H * REF.beta)[0])
        wa = (1j * kh) ** pw / fact
        wb = -((1j * kh) ** pw) * (1.0 - gamma ** (pw + 2)) / fact2
        out = []
        for na, nb, m_g, pinv, v in num:
            n_g = (np.tensordot(wa, na, 1) + np.tensordot(wb, nb, 1)) / (4.0 * np.pi * REF.mu)
            d_g = pinv @ d36 @ v
            out.append(m_g @ d_g @ np.linalg.inv(np.eye(len(m_g)) - n_g @ d_g))
        return out

    worst = max(np.abs(a - b).max() for a, b in zip(
        t_irreps(0.1), [irrep_t(r, values, 0.1, gamma, REF.mu, delta_at(0.1 * REF.beta)[0], n_max)
                        for r in reduced.values()], strict=True))
    print(f"numerical irrep blocks reproduce the exact ones to {worst:.1e} (absolute)")

    # the closed form's 36 x 36 against the single_site_t36 convention, once
    kh = 0.1
    t_q = single_site_t36(H, delta_at(kh * REF.beta), near_block((0, 0, 0), H, kh * REF.beta, REF, n_q=14))
    print(f"closed-form T36 against quadrature at k_S h = 0.1: "
          f"{np.abs(t36_closed(kh) - t_q).max() / np.abs(t_q).max():.1e}")

    # cost per frequency over a sweep (distinct frequencies, so no cache hits)
    sweep = np.linspace(0.01, 0.3, 24)
    rows = []
    t0 = time.perf_counter()
    for kh in sweep:
        single_site_t36(H, delta_at(kh * REF.beta), near_block((0, 0, 0), H, kh * REF.beta, REF, n_q=14))
    rows.append(("quadrature (near_block) + single_site_t36", (time.perf_counter() - t0) / len(sweep)))
    near_block_series((0, 0, 0), H, 0.5 * REF.beta, REF, n_source=10, n_test=4)  # fill the moment cache
    t0 = time.perf_counter()
    for kh in sweep:
        single_site_t36(H, delta_at(kh * REF.beta),
                        near_block_series((0, 0, 0), H, kh * REF.beta, REF, n_source=10, n_test=4))
    rows.append(("package series (near_block_series) + single_site_t36", (time.perf_counter() - t0) / len(sweep)))
    reps = 2000
    t0 = time.perf_counter()
    for i in range(reps):
        t36_closed(0.01 + 0.29 * i / reps)
    rows.append((f"closed form, 36 x 36 (n <= {n_max})", (time.perf_counter() - t0) / reps))
    t0 = time.perf_counter()
    for i in range(reps):
        t_irreps(0.01 + 0.29 * i / reps)
    rows.append((f"closed form, irreducible blocks (n <= {n_max})", (time.perf_counter() - t0) / reps))
    base = rows[0][1]
    print("cost per frequency:")
    for name, sec in rows:
        print(f"   {name:58s} {sec * 1e3:10.3f} ms   ({base / sec:8.0f}x)")

    # range: the truncated closed form against quadrature over k_S h
    print(f"accuracy of the closed form truncated at n = {n_max} (and at n = 4) against quadrature:")
    for kh in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        omega = kh * REF.beta
        t_q = single_site_t36(H, delta_at(omega), near_block((0, 0, 0), H, omega, REF, n_q=14))
        t_s = single_site_t36(H, delta_at(omega), near_block_series((0, 0, 0), H, omega, REF, n_source=10, n_test=4))
        dl = np.kron(np.eye(4), delta_at(omega)[0])
        errs = []
        for top in (n_max, 4):
            k_top = sum(kh**n * k_n[n] for n in range(top + 1))
            t_c = gram @ dl @ np.linalg.solve(gram - k_top @ dl, gram)
            errs.append(np.abs(t_c - t_q).max() / np.abs(t_q).max())
        print(f"   k_S h = {kh:<4}: n <= {n_max} {errs[0]:.1e}   n <= 4 {errs[1]:.1e}   "
              f"(package series against quadrature {np.abs(t_s - t_q).max() / np.abs(t_q).max():.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
