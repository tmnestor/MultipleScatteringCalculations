#!/usr/bin/env python3
"""The projection defect of the graded sphere's profile as the energy of its detail coefficients.

Each cubic cell divides into eight, so the cells at half-widths h, h/2, h/4, ... are the levels of an
octree.  Let V_j be the functions that are polynomials of degree p on every cell of level j (p = 0 is the
Haar case).  The spaces are nested, V_0 in V_1 in ..., so with P_j the L2 projection onto V_j,

    |s - P_0 s|^2 = sum_{j >= 0} E_j,      E_j = |P_(j+1) s - P_j s|^2 = |P_(j+1) s|^2 - |P_j s|^2,

and E_j is the sum of the squares of the level-j detail (wavelet) coefficients of s in an orthonormal
basis of V_(j+1) minus V_j: seven Haar wavelets per cell for p = 0, and for p >= 1 the multiwavelets
whose scaling functions are the cells' Legendre polynomials.

Checked here, for the smoothstep profile of ``measure_graded_voxel_resolution.py``:
  1. the detail energies sum to the defect D_p measured directly on the cells of the solver's grid;
  2. the energy falls by 4^(p+1) per level (the defect is of order h^(2p+2));
  3. the detail energy is concentrated: the share of level-0 cells that carries 90% of E_0.

Run:  conda run -n seismic python -u scripts/measure_graded_voxel_wavelet_detail.py [core_frac] [n ...]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.graded_voxel.basis import CONTRAST_NORMS, contrast_values  # noqa: E402
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402
from measure_graded_voxel_resolution import projection_defect  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS  # noqa: E402

LEVELS = 4  # detail levels summed: cells of half-width h down to h / 16
N_FUN = {0: 1, 1: 4, 2: 10}


def profile(r: np.ndarray, core: float) -> np.ndarray:
    """The smoothstep profile s5, vectorised: 1 in the core, 10x^3 - 15x^4 + 6x^5 across the shell."""
    x = np.clip((RADIUS - r) / (RADIUS - core), 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def projection_energy(centres: np.ndarray, h: float, core: float, level: int) -> tuple[np.ndarray, float]:
    """Per level-0 cell and per degree p: |P_level s|^2 over that cell; and int s^2 over all cells.

    Returns (energy[cell, p], total), the cell's 8^level sub-cells projected with a 6-point Gauss rule.
    """
    x, w = np.polynomial.legendre.leggauss(6)
    xi = np.stack([g.ravel() for g in np.meshgrid(x, x, x, indexing="ij")], axis=1)
    wt = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()
    basis = contrast_values(xi)  # (10, G)
    norms = np.array(CONTRAST_NORMS)
    m = 2**level
    hs = h / m
    shifts = (2 * np.arange(m) + 1 - m) * hs  # sub-cell centre offsets along one axis
    sub = np.stack([g.ravel() for g in np.meshgrid(shifts, shifts, shifts, indexing="ij")], axis=1)
    energy = np.zeros((len(centres), 3))
    total = 0.0
    for i, c in enumerate(centres):
        pts = (c + sub)[:, None, :] + hs * xi[None, :, :]  # (sub-cells, G, 3)
        vals = profile(np.linalg.norm(pts, axis=2), core)  # (sub-cells, G)
        coef = (vals * wt) @ basis.T / norms  # (sub-cells, 10)
        sq = coef**2 * norms * hs**3
        total += float(((vals**2) @ wt).sum() * hs**3)
        for p, n_fun in N_FUN.items():
            energy[i, p] = sq[:, :n_fun].sum()
    return energy, total


def main() -> int:
    args = sys.argv[1:]
    core = RADIUS * (float(args[0]) if args else 0.5)
    ns = [int(a) for a in args[1:]] or [8, 12, 16]
    ok = []
    print(f"core = {core / RADIUS} a; detail levels 0..{LEVELS - 1}")
    for n in ns:
        _, centres, h = _build_grid_index_map(
            RADIUS, n, lambda q, _h=RADIUS / n: bool(np.linalg.norm(q) < RADIUS + np.sqrt(3) * _h)
        )
        levels = [projection_energy(centres, h, core, j) for j in range(LEVELS + 1)]
        total = levels[-1][1]
        for p in (0, 1, 2):
            e = np.array(
                [levels[j + 1][0][:, p] - levels[j][0][:, p] for j in range(LEVELS)]
            )  # (level, cell)
            per_level = e.sum(axis=1) / total
            summed = per_level.sum()
            # the levels not computed: a geometric tail at the ratio of the last two
            ratio = per_level[-1] / per_level[-2]
            tail = per_level[-1] * ratio / (1.0 - ratio)
            if p == 0:
                direct = float((total - levels[0][0][:, 0].sum()) / total)
            else:
                direct = projection_defect("s5", core, centres, h, degree=p)
            share = np.sort(e[0])[::-1].cumsum() / e[0].sum()
            cells_90 = int(np.searchsorted(share, 0.9) + 1)
            agree = abs((summed + tail) / direct - 1.0)
            ok.append(agree < 2e-2)
            print(
                f"  n {n:3d} p {p}: defect {direct:.4e}; details {summed:.4e} + tail {tail:.1e} "
                f"(agree to {agree:.1e}); per level " + " ".join(f"{v:.2e}" for v in per_level)
            )
            print(
                "           level ratios "
                + " ".join(f"{per_level[j] / per_level[j + 1]:.1f}" for j in range(LEVELS - 1))
                + f" (4^(p+1) = {4 ** (p + 1)}); 90% of the level-0 detail energy in "
                f"{cells_90} of {len(centres)} cells ({cells_90 / len(centres):.0%})"
            )
    print(f"{sum(ok)}/{len(ok)} sums agree with the directly measured defect")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
