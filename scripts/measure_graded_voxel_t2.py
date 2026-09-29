"""The second-order Born term T2 of the graded voxel against the exact one, and its split by block type.

Scheme: psi0 = M^-1 <L, psi_inc>; psi1_m = M^-1 sum_n K(o_mn) E_n psi0_n; T2 = far field of Delta psi1.
Exact:  T2 = (f(d) + f(-d)) / (2 d^2) from the graded reference at contrast +-d (O(d^2) accurate).
Split:  the sum over n restricted to the self cell, the 26 touching cells, or the rest.
Channels: any of rho, lam, mu joined by "+"; p = 0 (mean only) or 1 (graded first moment).

Run:  conda run -n seismic python -u scripts/measure_graded_voxel_t2.py <k_S a> <p> <channels> <n ...>
      e.g. ... 0.5 1 rho+lam+mu 4 6 8 10 12  (the paper's T2 sequence)
"""

import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, str(W))
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from cubic_scattering import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.basis import gram_test, source_expansion  # noqa: E402
from cubic_scattering.graded_voxel.blocks import coupling_block  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.site import cell_contrast_coefficients  # noqa: E402
from cubic_scattering.graded_voxel.solver import GradedVoxelResult, plane_wave_moments  # noqa: E402
from cubic_scattering.sphere_scattering import (  # noqa: E402
    _plane_wave_strain_voigt,
    mie_scattered_displacement,
)
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST as GATE  # noqa: E402
from gate_sphere_cell_average_vs_mie import REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import CORE, RADIUS, THETA, profile  # noqa: E402

K = np.array([1.0, 0, 0])
CHANNEL = sys.argv[3]
CONTRAST = MaterialContrast(
    GATE.Dlambda if "lam" in CHANNEL else 0.0,
    GATE.Dmu if "mu" in CHANNEL else 0.0,
    GATE.Drho if "rho" in CHANNEL else 0.0,
)
ka = float(sys.argv[1])
P_DEG = int(sys.argv[2])
ns = [int(a) for a in sys.argv[4:]]
omega = ka * REF.beta / RADIUS
rf = 5e8 * RADIUS
pts = obs_points(rf, THETA)
d = 1e-3
fp = mie_scattered_displacement(
    graded_mie_result(
        omega,
        RADIUS,
        CORE,
        REF,
        MaterialContrast(d * CONTRAST.Dlambda, d * CONTRAST.Dmu, d * CONTRAST.Drho),
        12,
    ),
    pts,
)
fm = mie_scattered_displacement(
    graded_mie_result(
        omega,
        RADIUS,
        CORE,
        REF,
        MaterialContrast(-d * CONTRAST.Dlambda, -d * CONTRAST.Dmu, -d * CONTRAST.Drho),
        12,
    ),
    pts,
)
t2_exact = (fp + fm) / (2 * d * d)
t1_exact = (fp - fm) / (2 * d)
kP = omega / REF.alpha
amp = np.concatenate([K.astype(complex), _plane_wave_strain_voigt(K, K, kP)])
for n in ns:
    h = RADIUS / n
    grid, centres, _ = _build_grid_index_map(
        RADIUS, n, lambda q, _h=h: np.linalg.norm(q) < RADIUS + np.sqrt(3) * _h
    )
    N = len(centres)
    delta = np.array(
        [cell_contrast_coefficients(profile, c, h, CONTRAST, REF, omega, P_DEG) for c in centres]
    )
    E = np.array([source_expansion(dd) for dd in delta])  # (N, 10, 4, 9, 9)
    minv = np.linalg.inv(gram_test(h))
    psi0 = np.einsum("ab,nbi->nai", minv, plane_wave_moments(centres, h, kP * K, amp))
    if P_DEG == 0:
        psi0[:, 1:] = 0.0
    src0 = np.einsum("ncbij,nbj->nci", E, psi0)  # (N, 10, 9): source monomial coefficients
    parts = {
        "self": np.zeros((N, 4, 9), complex),
        "touch": np.zeros((N, 4, 9), complex),
        "far": np.zeros((N, 4, 9), complex),
    }
    blocks: dict = {}
    for m in range(N):
        for q in range(N):
            off = tuple(int(v) for v in grid[m] - grid[q])
            if off not in blocks:
                blocks[off] = coupling_block(off, h, omega, REF)
            key = "self" if off == (0, 0, 0) else "touch" if max(map(abs, off)) <= 1 else "far"
            parts[key][m] += np.einsum("acij,cj->ai", blocks[off], src0[q])
    errs = {}
    total = sum(parts.values())
    for label, rhs in [("all", total)] + [(k, v) for k, v in parts.items()]:
        psi1 = np.einsum("ab,nbi->nai", minv, rhs)
        if P_DEG == 0:
            psi1 = np.zeros_like(psi1)
            psi1[:, 0] = rhs[:, 0] / gram_test(h)[0, 0]
        res = GradedVoxelResult(centres, grid, h, omega, REF, delta, psi1, P_DEG, P_DEG)
        up, us = graded_far_field(res, pts / rf, rf, K, K, "P")
        errs[label] = up + us
    scale = np.abs(t2_exact).max()
    err_all = np.abs(errs["all"] - t2_exact).max() / scale
    sh = {k: np.abs(errs[k]).max() / scale for k in ("self", "touch", "far")}
    print(
        f"{CHANNEL} p={P_DEG} n={n}: |T2 scheme - T2 exact|/|T2| = {err_all:.4e}; "
        f"|T2|/|T1| = {scale / np.abs(t1_exact).max():.3e}; "
        f"shares self {sh['self']:.3f} touch {sh['touch']:.3f} far {sh['far']:.3f}",
        flush=True,
    )
