#!/usr/bin/env python3
"""How the graded voxel's approach to fourth order depends on the resolution of the contrast profile.

The graded sphere of ``pilot_graded_voxel_sphere.py``, with the shell a_c < r < a now taking one of three
profiles s(x), x = (a - r)/(a - a_c):
  s5    10x^3 - 15x^4 + 6x^5                                  (C2: the third derivative jumps at both ends)
  s9    126x^5 - 420x^6 + 540x^7 - 315x^8 + 70x^9             (C4)
  sinf  e^{-1/x} / (e^{-1/x} + e^{-1/(1-x)})                   (C-infinity)
and a core radius a_c = CORE_FRAC * a (0.5 in the paper's sphere; 0.25 widens the shell 1.5 times).

Measured, against the exact solution of the same body (``crosscheck_graded_sphere.graded_tmatrix``):
  T1  the Born term (the discretisation of the source), and
  T2  the second-order Born term (one application of the coupling, by FFT with the cube-group blocks),
from central differences in the contrast at steps d and 2d, Richardson-combined (error O(d^4)); and, with
--full, the full solve (``graded_voxel.fft.solve_graded_sphere_fft``). Errors: max over nine angles of
|scheme - exact| / max |exact|, at 5e8 radii.

Beside them, the projection defect of the profile over the kept cells,
    defect = sum_cells int (s - P1 s)^2 / sum_cells int s^2,
with P1 the L2 projection onto {1, x, y, z} in each cell (10-point Gauss rule per axis). The relative
error of T2 equals it to within a few per cent, at every grid, profile and shell width measured. defect2
is the same with the projection onto the ten polynomials of degree two: what a field quadratic in the
cell leaves.

Run small first:
    conda run -n seismic python -u scripts/measure_graded_voxel_resolution.py 0.5 0.5 s5,sinf 4
    ... <k_S a> <CORE_FRAC> <profiles> [--full] [--summary=path.json] <n ...>
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crosscheck_graded_sphere import MieResult, graded_tmatrix  # noqa: E402
from cubic_scattering import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.basis import (  # noqa: E402
    CONTRAST_NORMS,
    contrast_values,
    gram_test,
    source_expansion,
)
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import offset_blocks, solve_graded_sphere_fft  # noqa: E402
from cubic_scattering.graded_voxel.site import cell_contrast_coefficients  # noqa: E402
from cubic_scattering.graded_voxel.solver import GradedVoxelResult, plane_wave_moments  # noqa: E402
from cubic_scattering.sphere_scattering import (  # noqa: E402
    _plane_wave_strain_voigt,
    mie_scattered_displacement,
)
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

K_HAT = np.array([1.0, 0.0, 0.0])
#: Contrast step of the central differences; with the 2d step the extracted T1, T2 err at O(d^4).
D_STEP = 1e-2
N_MAX = 12


def _sinf(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    a, b = np.exp(-1.0 / x), np.exp(-1.0 / (1.0 - x))
    return float(a / (a + b))


SHAPES = {
    "s5": lambda x: x**3 * (10 - 15 * x + 6 * x**2),
    "s9": lambda x: x**5 * (126 - 420 * x + 540 * x**2 - 315 * x**3 + 70 * x**4),
    "sinf": _sinf,
}


def radial(shape: str, core: float, r: float) -> float:
    """The contrast factor at radius r: 1 in the core, the shell profile, 0 outside."""
    if r <= core:
        return 1.0
    if r >= RADIUS:
        return 0.0
    return SHAPES[shape]((RADIUS - r) / (RADIUS - core))


def exact_field(shape: str, core: float, omega: float, scale: float, pts: np.ndarray) -> np.ndarray:
    """Scattered displacement of the exact graded sphere at contrast ``scale`` x CONTRAST."""
    con = MaterialContrast(scale * CONTRAST.Dlambda, scale * CONTRAST.Dmu, scale * CONTRAST.Drho)
    kp, ks = omega / REF.alpha, omega / REF.beta
    a_n, b_n, c_n, a_sv, b_sv = (np.zeros(N_MAX + 1, dtype=complex) for _ in range(5))
    for n in range(N_MAX + 1):
        cp = (2 * n + 1) * (1j) ** n / (1j * kp)
        cs = (2 * n + 1) * (1j) ** n / (1j * ks)
        tm, tsh = graded_tmatrix(
            n,
            omega,
            RADIUS,
            core,
            REF,
            con,
            profile=lambda r: SHAPES[shape]((RADIUS - r) / (RADIUS - core)),
        )
        a_n[n] = tm[0, 0] * cp
        if n == 0:
            continue
        b_n[n], a_sv[n], b_sv[n] = tm[1, 0] * cp, tm[0, 1] * cs, tm[1, 1] * cs
        c_n[n] = tsh * cs
    mie = MieResult(
        a_n=a_n,
        b_n=b_n,
        c_n=c_n,
        a_n_sv=a_sv,
        b_n_sv=b_sv,
        n_max=N_MAX,
        omega=omega,
        radius=RADIUS,
        ref=REF,
        contrast=con,
        ka_P=omega * RADIUS / REF.alpha,
        ka_S=omega * RADIUS / REF.beta,
    )
    return mie_scattered_displacement(mie, pts)


def projection_defect(shape: str, core: float, centres: np.ndarray, h: float, degree: int = 1) -> float:
    """sum int (s - P s)^2 / sum int s^2 over the cells (half-width h), P the L2 projection onto the cell's
    polynomials of the given degree (1: the four linear functions; 2: the ten quadratic ones)."""
    x, w = np.polynomial.legendre.leggauss(10)
    xi = np.stack([g.ravel() for g in np.meshgrid(x, x, x, indexing="ij")], axis=1)
    wt = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()
    n_fun = 4 if degree == 1 else 10
    basis = contrast_values(xi)[:n_fun]
    norms = np.array(CONTRAST_NORMS[:n_fun])
    num = den = 0.0
    for c in centres:
        s = np.array([radial(shape, core, float(r)) for r in np.linalg.norm(c + h * xi, axis=1)])
        fit = ((basis * wt) @ s / norms) @ basis
        num += float((wt * (s - fit) ** 2).sum())
        den += float((wt * s * s).sum())
    return num / den


def rel_err(u: np.ndarray, exact: np.ndarray) -> float:
    return float(np.abs(u - exact).max() / np.abs(exact).max())


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    summary = next((Path(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--summary=")), None)
    ka, core = float(args[0]), RADIUS * float(args[1])
    shapes = args[2].split(",")
    ns = [int(a) for a in args[3:]]
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    kp = omega / REF.alpha
    amp = np.concatenate([K_HAT.astype(complex), _plane_wave_strain_voigt(K_HAT, K_HAT, kp)])
    d = D_STEP
    ref: dict = {}
    for shape in shapes:
        f = {s: exact_field(shape, core, omega, s, pts) for s in (d, -d, 2 * d, -2 * d)}
        t1 = (8 * (f[d] - f[-d]) - (f[2 * d] - f[-2 * d])) / (12 * d)
        t2 = (16 * (f[d] + f[-d]) - (f[2 * d] + f[-2 * d])) / (24 * d * d)
        ref[shape] = (t1, t2, exact_field(shape, core, omega, 1.0, pts) if full else None)
    out: dict = {"ka_s": ka, "core_frac": core / RADIUS, "n_sub": ns}
    out.update({s: {"t1": [], "t2": [], "full": [], "defect": [], "defect2": []} for s in shapes})
    print(f"k_S a = {ka}, core = {core / RADIUS} a, profiles {shapes}, full = {full}", flush=True)
    for n in ns:
        grid, centres, h = _build_grid_index_map(
            RADIUS, n, lambda q, _h=RADIUS / n: bool(np.linalg.norm(q) < RADIUS + np.sqrt(3) * _h)
        )
        blocks = offset_blocks(n, h, omega, REF)
        npad = 2 * n - 1
        kh = np.zeros((36, 90, npad, npad, npad), complex)
        for off, blk in blocks.items():
            kh[:, :, off[0] % npad, off[1] % npad, off[2] % npad] = blk.transpose(0, 2, 1, 3).reshape(
                36, 90
            )
        for row in range(36):
            kh[row] = np.fft.fftn(kh[row], axes=(1, 2, 3))
        kh = kh.reshape(36, 90, -1)
        g0, g1, g2 = grid[:, 0], grid[:, 1], grid[:, 2]
        minv = np.linalg.inv(gram_test(h))
        psi0 = np.einsum("ab,nbi->nai", minv, plane_wave_moments(centres, h, kp * K_HAT, amp))
        for shape in shapes:

            def prof(pos: np.ndarray, _s: str = shape) -> float:
                return radial(_s, core, float(np.linalg.norm(pos)))

            delta = np.array(
                [cell_contrast_coefficients(prof, c, h, CONTRAST, REF, omega, 1) for c in centres]
            )
            e_cells = np.array([source_expansion(dd) for dd in delta])
            src0 = np.einsum("ncbij,nbj->nci", e_cells, psi0).reshape(len(centres), 90)
            grid_src = np.zeros((90, npad, npad, npad), complex)
            grid_src[:, g0, g1, g2] = src0.T
            sh = np.fft.fftn(grid_src, axes=(1, 2, 3)).reshape(90, -1)
            yh = np.einsum("rcf,cf->rf", kh, sh).reshape(36, npad, npad, npad)
            rhs = np.fft.ifftn(yh, axes=(1, 2, 3))[:, g0, g1, g2].T.reshape(len(centres), 4, 9)
            psi1 = np.einsum("ab,nbi->nai", minv, rhs)
            t1_ex, t2_ex, full_ex = ref[shape]
            errs = []
            for psi, ex in ((psi0, t1_ex), (psi1, t2_ex)):
                res = GradedVoxelResult(centres, grid, h, omega, REF, delta, psi, 1, 1)
                up, us = graded_far_field(res, pts / rf, rf, K_HAT, K_HAT, "P")
                errs.append(rel_err(up + us, ex))
            defect = projection_defect(shape, core, centres, h)
            defect2 = projection_defect(shape, core, centres, h, degree=2)
            out[shape]["defect2"].append(defect2)
            line = f"  n {n:3d} {shape:5s} T1 {errs[0]:.4e}  T2 {errs[1]:.4e}"
            line += f"  defect {defect:.4e}  defect2 {defect2:.4e}"
            out[shape]["defect"].append(defect)
            out[shape]["t1"].append(errs[0])
            out[shape]["t2"].append(errs[1])
            if full:
                res = solve_graded_sphere_fft(
                    omega, RADIUS, REF, CONTRAST, n, prof, K_HAT, K_HAT, "P", blocks=blocks
                )
                up, us = graded_far_field(res, pts / rf, rf, K_HAT, K_HAT, "P")
                out[shape]["full"].append(rel_err(up + us, full_ex))
                line += f"  full {out[shape]['full'][-1]:.4e}"
            print(line, flush=True)
    for shape in shapes:
        for key in ("t2", "full"):
            e = out[shape][key]
            for n1, n2, e1, e2 in zip(ns, ns[1:], e, e[1:], strict=False):
                print(
                    f"  {shape} {key} apparent order {n1} -> {n2}: {np.log(e1 / e2) / np.log(n2 / n1):.2f}"
                )
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(out, indent=2) + "\n")
        print(f"  wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
