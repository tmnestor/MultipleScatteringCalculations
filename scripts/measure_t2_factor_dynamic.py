#!/usr/bin/env python3
"""The factor F of the second term of the error law at a finite frequency, for constant cells: a prediction
built from the derivation, to be set against the factor the scheme itself shows.

WHY. ``measure_t2_factor_fft.py`` showed that the factor F tends to the closed form 2 - m_ax / m_bar as
the cells shrink, but only at low frequency: at k_S a = 0.5 it settles on other values. The closed form
was derived with the incident and outgoing waves taken as uniform strains. This script tests what the
derivation gives when they are not.

THE DERIVATION, for cells of constant field and contrast. The scheme replaces the product of the profile
f and a plane wave w by the product of their cell means. Write F_in = f w_in (w_in = exp(i k_in.x)) and
F_out = f w_out (w_out = exp(-i k_out.x)), and g = F - (mean f)(mean w) for the part the cell loses. The
second-order term is bilinear, T2 = <F_out, K F_in>, so its error is

    T2 - T2_scheme = <g_out, K F_in> + <F_out, K g_in> - <g_out, K g_in> .

Inside a cell g is a sawtooth with the slope of F, grad F = w (grad f + i k f). The wave's own slope
stands beside the profile's. To leading order in the cell half-width h:

  * against a smooth field the sawtooth acts as (h^2 / 3) w (-lap f + k^2 f -+ i k.grad f) (- for the
    incident side, + for the outgoing side), so the first two terms are (h^2 / 3) times
    (k_in^2 + k_out^2) T2, which is closed, plus second-order terms of the same body with one profile
    replaced, T2[-lap f + i k_out.grad f ; f] + T2[f ; -lap f - i k_in.grad f];
  * the third term is local: a sawtooth along axis j sees the static strain Green operator along that
    axis, Gamma(e_j), and the product of the two sawtooths has mean
    (h^2 / 3) w_in w_out (d_j f - i k_out,j f)(d_j f + i k_in,j f).

Dividing by T2 and by the relative projection error E = (h^2 / 3) int |grad f|^2 / int f^2 gives the
factor F at any frequency, independent of h. At zero frequency it is 2 - m_ax / m_bar.

WHAT IS COMPUTED. The replaced-profile terms have no closed form, so they are evaluated numerically on a
fine grid of N cells across, with one application of the same kernel the solver uses: the fine grid plays
the part of the continuum. The local term is evaluated from Gamma(e_j) in closed form and volume integrals
of the profile. T2 itself is the exact one (the radial reference at contrasts +-d).

The output, angle by angle, is the predicted F for outgoing P and outgoing S, its parts, the zero-frequency
closed form, and, where ``measure_t2_factor_fft.py`` has stored it, the factor the scheme showed on its
finest grid. Written to ``scripts/measure_t2_factor_dynamic_ka<ka>_core<c>_fine<N>[_density].json``.

Run small first:
    conda run -n seismic python -u scripts/measure_t2_factor_dynamic.py 0.125 0.5 16
    ... <k_S a> <core_frac> <N fine> [--density] [--step=0.3]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST as GATE  # noqa: E402
from gate_sphere_cell_average_vs_mie import REF, obs_points  # noqa: E402
from measure_t2_law_3d import predicted_factor, split  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

from cubic_scattering import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.basis import gram_test  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import canonical_blocks, transformed_kernel  # noqa: E402
from cubic_scattering.graded_voxel.site import contrast_operator  # noqa: E402
from cubic_scattering.graded_voxel.solver import (  # noqa: E402
    GradedVoxelResult,
    field_sizes,
    plane_wave_moments,
)
from cubic_scattering.sphere_scattering import (  # noqa: E402
    _plane_wave_strain_voigt,
    mie_scattered_displacement,
)
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402

K = np.array([1.0, 0.0, 0.0])


def strain_to_voigt(eps: np.ndarray) -> np.ndarray:
    """A symmetric strain tensor in the package's six-component order, by the package's own routine."""
    eye = np.eye(3)
    return sum(
        eps[a, b] * _plane_wave_strain_voigt(eye[a], eye[b], 1.0) / 1j for a in range(3) for b in range(3)
    )


def local_strain(axis: int, tau: np.ndarray) -> np.ndarray:
    """-Gamma(e_axis) : tau, the strain a stress source tau varying rapidly along one axis produces there."""
    mu = REF.rho * REF.beta**2
    lam = REF.rho * REF.alpha**2 - 2 * mu
    xi = np.eye(3)[axis]
    t = tau @ xi
    gam = (np.outer(xi, t) + np.outer(t, xi)) / (2 * mu)
    gam = gam - (lam + mu) / (mu * (lam + 2 * mu)) * (xi @ tau @ xi) * np.outer(xi, xi)
    return -gam


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ka, core_frac, n_fine = float(args[0]), float(args[1]), int(args[2])
    density = "density" in flags
    d = float(opts.get("step", "0.3"))
    core = RADIUS * core_frac
    omega = ka * REF.beta / RADIUS
    k_p, k_s = omega / REF.alpha, omega / REF.beta
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    dirs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    tangents = np.stack([-dirs[:, 1], dirs[:, 0], np.zeros(len(dirs))], 1)
    drho = GATE.Drho if density else 0.0

    # ---------------------------------------------------------------- the exact second-order term
    def exact(scale: float) -> np.ndarray:
        c = MaterialContrast(scale * GATE.Dlambda, scale * GATE.Dmu, scale * drho)
        return mie_scattered_displacement(graded_mie_result(omega, RADIUS, core, REF, c, 12), pts)

    t2_ex = split((exact(d) + exact(-d)) / (2 * d * d), dirs)

    # ---------------------------------------------------------------- the fine grid and the profile on it
    t0 = time.perf_counter()
    h = RADIUS / n_fine
    grid_idx, centres, h = _build_grid_index_map(
        RADIUS, n_fine, lambda q: bool(np.linalg.norm(q) < RADIUS + np.sqrt(3.0) * h)
    )
    x1, w1 = leggauss(3)
    nodes = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel() / 8.0
    xyz = centres[:, None, :] + h * nodes[None, :, :]  # (N, 27, 3)
    r = np.linalg.norm(xyz, axis=-1)
    width = RADIUS - core
    x = np.clip((RADIUS - r) / width, 0.0, 1.0)
    f = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
    f_r = -30.0 * x**2 * (1.0 - x) ** 2 / width
    f_rr = 60.0 * x * (1.0 - x) * (1.0 - 2.0 * x) / width**2
    grad = f_r[..., None] * xyz / r[..., None]  # (N, 27, 3)
    lap = f_rr + 2.0 * f_r / r

    def mean(q: np.ndarray) -> np.ndarray:
        return q @ wts

    vol = (2 * h) ** 3
    norm_f = float(mean(f * f).sum() * vol)
    norm_grad = float(mean((grad**2).sum(-1)).sum() * vol)
    length2 = norm_f / norm_grad  # int f^2 / int |grad f|^2: E = (h^2 / 3) / length2 for constant cells

    # ---------------------------------------------------------------- the kernel, once
    _, n_field, _, n_source = field_sizes(0, 0)
    kh = transformed_kernel(canonical_blocks(n_fine, h, omega, REF, n_source, n_field), n_fine, 1, 1)
    npad = 2 * n_fine - 1
    g0, g1, g2 = grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]
    m00 = gram_test(h, n_field)[0, 0]
    base = contrast_operator(GATE.Dlambda, GATE.Dmu, drho, omega)
    amp = np.concatenate([K.astype(complex), _plane_wave_strain_voigt(K, K, k_p)])
    psi_in = plane_wave_moments(centres, h, k_p * K, amp, n_field)[:, 0] / m00  # (N, 9): the cell means
    kernel_s = time.perf_counter() - t0

    def scattered_once(profile_in: np.ndarray) -> np.ndarray:
        """The cell means of the field scattered once by the profile ``profile_in`` (one value a cell)."""
        src = profile_in[:, None] * (psi_in @ base.T)
        grid = np.zeros((9, npad, npad, npad), dtype=complex)
        grid[:, g0, g1, g2] = src.T
        sh = np.fft.fftn(grid, axes=(1, 2, 3)).reshape(9, -1)
        yh = np.empty_like(sh)
        for row in range(9):
            yh[row] = np.einsum("cf,cf->f", kh[row].reshape(9, -1), sh)
        return np.fft.ifftn(yh.reshape(9, npad, npad, npad), axes=(1, 2, 3))[:, g0, g1, g2].T / m00

    def read_out(profile_out: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The far field (its P and S parts) of the state ``psi`` weighted by the profile ``profile_out``."""
        delta = np.zeros((len(centres), 4, 9, 9), dtype=complex)
        delta[:, 0] = profile_out[:, None, None] * base[None]
        full = np.zeros((len(centres), n_field, 9), dtype=complex)
        full[:, 0] = psi
        res = GradedVoxelResult(centres, grid_idx, h, omega, REF, delta, full, 0, 0)
        return graded_far_field(res, dirs, rf, K, K, "P")

    f_m, lap_m = mean(f), mean(lap)
    grad_m = [mean(grad[..., j]) for j in range(3)]

    # the terms with one profile replaced
    psi_f = scattered_once(f_m)
    t2_fine = read_out(f_m, psi_f)  # T2[f ; f] on the fine grid: the check of the evaluation
    out_lap = read_out(-lap_m, psi_f)  # T2[-lap f ; f]
    out_grad = [read_out(grad_m[j], psi_f) for j in range(2)]  # T2[d_j f ; f]
    in_side = read_out(f_m, scattered_once(-lap_m - 1j * k_p * grad_m[0]))  # T2[f ; -lap f - i k_in.grad f]

    # the local term: the strain each axis's sawtooth leaves, read out with the product of the two slopes
    eps_in = 1j * k_p * np.outer(K, K)
    tau = GATE.Dlambda * np.trace(eps_in) * np.eye(3) + 2 * GATE.Dmu * eps_in
    phase = psi_in[:, 0] / amp[0]  # the cell mean of exp(i k_in.x)
    local = []
    for j in range(3):
        state = np.concatenate([np.zeros(3, complex), strain_to_voigt(local_strain(j, tau))])
        psi_j = phase[:, None] * state[None, :]
        local.append(
            {
                "gg": read_out(mean(grad[..., j] ** 2), psi_j),
                "fg": read_out(mean(f * grad[..., j]), psi_j),
                "ff": read_out(mean(f * f), psi_j),
            }
        )

    # ---------------------------------------------------------------- assemble, wave by wave
    stored = ROOT / (
        f"scripts/measure_t2_factor_fft_ka{ka:g}_core{core_frac:g}_p0"
        + ("_density" if density else "")
        + ".json"
    )
    scheme = json.loads(stored.read_text())["rows"][-1] if stored.exists() else None
    limits = {
        "P": [predicted_factor(np.outer(n, n)) for n in dirs],
        "S": [
            predicted_factor(0.5 * (np.outer(n, t) + np.outer(t, n)))
            for n, t in zip(dirs, tangents, strict=True)
        ],
    }
    print(
        f"factor of the second term at a finite frequency: k_S a = {ka}, core = {core_frac} a, constant "
        f"cells, {'stiffness and density' if density else 'stiffness'} contrast, fine grid {n_fine} across "
        f"({len(centres)} cells, {kernel_s:.0f} s);  int f^2 / int |grad f|^2 = {length2 / RADIUS**2:.5f} a^2",
        flush=True,
    )
    result: dict = {
        "ka_s": ka,
        "core_frac": core_frac,
        "n_fine": n_fine,
        "density": density,
        "theta": [float(t) for t in THETA],
        "length2_over_a2": length2 / RADIUS**2,
    }
    for w, (label, k_out) in enumerate((("P", k_p), ("S", k_s))):
        t2 = t2_ex[w]
        rows = []
        print(
            f"\n outgoing {label}:  angle | fine-grid T2 / exact - 1 | F predicted = wave part + profile part "
            f"- local part | F at zero frequency | F of the scheme"
            + (f" ({scheme['n']} cells)" if scheme else "")
        )
        for j, th in enumerate(THETA):
            ko = k_out * dirs[j]
            ki = k_p * K

            def ratio(v: np.ndarray, _j: int = j, _t2: np.ndarray = t2) -> complex:
                return complex((v @ np.conj(_t2[_j])) / np.vdot(_t2[_j], _t2[_j])) * length2

            wave = (k_p**2 + k_out**2) * length2
            prof = ratio(
                out_lap[w][j] + 1j * (ko[0] * out_grad[0][w][j] + ko[1] * out_grad[1][w][j]) + in_side[w][j]
            )
            loc = ratio(
                sum(
                    local[a]["gg"][w][j]
                    + 1j * (ki[a] - ko[a]) * local[a]["fg"][w][j]
                    + ko[a] * ki[a] * local[a]["ff"][w][j]
                    for a in range(3)
                )
            )
            pred = wave + prof - loc
            check = complex(((t2_fine[w][j] - t2[j]) @ np.conj(t2[j])) / np.vdot(t2[j], t2[j]))
            seen = scheme["F_" + label][j] if scheme else float("nan")
            rows.append(
                {
                    "theta": float(th),
                    "fine_check": abs(check),
                    "F_predicted": pred.real,
                    "F_predicted_imag": pred.imag,
                    "wave_part": wave,
                    "profile_part": prof.real,
                    "local_part": loc.real,
                    "F_static": limits[label][j],
                    "F_scheme": seen,
                }
            )
            print(
                f"   {th:5.2f} | {abs(check):.1e} | {pred.real: .4f} = {wave:.4f} + {prof.real: .4f} - "
                f"{loc.real: .4f} | {limits[label][j]: .4f} | {seen: .4f}",
                flush=True,
            )
        result["outgoing_" + label] = rows
        if scheme:
            result["scheme_n"] = scheme["n"]
    out = ROOT / (
        f"scripts/measure_t2_factor_dynamic_ka{ka:g}_core{core_frac:g}_fine{n_fine}"
        + ("_density" if density else "")
        + ".json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
