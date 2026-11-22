#!/usr/bin/env python3
"""The factor F of the second error term at a finite frequency, for constant and for degree-one cells,
predicted from the symbolic form of what a cell loses.

WHY A SECOND SCRIPT. ``measure_t2_factor_dynamic.py`` predicts F for constant cells from a derivation made
by hand, and agrees with the scheme to 2% from k_S a = 0.125 to 4. Degree-one cells need the same
derivation one order higher, where the bookkeeping is too long to trust by hand and the scheme treats its
two sides differently. Here the cell-level step is taken from ``derive_cell_residual_moments.py``, which
derives it symbolically for the three cases, and everything after it is common. Run with p = 0 the script
must reproduce ``measure_t2_factor_dynamic.py``; that is its own check.

THE DERIVATION. The second-order term is bilinear, T2 = <f w_out, K f w_in>, with f the profile,
w_in = exp(i k_in.x), w_out = exp(-i k_out.x) and K the coupling operator of the background at the
frequency of the wave. The scheme replaces each product f w by what its cells hold; call the part lost g.
Then exactly

    T2 - T2_scheme = <g_out, K f w_in> + <f w_out, K g_in> - <g_out, K g_in> .          (*)

  term A  <g_out, phi>, phi = K f w_in the field scattered once, smooth. Summed over the cells it is the
          integral of w_out h^n (L0 phi + L1.grad phi + L2:grad grad phi), with L0, L1, L2 from the
          symbolic step (n = 2 for constant cells, 4 for degree one). phi and its derivatives are taken
          from a fine grid by one application of the kernel and central differences; one derivative
          of the second-derivative term is first moved onto L2 analytically, so that the field is
          differenced once only.
  term B  <f w_out, K g_in>: g_in acts as the source h^n [w L0 - div(w L1) + div div(w L2)], formed on
          the fine grid by the same differences, then one application of the kernel.
  term C  <g_out, K g_in>, local. The lost parts are sawtooths (constant cells) or parabolas and products
          of two sawtooths (degree one), whose spectra lie on the cube's axes or on the lattice of a
          coordinate plane at wavenumbers of order 1 / h, where K is the static strain Green operator
          Gamma. With G = grad f + i k f the slopes and H the second derivatives of f w over w:
              constant     (h^2 / 3)  sum_j  Gamma(e_j)  G_out,j G_in,j
              degree one   h^4 [ (1/45) sum_j Gamma(e_j) H_out,jj H_in,jj
                               + (1/9) sum_{i<j} Gamma_ij H_out,ij H_in,ij ]
          Gamma_ij is the average of Gamma over the plane lattice with weights 1 / (m^2 n^2); Gamma is
          quartic in the direction, so that average is a (1 - b) : b mixture of the two axes and the two
          diagonals with b = 4 (6 G / pi^2 - 2 / 5), G Catalan's constant. For degree-one cells H_in has
          no cross term (the source keeps the product of the two linear forms whole) and H_out has it.

F is (A + B - C) over T2 and over the relative projection error E, which is h^n times an integral of the
squared first (or second) derivatives of f over int f^2. F does not depend on h.

Output, angle by angle and for outgoing P and S: the predicted F (real and imaginary parts), its parts,
and the factor the scheme showed where ``measure_t2_factor_fft.py`` has stored it. Written to
``scripts/measure_t2_factor_operators_ka<ka>_core<c>_p<p>_fine<N>[_density].json``.

Run small first:
    conda run -n seismic python -u scripts/measure_t2_factor_operators.py 0.125 0.5 0 16
    ... <k_S a> <core_frac> <p> <N fine> [--density] [--step=0.3]
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
from derive_cell_residual_moments import residual_operators  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST as GATE  # noqa: E402
from gate_sphere_cell_average_vs_mie import REF, obs_points  # noqa: E402
from measure_t2_law_3d import split  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

from cubic_scattering import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.basis import gram_test  # noqa: E402
from cubic_scattering.graded_voxel.fft import canonical_blocks, transformed_kernel  # noqa: E402
from cubic_scattering.graded_voxel.site import contrast_operator  # noqa: E402
from cubic_scattering.graded_voxel.solver import field_sizes  # noqa: E402
from cubic_scattering.sphere_scattering import (  # noqa: E402
    _plane_wave_strain_voigt,
    _voigt_to_tensor,
    mie_scattered_displacement,
)
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402

K = np.array([1.0, 0.0, 0.0])
CATALAN = 0.915965594177219015
DIAGONAL_WEIGHT = 4.0 * (6.0 * CATALAN / np.pi**2 - 0.4)  # b of the docstring
PAIRS = ((0, 1), (0, 2), (1, 2))


def strain_to_voigt(eps: np.ndarray) -> np.ndarray:
    """A symmetric strain tensor in the package's six-component order, by the package's own routine."""
    eye = np.eye(3)
    return sum(
        eps[a, b] * _plane_wave_strain_voigt(eye[a], eye[b], 1.0) / 1j for a in range(3) for b in range(3)
    )


def local_strain(xi: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """-Gamma(xi) : tau, the strain a stress source tau varying rapidly along the direction xi leaves."""
    mu = REF.rho * REF.beta**2
    lam = REF.rho * REF.alpha**2 - 2 * mu
    xi = xi / np.linalg.norm(xi)
    t = tau @ xi
    gam = (np.outer(xi, t) + np.outer(t, xi)) / (2 * mu)
    return -(gam - (lam + mu) / (mu * (lam + 2 * mu)) * (xi @ tau @ xi) * np.outer(xi, xi))


def plane_strain(i: int, j: int, tau: np.ndarray) -> np.ndarray:
    """-Gamma_ij : tau, Gamma averaged over the lattice of the (i, j) plane with weights 1 / (m^2 n^2)."""
    e = np.eye(3)
    axes = 0.5 * (local_strain(e[i], tau) + local_strain(e[j], tau))
    diagonals = 0.5 * (local_strain(e[i] + e[j], tau) + local_strain(e[i] - e[j], tau))
    return (1.0 - DIAGONAL_WEIGHT) * axes + DIAGONAL_WEIGHT * diagonals


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ka, core_frac, p, n_fine = float(args[0]), float(args[1]), int(args[2]), int(args[3])
    if p not in (0, 1):
        raise SystemExit(f"measure_t2_factor_operators: p must be 0 or 1, got {p}")
    density = "density" in flags
    d = float(opts.get("step", "0.3"))
    core = RADIUS * core_frac
    omega = ka * REF.beta / RADIUS
    k_p, k_s = omega / REF.alpha, omega / REF.beta
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    dirs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    drho = GATE.Drho if density else 0.0
    ops_in, power = residual_operators("constant" if p == 0 else "degree-one in")
    ops_out, _ = residual_operators("constant" if p == 0 else "degree-one out")

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
    side = 2 * h
    x1, w1 = leggauss(3)
    nodes = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel() / 8.0
    xyz = centres[:, None, :] + h * nodes[None, :, :]  # (N, 27, 3)
    r = np.linalg.norm(xyz, axis=-1)
    unit = xyz / r[..., None]
    width = RADIUS - core
    x = np.clip((RADIUS - r) / width, 0.0, 1.0)
    f = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
    f_r = -30.0 * x**2 * (1.0 - x) ** 2 / width
    f_rr = 60.0 * x * (1.0 - x) * (1.0 - 2.0 * x) / width**2
    fd: dict = {(): f}
    for i in range(3):
        fd[(i,)] = f_r * unit[..., i]
        for j in range(i, 3):
            fd[(i, j)] = (f_rr - f_r / r) * unit[..., i] * unit[..., j] + (f_r / r if i == j else 0.0)
    # third derivatives, for the one integration by parts done analytically below
    f_rrr = np.where((x > 0.0) & (x < 1.0), -60.0 * (1.0 - 6.0 * x + 6.0 * x**2) / width**3, 0.0)
    mixed = f_rr / r - f_r / r**2
    for i in range(3):
        for j in range(i, 3):
            for m in range(j, 3):
                fd[(i, j, m)] = (f_rrr - 3.0 * mixed) * unit[..., i] * unit[..., j] * unit[
                    ..., m
                ] + mixed * ((i == j) * unit[..., m] + (i == m) * unit[..., j] + (j == m) * unit[..., i])

    def lost_part(ops, kappa: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """(L0, V): the lost part of f exp(i kappa.x) acts on a smooth phi as the integral of
        exp(i kappa.x) h^n (L0 phi + V.grad phi). V is L1 less the divergence of L2 (with the wave's phase),
        one derivative of the second-derivative term moved off phi analytically: every L is linear in the
        derivatives of f, so its derivative along b is the same expression of the derivatives one order up."""
        l0, l1, l2 = ops(fd, kappa)
        vec = [l1[a] + 0.0 * f for a in range(3)]
        for b in range(3):
            shifted = {idx: fd[tuple(sorted((*idx, b)))] for idx in fd if len(idx) <= 2}
            _, _, dl2 = ops(shifted, kappa)
            for (a, c), val in l2.items():
                # the term L2_ac d_a d_c phi: for a == c one derivative moves; for a < c the symbol carries
                # both orders, so half moves along each axis
                share = 1.0 if a == c else 0.5
                if c == b:
                    vec[a] = vec[a] - share * (dl2[(a, c)] + 1j * kappa[b] * val)
                if a == b and a != c:
                    vec[c] = vec[c] - share * (dl2[(a, c)] + 1j * kappa[b] * val)
        return l0 + 0.0 * f, vec

    def mean(q: np.ndarray) -> np.ndarray:
        return q @ wts

    norm_f = float(mean(f * f).sum())
    if p == 0:
        energy = sum(float(mean(fd[(i,)] ** 2).sum()) for i in range(3)) / 3.0
    else:
        energy = (
            sum(float(mean(fd[(i, i)] ** 2).sum()) for i in range(3)) / 45.0
            + sum(float(mean(fd[pair] ** 2).sum()) for pair in PAIRS) / 9.0
        )
    scale_f = norm_f / energy  # (T2 - T2s) / (T2 E) = (A + B - C) / T2 * scale_f, the h^n cancelling

    # ---------------------------------------------------------------- the kernel, once
    _, n_field, _, n_source = field_sizes(0, 0)
    kh = transformed_kernel(canonical_blocks(n_fine, h, omega, REF, n_source, n_field), n_fine, 1, 1)
    npad = 2 * n_fine - 1
    g0, g1, g2 = grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]
    m00 = gram_test(h, n_field)[0, 0]
    base = contrast_operator(GATE.Dlambda, GATE.Dmu, drho, omega)
    amp = np.concatenate([K.astype(complex), _plane_wave_strain_voigt(K, K, k_p)])
    k_in = k_p * K
    phase_in = np.exp(1j * (centres @ k_in))  # exp(i k_in.x) at the cell centres
    dx = h * nodes  # (27, 3): the Gauss nodes from the cell centre
    wave_in = np.exp(1j * (dx @ k_in))  # (27,): exp(i k_in.x) at the nodes over its value at the centre
    kernel_s = time.perf_counter() - t0

    def to_box(values: np.ndarray) -> np.ndarray:
        """Values on the body's cells, (N, ...), as an array on the whole box, zero outside the body."""
        box = np.zeros((n_fine, n_fine, n_fine, *values.shape[1:]), dtype=complex)
        box[g0, g1, g2] = values
        return box

    def diff(box: np.ndarray, axis: int, compact: bool = False) -> np.ndarray:
        """The derivative of cell means along one axis by the fourth-order central difference. A field that
        vanishes outside the body (``compact``) is continued by zeros; any other keeps the second-order
        difference of ``np.gradient`` in the two outermost layers of the box, where the stencil would
        reach points the kernel does not cover."""
        moved = np.moveaxis(box, axis, 0)
        padded = np.concatenate([np.zeros_like(moved[:2]), moved, np.zeros_like(moved[:2])])
        fourth = (-padded[4:] + 8.0 * padded[3:-1] - 8.0 * padded[1:-3] + padded[:-4]) / (12.0 * side)
        if not compact:
            second = np.gradient(moved, side, axis=0)
            fourth[:2], fourth[-2:] = second[:2], second[-2:]
        return np.moveaxis(fourth, 0, axis)

    def scattered_once(source_scalar: np.ndarray) -> np.ndarray:
        """The field scattered once by the source ``source_scalar`` x (contrast x incident amplitude),
        ``source_scalar`` one complex value a cell (profile x incident phase); on the whole box."""
        src = source_scalar[:, None] * (amp @ base.T)[None, :]
        grid = np.zeros((9, npad, npad, npad), dtype=complex)
        grid[:, g0, g1, g2] = src.T
        sh = np.fft.fftn(grid, axes=(1, 2, 3)).reshape(9, -1)
        yh = np.empty_like(sh)
        for row in range(9):
            yh[row] = np.einsum("cf,cf->f", kh[row].reshape(9, -1), sh)
        full = np.fft.ifftn(yh.reshape(9, npad, npad, npad), axes=(1, 2, 3)) / m00
        return np.moveaxis(full[:, :n_fine, :n_fine, :n_fine], 0, -1)  # (n, n, n, 9)

    # the far field of point sources at the cell centres, all cells at once (the package's own readout loops
    # over the cells in Python; hundreds of readouts are needed here). The variation of the outgoing wave
    # across a cell is put into the weights by ``moments`` below, not here.
    to_tensor = np.stack([_voigt_to_tensor(e) for e in np.eye(6)], -1)  # (3, 3, 6)
    weight = side**3

    def read_out(profile_out: np.ndarray, psi: np.ndarray, sel: slice = slice(None)) -> tuple:
        """The far field (its P and S parts) of the state ``psi`` weighted by ``profile_out``, one value a
        cell, radiated from the cell centres."""
        src = profile_out[:, None] * (psi @ base.T) * weight  # (N, 9)
        these = dirs[sel]
        u_p = np.zeros((len(these), 3), dtype=complex)
        u_s = np.zeros((len(these), 3), dtype=complex)
        for o, rh in enumerate(these):
            proj = centres @ rh
            gp = np.exp(1j * k_p * (rf - proj)) / (4 * np.pi * REF.rho * REF.alpha**2 * rf)
            gs = np.exp(1j * k_s * (rf - proj)) / (4 * np.pi * REF.rho * REF.beta**2 * rf)
            sr = np.einsum("abk,nk,b->na", to_tensor, src[:, 3:], rh)
            u_p[o] = (gp * (src[:, :3] @ rh + 1j * k_p * (sr @ rh))).sum() * rh
            qs = src[:, :3] + 1j * k_s * sr
            qs = qs - np.outer(qs @ rh, rh)
            u_s[o] = (gs[:, None] * qs).sum(axis=0)
        return u_p, u_s

    def derivatives(box: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
        """A field on the box, and its first and second derivatives, on the body's cells."""
        first = [diff(box, a) for a in range(3)]
        second = [[diff(first[a], b)[g0, g1, g2] for b in range(3)] for a in range(3)]
        return box[g0, g1, g2], [q[g0, g1, g2] for q in first], second

    def integral(weight_nodes: np.ndarray, wave: np.ndarray, value: np.ndarray, slope: list, sel: slice):
        """The far field of the integral over each cell of (weight x wave x field), the weight and the wave
        known at the Gauss nodes and the field by its cell mean and its slope: the mean of the product is
        the product of the means plus the first moment of the weight against the slope, without which the
        product of two cell means is wrong at the second order of the fine cell."""
        weighted = weight_nodes * wave
        total = read_out(mean(weighted), value, sel)
        for b in range(3):
            extra = read_out(mean(weighted * dx[:, b]), slope[b], sel)
            total = (total[0] + extra[0], total[1] + extra[1])
        return total

    # the field scattered once by the profile itself, with its derivatives
    phi, d1, d2 = derivatives(scattered_once(mean(f * wave_in) * phase_in))

    # term B: the incident side's lost part as a source, and the field it scatters
    l0, vec = lost_part(ops_in, k_in)
    eff = to_box(mean(l0 * wave_in) * phase_in)
    for a in range(3):
        eff -= diff(to_box(mean(vec[a] * wave_in) * phase_in), a, compact=True)
    chi, chi1, _ = derivatives(scattered_once(eff[g0, g1, g2]))

    # the stress the incident strain leaves in the contrast, and the states of the local term
    eps_in = 1j * k_p * np.outer(K, K)
    tau = GATE.Dlambda * np.trace(eps_in) * np.eye(3) + 2 * GATE.Dmu * eps_in

    def state(strain: np.ndarray) -> np.ndarray:
        return phase_in[:, None] * np.concatenate([np.zeros(3, complex), strain_to_voigt(strain)])[None, :]

    axis_state = [state(local_strain(np.eye(3)[j], tau)) for j in range(3)]
    plane_state = {pair: state(plane_strain(*pair, tau)) for pair in PAIRS}
    hess_in = {(a, b): fd[(a, b)] - k_in[a] * k_in[b] * f for a in range(3) for b in range(a, 3)}

    stored = ROOT / (
        f"scripts/measure_t2_factor_fft_ka{ka:g}_core{core_frac:g}_p{p}"
        + ("_density" if density else "")
        + ".json"
    )
    scheme = json.loads(stored.read_text())["rows"][-1] if stored.exists() else None
    print(
        f"factor of the second term from the symbolic cell step: k_S a = {ka}, core = {core_frac} a, cells of "
        f"degree {p}, {'stiffness and density' if density else 'stiffness'} contrast, fine grid {n_fine} across "
        f"({len(centres)} cells, kernel {kernel_s:.0f} s)",
        flush=True,
    )
    result: dict = {
        "ka_s": ka,
        "core_frac": core_frac,
        "p": p,
        "n_fine": n_fine,
        "density": density,
        "theta": [float(t) for t in THETA],
        "scheme_n": scheme["n"] if scheme else None,
    }
    for w, (label, k_out) in enumerate((("P", k_p), ("S", k_s))):
        t2 = t2_ex[w]
        rows = []
        print(
            f"\n outgoing {label}:  angle | fine-grid T2 / exact - 1 | F predicted (real, imaginary) = A + B - C "
            f"(real parts) | F of the scheme" + (f" ({scheme['n']} cells)" if scheme else "")
        )
        for j, th in enumerate(THETA):
            sel = slice(j, j + 1)
            ko = k_out * dirs[j]

            def ratio(v: np.ndarray, _j: int = j, _t2: np.ndarray = t2) -> complex:
                return complex((v @ np.conj(_t2[_j])) / np.vdot(_t2[_j], _t2[_j])) * scale_f

            # term A: the outgoing side's lost part against the field scattered once
            wave_out = np.exp(-1j * (dx @ ko))  # exp(-i k_out.x) at the nodes over its value at the centre
            m0, mvec = lost_part(ops_out, -ko)
            term_a = integral(m0, wave_out, phi, d1, sel)[w][0]
            for a in range(3):
                term_a = term_a + integral(mvec[a], wave_out, d1[a], d2[a], sel)[w][0]
            term_b = integral(f, wave_out, chi, chi1, sel)[w][0]
            t2_fine = integral(f, wave_out, phi, d1, sel)[w][0]
            both = wave_out * wave_in  # the two waves together, for the local term
            # term C: local
            if p == 0:
                term_c = sum(
                    read_out(
                        mean((fd[(a,)] - 1j * ko[a] * f) * (fd[(a,)] + 1j * k_in[a] * f) * both) / 3.0,
                        axis_state[a],
                        sel,
                    )[w][0]
                    for a in range(3)
                )
            else:

                def hess_out(a: int, b: int, _ko: np.ndarray = ko) -> np.ndarray:
                    return fd[(a, b)] - 1j * (_ko[a] * fd[(b,)] + _ko[b] * fd[(a,)]) - _ko[a] * _ko[b] * f

                term_c = sum(
                    read_out(mean(hess_out(a, a) * hess_in[(a, a)] * both) / 45.0, axis_state[a], sel)[w][0]
                    for a in range(3)
                ) + sum(
                    read_out(mean(hess_out(*pair) * hess_in[pair] * both) / 9.0, plane_state[pair], sel)[w][
                        0
                    ]
                    for pair in PAIRS
                )
            fa, fb, fc = ratio(term_a), ratio(term_b), ratio(term_c)
            pred = fa + fb - fc
            check = abs(complex(((t2_fine - t2[j]) @ np.conj(t2[j])) / np.vdot(t2[j], t2[j])))
            seen = scheme["F_" + label][j] if scheme else float("nan")
            seen_i = scheme.get("F_" + label + "_imag", [float("nan")] * 9)[j] if scheme else float("nan")
            rows.append(
                {
                    "theta": float(th),
                    "fine_check": check,
                    "F_predicted": pred.real,
                    "F_predicted_imag": pred.imag,
                    "A": fa.real,
                    "B": fb.real,
                    "C": fc.real,
                    "F_scheme": seen,
                    "F_scheme_imag": seen_i,
                }
            )
            print(
                f"   {th:5.2f} | {check:.1e} | {pred.real: .4f} {pred.imag:+.4f}i = {fa.real: .4f} + {fb.real: .4f} "
                f"- {fc.real: .4f} | {seen: .4f} {seen_i:+.4f}i",
                flush=True,
            )
        result["outgoing_" + label] = rows
    out = ROOT / (
        f"scripts/measure_t2_factor_operators_ka{ka:g}_core{core_frac:g}_p{p}_fine{n_fine}"
        + ("_density" if density else "")
        + ".json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
