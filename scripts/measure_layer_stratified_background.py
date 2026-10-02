#!/usr/bin/env python3
"""The layer of voxels in a STRATIFIED background: do the convergence results of the whole space survive?

THE CONFIGURATION.  Normal incidence, P.  The background is two half-spaces joined at z = z_I: the medium
of the whole-space tests above, a faster and denser one below.  The scatterer is the layer 0 < z < D of
the whole-space tests (uniform, or with the smooth profile), cut into n cells, lying above the interface
(z_I >= D; z_I = D puts the lowest cell against it).  The voxels interact through the Green's function of
the STRATIFIED background,

    g(z, z') = (i / 2 M k) [ exp(i k |z - z'|) + rho_I exp(i k (2 z_I - z - z')) ],     z, z' < z_I,

with rho_I the displacement reflection coefficient of the interface: the whole-space term plus a
reverberation that is smooth (it carries no local term).  The incident field is the field of the source in
the background, a downgoing wave and its reflection.  The exact answer is the reflection of the whole
stack (the layer, the gap and the interface) less that of the background.

MEASURED
  [1] the order of the full solve for (p, r) = (0, 0), (1, 1), (2, 2) on the uniform layer and on the
      smooth profile, against min(2p + 2, 2r + 2);
  [2] the first-order (Born) term of the scheme against the prediction of the two-term analysis: each of
      the four paths (incident wave down or up, scattered wave leaving up or down) carries the wave factor
      of a cell for its own pair of wavenumbers (the one-dimensional case of ``octree.born_wave_factor``);
  [3] the law of the whole-space layer: the relative error of the term of second order in the contrast
      (isolated by differences in the contrast, for the scheme and the exact stack alike) over the
      relative projection error of the profile, which tends to one in the long-wave limit.

Run:  conda run -n seismic python -u scripts/measure_layer_stratified_background.py [omega]
SI units; e^{-i omega t}; z down.
"""

import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crosscheck_graded_contrast as cg  # noqa: E402
from crosscheck_second_moment_voxel import (  # noqa: E402
    ALPHA,
    CONTRAST,
    D_LAYER,
    M_P,
    RHO,
    Z_OBS_R,
    Z_SRC,
    gauss,
    kernel,
)

ALPHA_2, RHO_2 = 6500.0, 2800.0  # the half-space below the interface
M_2 = RHO_2 * ALPHA_2**2
NS = (2, 4, 8, 16)
D_STEP = 1e-2  # step in the contrast for the second-order term


def reflection_i() -> float:
    """Displacement reflection coefficient of the interface, for a wave arriving from above."""
    z1, z2 = RHO * ALPHA, RHO_2 * ALPHA_2
    return (z1 - z2) / (z1 + z2)


def k_reflected(k: float, z: np.ndarray, zp: np.ndarray, z_i: float) -> np.ndarray:
    """The reverberation part of the kernel, shape broadcast(z, zp) + (2, 2): rows (u, strain), columns
    (force, moment).  The moment column is minus the derivative on the source, the strain row the
    derivative on the receiver."""
    g = reflection_i() * 1j / (2 * M_P * k) * np.exp(1j * k * (2 * z_i - z - zp))
    one = np.ones_like(g)
    return g[..., None, None] * np.stack(
        [np.stack([one, 1j * k * one], -1), np.stack([-1j * k * one, k**2 * one], -1)], -2
    )


def incident(k: float, z: np.ndarray, z_i: float) -> np.ndarray:
    """(u, strain) of the background field of the source at Z_SRC, at depths Z_SRC < z < z_i."""
    down = 1j / (2 * M_P * k) * np.exp(1j * k * (z - Z_SRC))
    up = reflection_i() * 1j / (2 * M_P * k) * np.exp(1j * k * (2 * z_i - z - Z_SRC))
    return np.stack([down + up, 1j * k * (down - up)], -1)


def system(omega: float, n: int, p: int, r: int, profile, z_i: float, scale: float = 1.0):
    """(mass, coupling, rhs, readout) of the scheme in the stratified background."""
    d_lam, d_mu, d_rho = (scale * c for c in CONTRAST)
    k = omega / ALPHA
    h = D_LAYER / (2 * n)
    centres = (np.arange(n) + 0.5) * 2 * h
    dq = np.diag([omega**2 * d_rho, d_lam + 2 * d_mu])
    nb = p + 1
    s, ws = gauss(-h, h)
    phi = np.array([cg.leg(a, s / h) for a in range(nb)])
    coef = np.array(
        [
            [(2 * c + 1) / (2 * h) * np.sum(ws * profile(zc + s) * cg.leg(c, s / h)) for c in range(r + 1)]
            for zc in centres
        ]
    )

    def f_r(j: int, loc: np.ndarray) -> np.ndarray:
        return sum(coef[j, c] * cg.leg(c, loc / h) for c in range(r + 1))

    size = 2 * nb * n
    mass = np.zeros((size, size), dtype=complex)
    coup = np.zeros((size, size), dtype=complex)
    rhs = np.zeros(size, dtype=complex)
    for i in range(n):
        zi = centres[i] + s
        w0 = incident(k, zi, z_i)
        for a in range(nb):
            row = 2 * (nb * i + a)
            mass[row : row + 2, row : row + 2] += (2 * h / (2 * a + 1)) * np.eye(2)
            rhs[row : row + 2] = np.einsum("n,n,ni->i", phi[a], ws, w0)
        for j in range(n):
            zj = centres[j] + s
            fj = f_r(j, s)
            # the reverberation is smooth: a product Gauss rule for every pair of cells, the same cell too
            full = np.einsum(
                "an,n,bm,m,m,nmij->abij",
                phi,
                ws,
                phi,
                ws,
                fj,
                k_reflected(k, zi[:, None], zj[None, :], z_i),
            )
            if i == j:
                whole = np.array(
                    [
                        [cg.self_quad(a, b, k, h, lambda t, j=j: f_r(j, t)) for b in range(nb)]
                        for a in range(nb)
                    ]
                )
            else:
                kk = kernel(k, zi[:, None] - zj[None, :])
                whole = np.einsum("an,n,bm,m,m,nmij->abij", phi, ws, phi, ws, fj, kk)
            for a in range(nb):
                for b in range(nb):
                    rr, cc = 2 * (nb * i + a), 2 * (nb * j + b)
                    coup[rr : rr + 2, cc : cc + 2] += (whole[a, b] + full[a, b]) @ dq
    readout = np.zeros(size, dtype=complex)
    for j in range(n):
        zj = centres[j] + s
        kz = kernel(k, Z_OBS_R - zj) + k_reflected(k, np.full_like(zj, Z_OBS_R), zj, z_i)
        fj = f_r(j, s)
        for b in range(nb):
            col = 2 * (nb * j + b)
            readout[col : col + 2] = (np.einsum("n,n,n,nij->ij", phi[b], ws, fj, kz) @ dq)[0]
    return mass, coup, rhs, readout


def scattered(omega: float, n: int, p: int, r: int, profile, z_i: float, scale: float = 1.0) -> complex:
    mass, coup, rhs, readout = system(omega, n, p, r, profile, z_i, scale)
    return complex(readout @ np.linalg.solve(mass - coup, rhs))


def exact(omega: float, name: str, z_i: float, scale: float = 1.0) -> complex:
    """Scattered u at the observer: the stack's reflection less the background's, in 30-digit arithmetic.

    The layer's propagator on (u, y = M u' / M_P) comes from mpmath's Taylor-series ODE solver, the gap
    below it is a homogeneous propagator, and at the interface the state is that of the transmitted wave.
    """
    d_lam, d_mu, d_rho = (mp.mpf(scale) * mp.mpf(c) for c in CONTRAST)
    d_m = d_lam + 2 * d_mu
    mpp, rho, om = mp.mpf(M_P), mp.mpf(RHO), mp.mpf(omega)
    k1, k2 = om / ALPHA, om / ALPHA_2
    f = cg.MP_PROFILES[name]

    def rhs(z, y):
        return [mpp * y[1] / (mpp + d_m * f(z)), -(om**2) * (rho + d_rho * f(z)) * y[0] / mpp]

    def reflection(with_layer: bool) -> mp.mpc:
        if with_layer:
            cols = [
                mp.odefun(rhs, 0, ic)(D_LAYER) for ic in ([mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(1)])
            ]
            layer = mp.matrix([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]])
            gap_len = mp.mpf(z_i) - D_LAYER
        else:
            layer = mp.eye(2)
            gap_len = mp.mpf(z_i)
        c, sn = mp.cos(k1 * gap_len), mp.sin(k1 * gap_len)
        total = mp.matrix([[c, sn / k1], [-k1 * sn, c]]) * layer  # state at 0 to state at z_i
        # total (1 + R, i k1 (1 - R)) = T (1, i k2 M_2 / M_P)
        a = mp.matrix(
            [
                [total[0, 0] - 1j * k1 * total[0, 1], -1],
                [total[1, 0] - 1j * k1 * total[1, 1], -1j * k2 * mp.mpf(M_2) / mpp],
            ]
        )
        b = -mp.matrix([total[0, 0] + 1j * k1 * total[0, 1], total[1, 0] + 1j * k1 * total[1, 1]])
        return mp.lu_solve(a, b)[0]

    amp = 1j / (2 * mpp * k1) * mp.expj(-k1 * (mp.mpf(Z_SRC) + mp.mpf(Z_OBS_R)))
    return complex((reflection(True) - reflection(False)) * amp)


def wave_factor(k_in: float, k_out: float, h: float, p: int) -> float:
    """The one-dimensional wave factor of a cell of half-width h and field degree p."""
    num = sum((2 * a + 1) * spherical_jn(a, k_in * h) * spherical_jn(a, k_out * h) for a in range(p + 1))
    return float(num / spherical_jn(0, (k_in - k_out) * h))


def born_predicted(omega: float, n: int, p: int, z_i: float) -> complex:
    """The scheme's first-order term for the uniform layer, from the exact first-order term of each of
    the four paths times the wave factor of that path."""
    d_lam, d_mu, d_rho = CONTRAST
    k = omega / ALPHA
    h = D_LAYER / (2 * n)
    dq = np.diag([omega**2 * d_rho, d_lam + 2 * d_mu])
    r_i = reflection_i()
    pre = 1j / (2 * M_P * k)
    total = 0j
    for zc in (np.arange(n) + 0.5) * 2 * h:
        s, ws = gauss(zc - h, zc + h)
        # incident: wavenumber +k (down) and -k (up); state (u, strain) = amplitude (1, i k_in)
        for k_in, a_in in (
            (k, pre * np.exp(-1j * k * Z_SRC)),
            (-k, r_i * pre * np.exp(1j * k * (2 * z_i - Z_SRC))),
        ):
            # leaving: towards -z directly (wavevector -k), or towards +z and back off the interface (+k).
            # The row of the kernel for u at the observer is amplitude (1, i k_out) on (force, moment).
            for k_out, a_out in (
                (-k, pre * np.exp(-1j * k * Z_OBS_R)),
                (k, r_i * pre * np.exp(1j * k * (2 * z_i - Z_OBS_R))),
            ):
                row = a_out * np.array([1.0, 1j * k_out])
                state = a_in * np.array([1.0, 1j * k_in])
                path = (row @ dq @ state) * np.sum(ws * np.exp(1j * (k_in - k_out) * s))
                total += path * wave_factor(k_in, k_out, h, p)
    return complex(total)


def projection_error(n: int, q: int, profile) -> float:
    h = D_LAYER / (2 * n)
    s, w = gauss(-h, h)
    num = den = 0.0
    for zc in (np.arange(n) + 0.5) * 2 * h:
        vals = profile(zc + s)
        fit = sum(
            (2 * c + 1) / (2 * h) * np.sum(w * vals * cg.leg(c, s / h)) * cg.leg(c, s / h)
            for c in range(q + 1)
        )
        num += float(np.sum(w * (vals - fit) ** 2))
        den += float(np.sum(w * vals**2))
    return num / den


def second_order(f) -> complex:
    """The term of second order in the contrast of f(scale), by central differences at D_STEP and
    2 D_STEP, Richardson-combined."""
    d = D_STEP
    v = {sc: f(sc) for sc in (d, -d, 2 * d, -2 * d)}
    return (16 * (v[d] + v[-d]) - (v[2 * d] + v[-2 * d])) / (24 * d * d)


def t2_ratio(omega: float, n: int, p: int, z_i: float) -> float:
    """Relative error of the second-order term on the smooth profile, over E_p."""
    profile = cg.PROFILES["smooth"][0]
    t2_exact = second_order(lambda sc: exact(omega, "smooth", z_i, sc))
    t2 = second_order(lambda sc: scattered(omega, n, p, p, profile, z_i, sc))
    return abs(t2 - t2_exact) / abs(t2_exact) / projection_error(n, p, profile)


def main() -> int:
    omega = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    k = omega / ALPHA
    oks = []
    print(
        f"stratified background: interface reflection {reflection_i():+.3f}; omega = {omega}, "
        f"k D = {k * D_LAYER:.3f}"
    )
    for z_i in (5.0, D_LAYER):
        where = "against the layer" if z_i == D_LAYER else f"{z_i - D_LAYER:g} m below the layer"
        print(f"\ninterface at z = {z_i:g} ({where})")
        for name in ("const", "smooth"):
            profile = cg.PROFILES[name][0]
            ex = exact(omega, name, z_i)
            print(f"  profile '{name}'")
            for p in (0, 1, 2):
                errs, extra_vals = [], []
                for n in NS:
                    mass, coup, rhs, readout = system(omega, n, p, p, profile, z_i)
                    errs.append(abs(complex(readout @ np.linalg.solve(mass - coup, rhs)) - ex) / abs(ex))
                    if name == "smooth":
                        extra_vals.append(t2_ratio(omega, n, p, z_i))
                    else:
                        born = complex(readout @ np.linalg.solve(mass, rhs))
                        extra_vals.append(abs(born - born_predicted(omega, n, p, z_i)) / abs(born))
                orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(NS) - 1)]
                # an order is read only where both errors are clear of round-off
                usable = [o for e, o in zip(errs[1:], orders, strict=True) if e > 5e-15]
                ok_order = bool(usable) and abs(usable[-1] - (2 * p + 2)) < 0.3
                if name == "const":
                    ok_extra = max(extra_vals) < 1e-9
                    extra = "Born against the four-path prediction " + " ".join(
                        f"{v:.1e}" for v in extra_vals
                    )
                else:
                    ok_extra = all(abs(v - 1.0) < 0.05 for v in extra_vals[1:])
                    extra = "error of T2 / E_p " + " ".join(f"{v:.3f}" for v in extra_vals)
                oks += [ok_order, ok_extra]
                print(
                    f"    p = r = {p}: error "
                    + " ".join(f"{e:.2e}" for e in errs)
                    + " | orders "
                    + " ".join(f"{o:.2f}" for o in orders)
                    + f" (predicted {2 * p + 2}) {'PASS' if ok_order else 'FAIL'}"
                )
                print(f"               {extra}: {'PASS' if ok_extra else 'FAIL'}")
    # [3b] the law is exact in the long-wave limit: the ratio tends to one as the frequency falls
    print("\nlong-wave limit, smooth profile, p = r = 1, n = 8, error of T2 / E_1 - 1:")
    worst = []
    for z_i in (5.0, D_LAYER):
        gaps = [t2_ratio(om, 8, 1, z_i) - 1.0 for om in (600.0, 300.0, 100.0, 30.0)]
        worst.append(abs(gaps[-1]))
        print(
            f"   interface at {z_i:g}, k D = 0.24, 0.12, 0.04, 0.012: "
            + " ".join(f"{g:+.2e}" for g in gaps)
        )
    ok4 = max(worst) < 5e-3
    oks.append(ok4)
    print(f"   the ratio tends to one: {'PASS' if ok4 else 'FAIL'}")
    print(f"\n{sum(oks)}/{len(oks)} checks passed")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
