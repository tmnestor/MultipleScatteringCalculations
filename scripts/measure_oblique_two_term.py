#!/usr/bin/env python3
"""The two-term law at oblique incidence, in the whole space and in a stratified background.

The scheme and the backgrounds are those of ``measure_oblique_stratified_background.py``; here the
contrast of each plane of voxels may vary with depth inside the plane (a profile f(z), projected to
Legendre degree r in each plane), and the scattered wave is separated by its order in the contrast.

THE SYSTEM.  With M the Gram matrix, K the coupling (whole-space kernel plus reverberation, times the
projected contrast), b the moments of the background field and r the row that gives the change of the
same-type reflection coefficient,  u = r (M - K)^-1 b,  and the first two terms of its series in the
contrast are, with no differencing,

    T1 = r M^-1 b,        T2 = r M^-1 K M^-1 b.

[1] THE FIRST-ORDER TERM, uniform layer.  The background field in the layer is a sum of six plane waves
    (P, SV, SH going down and up) and the voxels radiate into six.  Prediction: T1 of the scheme is the
    sum over the 36 pairs of the EXACT first-order term of that pair times the wave factor of a cell,
    ``octree.born_wave_factor(k_in, k_out, h, p)``, with the pair's own two wavevectors.
[2] THE SECOND-ORDER TERM, smooth profile 1 + sin(2 pi z / D) / 2, contrast degree r = p.  Prediction: the
    relative error of T2 is the relative projection error E_p of the profile, in the long-wave limit.
    The exact T2 comes from the exact stack at four scaled contrasts (five-point differences); the exact
    stack is the propagator of the six-component state (u, traction) across the graded layer, integrated
    numerically with the local system matrix A(z) = F(z) diag(i kz) F(z)^-1 built from the plane-wave
    modes of the medium at depth z, between the radiation conditions of the two half-spaces.
    Checked first: with a constant profile the exact stack equals Kennett's recursion; and for the graded
    layer it equals the project's impedance march (invariant imbedding: the Riccati equation
    Y' = A21 + A22 Y - Y A11 - Y A12 Y integrated upwards from the downgoing impedance of the lower
    half-space, gate_first_order_impedance_march), an independent algorithm.

Run:  conda run -n seismic python -u scripts/measure_oblique_two_term.py [theta] [--part=2] [--cases=3]
          [--waves=SV,SH]
The whole run takes about two and a half hours; --part, --cases (indices 0 to 3 of the backgrounds) and
--waves select a part of it.
Seismic units (km, km/s, g/cm^3, GPa); e^{-i omega t}; z down.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crosscheck_graded_contrast as cg  # noqa: E402
from crosscheck_first_moment_voxel import N_GAUSS, gauss, lateral, phi  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel.octree import born_wave_factor  # noqa: E402
from cubic_scattering.sweep_modes import mode_basis, modes_to_state  # noqa: E402
from measure_oblique_stratified_background import (  # noqa: E402
    ABOVE,
    BASES,
    BELOW,
    CONTRAST,
    D_LAYER,
    LAYER,
    MODE,
    Order,
    StratifiedScheme,
    contrast_medium,
    kennett_same_type,
    traction_map,
)
from scripts.gate_first_order_impedance import ablocks  # noqa: E402
from scripts.gate_first_order_impedance_march import (  # noqa: E402
    mode_columns,
    step_mobius,
    step_rk4,
    y_downgoing,
)


def smooth(z: np.ndarray) -> np.ndarray:
    return 1.0 + 0.5 * np.sin(2 * np.pi * np.asarray(z) / D_LAYER)


def const(z: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(z, dtype=float))


class GradedScheme(StratifiedScheme):
    """The stratified scheme with a contrast that varies with depth inside each plane of voxels."""

    def __init__(self, omega, kx, n, p, incident, z_a, z_b, above, below, profile, r):
        super().__init__(omega, kx, n, BASES[p], 2, incident, z_a, z_b, above, below)
        self.r = r
        pz = max(b[0] for b in self.bas)
        self.zdeg = list(range(pz + r + 1))  # the source side needs depth degrees up to p + r
        s, w = gauss(-self.h, self.h, N_GAUSS)
        coef = np.array(
            [
                [
                    (2 * c + 1) / (2 * self.h) * np.sum(w * profile(zc + s) * phi(c, s, self.h))
                    for c in range(r + 1)
                ]
                for zc in self.zs
            ]
        )
        lin = np.array(
            [[[cg.lin_quad(c, b, e) for e in self.zdeg] for b in range(pz + 1)] for c in range(r + 1)]
        )
        self.g = np.einsum("jc,cbe->jbe", coef, lin)  # f P_b = sum_e g[j, b, e] P_e in plane j
        self.delta = self.dds[0]

    def system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(mass diagonal, K, b, r): u = r (M - K)^-1 b is the change of the same-type coefficient."""
        n, nb = self.n, len(self.bas)
        ztab = {0: self.z_same()}
        for m in range(1, n):
            ztab[m] = self.z_between(m)
            ztab[-m] = self.z_between(-m)
        srcs = sorted({(e, b[1], b[2]) for b in self.bas for e in self.zdeg})
        s_index = {t: k for k, t in enumerate(srcs)}
        rev = np.zeros((n, nb, n, len(srcs), 9, 9), dtype=complex)
        specular = None
        for ky in self.kys:
            for kxv in self.kxs:
                order = Order(kxv, ky, self.om, self.z_a, self.z_b, self.above, self.below, self.d)
                top, bot = self.moments(order)
                if abs(kxv - self.kx) < 1e-12 and abs(ky) < 1e-12:
                    specular = (order, top, bot)
                for j in range(n):
                    for si, b in enumerate(srcs):
                        d_a, u_b = order.radiated(top[j, b[0]], bot[j, b[0]])
                        lat_b = lateral(b[1], -kxv, self.h) * lateral(b[2], -ky, self.h)
                        for i in range(n):
                            for ai, a in enumerate(self.bas):
                                lat = lat_b * lateral(a[1], kxv, self.h) * lateral(a[2], ky, self.h)
                                rev[i, ai, j, si] += lat * (
                                    order.d_down @ np.diag(top[i, a[0]]) @ d_a
                                    + order.d_up @ np.diag(bot[i, a[0]]) @ u_b
                                )
        rev /= self.d**2
        size = 9 * nb * n
        k_mat = np.zeros((size, size), dtype=complex)
        mass = np.zeros(size)
        cache: dict = {}
        for i in range(n):
            for ai, a in enumerate(self.bas):
                r0 = 9 * (nb * i + ai)
                mass[r0 : r0 + 9] = self.d**3 * np.prod([1.0 / (2 * q + 1) for q in a])
                for j in range(n):
                    for bi, b in enumerate(self.bas):
                        blk = np.zeros((9, 9), dtype=complex)
                        for e in self.zdeg:
                            wgt = self.g[j, b[0], e]
                            if wgt == 0.0:
                                continue
                            src = (e, b[1], b[2])
                            key = (i - j, a, src)
                            if key not in cache:
                                cache[key] = self.coupling(ztab[i - j], a, src)
                            blk += wgt * (cache[key] + rev[i, ai, j, s_index[src]])
                        c0 = 9 * (nb * j + bi)
                        k_mat[r0 : r0 + 9, c0 : c0 + 9] = blk @ self.delta
        order, top, bot = specular
        amp = np.zeros(3)
        amp[MODE[self.incident]] = 1.0
        d0, u0 = order.background(amp)
        rhs = np.zeros(size, dtype=complex)
        row = np.zeros(size, dtype=complex)
        out = order.t_out[MODE[self.incident]]
        for i in range(n):
            for ai, a in enumerate(self.bas):
                state = order.d_down @ (top[i, a[0]] * d0) + order.d_up @ (bot[i, a[0]] * u0)
                r0 = 9 * (nb * i + ai)
                rhs[r0 : r0 + 9] = state * lateral(a[1], self.kx, self.h) * lateral(a[2], 0.0, self.h)
                op = np.zeros((3, 9), dtype=complex)
                for e in self.zdeg:
                    _, u_b = order.radiated(top[i, e], bot[i, e])
                    op += self.g[i, a[0], e] * (np.diag(top[i, e]) @ order.s_up + order.e_h @ u_b)
                lat = lateral(a[1], -self.kx, self.h) * lateral(a[2], 0.0, self.h)
                row[r0 : r0 + 9] = lat * (out @ op @ self.delta) / self.d**2
        self.specular = specular
        self.amp0 = (d0, u0)
        return mass, k_mat, rhs, row

    def born_predicted(self, p: int) -> complex:
        """T1 of the uniform layer from the 36 pairs of plane waves, each with its own wave factor."""
        order, _, _ = self.specular
        d0, u0 = self.amp0
        kz = order.kz
        # the six incident waves in the layer: amplitude, vertical phase reference, wavevector, state
        waves = []
        for m in range(3):
            waves.append(
                (d0[m], lambda z, m=m: np.exp(1j * kz[m] * (z - order.z_a)), kz[m], order.d_down[:, m])
            )
            waves.append(
                (u0[m], lambda z, m=m: np.exp(1j * kz[m] * (order.z_b - z)), -kz[m], order.d_up[:, m])
            )
        s, w = gauss(-self.h, self.h, 24)
        up_part = np.zeros(3, dtype=complex)
        down_part = np.zeros(3, dtype=complex)
        for zc in self.zs:
            z = zc + s
            for amp, phase, kin_z, state in waves:
                k_in = np.array([kin_z.real, self.kx, 0.0])
                q = self.delta @ state * amp
                for m in range(3):
                    gamma = kz[m]
                    # radiated upwards in mode m: wavevector (-gamma, kx, 0); downwards: (+gamma, kx, 0)
                    f_up = born_wave_factor(k_in, np.array([-gamma.real, self.kx, 0.0]), self.h, p)
                    f_dn = born_wave_factor(k_in, np.array([gamma.real, self.kx, 0.0]), self.h, p)
                    i_up = np.sum(w * phase(z) * np.exp(1j * gamma * (z - order.z_a)))
                    i_dn = np.sum(w * phase(z) * np.exp(1j * gamma * (order.z_b - z)))
                    up_part[m] += f_up * i_up * (order.s_up[m] @ q)
                    down_part[m] += f_dn * i_dn * (order.s_down[m] @ q)
        d_a = order.w @ order.r_a @ (up_part + order.e_h @ order.r_b @ down_part)
        u_b = order.r_b @ (down_part + order.e_h @ d_a)
        arriving = up_part + order.e_h @ u_b
        return complex((order.t_out @ arriving)[MODE[self.incident]])


# ---------------------------------------------------------------------------- the exact graded stack
def local_medium(z: float, profile, scale: float) -> ReferenceMedium:
    f = float(profile(np.array([z]))[0]) if 0.0 <= z <= D_LAYER else 0.0
    mu = LAYER.rho * LAYER.beta**2 + scale * CONTRAST["dmu"] * f
    lam = LAYER.rho * (LAYER.alpha**2 - 2 * LAYER.beta**2) + scale * CONTRAST["dlambda"] * f
    rho = LAYER.rho + scale * CONTRAST["drho"] * f
    return ReferenceMedium(float(np.sqrt((lam + 2 * mu) / rho)), float(np.sqrt(mu / rho)), rho)


def mode_frame(kx: float, omega: float, med: ReferenceMedium) -> tuple[np.ndarray, np.ndarray]:
    """(F, kz): the six modes' (u, traction) states and their signed vertical wavenumbers."""
    return traction_map(med) @ modes_to_state(kx, 0.0, omega, med), mode_basis(
        kx, 0.0, omega, med
    ).k_vectors[:, 0]


def exact_same_type(incident, kx, omega, z_a, z_b, above, below, profile, scale) -> complex:
    """Same-type reflection coefficient in A of the stack with the graded layer at contrast ``scale``."""

    def a_matrix(z: float) -> np.ndarray:
        f, kz = mode_frame(kx, omega, local_medium(z, profile, scale))
        return f @ np.diag(1j * kz) @ np.linalg.inv(f)

    def homogeneous(length: float) -> np.ndarray:
        f, kz = mode_frame(kx, omega, LAYER)
        return f @ np.diag(np.exp(1j * kz * length)) @ np.linalg.inv(f)

    sol = solve_ivp(
        lambda z, y: (a_matrix(z) @ y.reshape(6, 6)).ravel(),
        (0.0, D_LAYER),
        np.eye(6, dtype=complex).ravel(),
        rtol=1e-13,
        atol=1e-16,
        method="DOP853",
    )
    total = homogeneous(z_b - D_LAYER) @ sol.y[:, -1].reshape(6, 6) @ homogeneous(-z_a)
    fa, _ = mode_frame(kx, omega, above)
    fb, _ = mode_frame(kx, omega, below)
    amp = np.zeros(3)
    amp[MODE[incident]] = 1.0
    x = np.linalg.solve(np.hstack([total @ fa[:, 3:], -fb[:, :3]]), -total @ fa[:, :3] @ amp)
    return complex(x[MODE[incident]])


def imbedding_same_type(incident, kx, omega, z_a, z_b, above, below, profile, scale, nsub=400) -> complex:
    """The same coefficient by the impedance march: Y from the lower half-space up to z_A, then the
    reflection in A from continuity of displacement and traction.  Homogeneous gaps are stepped exactly;
    the graded layer by fourth-order Runge-Kutta with the medium sampled at the stages."""
    y = y_downgoing(below, omega, kx)
    if z_b > D_LAYER + 1e-15:
        y = step_mobius(y, ablocks(LAYER, omega, kx, 0.0), z_b - D_LAYER)

    def inside(z: float) -> ReferenceMedium:
        return local_medium(min(max(z, 0.0), D_LAYER), profile, scale)

    y = step_rk4(y, inside, D_LAYER, 0.0, omega, kx, nsub)
    if z_a < -1e-15:
        y = step_mobius(y, ablocks(LAYER, omega, kx, 0.0), -z_a)
    dd, du = mode_columns(above, omega, kx)
    r = -np.linalg.solve(du[3:, :] - y @ du[:3, :], dd[3:, :] - y @ dd[:3, :])
    return complex(r[MODE[incident], MODE[incident]])


def second_order(f, eps: float) -> complex:
    v = {s: f(s) for s in (eps, -eps, 2 * eps, -2 * eps)}
    return (16 * (v[eps] + v[-eps]) - (v[2 * eps] + v[-2 * eps])) / (24 * eps * eps)


def projection_error(n: int, q: int) -> float:
    h = D_LAYER / (2 * n)
    s, w = gauss(-h, h, 24)
    num = den = 0.0
    for zc in (np.arange(n) + 0.5) * 2 * h:
        vals = smooth(zc + s)
        fit = sum(
            (2 * c + 1) / (2 * h) * np.sum(w * vals * phi(c, s, h)) * phi(c, s, h) for c in range(q + 1)
        )
        num += float(np.sum(w * (vals - fit) ** 2))
        den += float(np.sum(w * vals**2))
    return num / den


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    theta = float(args[0]) if args else 20.0
    parts = [int(v) for v in opts.get("part", "0,1,2").split(",")]
    waves = tuple(opts.get("waves", "P,SV,SH").split(","))
    t0 = time.perf_counter()
    gap = 0.003
    cases = (
        ("whole space", 0.0, D_LAYER, LAYER, LAYER),
        ("two half-spaces, interface 3 m below", 0.0, D_LAYER + gap, LAYER, BELOW),
        ("three layers, interfaces 3 m away", -gap, D_LAYER + gap, ABOVE, BELOW),
        ("three layers, interfaces against the voxels", 0.0, D_LAYER, ABOVE, BELOW),
    )
    cases = tuple(cases[int(v)] for v in opts.get("cases", "0,1,2,3").split(","))
    oks = []

    def kx_of(incident: str, omega: float, above: ReferenceMedium) -> float:
        return omega / (above.alpha if incident == "P" else above.beta) * np.sin(np.deg2rad(theta))

    # ---- the exact graded stack against Kennett, constant profile
    worst = 0.0
    for _, z_a, z_b, above, below in cases if 0 in parts else ():
        for incident in waves:
            kx = kx_of(incident, 3000.0, above)
            mine = exact_same_type(incident, kx, 3000.0, z_a, z_b, above, below, const, 1.0)
            kn = kennett_same_type(incident, theta, 3000.0, z_a, z_b, above, below, contrast_medium())
            worst = max(worst, min(abs(mine - kn), abs(mine + kn)) / abs(kn))
    ok = worst < 1e-8
    oks.append(ok)
    print(
        f"exact stack by integration against Kennett, uniform layer, 12 cases: {worst:.1e} "
        f"{'PASS' if ok else 'FAIL'}"
    )

    # ---- the graded stack by two independent algorithms: the propagator and the impedance march
    worst = 0.0
    for _, z_a, z_b, above, below in cases if 0 in parts else ():
        for incident in waves:
            kx = kx_of(incident, 300.0, above)
            mine = exact_same_type(incident, kx, 300.0, z_a, z_b, above, below, smooth, 1.0)
            imb = imbedding_same_type(incident, kx, 300.0, z_a, z_b, above, below, smooth, 1.0)
            worst = max(worst, min(abs(mine - imb), abs(mine + imb)) / abs(mine))
    ok = worst < 1e-10
    oks.append(ok)
    print(
        f"graded stack, propagator against the impedance march, 12 cases: {worst:.1e} "
        f"{'PASS' if ok else 'FAIL'}"
    )

    # ---- [1] the first-order term from the 36 pairs
    print(
        "\n[1] first-order term of the scheme against the 36-pair prediction, uniform layer, "
        f"omega = 3000, {theta} deg"
    )
    for name, z_a, z_b, above, below in cases if 1 in parts else ():
        for incident in waves:
            kx = kx_of(incident, 3000.0, above)
            diffs = []
            for p in (0, 1, 2):
                for n in (1, 2):
                    sch = GradedScheme(3000.0, kx, n, p, incident, z_a, z_b, above, below, const, 0)
                    mass, _, rhs, row = sch.system()
                    t1 = complex(row @ (rhs / mass))
                    diffs.append(abs(t1 - sch.born_predicted(p)) / abs(t1))
            ok = max(diffs) < 1e-9
            oks.append(ok)
            print(
                f"  {name:44s} {incident:2s}: worst over p = 0, 1, 2 and n = 1, 2: {max(diffs):.1e} "
                f"{'PASS' if ok else 'FAIL'}  [{time.perf_counter() - t0:4.0f} s]",
                flush=True,
            )

    # ---- [2] the second-order term and the projection error
    print(
        "\n[2] smooth profile, r = p = 0, 1, 2: relative error of T2 over E_p (n = 2, 4 for each p), "
        "at k_S D = 0.2 and 0.067"
    )
    for name, z_a, z_b, above, below in cases if 2 in parts else ():
        for incident in waves:
            line = []
            ok = True
            for omega in (300.0, 100.0):
                kx = kx_of(incident, omega, above)

                def stack(s, kx=kx, omega=omega, geom=(incident, z_a, z_b, above, below)):
                    return exact_same_type(geom[0], kx, omega, *geom[1:], smooth, s)

                bg = stack(0.0)
                t2_exact = second_order(lambda s, bg=bg, stack=stack: stack(s) - bg, 0.1)
                for p in (0, 1, 2):
                    for n in (2, 4):
                        sch = GradedScheme(omega, kx, n, p, incident, z_a, z_b, above, below, smooth, p)
                        mass, k_mat, rhs, row = sch.system()
                        t2 = complex(row @ ((k_mat @ (rhs / mass)) / mass))
                        sign = 1.0 if abs(t2 - t2_exact) <= abs(t2 + t2_exact) else -1.0
                        ratio = abs(sign * t2 - t2_exact) / abs(t2_exact) / projection_error(n, p)
                        line.append(ratio)
                        if omega == 100.0:
                            ok = ok and abs(ratio - 1.0) < (0.05 if p < 2 else 0.1)
            oks.append(ok)
            print(
                f"  {name:44s} {incident:2s}: omega 300: "
                + " ".join(f"{v:.3f}" for v in line[:6])
                + " | omega 100: "
                + " ".join(f"{v:.3f}" for v in line[6:])
                + f"  {'PASS' if ok else 'FAIL'}  [{time.perf_counter() - t0:4.0f} s]",
                flush=True,
            )
    print(f"\n{sum(oks)}/{len(oks)} checks passed")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
