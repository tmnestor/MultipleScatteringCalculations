#!/usr/bin/env python3
"""Oblique incidence, P, SV and SH, with the layer of voxels inside a STRATIFIED background.

THE BACKGROUND has three parts: a half-space A above z = z_A, the layer medium between z_A and z_B, and
a half-space B below z_B.  The voxels (n planes, 0 < z < D, z_A <= 0 and z_B >= D) carry the departure
from the layer medium.  With A equal to the layer medium the background is two half-spaces joined at
z_B; with A different it is a three-layer background, and the voxels sit in its middle layer, which
reverberates between the two interfaces.

THE KERNEL.  At each lateral wavenumber the whole-space kernel factorises over the three downgoing and
three upgoing plane-wave modes, P_ws(dz) = D diag(exp(i kz |dz|)) S (``sweep_modes``).  The background
adds a reverberation: a source at z' radiates s_d = S_down q downwards and s_u = S_up q upwards, and with
R_A, R_B the 3 x 3 reflection matrices of the two interfaces as seen from inside the layer and
E(L) = diag(exp(i kz L)), H = z_B - z_A,

    D_A = (I - R_A E(H) R_B E(H))^-1 R_A [ E(z' - z_A) s_u + E(H) R_B E(z_B - z') s_d ],
    U_B = R_B [ E(z_B - z') s_d + E(H) D_A ],
    K_rev(z, z') q = D_down E(z - z_A) D_A + D_up E(z_B - z) U_B.

It is smooth in z and z' and separable, so its cell integrals are one-dimensional.  The reflection and
transmission matrices come from continuity of displacement and traction of the mode states at each
interface.  The incident field is the field of a plane wave arriving from A in the background, and the
answer is the change of the same-type reflection coefficient seen in A.

THE SCHEME is ``crosscheck_first_moment_voxel.Scheme`` (Bloch coupling by Poisson summation, every
integral by quadrature) with the reverberation added to every block, the background field as the
right-hand side, and the upgoing modes arriving at z_A, transmitted into A, as the output.

EXACT: the same-type reflection coefficient of the stack with the contrast layer, less that of the
background, from the package's Kennett recursion.  Checked first, with no voxels: the background's own
reflection from the modal construction against Kennett's.

Run:  conda run -n seismic python -u scripts/measure_oblique_stratified_background.py [omega] [theta]
Seismic units (km, km/s, g/cm^3, GPa); e^{-i omega t}; z down.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crosscheck_first_moment_voxel import N_GAUSS, Scheme, gauss, lateral, phi  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kennett_layers import IsotropicLayer, LayerStack, kennett_layers  # noqa: E402
from cubic_scattering.sweep_modes import modes_to_state, vertical_factorisation  # noqa: E402

LAYER = ReferenceMedium(5.0, 3.0, 2.5)
ABOVE = ReferenceMedium(4.2, 2.4, 2.3)  # the upper half-space of the three-layer background
BELOW = ReferenceMedium(6.5, 3.7, 2.8)
CONTRAST = {"dlambda": 2.0, "dmu": 1.0, "drho": 0.1}
D_LAYER = 0.002
BASES = {0: [1], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
MODE = {"P": 0, "SV": 1, "SH": 2}
NS = (1, 2, 4)


def traction_map(ref: ReferenceMedium) -> np.ndarray:
    """(u, traction on a plane z = const) from the nine-component state, shape (6, 9)."""
    mu = ref.rho * ref.beta**2
    lam = ref.rho * ref.alpha**2 - 2 * mu
    out = np.zeros((6, 9))
    out[:3, :3] = np.eye(3)
    out[3, 3:6] = lam
    out[3, 3] += 2 * mu  # sigma_zz
    out[4, 8] = mu  # sigma_zx from 2 e_zx
    out[5, 7] = mu  # sigma_zy from 2 e_zy
    return out


def interface(kx: float, ky: float, omega: float, top: ReferenceMedium, bot: ReferenceMedium) -> dict:
    """Reflection and transmission of the six modes at an interface, each a 3 x 3 matrix.

    r_down, t_down: a downgoing wave in ``top`` gives upgoing in ``top`` and downgoing in ``bot``;
    r_up, t_up: an upgoing wave in ``bot`` gives downgoing in ``bot`` and upgoing in ``top``.
    """
    ft = traction_map(top) @ modes_to_state(kx, ky, omega, top)
    fb = traction_map(bot) @ modes_to_state(kx, ky, omega, bot)
    x = np.linalg.solve(np.hstack([ft[:, 3:], -fb[:, :3]]), -ft[:, :3])
    y = np.linalg.solve(np.hstack([fb[:, :3], -ft[:, 3:]]), -fb[:, 3:])
    return {"r_down": x[:3], "t_down": x[3:], "r_up": y[:3], "t_up": y[3:]}


class Order:
    """Everything the reverberation needs at one lateral wavenumber."""

    def __init__(self, kx: float, ky: float, omega: float, z_a: float, z_b: float, above, below, d: float):
        self.z_a, self.z_b = z_a, z_b
        emb = modes_to_state(kx, ky, omega, LAYER)
        self.d_down, self.d_up = emb[:, :3], emb[:, 3:]
        fit = 0.1 * d
        down = vertical_factorisation(kx, ky, fit, omega, LAYER)
        up = vertical_factorisation(kx, ky, -fit, omega, LAYER)
        self.s_down, self.s_up = down.source[:3], up.source[3:]
        self.kz = down.kz[:3]
        top = interface(kx, ky, omega, above, LAYER)
        self.r_a, self.t_in, self.t_out, self.r_top = top["r_up"], top["t_down"], top["t_up"], top["r_down"]
        self.r_b = interface(kx, ky, omega, LAYER, below)["r_down"]
        self.e_h = np.diag(np.exp(1j * self.kz * (z_b - z_a)))
        self.w = np.linalg.inv(np.eye(3) - self.r_a @ self.e_h @ self.r_b @ self.e_h)

    def from_top(self, z: np.ndarray) -> np.ndarray:
        """exp(i kz (z - z_A)) per mode, shape (len(z), 3)."""
        return np.exp(1j * np.outer(z - self.z_a, self.kz))

    def to_bottom(self, z: np.ndarray) -> np.ndarray:
        """exp(i kz (z_B - z)) per mode."""
        return np.exp(1j * np.outer(self.z_b - z, self.kz))

    def radiated(self, e_top: np.ndarray, e_bot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(D_A, U_B) as 3 x 9 operators on the source, given the source cell's moments of the two
        exponentials (3-vectors)."""
        d_a = (
            self.w
            @ self.r_a
            @ (np.diag(e_top) @ self.s_up + self.e_h @ self.r_b @ np.diag(e_bot) @ self.s_down)
        )
        u_b = self.r_b @ (np.diag(e_bot) @ self.s_down + self.e_h @ d_a)
        return d_a, u_b

    def background(self, amp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(D0 at z_A, U0 at z_B): the field in the layer of a wave of mode amplitudes ``amp`` from A."""
        d0 = self.w @ self.t_in @ amp
        return d0, self.r_b @ self.e_h @ d0

    def reflection(self) -> np.ndarray:
        """The background's reflection matrix seen in A, referred to z_A."""
        return self.r_top + self.t_out @ self.e_h @ self.r_b @ self.e_h @ self.w @ self.t_in


class StratifiedScheme(Scheme):
    """The Galerkin chain with the reverberation of a three-part background."""

    def __init__(self, omega, kx, n, basis, p_max, incident, z_a, z_b, above, below):
        super().__init__(LAYER, CONTRAST, D_LAYER, omega, kx, n, basis, p_max, incident)
        self.z_a, self.z_b, self.above, self.below = z_a, z_b, above, below

    def moments(self, order: Order) -> tuple[np.ndarray, np.ndarray]:
        """Cell moments of the two exponentials: [plane, z-degree, mode]."""
        s, w = gauss(-self.h, self.h, N_GAUSS)
        top = np.zeros((self.n, max(self.zdeg) + 1, 3), dtype=complex)
        bot = np.zeros_like(top)
        for i, zc in enumerate(self.zs):
            for a in self.zdeg:
                wt = w * phi(a, s, self.h)
                top[i, a] = wt @ order.from_top(zc + s)
                bot[i, a] = wt @ order.to_bottom(zc + s)
        return top, bot

    def solve_change(self) -> complex:
        """The change of the same-type reflection coefficient in A caused by the voxels."""
        n, nb = self.n, len(self.bas)
        ztab = {0: self.z_same()}
        for m in range(1, n):
            ztab[m] = self.z_between(m)
            ztab[-m] = self.z_between(-m)
        big = np.zeros((9 * nb * n, 9 * nb * n), dtype=complex)
        cache: dict = {}
        # the reverberation, order by order of the reciprocal lattice
        rev = np.zeros((n, nb, n, nb, 9, 9), dtype=complex)
        specular = None
        for ky in self.kys:
            for kxv in self.kxs:
                order = Order(kxv, ky, self.om, self.z_a, self.z_b, self.above, self.below, self.d)
                top, bot = self.moments(order)
                if abs(kxv - self.kx) < 1e-12 and abs(ky) < 1e-12:
                    specular = (order, top, bot)
                for j in range(n):
                    for bi, b in enumerate(self.bas):
                        d_a, u_b = order.radiated(top[j, b[0]], bot[j, b[0]])
                        lat_b = lateral(b[1], -kxv, self.h) * lateral(b[2], -ky, self.h)
                        for i in range(n):
                            for ai, a in enumerate(self.bas):
                                lat = lat_b * lateral(a[1], kxv, self.h) * lateral(a[2], ky, self.h)
                                rev[i, ai, j, bi] += lat * (
                                    order.d_down @ np.diag(top[i, a[0]]) @ d_a
                                    + order.d_up @ np.diag(bot[i, a[0]]) @ u_b
                                )
        rev /= self.d**2
        for i in range(n):
            for j in range(n):
                m = i - j
                for ai, a in enumerate(self.bas):
                    for bi, b in enumerate(self.bas):
                        key = (m, a, b)
                        if key not in cache:
                            cache[key] = self.coupling(ztab[m], a, b)
                        blk = -(cache[key] + rev[i, ai, j, bi]) @ self.dds[j]
                        if i == j and ai == bi:
                            blk = blk + self.d**3 * np.prod([1.0 / (2 * q + 1) for q in a]) * np.eye(9)
                        r0, c0 = 9 * (nb * i + ai), 9 * (nb * j + bi)
                        big[r0 : r0 + 9, c0 : c0 + 9] = blk
        order, top, bot = specular
        amp = np.zeros(3)
        amp[MODE[self.incident]] = 1.0
        d0, u0 = order.background(amp)
        rhs = np.zeros(9 * nb * n, dtype=complex)
        for i in range(n):
            for ai, a in enumerate(self.bas):
                state = order.d_down @ (top[i, a[0]] * d0) + order.d_up @ (bot[i, a[0]] * u0)
                r0 = 9 * (nb * i + ai)
                rhs[r0 : r0 + 9] = state * lateral(a[1], self.kx, self.h) * lateral(a[2], 0.0, self.h)
        sol = np.linalg.solve(big, rhs)
        arriving = np.zeros(3, dtype=complex)  # upgoing mode amplitudes at z_A, from below
        for j in range(n):
            for bi, b in enumerate(self.bas):
                _, u_b = order.radiated(top[j, b[0]], bot[j, b[0]])
                op = np.diag(top[j, b[0]]) @ order.s_up + order.e_h @ u_b
                lat = lateral(b[1], -self.kx, self.h) * lateral(b[2], 0.0, self.h)
                c = sol[9 * (nb * j + bi) : 9 * (nb * j + bi) + 9]
                arriving += lat * (op @ self.dds[j] @ c) / self.d**2
        return complex((order.t_out @ arriving)[MODE[self.incident]])


def kennett_same_type(incident: str, theta: float, omega: float, z_a, z_b, above, below, layer_medium):
    """Same-type reflection coefficient in A of the stack whose slab 0 < z < D is ``layer_medium``."""
    parts = [IsotropicLayer(above.alpha, above.beta, above.rho, 100.0)]
    for med, thick in ((LAYER, -z_a), (layer_medium, D_LAYER), (LAYER, z_b - D_LAYER)):
        if thick > 1e-15:
            parts.append(IsotropicLayer(med.alpha, med.beta, med.rho, thick))
    parts.append(IsotropicLayer(below.alpha, below.beta, below.rho, np.inf))
    slowness = np.sin(np.deg2rad(theta)) / (above.alpha if incident == "P" else above.beta)
    res = kennett_layers(LayerStack(parts), slowness, np.array([omega]))
    return {
        "P": complex(res.RD_psv[0][0, 0]),
        "SV": complex(res.RD_psv[0][1, 1]),
        "SH": complex(res.RD_sh[0]),
    }[incident]


def contrast_medium() -> ReferenceMedium:
    mu = LAYER.rho * LAYER.beta**2 + CONTRAST["dmu"]
    lam = LAYER.rho * (LAYER.alpha**2 - 2 * LAYER.beta**2) + CONTRAST["dlambda"]
    rho = LAYER.rho + CONTRAST["drho"]
    return ReferenceMedium(float(np.sqrt((lam + 2 * mu) / rho)), float(np.sqrt(mu / rho)), rho)


def main() -> int:
    omega = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0
    theta = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    t0 = time.perf_counter()
    gap = 0.003
    cases = (
        ("two half-spaces, interface 3 m below the layer", 0.0, D_LAYER + gap, LAYER, BELOW),
        ("two half-spaces, interface against the lowest voxel", 0.0, D_LAYER, LAYER, BELOW),
        ("three-layer background, interfaces 3 m above and below", -gap, D_LAYER + gap, ABOVE, BELOW),
        ("three-layer background, interfaces against the voxels", 0.0, D_LAYER, ABOVE, BELOW),
    )
    print(f"omega = {omega}, incidence {theta} degrees in the upper half-space; planes n = {NS}")
    oks = []
    for name, z_a, z_b, above, below in cases:
        print(f"\n{name}")
        for incident in ("P", "SV", "SH"):
            k_inc = omega / (above.alpha if incident == "P" else above.beta)
            kx = k_inc * np.sin(np.deg2rad(theta))
            # the background's own reflection: the modal construction against Kennett's recursion
            modal = Order(kx, 0.0, omega, z_a, z_b, above, below, D_LAYER).reflection()
            modal = complex(modal[MODE[incident], MODE[incident]])
            kn_bg = kennett_same_type(incident, theta, omega, z_a, z_b, above, below, LAYER)
            bg_err = min(abs(modal - kn_bg), abs(modal + kn_bg)) / max(abs(kn_bg), 1e-300)
            sign = 1.0 if abs(modal - kn_bg) <= abs(modal + kn_bg) else -1.0
            change = sign * (
                kennett_same_type(incident, theta, omega, z_a, z_b, above, below, contrast_medium()) - kn_bg
            )
            ok_bg = bg_err < 1e-9 or abs(kn_bg) < 1e-12
            oks.append(ok_bg)
            print(
                f"  incident {incident:2s}: background reflection, modes against Kennett {bg_err:.1e} "
                f"{'PASS' if ok_bg else 'FAIL'}; |change due to the layer| = {abs(change):.3e}"
            )
            for p in (0, 1, 2):
                errs = []
                for n in NS:
                    got = StratifiedScheme(
                        omega, kx, n, BASES[p], 2, incident, z_a, z_b, above, below
                    ).solve_change()
                    errs.append(abs(got - change) / abs(change))
                orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(NS) - 1)]
                ok = abs(orders[-1] - (2 * p + 2)) < 0.4
                oks.append(ok)
                print(
                    f"     p = {p}: "
                    + "  ".join(f"{e:.2e}" for e in errs)
                    + "  orders "
                    + " ".join(f"{o:.2f}" for o in orders)
                    + f"  (predicted {2 * p + 2}) {'PASS' if ok else 'FAIL'}"
                    + f"  [{time.perf_counter() - t0:4.0f} s]",
                    flush=True,
                )
    print(f"\n{sum(oks)}/{len(oks)} checks passed")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
