#!/usr/bin/env python3
"""CROSS-CHECK: an independent implementation of the first-moment (Legendre Galerkin) voxel.

``Mathematica/ContinuumLimit_Oblique.wl`` (notebook 9) builds the scheme symbolically: partial fractions of
the transformed Green's tensor, closed-form Legendre moments, closed-form same-cell double integrals.
This script shares only the DEFINITION of the scheme and computes every ingredient another way:

  * the depth dependence of the transformed kernel from the package's residue construction
    ``sweep_kernels.vertical_kernel_9x9`` (validated against the spectral kernel in notebook 2a);
  * the same-plane delta weight as the large-k_z limit of the nine-component transform of the 3-D
    tensor, obtained by inverting the Christoffel matrix numerically (Richardson in 1/k_z^2);
  * every double integral by Gauss quadrature: tensor Gauss between planes; within a plane the inner
    integral is a polynomial in u = s - t, done exactly, and the outer by graded composite Gauss;
  * the P / S split of the reflected wave by sampling the upgoing field at two depths.

THE SCHEME (as notebook 9).  Voxel basis Legendre {1, z/h, x/h, y/h} on the 9-component state; tested
with the same functions; Bloch coupling by Poisson summation over the reciprocal lattice,
    C_ab(m) = (1/d^2) sum_g L_a(kappa) L_b(-kappa) int int phi_a(z) Ghat(kappa; m d + z - z') phi_b(z'),
the double integral over z and z'.
Compared: the raw specular reflection displacements {S, P} at |p|, |q| <= 2, for normal incidence and
20 degrees, n = 1, 2, 4, mean-only and mean + first moments, for an incident P wave
(``Mathematica/ContinuumLimit_oblique_ref.json``, notebook 9) and incident SV and SH waves
(``Mathematica/ContinuumLimit_incidentS_ref.json``, notebook 12), and a stratified model of eight
random cells, one contrast per plane (``Mathematica/ContinuumLimit_heterogeneous_ref.json``, notebook 13).

Run:  conda run -n seismic python scripts/crosscheck_first_moment_voxel.py
SI units.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.sweep_kernels import vertical_kernel_9x9  # noqa: E402

REFJSONS = (
    ROOT / "Mathematica" / "ContinuumLimit_oblique_ref.json",
    ROOT / "Mathematica" / "ContinuumLimit_incidentS_ref.json",
    ROOT / "Mathematica" / "ContinuumLimit_heterogeneous_ref.json",
)
VOIGT = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)  # (z, x, y) pairs: e_zz e_xx e_yy 2e_xy 2e_zy 2e_zx
ENG = (1, 1, 1, 2, 2, 2)
BASIS = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1)}  # Legendre degrees in (z, x, y)
N_GAUSS = 16
T_START = time.perf_counter()


def phi(deg: int, s: np.ndarray, h: float) -> np.ndarray:
    """Legendre P0 or P1 on [-h, h]."""
    return np.ones_like(s) if deg == 0 else s / h


def gauss(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def lateral(deg: int, kap: float, h: float) -> complex:
    """int_{-h}^{h} phi(x) e^{i kappa x} dx by Gauss (40 nodes)."""
    x, w = gauss(-h, h, 40)
    return complex(np.sum(w * phi(deg, x, h) * np.exp(1j * kap * x)))


def nine(uu: np.ndarray, kv: np.ndarray) -> np.ndarray:
    """9x9 from a 3x3 displacement response and a wavevector: strain rows i k_a, moment columns +i k_b."""

    def rows(col: np.ndarray) -> np.ndarray:
        strain = [
            ENG[v] * 0.5 * (1j * kv[a] * col[b] + 1j * kv[b] * col[a]) for v, (a, b) in enumerate(VOIGT)
        ]
        return np.concatenate([col, np.array(strain)])

    cols = [uu[:, j] for j in range(3)]
    cols += [0.5 * (1j * kv[e] * uu[:, c] + 1j * kv[c] * uu[:, e]) for (c, e) in VOIGT]
    return np.array([rows(c) for c in cols]).T


def delta_weight(ref: ReferenceMedium, omega: float, kx: float, ky: float) -> np.ndarray:
    """The delta weight of the transformed kernel: lim_{k_z -> inf} of the 9x9 transform, numerically."""
    lam, mu = ref.rho * (ref.alpha**2 - 2 * ref.beta**2), ref.rho * ref.beta**2

    def at(kz: float) -> np.ndarray:
        k = np.array([kz, kx, ky])
        christoffel = (
            (lam + mu) * np.outer(k, k) + mu * (k @ k) * np.eye(3) - ref.rho * omega**2 * np.eye(3)
        )
        return nine(np.linalg.inv(christoffel), k)

    big = 1e4 * (abs(kx) + abs(ky) + omega / ref.beta)

    # Entries with one k_z (u <- M, strain <- f) decay as 1/k_z and carry NO delta; averaging +-k_z cancels
    # every odd term exactly, and Richardson then removes the even O(1/k_z^2) remainder. Without the
    # symmetrisation a spurious O(1/k_z) weight survives in the odd blocks (measured 3e-4/mu).
    def even(kz: float) -> np.ndarray:
        return 0.5 * (at(kz) + at(-kz))

    return (4 * even(2 * big) - even(big)) / 3.0


class Scheme:
    """The Galerkin chain for one (omega, kx, n, basis, p_max)."""

    def __init__(
        self,
        ref: ReferenceMedium,
        contrast: dict,
        d_layer: float,
        omega: float,
        kx: float,
        n: int,
        basis: list[int],
        p_max: int,
        incident: str,
    ) -> None:
        self.ref, self.om, self.kx, self.n, self.p_max = ref, omega, kx, n, p_max
        self.incident = incident
        self.d = d_layer / n
        self.h = self.d / 2
        self.zs = (np.arange(n) + 0.5) * self.d
        self.bas = [BASIS[b] for b in basis]
        # one contrast for the layer, or one per plane (a stratified model: list of (dl, dm, dr))
        planes = (
            contrast
            if isinstance(contrast, list)
            else [(contrast["dlambda"], contrast["dmu"], contrast["drho"])] * n
        )
        if len(planes) != n:
            raise ValueError(f"{len(planes)} plane contrasts for n = {n} planes")
        self.dds = [self._delta(omega, *c) for c in planes]
        g = 2 * np.pi / self.d * np.arange(-p_max, p_max + 1)
        self.kxs = kx + g
        self.kys = g.copy()

    @staticmethod
    def _delta(omega: float, dl: float, dm: float, dr: float) -> np.ndarray:
        """The 9x9 contrast operator: diag(w^2 drho I3, the 6x6 stiffness with shear 2 dmu)."""
        c6 = np.zeros((6, 6))
        c6[:3, :3] = dl
        c6[np.arange(3), np.arange(3)] = dl + 2 * dm
        c6[np.arange(3, 6), np.arange(3, 6)] = 2 * dm  # the kernel's moment convention: shear 2 dmu
        dd = np.zeros((9, 9), dtype=complex)
        dd[:3, :3] = omega**2 * dr * np.eye(3)
        dd[3:, 3:] = c6
        return dd

    def kern(self, ky: float, z: float) -> np.ndarray:
        """Transformed kernel at all lateral kx nodes, shape (9, 9, n_kx)."""
        return vertical_kernel_9x9(self.kxs, ky, z, self.om, self.ref)

    def z_between(self, m: int) -> dict:
        """Z_ab for planes m apart (m != 0): tensor Gauss, per (ky index): (9, 9, n_kx) per basis z-pair."""
        s, w = gauss(-self.h, self.h, N_GAUSS)
        out = {}
        for iy, ky in enumerate(self.kys):
            acc = {(a, b): 0.0 for a in (0, 1) for b in (0, 1)}
            for i in range(N_GAUSS):
                for j in range(N_GAUSS):
                    k = self.kern(ky, m * self.d + s[i] - s[j])
                    for a in (0, 1):
                        for b in (0, 1):
                            acc[(a, b)] = (
                                acc[(a, b)] + w[i] * w[j] * phi(a, s[i], self.h) * phi(b, s[j], self.h) * k
                            )
            out[iy] = acc
        return out

    def z_same(self) -> dict:
        """Z_ab within a plane: delta + int_0^{2h} K(+-u) P^{+-}_ab(u) du, inner polynomial exact."""
        h = self.h
        # graded composite Gauss on [0, 2h] (the evanescent kernels decay like e^{-kappa u})
        edges = np.concatenate([[0.0], 2 * h * np.geomspace(1e-4, 1.0, 12)])
        nodes, weights = [], []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            x, w = gauss(a, b, 12)
            nodes.append(x)
            weights.append(w)
        u, wu = np.concatenate(nodes), np.concatenate(weights)

        def pplus(a: int, b: int, uu: float) -> float:  # int_{t=-h}^{h-u} phi_a(t + u) phi_b(t) dt
            t, w = gauss(-h, h - uu, 4)
            return float(np.sum(w * phi(a, t + uu, h) * phi(b, t, h)))

        def pminus(a: int, b: int, uu: float) -> float:  # int_{s=-h}^{h-u} phi_a(s) phi_b(s + u) ds
            t, w = gauss(-h, h - uu, 4)
            return float(np.sum(w * phi(a, t, h) * phi(b, t + uu, h)))

        gram = {(0, 0): 2 * h, (1, 1): 2 * h / 3, (0, 1): 0.0, (1, 0): 0.0}
        out = {}
        for iy, ky in enumerate(self.kys):
            acc = {(a, b): 0.0 for a in (0, 1) for b in (0, 1)}
            for q in range(len(u)):
                kp, km = self.kern(ky, u[q]), self.kern(ky, -u[q])
                for a in (0, 1):
                    for b in (0, 1):
                        acc[(a, b)] = acc[(a, b)] + wu[q] * (
                            pplus(a, b, u[q]) * kp + pminus(a, b, u[q]) * km
                        )
            c0 = np.stack([delta_weight(self.ref, self.om, kxv, ky) for kxv in self.kxs], axis=-1)
            for key in acc:
                acc[key] = acc[key] + gram[key] * c0
            out[iy] = acc
        return out

    def coupling(self, zt: dict, a: tuple, b: tuple) -> np.ndarray:
        """(1/d^2) sum_g L_a(kappa) L_b(-kappa) Z_ab(kappa)."""
        tot = np.zeros((9, 9), dtype=complex)
        for iy, ky in enumerate(self.kys):
            z = zt[iy][(a[0], b[0])]
            for ix, kxv in enumerate(self.kxs):
                lat = lateral(a[1], kxv, self.h) * lateral(a[2], ky, self.h)
                lat *= lateral(b[1], -kxv, self.h) * lateral(b[2], -ky, self.h)
                tot += lat * z[:, :, ix]
        return tot / self.d**2

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """Specular reflected displacement at z = 0, (S part, P part)."""
        n, nb = self.n, len(self.bas)
        ztab = {0: self.z_same()}
        for m in range(1, n):
            ztab[m] = self.z_between(m)
            ztab[-m] = self.z_between(-m)
        big = np.zeros((9 * nb * n, 9 * nb * n), dtype=complex)
        cache: dict = {}
        for i in range(n):
            for j in range(n):
                m = i - j
                for ai, a in enumerate(self.bas):
                    for bi, b in enumerate(self.bas):
                        key = (m, a, b)
                        if key not in cache:
                            cache[key] = self.coupling(ztab[m], a, b)
                        blk = -cache[key] @ self.dds[j]
                        if i == j and ai == bi:
                            blk = blk + self.d**3 * np.prod(
                                [1.0 if q == 0 else 1.0 / 3.0 for q in a]
                            ) * np.eye(9)
                        r0, c0 = 9 * (nb * i + ai), 9 * (nb * j + bi)
                        big[r0 : r0 + 9, c0 : c0 + 9] = blk
        # incident plane wave, unit displacement: P along k; SV in the (z, x) plane, normal to k; SH along y
        kw = self.om / (self.ref.alpha if self.incident == "P" else self.ref.beta)
        kin = np.array([np.sqrt(kw**2 - self.kx**2), self.kx, 0.0])
        uin = {
            "P": kin / kw,
            "SV": np.array([-self.kx, kin[0], 0.0]) / kw,
            "SH": np.array([0.0, 0.0, 1.0]),
        }[self.incident]
        psi = nine(np.outer(uin, [1, 0, 0]), kin)[:, 0]  # rows of the displacement column
        sz, wz = gauss(-self.h, self.h, N_GAUSS)
        rhs = np.zeros(9 * nb * n, dtype=complex)
        for i in range(n):
            for ai, a in enumerate(self.bas):
                iz = np.sum(wz * phi(a[0], sz, self.h) * np.exp(1j * kin[0] * sz))
                amp = (
                    np.exp(1j * kin[0] * self.zs[i])
                    * lateral(a[1], self.kx, self.h)
                    * lateral(a[2], 0.0, self.h)
                    * iz
                )
                r0 = 9 * (nb * i + ai)
                rhs[r0 : r0 + 9] = psi * amp
        sol = np.linalg.solve(big, rhs)

        def field(zo: float) -> np.ndarray:  # g = 0 scattered displacement at depth zo < 0
            tot = np.zeros(3, dtype=complex)
            for j in range(n):
                for bi, b in enumerate(self.bas):
                    kz = sum(
                        w
                        * phi(b[0], t, self.h)
                        * vertical_kernel_9x9(
                            np.array([self.kx]), 0.0, zo - self.zs[j] - t, self.om, self.ref
                        )[:, :, 0]
                        for t, w in zip(sz, wz, strict=True)
                    )
                    lat = lateral(b[1], -self.kx, self.h) * lateral(b[2], 0.0, self.h)
                    c = sol[9 * (nb * j + bi) : 9 * (nb * j + bi) + 9]
                    tot += (lat * kz @ self.dds[j] @ c)[:3] / self.d**2
            return tot

        # split upgoing P and S: U(z) = A_P e^{-i gP z} + A_S e^{-i gS z}, two depths
        gp = np.sqrt((self.om / self.ref.alpha) ** 2 - self.kx**2)
        gs = np.sqrt((self.om / self.ref.beta) ** 2 - self.kx**2)
        # the two depths a quarter beat apart: P and S differ in vertical wavenumber by only gs - gp, so
        # closely spaced depths make the 2x2 split nearly singular
        z1 = -0.5
        z2 = z1 - np.pi / (2 * abs(gs - gp)) if abs(gs - gp) > 0 else -1.3
        f1, f2 = field(z1), field(z2)
        mat = np.array(
            [[np.exp(-1j * gp * z1), np.exp(-1j * gs * z1)], [np.exp(-1j * gp * z2), np.exp(-1j * gs * z2)]]
        )
        amps = np.linalg.solve(mat, np.vstack([f1, f2]))
        return amps[1], amps[0]  # (S, P)


def main() -> int:
    """Compare every exported case.

    Returns:
        0 if every case agrees to 1e-8, else 1.
    """
    print("=" * 92)
    print("CROSS-CHECK: first-moment voxel, independent Python vs Mathematica notebooks 9 and 12")
    print("=" * 92)
    worst = 0.0
    for path in REFJSONS:
        ref = json.loads(path.read_text())
        medium = ReferenceMedium(ref["alpha"], ref["beta"], ref["rho"])
        # the stratified model (notebook 13) carries one contrast per model cell, one plane each (m = 1)
        contrast = (
            [tuple(c) for c in ref["cells_dlambda_dmu_drho"]]
            if "cells_dlambda_dmu_drho" in ref
            else ref["contrast"]
        )
        for c in ref["cases"]:
            inc = c.get("incident", "P")
            mm_s = np.array([complex(*v) for v in c["refl_S"]])
            mm_p = np.array([complex(*v) for v in c["refl_P"]])
            sch = Scheme(
                medium, contrast, ref["D"], c["omega"], c["kx"], c["n"], c["basis"], c["p_max"], inc
            )
            py_s, py_p = sch.solve()
            # each part relative to itself, or to the larger part where it vanishes (SH, SV at 0: no P)
            norm_p, norm_s = np.linalg.norm(mm_p), np.linalg.norm(mm_s)
            scale = max(norm_p, norm_s)
            dp = np.linalg.norm(py_p - mm_p) / (norm_p if norm_p >= 1e-6 * scale else scale)
            ds = np.linalg.norm(py_s - mm_s) / (norm_s if norm_s >= 1e-6 * scale else scale)
            worst = max(worst, dp, ds)
            print(
                f"  {inc:2s} theta {c['theta_deg']:2d}  n {c['n']}  basis {str(c['basis']):9s}:"
                f"  |dP| {dp:.2e}   |dS| {ds:.2e}   [{time.perf_counter() - T_START:5.0f} s]",
                flush=True,
            )
    ok = worst < 1e-8
    print(f"\n  worst disagreement {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
