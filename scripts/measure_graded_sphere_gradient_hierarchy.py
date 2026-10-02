#!/usr/bin/env python3
"""The gradient hierarchy on a lattice in a SMOOTH medium: the graded sphere, against its exact solution.

The body is the graded sphere of the paper (radius 10 m, full contrast in the core r < a/2, falling
smoothly to zero at the surface), voxelised on a grid of n_sub cells across. There are no edges or
corners: the medium is smooth, so the field inside every voxel is smooth and the order of the hierarchy
can show.

The scheme is that of ``gradient_voxel_lattice.py`` with one addition. The contrast now varies inside a
voxel, so it is expanded there too: the profile s(x) is projected (least squares, L2) on the polynomials
of total degree <= R_C in each voxel,

    s(x_c + xi) ~ sum_V c_V xi^V ,

and the source of a voxel is that polynomial times the Taylor polynomial of the field. The moments then
carry weights up to degree q + R_C:

    block(c <- c') = sum_V c^{c'}_V B_V(x_c - x_c') ,
    B_V : the block of the uniform-contrast scheme with every weight W replaced by W + V.

For c' = c the moments are the distributional cube moments (the closed forms of the hierarchy, evaluated
here by the independent route that agrees with them); for c' != c they are Gauss integrals of a smooth
integrand. The contrast degree is R_C = q, so that the medium is represented as well as the field.

Measured: far field at the paper's nine angles, 5e8 radii away, against the exact graded sphere;
error = largest component of the difference over the largest exact component. The same measure as the
paper's Table for the Legendre voxels, whose values are printed beside it for comparison.

Predicted: orders 2, 2, 4 for q = 1, 2, 3.

Run small first:  python -u scripts/measure_graded_sphere_gradient_hierarchy.py --q=1 4
"""

import functools
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gradient_voxel_lattice as lat  # noqa: E402
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_cube_gradient_hierarchy as cube  # noqa: E402
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from cubic_scattering.sphere_scattering import mie_scattered_displacement  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import CORE, RADIUS, THETA  # noqa: E402

AXES = (0, 1, 2)
K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
R_FAR = 5.0e8 * RADIUS


#: The radial profile across the shell, as a function of x = (a - r)/(a - core) in [0, 1]:
#:   "smoothstep"  10 x^3 - 15 x^4 + 6 x^5, the paper's: twice differentiable at both ends of the shell,
#:                 a polynomial of degree five whose higher derivatives are large;
#:   "sin2"        sin^2(pi x / 2): analytic inside the shell, once differentiable at its ends, gentle.
PROFILE = "smoothstep"


def shape(x):
    """The profile as a function of x = (a - r)/(a - core), 0 at the surface and 1 at the core."""
    if PROFILE == "sin2":
        return np.sin(0.5 * np.pi * x) ** 2
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def set_profile(name: str) -> None:
    """Choose the profile for the voxel schemes AND for the exact reference, which must be the same body."""
    global PROFILE  # noqa: PLW0603
    import crosscheck_graded_sphere

    PROFILE = name
    crosscheck_graded_sphere.smoothstep = lambda x: float(shape(np.asarray(x, dtype=float)))


def smoothstep(r: np.ndarray) -> np.ndarray:
    return shape(np.clip((RADIUS - r) / (RADIUS - CORE), 0.0, 1.0))


def sorted_indices(max_len: int) -> list[tuple[int, ...]]:
    return [w for m in range(max_len + 1) for w in itertools.combinations_with_replacement(AXES, m)]


def monomial(points: np.ndarray, w: tuple[int, ...]) -> np.ndarray:
    out = np.ones(len(points))
    for ax in w:
        out = out * points[:, ax]
    return out


# ------------------------------------------------------------ the moment tables, as arrays
def coupling_array(offset, side, omega, d_list, w_list, n_gauss) -> np.ndarray:
    """T[i, n, d, w] = int_cube (d_D G_in)(offset - xi) xi^W dxi."""
    x1, w1 = leggauss(n_gauss)
    x1, w1 = 0.5 * side * x1, 0.5 * side * w1
    xi = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel()
    monos = np.stack([wts * monomial(xi, w) for w in w_list])  # (nW, G)
    arg = np.asarray(offset)[None, :] - xi
    out = np.zeros((3, 3, len(d_list), len(w_list)), dtype=complex)
    for di, ds in enumerate(d_list):
        for i in AXES:
            for n in range(i, 3):
                vals = monos @ lat.green_derivative(i, n, ds, omega, arg)
                out[i, n, di] = vals
                out[n, i, di] = vals
    return out


def self_array(side, omega, d_list, w_list) -> np.ndarray:
    """The same at zero offset: (-1)^|D| times the distributional moment of the cube."""
    hier.scalar_moment = cube.cube_scalar_moment
    out = np.zeros((3, 3, len(d_list), len(w_list)), dtype=complex)
    for di, ds in enumerate(d_list):
        for wi, w in enumerate(w_list):
            if (len(ds) + len(w)) % 2:
                continue
            for i in AXES:
                for n in range(i, 3):
                    val = (-1) ** len(ds) * hier.tensor_moment(side, omega, i, n, ds, w)
                    out[i, n, di, wi] = val
                    out[n, i, di, wi] = val
    return out


class Assembler:
    """Turns a moment table into the blocks B_V, with the index bookkeeping done once."""

    def __init__(self, q: int, r_c: int, omega: float, contrast) -> None:
        self.q, self.r_c = q, r_c
        self.u_list = sorted_indices(q)  # unknowns and equations
        self.d_list = sorted_indices(q + 1)
        self.w_list = sorted_indices(q + r_c)
        self.v_list = sorted_indices(r_c)
        self.nu = len(self.u_list)
        d_pos = {d: n for n, d in enumerate(self.d_list)}
        w_pos = {w: n for n, w in enumerate(self.w_list)}
        u_pos = {w: n for n, w in enumerate(self.u_list)}
        self.a_rho = omega**2 * contrast.Drho
        self.dc = hier.stiffness(contrast)
        # ordered tuples W collapse to sorted ones with multiplicity; coefficient = multiplicity / |W|!
        self.coef = np.array(
            [1.0 / math.prod(math.factorial(w.count(ax)) for ax in AXES) for w in self.u_list]
        )
        self.p_idx = np.array([d_pos[p] for p in self.u_list])
        self.pk_idx = np.array([[d_pos[tuple(sorted((*p, k)))] for k in AXES] for p in self.u_list])
        # weight index of W + V, for every unknown index W and contrast monomial V
        self.wv = np.array([[w_pos[tuple(sorted((*w, *v)))] for w in self.u_list] for v in self.v_list])
        # stiffness: the source strain's coefficient of xi^W is U_{j; r W};
        # entries (W, r) -> column index of (r W)
        self.strain = [
            (wi, r, u_pos[tuple(sorted((r, *w)))])
            for wi, w in enumerate(self.u_list)
            if len(w) <= q - 1
            for r in AXES
        ]

    def blocks(self, table: np.ndarray) -> np.ndarray:
        """B[V, (i, P), (j, W)] from T[i, n, d, w]."""
        nu = self.nu
        out = np.zeros((len(self.v_list), 3, nu, 3, nu), dtype=complex)
        for vi in range(len(self.v_list)):
            wmap = self.wv[vi]
            if self.a_rho != 0.0:
                dens = table[:, :, self.p_idx][:, :, :, wmap]  # (i, j, P, W)
                out[vi] += self.a_rho * np.transpose(dens, (0, 2, 1, 3)) * self.coef[None, None, None, :]
            for k in AXES:
                tk = table[:, :, self.pk_idx[:, k]]  # (i, n, P, w)
                for wi, r, col in self.strain:
                    # sum_n T[i, n, P+k, W+V] dc[n, k, r, j]
                    out[vi, :, :, :, col] += self.coef[wi] * np.einsum(
                        "inp,nj->ipj", tk[:, :, :, wmap[wi]], self.dc[:, k, r, :]
                    )
        return out.reshape(len(self.v_list), 3 * nu, 3 * nu)


# ------------------------------------------------------------ the body
def build_cells(n_sub: int, r_c: int):
    """Centres of the voxels that carry contrast, and the projected profile's monomial coefficients."""
    side = 2.0 * RADIUS / n_sub
    c1 = (np.arange(n_sub) - 0.5 * (n_sub - 1)) * side
    x1, w1 = leggauss(8)
    x1, w1 = 0.5 * side * x1, 0.5 * side * w1
    xi = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel()
    v_list = sorted_indices(r_c)
    phi = np.stack([monomial(xi, v) for v in v_list], axis=1)  # (G, nV)
    gram = phi.T @ (wts[:, None] * phi)
    centres, coefs, grid = [], [], []
    for ijk in itertools.product(range(n_sub), repeat=3):
        xc = np.array([c1[ijk[0]], c1[ijk[1]], c1[ijk[2]]])
        s = smoothstep(np.linalg.norm(xc[None, :] + xi, axis=1))
        if not np.any(s > 0.0):
            continue
        centres.append(xc)
        coefs.append(np.linalg.solve(gram, phi.T @ (wts * s)))
        grid.append(ijk)
    return side, np.array(centres), np.array(coefs), np.array(grid)


def solve(n_sub: int, q: int, omega: float, contrast, r_c: int | None = None):
    r_c = q if r_c is None else r_c
    side, centres, coefs, grid = build_cells(n_sub, r_c)
    asm = Assembler(q, r_c, omega, contrast)
    nu3 = 3 * asm.nu
    nc = len(centres)
    kp = omega / hier.REF.alpha

    @functools.cache
    def offset_blocks(key: tuple[int, int, int]) -> np.ndarray:
        if key == (0, 0, 0):
            return asm.blocks(self_array(side, omega, asm.d_list, asm.w_list))
        reach = max(abs(v) for v in key)
        n_g = 20 if reach == 1 else 12 if reach == 2 else 8
        return asm.blocks(coupling_array(side * np.array(key), side, omega, asm.d_list, asm.w_list, n_g))

    # The incident P wave travels along axis 0 and is polarised along it, so the solution is symmetric
    # under the mirrors of axes 1 and 2: u_j(M x) = sigma_j u_j(x), sigma_j = -1 for j the mirrored axis.
    # A Taylor coefficient at the mirrored centre is then sigma_j (-1)^(count of that axis in W) times the
    # one at the original. Only the voxels with x_1 > 0 and x_2 > 0 are solved for (n_sub even: no voxel
    # lies on a mirror plane), a quarter of the unknowns.
    if n_sub % 2:
        raise ValueError("the mirror reduction needs an even number of voxels across")
    tgt = {tuple(int(v) for v in g): n for n, g in enumerate(grid)}
    quad = [c for c in range(nc) if centres[c][1] > 0 and centres[c][2] > 0]
    q_pos = {c: n for n, c in enumerate(quad)}
    rep = np.zeros(nc, dtype=int)
    sign = np.ones((nc, 3, asm.nu))
    for c in range(nc):
        g = [int(v) for v in grid[c]]
        for m in (1, 2):
            if centres[c][m] < 0:
                g[m] = n_sub - 1 - g[m]
                for j in AXES:
                    for wi, w in enumerate(asm.u_list):
                        sign[c, j, wi] *= (-1.0 if j == m else 1.0) * (-1.0) ** w.count(m)
        rep[c] = q_pos[tgt[tuple(g)]]
    sign = sign.reshape(nc, nu3)
    nq = len(quad)
    big = np.eye(nu3 * nq, dtype=complex)
    rhs = np.zeros(nu3 * nq, dtype=complex)
    for qc, c in enumerate(quad):
        phase = np.exp(1j * kp * (K_HAT @ centres[c]))
        for pi, p_idx in enumerate(asm.u_list):
            der = np.prod([1j * kp * K_HAT[ax] for ax in p_idx]) if p_idx else 1.0
            for i in AXES:
                rhs[qc * nu3 + i * asm.nu + pi] = POL[i] * der * phase
    for key in {tuple(int(v) for v in grid[c] - grid[c2]) for c in quad for c2 in range(nc)}:
        b_v = offset_blocks(key)  # (nV, nu3, nu3)
        for c2 in range(nc):
            c = tgt.get(tuple(int(v) for v in grid[c2] + np.array(key)))
            if c is not None and c in q_pos:
                blk = np.tensordot(coefs[c2], b_v, axes=1) * sign[c2][None, :]
                r0, c0 = q_pos[c] * nu3, rep[c2] * nu3
                big[r0 : r0 + nu3, c0 : c0 + nu3] -= blk
    sol_q = np.linalg.solve(big, rhs).reshape(nq, nu3)
    sol = (sign * sol_q[rep]).reshape(nc, 3, asm.nu)
    return side, centres, coefs, asm, sol


def far_field(side, centres, coefs, asm: Assembler, sol, omega, contrast, obs) -> np.ndarray:
    x1, w1 = leggauss(6)
    x1, w1 = 0.5 * side * x1, 0.5 * side * w1
    xi = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel()
    mono_u = np.stack([monomial(xi, w) for w in asm.u_list])  # (nu, G)
    mono_v = np.stack([monomial(xi, v) for v in asm.v_list])  # (nV, G)
    # gradient of each field monomial: d_r (xi^W) as a combination of lower monomials
    grad_u = np.zeros((3, asm.nu, len(xi)))
    for wi, w in enumerate(asm.u_list):
        for r in set(w):
            rest = list(w)
            rest.remove(r)
            grad_u[r, wi] = w.count(r) * monomial(xi, tuple(rest))
    dc = hier.stiffness(contrast)
    out = np.zeros((len(obs), 3), dtype=complex)
    dirs = obs / np.linalg.norm(obs, axis=1)[:, None]
    dist = np.linalg.norm(obs, axis=1)
    for xc, cf, s in zip(centres, coefs, sol, strict=True):
        prof = cf @ mono_v  # (G,)
        u = np.einsum("jw,w,wg->gj", s, asm.coef, mono_u)
        grad = np.einsum("jw,w,rwg->grj", s, asm.coef, grad_u)
        force = omega**2 * contrast.Drho * prof[:, None] * u
        tau = prof[:, None, None] * np.einsum("nkrj,grj->gnk", dc, grad)
        pts = xc[None, :] + xi
        for o, rh in enumerate(dirs):
            for k_c, speed, longitudinal in (
                (omega / hier.REF.alpha, hier.REF.alpha, True),
                (omega / hier.REF.beta, hier.REF.beta, False),
            ):
                phase = np.exp(-1j * k_c * (pts @ rh)) * wts
                amp = phase @ force + 1j * k_c * np.einsum("g,gnk,k->n", phase, tau, rh)
                amp = rh * (rh @ amp) if longitudinal else amp - rh * (rh @ amp)
                out[o] += (
                    np.exp(1j * k_c * dist[o]) / (4.0 * math.pi * hier.REF.rho * speed**2 * dist[o]) * amp
                )
    return out


def main() -> int:
    global CORE  # noqa: PLW0603
    ka_s, qs, ladder, legendre = 0.5, [1, 2, 3], [], False
    for a in sys.argv[1:]:
        if a.startswith("--ka="):
            ka_s = float(a.split("=", 1)[1])
        elif a.startswith("--core="):
            # radius of the homogeneous core in metres: a smaller core is a wider graded shell, which a
            # given grid resolves with more voxels
            CORE = float(a.split("=", 1)[1])
        elif a == "--legendre":
            legendre = True
        elif a.startswith("--profile="):
            set_profile(a.split("=", 1)[1])
        elif a.startswith("--q="):
            qs = [int(v) for v in a.split("=", 1)[1].split(",")]
        else:
            ladder.append(int(a))
    contrast = hier.CONTRAST
    omega = ka_s * hier.REF.beta / RADIUS
    obs = obs_points(R_FAR, THETA)
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    exact = mie_scattered_displacement(
        graded_mie_result(omega, RADIUS, CORE, hier.REF, contrast, n_max), obs
    )
    peak = float(np.max(np.abs(exact)))
    shell = RADIUS - CORE
    print(
        f"graded sphere, k_S a = {ka_s}, core {CORE} m, shell {shell} m wide, profile {PROFILE}",
        flush=True,
    )
    if legendre:
        # the Legendre (Galerkin) voxels on the same body and grids, for comparison
        from cubic_scattering.graded_voxel.farfield import graded_far_field
        from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft

        def prof(pos) -> float:
            return float(smoothstep(np.array([np.linalg.norm(pos)]))[0])

        for p in qs:
            rows = []
            for n in ladder or [4]:
                t0 = time.perf_counter()
                res = solve_graded_sphere_fft(
                    omega, RADIUS, hier.REF, contrast, n, prof, K_HAT, POL, "P", p=p, r=p
                )
                u_p, u_s = graded_far_field(res, obs / R_FAR, R_FAR, K_HAT, POL, "P")
                err = float(np.max(np.abs(u_p + u_s - exact)) / peak)
                rows.append((n, err))
                unknowns = len(res.centres) * 9 * (1, 4, 10)[p]
                print(
                    f"  Legendre degree {p}  n_sub {n:2d}  voxels {len(res.centres):4d}  "
                    f"unknowns {unknowns:6d}  error {err:.3e}   {time.perf_counter() - t0:7.1f} s   "
                    f"({shell / (2 * RADIUS / n):.1f} voxels "
                    "across the shell)",
                    flush=True,
                )
            for (n1, e1), (n2, e2) in zip(rows, rows[1:], strict=False):
                print(
                    f"  Legendre degree {p}  apparent order {n1} -> {n2}: "
                    f"{math.log(e1 / e2) / math.log(n2 / n1):.2f}"
                )
        return 0
    for q in qs:
        rows = []
        for n in ladder or [4]:
            t0 = time.perf_counter()
            side, centres, coefs, asm, sol = solve(n, q, omega, contrast)
            got = far_field(side, centres, coefs, asm, sol, omega, contrast, obs)
            err = float(np.max(np.abs(got - exact)) / peak)
            rows.append((n, err))
            print(
                f"  q = {q}  n_sub {n:2d}  voxels {len(centres):4d}  "
                f"unknowns {3 * asm.nu * len(centres):6d}  "
                f"error {err:.3e}   {time.perf_counter() - t0:7.1f} s",
                flush=True,
            )
        for (n1, e1), (n2, e2) in zip(rows, rows[1:], strict=False):
            print(f"  q = {q}  apparent order {n1} -> {n2}: {math.log(e1 / e2) / math.log(n2 / n1):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
