"""The layer recursion as the preconditioner of the interface solve: the two-way march (DownUp).

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6b.  The thesis's two-way march (Eqs. 5.60-5.62,
``phasescreen.f``'s ``DownUp``) for the Legendre cells near a welded interface, as the preconditioner of a
flexible GMRES.

THE SPLITTING.  The coupling, by depth plane (z points down, so "above" is the smaller layer index):
    K = K_0 + L + U + R
    K_0  a plane with itself (whole-space self plane, its reflection and corrections);
    L    from planes above: the whole-space coupling of the side, and the transmission A -> B;
    U    from planes below: the whole-space coupling of the side, and the transmission B -> A;
    R    the reflections between different planes of one side (interface blocks, same side): the backscatter.
Forward scattering (L in the down sweep, U in the up sweep) is exact within each sweep; the backscatter R is
lagged, evaluated once per sweep from the latest whole vector.

ONE PASS from x.  With A_i = M - K_0,ii E_i the plane's own system:
    down, i from the top:    x'_i  = A_i^-1 (b_i + (R x)_i + (U x)_i + sum_{j<i} L_ij E x'_j)
    up, i from the bottom:   x''_i = A_i^-1 (b_i + (R x')_i + (L x')_i + sum_{j>i} U_ij E x''_j)
(each coupling acting on E x).  The preconditioner is ``passes`` passes from x = 0; the planes' systems are
solved by an inner GMRES (the plane's lateral FFT products, block-Jacobi) to ``inner_tol``, so it varies
slightly between applications, and the outer solve is FGMRES.

This module applies the splitting with the stored kernels of ``LateralOperator(parts=True)``, O(L^2) pair
products per pass; the recursion (O(L) per pass) applies the same operator.
"""

import time

import numpy as np
import scipy.fft
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from . import stratified_apply as sap
from .effective_contrasts import MaterialContrast
from .graded_voxel.basis import source_expansion
from .interface_solver import (
    InterfaceCouplings,
    InterfaceResult,
    LateralOperator,
    cell_contrasts,
    reference_moments,
)

#: The preconditioner's own coarse nodes (``MatrixFreeDownUp(coarse=...)``): D's low band and the march at a coarse
#: low-band rule, NUFFT eps 1e-6, the march's lattice to q h = 4.  On 8 x 6 x 6 at omega h / beta = 1/2 it takes the
#: outer counts of the exact preconditioner (9 and 12 at 30 and 50 %) at ~1/4 of a product per application.
COARSE = (
    sap.LowBandRule(ramp_panels=4, panel_order=6, phi_margin=8, branch_order=12),
    1e-6,
    4.0,
)
#: ``COARSE`` with its nodes bounded independently of the box: the low band's polar rule built for a reach of at
#: most 32 h (16 cells), and the march's lattice period the box plus 8 cells.  The preconditioner's D is then
#: accurate only between nearby cells (D decays as ~h / rho); the product is unchanged, so only the outer count
#: can suffer.
COARSE_BOUNDED = COARSE + (32.0,)


def fgmres(
    matvec,
    precondition,
    b: NDArray,
    tol: float = 1e-10,
    restart: int = 50,
    max_iter: int = 1000,
) -> tuple[NDArray, int, list[float]]:
    """Flexible GMRES (Saad, Alg. 9.6), right-preconditioned and restarted, from x = 0.

    Returns the solution, the number of products with the operator and the relative residual after each.

    Raises:
        RuntimeError: when the relative residual does not reach ``tol`` within ``max_iter`` products.
    """
    b = np.asarray(b, dtype=complex)
    x = np.zeros_like(b)
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        return x, 0, []
    r, n_mv, history = b.copy(), 0, []
    while True:
        beta = np.linalg.norm(r)
        v = np.zeros((restart + 1, b.size), dtype=complex)
        z = np.zeros((restart, b.size), dtype=complex)
        hess = np.zeros((restart + 1, restart), dtype=complex)
        v[0] = r / beta
        rhs = np.zeros(restart + 1, dtype=complex)
        rhs[0] = beta
        y = np.zeros(0, dtype=complex)
        for k in range(restart):
            z[k] = precondition(v[k])
            w = matvec(z[k])
            n_mv += 1
            for _ in range(2):  # Gram-Schmidt, twice for orthogonality
                c = v[: k + 1].conj() @ w
                hess[: k + 1, k] += c
                w = w - c @ v[: k + 1]
            hess[k + 1, k] = np.linalg.norm(w)
            if hess[k + 1, k] > 0:
                v[k + 1] = w / hess[k + 1, k]
            # the small least-squares problem, solved directly (at most restart + 1 by restart)
            y, *_ = np.linalg.lstsq(hess[: k + 2, : k + 1], rhs[: k + 2], rcond=None)
            res = np.linalg.norm(rhs[: k + 2] - hess[: k + 2, : k + 1] @ y)
            history.append(res / bnorm)
            if history[-1] <= tol or n_mv >= max_iter or hess[k + 1, k] == 0:
                break
        x = x + y @ z[: len(y)]
        r = b - matvec(x)
        n_mv += 1
        if np.linalg.norm(r) <= tol * bnorm:
            return x, n_mv, history
        if n_mv >= max_iter:
            raise RuntimeError(
                f"fgmres: relative residual {np.linalg.norm(r) / bnorm:.2e} after {n_mv} products (tol {tol:g})."
            )


class DownUp:
    """The two-way march preconditioner on the stored kernels (module docstring).

    Args:
        op: ``LateralOperator(..., parts=True)``.
        e: (N, n_source, na, 9, 9), each cell's source expansion E (``source_expansion``).
        inner_tol: The planes' inner GMRES tolerance.
        passes: DownUp passes per application.
    """

    def __init__(
        self,
        op: LateralOperator,
        e: NDArray,
        inner_tol: float = 1e-3,
        passes: int = 1,
        inner_restart: int = 50,
    ) -> None:
        if not op.if_kernels:
            raise ValueError("DownUp: needs LateralOperator(..., parts=True).")
        self.op, self.e, self.inner_tol, self.passes = op, e, inner_tol, passes
        self.inner_restart = inner_restart
        self.layers = list(op.layers)
        self.cells = {z: op.by_layer[z] for z in self.layers}
        na = op.na
        self.rows = na * 9
        self.m_full = np.kron(op.m9, np.eye(9))
        # the split, by pair of planes
        self.k0, self.lo, self.up, self.refl = {}, {}, {}, {}
        for zr in self.layers:
            for zs in self.layers:
                if zr == zs:
                    self.k0[zr] = op.kernels[zr, zr]
                    continue
                forward = self.lo if zs < zr else self.up
                if (zr < 0) == (zs < 0):
                    forward[zr, zs] = op.ws_kernels[zr, zs]
                    self.refl[zr, zs] = op.if_kernels[zr, zs]
                else:
                    forward[zr, zs] = op.if_kernels[zr, zs]
        # block-Jacobi of each plane's inner solve
        self.jacobi = np.array(
            [
                np.linalg.inv(
                    self.m_full
                    - np.einsum(
                        "acij,cbjk->aibk", op.self_blocks[op.grid.idx[m, 0]], e[m]
                    ).reshape(self.rows, self.rows)
                )
                for m in range(len(e))
            ]
        )
        self.inner_products = 0
        self.inner_solves = 0

    # -- the pieces ------------------------------------------------------------------------------------------

    def _spectrum(self, z: int, x: NDArray) -> NDArray:
        """The plane's E x on the padded lateral grid, transformed: (n_source 9, F)."""
        op, cells = self.op, self.cells[z]
        src = np.einsum("ncbij,nbj->nci", self.e[cells], x).reshape(len(cells), -1)
        g = np.zeros((src.shape[1],) + op.shape, dtype=complex)
        g[:, op.pos[cells, 0], op.pos[cells, 1]] = src.T
        return np.fft.fft2(g, axes=(1, 2)).reshape(src.shape[1], -1)

    def _field(self, z: int, acc: NDArray) -> NDArray:
        """An accumulated transformed field (rows, F) at the plane's cells: (n_z, na, 9)."""
        op, cells = self.op, self.cells[z]
        y = np.fft.ifft2(acc.reshape((self.rows,) + op.shape), axes=(1, 2))
        return y[:, op.pos[cells, 0], op.pos[cells, 1]].T.reshape(len(cells), op.na, 9)

    def _apply(self, pairs: dict, spec: dict, zr: int) -> NDArray:
        """sum over the pairs (zr, zs) with zs in spec of K_{zr zs} E x_zs, transformed: (rows, F)."""
        acc = np.zeros((self.rows, self.k0[zr].shape[-1]), dtype=complex)
        for (r, s), k in pairs.items():
            if r == zr and s in spec:
                acc += np.einsum("rcf,cf->rf", k, spec[s])
        return acc

    def _solve_plane(self, z: int, rhs: NDArray) -> NDArray:
        """A_z^-1 rhs by GMRES with the plane's lateral FFT products and block-Jacobi."""
        cells, n = self.cells[z], len(self.cells[z])
        rows, m9 = self.rows, self.op.m9
        pre = self.jacobi[cells]

        def matvec(v):
            self.inner_products += 1
            x = v.reshape(n, self.op.na, 9)
            ky = self._field(
                z, np.einsum("rcf,cf->rf", self.k0[z], self._spectrum(z, x))
            )
            return (np.einsum("ab,nbi->nai", m9, x) - ky).ravel()

        def jac(v):
            return np.einsum("nij,nj->ni", pre, v.reshape(n, rows)).ravel()

        b = rhs.ravel()
        sol, _ = gmres(
            LinearOperator((b.size, b.size), matvec=matvec, dtype=complex),
            b,
            x0=jac(b),
            M=LinearOperator((b.size, b.size), matvec=jac, dtype=complex),
            rtol=self.inner_tol,
            atol=0.0,
            restart=self.inner_restart,
            maxiter=20,
        )
        self.inner_solves += 1
        return sol.reshape(n, self.op.na, 9)

    # -- the passes ------------------------------------------------------------------------------------------

    def one_pass(self, b: dict, x: dict | None) -> dict:
        """One DownUp pass from x (None: zero); b and x by plane, (n_z, na, 9)."""
        layers, n_f = self.layers, self.k0[self.layers[0]].shape[-1]

        def zeros():
            return np.zeros((self.rows, n_f), dtype=complex)

        # down: the lagged R x + U x, then the planes from the top with L of the fresh ones
        spec = {z: self._spectrum(z, x[z]) for z in layers} if x is not None else {}
        down, down_spec = {}, {}
        for z in layers:
            acc = zeros()
            if spec:
                acc += self._apply(self.refl, spec, z) + self._apply(self.up, spec, z)
            if down_spec:
                acc += self._apply(self.lo, down_spec, z)
            down[z] = self._solve_plane(z, b[z] + self._field(z, acc))
            down_spec[z] = self._spectrum(z, down[z])
        # up: the lagged R x' + L x', then the planes from the bottom with U of the fresh ones
        out, up_spec = {}, {}
        for z in reversed(layers):
            acc = self._apply(self.refl, down_spec, z) + self._apply(
                self.lo, down_spec, z
            )
            if up_spec:
                acc += self._apply(self.up, up_spec, z)
            out[z] = self._solve_plane(z, b[z] + self._field(z, acc))
            up_spec[z] = self._spectrum(z, out[z])
        return out

    def __call__(self, v: NDArray) -> NDArray:
        n = len(self.e)
        full = v.reshape(n, self.op.na, 9)
        b = {z: full[c] for z, c in self.cells.items()}
        x = None
        for _ in range(self.passes):
            x = self.one_pass(b, x)
        out = np.zeros_like(full)
        for z, c in self.cells.items():
            out[c] = x[z]
        return out.ravel()


class AveragedSweep:
    """Algorithm 5.1 of the thesis for the plane-averaged contrast, on the stored kernels.

    Each plane's cells take the plane's mean contrast, E_z = mean over its cells of E_m, so the system
    M - K E is invariant under lateral translations within each plane.  On the stored kernels' padded lateral
    grid (the circulant approximation of the box's convolutions) it then separates by lateral wavenumber: at each
    bin, a system over the L planes, M - sum_s K_hat(z_r, z_s) E_s.  Algorithm 5.1 solves that depth system
    exactly by the up/down Riccati sweep (pivots Kennett's net reflection operators); with the stored kernels a
    dense LU per bin is the same inverse, so the backscatter between planes of the averaged medium is exact and
    only the lateral fluctuations of the contrast are left to the outer iteration.  One application: an FFT per
    plane, a solve per bin, an inverse FFT per plane.  (A homogenised layer in place of the plane's mean contrast
    is the natural refinement.)

    Args:
        op: ``LateralOperator``.
        e: (N, n_source, na, 9, 9), each cell's source expansion E.
    """

    def __init__(self, op: LateralOperator, e: NDArray) -> None:
        self.op, self.e = op, e
        self.layers = list(op.layers)
        self.cells = {z: op.by_layer[z] for z in self.layers}
        na, ns = op.na, op.n_source
        rows, cols = na * 9, ns * 9
        L = len(self.layers)
        # each plane's mean contrast as a matrix (n_source 9) x (na 9): src[(c, i)] = sum ebar[c, b, i, j] x[(b, j)]
        ebar = {
            z: e[c].mean(axis=0).transpose(0, 2, 1, 3).reshape(cols, rows)
            for z, c in self.cells.items()
        }
        m_full = np.kron(op.m9, np.eye(9))
        n_f = op.kernels[self.layers[0], self.layers[0]].shape[-1]
        a = np.zeros((n_f, L * rows, L * rows), dtype=complex)
        for i, zr in enumerate(self.layers):
            a[:, i * rows : (i + 1) * rows, i * rows : (i + 1) * rows] += m_full
            for j, zs in enumerate(self.layers):
                kz = op.kernels[zr, zs]  # (rows, cols, F)
                a[:, i * rows : (i + 1) * rows, j * rows : (j + 1) * rows] -= np.einsum(
                    "rcf,cb->frb", kz, ebar[zs]
                )
        self.inv = np.linalg.inv(
            a
        )  # (F, L rows, L rows): the exact depth solve at each lateral bin
        self.rows = rows

    def __call__(self, v: NDArray) -> NDArray:
        op, rows = self.op, self.rows
        full = v.reshape(len(self.e), op.na, 9)
        spec = []
        for z in self.layers:
            cells = self.cells[z]
            g = np.zeros((rows,) + op.shape, dtype=complex)
            g[:, op.pos[cells, 0], op.pos[cells, 1]] = (
                full[cells].reshape(len(cells), rows).T
            )
            spec.append(np.fft.fft2(g, axes=(1, 2)).reshape(rows, -1))
        rhs = np.concatenate(spec, axis=0)  # (L rows, F)
        sol = np.einsum("fij,jf->if", self.inv, rhs)
        out = np.zeros_like(full)
        for i, z in enumerate(self.layers):
            cells = self.cells[z]
            y = np.fft.ifft2(
                sol[i * rows : (i + 1) * rows].reshape((rows,) + op.shape), axes=(1, 2)
            )
            out[cells] = y[:, op.pos[cells, 0], op.pos[cells, 1]].T.reshape(
                len(cells), op.na, 9
            )
        return out.ravel()


class MatrixFreeDownUp:
    """The two-way march preconditioner on the recursions: the splitting of ``DownUp`` applied by
    ``MatrixFreeOperator``'s pieces, O(L) per pass.

    - L and U, the whole-space part: the stored kernels for touching planes (within ``stored_planes``) from the
      planes already solved in the sweep, and the march beyond, alpha_i = p alpha_{i-1} + p^g beta_{i-g} (down)
      and its mirror (up), from their amplitudes.
    - L and U, the transmission: the summed mode amplitudes of the side solved first (A in the down sweep, B in
      the up sweep) onto each plane of the other, and the corrections of opposite sides.
    - R, the reflections: each side's summed amplitudes less the receiver plane's own, from the down sweep, and
      the corrections of the side's other planes.
    - K_0: the plane's own amplitudes onto itself, its stored whole-space kernel and its corrections.

    Args:
        op: ``MatrixFreeOperator`` (with ``cache=True``).
        e: (N, n_source, na, 9, 9), each cell's source expansion E.
        inner_tol: The planes' inner GMRES tolerance.
        self_reflection: "exact", a plane's own reflection (its D with itself) in its system K_0, as ``DownUp``;
            or "lagged", with the other reflections in R, so that K_0 is the plane's stored whole-space kernel and
            its corrections (the static image's near field) and the inner products are lateral FFTs alone.
        passes: DownUp passes per application (the later ones lag R x + U x of the one before, as ``DownUp``).
        bands: "both", D through its low and high bands as in the product; or "low", the low band alone (the
            branch points and the long-range tail; D_high is local), a coarser and cheaper preconditioner.
        coarse: (``stratified_apply.LowBandRule``, NUFFT eps[, qmax_h[, reach]]): D's low band at nodes of its own,
            coarser than the product's (implies ``bands="low"``); with qmax_h, the march's nodes too (that rule's low
            band and the lattice to q h = qmax_h); with reach (in h), the node bound: the low band's rule built for a
            reach of at most that, and the march's lattice period the box plus reach / 4 cells.
    """

    def __init__(
        self,
        op,
        e: NDArray,
        inner_tol: float = 1e-3,
        inner_restart: int = 50,
        self_reflection: str = "exact",
        passes: int = 1,
        bands: str = "both",
        coarse: tuple | None = None,
    ) -> None:
        self.passes = passes
        if bands not in ("both", "low"):
            raise ValueError("MatrixFreeDownUp: bands is 'both' or 'low'.")
        if self_reflection not in ("exact", "lagged"):
            raise ValueError(
                "MatrixFreeDownUp: self_reflection is 'exact' or 'lagged'."
            )
        self.self_reflection = self_reflection
        self.op, self.e, self.inner_tol, self.inner_restart = (
            op,
            e,
            inner_tol,
            inner_restart,
        )
        self.layers = list(op.layers)
        self.cells = {z: op.by_layer[z] for z in self.layers}
        nx, ny = op.shape
        self.lat = (2 * nx - 1, 2 * ny - 1)
        self.rows = op.na * 9
        self.ws = {
            z: (side, kern, march) for side, kern, march in op.sides for z in side
        }
        self.products = (op.low, op.high) if bands == "both" else (op.low,)
        if coarse is not None:
            # D's low band on its own, coarser nodes (the preconditioner need not be exact)
            rule, eps = coarse[:2]
            # the node bound: the rule's reach capped (coarse[3], in h), whatever the box
            box_reach = 2.0 * float(np.hypot(nx - 1, ny - 1))
            reach_d = min(box_reach, coarse[3]) if len(coarse) > 3 else box_reach
            g, cpl = op.grid, op.couplings
            lp, band = cpl.low_pass(op.omega), cpl.band_qh
            kx, ky, w = sap.low_band_nodes(
                reach_d,
                g.h,
                op.omega,
                g.above,
                g.below,
                band,
                lp,
                rule,
            )
            zu = [2 * z + 1 for z in self.layers]
            self.products = (
                sap.TwoSidedProduct(
                    zu,
                    zu,
                    op.shape,
                    kx,
                    ky,
                    w,
                    g.h,
                    op.omega,
                    g.above,
                    g.below,
                    op.n_source,
                    op.na,
                    band,
                    None,
                    eps,
                ),
            )
            if len(coarse) > 2:
                # and the march's nodes: the coarse low band and the lattice to q h = qmax_h
                qmax_h = coarse[2]
                if (
                    len(coarse) > 3
                ):  # the lattice's period: the box and half the capped reach beyond it
                    n_grid = scipy.fft.next_fast_len(max(nx, ny) + int(coarse[3] / 4))
                else:
                    reach = max(max(k[2], k[3]) for k in cpl.classes) // 2
                    n_grid = scipy.fft.next_fast_len(reach + 1 + cpl.split_pad)
                ws = {}
                for side, kern, march in op.sides:
                    if march is not None:
                        medium = g.above if side[0] < 0 else g.below
                        kx, ky, w, lat = sap.whole_space_nodes(
                            reach_d,
                            n_grid,
                            g.h,
                            op.omega,
                            medium,
                            band,
                            lp,
                            qmax_h=qmax_h,
                            rule=rule,
                        )
                        march = sap.WholeSpaceMarch(
                            side,
                            op.shape,
                            kx,
                            ky,
                            w,
                            g.h,
                            op.omega,
                            medium,
                            op.n_source,
                            op.na,
                            eps,
                            min_gap=march.min_gap,
                            lattice=lat,
                        )
                    for z in side:
                        ws[z] = (side, kern, march)
                self.ws = ws
        m_full = np.kron(op.m9, np.eye(9))
        self.jacobi = np.array(
            [
                np.linalg.inv(
                    m_full
                    - np.einsum(
                        "acij,cbjk->aibk", op.self_blocks[op.grid.idx[m, 0]], e[m]
                    ).reshape(self.rows, self.rows)
                )
                for m in range(len(e))
            ]
        )
        self.inner_products = 0
        self.inner_solves = 0

    # -- the pieces ------------------------------------------------------------------------------------------

    @staticmethod
    def _side(z: int) -> str:
        return "A" if z < 0 else "B"

    def _grid(self, z: int, x: NDArray) -> NDArray:
        """The plane's E x on the lateral box, (nx, ny, n_source, 9)."""
        op, cells = self.op, self.cells[z]
        g = np.zeros(op.shape + (op.n_source, 9), dtype=complex)
        g[op.pos[cells, 0], op.pos[cells, 1]] = np.einsum(
            "ncbij,nbj->nci", self.e[cells], x
        )
        return g

    def _gather(self, z: int, y: NDArray) -> NDArray:
        cells = self.cells[z]
        return y[self.op.pos[cells, 0], self.op.pos[cells, 1]]

    def _spec(self, g: NDArray) -> NDArray:
        nx, ny = self.op.shape
        return np.fft.fft2(
            np.pad(
                g.reshape(nx, ny, -1).transpose(2, 0, 1),
                ((0, 0), (0, nx - 1), (0, ny - 1)),
            ),
            axes=(1, 2),
        ).reshape(g.shape[2] * 9, -1)

    def _stored(self, zr: int, sources: dict) -> NDArray:
        """The stored whole-space kernels onto zr from the planes in ``sources`` ({z_s: spectrum}), (nx, ny, na, 9)."""
        op = self.op
        nx, ny = op.shape
        side, kernels, _ = self.ws[zr]
        acc = None
        for zs, spec in sources.items():
            dz = zr - zs
            if dz in kernels and zs in side:
                t = np.einsum("rcf,cf->rf", kernels[dz], spec)
                acc = t if acc is None else acc + t
        if acc is None:
            return 0.0
        y = np.fft.ifft2(acc.reshape((self.rows,) + self.lat), axes=(1, 2))
        return y[:, :nx, :ny].transpose(1, 2, 0).reshape(nx, ny, op.na, 9)

    def _stencils(self, zr: int, sources: dict) -> NDArray:
        """The corrections onto zr from the planes in ``sources`` ({z_s: grid}), (nx, ny, na, 9)."""
        op = self.op
        nx, ny = op.shape
        y = np.zeros((nx, ny, op.na, 9), dtype=complex)
        for zs, x in sources.items():
            for dx, dy, blk in op.stencils.get((zr, zs), ()):
                rx = slice(max(0, dx), min(nx, nx + dx))
                ry = slice(max(0, dy), min(ny, ny + dy))
                sx = slice(max(0, -dx), min(nx, nx - dx))
                sy = slice(max(0, -dy), min(ny, ny - dy))
                y[rx, ry] += np.einsum("acij,xycj->xyai", blk, x[sx, sy])
        return y

    def _d_amps(self, z: int, g: NDArray) -> list:
        return [p.amplitudes(2 * z + 1, g) for p in self.products]

    @staticmethod
    def _d_add(a: list | None, b: list, sign: float = 1.0) -> list:
        if a is None:
            return [sap.add_amplitudes(None, x, sign) for x in b]
        return [sap.add_amplitudes(x, y, sign) for x, y in zip(a, b, strict=True)]

    def _d_field(self, zr: int, amps: list, source_side: str) -> NDArray:
        return sum(
            p.fields(2 * zr + 1, a, source_side)
            for p, a in zip(self.products, amps, strict=True)
        )

    def _k0(self, z: int, g: NDArray) -> NDArray:
        y = self._stored(z, {z: self._spec(g)}) + self._stencils(z, {z: g})
        if self.self_reflection == "exact":
            y = y + self._d_field(z, self._d_amps(z, g), self._side(z))
        return y

    def _solve_plane(self, z: int, rhs: NDArray) -> NDArray:
        cells, n = self.cells[z], len(self.cells[z])
        na, rows, m9 = self.op.na, self.rows, self.op.m9
        pre = self.jacobi[cells]

        def matvec(v):
            self.inner_products += 1
            x = v.reshape(n, na, 9)
            ky = self._gather(z, self._k0(z, self._grid(z, x)))
            return (np.einsum("ab,nbi->nai", m9, x) - ky).ravel()

        def jac(v):
            return np.einsum("nij,nj->ni", pre, v.reshape(n, rows)).ravel()

        b = rhs.ravel()
        sol, _ = gmres(
            LinearOperator((b.size, b.size), matvec=matvec, dtype=complex),
            b,
            x0=jac(b),
            M=LinearOperator((b.size, b.size), matvec=jac, dtype=complex),
            rtol=self.inner_tol,
            atol=0.0,
            restart=self.inner_restart,
            maxiter=20,
        )
        self.inner_solves += 1
        return sol.reshape(n, na, 9)

    def _march_field(self, z: int, betas: dict, state: dict, sign: int) -> NDArray:
        """The march's field at z from the planes at least g away on the side the sweep came from: the recursion
        alpha_i = p alpha_{i -+ 1} + p^g beta_{i -+ g}, advanced over the side's contiguous layers ({z: amps})."""
        march = self.ws[z][2]
        if march is None:
            return 0.0
        state = state.setdefault(id(march), {})  # each side its own recursion
        g, steps = march.min_gap, march.steps()
        lay = march.layers if sign == 1 else march.layers[::-1]
        for zz in lay:
            if zz in state:
                continue
            prev, src = zz - sign, zz - sign * g
            amps = []
            for c, p in enumerate(steps):
                a = state[prev][c] * p if prev in state else 0.0
                if src in betas and src in march.layers:  # the march stays on its side
                    a = a + p**g * betas[src][c][sign]
                amps.append(np.zeros_like(p) + a)
            state[zz] = amps
            if zz == z:
                break
        return march.field(z, state[z], sign)

    # -- the passes -----------------------------------------------------------------------------------------

    def _state(self, z: int, x: NDArray, store: dict) -> None:
        """A solved plane's grid, spectrum, D amplitudes and march amplitudes, into ``store``."""
        g = self._grid(z, x)
        store["grid"][z], store["spec"][z] = g, self._spec(g)
        store["d"][z] = self._d_amps(z, g)
        tot = store.setdefault(
            "tot", {"A": None, "B": None}
        )  # each side's summed amplitudes, kept running
        tot[self._side(z)] = self._d_add(tot[self._side(z)], store["d"][z])
        march = self.ws[z][2]
        if march is not None:
            store["betas"][z] = march.betas(z, g)

    def _reflections(self, st: dict) -> dict:
        """R x for a whole vector's states: each side's reflections from its other planes (with
        ``self_reflection="lagged"``, from all its planes) and the corrections of its other planes."""
        tot = st["tot"]
        refl = {}
        for z in self.layers:
            sd = self._side(z)
            others = (
                self._d_add(tot[sd], st["d"][z], -1.0)
                if self.self_reflection == "exact"
                else tot[sd]
            )
            refl[z] = self._d_field(z, others, sd) + self._stencils(
                z,
                {
                    zs: g
                    for zs, g in st["grid"].items()
                    if zs != z and self._side(zs) == sd
                },
            )
        return refl

    def _from_below(self, z: int, st: dict, march_state: dict) -> NDArray:
        """U x at z from the planes below in ``st``: the side's stored kernels and march, and for z in A the
        transmission from B's summed amplitudes and the corrections of B's planes."""
        side = self.ws[z][0]
        y = self._stored(
            z, {zs: st["spec"][zs] for zs in st["grid"] if zs in side and zs > z}
        )
        y = y + self._march_field(z, st["betas"], march_state, -1)
        if z < 0:
            b_planes = [zs for zs in st["grid"] if zs >= 0]
            if b_planes:  # B lies wholly below an A plane: its summed amplitudes
                y = y + self._d_field(z, st["tot"]["B"], "B")
                y = y + self._stencils(z, {zs: st["grid"][zs] for zs in b_planes})
        return y

    def _from_above(self, z: int, st: dict, march_state: dict) -> NDArray:
        """L x at z from the planes above in ``st`` (the mirror of ``_from_below``)."""
        side = self.ws[z][0]
        y = self._stored(
            z, {zs: st["spec"][zs] for zs in st["grid"] if zs in side and zs < z}
        )
        y = y + self._march_field(z, st["betas"], march_state, 1)
        if z >= 0:
            a_planes = [zs for zs in st["grid"] if zs < 0]
            if a_planes:  # A lies wholly above a B plane: its summed amplitudes
                y = y + self._d_field(z, st["tot"]["A"], "A")
                y = y + self._stencils(z, {zs: st["grid"][zs] for zs in a_planes})
        return y

    def one_pass(self, b: dict, prev: dict | None) -> tuple[dict, dict]:
        """One DownUp pass from the states of x (None: zero).  Returns x'' by plane and its states."""
        layers = self.layers
        lag: dict = {}
        if prev is not None:  # the down sweep's lagged part, R x + U x
            refl, ms = self._reflections(prev), {}
            for z in reversed(layers):
                lag[z] = refl[z] + self._from_below(z, prev, ms)
        # down: L from the planes above, solved in this sweep
        dn = {"grid": {}, "spec": {}, "d": {}, "betas": {}}
        ms, l_field = {}, {}
        for z in layers:
            y = self._from_above(z, dn, ms)
            l_field[z] = y
            if z in lag:
                y = y + lag[z]
            rhs = b[z] + (self._gather(z, y) if not np.isscalar(y) else 0.0)
            self._state(z, self._solve_plane(z, rhs), dn)
        # the backscatter, lagged from the down sweep; up: U from the planes below, solved in this sweep
        refl = self._reflections(dn)
        up = {"grid": {}, "spec": {}, "d": {}, "betas": {}}
        ms, out = {}, {}
        for z in reversed(layers):
            y = l_field[z] + refl[z] + self._from_below(z, up, ms)
            out[z] = self._solve_plane(z, b[z] + self._gather(z, y))
            self._state(z, out[z], up)
        return out, up

    def __call__(self, v: NDArray) -> NDArray:
        full = v.reshape(len(self.e), self.op.na, 9)
        b = {z: full[c] for z, c in self.cells.items()}
        st = None
        for _ in range(self.passes):
            x, st = self.one_pass(b, st)
        out = np.zeros_like(full)
        for z, c in self.cells.items():
            out[c] = x[z]
        return out.ravel()


def solve_interface_fgmres(
    couplings: InterfaceCouplings,
    omega: float,
    contrasts: list[MaterialContrast | None],
    kx: float,
    ky: float,
    wave: str,
    p: int = 1,
    r: int = 0,
    operator=None,
    preconditioner: str = "downup",
    self_reflection: str = "exact",
    bands: str = "both",
    coarse: tuple | None = None,
    tol: float = 1e-10,
    inner_tol: float = 1e-3,
    passes: int = 1,
    restart: int = 50,
    max_iter: int = 1000,
) -> tuple[InterfaceResult, dict]:
    """``solve_interface`` by FGMRES with the lateral FFT products, preconditioned by the two-way march
    (``preconditioner="downup"``), by block-Jacobi (``"jacobi"``, the baseline of ``solve_interface_gmres``) or not at
    all (``"none"``, the benchmark), or by Algorithm 5.1 for the plane-averaged contrast (``"averaged"``,
    ``AveragedSweep``, stored kernels).
    With ``operator`` a ``MatrixFreeOperator`` the march is ``MatrixFreeDownUp`` (``self_reflection``, ``bands``), else
    ``DownUp`` on ``LateralOperator(parts=True)``.

    Returns the result and the counts: ``outer`` (products with the whole operator), ``inner`` (products with a
    plane's own operator), ``planes`` (in-plane solves) and ``seconds``.
    """
    grid = couplings.grid
    op = (
        LateralOperator(couplings, omega, p, r, parts=preconditioner == "downup")
        if operator is None
        else operator
    )
    na, n_field, n_source = op.na, op.n_field, op.n_source
    n = len(grid.idx)
    delta = cell_contrasts(grid, omega, contrasts, r)
    e = np.array([source_expansion(d, n_field)[:n_source, :na] for d in delta])
    rows = na * 9

    def matvec(v: NDArray) -> NDArray:
        x = v.reshape(n, na, 9)
        src = np.einsum("ncbij,nbj->nci", e, x).reshape(n, n_source * 9)
        return (np.einsum("ab,nbi->nai", op.m9, x) - op._convolve(src)).ravel()

    if preconditioner == "downup" and isinstance(op, LateralOperator):
        pre = DownUp(op, e, inner_tol=inner_tol, passes=passes)
    elif preconditioner == "downup":
        pre = MatrixFreeDownUp(
            op,
            e,
            inner_tol=inner_tol,
            self_reflection=self_reflection,
            passes=passes,
            bands=bands,
            coarse=coarse,
        )
    elif preconditioner == "jacobi":
        m_full = np.kron(op.m9, np.eye(9))
        inv = np.array(
            [
                np.linalg.inv(
                    m_full
                    - np.einsum(
                        "acij,cbjk->aibk", op.self_blocks[grid.idx[m, 0]], e[m]
                    ).reshape(rows, rows)
                )
                for m in range(n)
            ]
        )

        def pre(v):
            return np.einsum("nij,nj->ni", inv, v.reshape(n, rows)).ravel()
    elif preconditioner == "averaged":  # Algorithm 5.1 for the plane-averaged contrast
        if not isinstance(op, LateralOperator):
            raise ValueError(
                "solve_interface_fgmres: 'averaged' needs the stored kernels (LateralOperator)."
            )
        pre = AveragedSweep(op, e)
    elif preconditioner == "none":  # the benchmark: no preconditioning at all

        def pre(v):
            return v

    else:
        raise ValueError(
            f"solve_interface_fgmres: unknown preconditioner {preconditioner!r}."
        )

    ref = reference_moments(grid, omega, kx, ky, wave, n_field)
    t0 = time.time()
    sol, n_outer, history = fgmres(
        matvec, pre, ref[:, :na].ravel(), tol=tol, restart=restart, max_iter=max_iter
    )
    stats = {
        "outer": n_outer,
        "inner": getattr(pre, "inner_products", 0),
        "planes": getattr(pre, "inner_solves", 0),
        "seconds": time.time() - t0,
        "history": history,
    }
    full = np.zeros((n, n_field, 9), dtype=complex)
    full[:, :na] = sol.reshape(n, na, 9)
    return InterfaceResult(grid, omega, delta, ref, full, p, r), stats
