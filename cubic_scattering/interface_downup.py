"""The layer recursion as the preconditioner of the interface solve: the two-way march (DownUp).

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6b.  The thesis's two-way march (Eqs. 5.60-5.62,
``phasescreen.f``'s ``DownUp``) for the Legendre cells near a welded interface, as the preconditioner of a
flexible GMRES.

THE SPLITTING.  The coupling, by depth plane (z points down, so "above" is the smaller layer index):
    K = K_0 + L + U + R
    K_0  a plane with itself (whole-space self plane, its reflection and corrections);
    L    from planes above: the whole-space coupling of the side, and the transmission A -> B;
    U    from planes below: the whole-space coupling of the side, and the transmission B -> A;
    R    the reflections between different planes of one side (interface blocks, same side): the
         backscatter.
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

#: The preconditioner's own coarse nodes (``MatrixFreeDownUp(coarse=...)``): D's low band and the march at a
#: coarse low-band rule, NUFFT eps 1e-6, the march's lattice to q h = 4.  On 8 x 6 x 6 at omega h / beta =
#: 1/2 it takes the outer counts of the exact preconditioner (9 and 12 at 30 and 50 %) at ~1/4 of a product
#: per application.
COARSE = (
    sap.LowBandRule(ramp_panels=4, panel_order=6, phi_margin=8, branch_order=12),
    1e-6,
    4.0,
)
#: ``COARSE`` with its nodes bounded independently of the box: the low band's polar rule built for a reach
#: of at most 32 h (16 cells), and the march's lattice period the box plus 8 cells.  The preconditioner's D
#: is then accurate only between nearby cells (D decays as ~h / rho); the product is unchanged, so only the
#: outer count can suffer.
COARSE_BOUNDED = COARSE + (32.0,)


def fgmres(
    matvec,
    precondition,
    b: NDArray,
    tol: float = 1e-10,
    restart: int = 50,
    max_iter: int = 1000,
    x0: NDArray | None = None,
) -> tuple[NDArray, int, list[float]]:
    """Flexible GMRES (Saad, Alg. 9.6), right-preconditioned and restarted, from x = 0 or from ``x0``.

    Returns the solution, the number of products with the operator (with ``x0``, its residual's product
    included) and the relative residual after each.

    Raises:
        RuntimeError: when the relative residual does not reach ``tol`` within ``max_iter`` products.
    """
    b = np.asarray(b, dtype=complex)
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        return np.zeros_like(b), 0, []
    history: list[float] = []
    if x0 is None:
        x = np.zeros_like(b)
        r, n_mv = b.copy(), 0
    else:
        x = np.array(x0, dtype=complex).ravel()
        r, n_mv = b - matvec(x), 1
        history = [float(np.linalg.norm(r) / bnorm)]
        if history[-1] <= tol:
            return x, n_mv, history
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
            history.append(float(res / bnorm))
            if history[-1] <= tol or n_mv >= max_iter or hess[k + 1, k] == 0:
                break
        x = x + y @ z[: len(y)]
        r = b - matvec(x)
        n_mv += 1
        if np.linalg.norm(r) <= tol * bnorm:
            return x, n_mv, history
        if n_mv >= max_iter:
            raise RuntimeError(
                f"fgmres: relative residual {np.linalg.norm(r) / bnorm:.2e} after {n_mv} products "
                f"(tol {tol:g})."
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
        self.k0: dict[int, NDArray] = {}
        self.lo: dict[tuple[int, int], NDArray] = {}
        self.up: dict[tuple[int, int], NDArray] = {}
        self.refl: dict[tuple[int, int], NDArray] = {}
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
                    - np.einsum("acij,cbjk->aibk", op.self_blocks[op.grid.idx[m, 0]], e[m]).reshape(
                        self.rows, self.rows
                    )
                )
                for m in range(len(e))
            ]
        )
        self.inner_products = 0
        self.inner_solves = 0

    # -- the pieces ----------------------------------------------------------------------------------------

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
            ky = self._field(z, np.einsum("rcf,cf->rf", self.k0[z], self._spectrum(z, x)))
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

    # -- the passes ----------------------------------------------------------------------------------------

    def one_pass(self, b: dict, x: dict | None) -> dict:
        """One DownUp pass from x (None: zero); b and x by plane, (n_z, na, 9)."""
        layers, n_f = self.layers, self.k0[self.layers[0]].shape[-1]

        def zeros():
            return np.zeros((self.rows, n_f), dtype=complex)

        # down: the lagged R x + U x, then the planes from the top with L of the fresh ones
        spec = {z: self._spectrum(z, x[z]) for z in layers} if x is not None else {}
        down: dict[int, NDArray] = {}
        down_spec: dict[int, NDArray] = {}
        for z in layers:
            acc = zeros()
            if spec:
                acc += self._apply(self.refl, spec, z) + self._apply(self.up, spec, z)
            if down_spec:
                acc += self._apply(self.lo, down_spec, z)
            down[z] = self._solve_plane(z, b[z] + self._field(z, acc))
            down_spec[z] = self._spectrum(z, down[z])
        # up: the lagged R x' + L x', then the planes from the bottom with U of the fresh ones
        out: dict[int, NDArray] = {}
        up_spec: dict[int, NDArray] = {}
        for z in reversed(layers):
            acc = self._apply(self.refl, down_spec, z) + self._apply(self.lo, down_spec, z)
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
        assert x is not None  # passes >= 1
        out = np.zeros_like(full)
        for z, c in self.cells.items():
            out[c] = x[z]
        return out.ravel()


def _averaged_system(op: LateralOperator, e: NDArray) -> tuple[NDArray, dict]:
    """The plane-averaged system at every lateral bin of the stored kernels' padded grid,
    (F, L rows, L rows), and each plane's mean contrast as a matrix (n_source 9) x (na 9):
    src[(c, i)] = sum ebar[c, b, i, j] x[(b, j)]."""
    layers = list(op.layers)
    rows, cols = op.na * 9, op.n_source * 9
    ebar = {
        z: e[c].mean(axis=0).transpose(0, 2, 1, 3).reshape(cols, rows)
        for z, c in ((z, op.by_layer[z]) for z in layers)
    }
    m_full = np.kron(op.m9, np.eye(9))
    n_f = op.kernels[layers[0], layers[0]].shape[-1]
    a = np.zeros((n_f, len(layers) * rows, len(layers) * rows), dtype=complex)
    for i, zr in enumerate(layers):
        a[:, i * rows : (i + 1) * rows, i * rows : (i + 1) * rows] += m_full
        for j, zs in enumerate(layers):
            a[:, i * rows : (i + 1) * rows, j * rows : (j + 1) * rows] -= np.einsum(
                "rcf,cb->frb", op.kernels[zr, zs], ebar[zs]
            )
    return a, ebar


class AveragedSweep:
    """Algorithm 5.1 of the thesis for the plane-averaged contrast, on the stored kernels.

    Each plane's cells take the plane's mean contrast, E_z = mean over its cells of E_m, so the system
    M - K E is invariant under lateral translations within each plane.  On the stored kernels' padded
    lateral grid (the circulant approximation of the box's convolutions) it then separates by lateral
    wavenumber: at each bin, a system over the L planes, M - sum_s K_hat(z_r, z_s) E_s.  Algorithm 5.1
    solves that depth system exactly by the up/down Riccati sweep (pivots Kennett's net reflection
    operators); with the stored kernels a dense LU per bin is the same inverse, so the backscatter between
    planes of the averaged medium is exact and only the lateral fluctuations of the contrast are left to the
    outer iteration.  One application: an FFT per plane, a solve per bin, an inverse FFT per plane.  (A
    homogenised layer in place of the plane's mean contrast is the natural refinement.)

    Args:
        op: ``LateralOperator``.
        e: (N, n_source, na, 9, 9), each cell's source expansion E.
    """

    def __init__(self, op: LateralOperator, e: NDArray) -> None:
        self.op, self.e = op, e
        self.layers = list(op.layers)
        self.cells = {z: op.by_layer[z] for z in self.layers}
        a, _ = _averaged_system(op, e)
        self.inv = np.linalg.inv(a)  # (F, L rows, L rows): the exact depth solve at each lateral bin
        self.rows = op.na * 9

    def __call__(self, v: NDArray) -> NDArray:
        op, rows = self.op, self.rows
        full = v.reshape(len(self.e), op.na, 9)
        spec = []
        for z in self.layers:
            cells = self.cells[z]
            g = np.zeros((rows,) + op.shape, dtype=complex)
            g[:, op.pos[cells, 0], op.pos[cells, 1]] = full[cells].reshape(len(cells), rows).T
            spec.append(np.fft.fft2(g, axes=(1, 2)).reshape(rows, -1))
        rhs = np.concatenate(spec, axis=0)  # (L rows, F)
        sol = np.einsum("fij,jf->if", self.inv, rhs)
        out = np.zeros_like(full)
        for i, z in enumerate(self.layers):
            cells = self.cells[z]
            y = np.fft.ifft2(sol[i * rows : (i + 1) * rows].reshape((rows,) + op.shape), axes=(1, 2))
            out[cells] = y[:, op.pos[cells, 0], op.pos[cells, 1]].T.reshape(len(cells), op.na, 9)
        return out.ravel()


class DtNSweep:
    """Algorithm 5.1 in its Dirichlet-to-Neumann form with the true contrast, the pivots solved partially,
    on the stored kernels.

    The exact factorisation in depth, eliminated from the top, is A = L S U with unit block-triangular L and
    U and pivots S_z, the discrete Dirichlet-to-Neumann maps of the half-domains above each plane; with a
    laterally varying contrast S_z is dense within its plane.  Here the factors come from the plane-averaged
    medium (``AveragedSweep``), where at each lateral wavenumber (the stored kernels' padded grid) the block
    LDU of the depth system is exact: L, U and the averaged pivots S_z.  Each pivot then takes the plane's
    own contrast back, D_z = S_z + Delta A_zz, Delta A_zz = -K_zz (E - E_z) the plane's fluctuation about
    its mean on the whole padded grid (E = E_m at the cells and 0 at the grid's points outside the box), so
    the backscatter from above is the averaged medium's and the plane's own contrast, its box's edges
    included, is exact.  One application is x = U^-1 D^-1 L^-1 b: L and U at each wavenumber, and each
    D_z^-1 by an inner GMRES preconditioned with S_z^-1 to ``inner_tol`` -- a partial solve, which flexible
    GMRES accepts since the preconditioner may then vary.  With one plane it is the plane's exact inverse
    (to ``inner_tol``).

    Args:
        op: ``LateralOperator``.
        e: (N, n_source, na, 9, 9), each cell's source expansion E.
        inner_tol: The pivots' inner GMRES tolerance.
        window: The planes above each plane whose true contrast its pivot carries: 0, the plane's own alone;
            None, all the planes above (the exact pivot S_z of the true medium, solved partially); between,
            the window's planes true and those above it through the averaged medium's Dirichlet-to-Neumann
            map.
        inner_iterations: None, each pivot solved to ``inner_tol``; or exactly this many inner iterations, a
            fixed cost per pivot (0: S_z^-1 alone, so that the whole is ``AveragedSweep``).
        depth_band: None, the averaged system's coupling between all the planes; or only between planes
            at most this many apart, the rest dropped before the factorisation.  A block-banded system
            factorises with no fill outside its band, so the build is O(L b^2) blocks and an application
            O(L b), where the full one is O(L^3) and O(L^2).  ``DtNSweep`` alone (``TrueSweep`` reads the
            factors at every depth).
    """

    def __init__(
        self,
        op: LateralOperator,
        e: NDArray,
        inner_tol: float = 1e-2,
        inner_restart: int = 30,
        window: int | None = 0,
        inner_iterations: int | None = None,
        depth_band: int | None = None,
    ) -> None:
        if depth_band is not None and type(self) is not DtNSweep:
            raise ValueError(
                f"{type(self).__name__}: depth_band applies to DtNSweep alone, whose factors are read"
                " within the band; leave it None here."
            )
        if inner_iterations is not None and inner_iterations < 0:
            raise ValueError(f"DtNSweep: inner_iterations is None or a count >= 0, not {inner_iterations}.")
        self.window = window
        self.inner_iterations = inner_iterations
        self.op, self.e, self.inner_tol, self.inner_restart = (
            op,
            e,
            inner_tol,
            inner_restart,
        )
        self.layers = list(op.layers)
        self.cells = {z: op.by_layer[z] for z in self.layers}
        rows, cols = op.na * 9, op.n_source * 9
        self.rows, self.cols = rows, cols
        n_l = len(self.layers)
        # the band: planes within b of each other (all of them by default)
        self.band = n_l - 1 if depth_band is None else min(depth_band, n_l - 1)
        a, ebar = _averaged_system(op, e)
        blk = [
            [
                a[:, i * rows : (i + 1) * rows, j * rows : (j + 1) * rows].copy()
                if abs(i - j) <= self.band
                else None
                for j in range(n_l)
            ]
            for i in range(n_l)
        ]
        # the exact block LDU of the averaged depth system (within the band) at every wavenumber,
        # right-looking elimination; a block-banded system has no fill outside its band
        self.lo, self.up, self.s, self.s_inv = {}, {}, [], []
        for k in range(n_l):
            sk = blk[k][k]
            assert sk is not None  # within the band
            sk_inv = np.linalg.inv(sk)
            self.s.append(sk)
            self.s_inv.append(sk_inv)
            below = range(k + 1, min(n_l, k + self.band + 1))
            for i in below:
                bik = blk[i][k]
                assert bik is not None  # within the band
                self.lo[i, k] = bik @ sk_inv
            for j in below:
                bkj = blk[k][j]
                assert bkj is not None  # within the band
                self.up[k, j] = sk_inv @ bkj
            for i in below:
                for j in below:
                    bij, bkj = blk[i][j], blk[k][j]
                    assert bij is not None and bkj is not None  # within the band
                    bij -= self.lo[i, k] @ bkj
        # each cell's contrast fluctuation about its plane's mean, (n_source 9) x (na 9)
        # each cell's contrast, and the plane's mean, as matrices (n_source 9) x (na 9); the true contrast
        # is zero at the padded grid's points outside the box, so the fluctuation there is -E_z
        self.e_cells = {
            z: e[c].transpose(0, 1, 3, 2, 4).reshape(len(c), cols, rows) for z, c in self.cells.items()
        }
        self.ebar = ebar
        self.inner_products = 0
        self.inner_solves = 0

    def sharing(self, cls: type | None = None, **settings) -> "DtNSweep":
        """A sweep that shares this one's averaged factorisation, its build (per frequency and contrast, the
        expensive part), with settings of its own: ``window``, ``inner_tol``, ``inner_restart`` and
        ``inner_iterations``; ``cls`` ``TrueSweep`` for the sweep on the true system (default this class).
        Its counts start from zero; the factors are shared, not copied.

        Raises:
            ValueError: for a setting other than those four, ``cls`` not a ``DtNSweep``, or a ``TrueSweep``
                from a factorisation within a depth band (it reads the factors at every depth).
        """
        cls = type(self) if cls is None else cls
        if not (isinstance(cls, type) and issubclass(cls, DtNSweep)):
            raise ValueError(f"DtNSweep.sharing: cls is DtNSweep or a subclass, not {cls!r}.")
        unknown = set(settings) - {"window", "inner_tol", "inner_restart", "inner_iterations"}
        if unknown:
            raise ValueError(
                f"DtNSweep.sharing: unknown settings {sorted(unknown)}; the factorisation's own"
                " (depth_band) cannot change without a new build."
            )
        if cls is not DtNSweep and self.band < len(self.layers) - 1:
            raise ValueError(
                f"DtNSweep.sharing: {cls.__name__} needs the full factorisation, this one is within"
                f" depth_band={self.band}."
            )
        k = settings.get("inner_iterations")
        if k is not None and k < 0:
            raise ValueError(f"DtNSweep: inner_iterations is None or a count >= 0, not {k}.")
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        new.__dict__.update(settings)
        new.inner_products = 0
        new.inner_solves = 0
        return new

    def _to_grid(self, z: int, x: NDArray) -> NDArray:
        """A plane's cell vectors (n_z, k) onto the padded grid, transformed: (k, F)."""
        op, cells = self.op, self.cells[z]
        g = np.zeros((x.shape[1],) + op.shape, dtype=complex)
        g[:, op.pos[cells, 0], op.pos[cells, 1]] = x.T
        return np.fft.fft2(g, axes=(1, 2)).reshape(x.shape[1], -1)

    def _at_cells(self, z: int, xh: NDArray) -> NDArray:
        """A transformed grid vector (k, F) at the plane's cells: (n_z, k)."""
        op, cells = self.op, self.cells[z]
        y = np.fft.ifft2(xh.reshape((xh.shape[0],) + op.shape), axes=(1, 2))
        return y[:, op.pos[cells, 0], op.pos[cells, 1]].T

    def _fluctuation(self, z: int, uh: NDArray) -> NDArray:
        """The plane's contrast fluctuation on the whole padded grid acting on uh (rows, F): the sources
        (E - E_z) u, E_m at the cells and 0 elsewhere, transformed (cols, F)."""
        op, cells = self.op, self.cells[z]
        ug = np.fft.ifft2(uh.reshape((uh.shape[0],) + op.shape), axes=(1, 2))
        src = -np.einsum("cr,rxy->cxy", self.ebar[z], ug)
        src[:, op.pos[cells, 0], op.pos[cells, 1]] += np.einsum(
            "ncr,rn->cn", self.e_cells[z], ug[:, op.pos[cells, 0], op.pos[cells, 1]]
        )
        return np.fft.fft2(src, axes=(1, 2)).reshape(src.shape[0], -1)

    def _trailing(self, win: list, v: list) -> list:
        """The averaged medium's trailing system on the planes ``win`` (those above eliminated:
        its Dirichlet-to-Neumann boundary) applied at each wavenumber, L_W S_W U_W v."""
        n, band = len(win), self.band  # the window's planes are consecutive
        u = [
            v[a]
            + sum(
                np.einsum("frs,sf->rf", self.up[win[a], win[b]], v[b])
                for b in range(a + 1, min(n, a + band + 1))
            )
            for a in range(n)
        ]
        t = [np.einsum("frs,sf->rf", self.s[win[a]], u[a]) for a in range(n)]
        return [
            t[a]
            + sum(
                np.einsum("frs,sf->rf", self.lo[win[a], win[b]], t[b]) for b in range(max(0, a - band), a)
            )
            for a in range(n)
        ]

    def _trailing_solve(self, win: list, v: list) -> list:
        """(L_W S_W U_W)^-1 v at each wavenumber."""
        n, band = len(win), self.band
        p: list[NDArray] = []
        for a in range(n):
            pa = v[a].copy()
            for b in range(max(0, a - band), a):
                pa -= np.einsum("frs,sf->rf", self.lo[win[a], win[b]], p[b])
            p.append(pa)
        q = [np.einsum("frs,sf->rf", self.s_inv[win[a]], p[a]) for a in range(n)]
        r: list[NDArray | None] = [None] * n
        for a in reversed(range(n)):
            ra = q[a].copy()
            for b in range(a + 1, min(n, a + band + 1)):
                rb = r[b]
                assert rb is not None  # solved earlier in the back substitution
                ra -= np.einsum("frs,sf->rf", self.up[win[a], win[b]], rb)
            r[a] = ra
        return r

    def _window(self, i: int) -> list:
        lo_plane = 0 if self.window is None else max(0, i - self.window)
        return list(range(lo_plane, i + 1))

    def _window_solve(self, win: list, rhs: list) -> list:
        """The window's system, the averaged trailing system (the planes above the window through the
        averaged medium's Dirichlet-to-Neumann map) less the true contrast fluctuation on the window's
        planes, solved for ``rhs`` by GMRES on the padded grid, preconditioned with the window's averaged
        factorisation."""
        zs = [self.layers[a] for a in win]
        n, shape = len(win), rhs[0].shape
        size = rhs[0].size

        def split(u):
            return [u[a * size : (a + 1) * size].reshape(shape) for a in range(n)]

        def matvec(u):
            self.inner_products += 1
            v = split(u)
            out = self._trailing(win, v)
            src = [self._fluctuation(z, va) for z, va in zip(zs, v, strict=True)]
            for a in range(n):
                for b in range(n):
                    out[a] = out[a] - np.einsum("rcf,cf->rf", self.op.kernels[zs[a], zs[b]], src[b])
            return np.concatenate([o.ravel() for o in out])

        def pre(u):
            return np.concatenate([o.ravel() for o in self._trailing_solve(win, split(u))])

        b = np.concatenate([r.ravel() for r in rhs])
        self.inner_solves += 1
        k = self.inner_iterations
        if k == 0:  # the averaged factorisation's solve alone
            return split(pre(b))
        sol, _ = gmres(
            LinearOperator((b.size, b.size), matvec=matvec, dtype=complex),
            b,
            x0=pre(b),
            M=LinearOperator((b.size, b.size), matvec=pre, dtype=complex),
            # a tolerance, or exactly k iterations: one cycle of k, never stopped by a tolerance
            rtol=self.inner_tol if k is None else 0.0,
            atol=0.0,
            restart=self.inner_restart if k is None else k,
            maxiter=20 if k is None else 1,
        )
        return split(sol)

    def _pivot_solve(self, i: int, wh: NDArray) -> NDArray:
        """The pivot's inverse with the true contrast of the plane and of the ``window`` planes above it:
        the plane's component of the window's system solved for a source on the plane."""
        win = self._window(i)
        rhs = [np.zeros_like(wh) for _ in win[:-1]] + [wh]
        return self._window_solve(win, rhs)[-1]

    def __call__(self, v: NDArray) -> NDArray:
        op, n_l = self.op, len(self.layers)
        full = v.reshape(len(self.e), op.na, 9)
        bh = [self._to_grid(z, full[self.cells[z]].reshape(len(self.cells[z]), -1)) for z in self.layers]
        band = self.band
        # L^-1: down, at each wavenumber
        w: list[NDArray] = []
        for i in range(n_l):
            wi = bh[i].copy()
            for k in range(max(0, i - band), i):
                wi -= np.einsum("frs,sf->rf", self.lo[i, k], w[k])
            w.append(wi)
        # D^-1: each pivot, partially
        y = [self._pivot_solve(i, w[i]) for i in range(n_l)]
        # U^-1: up, at each wavenumber
        x: list[NDArray | None] = [None] * n_l
        for k in reversed(range(n_l)):
            xk = y[k].copy()
            for j in range(k + 1, min(n_l, k + band + 1)):
                xj = x[j]
                assert xj is not None  # solved earlier in the back substitution
                xk -= np.einsum("frs,sf->rf", self.up[k, j], xj)
            x[k] = xk
        out = np.zeros_like(full)
        for i, z in enumerate(self.layers):
            xi = x[i]
            assert xi is not None  # every plane solved
            out[self.cells[z]] = self._at_cells(z, xi).reshape(len(self.cells[z]), op.na, 9)
        return out.ravel()


class TrueSweep(DtNSweep):
    """Algorithm 5.1 for the true medium, solved approximately: the block LDU solve of the true system
    written as solves of leading systems, each approximated in a window.

    Eliminated from the top, the exact solve is, plane by plane from the bottom,
        x_z = [ (A_{<=z})^-1 (b_{<=z} - A_{<=z, >z} x_{>z}) ]_z,
    the plane's component of the leading system of planes 1 to z, the planes below (already solved) entering
    through the true coupling.  Each leading solve is approximated as in ``DtNSweep``: the plane and
    ``window`` planes above it with their true contrast, the planes above the window through the averaged
    medium's Dirichlet-to-Neumann map (their right-hand sides carried by the averaged forward substitution),
    by an inner GMRES to ``inner_tol``.  No averaged triangular factor couples solved planes to unsolved
    ones: the coupling from below is the true one, accumulated as each plane is solved (O(L^2) plane
    couplings per application).  With ``window=None`` (all the planes above) and a tight inner solve it is
    the exact inverse of the true system.
    """

    def _true_coupling(self, zr: int, zs: int, xh: NDArray) -> NDArray:
        """A_{zr zs} x for zr != zs: -K_{zr zs} E x, the true contrast at zs's cells only, transformed
        (rows, F)."""
        op, cells = self.op, self.cells[zs]
        ug = np.fft.ifft2(xh.reshape((xh.shape[0],) + op.shape), axes=(1, 2))
        src = np.zeros((self.cols,) + op.shape, dtype=complex)
        src[:, op.pos[cells, 0], op.pos[cells, 1]] = np.einsum(
            "ncr,rn->cn", self.e_cells[zs], ug[:, op.pos[cells, 0], op.pos[cells, 1]]
        )
        srch = np.fft.fft2(src, axes=(1, 2)).reshape(self.cols, -1)
        return -np.einsum("rcf,cf->rf", op.kernels[zr, zs], srch)

    def __call__(self, v: NDArray) -> NDArray:
        op, n_l = self.op, len(self.layers)
        full = v.reshape(len(self.e), op.na, 9)
        bh = [self._to_grid(z, full[self.cells[z]].reshape(len(self.cells[z]), -1)) for z in self.layers]
        acc = [np.zeros_like(b) for b in bh]  # sum over the solved planes k of A_{jk} x_k
        x: list[NDArray | None] = [None] * n_l
        for i in reversed(range(n_l)):
            r = [bh[j] - acc[j] for j in range(i + 1)]
            win = self._window(i)
            # the planes above the window, eliminated in the averaged medium: their right-hand sides forward
            f: list[NDArray] = []
            for k in range(win[0]):
                fk = r[k].copy()
                for m in range(k):
                    fk -= np.einsum("frs,sf->rf", self.lo[k, m], f[m])
                f.append(fk)
            rhs = []
            for j in win:
                rj = r[j].copy()
                for m in range(win[0]):
                    rj -= np.einsum("frs,sf->rf", self.lo[j, m], f[m])
                rhs.append(rj)
            xi = x[i] = self._window_solve(win, rhs)[-1]
            zi = self.layers[i]
            for j in range(i):
                acc[j] += self._true_coupling(self.layers[j], zi, xi)
        out = np.zeros_like(full)
        for i, z in enumerate(self.layers):
            xs = x[i]
            assert xs is not None  # every plane solved
            out[self.cells[z]] = self._at_cells(z, xs).reshape(len(self.cells[z]), op.na, 9)
        return out.ravel()


class LeadingSweep:
    """``TrueSweep`` with whole windows, on any operator's product: the leading systems solved in full.

    Eliminated from the top, the exact solve is, plane by plane from the bottom,
        x_z = [ (A_{<=z})^-1 (b_{<=z} - A_{<=z, >z} x_{>z}) ]_z,
    the plane's component of the leading system of planes 1 to z, the planes below (already solved)
    entering through the true coupling.  ``TrueSweep(window=None)`` is the same solve on the stored
    kernels, its inner GMRES preconditioned by the plane-averaged factorisation; here nothing averaged
    enters, so the class needs only the operator's product (``_convolve``, ``m9``) and runs on
    ``MatrixFreeOperator`` (route A of the matrix-free sweep: the window's planes above are the true
    ones).  Each leading system is solved by an inner GMRES to ``inner_tol``, its products restricted to
    the leading planes, preconditioned cell by cell (block-Jacobi); the coupling from the solved planes
    below is one product with those planes as sources.  With a tight inner solve it is the exact inverse.

    Args:
        op: ``LateralOperator`` or ``MatrixFreeOperator``.
        e: (N, n_source, na, 9, 9), each cell's source expansion E.
        inner_tol: The leading systems' inner GMRES tolerance.
        inner_restart: Its restart.
        inner_maxiter: Its cap on restart cycles.
    """

    def __init__(
        self,
        op,
        e: NDArray,
        inner_tol: float = 1e-2,
        inner_restart: int = 50,
        inner_maxiter: int = 20,
    ) -> None:
        self.op, self.e = op, e
        self.inner_tol, self.inner_restart, self.inner_maxiter = inner_tol, inner_restart, inner_maxiter
        self.layers = list(op.layers)
        self.cells = {z: op.by_layer[z] for z in self.layers}
        self.rows = op.na * 9
        m_full = np.kron(op.m9, np.eye(9))
        self.jacobi = np.array(
            [
                np.linalg.inv(
                    m_full
                    - np.einsum("acij,cbjk->aibk", op.self_blocks[op.grid.idx[m, 0]], e[m]).reshape(
                        self.rows, self.rows
                    )
                )
                for m in range(len(e))
            ]
        )
        self.inner_products = 0
        self.inner_solves = 0
        self.outer_products = 0

    def _coupling(self, x: NDArray, sources: NDArray) -> NDArray:
        """K E x with x nonzero only at the cells ``sources``: (N, na, 9), every cell's field."""
        op = self.op
        src = np.zeros((len(self.e), op.n_source * 9), dtype=complex)
        src[sources] = np.einsum("ncbij,nbj->nci", self.e[sources], x[sources]).reshape(
            len(sources), op.n_source * 9
        )
        return op._convolve(src)

    def _leading_solve(self, cells: NDArray, rhs: NDArray) -> NDArray:
        """A_{<=z}^-1 rhs on the leading planes' cells by GMRES, block-Jacobi preconditioned: (n, na, 9)."""
        op, n_all, n = self.op, len(self.e), len(cells)
        pre = self.jacobi[cells]

        def matvec(u):
            self.inner_products += 1
            x = np.zeros((n_all, op.na, 9), dtype=complex)
            x[cells] = u.reshape(n, op.na, 9)
            y = np.einsum("ab,nbi->nai", op.m9, x[cells]) - self._coupling(x, cells)[cells]
            return y.ravel()

        def jac(u):
            return np.einsum("nij,nj->ni", pre, u.reshape(n, self.rows)).ravel()

        b = rhs.ravel()
        sol, _ = gmres(
            LinearOperator((b.size, b.size), matvec=matvec, dtype=complex),
            b,
            x0=jac(b),
            M=LinearOperator((b.size, b.size), matvec=jac, dtype=complex),
            rtol=self.inner_tol,
            atol=0.0,
            restart=self.inner_restart,
            maxiter=self.inner_maxiter,
        )
        self.inner_solves += 1
        return sol.reshape(n, op.na, 9)

    def __call__(self, v: NDArray) -> NDArray:
        op = self.op
        full = v.reshape(len(self.e), op.na, 9)
        x = np.zeros_like(full)
        solved = np.zeros(0, dtype=int)
        for i in reversed(range(len(self.layers))):
            lead = np.concatenate([self.cells[z] for z in self.layers[: i + 1]])
            rhs = full[lead].copy()
            if solved.size:  # the true coupling from the solved planes below: one product
                self.outer_products += 1
                rhs = rhs + self._coupling(x, solved)[lead]
            y = self._leading_solve(lead, rhs)
            here = self.cells[self.layers[i]]
            x[here] = y[len(lead) - len(here) :]  # plane i's cells are the leading set's last block
            solved = np.concatenate([solved, here])
        return x.ravel()


class MatrixFreeDownUp:
    """The two-way march preconditioner on the recursions: the splitting of ``DownUp`` applied by
    ``MatrixFreeOperator``'s pieces, O(L) per pass.

    - L and U, the whole-space part: the stored kernels for touching planes (within ``stored_planes``) from
      the planes already solved in the sweep, and the march beyond, alpha_i = p alpha_{i-1} + p^g beta_{i-g}
      (down) and its mirror (up), from their amplitudes.
    - L and U, the transmission: the summed mode amplitudes of the side solved first (A in the down sweep, B
      in the up sweep) onto each plane of the other, and the corrections of opposite sides.
    - R, the reflections: each side's summed amplitudes less the receiver plane's own, from the down sweep,
      and the corrections of the side's other planes.
    - K_0: the plane's own amplitudes onto itself, its stored whole-space kernel and its corrections.

    Args:
        op: ``MatrixFreeOperator`` (with ``cache=True``).
        e: (N, n_source, na, 9, 9), each cell's source expansion E.
        inner_tol: The planes' inner GMRES tolerance.
        self_reflection: "exact", a plane's own reflection (its D with itself) in its system K_0, as
            ``DownUp``; or "lagged", with the other reflections in R, so that K_0 is the plane's stored
            whole-space kernel and its corrections (the static image's near field) and the inner products
            are lateral FFTs alone.
        passes: DownUp passes per application (the later ones lag R x + U x of the one before, as
            ``DownUp``).
        bands: "both", D through its low and high bands as in the product; or "low", the low band alone (the
            branch points and the long-range tail; D_high is local), a coarser and cheaper preconditioner.
        coarse: (``stratified_apply.LowBandRule``, NUFFT eps[, qmax_h[, reach]]): D's low band at nodes of
            its own, coarser than the product's (implies ``bands="low"``); with qmax_h, the march's nodes
            too (that rule's low band and the lattice to q h = qmax_h); with reach (in h), the node bound:
            the low band's rule built for a reach of at most that, and the march's lattice period the box
            plus reach / 4 cells.
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
            raise ValueError("MatrixFreeDownUp: self_reflection is 'exact' or 'lagged'.")
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
        self.ws = {z: (side, kern, march) for side, kern, march in op.sides for z in side}
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
                if len(coarse) > 3:  # the lattice's period: the box and half the capped reach beyond it
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
                    - np.einsum("acij,cbjk->aibk", op.self_blocks[op.grid.idx[m, 0]], e[m]).reshape(
                        self.rows, self.rows
                    )
                )
                for m in range(len(e))
            ]
        )
        self.inner_products = 0
        self.inner_solves = 0

    # -- the pieces ----------------------------------------------------------------------------------------

    @staticmethod
    def _side(z: int) -> str:
        return "A" if z < 0 else "B"

    def _grid(self, z: int, x: NDArray) -> NDArray:
        """The plane's E x on the lateral box, (nx, ny, n_source, 9)."""
        op, cells = self.op, self.cells[z]
        g = np.zeros(op.shape + (op.n_source, 9), dtype=complex)
        g[op.pos[cells, 0], op.pos[cells, 1]] = np.einsum("ncbij,nbj->nci", self.e[cells], x)
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

    def _stored(self, zr: int, sources: dict) -> NDArray | float:
        """The stored whole-space kernels onto zr from the planes in ``sources`` ({z_s: spectrum}),
        (nx, ny, na, 9)."""
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
        return sum(p.fields(2 * zr + 1, a, source_side) for p, a in zip(self.products, amps, strict=True))

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

    def _march_field(self, z: int, betas: dict, state: dict, sign: int) -> NDArray | float:
        """The march's field at z from the planes at least g away on the side the sweep came from: the
        recursion alpha_i = p alpha_{i -+ 1} + p^g beta_{i -+ g}, advanced over the side's contiguous layers
        ({z: amps})."""
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

    # -- the passes ----------------------------------------------------------------------------------------

    def _state(self, z: int, x: NDArray, store: dict) -> None:
        """A solved plane's grid, spectrum, D amplitudes and march amplitudes, into ``store``."""
        g = self._grid(z, x)
        store["grid"][z], store["spec"][z] = g, self._spec(g)
        store["d"][z] = self._d_amps(z, g)
        tot = store.setdefault("tot", {"A": None, "B": None})  # each side's summed amplitudes, kept running
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
            others = self._d_add(tot[sd], st["d"][z], -1.0) if self.self_reflection == "exact" else tot[sd]
            refl[z] = self._d_field(z, others, sd) + self._stencils(
                z,
                {zs: g for zs, g in st["grid"].items() if zs != z and self._side(zs) == sd},
            )
        return refl

    def _from_below(self, z: int, st: dict, march_state: dict) -> NDArray | float:
        """U x at z from the planes below in ``st``: the side's stored kernels and march, and for z in A the
        transmission from B's summed amplitudes and the corrections of B's planes."""
        side = self.ws[z][0]
        y = self._stored(z, {zs: st["spec"][zs] for zs in st["grid"] if zs in side and zs > z})
        y = y + self._march_field(z, st["betas"], march_state, -1)
        if z < 0:
            b_planes = [zs for zs in st["grid"] if zs >= 0]
            if b_planes:  # B lies wholly below an A plane: its summed amplitudes
                y = y + self._d_field(z, st["tot"]["B"], "B")
                y = y + self._stencils(z, {zs: st["grid"][zs] for zs in b_planes})
        return y

    def _from_above(self, z: int, st: dict, march_state: dict) -> NDArray | float:
        """L x at z from the planes above in ``st`` (the mirror of ``_from_below``)."""
        side = self.ws[z][0]
        y = self._stored(z, {zs: st["spec"][zs] for zs in st["grid"] if zs in side and zs < z})
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
            refl = self._reflections(prev)
            ms: dict[int, dict] = {}
            for z in reversed(layers):
                lag[z] = refl[z] + self._from_below(z, prev, ms)
        # down: L from the planes above, solved in this sweep
        dn: dict[str, dict] = {"grid": {}, "spec": {}, "d": {}, "betas": {}}
        ms, l_field = {}, {}
        for z in layers:
            y = self._from_above(z, dn, ms)
            l_field[z] = y
            if z in lag:
                y = y + lag[z]
            # y is the scalar 0.0 when no plane above contributes, else an array
            rhs = b[z] + (self._gather(z, y) if isinstance(y, np.ndarray) else 0.0)
            self._state(z, self._solve_plane(z, rhs), dn)
        # the backscatter, lagged from the down sweep; up: U from the planes below, solved in this sweep
        refl = self._reflections(dn)
        up: dict[str, dict] = {"grid": {}, "spec": {}, "d": {}, "betas": {}}
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


def source_expansions(op, delta: NDArray) -> NDArray:
    """Each cell's source expansion E, (N, n_source, na, 9, 9), from its contrast ``delta``
    (``cell_contrasts``) for the field sizes of ``op``: what a preconditioner is built from."""
    return np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])


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
    dtn_window: int | None = 0,
    tol: float = 1e-10,
    inner_tol: float = 1e-3,
    passes: int = 1,
    restart: int = 50,
    max_iter: int = 1000,
    precondition=None,
    x0: NDArray | None = None,
) -> tuple[InterfaceResult, dict]:
    """``solve_interface`` by FGMRES with the lateral FFT products, preconditioned by the two-way march
    (``preconditioner="downup"``), by block-Jacobi (``"jacobi"``, the baseline of ``solve_interface_gmres``)
    or not at all (``"none"``, the benchmark), or by Algorithm 5.1 for the plane-averaged contrast
    (``"averaged"``, ``AveragedSweep``, stored kernels) or in its Dirichlet-to-Neumann form with the true
    contrast, the pivots solved partially (``"dtn"``, ``DtNSweep``, stored kernels).  ``"true-sweep"``
    (``TrueSweep``, stored kernels) solves the true system's leading systems in windows; ``"leading"``
    (``LeadingSweep``, either operator, matrix-free included) solves them over all the planes above.
    With ``operator`` a ``MatrixFreeOperator`` the march is ``MatrixFreeDownUp`` (``self_reflection``,
    ``bands``), else ``DownUp`` on ``LateralOperator(parts=True)``.

    ``precondition``, a preconditioner already built (any of the above, or a callable), is used as it is in
    place of ``preconditioner``: one built at another frequency, for instance, since FGMRES accepts an
    inexact preconditioner.  ``x0`` (N, n_field, 9), a previous solution's ``psi`` (at a neighbouring
    frequency, say), starts the iteration; its residual's product is counted in ``outer``.

    Returns the result and the counts: ``outer`` (products with the whole operator), ``inner`` (products
    with a plane's own operator), ``planes`` (in-plane solves) and ``seconds``, those of this solve alone.
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
    e = source_expansions(op, delta)
    rows = na * 9

    def matvec(v: NDArray) -> NDArray:
        x = v.reshape(n, na, 9)
        src = np.einsum("ncbij,nbj->nci", e, x).reshape(n, n_source * 9)
        return (np.einsum("ab,nbi->nai", op.m9, x) - op._convolve(src)).ravel()

    if precondition is not None:
        pre = precondition
    elif preconditioner == "downup" and isinstance(op, LateralOperator):
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
                    - np.einsum("acij,cbjk->aibk", op.self_blocks[grid.idx[m, 0]], e[m]).reshape(rows, rows)
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
    elif preconditioner == "dtn":  # Algorithm 5.1, Dirichlet-to-Neumann form, pivots solved partially
        if not isinstance(op, LateralOperator):
            raise ValueError("solve_interface_fgmres: 'dtn' needs the stored kernels (LateralOperator).")
        pre = DtNSweep(op, e, inner_tol=inner_tol, window=dtn_window)
    elif (
        preconditioner == "true-sweep"
    ):  # Algorithm 5.1 for the true medium, the leading solves approximated
        if not isinstance(op, LateralOperator):
            raise ValueError(
                "solve_interface_fgmres: 'true-sweep' needs the stored kernels (LateralOperator)."
            )
        pre = TrueSweep(op, e, inner_tol=inner_tol, window=dtn_window)
    elif preconditioner == "leading":  # the true sweep with whole windows, on either operator (route A)
        pre = LeadingSweep(op, e, inner_tol=inner_tol)
    elif preconditioner == "none":  # the benchmark: no preconditioning at all

        def pre(v):
            return v

    else:
        raise ValueError(f"solve_interface_fgmres: unknown preconditioner {preconditioner!r}.")

    ref = reference_moments(grid, omega, kx, ky, wave, n_field)
    # a preconditioner used before keeps counting: this solve's counts are the increments
    inner0 = getattr(pre, "inner_products", 0)
    planes0 = getattr(pre, "inner_solves", 0)
    t0 = time.time()
    sol, n_outer, history = fgmres(
        matvec,
        pre,
        ref[:, :na].ravel(),
        tol=tol,
        restart=restart,
        max_iter=max_iter,
        x0=None if x0 is None else np.asarray(x0)[:, :na],
    )
    stats = {
        "outer": n_outer,
        "inner": getattr(pre, "inner_products", 0) - inner0,
        "planes": getattr(pre, "inner_solves", 0) - planes0,
        "seconds": time.time() - t0,
        "history": history,
    }
    full = np.zeros((n, n_field, 9), dtype=complex)
    full[:, :na] = sol.reshape(n, na, 9)
    return InterfaceResult(grid, omega, delta, ref, full, p, r), stats
