"""The interface coupling of Legendre cells applied through plane-wave modes, without stored blocks.

Plans: ``docs/2026-10-15-interface-solver-scaling-plan.md`` (S5), ``docs/2026-10-10-stratified-reference-legendre-
cells-3d.md`` (Stage A).

THE FACTORISATION.  At a lateral wavenumber k the integrand of the plane-wave part D of an interface block
(``interface_spectral.integrand_points``) separates in the two cells' depths.  For a source in A at depth z_s and a
receiver at z_r,

    F(k)[a, c, i, j] = sum_{m, n} U(z_r; k)[a, i, m] R(k)[m, n] V(z_s; k)[n, c, j]
                     + sum_{alpha, beta} Ur(z_r; k)[a, alpha] C(k)[i, j, alpha, beta] Vs(z_s; k)[beta, c],

the first sum over the three down-going modes of the source (P, SV, SH) and the three up-going (reflected,
R = R_d) or down-going in B (transmitted, R = T_d) modes of the receiver, the second over the vertical moment
orders of the window's static and omega^2 parts, chi (Y0 + omega^2 Y1), which are separable in depth through the
cells' vertical moments.  Each factor carries its cell's lateral moments, and the rotation to k's direction.

So a product sums the source layers' mode amplitudes at each k (3 per k, plus the static pseudo-modes), applies R,
and reconstructs at each receiver layer: no block of a pair of depths is formed or stored.  A source in B is the
mirrored problem (z -> -z, media exchanged; ``interface_spectral._mirror``).

Conventions as ``interface_spectral``: cell centres' z in units of h (odd), z < 0 in A.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import spherical_jn

from . import interface_spectral as isp
from .effective_contrasts import ReferenceMedium
from .graded_voxel.basis import SOURCE_EXPONENTS_QUARTIC
from .graded_voxel.farfield import monomial_fourier
from .stratified_march import SOURCE_PAIRING, interface_rt, mode_matrices

_M_INV = np.diag(1.0 / SOURCE_PAIRING)


@dataclass
class ModeFactors:
    """The factors of D's integrand at P lateral wavenumbers, for sources in A, kept factored: each is a cell's
    lateral moments times a mode map times its layer's phase e^{-+i k_z z}, so only the layer-independent pieces are
    stored and the phase is applied per layer (``src_amp``, ``rec_out``).

    Attributes:
        lf, ls: (P, n_test), (P, n_source), the cells' lateral moments.
        l_of, e_of: The field functions' and source monomials' vertical exponents, as indices into the unique ones.
        v0: (P, n_e, 3, 9) per source vertical exponent e, the map from a source's coefficients onto the three
            down-going modes without the layer's phase: (diag(h F_e(-k_z h)) right M^-1 T^T) * pairing.
        kzs: (P, 3) the down-going modes' k_z in A.
        u0: {"R", "T"}: (P, n_l, 9, 3) per field vertical exponent l, the outgoing modes onto a receiver without the
            layer's phase: T states diag(h 2 i^l j_l(k_z h)).
        kzw: {"R", "T"}: (P, 3), the outgoing modes' k_z (as in the receiver's phase e^{i k_z z}).
        mid: {"R": R_d, "T": T_d}, each (P, 3, 3).
        weight: {(kind, order): (P,)} the window's weight of the static parts, -chi (times omega^2 for order 1).
        vs: {(kind, order): {z_s: (P, n_e, n_beta)}} the sources' vertical moments of the static parts.
        vr: {(kind, order): {z_r: (P, n_l, n_alpha)}} the receivers'.
        mid_stat: {(kind, order): (C, q^powers)}: the static kernel in the frame of k (times M^-1) is the polynomial
            sum_m C[i, j, alpha, beta, m] q^(m0 + m), depth-independent; the rotation to k's direction is applied to
            the vectors (``stat_apply``), not formed.
        rot: (P, 9, 9), T.
        h: The cell half-width.
    """

    lf: NDArray
    ls: NDArray
    l_of: NDArray
    e_of: NDArray
    v0: NDArray
    kzs: NDArray
    u0: dict
    kzw: dict
    mid: dict
    weight: dict
    vs: dict
    vr: dict
    mid_stat: dict
    rot: NDArray
    h: float

    def by_exponent(self, xh: NDArray) -> NDArray:
        """Y[p, e, j] = sum over the source monomials c of vertical exponent e of ls[p, c] xh[p, c, j]."""
        one_hot = np.eye(self.v0.shape[1])[self.e_of]  # (n_source, n_e)
        return np.einsum("pc,ce,pcj->pej", self.ls, one_hot, xh, optimize=True)

    def src_amp(self, zs: int, y: NDArray) -> NDArray:
        """The three down-going modes' amplitudes from a source layer's ``by_exponent`` sums, (P, 3)."""
        return np.einsum("pemj,pej->pm", self.v0, y, optimize=True) * np.exp(
            -1j * self.kzs * zs * self.h
        )

    def rec_out(self, zr: int, amp: NDArray) -> NDArray:
        """A receiver layer's coefficients from the incoming down-going amplitudes (P, 3), (P, n_test, 9)."""
        kind = "R" if zr < 0 else "T"
        b = np.einsum("pmn,pn->pm", self.mid[kind], amp) * np.exp(
            1j * self.kzw[kind] * zr * self.h
        )
        g = np.einsum("plim,pm->pli", self.u0[kind], b, optimize=True)
        return self.lf[:, :, None] * g[:, self.l_of, :]

    def stat_src_amp(self, key: tuple, zs: int, y: NDArray) -> NDArray:
        """The static pseudo-modes' amplitudes from a source layer's ``by_exponent`` sums, (P, n_beta, 9)."""
        return self.weight[key][:, None, None] * np.einsum(
            "pex,pej->pxj", self.vs[key][zs], y, optimize=True
        )

    def stat_rec_out(self, key: tuple, zr: int, w: NDArray) -> NDArray:
        """A receiver layer's coefficients from the static kernel applied to the amplitudes (``stat_apply``),
        (P, n_test, 9)."""
        g = np.einsum("plx,pix->pli", self.vr[key][zr], w, optimize=True)
        return self.lf[:, :, None] * g[:, self.l_of, :]

    def src_full(self, zs: int) -> NDArray:
        """The expanded source map, (P, 3, n_source, 9) (for checks)."""
        v = (
            self.v0[:, self.e_of]
            * np.exp(-1j * self.kzs * zs * self.h)[:, None, :, None]
        )  # (P, c, 3, 9)
        return (self.ls[:, :, None, None] * v).transpose(0, 2, 1, 3)

    def rec_full(self, zr: int) -> NDArray:
        """The expanded receiver map, (P, n_test, 9, 3) (for checks)."""
        kind = "R" if zr < 0 else "T"
        u = (
            self.u0[kind][:, self.l_of]
            * np.exp(1j * self.kzw[kind] * zr * self.h)[:, None, None, :]
        )
        return self.lf[:, :, None, None] * u

    def rotated_stat(self, key: tuple) -> NDArray:
        """The static kernel rotated to k's direction, (P, 9, 9, n_alpha, n_beta) (for checks)."""
        t = self.rot
        c, qpow = self.mid_stat[key]
        cq = np.einsum("ijabm,mp->pijab", c, qpow, optimize=True)
        out = np.einsum("pik,pklab,pjl->pijab", t, cq, t, optimize=True)
        return out * SOURCE_PAIRING[None, None, :, None, None]

    def stat_apply(self, key: tuple, amp: NDArray) -> NDArray:
        """sum_{j, beta} (T C T^T)[i, j, alpha, beta] P_j amp[beta, j], (P, 9, n_alpha), with T on the vectors."""
        t = self.rot
        v = np.einsum("pjl,pyj->pyl", t, amp * SOURCE_PAIRING, optimize=True)
        c, qpow = self.mid_stat[key]
        u = np.einsum(
            "pkxm,mp->pkx",
            np.einsum("klxym,pyl->pkxm", c, v, optimize=True),
            qpow,
            optimize=True,
        )
        return np.einsum("pik,pkx->pix", t, u, optimize=True)


def mode_factors(
    rec_zs: list[int],
    src_zs: list[int],
    kx: NDArray,
    ky: NDArray,
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
    window: isp.Window | None = None,
) -> ModeFactors:
    """The factors of D's integrand (module docstring) at lateral wavenumbers (kx, ky), q > 0.

    Args:
        rec_zs: Receiver layers' centres' z in h (odd), either side.
        src_zs: Source layers' centres' z in h, in A (z < 0).
        window: The split's window (None: no static parts).
    """
    if any(z >= 0 for z in src_zs):
        raise ValueError(
            "mode_factors: sources in A only (mirror a source in B first)."
        )
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    lu = sorted({e[0] for e in exps_f})
    eu = sorted({e[0] for e in exps_s})
    l_of = np.array([lu.index(e[0]) for e in exps_f])
    e_of = np.array([eu.index(e[0]) for e in exps_s])
    q = np.hypot(kx, ky)
    zero = np.zeros_like(q)
    T = isp.rotation(np.arctan2(ky, kx))
    lf = isp._lateral(kx, ky, h, exps_f, True)  # (P, n_test)
    ls = isp._lateral(kx, ky, h, exps_s, False)  # (P, n_source)

    st_a, _, w_a, kz_a = mode_matrices(q, zero, omega, above)
    rt = interface_rt(q, zero, omega, above, below)
    kzs = kz_a[:, :3]
    # the source modes, columns paired as the integrand, then rotated: (diag(v_e) right M^-1 T^T) * pairing
    right = (
        st_a[:, :, :3].swapaxes(1, 2) * w_a[:, :3, None] * SOURCE_PAIRING
    )  # (P, 3, 9)
    v0 = np.stack(
        [
            ((h * monomial_fourier(e, -kzs * h))[:, :, None] * right)
            @ _M_INV
            @ T.swapaxes(1, 2)
            * SOURCE_PAIRING
            for e in eu
        ],
        1,
    )  # (P, n_e, 3, 9)
    u0, kzw, sides = {}, {}, set()
    if any(z < 0 for z in rec_zs):
        sides.add("R")
        u0["R"], kzw["R"] = st_a[:, :, 3:], -kz_a[:, 3:]
    if any(z > 0 for z in rec_zs):
        sides.add("T")
        st_b, _, _, kz_b = mode_matrices(q, zero, omega, below)
        u0["T"], kzw["T"] = st_b[:, :, :3], kz_b[:, :3]
    for kind in sides:
        states, kz = u0[kind], kzw[kind]
        u0[kind] = np.stack(
            [
                T @ (states * (h * 2.0 * 1j**l * spherical_jn(l, kz * h))[:, None, :])
                for l in lu
            ],
            1,
        )  # (P, n_l, 9, 3)
    mid = {"R": rt["R_d"], "T": rt["T_d"]}

    weight, vs, vr, mid_stat = {}, {}, {}, {}
    if window is not None:
        chi = window(q * h, q0h)
        pf, ps = isp._vertical_polys(exps_f, True), isp._vertical_polys(exps_s, False)
        for kind in sorted(sides):
            s = 1 if kind == "R" else -1
            for order in (0, 1):
                C, m0 = isp.frame_laurent(kind, order, isp._key(above), isp._key(below))
                qpow = q[None, :] ** (m0 + np.arange(C.shape[4]))[:, None]
                # C M^-1, kept with the powers of q (ModeFactors.stat_apply rotates on the vectors)
                mid_stat[kind, order] = (
                    C * (1.0 / SOURCE_PAIRING)[None, :, None, None, None],
                    qpow,
                )
                weight[kind, order] = -chi * (omega**2 if order == 1 else 1.0)
                vs[kind, order] = {
                    zs: np.stack(
                        [
                            isp._vertical_moments(
                                ps[e], zs * h, h, 1, q, C.shape[3] - 1
                            ).T
                            for e in eu
                        ],
                        1,
                    )
                    for zs in src_zs
                }
                vr[kind, order] = {
                    zr: np.stack(
                        [
                            isp._vertical_moments(
                                pf[l], zr * h, h, s, q, C.shape[2] - 1
                            ).T
                            for l in lu
                        ],
                        1,
                    )
                    for zr in rec_zs
                    if (zr < 0) == (kind == "R")
                }
    return ModeFactors(
        lf, ls, l_of, e_of, v0, kzs, u0, kzw, mid, weight, vs, vr, mid_stat, T, h
    )


def integrand_from_factors(f: ModeFactors, zr: int, zs: int) -> NDArray:
    """D's integrand for receiver layer z_r and source layer z_s from the factors, (P, n_test, n_source, 81): the
    reference against ``interface_spectral.integrand_points``."""
    kind = "R" if zr < 0 else "T"
    out = np.einsum(
        "paim,pmn,pncj->pacij",
        f.rec_full(zr),
        f.mid[kind],
        f.src_full(zs),
        optimize=True,
    )
    for key in f.mid_stat:
        if key[0] == kind:
            ur = f.lf[:, :, None] * f.vr[key][zr][:, f.l_of]  # (P, n_test, n_alpha)
            us = (
                f.weight[key][:, None, None]
                * f.ls[:, None, :]
                * f.vs[key][zs][:, f.e_of].transpose(0, 2, 1)
            )
            out = out + np.einsum(
                "pax,pijxy,pyc->pacij", ur, f.rotated_stat(key), us, optimize=True
            )
    return out.reshape(out.shape[0], out.shape[1], out.shape[2], 81)


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------


class _Chunk:
    """A chunk of nodes: on the box's lattice (``idx``: m mod N flattened) or not (``t1``, ``t2``, ``shift``: the
    NUFFT's points and the centring phase)."""

    def __init__(self, qx, qy, w, h, shape, idx=None, lattice_n=0):
        self.qx, self.qy, self.w, self.idx = qx, qy, w, idx
        if idx is not None:
            import scipy.sparse

            n_sq = lattice_n * lattice_n
            # the fold: node p onto bin idx[p] (aliases share a bin)
            self.fold = scipy.sparse.csr_matrix(
                (np.ones(len(idx)), (idx, np.arange(len(idx)))), shape=(n_sq, len(idx))
            )
        else:
            nx, ny = shape
            # e^{-i k . 2h (ix, iy)}: points t = 2h k, modes ix - nx//2 (finufft's centred order)
            self.t1 = np.mod(2 * h * qx + np.pi, 2 * np.pi) - np.pi
            self.t2 = np.mod(2 * h * qy + np.pi, 2 * np.pi) - np.pi
            self.shift = np.exp(-1j * (self.t1 * (nx // 2) + self.t2 * (ny // 2)))


def _chunks(kx, ky, wts, h, shape, chunk, lattice):
    """The chunks of the nodes; ``lattice`` = (mx, my, n_grid) for the last len(mx) nodes, which lie at
    k = 2 pi m / (n_grid 2h): those chunks are applied by FFTs, not NUFFTs."""
    n_polar = len(kx) - (0 if lattice is None else len(lattice[0]))
    out = []
    for lo, hi in ((0, n_polar), (n_polar, len(kx))):
        for i0 in range(lo, hi, chunk):
            sl = slice(i0, min(i0 + chunk, hi))
            idx, n = None, 0
            if lo == n_polar and lattice is not None:
                mx, my, n = lattice
                j = slice(sl.start - n_polar, sl.stop - n_polar)
                idx = (mx[j] % n) * n + (my[j] % n)
            out.append(_Chunk(kx[sl], ky[sl], wts[sl], h, shape, idx, n))
    return out


class _Transforms:
    """Grid to nodes and nodes to grid for one product.  Off the lattice: NUFFT plans (type 2 and 1), their points
    set once per chunk.  On the lattice: e^{-2 pi i m j / N} on the cell grid, so one zero-padded N x N FFT per layer
    gives every lattice node exactly (aliases m and m + N alike), and the way back folds the nodes by m mod N and
    takes one inverse FFT per layer: no spreading and no tolerance."""

    def __init__(self, shape, n_in, n_out, eps, n_grid):
        import finufft

        self.shape, self.n_in, self.n_out, self.n = shape, n_in, n_out, n_grid
        assert n_grid is None or n_grid >= max(shape), (
            "the lattice's box holds the grid"
        )
        self.to_k = finufft.Plan(2, shape, n_trans=n_in, eps=eps, isign=-1)
        self.to_grid = finufft.Plan(1, shape, n_trans=n_out, eps=eps, isign=1)
        self.spec: dict = {}
        self.fold: dict = {}
        self.current = None

    def chunk(self, ch: _Chunk) -> None:
        self.current = ch
        if ch.idx is None:
            self.to_k.setpts(ch.t1, ch.t2)
            self.to_grid.setpts(ch.t1, ch.t2)

    def to_nodes(self, z, g: NDArray) -> NDArray:
        """(n_in, nx, ny) grid -> (P, n_in) at the current chunk's nodes."""
        ch = self.current
        if ch.idx is None:
            return (self.to_k.execute(g) * ch.shift).T
        if z not in self.spec:
            n, (nx, ny) = self.n, self.shape
            self.spec[z] = np.fft.fft2(
                np.pad(g, ((0, 0), (0, n - nx), (0, n - ny))), axes=(1, 2)
            ).reshape(self.n_in, -1)
        return self.spec[z][:, ch.idx].T

    def to_grid_add(self, z, c: NDArray, out: NDArray) -> None:
        """(P, n_out) node values (weights included) -> added onto out (nx, ny, n_out) or folded for ``finish``."""
        ch = self.current
        if ch.idx is None:
            got = self.to_grid.execute(np.ascontiguousarray((c / ch.shift[:, None]).T))
            out += got.transpose(1, 2, 0)
            return
        if z not in self.fold:
            self.fold[z] = np.zeros((self.n * self.n, self.n_out), dtype=complex)
        self.fold[z] += ch.fold @ c

    def finish(self, z, out: NDArray) -> None:
        """The folded lattice nodes onto the grid: y_j = sum_m c_m e^{2 pi i m j / N} = N^2 ifft2(fold)."""
        if z not in self.fold:
            return
        n, (nx, ny) = self.n, self.shape
        y = np.fft.ifft2(self.fold[z].reshape(n, n, -1), axes=(0, 1)) * n * n
        out += y[:nx, :ny]


class NodeProduct:
    """sum_k wts_k F(k) e^{i k . (x_r - x_s)} x_s over non-uniform nodes, sources in A: type-2 NUFFTs of the source
    layers, the mode factors, type-1 NUFFTs onto the receiver layers.  Grids (nx, ny, n_source or n_test, 9).

    The factors depend on the nodes and the frequency only, so with ``cache`` they are computed once and every
    later product (a GMRES iteration) costs the NUFFTs and the contractions alone.  ``lattice`` (``lattice_of``)
    marks the last nodes as the periodic box's lattice: those go by FFTs of the zero-padded layers instead.
    """

    def __init__(
        self,
        src_zs: list[int],
        rec_zs: list[int],
        shape: tuple[int, int],
        kx: NDArray,
        ky: NDArray,
        wts: NDArray,
        h: float,
        omega: float,
        above: ReferenceMedium,
        below: ReferenceMedium,
        n_source: int,
        n_test: int,
        q0h: float,
        window: isp.Window | None,
        eps: float = 1e-11,
        chunk: int = 4096,
        cache: bool = True,
        lattice: tuple | None = None,
    ) -> None:
        self.src_zs, self.rec_zs, self.shape = sorted(src_zs), list(rec_zs), shape
        self.n_source, self.n_test, self.eps, self.cache = n_source, n_test, eps, cache
        self._args = (h, omega, above, below, n_source, n_test, q0h, window)
        self.n_grid = None if lattice is None else lattice[2]
        self.chunks = _chunks(kx, ky, wts, h, shape, chunk, lattice)
        self.factors = [
            self._factors(ch.qx, ch.qy) if cache else None for ch in self.chunks
        ]

    def _factors(self, qx: NDArray, qy: NDArray) -> ModeFactors:
        return mode_factors(self.rec_zs, self.src_zs, qx, qy, *self._args)

    def apply(self, src_grids: dict) -> dict:
        nx, ny = self.shape
        n_source, n_test = self.n_source, self.n_test
        out = {zr: np.zeros((nx, ny, n_test * 9), dtype=complex) for zr in self.rec_zs}
        grids = {
            zs: np.ascontiguousarray(
                src_grids[zs].reshape(nx, ny, -1).transpose(2, 0, 1)
            )
            for zs in self.src_zs
            if zs in src_grids
        }
        if not grids:
            return {zr: o.reshape(nx, ny, n_test, 9) for zr, o in out.items()}
        tr = _Transforms(self.shape, n_source * 9, n_test * 9, self.eps, self.n_grid)
        for ch, f in zip(self.chunks, self.factors, strict=True):
            if f is None:
                f = self._factors(ch.qx, ch.qy)
            tr.chunk(ch)
            amp = np.zeros((len(ch.qx), 3), dtype=complex)
            amp_stat = {key: 0.0 for key in f.mid_stat}
            for zs, g in grids.items():
                y = f.by_exponent(tr.to_nodes(zs, g).reshape(-1, n_source, 9))
                amp += f.src_amp(zs, y)
                for key in f.mid_stat:
                    amp_stat[key] = amp_stat[key] + f.stat_src_amp(key, zs, y)
            stat_cache: dict = {}  # the static kernel on the amplitudes, shared by the receiver layers
            for zr in self.rec_zs:
                kind = "R" if zr < 0 else "T"
                b = f.rec_out(zr, amp)
                for key in f.mid_stat:
                    if key[0] == kind:
                        if key not in stat_cache:
                            stat_cache[key] = f.stat_apply(key, amp_stat[key])
                        b = b + f.stat_rec_out(key, zr, stat_cache[key])
                tr.to_grid_add(
                    zr, (b * ch.w[:, None, None]).reshape(len(ch.qx), -1), out[zr]
                )
        for zr in self.rec_zs:
            tr.finish(zr, out[zr])
        return {zr: o.reshape(nx, ny, n_test, 9) for zr, o in out.items()}


class TwoSidedProduct:
    """``NodeProduct`` for sources on both sides: sources in A directly, sources in B by the mirrored problem
    (z -> -z, media exchanged), with the signs of ``interface_spectral._mirror`` on the coefficients going in and
    coming out."""

    def __init__(
        self,
        src_zs,
        rec_zs,
        shape,
        kx,
        ky,
        wts,
        h,
        omega,
        above,
        below,
        n_source,
        n_test,
        q0h,
        window,
        eps=1e-11,
        chunk=4096,
        cache=True,
        lattice=None,
    ) -> None:
        exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
        exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
        self.sa = np.array([(-1.0) ** e[0] for e in exps_f])[:, None] * isp.MIRROR_9
        self.sc = np.array([(-1.0) ** e[0] for e in exps_s])[:, None] * isp.MIRROR_9
        self.rec_zs, self.shape, self.n_test = list(rec_zs), shape, n_test
        common = (shape, kx, ky, wts, h, omega)
        rest = (n_source, n_test, q0h, window, eps, chunk, cache, lattice)
        in_a = [z for z in src_zs if z < 0]
        in_b = [-z for z in src_zs if z > 0]
        self.a = (
            NodeProduct(in_a, rec_zs, *common, above, below, *rest) if in_a else None
        )
        self.b = (
            NodeProduct(in_b, [-z for z in rec_zs], *common, below, above, *rest)
            if in_b
            else None
        )

    def apply(self, x_grids: dict) -> dict:
        out = {
            zr: np.zeros(self.shape + (self.n_test, 9), dtype=complex)
            for zr in self.rec_zs
        }
        if self.a is not None:
            got = self.a.apply({z: g for z, g in x_grids.items() if z < 0})
            for zr in self.rec_zs:
                out[zr] += got[zr]
        if self.b is not None:
            got = self.b.apply({-z: g * self.sc for z, g in x_grids.items() if z > 0})
            for zr in self.rec_zs:
                out[zr] += got[-zr] * self.sa
        return out


def low_band_nodes(
    reach: float,
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    band_qh: float,
    low_pass: isp.Window,
) -> tuple[NDArray, NDArray, NDArray]:
    """The polar quadrature of the low band, (kx, ky, weights of (1/4 pi^2) d^2k times 1 - low_pass): the radial
    rule of ``truncated_blocks`` (branch points as panel ends, the band's ramp as breaks) for lateral offsets up to
    ``reach`` (in h), and the azimuthal trapezoid rule of ``interface_spectral._grid``."""
    branch = [omega * h / c for m in (above, below) for c in (m.alpha, m.beta)]
    qh, w = isp.radial_rule(
        band_qh,
        branch,
        isp._radial_width(reach),
        breaks=tuple(low_pass.ramp_edges(band_qh)),
    )
    w = w * (1.0 - low_pass(qh, band_qh))
    keep = w != 0
    _, kx, ky, wt = isp._grid(qh[keep], w[keep], h, reach)
    return kx, ky, wt


def apply_low_band(
    x_grids: dict,
    rec_zs: list[int],
    shape: tuple[int, int],
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    band_qh: float,
    low_pass: isp.Window,
    n_source: int = 10,
    n_test: int = 4,
    eps: float = 1e-11,
    chunk: int = 4096,
) -> dict:
    """D_low x for a lateral box of shape (nx, ny) cells: every source layer's cells (coefficients x_grids[z_s],
    (nx, ny, n_source, 9), either side) onto every receiver layer's (rec_zs), (nx, ny, n_test, 9).  Exact and
    non-periodic: the polar quadrature of the low band at its non-uniform nodes, by NUFFTs.  A source in B is the
    mirrored problem."""
    reach = 2.0 * float(np.hypot(shape[0] - 1, shape[1] - 1))
    kx, ky, wts = low_band_nodes(reach, h, omega, above, below, band_qh, low_pass)
    return _apply_both_sides(
        x_grids,
        rec_zs,
        shape,
        kx,
        ky,
        wts,
        h,
        omega,
        above,
        below,
        n_source,
        n_test,
        band_qh,
        None,
        eps,
        chunk,
    )


def _apply_both_sides(
    x_grids,
    rec_zs,
    shape,
    kx,
    ky,
    wts,
    h,
    omega,
    above,
    below,
    n_source,
    n_test,
    q0h,
    window,
    eps,
    chunk,
    lattice=None,
) -> dict:
    """One product, factors not kept (``TwoSidedProduct`` keeps them for repeated products)."""
    return TwoSidedProduct(
        list(x_grids),
        rec_zs,
        shape,
        kx,
        ky,
        wts,
        h,
        omega,
        above,
        below,
        n_source,
        n_test,
        q0h,
        window,
        eps,
        chunk,
        cache=False,
        lattice=lattice,
    ).apply(x_grids)


def high_band_nodes(
    n_grid: int, h: float, q0h: float, band_qh: float, low_pass: isp.Window
) -> tuple[NDArray, NDArray, NDArray]:
    """The wavenumbers of a periodic box of n_grid cells, 2 pi m / (n_grid 2h), in the high band (low_pass > 0,
    q < Q0), each alias its own node, with weights dk^2 / 4 pi^2 times low_pass: sampled there, D_high is the
    periodised kernel of ``interface_spectral.high_band_blocks``."""
    dk = 2 * np.pi / (n_grid * 2 * h)
    m = np.arange(-int(np.ceil(q0h / h / dk)), int(np.ceil(q0h / h / dk)) + 1)
    mx, my = np.meshgrid(m, m, indexing="ij")
    kx, ky = (mx * dk).ravel(), (my * dk).ravel()
    qh = np.hypot(kx, ky) * h
    w = low_pass(qh, band_qh) * dk**2 / (4 * np.pi**2)
    keep = (w > 0) & (qh < q0h)
    return kx[keep], ky[keep], w[keep]


def lattice_of(kx: NDArray, ky: NDArray, h: float, n_grid: int) -> tuple:
    """(mx, my, n_grid) of nodes on the periodic box's lattice, k = 2 pi m / (n_grid 2h): the ``lattice`` argument
    of ``NodeProduct`` and ``WholeSpaceMarch``, which then apply those nodes by FFTs."""
    dk = 2 * np.pi / (n_grid * 2 * h)
    mx, my = np.rint(kx / dk).astype(int), np.rint(ky / dk).astype(int)
    assert np.allclose(mx * dk, kx) and np.allclose(my * dk, ky)
    return mx, my, n_grid


def apply_high_band(
    x_grids: dict,
    rec_zs: list[int],
    shape: tuple[int, int],
    n_grid: int,
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    q0h: float,
    window: isp.Window,
    band_qh: float,
    low_pass: isp.Window,
    n_source: int = 10,
    n_test: int = 4,
    eps: float = 1e-11,
    chunk: int = 4096,
) -> dict:
    """D_high x for a lateral box of shape (nx, ny) cells, with the window's static parts: at the wavenumbers of a
    periodic box of n_grid cells (``high_band_nodes``), so its images lie n_grid cells apart (D_high decays faster
    than any power; n_grid >= box + the padding of ``InterfaceCouplings.split_pad``)."""
    kx, ky, wts = high_band_nodes(n_grid, h, q0h, band_qh, low_pass)
    return _apply_both_sides(
        x_grids,
        rec_zs,
        shape,
        kx,
        ky,
        wts,
        h,
        omega,
        above,
        below,
        n_source,
        n_test,
        q0h,
        window,
        eps,
        chunk,
        lattice_of(kx, ky, h, n_grid),
    )


# ---------------------------------------------------------------------------
# The whole-space coupling between planes that do not touch
# ---------------------------------------------------------------------------


@dataclass
class WholeSpaceModes:
    """The whole-space integrand's factors at P lateral wavenumbers, for one medium (gate G2's factorisation).

    Between cells in planes at least two apart, receiver at depth offset dz (in cells) from the source, the
    integrand is sum_m w_m R_a(k_m) d_m[i] S_c(k_m) (d_m M)[j] e^{i k_m . r}, the modes m (P, SV, SH) going down
    (dz > 0) or up (dz < 0), each cell's moments about its own centre.  So the factors do not depend on the layer,
    and the vertical phase is p^|dz|, p = e^{i k_z 2h} per mode.  Kept factored (the cells' moments and the mode
    states apart).

    Attributes:
        ra: {+1: down-going, -1: up-going}, (P, 3, n_test), the receiver's moments R_a(k_m).
        ws: {+1, -1}, (P, 3, n_source), w_m S_c(k_m).
        d: {+1, -1}, (P, 3, 9), the mode states.
        step: (P, 3), p.
    """

    ra: dict
    ws: dict
    d: dict
    step: NDArray

    def src_amp(self, sign: int, xh: NDArray) -> NDArray:
        """beta_m = sum_{c, j} w_m S_c (d_m M)_j xh[c, j], (P, 3)."""
        return np.einsum(
            "pmc,pmj,pcj->pm",
            self.ws[sign],
            self.d[sign] * SOURCE_PAIRING,
            xh,
            optimize=True,
        )

    def rec_out(self, sign: int, amp: NDArray) -> NDArray:
        """sum_m R_a(k_m) d_m[i] amp_m, (P, n_test, 9)."""
        return np.einsum(
            "pma,pmi,pm->pai", self.ra[sign], self.d[sign], amp, optimize=True
        )


def whole_space_modes(
    kx: NDArray,
    ky: NDArray,
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
) -> WholeSpaceModes:
    """The factors of ``WholeSpaceModes`` at lateral wavenumbers (kx, ky)."""
    from .stratified_march import mode_family, receiver_moments, source_moments_local

    ra, ws, d = {}, {}, {}
    step = None
    for sign in (1.0, -1.0):
        modes = mode_family(kx, ky, omega, ref, "P", sign) + mode_family(
            kx, ky, omega, ref, "S", sign
        )
        ra[int(sign)] = np.stack(
            [receiver_moments(k, h, n_test) for k, _, _ in modes], 1
        )
        ws[int(sign)] = np.stack(
            [w[:, None] * source_moments_local(k, h, n_source) for k, _, w in modes], 1
        )
        d[int(sign)] = np.stack([dm for _, dm, _ in modes], 1)
        if step is None:
            step = np.stack(
                [np.exp(1j * k[:, 0] * sign * 2 * h) for k, _, _ in modes], 1
            )
    return WholeSpaceModes(ra, ws, d, step)


class WholeSpaceMarch:
    """The whole-space coupling of one side's cells between planes at least two apart, by a march in depth at
    non-uniform lateral nodes: type-2 NUFFTs of each layer, the source factors, a downsweep and an upsweep of three
    mode amplitudes per node (alpha_i = p alpha_{i-1} + p^2 beta_{i-2}, and the mirror), the receiver factors,
    type-1 NUFFTs onto each layer.  |p| <= 1 in every step.  Planes fewer than ``min_gap`` apart are left to the
    stored kernels (alpha_i = p alpha_{i-1} + p^g beta_{i-g}, g = min_gap).

    Args:
        layers: The side's layer indices (cells), contiguous (an empty layer has zero coefficients).
        cache: Keep the factors between products (they depend on the nodes and the frequency only).
    """

    def __init__(
        self,
        layers: list[int],
        shape: tuple[int, int],
        kx: NDArray,
        ky: NDArray,
        wts: NDArray,
        h: float,
        omega: float,
        ref: ReferenceMedium,
        n_source: int = 10,
        n_test: int = 4,
        eps: float = 1e-11,
        chunk: int = 4096,
        cache: bool = True,
        min_gap: int = 2,
        lattice: tuple | None = None,
    ) -> None:
        self.layers = list(range(min(layers), max(layers) + 1))
        self.shape, self.n_source, self.n_test, self.eps = shape, n_source, n_test, eps
        self.min_gap = min_gap
        self._args = (h, omega, ref, n_source, n_test)
        self.n_grid = None if lattice is None else lattice[2]
        self.chunks = _chunks(kx, ky, wts, h, shape, chunk, lattice)
        self.factors = [
            whole_space_modes(ch.qx, ch.qy, *self._args) if cache else None
            for ch in self.chunks
        ]

    def apply(self, x_grids: dict) -> dict:
        nx, ny = self.shape
        n_source, n_test = self.n_source, self.n_test
        n_l = len(self.layers)
        out = {z: np.zeros((nx, ny, n_test * 9), dtype=complex) for z in self.layers}
        g = self.min_gap
        if n_l <= g:
            return {z: o.reshape(nx, ny, n_test, 9) for z, o in out.items()}
        tr = _Transforms(self.shape, n_source * 9, n_test * 9, self.eps, self.n_grid)
        grids = {
            z: np.ascontiguousarray(x_grids[z].reshape(nx, ny, -1).transpose(2, 0, 1))
            for z in self.layers
            if z in x_grids
        }
        for ch, f in zip(self.chunks, self.factors, strict=True):
            if f is None:
                f = whole_space_modes(ch.qx, ch.qy, *self._args)
            tr.chunk(ch)
            p = len(ch.qx)
            beta = {s: np.zeros((n_l, p, 3), dtype=complex) for s in (1, -1)}
            for i, z in enumerate(self.layers):
                if z in grids:
                    xh = tr.to_nodes(z, grids[z]).reshape(-1, n_source, 9)
                    for s in (1, -1):
                        beta[s][i] = f.src_amp(s, xh)
            p1, pg = f.step, f.step**g
            down = np.zeros(
                (n_l, p, 3), dtype=complex
            )  # from the planes at least two above
            up = np.zeros(
                (n_l, p, 3), dtype=complex
            )  # from the planes at least two below
            for i in range(g, n_l):
                down[i] = p1 * down[i - 1] + pg * beta[1][i - g]
            for i in range(n_l - 1 - g, -1, -1):
                up[i] = p1 * up[i + 1] + pg * beta[-1][i + g]
            for i, z in enumerate(self.layers):
                b = f.rec_out(1, down[i]) + f.rec_out(-1, up[i])
                tr.to_grid_add(z, (b * ch.w[:, None, None]).reshape(p, -1), out[z])
        for z in self.layers:
            tr.finish(z, out[z])
        return {z: o.reshape(nx, ny, n_test, 9) for z, o in out.items()}


def whole_space_nodes(
    reach: float,
    n_grid: int,
    h: float,
    omega: float,
    ref: ReferenceMedium,
    band_qh: float,
    low_pass: isp.Window,
    qmax_h: float = 16.0,
) -> tuple[NDArray, NDArray, NDArray, tuple]:
    """The whole-space march's nodes: the low band's polar quadrature (as D's, the medium's branch points) and the
    periodic box's lattice above it to q h = qmax_h.  Between planes g apart the integrand falls as
    e^{-2 (g - 1) q h}, ~1e-14 at q h = 16 / (g - 1): 16 for the march from g = 2, 8 from g = 3.  Returns
    (kx, ky, weights, lattice): the lattice nodes last, ``lattice`` their indices (``lattice_of``)."""
    lo = low_band_nodes(reach, h, omega, ref, ref, band_qh, low_pass)
    hi = high_band_nodes(n_grid, h, qmax_h, band_qh, low_pass)
    kx, ky, w = (np.concatenate([a, b]) for a, b in zip(lo, hi, strict=True))
    return kx, ky, w, lattice_of(hi[0], hi[1], h, n_grid)
