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
    """The factors of D's integrand at P lateral wavenumbers, for sources in A.

    Attributes:
        src: {z_s: (P, 3, n_source, 9)} the source layer's map from its cells' coefficients (at k) to the
            amplitudes of the three down-going modes.
        rec: {z_r: (P, n_test, 9, 3)} the receiver layer's map from the three outgoing modes to its cells'
            coefficients (reflected up-going in A, or transmitted down-going in B).
        mid: {"R": R_d, "T": T_d}, each (P, 3, 3).
        src_stat: {(kind, order): {z_s: (P, n_beta, n_source)}} the static pseudo-modes' source maps, with the
            window's weight -chi (and omega^2 for order 1).
        rec_stat: {(kind, order): {z_r: (P, n_test, n_alpha)}}.
        mid_stat: {(kind, order): (P, 9, 9, n_alpha, n_beta)}.
    """

    src: dict
    rec: dict
    mid: dict
    src_stat: dict
    rec_stat: dict
    mid_stat: dict


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
    lz = np.array([e[0] for e in exps_f])
    ez = np.array([e[0] for e in exps_s])
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
    src = {}
    for zs in src_zs:
        v = {
            e: h * np.exp(-1j * kzs * zs * h) * monomial_fourier(e, -kzs * h)
            for e in set(ez)
        }
        ve = {
            e: (v[e][:, :, None] * right) @ _M_INV @ T.swapaxes(1, 2) * SOURCE_PAIRING
            for e in v
        }
        src[zs] = ls[:, None, :, None] * np.stack(
            [ve[e] for e in ez], 2
        )  # (P, 3, n_source, 9)
    rec, sides = {}, {}
    if any(z < 0 for z in rec_zs):
        sides["R"] = (st_a[:, :, 3:], -kz_a[:, 3:])
    if any(z > 0 for z in rec_zs):
        st_b, _, _, kz_b = mode_matrices(q, zero, omega, below)
        sides["T"] = (st_b[:, :, :3], kz_b[:, :3])
    for zr in rec_zs:
        states, kzw = sides["R" if zr < 0 else "T"]
        u = {
            l: h * np.exp(1j * kzw * zr * h) * 2.0 * 1j**l * spherical_jn(l, kzw * h)
            for l in set(lz)
        }
        ul = {l: T @ (states * u[l][:, None, :]) for l in u}  # (P, 9, 3)
        rec[zr] = lf[:, :, None, None] * np.stack(
            [ul[l] for l in lz], 1
        )  # (P, n_test, 9, 3)
    mid = {"R": rt["R_d"], "T": rt["T_d"]}

    src_stat, rec_stat, mid_stat = {}, {}, {}
    if window is not None:
        chi = window(q * h, q0h)
        pf, ps = isp._vertical_polys(exps_f, True), isp._vertical_polys(exps_s, False)
        for kind in [k for k in ("R", "T") if k in sides]:
            s = 1 if kind == "R" else -1
            for order in (0, 1):
                C, m0 = isp.frame_laurent(kind, order, isp._key(above), isp._key(below))
                qpow = q[None, :] ** (m0 + np.arange(C.shape[4]))[:, None]
                cq = (
                    np.einsum("ijabm,mn->nijab", C, qpow)
                    * (1.0 / SOURCE_PAIRING)[None, None, :, None, None]
                )
                # rotated as the dynamic frame: T (C M^-1) T^T, columns paired
                mid_stat[kind, order] = (
                    np.einsum("pik,pklab,pjl->pijab", T, cq, T)
                    * SOURCE_PAIRING[None, None, :, None, None]
                )
                weight = -chi * (omega**2 if order == 1 else 1.0)
                src_stat[kind, order] = {
                    zs: weight[:, None, None]
                    * ls[:, None, :]
                    * np.stack(
                        [
                            isp._vertical_moments(
                                ps[e], zs * h, h, 1, q, C.shape[3] - 1
                            ).T
                            for e in ez
                        ],
                        2,
                    )
                    for zs in src_zs
                }
                rec_stat[kind, order] = {
                    zr: lf[:, :, None]
                    * np.stack(
                        [
                            isp._vertical_moments(
                                pf[l], zr * h, h, s, q, C.shape[2] - 1
                            ).T
                            for l in lz
                        ],
                        1,
                    )
                    for zr in rec_zs
                    if (zr < 0) == (kind == "R")
                }
    return ModeFactors(src, rec, mid, src_stat, rec_stat, mid_stat)


def integrand_from_factors(f: ModeFactors, zr: int, zs: int) -> NDArray:
    """D's integrand for receiver layer z_r and source layer z_s from the factors, (P, n_test, n_source, 81): the
    reference against ``interface_spectral.integrand_points``."""
    kind = "R" if zr < 0 else "T"
    out = np.einsum("paim,pmn,pncj->pacij", f.rec[zr], f.mid[kind], f.src[zs])
    for (k, order), m in f.mid_stat.items():
        if k == kind:
            out = out + np.einsum(
                "pax,pijxy,pyc->pacij",
                f.rec_stat[k, order][zr],
                m,
                f.src_stat[k, order][zs],
            )
    return out.reshape(out.shape[0], out.shape[1], out.shape[2], 81)


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------


def _apply_at_nodes(
    src_grids: dict,
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
    eps: float,
    chunk: int,
) -> dict:
    """sum_k wts_k F(k) e^{i k . (x_r - x_s)} x_s over non-uniform nodes, sources in A: type-2 NUFFTs of the source
    layers, the mode factors, type-1 NUFFTs onto the receiver layers.  Grids (nx, ny, n_source or n_test, 9)."""
    import finufft

    nx, ny = shape
    out = {zr: np.zeros((nx, ny, n_test, 9), dtype=complex) for zr in rec_zs}
    srcs = sorted(src_grids)
    for i0 in range(0, len(kx), chunk):
        qx, qy, w = kx[i0 : i0 + chunk], ky[i0 : i0 + chunk], wts[i0 : i0 + chunk]
        # e^{-i k . 2h (ix, iy)}: points t = 2h k, modes ix - nx//2 (finufft's centred order)
        t1, t2 = (
            np.mod(2 * h * qx + np.pi, 2 * np.pi) - np.pi,
            np.mod(2 * h * qy + np.pi, 2 * np.pi) - np.pi,
        )
        shift = np.exp(-1j * (t1 * (nx // 2) + t2 * (ny // 2)))
        f = mode_factors(
            rec_zs, srcs, qx, qy, h, omega, above, below, n_source, n_test, q0h, window
        )
        amp = np.zeros((len(qx), 3), dtype=complex)
        amp_stat = {key: 0.0 for key in f.mid_stat}
        for zs in srcs:
            g = np.ascontiguousarray(
                src_grids[zs].reshape(nx, ny, -1).transpose(2, 0, 1)
            )
            xh = (finufft.nufft2d2(t1, t2, g, isign=-1, eps=eps) * shift).T.reshape(
                -1, n_source, 9
            )
            amp += np.einsum("pmcj,pcj->pm", f.src[zs], xh)
            for key in f.mid_stat:
                amp_stat[key] = amp_stat[key] + np.einsum(
                    "pyc,pcj->pyj", f.src_stat[key][zs], xh
                )
        for zr in rec_zs:
            kind = "R" if zr < 0 else "T"
            b = np.einsum("paim,pmn,pn->pai", f.rec[zr], f.mid[kind], amp)
            for (k, order), m in f.mid_stat.items():
                if k == kind:
                    b = b + np.einsum(
                        "pax,pijxy,pyj->pai",
                        f.rec_stat[k, order][zr],
                        m,
                        amp_stat[k, order],
                    )
            c = np.ascontiguousarray(
                (b * (w / shift)[:, None, None]).reshape(len(qx), -1).T
            )
            got = finufft.nufft2d1(t1, t2, c, (nx, ny), isign=1, eps=eps)
            out[zr] += got.transpose(1, 2, 0).reshape(nx, ny, n_test, 9)
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
    eps: float = 1e-14,
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
) -> dict:
    """Sources in A directly; sources in B by the mirrored problem (z -> -z, media exchanged), with the signs of
    ``interface_spectral._mirror`` on the coefficients going in and coming out."""
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    sa = np.array([(-1.0) ** e[0] for e in exps_f])
    sc = np.array([(-1.0) ** e[0] for e in exps_s])
    out = {zr: np.zeros(shape + (n_test, 9), dtype=complex) for zr in rec_zs}
    in_a = {z: g for z, g in x_grids.items() if z < 0}
    in_b = {-z: g * sc[:, None] * isp.MIRROR_9 for z, g in x_grids.items() if z > 0}
    common = (shape, kx, ky, wts, h, omega)
    if in_a:
        got = _apply_at_nodes(
            in_a,
            rec_zs,
            *common,
            above,
            below,
            n_source,
            n_test,
            q0h,
            window,
            eps,
            chunk,
        )
        for zr in rec_zs:
            out[zr] += got[zr]
    if in_b:
        got = _apply_at_nodes(
            in_b,
            [-z for z in rec_zs],
            *common,
            below,
            above,
            n_source,
            n_test,
            q0h,
            window,
            eps,
            chunk,
        )
        for zr in rec_zs:
            out[zr] += got[-zr] * sa[:, None] * isp.MIRROR_9
    return out


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
    eps: float = 1e-14,
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
    )
