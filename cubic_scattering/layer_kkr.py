"""Layer-KKR for elastic spheres: vector multipoles and their lattice coupling.

THE BASIS.  For order (n, m), radial function z_n = j_n (regular) or h_n^(1)
(outgoing), and Y_nm orthonormal with the Condon-Shortley phase:

    L_nm = grad(z_n(k_P r) Y_nm)          (P)
    M_nm = curl(r z_n(k_S r) Y_nm)        (SH, toroidal)
    N_nm = curl M_nm                      (SV, poloidal)

These are the potentials of ``sphere_scattering`` (u = grad phi + curl curl(r psi)
+ curl(r chi)), so the single-site T-matrix in this basis is diagonal in
(n, m), independent of m, and given by ``mie_tmatrix_psv`` on (L, N) and
``mie_tmatrix_sh`` on M.

THE COUPLING.  Outgoing multipoles at every lattice site but the origin,
re-expanded as regular multipoles at the origin:

    sum_{R != 0} e^{i k.R} F^h_nm(r - R) = sum G0[(F' nu mu), (F n m)] F'^j_numu(r).

  * L -> L is the scalar translation, the Gaunt contraction of the structure
    constants D[q, s] of ``planar_kambe``:
        C[(nu mu), (n m)] = 4 pi (-1)^m sum_q i^(nu+q-n) (-1)^q D[q, m-mu]
                            G(n, m; nu, -mu; q, mu-m) ,
    because the gradient commutes with the lattice sum.
  * M: M^h_nm = -i L psi_nm, with L = -i s x grad the angular momentum about
    the source's OWN centre; L maps psi_nm into psi_{n,m-1..m+1} times fixed
    Cartesian vectors, so the lattice sum of M is scalar lattice sums.
  * N = curl M, so its lattice sum is the curl of M's.
  M and N lattice fields are then read off as regular M, N by projection on a
  sphere about the origin: the radial component carries N alone, the
  r^ x grad_s Y component M alone.

Every identity used here is checked numerically in
``tests/test_layer_kkr_coupling.py``: one translation against direct
evaluation, and the damped lattice against the plain direct sum.
"""

from __future__ import annotations

from math import factorial
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm_y, spherical_jn

from .effective_contrasts import MaterialContrast, ReferenceMedium
from .sphere_scattering import mie_tmatrix_psv, mie_tmatrix_sh

#: Orders beyond N_max kept when the coupling is formed and projected.  The
#: lattice field's content at the projection sphere falls like (rho0/a_L)^nu.
DEFAULT_N_EVAL_EXTRA = 16

Kind = Literal["j", "h"]
Family = Literal["L", "M", "N"]


# ---------------------------------------------------------------------------
# Radial and angular functions, and the three vector families
# ---------------------------------------------------------------------------


def _h(n: int, x: NDArray) -> NDArray:
    """h_n^(1)(x) from its closed-form finite series, valid for complex x.

    ⚠ NOT j_n + i y_n.  For Im x > 0 both j_n and y_n grow like e^{Im x}
    while h_n decays like e^{-Im x}, and their sum cancels catastrophically:
    against mpmath, scipy's j + i y is wrong by 1e-11 at x = 30 + 6i and
    entirely by x = 120 + 24i.  The series

        h_n(x) = (-i)^(n+1) (e^{ix}/x) sum_k i^k (n+k)! / (k! (n-k)! (2x)^k)

    has no such cancellation.

    Args:
        n: Order.
        x: Argument, nonzero.

    Returns:
        h_n^(1)(x).
    """
    x = np.asarray(x, dtype=complex)
    acc = np.zeros_like(x)
    coef = 1.0
    for k in range(n + 1):
        if k > 0:
            coef *= (n + k) * (n - k + 1) / k
        acc = acc + coef * (1j**k) / (2.0 * x) ** k
    return (-1j) ** (n + 1) * np.exp(1j * x) / x * acc


def _z(n: int, x: NDArray, kind: Kind) -> NDArray:
    if kind == "j":
        return spherical_jn(n, x)
    return _h(n, x)


def _zp(n: int, x: NDArray, kind: Kind) -> NDArray:
    if kind == "j":
        return spherical_jn(n, x, derivative=True)
    return (n / np.asarray(x, dtype=complex)) * _h(n, x) - _h(n + 1, x)


def _y(n: int, m: int, th: NDArray, ph: NDArray) -> NDArray:
    if abs(m) > n:
        return np.zeros_like(th, dtype=complex)
    return sph_harm_y(n, m, th, ph)


def _dth_y(n: int, m: int, th: NDArray, ph: NDArray) -> NDArray:
    """dY_nm/dtheta = m cot(theta) Y_nm + sqrt((n-m)(n+m+1)) e^{-i phi} Y_{n,m+1}."""
    return m / np.tan(th) * _y(n, m, th, ph) + np.sqrt(max((n - m) * (n + m + 1), 0)) * np.exp(
        -1j * ph
    ) * _y(n, m + 1, th, ph)


def _frame(pts: NDArray) -> tuple[NDArray, ...]:
    r = np.linalg.norm(pts, axis=-1)
    th = np.arccos(np.clip(pts[..., 2] / r, -1.0, 1.0))
    ph = np.arctan2(pts[..., 1], pts[..., 0])
    rh = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    tt = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
    pp = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], -1)
    return r, th, ph, rh, tt, pp


def _grad_s_y(n: int, m: int, th: NDArray, ph: NDArray, tt: NDArray, pp: NDArray) -> NDArray:
    """The surface gradient r grad_s Y_nm = dY/dth th^ + (i m / sin th) Y ph^."""
    return _dth_y(n, m, th, ph)[..., None] * tt + (1j * m / np.sin(th) * _y(n, m, th, ph))[..., None] * pp


def l_field(n: int, m: int, k: complex, pts: NDArray, kind: Kind) -> NDArray:
    """L_nm = grad(z_n(k r) Y_nm) at points, shape (N, 3), Cartesian (x, y, z)."""
    r, th, ph, rh, tt, pp = _frame(np.asarray(pts, dtype=float))
    zz = _z(n, k * r, kind)
    return (k * _zp(n, k * r, kind) * _y(n, m, th, ph))[..., None] * rh + (zz / r)[..., None] * _grad_s_y(
        n, m, th, ph, tt, pp
    )


def m_field(n: int, m: int, k: complex, pts: NDArray, kind: Kind) -> NDArray:
    """M_nm = curl(r z_n Y_nm) = z_n (grad_s Y x r^)."""
    r, th, ph, rh, tt, pp = _frame(np.asarray(pts, dtype=float))
    return _z(n, k * r, kind)[..., None] * np.cross(_grad_s_y(n, m, th, ph, tt, pp), rh)


def n_field(n: int, m: int, k: complex, pts: NDArray, kind: Kind) -> NDArray:
    """N_nm = curl M_nm = n(n+1) (z_n/r) Y r^ + (z_n/r + k z_n') grad_s Y."""
    r, th, ph, rh, tt, pp = _frame(np.asarray(pts, dtype=float))
    zz = _z(n, k * r, kind)
    radial = (n * (n + 1) * zz / r * _y(n, m, th, ph))[..., None] * rh
    return radial + (zz / r + k * _zp(n, k * r, kind))[..., None] * _grad_s_y(n, m, th, ph, tt, pp)


# ---------------------------------------------------------------------------
# Gaunt coefficients by quadrature
# ---------------------------------------------------------------------------


class GauntTable:
    """G(l1 m1; l2 m2; l3, m3 = -(m1+m2)) = Int Y_l1m1 Y_l2m2 Y_l3m3 dOmega.

    The azimuthal integral is 2 pi when the m's sum to zero, and the polar one
    is a product of three normalised associated Legendre functions -- a
    polynomial of degree <= l1 + l2 + l3 in cos(theta) times powers of
    sin(theta) whose parities pair up -- integrated exactly by Gauss-Legendre
    with l1max + l2max + 2 nodes.  Sympy's exact Gaunt is the check.

    Args:
        l1max: Highest first degree.
        l2max: Highest second degree.
    """

    def __init__(self, l1max: int, l2max: int) -> None:
        self.l1max, self.l2max = l1max, l2max
        lmax = l1max + l2max
        nx = lmax + 2
        x, w = np.polynomial.legendre.leggauss(nx)
        th = np.arccos(x)
        self._off = lmax
        pt = np.zeros((lmax + 1, 2 * lmax + 1, nx))
        for ll in range(lmax + 1):
            for mm in range(-ll, ll + 1):
                pt[ll, mm + lmax] = sph_harm_y(ll, mm, th, np.zeros_like(th)).real
        self._tab: dict[tuple[int, int], NDArray] = {}
        m2 = np.arange(-l2max, l2max + 1)
        for l1 in range(l1max + 1):
            for m1 in range(-l1, l1 + 1):
                m3 = -(m1 + m2)
                ok = np.abs(m3) <= lmax
                prod = pt[l1, m1 + lmax][None, None, :] * pt[: l2max + 1][:, m2 + lmax, :]
                p3 = np.zeros((len(m2), lmax + 1, nx))
                p3[ok] = np.transpose(pt[:, m3[ok] + lmax, :], (1, 0, 2))
                self._tab[l1, m1] = 2.0 * np.pi * np.einsum("abx,bcx,x->abc", prod, p3, w)

    def row(self, l1: int, m1: int) -> NDArray:
        """Array over (l2, m2 + l2max, l3) for fixed (l1, m1).

        Args:
            l1: First degree.
            m1: First order.

        Returns:
            Shape (l2max + 1, 2 l2max + 1, l1max + l2max + 1).
        """
        return self._tab[l1, m1]

    def value(self, l1: int, m1: int, l2: int, m2: int, l3: int) -> float:
        """One coefficient.

        Args:
            l1: First degree.
            m1: First order.
            l2: Second degree.
            m2: Second order.
            l3: Third degree; m3 = -(m1 + m2).

        Returns:
            The Gaunt coefficient.
        """
        return float(self._tab[l1, m1][l2, m2 + self.l2max, l3])


# ---------------------------------------------------------------------------
# Translation and lattice coupling
# ---------------------------------------------------------------------------


def scalar_index(nmin: int, nmax: int) -> list[tuple[int, int]]:
    """[(n, m)] for nmin <= n <= nmax, m = -n..n."""
    return [(n, m) for n in range(nmin, nmax + 1) for m in range(-n, n + 1)]


def full_index(nmax: int) -> list[tuple[str, tuple[int, int]]]:
    """[(family, (n, m))]: L from n = 0, M and N from n = 1, up to nmax."""
    out: list[tuple[str, tuple[int, int]]] = [("L", nm) for nm in scalar_index(0, nmax)]
    out += [(f, nm) for f in ("M", "N") for nm in scalar_index(1, nmax)]
    return out


def single_site_constants(k: complex, rvec: NDArray, qmax: int) -> dict[tuple[int, int], complex]:
    """The structure constants of ONE site at rvec: h_q(k|R|) Y_q^s(R^).

    With these, ``vector_coupling`` is the translation of a single multipole.

    Args:
        k: Wavenumber.
        rvec: Site position.
        qmax: Highest order.

    Returns:
        {(q, s): value}.
    """
    r = float(np.linalg.norm(rvec))
    th = float(np.arccos(rvec[2] / r))
    ph = float(np.arctan2(rvec[1], rvec[0]))
    return {
        (q, s): complex(_z(q, np.asarray(k * r), "h") * sph_harm_y(q, s, th, ph))
        for q in range(qmax + 1)
        for s in range(-q, q + 1)
    }


def _translation(
    d: dict[tuple[int, int], complex],
    src: list[tuple[int, int]],
    tgt: list[tuple[int, int]],
    tab: GauntTable,
) -> NDArray:
    """Scalar translation C[tgt, src] from structure constants.

    Args:
        d: Structure constants, keys (q, s); must reach q = n + nu for every pair.
        src: Source orders.
        tgt: Target orders.
        tab: Gaunt table covering (src n, tgt nu).

    Returns:
        Shape (len(tgt), len(src)).
    """
    qmax = max(q for q, _ in d)
    darr = np.zeros((qmax + 1, 2 * qmax + 1), dtype=complex)
    for (q, s), v in d.items():
        darr[q, s + qmax] = v
    need = max(n for n, _ in src) + max(nu for nu, _ in tgt)
    if need > qmax:
        raise ValueError(
            f"structure constants reach q = {qmax} but the coupling needs q = {need}; "
            f"compute them with qmax >= {need}"
        )
    tpos = {nm: i for i, nm in enumerate(tgt)}
    out = np.zeros((len(tgt), len(src)), dtype=complex)
    l2max = tab.l2max
    for j, (n, m) in enumerate(src):
        row = tab.row(n, m)
        for nu in range(max(nu for nu, _ in tgt) + 1):
            mus = np.arange(-nu, nu + 1)
            qs = np.arange(abs(n - nu), n + nu + 1)
            g = row[nu, -mus + l2max][:, qs]  # (mu, q), m2 = -mu
            ss = m - mus
            dq = np.where(np.abs(ss)[:, None] <= qs[None, :], darr[qs[None, :], ss[:, None] + qmax], 0.0)
            phase = (1j ** (nu + qs - n) * (-1.0) ** qs)[None, :]
            vals = 4.0 * np.pi * (-1.0) ** m * np.sum(phase * dq * g, axis=1)
            for mu, v in zip(mus, vals, strict=True):
                if (nu, int(mu)) in tpos:
                    out[tpos[nu, int(mu)], j] = v
    return out


def vector_coupling(
    d_p: dict[tuple[int, int], complex],
    d_s: dict[tuple[int, int], complex],
    k_p: complex,
    k_s: complex,
    nmax: int,
    rho0: float,
    n_eval_extra: int = DEFAULT_N_EVAL_EXTRA,
    n_theta: int | None = None,
) -> tuple[dict[tuple[tuple[str, int, int], tuple[str, int, int]], complex], list[tuple[str, int, int]]]:
    """The coupling G0 of outgoing L, M, N sources (n <= nmax) into regular multipoles.

    Targets run to n_eval = nmax + n_eval_extra, so that the coupling can also
    reconstruct the field; the solve truncates them to nmax.

    Args:
        d_p: Structure constants at k_P (lattice, or ``single_site_constants``).
        d_s: Structure constants at k_S.
        k_p: P wavenumber.
        k_s: S wavenumber.
        nmax: Highest source order.
        rho0: Projection radius; inside the nearest other site, with
            j_nu(k_S rho0) clear of its zeros for nu <= n_eval.
        n_eval_extra: Target orders beyond nmax.
        n_theta: Projection quadrature in cos(theta); default n_eval + 8.

    Returns:
        (G0 as {((F', nu, mu), (F, n, m)): value}, source list).
    """
    n_eval = nmax + n_eval_extra
    tab = GauntTable(nmax, n_eval)
    src_l = scalar_index(0, nmax)
    tgt = scalar_index(0, n_eval)
    g0: dict[tuple[tuple[str, int, int], tuple[str, int, int]], complex] = {}

    # L -> L: the scalar translation itself.
    c_p = _translation(d_p, src_l, tgt, tab)
    for j, (n, m) in enumerate(src_l):
        for i, (nu, mu) in enumerate(tgt):
            g0[("L", nu, mu), ("L", n, m)] = complex(c_p[i, j])

    # M, N: scalar lattice sums of psi_{n, m-1..m+1}, then projection.
    c_s = _translation(d_s, src_l, tgt, tab)
    nt = n_theta if n_theta is not None else n_eval + 8
    u, wu = np.polynomial.legendre.leggauss(nt)
    nph = 2 * nt
    th_g, ph_g = np.meshgrid(np.arccos(u), 2.0 * np.pi * np.arange(nph) / nph, indexing="ij")
    wts = (wu[:, None] * np.full(nph, 2.0 * np.pi / nph)[None, :]).ravel()
    th, ph = th_g.ravel(), ph_g.ravel()
    pts = rho0 * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    _, _, _, rh, tt, pp = _frame(pts)
    psi = np.array([_z(nu, np.asarray(k_s * rho0), "j") * _y(nu, mu, th, ph) for nu, mu in tgt])
    gpsi = np.array([l_field(nu, mu, k_s, pts, "j") for nu, mu in tgt])  # grad psi_t, (nt, npts, 3)
    ys = [_y(nu, mu, th, ph) for nu, mu in tgt]
    xs = [np.cross(_grad_s_y(nu, mu, th, ph, tt, pp), rh) for nu, mu in tgt]
    spos = {nm: i for i, nm in enumerate(src_l)}

    for n in range(1, nmax + 1):
        for m in range(-n, n + 1):
            vm = np.zeros((len(th), 3), dtype=complex)
            vn = np.zeros((len(th), 3), dtype=complex)
            for mp, amp, vec in (
                (m + 1, np.sqrt(max((n - m) * (n + m + 1), 0)), np.array([0.5, 0.5 / 1j, 0.0])),
                (m - 1, np.sqrt(max((n + m) * (n - m + 1), 0)), np.array([0.5, -0.5 / 1j, 0.0])),
                (m, float(m), np.array([0.0, 0.0, 1.0])),
            ):
                if abs(mp) > n or amp == 0.0:
                    continue
                col = c_s[:, spos[n, mp]]
                sval = col @ psi  # scalar lattice field of psi_{n,mp} at the points
                sgrad = np.einsum("t,tpc->pc", col, gpsi)
                w = -1j * amp * vec
                vm += sval[:, None] * w[None, :]
                vn += np.cross(sgrad, w[None, :])  # curl(w f) = grad f x w
            for (nu, mu), y, x in zip(tgt, ys, xs, strict=True):
                if nu == 0:
                    continue
                jn = spherical_jn(nu, k_s * rho0)
                nn1 = nu * (nu + 1)
                for fam, v in (("M", vm), ("N", vn)):
                    beta = np.sum(wts * np.sum(v * rh, -1) * np.conj(y)) / (nn1 * jn / rho0)
                    alpha = np.sum(wts * np.sum(v * np.conj(x), -1)) / (jn * nn1)
                    g0[("M", nu, mu), (fam, n, m)] = complex(alpha)
                    g0[("N", nu, mu), (fam, n, m)] = complex(beta)
    sources = [(f, n, m) for f, (n, m) in full_index(nmax)]
    return g0, sources


# ---------------------------------------------------------------------------
# Angular functions continued to complex directions, and plane-wave spectra
# ---------------------------------------------------------------------------


def _legendre_deriv(n: int, k: int, u: NDArray) -> NDArray:
    """d^k P_n / du^k at (possibly complex) u."""
    c = np.zeros(n + 1)
    c[n] = 1.0
    return np.polynomial.legendre.legval(u, np.polynomial.legendre.legder(c, k) if k > 0 else c)


def angular_triple(
    n: int, m: int, u: NDArray, s: NDArray, cph: NDArray, sph: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """{Y_nm, dY/dtheta, (1/sin theta) dY/dphi} at a direction given by cos, sin, azimuth.

    Written as Y_nm = c_nm s^|m| P_n^(|m|)(u) e^{i m phi}, a polynomial in u =
    cos(theta) and s = sin(theta), so it continues analytically to the COMPLEX
    directions of evanescent plane waves (u = +-i kap''/K, s = q/K > 1).  On the
    axis (s = 0) the azimuth is undefined; the caller passes the phi = 0 branch,
    the one ``gate_sphere_vs_impedance_march.mode_matrix`` takes.

    Args:
        n: Degree.
        m: Order.
        u: cos(theta), complex allowed.
        s: sin(theta), taken as q/K >= 0.
        cph: cos(phi).
        sph: sin(phi).

    Returns:
        (Y, dY/dtheta, (1/sin theta) dY/dphi).
    """
    am = abs(m)
    norm = np.sqrt((2 * n + 1) / (4.0 * np.pi) * factorial(n - m) / factorial(n + m))
    c = norm * ((-1.0) ** m if m >= 0 else factorial(n - am) / factorial(n + am))
    eim = (np.asarray(cph) + 1j * np.asarray(sph)) ** m
    f = _legendre_deriv(n, am, u)
    fp = _legendre_deriv(n, am + 1, u) if am + 1 <= n else 0.0 * u
    s = np.asarray(s, dtype=complex)
    y = c * s**am * f * eim
    # d/dtheta [s^|m| f(u)] = |m| u s^(|m|-1) f - s^(|m|+1) f'(u)
    ds = (am * u * s ** (am - 1) * f if am > 0 else 0.0 * u) - s ** (am + 1) * fp
    dth = c * ds * eim
    dph = 1j * m * c * s ** (am - 1) * f * eim if am > 0 else 0.0 * y
    return y, dth, dph


def plane_wave_amplitude(
    fam: str, n: int, m: int, kx: NDArray, ky: NDArray, k_p: complex, k_s: complex, upward: bool
) -> NDArray:
    """uhat(q) of one OUTGOING multipole: its field is Int d^2q uhat e^{i(q.rho + kap |z|)}.

    Section 7 of ``Mathematica/MieSphericalWaves.wl`` (validated there for
    m = 0, +-1 against the exact field), for general m:

        L : (i K / D) Y e_P
        N : (i K / D) (dY/dth e_SV + (1/sin) dY/dph e_SH)
        M : (-1 / D) (-(1/sin) dY/dph e_SV + dY/dth e_SH) ,     D = 2 pi K i^n kap ,

    angular functions at the plane-wave direction (kx, ky, +-kap)/K, polarisations
    e_P = k^, e_SV = theta^, e_SH = phi^ there.

    Args:
        fam: "L", "M" or "N".
        n: Degree.
        m: Order.
        kx: Lateral wavenumber, x (array).
        ky: Lateral wavenumber, y (array).
        k_p: P wavenumber.
        k_s: S wavenumber.
        upward: True for the half-space z < 0.

    Returns:
        Shape kx.shape + (3,), Cartesian (x, y, z).
    """
    kk = k_p if fam == "L" else k_s
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    q = np.hypot(kx, ky)
    kap = np.sqrt(np.asarray(kk**2 - q**2, dtype=complex))
    kap = np.where(kap.imag < 0, -kap, kap)
    kzd = -kap if upward else kap
    on = q < 1e-300
    qs = np.where(on, 1.0, q)
    cph = np.where(on, 1.0, kx / qs)
    sph = np.where(on, 0.0, ky / qs)
    u, s = kzd / kk, q / kk
    y, dth, dph = angular_triple(n, m, u, s, cph, sph)
    e_p = np.stack([kx / kk, ky / kk, kzd / kk], -1)
    e_sv = np.stack([u * cph, u * sph, -s + 0j], -1)
    e_sh = np.stack([-sph, cph, np.zeros_like(cph)], -1).astype(complex)
    c_p, c_sv, c_sh = plane_wave_modes(fam, n, m, kx, ky, k_p, k_s, upward)
    return c_p[..., None] * e_p + c_sv[..., None] * e_sv + c_sh[..., None] * e_sh


def plane_wave_modes(
    fam: str, n: int, m: int, kx: NDArray, ky: NDArray, k_p: complex, k_s: complex, upward: bool
) -> tuple[NDArray, NDArray, NDArray]:
    """The (P, SV, SH) mode coefficients of ``plane_wave_amplitude``.

    Returned as coefficients of e_P, e_SV, e_SH rather than projected with a dot
    product: for evanescent orders the polarisation vectors are complex and not
    orthonormal under the Hermitian product.

    Args:
        fam: "L", "M" or "N".
        n: Degree.
        m: Order.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.
        k_p: P wavenumber.
        k_s: S wavenumber.
        upward: True for the half-space z < 0.

    Returns:
        (c_P, c_SV, c_SH), each of kx's shape.
    """
    kk = k_p if fam == "L" else k_s
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    q = np.hypot(kx, ky)
    kap = np.sqrt(np.asarray(kk**2 - q**2, dtype=complex))
    kap = np.where(kap.imag < 0, -kap, kap)
    kzd = -kap if upward else kap
    on = q < 1e-300
    qs = np.where(on, 1.0, q)
    cph = np.where(on, 1.0, kx / qs)
    sph = np.where(on, 0.0, ky / qs)
    y, dth, dph = angular_triple(n, m, kzd / kk, q / kk, cph, sph)
    d = 2.0 * np.pi * kk * (1j**n) * kap
    zero = np.zeros_like(y)
    if fam == "L":
        return 1j * kk / d * y, zero, zero
    if fam == "N":
        return zero, 1j * kk / d * dth, 1j * kk / d * dph
    return zero, dph / d, -dth / d


# ---------------------------------------------------------------------------
# The incident wave, the single-site T-matrix, and the collective solve
# ---------------------------------------------------------------------------


def project_regular(
    field, k_p: complex, k_s: complex, nmax: int, rho0: float, n_theta: int | None = None
) -> dict[tuple[str, int, int], complex]:
    """Regular L, M, N coefficients of a field regular about the origin.

    On a sphere of radius rho0: the r^ x grad_s Y component is M alone; the
    radial and grad_s Y components carry L and N together, one 2x2 per (n, m).

    Args:
        field: Callable, points (N, 3) -> vectors (N, 3).
        k_p: P wavenumber.
        k_s: S wavenumber.
        nmax: Highest order.
        rho0: Projection radius.
        n_theta: Quadrature nodes in cos(theta); default nmax + 12.

    Returns:
        {(family, n, m): coefficient}.
    """
    nt = n_theta if n_theta is not None else nmax + 12
    u, wu = np.polynomial.legendre.leggauss(nt)
    nph = 2 * nt
    th_g, ph_g = np.meshgrid(np.arccos(u), 2.0 * np.pi * np.arange(nph) / nph, indexing="ij")
    wts = (wu[:, None] * np.full(nph, 2.0 * np.pi / nph)[None, :]).ravel()
    th, ph = th_g.ravel(), ph_g.ravel()
    pts = rho0 * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    _, _, _, rh, tt, pp = _frame(pts)
    v = field(pts)
    vr = np.sum(v * rh, -1)
    out: dict[tuple[str, int, int], complex] = {}
    for n in range(0, nmax + 1):
        for m in range(-n, n + 1):
            y = _y(n, m, th, ph)
            pr = np.sum(wts * vr * np.conj(y))
            jp, jpp = spherical_jn(n, k_p * rho0), spherical_jn(n, k_p * rho0, derivative=True)
            if n == 0:
                out["L", 0, 0] = complex(pr / (k_p * jpp))
                continue
            gs = _grad_s_y(n, m, th, ph, tt, pp)
            nn1 = n * (n + 1)
            pg = np.sum(wts * np.sum(v * np.conj(gs), -1)) / nn1
            pm = np.sum(wts * np.sum(v * np.conj(np.cross(gs, rh)), -1)) / nn1
            js, jsp = spherical_jn(n, k_s * rho0), spherical_jn(n, k_s * rho0, derivative=True)
            out["M", n, m] = complex(pm / js)
            a = np.array([[k_p * jpp, nn1 * js / rho0], [jp / rho0, js / rho0 + k_s * jsp]], dtype=complex)
            sol = np.linalg.solve(a, np.array([pr, pg], dtype=complex))
            out["L", n, m], out["N", n, m] = complex(sol[0]), complex(sol[1])
    return out


def incident_coefficients(
    mode: str, khat: NDArray, k_p: complex, k_s: complex, nmax: int, pol: NDArray | None = None
) -> dict[tuple[str, int, int], complex]:
    """Regular-multipole coefficients of a unit plane wave e e^{i K khat.r}.

    P (e = khat): e^{ik.r} = 4 pi sum i^n j_n Y_nm(r^) conj(Y_nm(k^)), and
    u = grad(e^{ik.r})/(iK), so a^L_nm = 4 pi i^n conj(Y_nm(k^))/(i K).
    S (e = pol, transverse): by ``project_regular``.

    Args:
        mode: "P" or "S".
        khat: Real unit propagation direction.
        k_p: P wavenumber.
        k_s: S wavenumber.
        nmax: Highest order.
        pol: Unit polarisation, for S.

    Returns:
        {(family, n, m): coefficient}.
    """
    if mode == "P":
        th = float(np.arccos(np.clip(khat[2], -1.0, 1.0)))
        ph = float(np.arctan2(khat[1], khat[0]))
        return {
            ("L", n, m): complex(4.0 * np.pi * 1j**n * np.conj(sph_harm_y(n, m, th, ph)) / (1j * k_p))
            for n in range(nmax + 1)
            for m in range(-n, n + 1)
        }
    if pol is None:
        raise ValueError("an S plane wave needs its polarisation: pass pol=np.array([...])")
    kvec = k_s * np.asarray(khat, dtype=float)
    return project_regular(
        lambda pts: np.exp(1j * pts @ kvec)[:, None] * np.asarray(pol)[None, :],
        k_p,
        k_s,
        nmax,
        rho0=1.0 / abs(k_s),
    )


def sphere_tmatrix(
    omega: float, radius: float, ref: ReferenceMedium, contrast: MaterialContrast, nmax: int
) -> dict[int, tuple[NDArray, complex]]:
    """Per order: the (L, N) 2x2 T-matrix and the M scalar, from Mie.

    The basis's potentials are ``sphere_scattering``'s, and T is independent of
    m, so these ARE ``mie_tmatrix_psv`` and ``mie_tmatrix_sh``.

    Args:
        omega: Angular frequency.
        radius: Sphere radius.
        ref: Background.
        contrast: Contrast.
        nmax: Highest order.

    Returns:
        {n: (T_psv, T_sh)}.
    """
    return {
        n: (
            mie_tmatrix_psv(n, omega, radius, ref, contrast),
            mie_tmatrix_sh(n, omega, radius, ref, contrast) if n else 0j,
        )
        for n in range(nmax + 1)
    }


def solve_array(
    g0: dict[tuple[tuple[str, int, int], tuple[str, int, int]], complex],
    tm: dict[int, tuple[NDArray, complex]],
    a_inc: dict[tuple[str, int, int], complex],
    nmax: int,
) -> dict[tuple[str, int, int], complex]:
    """Scattered coefficients of every sphere of the array: b = T (I - G0 T)^-1 a_inc.

    Args:
        g0: Lattice coupling (``vector_coupling``); pass {} for the isolated sphere.
        tm: ``sphere_tmatrix``.
        a_inc: Incident regular coefficients.
        nmax: Truncation order.

    Returns:
        {(family, n, m): b}.
    """
    idx = [(f, n, m) for f, (n, m) in full_index(nmax)]
    pos = {k: i for i, k in enumerate(idx)}
    nd = len(idx)
    t = np.zeros((nd, nd), dtype=complex)
    for f, n, m in idx:
        i = pos[f, n, m]
        tp, ts = tm[n]
        if f == "M":
            t[i, i] = ts
        elif f == "L":
            t[i, i] = tp[0, 0]
            if n >= 1:
                t[i, pos["N", n, m]] = tp[0, 1]
        else:
            t[i, i] = tp[1, 1]
            t[i, pos["L", n, m]] = tp[1, 0]
    g = np.zeros((nd, nd), dtype=complex)
    for (tgt, src), v in g0.items():
        if tgt in pos and src in pos:
            g[pos[tgt], pos[src]] = v
    a = np.array([a_inc.get(k, 0.0) for k in idx], dtype=complex)
    exc = np.linalg.solve(np.eye(nd) - g @ t, a)
    b = t @ exc
    return {k: complex(b[i]) for k, i in pos.items()}
