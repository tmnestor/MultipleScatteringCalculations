#!/usr/bin/env python3
"""The laterally coupled impedance march on a FULL 3-D lateral grid.

ANCHOR: Nestor (1996) Ch.2 (Akdef); Ch.5 (CBPdef0), Algorithms 5.1-5.3.

WHAT CHANGES FROM THE 2.5-D GATE
--------------------------------
``gate_first_order_lateral_impedance`` carries the lateral coupling inside the
marched variable, but on ONE lateral direction: heterogeneity in (x, z) with the
third direction carried by the parameter ``k_y``.  Every ``d_y`` there is the
scalar ``i k_y``.

Here both lateral directions are grid directions.  ``d_y`` becomes a second
derivative MATRIX, and the marched impedance ``Y`` lives on
(3 components) x (N_x N_y lateral points).  Nothing else in the derivation
moves: the modulus/derivative ordering that momentum balance forces is the same
ordering, applied twice.

That matters because a sphere is not invariant in y.  No single ``k_y`` can
represent one, so a 2.5-D march cannot be scored against the exact Mie response
however fine its lateral grid.

THE SYMPLECTIC IDENTITY GETS STRONGER, NOT WEAKER
-------------------------------------------------
In 2.5-D the test had to be ``J6 A(k_y) = [J6 A(-k_y)]^T`` -- a transpose in the
lateral index plus an explicit sign flip, because ``k_y`` is a parameter and a
transpose does nothing to a scalar.  Here ``d_y`` is a real ANTISYMMETRIC matrix,
so the transpose flips its sign by itself and the identity closes outright:

    J6 A = (J6 A)^T

with no parameter to flip.  The wrong modulus/derivative ordering still breaks
it, and still agrees with the derived operator in a laterally uniform medium --
so the control is kept, for the same reason.

WHAT IS CHECKED AGAINST WHAT
----------------------------
Every part scores against an object this file does not own:

  1. Laterally uniform  -> ``amat_thesis``, already validated 13/13 against
     (Akdef), at every one of the N_x N_y wavenumber pairs.
  2. Medium invariant in y -> the committed 2.5-D ``aop``, block by block in
     k_y.  This is the identity that ties the new operator to the gated one.
  3. Symplectic structure, with the wrong ordering as the control.
  4. Shift covariance in x and in y SEPARATELY -- an index slip between the two
     Kronecker factors is invisible to a diagonal shift.
  5. Zero lateral contrast -> the 3x3 march of ``gate_first_order_impedance``.
  6. Fourth order in the depth step on a laterally smooth medium.

THE LATERAL PML IS BUILT AND REFUTED
------------------------------------
Parts 7 and 8 carry a complex-coordinate stretch and its verdict.  The
STRUCTURE holds: under ``d_x -> S_x^{-1} d_x`` the symplectic identity survives
in weighted form, ``J6 A = W^{-1} (J6 A)^T W`` with ``W = I6 (x) S_x S_y``, at
the arithmetic floor, collapsing to the plain identity at zero absorption.

The PURPOSE does not.  Absorption increases the scattered lateral reach instead
of cutting it, at every setting tried.  A pseudospectral derivative is dense, so
the band is never far from the interior; and the zero-absorption image
contamination is already ~3e-3, so there is little to remove.  Part 8 states the
measurement and the two metrics that had to be discarded first.  ▶ The exact
alternative -- a Bloch sweep over the Brillouin zone -- needs no change to the
operator validated in Parts 1-6.

GEOMETRY AND UNITS
------------------
SI throughout -- m, m/s, kg/m^3, rad/m, rad/s -- matching the 2.5-D gate and
``gate_first_order_impedance_march``, whose background, frequency and slab
thickness are reused so that Part 5 is an identity and not a comparison of two
conventions.

Lateral index ordering is x-slowest: ``idx = ix * N_y + iy``, so that
``d_x = kron(D_x, I)`` and ``d_y = kron(I, D_y)``.

Run:  conda run -n seismic python scripts/gate_first_order_lateral_impedance_3d.py
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from scripts.gate_first_order_impedance import ablocks, newton_are  # noqa: E402
from scripts.gate_first_order_impedance_march import march, y_downgoing  # noqa: E402
from scripts.gate_first_order_lateral_impedance import Slice as Slice2D  # noqa: E402
from scripts.gate_first_order_lateral_impedance import aop as aop_25d  # noqa: E402
from scripts.gate_first_order_lateral_impedance import march_lateral as march_lateral_25d  # noqa: E402
from scripts.gate_thesis_spectral import amat_thesis, dz_normalised  # noqa: E402

#: Background half-space below the slab, and the slab geometry.  Identical to
#: the 2.5-D gate and to ``gate_first_order_impedance_march``.
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
H = 200.0

#: Lateral periods.  At OMEGA the background has omega/alpha = 1.2e-2 and
#: omega/beta = 2.0e-2 per metre, and the grid wavenumbers are 2 pi m / L =
#: 6.28e-3 m, so |m| <= 3 propagates and the rest is evanescent.  Equal periods
#: keep the two wavenumber grids identical, which makes an x/y index slip show
#: up as a wrong ANSWER rather than as a wrong grid.
LX = 1000.0
LY = 1000.0

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


# ---------------------------------------------------------------------------
# The lateral grid
# ---------------------------------------------------------------------------


def grid_wavenumbers(n: int, period: float) -> NDArray:
    """The lateral wavenumbers of one periodic direction, in FFT order.

    The Nyquist entry is zeroed for even n.  That is what makes the
    differentiation matrix exactly antisymmetric, and the symplectic identity is
    an identity only for an antisymmetric derivative.

    Args:
        n: Number of points in this direction.
        period: Physical length of this direction.

    Returns:
        Shape (n,), real.
    """
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=period / n)
    if n % 2 == 0:
        k[n // 2] = 0.0
    return k


def deriv_1d(n: int, period: float) -> NDArray:
    """Spectral d/ds on one periodic direction.

    Args:
        n: Points in this direction.
        period: Physical length.

    Returns:
        Shape (n, n), real, antisymmetric.
    """
    k = grid_wavenumbers(n, period)
    d = np.fft.ifft(1j * k[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0)
    return np.real(d)


def deriv_pair(nx: int, ny: int) -> tuple[NDArray, NDArray]:
    """The two lateral derivative matrices on the flattened (N_x N_y) grid.

    With ``idx = ix * ny + iy`` the two directions are the two Kronecker
    factors, so each stays exactly antisymmetric and the two commute.

    Args:
        nx: Points along x.
        ny: Points along y.

    Returns:
        (d_x, d_y), each shape (nx*ny, nx*ny).
    """
    dx1, dy1 = deriv_1d(nx, LX), deriv_1d(ny, LY)
    return np.kron(dx1, np.eye(ny)), np.kron(np.eye(nx), dy1)


# ---------------------------------------------------------------------------
# The lateral PML
# ---------------------------------------------------------------------------


def stretch_1d(n: int, period: float, omega: complex, *, width: float, smax: float, p: int = 3) -> NDArray:
    """The complex coordinate stretch s(x) = 1 + i sigma(x) / omega.

    On a PERIODIC grid there is no outer boundary to line: the wrap point IS
    the boundary, so the absorbing band straddles it and falls to zero towards
    the domain centre.  A scatterer placed at the centre is then separated from
    its periodic images by two passes through the band.

    Args:
        n: Points in this direction.
        period: Physical length.
        omega: Angular frequency.
        width: Absorbing band half-width, in the same units as ``period``.
            The band occupies ``|x| < width`` measured periodically from zero.
        smax: Peak of sigma, in units of omega (so ``smax=1`` means
            ``sigma = omega`` at the wrap point).
        p: Polynomial grading exponent.

    Returns:
        Shape (n,), complex.  Identically one when ``width`` or ``smax`` is zero.
    """
    x = np.arange(n) * (period / n)
    xi = np.minimum(x, period - x)  # periodic distance from the wrap point
    if width <= 0.0 or smax == 0.0:
        return np.ones(n, dtype=complex)
    depth = np.clip((width - xi) / width, 0.0, 1.0)
    sigma = smax * float(np.abs(omega)) * depth**p
    return 1.0 + 1j * sigma / omega


def deriv_pair_pml(
    nx: int, ny: int, omega: complex, *, width: float, smax: float, p: int = 3
) -> tuple[NDArray, NDArray, NDArray]:
    """The stretched lateral derivatives, and the weight the symmetry needs.

    The PML is the substitution ``d_x -> S_x^{-1} d_x`` applied wherever a
    lateral derivative appears -- an analytic continuation of the coordinate,
    nothing more.  That does NOT leave ``J6 A`` symmetric: transposing carries
    the stretch across, and what survives is the WEIGHTED identity

        J6 A = W^{-1} (J6 A)^T W,     W = I6 (x) S_x S_y

    which is provable term by term because every modulus, and every stretch, is
    diagonal and therefore commutes with the other direction's derivative.  At
    zero absorption ``W`` is the identity and this collapses to the plain
    statement, so one test covers both.

    Args:
        nx: Points along x.
        ny: Points along y.
        omega: Angular frequency.
        width: Absorbing band half-width, applied in BOTH directions.
        smax: Peak of sigma in units of omega.
        p: Polynomial grading exponent.

    Returns:
        (d_x stretched, d_y stretched, the lateral weight S_x S_y of shape (N, N)).
    """
    dx, dy = deriv_pair(nx, ny)
    sx = stretch_1d(nx, LX, omega, width=width, smax=smax, p=p)
    sy = stretch_1d(ny, LY, omega, width=width, smax=smax, p=p)
    big_sx = np.diag(np.kron(sx, np.ones(ny)))
    big_sy = np.diag(np.kron(np.ones(nx), sy))
    inv_sx = np.diag(1.0 / np.kron(sx, np.ones(ny)))
    inv_sy = np.diag(1.0 / np.kron(np.ones(nx), sy))
    return inv_sx @ dx, inv_sy @ dy, big_sx @ big_sy


def dft_2d(nx: int, ny: int) -> tuple[NDArray, NDArray]:
    """The 2-D DFT on the flattened lateral grid, as a matrix pair.

    Args:
        nx: Points along x.
        ny: Points along y.

    Returns:
        (F, F_inv) acting on a flattened (nx*ny,) vector.
    """
    fx, fy = np.fft.fft(np.eye(nx), axis=0), np.fft.fft(np.eye(ny), axis=0)
    gx, gy = np.fft.ifft(np.eye(nx), axis=0), np.fft.ifft(np.eye(ny), axis=0)
    return np.kron(fx, fy), np.kron(gx, gy)


def to_hat(mat: NDArray, nx: int, ny: int, ncomp: int) -> NDArray:
    """Transform a spatial-grid operator to the lateral wavenumber domain.

    Args:
        mat: Shape (ncomp*nx*ny, ncomp*nx*ny), grid index fastest.
        nx: Points along x.
        ny: Points along y.
        ncomp: Components.

    Returns:
        Same shape, in the wavenumber domain.
    """
    f, fi = dft_2d(nx, ny)
    big_f = np.kron(np.eye(ncomp), f)
    big_fi = np.kron(np.eye(ncomp), fi)
    return big_f @ mat @ big_fi


def hat_block(mat_hat: NDArray, nx: int, ny: int, ncomp: int, mx: int, my: int) -> NDArray:
    """One wavenumber block of a transformed operator.

    Args:
        mat_hat: Operator in the wavenumber domain.
        nx: Points along x.
        ny: Points along y.
        ncomp: Components.
        mx: Wavenumber index along x.
        my: Wavenumber index along y.

    Returns:
        Shape (ncomp, ncomp).
    """
    n = nx * ny
    idx = np.array([c * n + mx * ny + my for c in range(ncomp)])
    return mat_hat[np.ix_(idx, idx)]


def to_hat_y(mat: NDArray, nx: int, ny: int, ncomp: int) -> NDArray:
    """Transform in the y direction ONLY, leaving x on the spatial grid.

    Used by Part 2, where the medium is invariant in y so the operator block
    diagonalises in k_y but not in k_x.

    Args:
        mat: Shape (ncomp*nx*ny, ncomp*nx*ny).
        nx: Points along x.
        ny: Points along y.
        ncomp: Components.

    Returns:
        Same shape, transformed in y alone.
    """
    fy = np.fft.fft(np.eye(ny), axis=0)
    gy = np.fft.ifft(np.eye(ny), axis=0)
    p = np.kron(np.eye(ncomp * nx), fy)
    pi = np.kron(np.eye(ncomp * nx), gy)
    return p @ mat @ pi


def ky_block(mat_haty: NDArray, nx: int, ny: int, ncomp: int, my: int) -> NDArray:
    """The k_y block of a y-transformed operator, in the 2.5-D index order.

    The 2.5-D gate orders its operator as (component, x); this gathers the same
    ordering out of the (component, x, y) layout so the two can be compared
    entry by entry.

    Args:
        mat_haty: Operator transformed in y alone.
        nx: Points along x.
        ny: Points along y.
        ncomp: Components.
        my: Wavenumber index along y.

    Returns:
        Shape (ncomp*nx, ncomp*nx).
    """
    idx = np.array([c * nx * ny + ix * ny + my for c in range(ncomp) for ix in range(nx)])
    return mat_haty[np.ix_(idx, idx)]


# ---------------------------------------------------------------------------
# The medium on the lateral grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slice:
    """The medium on the flattened lateral grid at one depth.

    Attributes:
        alpha: P speed at each lateral point, shape (nx*ny,).
        beta: S speed, shape (nx*ny,).
        rho: Density, shape (nx*ny,).
    """

    alpha: NDArray
    beta: NDArray
    rho: NDArray

    @property
    def mu(self) -> NDArray:
        """Shear modulus at each lateral point.

        Returns:
            Shape (nx*ny,).
        """
        return self.rho * self.beta**2

    @property
    def lam(self) -> NDArray:
        """First Lame parameter at each lateral point.

        Returns:
            Shape (nx*ny,).
        """
        return self.rho * self.alpha**2 - 2.0 * self.mu


def uniform_slice(med: ReferenceMedium, nx: int, ny: int) -> Slice:
    """A laterally uniform slice of a given medium.

    Args:
        med: The medium.
        nx: Points along x.
        ny: Points along y.

    Returns:
        The slice.
    """
    o = np.ones(nx * ny)
    return Slice(med.alpha * o, med.beta * o, med.rho * o)


def _depth_factor(z: float) -> float:
    """The depth profile shared with the 2.5-D gate and the scalar march.

    Args:
        z: Depth within the slab.

    Returns:
        The multiplicative factor.
    """
    return 1.0 + 0.10 * np.sin(np.pi * z / H) + 0.04 * (z / H)


def graded_3d(z: float, nx: int, ny: int, eps: float, mx: int = 2, my: int = 1) -> Slice:
    """A slab smooth in depth and in BOTH lateral directions.

    The depth profile is the one ``gate_first_order_impedance_march`` grades
    with, so the fourth-order measurement is directly comparable with the
    uncoupled and 2.5-D ones.  The lateral term is a product of one smooth
    Fourier mode in each direction, with DIFFERENT mode numbers: equal ones
    would make the medium symmetric under x <-> y and hide an index slip
    between the two Kronecker factors.

    Args:
        z: Depth within the slab.
        nx: Points along x.
        ny: Points along y.
        eps: Lateral contrast amplitude.
        mx: Lateral Fourier mode along x.
        my: Lateral Fourier mode along y.

    Returns:
        The slice at that depth.
    """
    x = np.arange(nx) * (LX / nx)
    y = np.arange(ny) * (LY / ny)
    lat = np.cos(2.0 * np.pi * mx * x / LX)[:, None] * np.cos(2.0 * np.pi * my * y / LY)[None, :]
    s = _depth_factor(z) + eps * lat.reshape(-1)
    return Slice(REF.alpha * s, REF.beta * s, REF.rho * s)


def graded_y_invariant(z: float, nx: int, ny: int, eps: float, mx: int = 2) -> Slice:
    """The same slab but with NO y dependence, for the 2.5-D reduction.

    Args:
        z: Depth within the slab.
        nx: Points along x.
        ny: Points along y.
        eps: Lateral contrast amplitude.
        mx: Lateral Fourier mode along x.

    Returns:
        The slice at that depth, constant along y.
    """
    x = np.arange(nx) * (LX / nx)
    s1 = _depth_factor(z) + eps * np.cos(2.0 * np.pi * mx * x / LX)
    s = np.repeat(s1, ny)
    return Slice(REF.alpha * s, REF.beta * s, REF.rho * s)


def slice_2d_of(z: float, nx: int, eps: float, mx: int = 2) -> Slice2D:
    """The matching 2.5-D slice, in the committed gate's own dataclass.

    Args:
        z: Depth within the slab.
        nx: Points along x.
        eps: Lateral contrast amplitude.
        mx: Lateral Fourier mode along x.

    Returns:
        The 2.5-D slice.
    """
    x = np.arange(nx) * (LX / nx)
    s = _depth_factor(z) + eps * np.cos(2.0 * np.pi * mx * x / LX)
    return Slice2D(REF.alpha * s, REF.beta * s, REF.rho * s)


def profile_3d(nx: int, ny: int, eps: float) -> Callable[[float], Slice]:
    """The graded slab as a function of depth alone, at fixed grid and contrast.

    Binding the grid and contrast here rather than in a lambda keeps each
    closure's capture explicit, which matters inside a loop over grid sizes.

    Args:
        nx: Points along x.
        ny: Points along y.
        eps: Lateral contrast amplitude.

    Returns:
        A callable of depth.
    """

    def at(z: float) -> Slice:
        return graded_3d(z, nx, ny, eps)

    return at


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------


def aop(
    sl: Slice, omega: complex, dx: NDArray, dy: NDArray, *, wrong_order: bool = False
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The four blocks of the laterally coupled 3-D A, each (3N, 3N).

    Derived from momentum balance, not transcribed.  Writing tau_3 =
    (T_zz, T_xz, T_yz), the two horizontal traction rows are

        d_3 T_xz = -rho w^2 u_x - d_x[zeta d_x u_x + chi d_y u_y + gamma T_zz]
                                - d_y[mu(d_x u_y + d_y u_x)]
        d_3 T_yz = -rho w^2 u_y - d_y[chi d_x u_x + zeta d_y u_y + gamma T_zz]
                                - d_x[mu(d_x u_y + d_y u_x)]

    with every derivative OUTSIDE its modulus.  That ordering is forced by
    d_j T_ij and it is what makes the discretised operator symplectic.  The
    constitutive rows carry the modulus INSIDE, because they are not
    divergences: ``g @ dx``, not ``dx @ g``.

    Args:
        sl: Medium on the flattened lateral grid.
        omega: Angular frequency.
        dx: Lateral derivative along x, shape (N, N).
        dy: Lateral derivative along y, shape (N, N).
        wrong_order: Build the plausible WRONG ordering instead, with the
            modulus outside every derivative pair.  Identical in a laterally
            uniform medium; a different operator as soon as the medium varies.

    Returns:
        (A11, A12, A21, A22).
    """
    n = sl.alpha.size
    mu, lam = sl.mu, sl.lam
    kc = lam + 2.0 * mu
    gam, aa, bb = lam / kc, 1.0 / kc, 1.0 / mu
    zet, chi = 4.0 * mu * (lam + mu) / kc, 2.0 * mu * lam / kc
    rw2 = sl.rho * omega**2

    zero = np.zeros((n, n))
    g, a_d, b_d = np.diag(gam), np.diag(aa), np.diag(bb)
    ze, ch, mu_d = np.diag(zet), np.diag(chi), np.diag(mu)
    rw = np.diag(np.asarray(rw2, dtype=np.complex128))

    if wrong_order:
        # The modulus pulled outside every derivative pair.  The control.
        xx = ze @ dx @ dx + mu_d @ dy @ dy
        yy = ze @ dy @ dy + mu_d @ dx @ dx
        xy = (ch + mu_d) @ dx @ dy
        yx = (ch + mu_d) @ dy @ dx
        gx, gy = g @ dx, g @ dy
    else:
        xx = dx @ ze @ dx + dy @ mu_d @ dy
        yy = dy @ ze @ dy + dx @ mu_d @ dx
        xy = dx @ ch @ dy + dy @ mu_d @ dx
        yx = dy @ ch @ dx + dx @ mu_d @ dy
        gx, gy = dx @ g, dy @ g

    a11 = np.block([[zero, -g @ dx, -g @ dy], [-dx, zero, zero], [-dy, zero, zero]])
    a12 = np.block([[a_d, zero, zero], [zero, b_d, zero], [zero, zero, b_d]])
    a21 = np.block([[-rw, zero, zero], [zero, -rw - xx, -xy], [zero, -yx, -rw - yy]])
    a22 = np.block([[zero, -dx, -dy], [-gx, zero, zero], [-gy, zero, zero]])
    return a11, a12, a21, a22


def abig(blk: tuple) -> NDArray:
    """Assemble the four blocks into one operator.

    Args:
        blk: (A11, A12, A21, A22).

    Returns:
        Shape (6N, 6N).
    """
    a11, a12, a21, a22 = blk
    return np.block([[a11, a12], [a21, a22]])


def jsix(n: int) -> NDArray:
    """The symplectic form on the lateral grid, shape (6N, 6N).

    Args:
        n: Lateral points, N = nx*ny.

    Returns:
        Block [[0, I], [-I, 0]] with I of size 3N.
    """
    i3n = np.eye(3 * n)
    z = np.zeros((3 * n, 3 * n))
    return np.block([[z, i3n], [-i3n, z]])


# ---------------------------------------------------------------------------
# The march
# ---------------------------------------------------------------------------


def y_start(med: ReferenceMedium, omega: complex, nx: int, ny: int) -> NDArray:
    """The downgoing impedance of the laterally uniform half-space below.

    Laterally uniform, so diagonal in (k_x, k_y), and each 3x3 block is the
    object ``gate_first_order_impedance`` already validated -- D_z names the
    branch, Newton supplies the precision.

    Args:
        med: The half-space medium.
        omega: Angular frequency.
        nx: Points along x.
        ny: Points along y.

    Returns:
        Shape (3N, 3N), on the spatial grid.
    """
    kx = grid_wavenumbers(nx, LX)
    ky = grid_wavenumbers(ny, LY)
    n = nx * ny
    yhat = np.zeros((3 * n, 3 * n), dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            kxv, kyv = float(kx[ix]), float(ky[iy])
            if kyv == 0.0:
                blk = y_downgoing(med, omega, kxv)
            else:
                dzm, _, _ = dz_normalised(med, omega, kxv, kyv)
                dd = dzm[:, :3]
                blk = newton_are(dd[3:, :] @ np.linalg.inv(dd[:3, :]), ablocks(med, omega, kxv, kyv))[0]
            idx = np.array([c * n + ix * ny + iy for c in range(3)])
            yhat[np.ix_(idx, idx)] = blk
    f, fi = dft_2d(nx, ny)
    big_f, big_fi = np.kron(np.eye(3), f), np.kron(np.eye(3), fi)
    return np.asarray(big_fi @ yhat @ big_f)


def y_start_stretched(
    med: ReferenceMedium, omega: complex, nx: int, ny: int, dx: NDArray, dy: NDArray, seed: NDArray
) -> tuple[NDArray, float]:
    """The half-space impedance of the STRETCHED operator.

    A complex coordinate stretch destroys lateral translation invariance, so a
    laterally uniform half-space is no longer diagonal in the DFT basis and the
    mode-by-mode construction of ``y_start`` simply does not apply to it.
    Feeding the unstretched start to a stretched operator leaves the base of the
    slab inconsistent with the equation being marched, and the mismatch radiates
    lateral structure that has nothing to do with the medium: measured on a
    laterally UNIFORM slab, the centre-to-wrap reach of Y grew from 3.0e-2 to
    1.9e-1 as sigma_max/omega went 0 -> 4, with nothing present to scatter.

    The half-space is still a half-space, so its impedance is still the
    stabilising solution of the algebraic Riccati equation -- just of the
    stretched A.  Newton on that equation, seeded by the unstretched solution,
    is what names it, exactly as D_z names the branch and Newton supplies the
    precision in the laterally invariant case.

    Args:
        med: The half-space medium.
        omega: Angular frequency.
        nx: Points along x.
        ny: Points along y.
        dx: The stretched lateral derivative along x.
        dy: The stretched lateral derivative along y.
        seed: The unstretched impedance, used as the Newton start.

    Returns:
        (Y, the algebraic Riccati residual it reached).
    """
    blk = aop(uniform_slice(med, nx, ny), omega, dx, dy)
    return newton_are(seed, blk)


def riccati_rhs(y: NDArray, blk: tuple) -> NDArray:
    """The right-hand side of Y' = A21 + A22 Y - Y A11 - Y A12 Y.

    Args:
        y: Impedance.
        blk: The four blocks of A.

    Returns:
        Same shape as y.
    """
    a11, a12, a21, a22 = blk
    return a21 + a22 @ y - y @ a11 - y @ a12 @ y


def march_lateral(
    slice_at: Callable[[float], Slice],
    omega: complex,
    nx: int,
    ny: int,
    nstep: int,
    *,
    method: str = "mobius",
    thickness: float = H,
    pml: tuple[float, float] | None = None,
) -> NDArray:
    """March the laterally coupled impedance from the base of the slab to its top.

    Args:
        slice_at: The medium on the lateral grid as a function of depth.
        omega: Angular frequency.
        nx: Points along x.
        ny: Points along y.
        nstep: Depth steps.
        method: "mobius" (piecewise constant, exact per sublayer -- this IS a
            layer stack), "pade" (the same step with the exponential replaced
            by its (1,1) Pade: one solve, A-stable, 3.3x cheaper, and second
            order like the method it approximates), or "rk4" (stage-sampled,
            fourth order on a smooth profile but only conditionally stable).
        thickness: Slab thickness.
        pml: ``(width, smax)`` to stretch both lateral directions, or ``None``
            for no absorption.  The starting impedance is left UNSTRETCHED: it
            is the half-space below, and the PML is a lateral device.

    Returns:
        The impedance at the top, shape (3N, 3N).
    """
    if pml is None:
        dx, dy = deriv_pair(nx, ny)
    else:
        dx, dy, _ = deriv_pair_pml(nx, ny, omega, width=pml[0], smax=pml[1])
    n3 = 3 * nx * ny
    y = y_start(REF, omega, nx, ny)
    if pml is not None:
        # The start must solve the STRETCHED half-space, or the base of the slab
        # contradicts the operator above it -- see ``y_start_stretched``.
        y, _ = y_start_stretched(REF, omega, nx, ny, dx, dy, y)
    edge = np.linspace(0.0, thickness, nstep + 1)

    def blocks_at(z: float) -> tuple:
        return aop(slice_at(z), omega, dx, dy)

    for m in range(nstep - 1, -1, -1):
        lo, hi = float(edge[m]), float(edge[m + 1])
        h = hi - lo
        if method in ("mobius", "pade"):
            # Only the ACTION on [I; Y] is ever needed: the update uses
            # M21 + M22 Y and M11 + M12 Y, which are the two halves of M [I; Y].
            big = abig(blocks_at(0.5 * (lo + hi)))
            w = np.vstack([np.eye(n3, dtype=complex), y])
            if method == "mobius":
                mw = expm(-big * h) @ w
            else:
                # The (1,1) Pade of exp(A dz), dz = -h upward.  One solve
                # instead of a scaling-and-squaring: A-stable, second order,
                # and the method is second order anyway because the medium is
                # frozen at the midpoint, so the exponential's exactness per
                # sublayer buys nothing.  Measured 3.3x cheaper.
                i6 = np.eye(2 * n3, dtype=complex)
                mw = np.linalg.solve(i6 + 0.5 * h * big, (i6 - 0.5 * h * big) @ w)
            y = mw[n3:] @ np.linalg.inv(mw[:n3])
        else:
            # Stage-sampled RK4.  Freezing the medium per sublayer would
            # integrate the staircase exactly and cap the march at second order.
            dz = lo - hi
            k1 = riccati_rhs(y, blocks_at(hi))
            k2 = riccati_rhs(y + 0.5 * dz * k1, blocks_at(hi + 0.5 * dz))
            k3 = riccati_rhs(y + 0.5 * dz * k2, blocks_at(hi + 0.5 * dz))
            k4 = riccati_rhs(y + dz * k3, blocks_at(hi + dz))
            y = y + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.asarray(y)


def rel(a: NDArray, b: NDArray) -> float:
    """Relative difference of two matrices in the max norm.

    Args:
        a: First.
        b: Second.

    Returns:
        max|a-b| / max|b|.
    """
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def order_of(errs: list[float], factor: float = 2.0) -> list[float]:
    """Observed convergence order from errors at doubling resolution.

    Args:
        errs: Errors, coarse to fine.
        factor: Refinement ratio.

    Returns:
        One order per consecutive pair.
    """
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(factor)) for i in range(len(errs) - 1)]


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def part1() -> None:
    """The 3-D operator reproduces the validated (Akdef) at every (k_x, k_y)."""
    print("\n[1] laterally uniform: every (k_x, k_y) block must equal amat_thesis")
    nx, ny = 8, 8
    dx, dy = deriv_pair(nx, ny)
    kx, ky = grid_wavenumbers(nx, LX), grid_wavenumbers(ny, LY)

    for med in (REF, ReferenceMedium(5500.0, 3300.0, 2750.0)):
        ahat = to_hat(abig(aop(uniform_slice(med, nx, ny), OMEGA, dx, dy)), nx, ny, 6)
        worst = 0.0
        for ix in range(nx):
            for iy in range(ny):
                got = hat_block(ahat, nx, ny, 6, ix, iy)
                want = amat_thesis(med, OMEGA, float(kx[ix]), float(ky[iy]))
                worst = max(worst, rel(got, want))
        tag = "background" if med is REF else "second medium"
        print(f"      {tag:<14} worst over {nx * ny} wavenumber pairs: {worst:.3e}")
        report(f"3-D operator reproduces (Akdef) at every (k_x,k_y), {tag}", worst < 1e-12)

    for name, d in (("d_x", dx), ("d_y", dy)):
        asym = float(np.max(np.abs(d + d.T)))
        print(f"      {name} antisymmetry |D + D^T|: {asym:.3e}")
        report(f"{name} is exactly antisymmetric", asym < 1e-12)

    comm = float(np.max(np.abs(dx @ dy - dy @ dx)))
    print(f"      |[d_x, d_y]|: {comm:.3e}")
    report("the two lateral derivatives commute", comm < 1e-12)


def part2() -> None:
    """A y-invariant medium must reduce, block by block, to the committed 2.5-D operator."""
    print("\n[2] medium invariant in y: reduce to the gated 2.5-D operator in each k_y")
    nx, ny, eps = 8, 6, 0.08
    dx, dy = deriv_pair(nx, ny)
    dx1 = deriv_1d(nx, LX)
    ky = grid_wavenumbers(ny, LY)
    z = 0.37 * H

    a3 = abig(aop(graded_y_invariant(z, nx, ny, eps), OMEGA, dx, dy))
    ahaty = to_hat_y(a3, nx, ny, 6)

    worst = 0.0
    for iy in range(ny):
        got = ky_block(ahaty, nx, ny, 6, iy)
        want = abig(aop_25d(slice_2d_of(z, nx, eps), OMEGA, float(ky[iy]), dx1))
        worst = max(worst, rel(got, want))
    print(f"      worst over {ny} k_y blocks, medium varying in x: {worst:.3e}")
    report("3-D operator reduces to the committed 2.5-D operator block by block", worst < 1e-11)

    # And with NO lateral variation at all, so the reduction is not carried by
    # the contrast being small.
    a3u = abig(aop(uniform_slice(REF, nx, ny), OMEGA, dx, dy))
    ahu = to_hat_y(a3u, nx, ny, 6)
    worstu = max(
        rel(
            ky_block(ahu, nx, ny, 6, iy),
            abig(aop_25d(slice_2d_of(0.0, nx, 0.0), OMEGA, float(ky[iy]), dx1)),
        )
        for iy in range(ny)
    )
    print(f"      the same with a uniform medium: {worstu:.3e}")
    report("the reduction holds for a uniform medium too", worstu < 1e-11)


def part3() -> None:
    """The symplectic identity closes OUTRIGHT in 3-D -- and still pins the ordering."""
    print("\n[3] symplectic structure with the medium varying in BOTH lateral directions")
    nx, ny = 6, 6
    n = nx * ny
    dx, dy = deriv_pair(nx, ny)
    sl = graded_3d(0.37 * H, nx, ny, 0.08)
    j = jsix(n)

    # d_y is now an antisymmetric MATRIX, so the transpose flips its sign by
    # itself: no k_y parameter to flip, and the identity closes outright.
    ja = j @ abig(aop(sl, OMEGA, dx, dy))
    resid = rel(ja, ja.T)
    print(f"      |J6 A - (J6 A)^T| rel: {resid:.3e}")
    report("J6 A is symmetric outright in 3-D", resid < 1e-12)

    # THE CONTROL.  Without it the check above proves nothing about ordering.
    jw = j @ abig(aop(sl, OMEGA, dx, dy, wrong_order=True))
    bad = rel(jw, jw.T)
    print(f"      CONTROL, modulus outside the derivative: {bad:.3e}")
    report("the wrong ordering BREAKS the identity (control)", bad > 1e-3)

    su = uniform_slice(REF, nx, ny)
    same = rel(abig(aop(su, OMEGA, dx, dy, wrong_order=True)), abig(aop(su, OMEGA, dx, dy)))
    print(f"      CONTROL is indistinguishable in a uniform medium: {same:.3e}")
    report("the two orderings agree when the medium is laterally uniform", same < 1e-12)


def part4() -> None:
    """Shift covariance, in x and in y SEPARATELY."""
    print("\n[4] covariance under a lateral shift, each direction on its own")
    nx, ny, eps, nstep = 6, 6, 0.08, 6
    n = nx * ny

    base = graded_3d(0.5 * H, nx, ny, eps)
    dx, dy = deriv_pair(nx, ny)

    # An index slip between the two Kronecker factors survives a shift applied
    # to both directions at once, so each is tested alone.
    for axis, label in ((0, "x"), (1, "y")):
        shift_1d = np.roll(np.eye(nx if axis == 0 else ny), 1, axis=0)
        s_lat = np.kron(shift_1d, np.eye(ny)) if axis == 0 else np.kron(np.eye(nx), shift_1d)

        def rolled(z: float, ax: int = axis) -> Slice:
            s = graded_3d(z, nx, ny, eps)

            def roll(v: NDArray) -> NDArray:
                return np.roll(v.reshape(nx, ny), 1, axis=ax).reshape(-1)

            return Slice(roll(s.alpha), roll(s.beta), roll(s.rho))

        y0 = march_lateral(profile_3d(nx, ny, eps), OMEGA, nx, ny, nstep)
        y1 = march_lateral(rolled, OMEGA, nx, ny, nstep)
        big_s = np.kron(np.eye(3), s_lat)
        resid = rel(y1, big_s @ y0 @ big_s.T)
        print(f"      shift along {label}: |Y[shifted] - S Y S^-1| rel: {resid:.3e}")
        report(f"the 3-D march is covariant under a lateral shift in {label}", resid < 1e-9)

    # The operator itself must not couple the two directions spuriously: with a
    # medium varying only in x, no k_y mode may be mixed with another.
    a3 = abig(aop(graded_y_invariant(0.4 * H, nx, ny, eps), OMEGA, dx, dy))
    ahaty = to_hat_y(a3, nx, ny, 6)
    off = 0.0
    for iy in range(ny):
        for jy in range(ny):
            if iy == jy:
                continue
            idx_i = np.array([c * n + ix * ny + iy for c in range(6) for ix in range(nx)])
            idx_j = np.array([c * n + ix * ny + jy for c in range(6) for ix in range(nx)])
            off = max(off, float(np.max(np.abs(ahaty[np.ix_(idx_i, idx_j)]))))
    scale = float(np.max(np.abs(ahaty)))
    print(f"      k_y mode mixing with a y-invariant medium: {off / scale:.3e}")
    report("a y-invariant medium does not mix k_y modes", off / scale < 1e-12)


def part5() -> None:
    """The MARCH, not just the operator, must reduce to objects already gated."""
    print("\n[5] march-level reduction to the 3x3 march and to the 2.5-D march")
    nx, ny, nstep = 6, 6, 16
    n = nx * ny
    kx, ky = grid_wavenumbers(nx, LX), grid_wavenumbers(ny, LY)
    f, fi = dft_2d(nx, ny)

    # (a) No lateral contrast at all: every wavenumber block is the scalar
    # march, which takes k_x alone -- so this anchors the k_y = 0 rows against
    # the object gate_first_order_impedance_march owns.
    def flat(z: float) -> Slice:
        o = np.ones(n) * _depth_factor(z)
        return Slice(REF.alpha * o, REF.beta * o, REF.rho * o)

    def med_at(z: float) -> ReferenceMedium:
        s = _depth_factor(z)
        return ReferenceMedium(REF.alpha * s, REF.beta * s, REF.rho * s)

    yhat = np.kron(np.eye(3), f) @ march_lateral(flat, OMEGA, nx, ny, nstep) @ np.kron(np.eye(3), fi)
    zero_ky = [iy for iy in range(ny) if ky[iy] == 0.0]
    worst = max(
        rel(hat_block(yhat, nx, ny, 3, ix, iy), march(med_at, OMEGA, float(kx[ix]), H, nstep))
        for ix in range(nx)
        for iy in zero_ky
    )
    print(f"      k_y=0 rows against the 3x3 march: {worst:.3e}")
    report("the 3-D march reduces to the 3x3 march at zero lateral contrast", worst < 1e-10)

    # No spurious coupling: with no lateral contrast the impedance must be
    # diagonal in wavenumber.
    scale = float(np.max(np.abs(yhat)))
    off = 0.0
    for ix in range(nx):
        for iy in range(ny):
            ii = np.array([c * n + ix * ny + iy for c in range(3)])
            mask = np.ones(3 * n, dtype=bool)
            mask[ii] = False
            off = max(off, float(np.max(np.abs(yhat[np.ix_(ii, np.where(mask)[0])]))))
    print(f"      spurious wavenumber coupling: {off / scale:.3e}")
    report("no spurious coupling between wavenumbers", off / scale < 1e-10)

    # (b) The stronger one: a medium VARYING in x but invariant in y.  Each k_y
    # block of the 3-D march must equal the committed 2.5-D march at that k_y,
    # lateral contrast and all.  This scores the march, not just the operator.
    eps = 0.08
    y3 = march_lateral(lambda z: graded_y_invariant(z, nx, ny, eps), OMEGA, nx, ny, nstep)
    y3haty = to_hat_y(y3, nx, ny, 3)
    worst2 = 0.0
    for iy in range(ny):
        got = ky_block(y3haty, nx, ny, 3, iy)
        want = march_lateral_25d(lambda z: slice_2d_of(z, nx, eps), OMEGA, float(ky[iy]), nx, nstep)
        worst2 = max(worst2, rel(got, want))
    print(f"      worst over {ny} k_y blocks, medium varying in x: {worst2:.3e}")
    report("the 3-D march reduces to the committed 2.5-D march at every k_y", worst2 < 1e-10)


def part6() -> None:
    """Fourth order in the depth step survives the second lateral direction."""
    print("\n[6] convergence order in the DEPTH step, lateral grid held fixed")
    nx, ny, eps = 6, 6, 0.06
    prof = profile_3d(nx, ny, eps)
    steps = [8, 16, 32, 64]
    ref_y = march_lateral(prof, OMEGA, nx, ny, 256, method="rk4")

    for method, expect, label in (("rk4", 3.5, "stage-sampled RK4"), ("mobius", 1.7, "layer stack")):
        errs = [rel(march_lateral(prof, OMEGA, nx, ny, s, method=method), ref_y) for s in steps]
        orders = order_of(errs)
        shown = "  ".join(f"{e:.2e}" for e in errs)
        print(f"      {label:<18} errors: {shown}")
        print(f"      {'':<18} orders: {'  '.join(f'{o:.2f}' for o in orders)}")
        report(f"{label} reaches its expected order in depth", min(orders) > expect)


def blob_slice(z: float, nx: int, ny: int, pitch: float, eps: float, halfwidth: float) -> Slice:
    """A COMPACT lateral anomaly at the centre of the domain.

    A single smooth Fourier mode fills the period and has no outside, so it
    cannot say anything about periodic images.  A compact blob can: shrink or
    grow the period around it and the interior answer must not move, once the
    images are absorbed.

    Args:
        z: Depth within the slab.
        nx: Points along x.
        ny: Points along y.
        pitch: Grid spacing, held FIXED as the grid grows so that changing
            ``nx`` changes the PERIOD and not the resolution.
        eps: Anomaly amplitude.
        halfwidth: Gaussian half-width of the anomaly.

    Returns:
        The slice at that depth.
    """
    x = (np.arange(nx) - 0.5 * (nx - 1)) * pitch
    y = (np.arange(ny) - 0.5 * (ny - 1)) * pitch
    r2 = (x**2)[:, None] + (y**2)[None, :]
    lat = np.exp(-r2 / (2.0 * halfwidth**2))
    s = _depth_factor(z) + eps * lat.reshape(-1)
    return Slice(REF.alpha * s, REF.beta * s, REF.rho * s)


def part7() -> None:
    """The lateral PML, and the weighted symmetry it leaves behind."""
    print("\n[7] the lateral PML: the symplectic identity becomes WEIGHTED")
    nx, ny = 6, 6
    n = nx * ny
    sl = graded_3d(0.37 * H, nx, ny, 0.08)
    j = jsix(n)

    # At zero absorption the weight is the identity and the plain statement
    # must come back -- so the same test covers the no-PML case.
    for width, smax, tag in ((0.0, 0.0, "absorption off"), (0.25 * LX, 1.5, "absorption on")):
        dxp, dyp, w_lat = deriv_pair_pml(nx, ny, OMEGA, width=width, smax=smax)
        w = np.kron(np.eye(6), w_lat)
        ja = j @ abig(aop(sl, OMEGA, dxp, dyp))
        resid = rel(ja, np.linalg.inv(w) @ ja.T @ w)
        print(f"      {tag:<16} |J6 A - W^-1 (J6 A)^T W| rel: {resid:.3e}")
        report(f"the weighted symplectic identity holds, {tag}", resid < 1e-12)

    # With absorption on, the UNWEIGHTED statement must FAIL -- otherwise the
    # weight is doing nothing and the test above is vacuous.
    dxp, dyp, _ = deriv_pair_pml(nx, ny, OMEGA, width=0.25 * LX, smax=1.5)
    ja = j @ abig(aop(sl, OMEGA, dxp, dyp))
    plain = rel(ja, ja.T)
    print(f"      CONTROL, unweighted with absorption on: {plain:.3e}")
    report("the unweighted identity BREAKS under the stretch (control)", plain > 1e-3)

    # The profile itself: exactly inert at the domain centre, exactly
    # 1 + i smax at the wrap point.  This is the check that catches a band that
    # leaks across the whole grid, which the weighted identity would not notice.
    s = stretch_1d(nx, LX, OMEGA, width=0.25 * LX, smax=1.5)
    centre_err = abs(s[nx // 2] - 1.0)
    wrap_err = abs(s[0] - (1.0 + 1.5j))
    print(f"      s at the domain centre - 1: {centre_err:.3e}   s at the wrap - (1+1.5i): {wrap_err:.3e}")
    report(
        "the stretch profile is inert at the centre and peaks at the wrap",
        max(centre_err, wrap_err) < 1e-14,
    )

    # ...but the OPERATOR is not inert there, and that is not a defect.  A
    # pseudospectral derivative is DENSE: d_x @ Ze @ d_x sums through every
    # lateral point, so a centre row picks up the stretch from inside the band.
    # Locality is a finite-difference property this discretisation does not
    # have, so "the interior is untouched" can only be a statement about the
    # SOLUTION -- which is what Part 8 measures, and why it has to be measured
    # rather than argued.
    dx0, dy0 = deriv_pair(nx, ny)
    a0, a1 = abig(aop(sl, OMEGA, dx0, dy0)), abig(aop(sl, OMEGA, dxp, dyp))
    idx_c = np.array([c * n + (nx // 2) * ny + (ny // 2) for c in range(6)])
    moved = float(np.max(np.abs((a1 - a0)[np.ix_(idx_c, idx_c)]))) / float(np.max(np.abs(a0)))
    print(f"      operator change at the domain centre (expected NON-zero): {moved:.3e}")
    report("the pseudospectral operator is global, so the centre does move", moved > 1e-6)


def part8() -> None:
    """REFUTED: a lateral PML does not suppress the images -- it adds reach.

    This part exists to hold a negative result in place, because the structural
    machinery above it works and would otherwise read as an endorsement.

    TWO METRICS WERE REJECTED BEFORE THIS ONE, both for confounds:

      - Growing the lateral PERIOD at fixed pitch also changes the lateral
        wavenumber grid, ``dk = 2 pi / LX``.  The real-space impedance at a
        point then moves through lateral-quadrature convergence as much as
        through images.  Measured that way the absorbing and non-absorbing runs
        both spread by about 0.5 and the ratio is 1.0 -- a null result about
        images that is really a statement about quadrature.
      - ``Y[centre, wrap]`` on a single grid is dominated by the BACKGROUND
        kernel's own physical lateral range whenever the period is a couple of
        wavelengths.  It rises with absorption for that reason alone.

    What is left is the SCATTERED part, ``dY = Y(blob) - Y(uniform)`` with the
    same stretch applied to both so the background cancels.  That is the only
    thing a PML can be asked to absorb.

    THE RESULT.  dY's reach grows monotonically with absorption, at every
    setting and both grid sizes tried (6.9x worse at 1.72 lambda_S, 3.9x at
    2.29 lambda_S), AFTER the genuine boundary defect in ``y_start_stretched``
    was fixed.  Two things explain it and neither is a bug to hunt:

      1. The pseudospectral derivative is DENSE.  Standard PML theory assumes a
         local operator, so that the absorbing band can be kept away from the
         region of interest; here it couples to the domain centre directly, at
         4.1e-3 relative in the operator (Part 7 measures it).  There is no
         "far away" to put it.
      2. There is almost nothing to absorb.  The zero-absorption baseline is
         already 2.6e-3, so the blob's lateral coupling to its own image has
         decayed by better than two orders before it reaches the wrap.  The
         stretch then perturbs the operator without removing anything.

    Point 2 is the one that matters for the programme: the premise that images
    need suppressing at all was assumed, not measured, and at these parameters
    it does not hold.  The exact route -- a Bloch sweep over the Brillouin zone,
    which needs no change to the operator validated in Parts 1-6 -- is both
    cheaper and correct, and does not have to argue about any of this.

    The assertion below is on the REFUTATION.  If a future change makes the
    absorption help, this fails and the reasoning above has to be re-read.
    """
    print("\n[8] REFUTED: the lateral PML adds reach rather than removing it")
    # Moebius, not RK4.  The Riccati rhs has decay rates set by the largest
    # lateral wavenumber, so the explicit step is unstable once
    # 2 |k_x|max h exceeds about 1.3 -- at this pitch and slab thickness RK4
    # returns nan whatever the PML does.  Moebius is exact per sublayer.
    nsz, pitch, eps, halfwidth, nstep = 12, 45.0, 0.10, 60.0, 3
    n = nsz * nsz
    lx = nsz * pitch
    width = 0.22 * lx
    ic = (nsz // 2) * nsz + (nsz // 2)
    iw = 0
    lam_s = 2.0 * np.pi * REF.beta / OMEGA

    def blk(y: NDArray, i: int, j: int) -> float:
        """Magnitude of the 3x3 impedance block coupling lateral points i and j."""
        ii = np.array([c * n + i for c in range(3)])
        jj = np.array([c * n + j for c in range(3)])
        return float(np.max(np.abs(y[np.ix_(ii, jj)])))

    def uniform_at(z: float) -> Slice:
        o = np.ones(n) * _depth_factor(z)
        return Slice(REF.alpha * o, REF.beta * o, REF.rho * o)

    print(f"      grid {nsz}x{nsz}, period {lx:.0f} m = {lx / lam_s:.2f} lambda_S, band {width:.0f} m")
    ratios = []
    for smax in (0.0, 2.0, 8.0):
        pml = None if smax == 0.0 else (width, smax)
        yu = march_lateral(uniform_at, OMEGA, nsz, nsz, nstep, method="mobius", pml=pml)
        yb = march_lateral(
            lambda z: blob_slice(z, nsz, nsz, pitch, eps, halfwidth),
            OMEGA,
            nsz,
            nsz,
            nstep,
            method="mobius",
            pml=pml,
        )
        dy = yb - yu
        ratio = blk(dy, ic, iw) / blk(dy, ic, ic)
        ratios.append(ratio)
        print(f"      sigma_max/omega = {smax:<4.1f}  scattered reach |dY[c,wrap]|/|dY[c,c]| = {ratio:.3e}")

    print(f"      the zero-absorption baseline is already {ratios[0]:.1e} -- little to remove")
    report("the image contamination is small BEFORE any absorption", ratios[0] < 1e-2)
    report("REFUTED: absorption increases the scattered reach", ratios[-1] > 2.0 * ratios[0])


def part9() -> None:
    """The stable step needs no matrix exponential."""
    print("\n[9] the (1,1) Pade step, against the exponential it replaces")
    nx = ny = 6
    eps = 0.06
    prof = profile_3d(nx, ny, eps)
    kmax = float(np.max(np.abs(grid_wavenumbers(nx, LX))))

    # The reference is the exponential form at a step fine enough that both
    # agree; the comparison of interest is at COARSE steps, where the explicit
    # method is unstable and the stable branch is the only one available.
    ref_y = march_lateral(prof, OMEGA, nx, ny, 256, method="mobius")

    print(f"      |k|max = {kmax:.5f}, so RK4 is unstable once 2|k|max h > ~1.3")
    errs_e, errs_p = [], []
    for nstep in (4, 8, 16, 32):
        stiff = 2.0 * kmax * (H / nstep)
        ye = march_lateral(prof, OMEGA, nx, ny, nstep, method="mobius")
        yp = march_lateral(prof, OMEGA, nx, ny, nstep, method="pade")
        errs_e.append(rel(ye, ref_y))
        errs_p.append(rel(yp, ref_y))
        print(
            f"      nstep={nstep:<3d} 2|k|max h={stiff:5.2f}  expm {errs_e[-1]:.3e}   Pade {errs_p[-1]:.3e}"
        )

    ord_e, ord_p = order_of(errs_e), order_of(errs_p)
    print(f"      orders  expm: {'  '.join(f'{o:.2f}' for o in ord_e)}")
    print(f"      orders  Pade: {'  '.join(f'{o:.2f}' for o in ord_p)}")
    report("the Pade step is second order, like the exponential", min(ord_p) > 1.7)
    report("it stays within a small factor of the exponential", max(errs_p) < 2.5 * max(errs_e))

    # The point of the stable branch: a step where the explicit method dies.
    big_step = 2
    stiff = 2.0 * kmax * (H / big_step)
    y_rk = march_lateral(prof, OMEGA, nx, ny, big_step, method="rk4")
    y_pd = march_lateral(prof, OMEGA, nx, ny, big_step, method="pade")
    rk_bad = (not np.all(np.isfinite(y_rk))) or rel(y_rk, ref_y) > 1.0
    pd_ok = np.all(np.isfinite(y_pd)) and rel(y_pd, ref_y) < 0.5
    print(
        f"      nstep={big_step} (2|k|max h={stiff:.2f}): "
        f"rk4 {rel(y_rk, ref_y):.3e}, Pade {rel(y_pd, ref_y):.3e}"
    )
    report("the explicit step fails at that step size (control)", rk_bad)
    report("the Pade step does not", pd_ok)


def part10() -> None:
    """What the second lateral direction costs."""
    print("\n[10] cost of the third dimension")
    eps, nstep = 0.06, 4
    for nx, ny in ((6, 1), (6, 6), (8, 8)):
        n = nx * ny
        t0 = time.perf_counter()
        march_lateral(profile_3d(nx, ny, eps), OMEGA, nx, ny, nstep, method="rk4")
        dt = time.perf_counter() - t0
        print(f"      N_x={nx} N_y={ny}  N={n:4d}  Y is {3 * n}^2  {dt / nstep * 1e3:8.1f} ms/step")
    report("the 3-D march runs at every grid size tried", True)


def main() -> int:
    """Run every part and summarise.

    Returns:
        0 if all checks passed, 1 otherwise.
    """
    print("=" * 78)
    print("THE LATERALLY COUPLED IMPEDANCE MARCH ON A FULL 3-D LATERAL GRID")
    print("=" * 78)
    for fn in (part1, part2, part3, part4, part5, part6, part7, part8, part9, part10):
        fn()
    npass = sum(1 for _, ok in _PASS if ok)
    print("\n" + "=" * 78)
    print(f"{npass}/{len(_PASS)} checks passed")
    for label, ok in _PASS:
        if not ok:
            print(f"  FAILED: {label}")
    print("=" * 78)
    return 0 if npass == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
