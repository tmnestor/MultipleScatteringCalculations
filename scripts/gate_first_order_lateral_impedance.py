#!/usr/bin/env python3
"""The impedance march with the lateral coupling INSIDE the marched variable.

ANCHOR: Nestor (1996) Ch.2 (Akdef); Ch.5 (CBPdef0), (dCBC), (dCBCrho),
Algorithms 5.1-5.3.

WHAT THIS IS, AND HOW IT DEPARTS FROM THE THESIS
------------------------------------------------
The thesis solves the laterally varying problem by eq. (CBPdef0),

    P_{r,s} = P^z_{r,s} + P^x_{r,s}
            = -i Theta^z_in Q^partial Theta^z_ou
              + diag[P^x_{r,s,1}, ..., P^x_{r,s,L}]

acting on an 8 L N_x vector.  There the vertical operator Q^partial is DIAGONAL
in lateral wavenumber -- Algorithm 5.3 is a mode-domain R/T recursion at fixed
k_x -- and the lateral coupling lives entirely in the site-diagonal contrast
Delta C_eff (dCBC, dCBCrho) and in the sideways sweeps P^x of Algorithm 5.2.
The two are then reconciled by an OUTER iteration on (I - P Delta C_eff).

This gate does the opposite, which is the only part that is new.  The lateral
grid is carried INSIDE the marched variable: Y becomes a matrix on
(3 components) x (N_x lateral points), and the lateral coupling is integrated
along with depth.  There is no outer iteration for it at all.

That is worth building only if it survives contact with the existing evidence,
so every check below is either an identity or a comparison against an object
this file does not own.

THE OPERATOR IS DERIVED, NOT TRANSCRIBED
----------------------------------------
A laterally varying A cannot be read off (Akdef), because (Akdef) is a symbol in
k_x and says nothing about where a spatially varying modulus sits relative to a
derivative.  Momentum balance does say.  Writing tau_3 = (T_zz, T_xz, T_yz) and
using T_xx = zeta d_x u_x + chi d_y u_y + gamma T_zz, T_xy = mu(d_x u_y +
d_y u_x), the fifth row of the system is

    d_3 T_xz = -rho w^2 u_x - d_x[zeta d_x u_x + chi d_y u_y + gamma T_zz]
                            - d_y[mu(d_x u_y + d_y u_x)]

with the derivative OUTSIDE the modulus.  That ordering is forced by
d_j T_ij, not chosen, and it is what makes the discretised operator symplectic.
Part 2 measures the difference against the plausible wrong ordering
(mu d_x^2 in place of d_x mu d_x), which agrees in a uniform medium and is a
different operator as soon as the medium varies.

Every entry is nevertheless checked rather than trusted: Part 1 transforms the
operator built here to the lateral wavenumber domain in a laterally UNIFORM
medium, where it must reproduce ``amat_thesis`` -- already validated 13/13
against (Akdef) -- block by block, at the arithmetic floor.

GEOMETRY
--------
2.5-D, matching the thesis: heterogeneity in (x, z), the third direction
carried by the parameter k_y, N_x lateral points on a period LX.  The lateral
derivative is spectral, so it is exactly antisymmetric and the symplectic
identity is available as a test rather than as an approximation.

Units are SI throughout -- m, m/s, kg/m^3, rad/m, rad/s -- matching
``gate_first_order_impedance_march``, whose background, frequency, slab
thickness and starting impedance are reused so the reduction in Part 4 is an
identity and not a comparison of two conventions.

Run:  conda run -n seismic python scripts/gate_first_order_lateral_impedance.py
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
from scripts.gate_thesis_spectral import amat_thesis, dz_normalised  # noqa: E402

#: Background half-space below the slab, and the slab geometry.  Identical to
#: ``gate_first_order_impedance_march`` so Part 4 can be an identity.
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
H = 200.0

#: Lateral period.  At OMEGA the background has omega/alpha = 1.2e-2 and
#: omega/beta = 2.0e-2 per metre, and the grid wavenumbers are 2 pi n / LX =
#: 6.28e-3 n, so |n| <= 3 propagates and the rest is evanescent.  Both branches
#: are therefore exercised at N = 16 without choosing them by hand.
LX = 1000.0

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


def grid_wavenumbers(n: int) -> NDArray:
    """The lateral wavenumbers of the periodic grid, in FFT order.

    The Nyquist entry is set to zero for even n.  That is the standard choice
    and it is not cosmetic here: it is what makes the differentiation matrix
    exactly antisymmetric, and the symplectic identity of Part 2 is an identity
    only for an antisymmetric derivative.

    Args:
        n: Number of lateral points.

    Returns:
        Shape (n,), real.
    """
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=LX / n)
    if n % 2 == 0:
        k[n // 2] = 0.0
    return k


def deriv_matrix(n: int) -> NDArray:
    """Spectral d/dx on the periodic lateral grid.

    Args:
        n: Number of lateral points.

    Returns:
        Shape (n, n), real, antisymmetric.
    """
    k = grid_wavenumbers(n)
    d = np.fft.ifft(1j * k[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0)
    return np.real(d)


def dft_pair(n: int) -> tuple[NDArray, NDArray]:
    """The DFT matrix and its inverse, as matrices.

    Args:
        n: Number of lateral points.

    Returns:
        (F, F_inv) with ``F @ v == np.fft.fft(v)``.
    """
    return np.fft.fft(np.eye(n), axis=0), np.fft.ifft(np.eye(n), axis=0)


def to_hat(mat: NDArray, n: int, ncomp: int) -> NDArray:
    """Transform a spatial-grid operator to the lateral wavenumber domain.

    Args:
        mat: Shape (ncomp*n, ncomp*n), grid index fastest.
        n: Lateral points.
        ncomp: Components.

    Returns:
        Same shape, in the wavenumber domain.
    """
    f, fi = dft_pair(n)
    big_f, big_fi = np.kron(np.eye(ncomp), f), np.kron(np.eye(ncomp), fi)
    return np.asarray(big_f @ mat @ big_fi)


def from_hat(mat: NDArray, n: int, ncomp: int) -> NDArray:
    """Transform a wavenumber-domain operator back to the spatial grid.

    Args:
        mat: Shape (ncomp*n, ncomp*n).
        n: Lateral points.
        ncomp: Components.

    Returns:
        Same shape, on the spatial grid.
    """
    f, fi = dft_pair(n)
    big_f, big_fi = np.kron(np.eye(ncomp), f), np.kron(np.eye(ncomp), fi)
    return np.asarray(big_fi @ mat @ big_f)


def hat_block(mat_hat: NDArray, n: int, ncomp: int, idx: int) -> NDArray:
    """The ncomp x ncomp block of a wavenumber-domain operator at one k_x.

    Args:
        mat_hat: Wavenumber-domain operator.
        n: Lateral points.
        ncomp: Components.
        idx: Wavenumber index.

    Returns:
        Shape (ncomp, ncomp).
    """
    sel = idx + n * np.arange(ncomp)
    return np.asarray(mat_hat[np.ix_(sel, sel)])


def blocks_to_hat(blocks: NDArray, n: int) -> NDArray:
    """Scatter per-wavenumber 3x3 blocks into a (3n, 3n) wavenumber operator.

    Args:
        blocks: Shape (n, 3, 3).
        n: Lateral points.

    Returns:
        Shape (3n, 3n).
    """
    out = np.zeros((3 * n, 3 * n), dtype=np.complex128)
    for m in range(n):
        sel = m + n * np.arange(3)
        out[np.ix_(sel, sel)] = blocks[m]
    return out


# ---------------------------------------------------------------------------
# The medium on the lateral grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slice:
    """The medium along the lateral grid at one depth.

    Attributes:
        alpha: P speed at each lateral point, shape (n,).
        beta: S speed, shape (n,).
        rho: Density, shape (n,).
    """

    alpha: NDArray
    beta: NDArray
    rho: NDArray

    @property
    def mu(self) -> NDArray:
        """Shear modulus at each lateral point.

        Returns:
            Shape (n,).
        """
        return self.rho * self.beta**2

    @property
    def lam(self) -> NDArray:
        """First Lame parameter at each lateral point.

        Returns:
            Shape (n,).
        """
        return self.rho * self.alpha**2 - 2.0 * self.mu


def uniform_slice(med: ReferenceMedium, n: int) -> Slice:
    """A laterally uniform slice of a given medium.

    Args:
        med: The medium.
        n: Lateral points.

    Returns:
        The slice.
    """
    o = np.ones(n)
    return Slice(med.alpha * o, med.beta * o, med.rho * o)


def graded_2d(z: float, n: int, eps: float, mode: int = 2) -> Slice:
    """A slab smooth in BOTH depth and lateral position.

    The depth profile is the one ``gate_first_order_impedance_march`` grades
    with, so the fourth-order measurement of Part 6 is directly comparable with
    the laterally uncoupled one.  The lateral term is a single smooth Fourier
    mode of amplitude ``eps``: smooth, so the depth march's order is not capped
    by a lateral discontinuity, and single-mode, so the coupling it induces is
    to k_x +/- 2 pi mode / LX and can be looked at directly.

    Args:
        z: Depth within the slab.
        n: Lateral points.
        eps: Lateral contrast amplitude.
        mode: Lateral Fourier mode number.

    Returns:
        The slice at that depth.
    """
    x = np.arange(n) * (LX / n)
    s = 1.0 + 0.10 * np.sin(np.pi * z / H) + 0.04 * (z / H) + eps * np.cos(2.0 * np.pi * mode * x / LX)
    return Slice(REF.alpha * s, REF.beta * s, REF.rho * s)


def lateral_profile(n: int, eps: float) -> Callable[[float], Slice]:
    """The graded slab as a function of depth alone, at fixed grid and contrast.

    Binding the grid size and contrast here rather than in a lambda at the call
    site keeps each closure's capture explicit, which matters inside a loop over
    grid sizes.

    Args:
        n: Lateral points.
        eps: Lateral contrast amplitude.

    Returns:
        A callable of depth.
    """

    def at(z: float) -> Slice:
        return graded_2d(z, n, eps)

    return at


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------


def aop(
    sl: Slice, omega: complex, ky: float, dmat: NDArray, *, wrong_order: bool = False
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The four blocks of the laterally coupled A, each (3n, 3n).

    Derived from the elastodynamic equations rather than transcribed, as the
    module docstring sets out.  Every modulus is a multiplication operator --
    a diagonal matrix on the lateral grid -- and every d_x is ``dmat``.  The
    orderings that matter are ``dmat @ Ze @ dmat`` and ``dmat @ G``, both of
    which carry the derivative outside the modulus because they come from
    d_j T_ij.

    Args:
        sl: Medium on the lateral grid.
        omega: Angular frequency.
        ky: The 2.5-D lateral parameter.
        dmat: The lateral derivative matrix, shape (n, n).
        wrong_order: Build the plausible WRONG ordering instead, with the
            modulus outside the derivative.  Identical in a uniform medium; a
            different operator as soon as the medium varies.  Part 2's control.

    Returns:
        (A11, A12, A21, A22).
    """
    n = sl.alpha.size
    mu, lam = sl.mu, sl.lam
    kc = lam + 2.0 * mu
    gam, aa, bb = lam / kc, 1.0 / kc, 1.0 / mu
    zet, chi = 4.0 * mu * (lam + mu) / kc, 2.0 * mu * lam / kc
    rw2 = sl.rho * omega**2

    eye, zero = np.eye(n), np.zeros((n, n))
    g, a_d, b_d = np.diag(gam), np.diag(aa), np.diag(bb)
    ze, ch, mu_d = np.diag(zet), np.diag(chi), np.diag(mu)
    rw = np.diag(np.asarray(rw2, dtype=np.complex128))
    d = dmat
    iy = 1j * ky

    if wrong_order:
        # The modulus pulled outside every derivative pair.
        dzd, dmd, dchd_xy, dmud_yx = ze @ d @ d, mu_d @ d @ d, (ch + mu_d) @ d, (mu_d + ch) @ d
        dg = g @ d
    else:
        dzd, dmd = d @ ze @ d, d @ mu_d @ d
        dchd_xy, dmud_yx = d @ ch + mu_d @ d, d @ mu_d + ch @ d
        dg = d @ g

    a11 = np.block([[zero, -g @ d, -iy * g], [-d, zero, zero], [-iy * eye, zero, zero]])
    a12 = np.block([[a_d, zero, zero], [zero, b_d, zero], [zero, zero, b_d]])
    a21 = np.block(
        [
            [-rw, zero, zero],
            [zero, -rw - dzd + ky**2 * mu_d, -iy * dchd_xy],
            [zero, -iy * dmud_yx, -rw - dmd + ky**2 * ze],
        ]
    )
    a22 = np.block([[zero, -d, -iy * eye], [-dg, zero, zero], [-iy * g, zero, zero]])
    return a11, a12, a21, a22


def abig(blk: tuple) -> NDArray:
    """Assemble the four blocks into one operator.

    Args:
        blk: (A11, A12, A21, A22).

    Returns:
        Shape (6n, 6n).
    """
    a11, a12, a21, a22 = blk
    return np.block([[a11, a12], [a21, a22]])


def jsix(n: int) -> NDArray:
    """The symplectic form on the lateral grid, shape (6n, 6n).

    Args:
        n: Lateral points.

    Returns:
        Block [[0, I], [-I, 0]] with I of size 3n.
    """
    i3n = np.eye(3 * n)
    z = np.zeros((3 * n, 3 * n))
    return np.block([[z, i3n], [-i3n, z]])


# ---------------------------------------------------------------------------
# The march
# ---------------------------------------------------------------------------


def y_start(med: ReferenceMedium, omega: complex, n: int, ky: float) -> NDArray:
    """The downgoing impedance of the laterally uniform half-space below.

    The background below the slab is laterally uniform, so its impedance is
    diagonal in lateral wavenumber and each 3x3 block is the object
    ``gate_first_order_impedance`` already validated -- D_z names the branch,
    Newton supplies the precision.  Building the start this way rather than by
    a big Newton solve is what makes the reduction in Part 4 an identity.

    Args:
        med: The half-space medium.
        omega: Angular frequency.
        n: Lateral points.
        ky: The 2.5-D lateral parameter.

    Returns:
        Shape (3n, 3n), on the spatial grid.
    """
    kx = grid_wavenumbers(n)
    blocks = np.zeros((n, 3, 3), dtype=np.complex128)
    for m in range(n):
        if ky == 0.0:
            blocks[m] = y_downgoing(med, omega, float(kx[m]))
        else:
            dzm, _, _ = dz_normalised(med, omega, float(kx[m]), ky)
            dd = dzm[:, :3]
            y0 = dd[3:, :] @ np.linalg.inv(dd[:3, :])
            blocks[m] = newton_are(y0, ablocks(med, omega, float(kx[m]), ky))[0]
    return from_hat(blocks_to_hat(blocks, n), n, 3)


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
    ky: float,
    n: int,
    nstep: int,
    *,
    method: str = "mobius",
    thickness: float = H,
    wrong_order: bool = False,
) -> NDArray:
    """March the laterally coupled impedance from the base of the slab to its top.

    Args:
        slice_at: The medium on the lateral grid as a function of depth.
        omega: Angular frequency.
        ky: The 2.5-D lateral parameter.
        n: Lateral points.
        nstep: Depth steps.
        method: "mobius" (piecewise constant, exact per sublayer) or "rk4"
            (stage-sampled, fourth order on a smooth profile).
        thickness: Slab thickness.
        wrong_order: Pass the control ordering through to ``aop``.

    Returns:
        The impedance at the top, shape (3n, 3n).
    """
    dmat = deriv_matrix(n)
    y = y_start(REF, omega, n, ky)
    edge = np.linspace(0.0, thickness, nstep + 1)

    def blocks_at(z: float) -> tuple:
        return aop(slice_at(z), omega, ky, dmat, wrong_order=wrong_order)

    for m in range(nstep - 1, -1, -1):
        lo, hi = float(edge[m]), float(edge[m + 1])
        h = hi - lo
        if method == "mobius":
            # Piecewise constant, sampled at the midpoint: this IS a layer
            # stack, and Part 6 scores it as one.
            mexp = expm(-abig(blocks_at(0.5 * (lo + hi))) * h)
            num = mexp[3 * n :, : 3 * n] + mexp[3 * n :, 3 * n :] @ y
            den = mexp[: 3 * n, : 3 * n] + mexp[: 3 * n, 3 * n :] @ y
            y = num @ np.linalg.inv(den)
        else:
            # Stage-sampled RK4.  Freezing the medium per sublayer would
            # integrate the staircase exactly and cap the march at second
            # order; see the march gate's step_rk4 for the measurement.
            dz = lo - hi
            k1 = riccati_rhs(y, blocks_at(hi))
            k2 = riccati_rhs(y + 0.5 * dz * k1, blocks_at(hi + 0.5 * dz))
            k3 = riccati_rhs(y + 0.5 * dz * k2, blocks_at(hi + 0.5 * dz))
            k4 = riccati_rhs(y + dz * k3, blocks_at(hi + dz))
            y = y + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.asarray(y)


def march_linear(
    slice_at: Callable[[float], Slice],
    omega: complex,
    ky: float,
    n: int,
    nstep: int,
    *,
    thickness: float = H,
) -> tuple[NDArray, float]:
    """The same answer by integrating the LINEAR system and dividing at the end.

    Rather than marching the quadratic Riccati for a (3n, 3n) impedance, this
    marches the linear system for a (6n, 3n) fundamental block started at
    [I; Y0] and forms Y = T U^{-1} only at the top.  Two different ODEs, one
    answer -- so agreement tests the Riccati reduction itself, and the
    condition number of U at the top measures how far the linear route can be
    pushed before the growing evanescent branch destroys it.  That degradation
    is the whole reason the impedance is the marched variable.

    Args:
        slice_at: The medium on the lateral grid as a function of depth.
        omega: Angular frequency.
        ky: The 2.5-D lateral parameter.
        n: Lateral points.
        nstep: Depth steps.
        thickness: Slab thickness.

    Returns:
        (Y at the top, condition number of U at the top).
    """
    dmat = deriv_matrix(n)
    w = np.vstack([np.eye(3 * n, dtype=np.complex128), y_start(REF, omega, n, ky)])
    edge = np.linspace(0.0, thickness, nstep + 1)

    def amat_at(z: float) -> NDArray:
        return abig(aop(slice_at(z), omega, ky, dmat))

    for m in range(nstep - 1, -1, -1):
        lo, hi = float(edge[m]), float(edge[m + 1])
        dz = lo - hi
        k1 = amat_at(hi) @ w
        k2 = amat_at(hi + 0.5 * dz) @ (w + 0.5 * dz * k1)
        k3 = amat_at(hi + 0.5 * dz) @ (w + 0.5 * dz * k2)
        k4 = amat_at(hi + dz) @ (w + dz * k3)
        w = w + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    u, t = w[: 3 * n], w[3 * n :]
    return np.asarray(t @ np.linalg.inv(u)), float(np.linalg.cond(u))


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
    """Observed convergence order from a sequence of errors at doubling resolution.

    Args:
        errs: Errors, coarse to fine.
        factor: Refinement ratio.

    Returns:
        One order per consecutive pair.
    """
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(factor)) for i in range(len(errs) - 1)]


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def part1() -> None:
    """The operator reproduces the validated (Akdef) block by block."""
    print("\n[1] laterally uniform: every k_x block must equal amat_thesis")
    n = 16
    dmat = deriv_matrix(n)
    kx = grid_wavenumbers(n)
    for ky in (0.0, 0.004):
        sl = uniform_slice(REF, n)
        ahat = to_hat(abig(aop(sl, OMEGA, ky, dmat)), n, 6)
        worst = 0.0
        for m in range(n):
            got = hat_block(ahat, n, 6, m)
            want = amat_thesis(REF, OMEGA, float(kx[m]), ky)
            worst = max(worst, rel(got, want))
        print(f"      ky={ky:<6.4f}  worst block over {n} wavenumbers: {worst:.3e}")
        report(f"operator reproduces (Akdef) at every k_x, ky={ky}", worst < 1e-12)

    # A second medium, so the agreement is not an accident of one parameter set.
    other = ReferenceMedium(5500.0, 3300.0, 2750.0)
    ahat = to_hat(abig(aop(uniform_slice(other, n), OMEGA, 0.003, dmat)), n, 6)
    worst = max(
        rel(hat_block(ahat, n, 6, m), amat_thesis(other, OMEGA, float(kx[m]), 0.003)) for m in range(n)
    )
    print(f"      second medium, ky=0.0030: {worst:.3e}")
    report("operator reproduces (Akdef) for a second medium", worst < 1e-12)

    d = deriv_matrix(12)
    asym = float(np.max(np.abs(d + d.T)))
    print(f"      derivative matrix antisymmetry |D + D^T|: {asym:.3e}")
    report("the spectral derivative is exactly antisymmetric", asym < 1e-12)


def part2() -> None:
    """The symplectic structure survives lateral variation -- and pins the ordering."""
    print("\n[2] symplectic identity with the medium VARYING laterally")
    n = 12
    dmat = deriv_matrix(n)
    sl = graded_2d(0.37 * H, n, 0.08)
    j = jsix(n)

    # J6 A(ky) must equal [J6 A(-ky)]^T.  Transpose is in the lateral grid only,
    # so k_y -- a parameter, not a grid direction -- flips with it.  At ky = 0
    # this reduces to J6 A being symmetric outright.
    for ky in (0.0, 0.005):
        ja = j @ abig(aop(sl, OMEGA, ky, dmat))
        jam = j @ abig(aop(sl, OMEGA, -ky, dmat))
        resid = rel(ja, jam.T)
        print(f"      ky={ky:<6.4f}  |J6 A(ky) - [J6 A(-ky)]^T| rel: {resid:.3e}")
        report(f"J6 A symmetric under the lateral transpose, ky={ky}", resid < 1e-12)

    # THE CONTROL.  The wrong ordering is a different operator and must fail --
    # otherwise the check above proves nothing about where the modulus sits.
    jw = j @ abig(aop(sl, OMEGA, 0.0, dmat, wrong_order=True))
    bad = rel(jw, jw.T)
    print(f"      CONTROL, modulus outside the derivative: {bad:.3e}")
    report("the wrong ordering BREAKS the identity (control)", bad > 1e-3)

    # And the control must agree with the derived operator in a UNIFORM medium,
    # which is why no laterally invariant test could have caught it.
    su = uniform_slice(REF, n)
    same = rel(abig(aop(su, OMEGA, 0.0, dmat, wrong_order=True)), abig(aop(su, OMEGA, 0.0, dmat)))
    print(f"      CONTROL is indistinguishable in a uniform medium: {same:.3e}")
    report("the two orderings agree when the medium is laterally uniform", same < 1e-12)


def part3() -> None:
    """Structural identities the march must inherit."""
    print("\n[3] decoupling and shift covariance")
    n = 12
    dmat = deriv_matrix(n)
    sl = graded_2d(0.5 * H, n, 0.08)

    # At ky = 0, SH (u_y, T_yz) must close on itself however the medium varies
    # laterally.  Components are ordered (u_z, u_x, u_y, T_zz, T_xz, T_yz).
    a = abig(aop(sl, OMEGA, 0.0, dmat))
    sh = np.concatenate([2 * n + np.arange(n), 5 * n + np.arange(n)])
    psv = np.array([i for i in range(6 * n) if i not in set(sh.tolist())])
    leak = max(float(np.max(np.abs(a[np.ix_(psv, sh)]))), float(np.max(np.abs(a[np.ix_(sh, psv)]))))
    print(f"      P-SV <-> SH leakage at ky=0: {leak:.3e}")
    report("SH decouples from P-SV at ky=0 despite lateral variation", leak < 1e-14)

    # Shifting the medium one grid point must conjugate the marched impedance by
    # the shift.  This is an exact identity and it is what catches an index slip
    # between the medium array and the derivative matrix.
    def shifted(z: float) -> Slice:
        s = graded_2d(z, n, 0.08)
        return Slice(np.roll(s.alpha, 1), np.roll(s.beta, 1), np.roll(s.rho, 1))

    y0 = march_lateral(lambda z: graded_2d(z, n, 0.08), OMEGA, 0.0, n, 8)
    y1 = march_lateral(shifted, OMEGA, 0.0, n, 8)
    shift = np.kron(np.eye(3), np.roll(np.eye(n), 1, axis=0))
    resid = rel(y1, shift @ y0 @ shift.T)
    print(f"      |Y[shifted medium] - S Y S^-1| rel: {resid:.3e}")
    report("the march is covariant under a lateral shift", resid < 1e-10)


def part4() -> None:
    """Zero lateral contrast reduces EXACTLY to the validated 3x3 march."""
    print("\n[4] lateral contrast off: must reproduce the 14/14 march exactly")
    n = 16
    kx = grid_wavenumbers(n)
    for method, nstep in (("mobius", 12), ("rk4", 6)):
        big = march_lateral(lambda z: graded_2d(z, n, 0.0), OMEGA, 0.0, n, nstep, method=method)
        bhat = to_hat(big, n, 3)

        # Off-diagonal in k_x must vanish: with no lateral contrast nothing
        # couples wavenumbers, and a non-zero here would mean the operator was
        # manufacturing coupling out of the discretisation.
        mask = np.ones((3 * n, 3 * n), dtype=bool)
        for m in range(n):
            sel = m + n * np.arange(3)
            mask[np.ix_(sel, sel)] = False
        off = float(np.max(np.abs(bhat[mask]))) / float(np.max(np.abs(bhat)))
        print(f"      {method:<7} off-diagonal coupling in k_x: {off:.3e}")
        report(f"no spurious lateral coupling, {method}", off < 1e-12)

        worst = 0.0
        for m in range(n):
            got = hat_block(bhat, n, 3, m)
            want = march(lambda z: _scalar_graded(z), OMEGA, float(kx[m]), H, nstep, method=method)
            worst = max(worst, rel(got, want))
        print(f"      {method:<7} worst block vs the 3x3 march: {worst:.3e}")
        report(f"reduces to the validated march, {method}", worst < 1e-10)


def _scalar_graded(z: float) -> ReferenceMedium:
    """The laterally uniform depth profile, for the 3x3 comparison march.

    Args:
        z: Depth within the slab.

    Returns:
        The medium there.
    """
    s = 1.0 + 0.10 * np.sin(np.pi * z / H) + 0.04 * (z / H)
    return ReferenceMedium(REF.alpha * s, REF.beta * s, REF.rho * s)


def part5() -> None:
    """The Riccati march against an independent algorithm on the same operator."""
    print("\n[5] Riccati march vs marching the LINEAR system and dividing at the end")
    n = 12

    # Both routes are fourth order, so at any finite step count they differ BY
    # their own truncation error -- Part 6 measures RK4's at about 4e-7 near 64
    # steps.  Asking two fourth-order approximations to agree to a fixed small
    # number is asking them to agree better than either is accurate, and this
    # gate's first run duly failed its own bar at 5.3e-7.  The discriminating
    # test is the FALL: truncation falls at fourth order, whereas a genuine
    # error in the Riccati reduction would plateau at a fixed level.
    print("      the routes agree to their own truncation, which must FALL at 4th order:")
    errs = []
    for nstep in (24, 48, 96, 192):
        yr = march_lateral(lambda z: graded_2d(z, n, 0.08), OMEGA, 0.0, n, nstep, method="rk4")
        yl, cond = march_linear(lambda z: graded_2d(z, n, 0.08), OMEGA, 0.0, n, nstep)
        errs.append(rel(yr, yl))
        print(f"        {nstep:4d} steps   disagreement {errs[-1]:.3e}   cond(U) {cond:.2e}")
    ords = order_of(errs)
    print(f"        order:  {'  '.join(f'{o:.2f}' for o in ords)}")
    report(
        "two ODEs converge to one answer, at fourth order",
        3.4 < float(np.mean(ords[-2:])) < 4.6,
    )

    # How far the linear route can be pushed.  The impedance is bounded by
    # construction; U is not, and this measures the difference.  |Y| is printed
    # alongside so that the linear route's collapse cannot be mistaken for the
    # problem itself becoming hard.
    # The step size is held FIXED as the slab thickens, by scaling the step
    # count with the thickness.  Running all four thicknesses at one step count
    # confounds two different failures: this gate's first run did exactly that
    # and reported |Y| = nan at 2000 m, which is not the impedance failing but
    # the explicit integrator going unstable at h = 20.8 m -- measured directly
    # below.  Holding h fixed isolates the conditioning.
    print("      the linear route degrades as the slab thickens; the impedance does not:")
    broke = False
    norms = []
    for thick in (200.0, 600.0, 1200.0, 2000.0):
        nstep = int(round(96 * thick / 200.0))
        yr = march_lateral(lateral_profile(n, 0.08), OMEGA, 0.0, n, nstep, method="rk4", thickness=thick)
        yl, cond = march_linear(lateral_profile(n, 0.08), OMEGA, 0.0, n, nstep, thickness=thick)
        norms.append(float(np.linalg.norm(yr)))
        print(
            f"        thickness {thick:7.0f} m  ({nstep:4d} steps)   cond(U) {cond:.2e}"
            f"   disagreement {rel(yr, yl):.2e}   |Y| {norms[-1]:.4e}"
        )
        if cond > 1e12:
            broke = True
    report("cond(U) grows with thickness while the impedance march does not", broke)
    report("the impedance stays finite at every thickness", bool(np.all(np.isfinite(norms))))

    # THE STEP LIMIT, which is the price of the explicit integrator and which
    # the laterally coupled problem pays more heavily than the 3x3 one.  The
    # Riccati right-hand side has decay rates set by the vertical wavenumbers,
    # so the largest LATERAL wavenumber on the grid sets the largest rate, and
    # explicit RK4 is stable only while 2|k_x|_max h is below order one.  The
    # Mobius step is exact per sublayer and has no such limit.  This couples the
    # lateral mesh to the depth step: refining laterally raises |k_x|_max and
    # therefore forces a smaller depth step, which offsets part of the fourth
    # order economy of section "What the continuum march buys".
    print("      the explicit step limit, at 2000 m:")
    kmax = float(np.max(np.abs(grid_wavenumbers(n))))
    first_stable = None
    for nstep in (96, 128, 160, 192, 384):
        h = 2000.0 / nstep
        yr = march_lateral(lateral_profile(n, 0.08), OMEGA, 0.0, n, nstep, method="rk4", thickness=2000.0)
        ym = march_lateral(
            lateral_profile(n, 0.08), OMEGA, 0.0, n, nstep, method="mobius", thickness=2000.0
        )
        ok = bool(np.all(np.isfinite(yr)))
        if ok and first_stable is None:
            first_stable = 2.0 * kmax * h
        print(
            f"        {nstep:4d} steps  h {h:6.2f} m  2|k_x|h {2.0 * kmax * h:5.2f}"
            f"   rk4 {'stable' if ok else 'UNSTABLE':>8}   mobius"
            f" {'stable' if np.all(np.isfinite(ym)) else 'UNSTABLE':>8}"
        )
    print(f"        explicit RK4 becomes stable near 2|k_x|h = {first_stable:.2f}")
    report(
        "the explicit march has a step limit set by the largest lateral wavenumber",
        first_stable is not None and first_stable < 1.4,
    )
    report(
        "the Mobius step has no such limit",
        bool(
            np.all(
                np.isfinite(
                    march_lateral(
                        lateral_profile(n, 0.08),
                        OMEGA,
                        0.0,
                        n,
                        96,
                        method="mobius",
                        thickness=2000.0,
                    )
                )
            )
        ),
    )


def part6() -> None:
    """The fourth order survives the lateral coupling."""
    print("\n[6] depth convergence order, WITH lateral coupling on")
    n = 8
    eps = 0.08
    ref_y = march_lateral(lambda z: graded_2d(z, n, eps), OMEGA, 0.0, n, 3072, method="rk4")

    rows: dict[str, list[float]] = {}
    steps = [12, 24, 48, 96]
    for method in ("mobius", "rk4"):
        errs = [
            rel(march_lateral(lambda z: graded_2d(z, n, eps), OMEGA, 0.0, n, s, method=method), ref_y)
            for s in steps
        ]
        rows[method] = errs
        ords = order_of(errs)
        txt = "  ".join(f"{e:.2e}" for e in errs)
        otx = "  ".join(f"{o:.2f}" for o in ords)
        print(f"      {method:<7} err {txt}")
        print(f"      {method:<7} ord        {otx}")

    stair = float(np.mean(order_of(rows["mobius"])[-2:]))
    cont = float(np.mean(order_of(rows["rk4"])[-2:]))
    print(f"      staircase order {stair:.2f}   continuum order {cont:.2f}")
    report("the layer-stack march is second order", 1.7 < stair < 2.3)
    report("the continuum march is fourth order WITH lateral coupling", 3.6 < cont < 4.4)


def part7() -> None:
    """Lateral coupling is real, and the quadratic term is the cost."""
    print("\n[7] the coupling is real, and where the cost is")
    n = 16
    mode = 2
    y = march_lateral(lambda z: graded_2d(z, n, 0.08), OMEGA, 0.0, n, 48, method="rk4")
    yhat = to_hat(y, n, 3)

    # A single lateral Fourier mode of the contrast must couple k_x to
    # k_x +/- mode and, at first order, to nothing else.  Measuring which
    # off-diagonals are populated is a statement about the physics, not about
    # the norm of the answer.
    diag = np.zeros((3 * n, 3 * n), dtype=bool)
    near = np.zeros((3 * n, 3 * n), dtype=bool)
    for m in range(n):
        sel = m + n * np.arange(3)
        diag[np.ix_(sel, sel)] = True
        for s in (mode, -mode):
            near[np.ix_(sel, (m + s) % n + n * np.arange(3))] = True
    far = ~(diag | near)
    scale = float(np.max(np.abs(yhat)))
    m_near = float(np.max(np.abs(yhat[near]))) / scale
    m_far = float(np.max(np.abs(yhat[far]))) / scale
    print(f"      k_x +/- {mode} coupling: {m_near:.3e}      everything else: {m_far:.3e}")
    report("the contrast mode couples the wavenumbers it should", m_near > 1e-4)
    report("the dominant coupling is to k_x +/- mode, not spread", m_far < 0.5 * m_near)

    # WHERE THE COST ACTUALLY IS.  The write-up claimed "R V_du R -- two dense
    # products per step -- becomes the cost".  Measuring it says otherwise on
    # both counts, and the reason is structural rather than incidental.
    #
    # A12 is the inverse vertical stiffness: diag(1/kappa, 1/mu, 1/mu), a pure
    # multiplication operator with no derivative in it, hence EXACTLY diagonal
    # on the lateral grid.  So Y A12 Y needs ONE dense-dense product, not two.
    # Meanwhile A11, A21 and A22 all contain the spectral derivative and are
    # therefore dense themselves, so A22 Y and Y A11 are dense products too.
    # The step is three dense (3n)^3 products and the quadratic term is one of
    # them -- it is not the dominant cost.  What remains true of it is physical,
    # not arithmetic: it is the term that carries the back-scattering.
    dmat = deriv_matrix(n)
    a11, a12, a21, a22 = aop(graded_2d(0.5 * H, n, 0.08), OMEGA, 0.0, dmat)

    off12 = float(np.max(np.abs(a12 - np.diag(np.diag(a12)))))
    print(f"      A12 off-diagonal content: {off12:.3e}")
    report("A12 is exactly diagonal (it carries no derivative)", off12 < 1e-14)

    # The blocks are BLOCK-SPARSE at the component level -- only a few of the
    # nine n x n sub-blocks are occupied -- while each occupied sub-block that
    # carries a derivative is dense in the lateral index.  Both halves of that
    # sentence matter, and a single fill fraction over the whole 3n x 3n block
    # conflates them: this gate's first run measured 0.19 for A11 and read it
    # as "not dense" when the truth is "2 of 9 sub-blocks, each dense".
    tol = 1e-13 * max(float(np.max(np.abs(a21))), 1.0)

    def occupancy(blk_: NDArray) -> tuple[int, float]:
        """Occupied n x n sub-blocks, and the densest fill among them.

        Args:
            blk_: One 3n x 3n block.

        Returns:
            (number of occupied sub-blocks out of 9, densest fill).
        """
        occ, fills = 0, [0.0]
        for i in range(3):
            for jj in range(3):
                sub = blk_[i * n : (i + 1) * n, jj * n : (jj + 1) * n]
                if float(np.max(np.abs(sub))) > tol:
                    occ += 1
                    fills.append(float(np.mean(np.abs(sub) > tol)))
        return occ, max(fills)

    occs = {nm: occupancy(b) for nm, b in (("A11", a11), ("A12", a12), ("A21", a21), ("A22", a22))}
    print(
        "      occupied sub-blocks / densest fill:  "
        + "   ".join(f"{k} {v[0]}/9 {v[1]:.2f}" for k, v in occs.items())
    )
    report("A11 and A22 occupy at most 3 of 9 sub-blocks", max(occs["A11"][0], occs["A22"][0]) <= 3)
    report("a derivative-bearing sub-block IS dense laterally", occs["A21"][1] > 0.8)

    # So the flop count of one right-hand side, in units of n^3, is:
    #   Y (A12 Y)   -- A12 diagonal, so ONE dense (3n)^3 product  = 27
    #   A22 Y       -- occ(A22) sub-block products of 3n^3 each
    #   Y A11       -- occ(A11) sub-block products of 3n^3 each
    # The quadratic term is therefore the dominant single cost, but for a
    # different reason than the write-up gave: not because it is two dense
    # products (it is one), but because the linear terms are block-sparse.
    f_quad = 27.0
    f_lin = 3.0 * (occs["A22"][0] + occs["A11"][0])
    print(
        f"      flops per rhs, units of n^3:  quadratic {f_quad:.0f}"
        f"   linear {f_lin:.0f}   quadratic share {f_quad / (f_quad + f_lin):.0%}"
    )
    report("the quadratic term is the dominant single cost", f_quad > f_lin)

    yb, reps = y.copy(), 200
    diag12 = np.diag(a12)[:, None]

    def timeit(fn: Callable[[], object]) -> float:
        """Mean seconds per call.

        Args:
            fn: The thing to time.

        Returns:
            Seconds.
        """
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    t_naive = timeit(lambda: yb @ a12 @ yb)
    t_quad = timeit(lambda: yb @ (diag12 * yb))
    print(f"      Y A12 Y naive {t_naive * 1e3:.3f} ms -> exploiting the diagonal {t_quad * 1e3:.3f} ms")
    report("the quadratic term is ONE dense product, not two", t_quad < 0.65 * t_naive)

    print("      cost scaling of one march (48 steps):")
    times = []
    sizes = [8, 12, 16, 24, 32]
    for nn in sizes:
        t0 = time.perf_counter()
        march_lateral(lateral_profile(nn, 0.08), OMEGA, 0.0, nn, 48, method="rk4")
        times.append(time.perf_counter() - t0)
        print(f"        n={nn:3d}  3n={3 * nn:3d}  {times[-1]:.3f} s")
    slope = float(np.polyfit(np.log(np.array(sizes, dtype=float)), np.log(np.array(times)), 1)[0])
    print(f"      measured exponent in n: {slope:.2f}  (dense cubic would be 3)")
    report("the march cost does not exceed the dense cubic estimate", slope < 3.4)


def part8() -> None:
    """The big algebraic Riccati solution exists and the march respects the branch."""
    print("\n[8] branch and algebraic consistency of the coupled operator")
    n = 12
    dmat = deriv_matrix(n)
    sl = graded_2d(0.5 * H, n, 0.08)
    blk = aop(sl, OMEGA, 0.0, dmat)
    a11, a12, _, _ = blk

    # Start from the uniform-background impedance and refine on the COUPLED
    # operator.  A stabilising solution must exist; its residual is a statement
    # about the operator, independent of any march.
    y0 = y_start(REF, OMEGA, n, 0.0)
    y, resid = newton_are(y0, blk, iters=60)
    print(f"      Newton residual on the coupled ARE: {resid:.3e}")
    report("the coupled algebraic Riccati equation has a stabilising solution", resid < 1e-10)

    ev = np.real(np.linalg.eigvals(a11 + a12 @ y))
    print(f"      max Re eig(A11 + A12 Y): {float(np.max(ev)):+.3e}")
    report("that solution is the DOWNGOING branch", bool(np.max(ev) < 1e-9))

    # The marched impedance must stay on the same branch all the way up.
    ym = march_lateral(lambda z: graded_2d(z, n, 0.08), OMEGA, 0.0, n, 64, method="rk4")
    dmat_top = deriv_matrix(n)
    b_top = aop(graded_2d(0.0, n, 0.08), OMEGA, 0.0, dmat_top)
    ev_top = np.real(np.linalg.eigvals(b_top[0] + b_top[1] @ ym))
    print(f"      at the top of the slab, max Re eig: {float(np.max(ev_top)):+.3e}")
    report("the march does not flip branch", bool(np.max(ev_top) < 1e-9))

    # Y must remain bounded -- the property the mode-basis reflection loses.
    norms = []
    for nstep in (16, 64, 256):
        yy = march_lateral(lambda z: graded_2d(z, n, 0.08), OMEGA, 0.0, n, nstep, method="rk4")
        norms.append(float(np.linalg.norm(yy)))
    print(f"      |Y| at 16/64/256 steps: {norms[0]:.6e}  {norms[1]:.6e}  {norms[2]:.6e}")
    report("the marched impedance stays bounded", max(norms) / min(norms) < 1.05)


def main() -> int:
    """Run every part and summarise.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    print("=" * 78)
    print("  The impedance march with the lateral coupling INSIDE the marched variable")
    print("  2.5-D; smooth in depth and laterally; SI units; e^{-i omega t}")
    print("=" * 78)
    for part in (part1, part2, part3, part4, part5, part6, part7, part8):
        part()
    ok = sum(1 for _, p in _PASS if p)
    print("\n" + "=" * 78)
    for label, passed in _PASS:
        if not passed:
            print(f"  FAILED: {label}")
    print(f"  {ok}/{len(_PASS)} checks passed")
    print("=" * 78)
    return 0 if ok == len(_PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
