#!/usr/bin/env python3
"""How the augmentation composes with the Schwinger form -- settled, and gated.

Task 1 of docs/plans/2026-09-18-first-order-propagator-moment.md, resuming from
the second revision banner.  That banner recorded a DESIGN question, not a bug:

    the transferred bilinear b[phi, psi] = <phi, J6 DeltaA psi> carries a SINGLE
    DeltaA with the derivatives already distributed between the two sides, which
    is what keeps the indicator of the scatterer from ever being differentiated;
    but the Schwinger object <J6 phi, DeltaA Gamma DeltaA psi> needs DeltaA psi
    as an actual SOURCE between two propagator legs.

THE SETTLEMENT
--------------
The two are not in conflict, and the reason is that the Born term and the
Schwinger term are different in a way that matters:

  * In the Born term the contrast operator's indicator and the test function meet
    at the SAME point.  Writing d_alpha[1_V c] there and pairing it with another
    object carrying 1_V would multiply a surface layer by an indicator, which is
    not defined.  The transfer is not a convenience, it is the only well-posed
    form, and DeltaC_eff is right.

  * In the Schwinger term the two indicators are separated by Gamma.  The source
    d_alpha[1_V c] at x' is paired, through a kernel, with a test object at x.
    Nothing coincides, so the surface layer is harmless -- and in the lateral
    Fourier domain it never has to be formed at all, because

        FT[ d_alpha ( 1_V c ) ] = i k_alpha  FT[ 1_V c ]

    is an exact identity: the derivative is carried by the TRANSFORM, and the
    indicator stays inside the form factor where it was.

So route 2 of the banner is correct, and it is exact rather than approximate.
Route 1 -- putting the surface terms back and evaluating them on the cube faces
-- computes the same number the hard way and is not needed.

The bookkeeping that follows is then completely uniform.  Every derivative in
the first-order system is LATERAL (the depth derivative is the left-hand side of
d3 q = A q + d), so in the lateral Fourier domain every one of them is a factor
i k_alpha at the single wavenumber the integral runs over.  The four derivative
slots of one Schwinger entry land as:

    left  leg, OUTER derivative -> onto the test polynomial   (the augmentation)
    left  leg, INNER derivative -> onto the middle field      -> i k
    right leg, OUTER derivative -> onto the source transform  -> i k
    right leg, INNER derivative -> onto the trial polynomial  (the augmentation)

The two OUTER slots are the ones the banner could not place.  The left one goes
where DeltaC_eff already puts it; the right one goes onto the kernel.

WHAT THIS SCRIPT CHECKS
-----------------------
Part 1  the identity above, numerically, against a real-space answer, with a
        negative control on the sign.  This is the design question itself.
Part 2  fault 1 of the banner: the system matrix in the PAPER basis, checked
        against the thesis-basis matrix through the constant map S.
Part 3  fault 2 of the banner: the Schwinger assembly with BOTH DeltaA in place,
        on the density channel, against the validated Kelvin-route answer.

Run:  conda run -n seismic python scripts/gate_first_order_schwinger.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from scripts.gate_first_order_lateral_moment import amat, tri  # noqa: E402
from scripts.gate_first_order_tmatrix import (  # noqa: E402
    coupling_first_order,
    delta_a_terms,
    propagator_moment,
    qfield_of,
)

J6 = np.zeros((6, 6))
J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)

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
# Part 1.  The derivative of the indicator, carried by the transform
# ---------------------------------------------------------------------------


def _gauss_hat(k1: np.ndarray, k2: np.ndarray, x0: np.ndarray, s: float) -> np.ndarray:
    """Lateral transform of a unit Gaussian bump centred at x0.

    With the convention fhat(k) = int f(x) e^{-i k.x} d2x, a bump
    f(x) = exp(-|x - x0|^2 / 2 s^2) transforms to
    2 pi s^2 exp(-i k.x0) exp(-s^2 |k|^2 / 2).

    Args:
        k1: First wavenumber component.
        k2: Second wavenumber component.
        x0: Centre, shape (2,).
        s: Width.

    Returns:
        Same shape as k1.
    """
    ph = np.exp(-1j * (k1 * x0[0] + k2 * x0[1]))
    return 2.0 * np.pi * s**2 * ph * np.exp(-0.5 * s**2 * (k1**2 + k2**2))


def _ff_grid(k1: np.ndarray, k2: np.ndarray, a: float) -> np.ndarray:
    """Form factor of the square [-a, a]^2 on a grid of wavenumbers.

    Args:
        k1: First wavenumber component.
        k2: Second wavenumber component.
        a: Half-width.

    Returns:
        Same shape as k1, real.
    """

    def g(k: np.ndarray) -> np.ndarray:
        return np.where(np.abs(k) < 1e-12, 2.0 * a, 2.0 * np.sin(k * a) / np.where(k == 0.0, 1.0, k))

    return g(k1) * g(k2)


def part1_indicator_derivative(a: float = 0.5, s: float = 0.37) -> None:
    """Check FT[d_alpha(1_V c)] = i k_alpha FT[1_V c] against real space.

    The claim being tested is the whole content of the design question: that the
    outer derivative of a contrast term may be carried by the transform, leaving
    the indicator undifferentiated.  Paired with a smooth bump f the identity
    says

        int f d_1[1_S] d2x  =  - int_S d_1 f d2x ,

    the right side an ordinary integral over the square.  The left side is
    evaluated in the lateral Fourier domain as (2 pi)^-2 int fhat(-k) i k_1 F(k).

    Args:
        a: Half-width of the square.
        s: Width of the Gaussian bump.
    """
    print("")
    print("--- Part 1: the outer derivative goes onto the transform ----------")
    x0 = np.array([0.21, -0.13])

    # Real-space right-hand side, by Gauss quadrature over the square.
    ng = 400
    t, wt = np.polynomial.legendre.leggauss(ng)
    xs, ws = a * t, a * wt
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    wwx, wwy = np.meshgrid(ws, ws, indexing="ij")
    d1f = -((xx - x0[0]) / s**2) * np.exp(-((xx - x0[0]) ** 2 + (yy - x0[1]) ** 2) / (2.0 * s**2))
    rhs = -np.sum(wwx * wwy * d1f)

    # Fourier-domain left-hand side.  The Gaussian damps the integrand, so a
    # plain tensor-product Gauss rule on a box of a few widths is ample.
    kmax = 14.0 / s
    nk = 900
    tk, wk = np.polynomial.legendre.leggauss(nk)
    kv, kw = kmax * tk, kmax * wk
    k1, k2 = np.meshgrid(kv, kv, indexing="ij")
    w1, w2 = np.meshgrid(kw, kw, indexing="ij")
    integ = _gauss_hat(-k1, -k2, x0, s) * (1j * k1) * _ff_grid(k1, k2, a)
    lhs = np.sum(w1 * w2 * integ) / (2.0 * np.pi) ** 2

    rel = abs(lhs - rhs) / abs(rhs)
    print(f"    real space   -int_S d_1 f  = {rhs.real: .12e}")
    print(f"    k space      (2pi)^-2 int fhat(-k) i k_1 F(k) = {lhs.real: .12e}")
    print(f"    imaginary residue of the k-space value        = {lhs.imag: .3e}")
    print(f"    relative difference                           = {rel:.3e}")
    report("FT[d_1(1_V c)] = i k_1 FT[1_V c], against real space", rel < 1e-9)
    report("the k-space value is real, as it must be", abs(lhs.imag) < 1e-12 * abs(rhs))

    # Negative control: the opposite sign is a different number by a wide margin.
    bad = abs(-lhs - rhs) / abs(rhs)
    print(f"    NEGATIVE CONTROL, wrong sign: relative difference = {bad:.3e}")
    report("NEGATIVE CONTROL: the opposite sign is rejected", bad > 0.5)


# ---------------------------------------------------------------------------
# Part 2.  The system matrix in the paper basis (fault 1)
# ---------------------------------------------------------------------------


def amat_paper(ref: ReferenceMedium, omega: complex, k1: float, k2: float) -> np.ndarray:
    """The 6x6 system matrix in the PAPER basis (-t13, -t23, -t33, v1, v2, v3).

    Transcribed from the blocks derived and validated in
    Mathematica/MatrixVectorWaveEquation.wl (D.20-D.23, checked against Appendix
    J), with every lateral derivative replaced by i k_alpha -- the same e^{+ikx}
    convention the form factors use.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        k1: Lateral wavenumber, first component.
        k2: Lateral wavenumber, second component.

    Returns:
        Shape (6, 6) complex.
    """
    lam, mu = ref.lam, ref.mu
    kc = lam + 2.0 * mu
    gam = lam / kc
    nu1, nu2 = 4.0 * mu * (lam + mu) / kc, 2.0 * mu * lam / kc
    iw = 1j * omega
    j = 1j
    a6 = np.zeros((6, 6), dtype=np.complex128)
    # A11 = -d_a C_a3 C33^-1
    a6[0, 2], a6[1, 2] = -j * gam * k1, -j * gam * k2
    a6[2, 0], a6[2, 1] = -j * k1, -j * k2
    # A12 = i w rho - (1/(i w)) d_a U_ab d_b
    a6[0, 3] = iw * ref.rho + (nu1 * k1**2 + mu * k2**2) / iw
    a6[0, 4] = k1 * k2 * (mu + nu2) / iw
    a6[1, 3] = k1 * k2 * (nu2 + mu) / iw
    a6[1, 4] = iw * ref.rho + (mu * k1**2 + nu1 * k2**2) / iw
    a6[2, 5] = iw * ref.rho
    # A21 = i w C33^-1
    a6[3, 0], a6[4, 1], a6[5, 2] = iw / mu, iw / mu, iw / kc
    # A22 = -C33^-1 C_3b d_b
    a6[3, 5], a6[4, 5] = -j * k1, -j * k2
    a6[5, 3], a6[5, 4] = -j * gam * k1, -j * gam * k2
    return a6


def amat_paper_batch(ref: ReferenceMedium, omega: complex, k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """The paper-basis system matrix over a grid of lateral wavenumbers.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        k1: Lateral wavenumbers, shape (N,).
        k2: Lateral wavenumbers, shape (N,).

    Returns:
        Shape (N, 6, 6) complex.
    """
    lam, mu = ref.lam, ref.mu
    kc = lam + 2.0 * mu
    gam = lam / kc
    nu1, nu2 = 4.0 * mu * (lam + mu) / kc, 2.0 * mu * lam / kc
    iw = 1j * omega
    j = 1j
    a6 = np.zeros((k1.shape[0], 6, 6), dtype=np.complex128)
    a6[:, 0, 2], a6[:, 1, 2] = -j * gam * k1, -j * gam * k2
    a6[:, 2, 0], a6[:, 2, 1] = -j * k1, -j * k2
    a6[:, 0, 3] = iw * ref.rho + (nu1 * k1**2 + mu * k2**2) / iw
    a6[:, 0, 4] = k1 * k2 * (mu + nu2) / iw
    a6[:, 1, 3] = k1 * k2 * (nu2 + mu) / iw
    a6[:, 1, 4] = iw * ref.rho + (mu * k1**2 + nu1 * k2**2) / iw
    a6[:, 2, 5] = iw * ref.rho
    a6[:, 3, 0], a6[:, 4, 1], a6[:, 5, 2] = iw / mu, iw / mu, iw / kc
    a6[:, 3, 5], a6[:, 4, 5] = -j * k1, -j * k2
    a6[:, 5, 3], a6[:, 5, 4] = -j * gam * k1, -j * gam * k2
    return a6


def smap(omega: complex) -> np.ndarray:
    """The constant map b = S q from the paper basis to the thesis basis.

    Thesis b = (u_z, u_x, u_y, T_zz, T_xz, T_yz), paper
    q = (-t13, -t23, -t33, v1, v2, v3), with u = v / (-i omega), tau_a3 = -q_a
    and the index relabelling (1, 2, 3) -> (3, 1, 2).

    Args:
        omega: Angular frequency, complex.

    Returns:
        Shape (6, 6) complex.
    """
    s = np.zeros((6, 6), dtype=np.complex128)
    inv = 1.0 / (-1j * omega)
    s[0, 5], s[1, 3], s[2, 4] = inv, inv, inv
    s[3, 2], s[4, 0], s[5, 1] = -1.0, -1.0, -1.0
    return s


def part2_basis(ref: ReferenceMedium, omega: complex) -> None:
    """Check the paper-basis matrix against the thesis-basis one through S.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
    """
    print("")
    print("--- Part 2: the basis fault, A_paper = S^-1 A_thesis S ------------")
    rng = np.random.default_rng(20260918)
    worst = 0.0
    for _ in range(6):
        k1, k2 = rng.uniform(-3.0, 3.0, size=2)
        ap = amat_paper(ref, omega, k1, k2)
        at = amat(ref, omega, k1, k2)
        s = smap(omega)
        got = np.linalg.solve(s, at @ s)
        worst = max(worst, np.max(np.abs(got - ap)) / max(np.max(np.abs(ap)), 1.0))
    print(f"    worst relative entry difference over 6 wavenumbers = {worst:.3e}")
    report("A_paper == S^-1 A_thesis S, entry by entry", worst < 1e-10)

    # The spectra must agree regardless of basis; a cheap independent statement.
    k1, k2 = 0.7, -1.9
    ep = np.sort_complex(np.linalg.eigvals(amat_paper(ref, omega, k1, k2)))
    et = np.sort_complex(np.linalg.eigvals(amat(ref, omega, k1, k2)))
    dsp = np.max(np.abs(ep - et)) / np.max(np.abs(et))
    print(f"    spectra agree to {dsp:.3e}")
    report("the two bases carry the same spectrum", dsp < 1e-10)

    # Quasi-Hamiltonian symmetry in the paper basis.  The operator statement is
    # A^t J6 = -J6 A with the OPERATOR transpose, and d_alpha^t = -d_alpha; in
    # the lateral Fourier domain that transpose is the matrix transpose taken at
    # REVERSED wavenumber.  Testing it at fixed k instead is simply a different
    # claim, and a false one -- worth stating, since it is an easy slip.
    ap = amat_paper(ref, omega, k1, k2)
    am = amat_paper(ref, omega, -k1, -k2)
    # In SI the entries of A span seventeen orders, so both residuals are read
    # against the arithmetic floor of the largest entry rather than against it.
    floor = float(np.finfo(float).eps * np.max(np.abs(ap)))
    res = float(np.max(np.abs(am.T @ J6 + J6 @ ap)))
    naive = float(np.max(np.abs(ap.T @ J6 + J6 @ ap)))
    print(f"    arithmetic floor              = {floor:.3e}")
    print(f"    A(-k)^T J6 + J6 A(k) residual = {res:.3e}   ({res / floor:.2f} floors)")
    print(f"    the same test at fixed k      = {naive:.3e}   ({naive / floor:.2e} floors)")
    report("A is quasi-Hamiltonian: A(-k)^T J6 = -J6 A(k)", res <= 4.0 * floor)
    report("NEGATIVE CONTROL: the fixed-k form of the relation is false", naive > 1e3 * floor)


# ---------------------------------------------------------------------------
# Part 3.  The Schwinger assembly, with BOTH contrast operators
# ---------------------------------------------------------------------------


def form_factors_batch(k1: np.ndarray, k2: np.ndarray, a: float) -> tuple:
    """Lateral transforms of 1, x1 and x2 over the square, over a grid.

    The scalar version in the lateral gate is exact; this is the same thing
    written to take arrays, because the quadrature this integral needs cannot
    afford a Python call per node.

    Args:
        k1: First wavenumber component, shape (N,).
        k2: Second wavenumber component, shape (N,).
        a: Cube half-width.

    Returns:
        (F, F1, F2), each shape (N,) complex.
    """

    def g(k: np.ndarray) -> np.ndarray:
        safe = np.where(np.abs(k) < 1e-12, 1.0, k)
        return np.where(np.abs(k) < 1e-12, 2.0 * a, 2.0 * np.sin(k * a) / safe)

    def dg(k: np.ndarray) -> np.ndarray:
        safe = np.where(np.abs(k) < 1e-9, 1.0, k)
        val = 2.0 * (a * np.cos(k * a) / safe - np.sin(k * a) / safe**2)
        return np.where(np.abs(k) < 1e-9, 0.0, val)

    g1, g2 = g(k1), g(k2)
    return (g1 * g2).astype(complex), 1j * dg(k1) * g2, 1j * g1 * dg(k2)


def kernels_paper(ref: ReferenceMedium, omega: complex, k1: np.ndarray, k2: np.ndarray, a: float) -> dict:
    """The four z-integrated propagator kernels in the PAPER basis.

    Gamma is built from the spectral projectors of A -- never from a matrix
    exponential, which is unstable in the evanescent regime -- and the
    eigen-decomposition is done on the SCALED matrix, since on the unscaled one
    the up- and down-going eigenvectors become nearly parallel as k grows.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        k1: Lateral wavenumbers, shape (N,).
        k2: Lateral wavenumbers, shape (N,).
        a: Cube half-width.

    Returns:
        Dict keyed by (m, n) with (N, 6, 6) complex values, indexed
        [receiver component, source component].
    """
    a6 = amat_paper_batch(ref, omega, k1, k2)
    scaled, sd = balance_batch(a6)
    ev, rv = np.linalg.eig(scaled)
    lv = np.linalg.inv(rv)
    dn = downgoing(ev)
    out = {}
    for m in (0, 1):
        for nn in (0, 1):
            w = np.where(dn, tri(m, nn, -ev, a), -tri(nn, m, ev, a))
            acc = np.einsum("nij,nj,njl->nil", rv, w, lv)
            out[(m, nn)] = acc * sd[:, :, None] / sd[:, None, :]
    return out


def panel_axis(kmax: float, nk: int, ksplit: float, nin: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of a symmetric three-panel Gauss rule on [-kmax, kmax].

    A single Gauss rule over a wide interval puts its nodes at the ENDS, which is
    the opposite of where these integrands keep their mass; panelling is what
    makes the centre resolved without making the tail expensive.

    Args:
        kmax: Half-width of the box.
        nk: Nodes per outer panel.
        ksplit: Inner panel half-width; 0 for a single panel.
        nin: Nodes in the inner panel.

    Returns:
        (nodes, weights), both shape (2 nk + nin,) or (nk,).
    """
    if ksplit <= 0.0:
        panels = [(-kmax, kmax, nk)]
    else:
        panels = [(-kmax, -ksplit, nk), (-ksplit, ksplit, nin), (ksplit, kmax, nk)]
    nodes, wts = [], []
    for lo, hi, n in panels:
        t, w = np.polynomial.legendre.leggauss(n)
        nodes.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
        wts.append(0.5 * (hi - lo) * w)
    return np.concatenate(nodes), np.concatenate(wts)


def balance_batch(a6: np.ndarray, sweeps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Balance a stack of matrices by a diagonal similarity, Parlett--Reinsch.

    The spectral construction of Gamma needs the eigenvectors of A, and at large
    k the up- and down-going ones become nearly parallel.  The fixed scaling
    ``diag(1,1,1,k,k,k)`` that this work previously used is in the WRONG
    DIRECTION: measured against the unscaled matrix it makes the eigenvector
    condition number worse at every k above the propagating window, by two orders
    at k = 60.  Balancing beats both by about twelve orders there
    (8.7e7 against 1.6e20), because it is free to treat the third component
    differently from the first two, which no scalar rule can.

    Returns S with ``a6 = S M S^-1``, so a projector computed from M maps back as
    ``S P S^-1`` -- the same unscaling the fixed rule used.

    Args:
        a6: Shape (N, 6, 6) complex.
        sweeps: Balancing sweeps.

    Returns:
        (balanced, s) with balanced shape (N, 6, 6) and s shape (N, 6).
    """
    m = a6.astype(np.complex128).copy()
    n = m.shape[0]
    s = np.ones((n, 6), dtype=np.complex128)
    for _ in range(sweeps):
        for i in range(6):
            dia = np.abs(m[:, i, i])
            r = np.sum(np.abs(m[:, i, :]), axis=1) - dia
            c = np.sum(np.abs(m[:, :, i]), axis=1) - dia
            ok = (r > 0.0) & (c > 0.0)
            f = np.where(ok, np.sqrt(np.divide(c, r, out=np.ones_like(r), where=ok)), 1.0)
            m[:, i, :] *= f[:, None]
            m[:, :, i] /= f[:, None]
            s[:, i] /= f
    return m, s


def downgoing(ev: np.ndarray) -> np.ndarray:
    """Split the spectrum of A by the radiation condition.

    Classifying by ``Re(ev) < 0`` alone is correct for evanescent modes and
    MEANINGLESS for propagating ones: below k = omega/beta the vertical
    wavenumber is real, Re(ev) is zero up to round-off, and its sign is then
    decided by arithmetic noise -- independently at +k and -k, which destroys
    reciprocity outright (measured: a residual of 1.2 at |k| = 1.1e-3).

    The limit omega -> omega(1 + i eps) settles it without needing eps.  A
    propagating mode has ev = i k_z, and the perturbation gives k_z a positive
    imaginary part, so Re(ev) -> 0^- exactly when Im(ev) > 0.  Downgoing is
    therefore ``Re(ev) < 0`` where that is meaningful and ``Im(ev) > 0`` where it
    is not, and the two agree wherever both apply.

    Args:
        ev: Eigenvalues, any shape.

    Returns:
        Boolean array of the same shape.
    """
    scale = np.maximum(np.abs(ev), np.finfo(float).tiny)
    propagating = np.abs(np.real(ev) / scale) < 1e-8
    return np.where(propagating, np.imag(ev) > 0.0, np.real(ev) < 0.0)


def radial_panels(edges: list[float], nper: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss nodes and weights on [edges[0], edges[-1]], panel by panel.

    Args:
        edges: Increasing panel boundaries.
        nper: Nodes per panel.

    Returns:
        (nodes, weights).
    """
    t, w = np.polynomial.legendre.leggauss(nper)
    nodes, wts = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        nodes.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
        wts.append(0.5 * (hi - lo) * w)
    return np.concatenate(nodes), np.concatenate(wts)


def gamma_at(ref: ReferenceMedium, omega: complex, k1: np.ndarray, k2: np.ndarray, dz: float) -> np.ndarray:
    """Gamma(k; dz), the propagator at one depth offset, from spectral projectors.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        k1: Lateral wavenumbers, shape (N,).
        k2: Lateral wavenumbers, shape (N,).
        dz: Depth offset, receiver minus source.

    Returns:
        Shape (N, 6, 6) complex.
    """
    a6 = amat_paper_batch(ref, omega, k1, k2)
    scaled, sd = balance_batch(a6)
    ev, rv = np.linalg.eig(scaled)
    lv = np.linalg.inv(rv)
    dn = downgoing(ev)
    keep = dn if dz > 0.0 else ~dn
    sgn = 1.0 if dz > 0.0 else -1.0
    w = np.where(keep, sgn * np.exp(ev * dz), 0.0)
    acc = np.einsum("nij,nj,njl->nil", rv, w, lv)
    return acc * sd[:, :, None] / sd[:, None, :]


def kelvin_static(ref: ReferenceMedium, r: np.ndarray) -> np.ndarray:
    """The static Kelvin tensor u_i = G_ij f_j.

    Used rather than the package's elastodynamic tensor because at k_S r ~ 0.01
    that routine loses about twelve digits to a cancellation that the static form
    does not have.  The omega^2 correction it drops is of order (k_S r)^2 ~ 1e-4,
    far below the discrepancy this check exists to resolve.

    Args:
        ref: Background medium.
        r: Separation vector, shape (3,).

    Returns:
        Shape (3, 3).
    """
    lam, mu = ref.lam, ref.mu
    nu = lam / (2.0 * (lam + mu))
    rr = float(np.linalg.norm(r))
    nhat = r / rr
    return ((3.0 - 4.0 * nu) * np.eye(3) + np.outer(nhat, nhat)) / (16.0 * np.pi * mu * (1.0 - nu) * rr)


def part3a_propagator_pointwise(ref: ReferenceMedium, omega: complex, dz: float = 0.5) -> None:
    """Gamma inverted laterally, against the analytic Kelvin tensor.

    This is the one arbiter in the chain that owes nothing to the moment
    machinery: it asks only whether the propagator built from the spectral
    projectors of A is the Green's matrix with the normalisation assumed, that a
    force f_i enters row i of d3 q = A q + s and that q_{3+i} = v_i = -i omega
    u_i comes out.  Nothing here involves a form factor, a volume, or a moment.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        dz: Vertical separation.
    """
    print("")
    print("--- Part 3a: the propagator itself, against the Kelvin tensor -----")
    gref = kelvin_static(ref, np.array([0.0, 0.0, dz]))
    rel = 1.0
    gnum = np.zeros((3, 3), dtype=complex)
    # POLAR, deliberately: the kernel depends on |k| and on direction separately,
    # so it is not smooth at the origin and a Cartesian tensor rule converges
    # only as 1/n there.  In polar the Jacobian k removes that, and the angular
    # integrand is periodic, where the uniform rule is spectrally accurate.
    # (The lateral moment in Part 3 is the opposite case: its form factors are
    # separable sinc products that polar cannot resolve.)
    for nr, nth in ((200, 48), (400, 64), (800, 96)):
        # Panel edges at the two branch radii, where the vertical wavenumber
        # vanishes: the radiation reaction comes entirely from k < omega/beta,
        # which without these edges falls inside one panel spanning [0, 8] and
        # gets perhaps one node.
        ka, kb = float(abs(omega) / ref.alpha), float(abs(omega) / ref.beta)
        edges = [0.0, ka, kb, 4.0 / dz, 20.0 / dz, 80.0 / dz]
        kr, wr = radial_panels(edges, nr // 5)
        th = 2.0 * np.pi * np.arange(nth) / nth
        kg, tg = np.meshgrid(kr, th, indexing="ij")
        wg = np.meshgrid(wr * kr, np.full(nth, 2.0 * np.pi / nth), indexing="ij")
        gam = gamma_at(ref, omega, (kg * np.cos(tg)).ravel(), (kg * np.sin(tg)).ravel(), dz)
        got = np.einsum("n,nij->ij", (wg[0] * wg[1]).ravel(), gam) / (2.0 * np.pi) ** 2
        # q_{3+i} = v_i = -i omega u_i, and the source in row j is the force f_j
        gnum = got[3:, :3] / (-1j * omega)
        rel = float(np.max(np.abs(gnum.real - gref)) / np.max(np.abs(gref)))
        print(f"    nr={nr:4d} nth={nth:3d}   worst relative difference = {rel:.3e}")
    print(f"    G_11 from the first-order system = {gnum[0, 0].real: .10e}")
    print(f"    G_11 analytic (static Kelvin)    = {gref[0, 0]: .10e}")
    print(f"    G_33 from the first-order system = {gnum[2, 2].real: .10e}")
    print(f"    G_33 analytic (static Kelvin)    = {gref[2, 2]: .10e}")
    off = float(np.max(np.abs(gnum - np.diag(np.diag(gnum)))) / np.max(np.abs(gref)))
    print(f"    off-diagonal leakage             = {off:.3e}")
    report("Gamma inverted laterally IS the Green's tensor, normalisation included", rel < 2e-3)

    # The imaginary part has no static counterpart at all: it is the radiation
    # reaction, and it is here only because the up/down split now follows the
    # radiation condition rather than the sign of a quantity that vanishes.
    rad = float(omega.real * (1.0 / ref.alpha**3 + 2.0 / ref.beta**3) / (12.0 * np.pi * ref.rho))
    relr = float(abs(gnum[0, 0].imag - rad) / rad)
    print(f"    Im G_11 from the first-order system = {gnum[0, 0].imag: .6e}")
    print(f"    Im G_11 = w(1/a^3 + 2/b^3)/12 pi rho = {rad: .6e}   rel {relr:.3e}")
    report("the radiation reaction comes out right, with no static counterpart", relr < 5e-3)


def weight_coeffs(q0: np.ndarray, qg: np.ndarray, comp: int, d: int, ff: tuple) -> tuple:
    """The (z^0, z^1) lateral coefficients of one component, possibly differentiated.

    Args:
        q0: Constant part of the q-field, shape (6,).
        qg: Gradient of the q-field, shape (6, 3).
        comp: Component index.
        d: 0 for the field, 1 or 2 for d/dx_1 or d/dx_2.
        ff: (F, F1, F2) form factors.

    Returns:
        (coefficient of z^0, coefficient of z^1).
    """
    f, f1, f2 = ff
    if d != 0:
        return qg[comp, d - 1] * f, 0.0 + 0.0j
    return q0[comp] * f + qg[comp, 0] * f1 + qg[comp, 1] * f2, qg[comp, 2] * f


def schwinger_integrand(
    ref: ReferenceMedium,
    omega: complex,
    con: MaterialContrast,
    a: float,
    kk: int,
    ll: int,
    k1: np.ndarray,
    k2: np.ndarray,
    kern: dict,
) -> np.ndarray:
    """One entry of <J6 psi_k, DeltaA Gamma DeltaA psi_l> at one wavenumber.

    The two contrast operators are looped over independently: the LEFT one is
    read in its transferred form, since its outer derivative meets the test
    polynomial; the RIGHT one is read in its original form, since its outer
    derivative meets the kernel and becomes i k.  The kernel connects the
    component the left operator READS to the component the right operator
    WRITES, which is what the single-operator assembly got wrong.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        con: Material contrast.
        a: Cube half-width.
        kk: Test index, 0..8.
        ll: Trial index, 0..8.
        k1: Lateral wavenumbers, shape (N,).
        k2: Lateral wavenumbers, shape (N,).
        kern: Output of kernels_paper at the same wavenumbers.

    Returns:
        Shape (N,) complex.
    """
    lam_t, mu_t = ref.lam + con.Dlambda, ref.mu + con.Dmu
    p0k, pgk = qfield_of(kk, lam_t, mu_t, omega)
    p0l, pgl = qfield_of(ll, lam_t, mu_t, omega)
    ffm = form_factors_batch(-k1, -k2, a)
    ffp = form_factors_batch(k1, k2, a)
    terms = delta_a_terms(ref.lam, ref.mu, ref.rho, con, omega)
    one = np.ones_like(k1, dtype=complex)
    kvec = (one, k1.astype(complex), k2.astype(complex))

    tot = np.zeros_like(k1, dtype=complex)
    for rowl, coll, coefl, dphil, dpsil in terms:
        if abs(coefl) == 0.0:
            continue
        idxl, sgnl = (rowl + 3, -1.0) if rowl < 3 else (rowl - 3, 1.0)
        ck = weight_coeffs(p0k, pgk, idxl, dphil, ffm)
        # the left leg's INNER derivative acts on the propagated field
        fl = one if dpsil == 0 else 1j * kvec[dpsil]
        for rowr, colr, coefr, dphir, dpsir in terms:
            if abs(coefr) == 0.0:
                continue
            # the right leg is NOT transferred: undo the sign the transfer put in,
            # and let its outer derivative become i k on the source transform
            fr = one if dphir == 0 else -1j * kvec[dphir]
            cl = weight_coeffs(p0l, pgl, colr, dpsir, ffp)
            pre = sgnl * coefl * coefr * fl * fr
            for m in (0, 1):
                if np.all(ck[m] == 0.0):
                    continue
                for nn in (0, 1):
                    if np.all(cl[nn] == 0.0):
                        continue
                    tot += pre * ck[m] * kern[(m, nn)][:, coll, rowr] * cl[nn]
    return tot


def lateral_grid(
    ref: ReferenceMedium,
    omega: complex,
    kmax: float,
    nk: int,
    ksplit: float,
    nin: int,
    nth: int = 10,
    nrad: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes and weights for the lateral integral, Cartesian outside, polar in.

    Two scales, two geometries, and neither rule works for both:

      * OUTSIDE the propagating window everything is evanescent and smooth, and
        the integrand is a product of sinc-type form factors -- separable in
        (k1, k2), which is what a Cartesian tensor rule is for and what a polar
        rule cannot resolve.
      * INSIDE it the kernel has square-root branch points where the vertical
        wavenumber vanishes, at |k| = omega/alpha and omega/beta.  Those are
        CIRCLES: no Cartesian panel can follow one, and Gauss quadrature across
        an unresolved square-root costs several per cent.  In polar they are
        radial panel edges.  Over that window k a < 0.02, so the form factors
        are constant there to 1e-4 and the polar rule loses nothing.

    The central block of the Cartesian grid is exactly the square
    [-ksplit, ksplit]^2, so it is dropped and replaced by the polar rule on the
    same square -- the two tile the box with no overlap and no gap.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        kmax: Half-width of the wavenumber box.
        nk: Nodes per outer Cartesian panel per axis.
        ksplit: Half-width of the inner square.
        nin: Unused placeholder kept for signature stability.
        nth: Angular Gauss nodes per octant of the inner square.
        nrad: Radial Gauss nodes per inner panel.

    Returns:
        (k1, k2, weights), each shape (N,).
    """
    # --- outer: Cartesian, central block removed ---
    kv, kw = panel_axis(kmax, nk, ksplit, 2)
    k1g, k2g = np.meshgrid(kv, kv, indexing="ij")
    w1g, w2g = np.meshgrid(kw, kw, indexing="ij")
    inner = (np.abs(k1g) < ksplit) & (np.abs(k2g) < ksplit)
    wout = np.where(inner, 0.0, w1g * w2g)
    k1o, k2o, wo = k1g.ravel(), k2g.ravel(), wout.ravel()
    keep = wo != 0.0

    # --- inner: polar on the square, panelled at the two branch radii ---
    ka, kb = float(abs(omega) / ref.alpha), float(abs(omega) / ref.beta)
    tt, tw = np.polynomial.legendre.leggauss(nth)
    rt, rw = np.polynomial.legendre.leggauss(nrad)
    k1i, k2i, wi = [], [], []
    for oct_ in range(8):
        lo = oct_ * np.pi / 4.0
        th = 0.5 * (np.pi / 4.0) * (tt + 1.0) + lo
        wth = 0.5 * (np.pi / 4.0) * tw
        rmax = ksplit / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
        for elo, ehi in ((0.0, ka), (ka, kb), (kb, None)):
            hi = rmax if ehi is None else np.full_like(rmax, ehi)
            lo_r = np.full_like(rmax, elo)
            rr = 0.5 * (hi - lo_r)[:, None] * rt[None, :] + 0.5 * (hi + lo_r)[:, None]
            wr = 0.5 * (hi - lo_r)[:, None] * rw[None, :]
            ww = wth[:, None] * wr * rr
            k1i.append((rr * np.cos(th)[:, None]).ravel())
            k2i.append((rr * np.sin(th)[:, None]).ravel())
            wi.append(ww.ravel())
    return (
        np.concatenate([k1o[keep], *k1i]),
        np.concatenate([k2o[keep], *k2i]),
        np.concatenate([wo[keep], *wi]),
    )


def schwinger_entry(
    ref: ReferenceMedium,
    omega: complex,
    con: MaterialContrast,
    a: float,
    kk: int,
    ll: int,
    kmax: float,
    nk: int,
    ksplit: float = 0.0,
    nin: int = 0,
) -> complex:
    """The lateral integral of one Schwinger entry, on a panelled Cartesian grid.

    Cartesian rather than polar: the integrand is a product of sinc-type form
    factors, which is separable in (k1, k2) and which a polar rule cannot
    resolve.  Panelled because the integrand carries two very different scales
    -- the form factors vary on 1/a while the medium varies on omega/beta -- and
    a single Gauss rule wide enough for the first leaves the second with no
    nodes at all.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        con: Material contrast.
        a: Cube half-width.
        kk: Test index.
        ll: Trial index.
        kmax: Half-width of the wavenumber box.
        nk: Nodes per outer panel per axis.
        ksplit: Inner panel half-width; 0 for a single panel.
        nin: Nodes in the inner panel per axis.

    Returns:
        The integral, with the (2 pi)^-2 of the inverse transform included.
    """
    k1f, k2f, wq = lateral_grid(ref, omega, kmax, nk, ksplit, nin)
    kern = kernels_paper(ref, omega, k1f, k2f, a)
    vals = schwinger_integrand(ref, omega, con, a, kk, ll, k1f, k2f, kern)
    return complex(np.sum(wq * vals) / (2.0 * np.pi) ** 2)


def duffy_moment(a: float, n: int, doubled: bool) -> float:
    """The cube's geometric moment of 1/r, by a Duffy-transformed quadrature.

    Two different objects, and the difference between them is the difference
    between a Galerkin and a collocation formulation:

      doubled : int_V int_V 1/|x - x'| dV dV', written as a single integral
                against the autocorrelation weight (2a-|u1|)(2a-|u2|)(2a-|u3|)
                over [-2a, 2a]^3 -- exact, and three dimensions cheaper;
      point   : int_V 1/|x| dV, the moment seen from the cube's centre.

    Both integrands are singular at the origin.  Splitting the octant by which
    coordinate is largest and substituting u = t (xi, eta, 1) gives a Jacobian
    t^2 that cancels the 1/r outright, leaving a smooth integrand.

    Args:
        a: Cube half-width.
        n: Gauss nodes per dimension.
        doubled: True for the volume-volume moment, False for the point-volume.

    Returns:
        The moment.
    """
    top = 2.0 * a if doubled else a
    t, w = np.polynomial.legendre.leggauss(n)
    tv, tw = 0.5 * top * (t + 1.0), 0.5 * top * w
    uv, uw = 0.5 * (t + 1.0), 0.5 * w
    tg, xg, yg = np.meshgrid(tv, uv, uv, indexing="ij")
    wg = np.einsum("i,j,k->ijk", tw, uw, uw)
    rad = np.sqrt(1.0 + xg**2 + yg**2)
    if doubled:
        wgt = (top - tg * xg) * (top - tg * yg) * (top - tg)
    else:
        wgt = np.ones_like(tg)
    # 8 octants x 3 sub-simplices, all congruent
    return float(24.0 * np.sum(wg * tg * wgt / rad))


def part3_density(ref: ReferenceMedium, omega: complex, a: float) -> None:
    """The density channel, assembled laterally, against the Kelvin route.

    With Dlambda = Dmu = 0 only the multiplicative i w Drho term of DeltaA
    survives, so this exercises the basis, the J6 contraction, the kernel, the
    triangular z integrals, the form factors and the quadrature -- everything
    except the derivative bookkeeping, which Part 1 settles on its own.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        a: Cube half-width.
    """
    print("")
    print("--- Part 3: the density channel, both contrast operators ----------")
    con = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=100.0)
    drho = con.Drho

    # The arbiter, in real space and owing nothing to the lateral route.  With
    # Dlambda = Dmu = 0 the only surviving term is multiplicative, so both legs
    # reduce to constants and the whole object is
    #     i omega^5 Drho^2 M ,   M = int_V int_V G_11 ,
    # with M the Kelvin double-volume moment (a0 + b0/3) D0.
    lam, mu = ref.lam, ref.mu
    kc = lam + 2.0 * mu
    a0 = (lam + 3.0 * mu) / (8.0 * np.pi * mu * kc)
    b0 = (lam + mu) / (8.0 * np.pi * mu * kc)
    d0 = duffy_moment(a, 60, doubled=True)
    d0c = duffy_moment(a, 80, doubled=True)
    # The moment is complex.  Its real part is the static double-volume Kelvin
    # moment; its imaginary part is the radiation reaction, which at k_S a = 0.01
    # is constant across the cube to O((k_S a)^2) and so contributes V^2 times
    # its value at the origin.
    vol = (2.0 * a) ** 3
    rad = omega.real * (1.0 / ref.alpha**3 + 2.0 / ref.beta**3) / (12.0 * np.pi * ref.rho)
    mgal = (a0 + b0 / 3.0) * d0 + 1j * vol**2 * rad
    want = 1j * omega**5 * drho**2 * mgal
    print(f"    D0 = int_V int_V 1/r        = {d0:.10f}   (n=80: {d0c:.10f})")
    print(f"    M  = int_V int_V G_11       = {mgal.real: .8e} {mgal.imag:+.4e}i")
    print(f"    arbiter  i w^5 Drho^2 M     = {want.real: .6e} {want.imag:+.8e}i")

    ks = 4.0 * omega.real / ref.beta
    prev = None
    for nk, nin, kmax in ((120, 60, 90.0), (180, 90, 140.0), (260, 120, 200.0)):
        got = schwinger_entry(ref, omega, con, a, 0, 0, kmax, nk, ksplit=ks, nin=nin)
        rel = abs(got - want) / abs(want)
        print(f"    nk={nk:4d} kmax={kmax:6.1f}  {got.real: .3e} {got.imag:+.8e}i  rel {rel:.3e}")
        prev = got
    assert prev is not None
    ok = abs(prev - want) / abs(want) < 5e-3
    report("the two-DeltaA assembly IS the Galerkin double-volume moment", ok)

    # Where the T-matrix gate's nine-parameter G sits relative to this.  It is a
    # different object by construction, and the difference factorises cleanly.
    cpl = coupling_first_order(ref.lam, ref.mu, ref.rho, con, omega, a)
    gmom = propagator_moment(ref, omega, a)
    alt = (cpl @ gmom @ cpl)[0, 0]
    d1 = duffy_moment(a, 80, doubled=False)
    vol = (2.0 * a) ** 3
    # the static point-volume moment, and the radiation reaction that goes with
    # it -- the latter in the closed form the moment export is itself gated on
    reg = (a0 + b0 / 3.0) * d1
    radv = vol * omega * (1.0 / ref.alpha**3 + 2.0 / ref.beta**3) / (12.0 * np.pi * ref.rho)
    pred = 1j * omega * (reg + 1j * radv) / mgal
    print("")
    print(f"    [DeltaC G DeltaC]_00 from the T-matrix gate = {alt.real: .6e} {alt.imag:+.4e}i")
    print(f"    ratio to the arbiter                        = {alt / want: .6f}")
    print(f"    predicted                                   = {pred: .6f}")
    print(f"      point-volume int_V 1/r   = {d1:.10f}   (Galerkin int_V int_V = {d0:.10f})")
    print(f"      collocation / Galerkin   = {d1 / d0:.10f}")
    print(f"      Re G_point = {reg: .6e}   Im G_point (radiation) = {rad: .6e}")
    res = abs(alt / want - pred) / abs(pred)
    print(f"    residual of that account                    = {res:.3e}")
    report("the gap to the T-matrix gate's G is (i omega) x collocation/Galerkin", res < 2e-3)


def part3b_reciprocity(ref: ReferenceMedium, omega: complex, dz: float = 0.37) -> None:
    """The propagator's reciprocity, in the form the symmetry test needs.

    From A(-k)^T J6 = -J6 A(k) and the adjoint equation for Gamma,

        Gamma(-k; z', z)^T = J6 Gamma(k; z, z') J6 ,

    using J6^-1 = -J6.  This is the statement that makes the assembled Schwinger
    matrix symmetric even though its two legs are built by different rules, so it
    is worth checking on its own before relying on it.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        dz: Depth offset.
    """
    print("")
    print("--- Part 3b: reciprocity of Gamma ---------------------------------")
    rng = np.random.default_rng(18092026)
    worst = 0.0
    for _ in range(5):
        k1, k2 = rng.uniform(-4.0, 4.0, size=2)
        kpv = (np.array([k1]), np.array([k2]))
        kmv = (np.array([-k1]), np.array([-k2]))
        gp = gamma_at(ref, omega, kpv[0], kpv[1], dz)[0]
        gm = gamma_at(ref, omega, kmv[0], kmv[1], -dz)[0]
        res = np.max(np.abs(gm.T - J6 @ gp @ J6)) / np.max(np.abs(gp))
        worst = max(worst, float(res))
    print(f"    worst relative residual over 5 wavenumbers = {worst:.3e}")
    report("Gamma(-k; z', z)^T = J6 Gamma(k; z, z') J6", worst < 1e-10)

    # The assembly does not use Gamma pointwise, it uses the z-integrated
    # kernels; integrating the identity above over z and z' and relabelling gives
    #     K^{mn}(-k)^T = J6 K^{nm}(k) J6 ,
    # which is what makes the assembled matrix symmetric.  Checking it separately
    # localises any symmetry defect to the kernel or to the assembly, instead of
    # leaving it to be argued about.
    a = 0.5
    kk1 = np.array([0.8, -2.3, 1.7, 0.05, 4.1])
    kk2 = np.array([-1.4, 0.6, 1.7, -0.02, 0.9])
    kerp = kernels_paper(ref, omega, kk1, kk2, a)
    kerm = kernels_paper(ref, omega, -kk1, -kk2, a)
    worst_k = 0.0
    for m in (0, 1):
        for n in (0, 1):
            lhs = np.swapaxes(kerm[(m, n)], 1, 2)
            rhs = np.einsum("ij,njl,lm->nim", J6, kerp[(n, m)], J6)
            scale = np.max(np.abs(rhs), axis=(1, 2))
            worst_k = max(worst_k, float(np.max(np.abs(lhs - rhs) / scale[:, None, None])))
    print(f"    z-integrated:  K^mn(-k)^T vs J6 K^nm(k) J6 = {worst_k:.3e}")
    report("the z-integrated kernels carry the same reciprocity", worst_k < 1e-10)


def schwinger_matrix(
    ref: ReferenceMedium,
    omega: complex,
    con: MaterialContrast,
    a: float,
    kmax: float,
    nk: int,
    ksplit: float,
    nin: int,
    flip: bool = False,
    multiplicative_only: bool = False,
) -> np.ndarray:
    """The full 9x9 Schwinger matrix, assembled in one pass over term pairs.

    The quadrature grid is symmetric under k -> -k, which matters: the symmetry
    of the result is an algebraic property relating the integrand at +k and -k,
    so it holds to near machine precision on a symmetric grid even when the grid
    is far too coarse to give the VALUE accurately.  That is what makes the
    symmetry a cheap test and an accuracy-independent one.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        con: Material contrast.
        a: Cube half-width.
        kmax: Half-width of the wavenumber box.
        nk: Nodes per outer panel per axis.
        ksplit: Inner panel half-width.
        nin: Nodes in the inner panel per axis.
        flip: Negative control -- reverse the sign of the right leg's i k.
        multiplicative_only: Drop every term that carries a derivative.

    Returns:
        Shape (9, 9) complex.
    """
    k1f, k2f, wq = lateral_grid(ref, omega, kmax, nk, ksplit, nin)
    wts = wq / (2.0 * np.pi) ** 2
    kern = kernels_paper(ref, omega, k1f, k2f, a)
    ffm = form_factors_batch(-k1f, -k2f, a)
    ffp = form_factors_batch(k1f, k2f, a)

    lam_t, mu_t = ref.lam + con.Dlambda, ref.mu + con.Dmu
    fields = [qfield_of(p, lam_t, mu_t, omega) for p in range(9)]
    one = np.ones_like(k1f, dtype=complex)
    kvec = (one, k1f.astype(complex), k2f.astype(complex))

    def stack(ff: tuple, comp: int, d: int, m: int) -> np.ndarray:
        """Weight coefficient of every trial function, as a (9, N) array."""
        return np.stack([weight_coeffs(q0, qg, comp, d, ff)[m] * one for q0, qg in fields])

    cache_m: dict = {}
    cache_p: dict = {}
    terms = delta_a_terms(ref.lam, ref.mu, ref.rho, con, omega)
    if multiplicative_only:
        terms = [t for t in terms if t[3] == 0 and t[4] == 0]
    terms = [t for t in terms if abs(t[2]) > 0.0]

    out = np.zeros((9, 9), dtype=complex)
    for rowl, coll, coefl, dphil, dpsil in terms:
        idxl, sgnl = (rowl + 3, -1.0) if rowl < 3 else (rowl - 3, 1.0)
        fl = one if dpsil == 0 else 1j * kvec[dpsil]
        for rowr, colr, coefr, dphir, dpsir in terms:
            sgn = -1.0 if (flip and dphir != 0) else 1.0
            fr = one if dphir == 0 else -sgn * 1j * kvec[dphir]
            pre = sgnl * coefl * coefr * fl * fr * wts
            for m in (0, 1):
                key_m = (idxl, dphil, m)
                if key_m not in cache_m:
                    cache_m[key_m] = stack(ffm, idxl, dphil, m)
                ck = cache_m[key_m]
                if not np.any(ck):
                    continue
                base = pre * kern[(m, 0)][:, coll, rowr]
                for n in (0, 1):
                    key_p = (colr, dpsir, n)
                    if key_p not in cache_p:
                        cache_p[key_p] = stack(ffp, colr, dpsir, n)
                    cl = cache_p[key_p]
                    if not np.any(cl):
                        continue
                    if n == 1:
                        base = pre * kern[(m, 1)][:, coll, rowr]
                    out += (ck * base) @ cl.T
    return out


def part4_derivative_channels(ref: ReferenceMedium, omega: complex, a: float) -> None:
    """The derivative-carrying channels, tested by the symmetry they must have.

    The density channel of Part 3 settles the basis, the kernel and the
    quadrature, but its contrast operator is MULTIPLICATIVE: it does not exercise
    the i k rule at all.  The channels that do are the ones carrying Dlambda and
    Dmu, and for those the sharp statement available without a new arbiter is
    symmetry.  J6 DeltaA is symmetric -- that is what derived the test space --
    and Gamma is reciprocal by Part 3b, so the assembled matrix must be
    symmetric.  The assembly builds its two legs by opposite rules, transferring
    the left outer derivative onto the test polynomial and carrying the right one
    on the kernel as i k, so this is a genuine test of that bookkeeping rather
    than an identity it satisfies by construction.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        a: Cube half-width.
    """
    print("")
    print("--- Part 4: the derivative channels, by symmetry -------------------")
    con = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    ks = 4.0 * omega.real / ref.beta
    mat = schwinger_matrix(ref, omega, con, a, 60.0, 40, ks, 20)
    asym = float(np.max(np.abs(mat - mat.T)) / np.max(np.abs(mat)))
    print(f"    ||S - S^T|| / ||S||                    = {asym:.3e}")
    report("the assembled Schwinger matrix is symmetric", asym < 1e-10)

    # The test must not be vacuous: the derivative terms have to be doing work.
    mult = schwinger_matrix(ref, omega, con, a, 60.0, 40, ks, 20, multiplicative_only=True)
    share = float(np.max(np.abs(mat - mult)) / np.max(np.abs(mat)))
    print(f"    share of S the derivative terms carry  = {share:.3e}")
    report("the derivative terms are not a negligible part of S", share > 0.1)

    # And it must have teeth on exactly the disputed sign.
    bad = schwinger_matrix(ref, omega, con, a, 60.0, 40, ks, 20, flip=True)
    basym = float(np.max(np.abs(bad - bad.T)) / np.max(np.abs(bad)))
    print(f"    NEGATIVE CONTROL, right-leg i k flipped: {basym:.3e}")
    report("NEGATIVE CONTROL: the wrong sign on the right leg breaks symmetry", basym > 1e-3)

    # The density channel must survive being embedded in the full matrix.
    dens = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=100.0)
    dmat = schwinger_matrix(ref, omega, dens, a, 90.0, 120, ks, 60)
    lam, mu = ref.lam, ref.mu
    kc = lam + 2.0 * mu
    a0 = (lam + 3.0 * mu) / (8.0 * np.pi * mu * kc)
    b0 = (lam + mu) / (8.0 * np.pi * mu * kc)
    rad = omega.real * (1.0 / ref.alpha**3 + 2.0 / ref.beta**3) / (12.0 * np.pi * ref.rho)
    mg = (a0 + b0 / 3.0) * duffy_moment(a, 60, doubled=True) + 1j * (2.0 * a) ** 6 * rad
    want = 1j * omega**5 * dens.Drho**2 * mg
    rel = abs(dmat[0, 0] - want) / abs(want)
    got = dmat[0, 0]
    print(f"    density entry via the full assembly    = {got.real: .4e} {got.imag:+.8e}i  rel {rel:.3e}")
    report("the one-pass assembly reproduces the density channel", rel < 5e-3)


def main() -> int:
    """Run the settlement gate.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 70)
    print("  The augmentation and the Schwinger form -- the settlement, gated")
    print("=" * 70)
    # SI, and the same point the T-matrix gate and the moment export use:
    # k_S a = 0.01, deep in the Rayleigh regime where the affine trial space is
    # the right one.
    ref = ReferenceMedium(5000.0, 3000.0, 2500.0)
    a = 0.5
    omega = 60.0

    part1_indicator_derivative()
    part2_basis(ref, omega)
    part3a_propagator_pointwise(ref, omega)
    part3b_reciprocity(ref, omega)
    part3_density(ref, omega, a)
    part4_derivative_channels(ref, omega, a)

    print("")
    print("=" * 70)
    ok = sum(1 for _, p in _PASS if p)
    print(f"  {ok}/{len(_PASS)} checks passed")
    for label, passed in _PASS:
        if not passed:
            print(f"    FAILED: {label}")
    print("=" * 70)
    return 0 if ok == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
