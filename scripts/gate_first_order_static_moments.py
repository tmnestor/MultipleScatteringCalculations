#!/usr/bin/env python3
"""The cube's static moments A, B and C, from the first-order system.

Task 1 of docs/plans/2026-09-18-first-order-propagator-moment.md: obtain the
cube's propagator moment from d3 q = A q + d rather than from the Kelvin tensor,
and compare against closed forms the first-order machinery knows nothing about:

    B = sqrt(3) (lam + mu) / (6 pi mu (lam + 2 mu)),
    A = B - 1/(3 mu),        C = -(5 - 2 pi/sqrt(3)) B .

All three are obtained INDEPENDENTLY, so the last two relations are tested
rather than used.

WHICH REALISATIONS
------------------
With itens(i,j,k,m) = int_V d_i d_j G_km dV,

    itens(x,y,x,y) = B,   itens(x,x,y,y) = A,   itens(x,x,x,x) = A + 2B + C,

and every one of those carries LATERAL derivatives only -- they become i k with
no d_z and hence no A Gamma product.  Any other realisation needs d_z.

THE DELTA, AND WHY B IS THE EASY ONE
------------------------------------
B has i /= j and so no Eshelby delta: its k integral converges outright, needs no
subtraction, and reaches 1.7e-5.  A and C sit in the delta_ij channels where the
delta lives; their integrands do not decay and they need the UV piece removed.
Convergence tracking the delta's presence is a check on the setup in itself.

SPLIT BY REPRESENTATION, NOT BY MAGNITUDE
-----------------------------------------
The asymptotic integrand is -k_x^2 F(k) C(theta)/k^2 = -cos^2(theta) C(theta)
F(k): a function of DIRECTION only.  Its k-space integral is a pure pole with no
finite part, so adding it back would restore exactly the divergence the
subtraction removed -- the split cannot be closed in k-space at all.  It closes
by representation:

    remainder  ->  stays in k.  Regular at the origin, because -k_x^2 times
                   -C/k^2 is +cos^2(theta) C, bounded.  No taper, no cutoff:
                   independent of the box size to seven digits.
    local      ->  real space.  n = 0 is a delta giving h_0; n /= 0 is a
                   principal-value dipole whose integral over the square is
                   int e^{i n phi} ln R(phi) dphi.  D4 admits n = 0, +-4, ...,
                   and nothing above n = 4 measures.

THE SIGN, PINNED
----------------
(i k_x)(i k_y) = -k_x k_y; dropping the two i's costs exactly a sign, invisible
against |B|.  It is pinned by an evaluation owing nothing to Gamma:

    int_V d_y d_x G_xy dV = closed surface integral of n_y (d_x G_xy) dA,

complete as well as legitimate, because the divergence theorem on a locally
integrable F returns the full DISTRIBUTIONAL integral -- so a delta at the origin
is included automatically.

QUADRATURE
----------
CARTESIAN, on panels cut at the sinc half-period pi/a.  The integrand is a
product of sincs separable in the CARTESIAN components, and on a polar grid it
does not converge even at FIXED cutoff; read as cutoff dependence, that drift
looks exactly like a log divergence and invites a hunt for a delta that is not
there.  Only a resolution study at fixed cutoff separates the two.

The epsilon normalisations of the thesis eigenvectors are neither needed nor
computed: epsilon scales column i of D_z and divides row i of its inverse, so it
cancels out of every projector.  Dropping it removes the one step that cancels,
and lets the whole construction batch.

Run:  conda run -n seismic python scripts/gate_first_order_static_moments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from scripts.gate_thesis_spectral import dz_balanced  # noqa: E402

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def b_closed(ref: ReferenceMedium) -> float:
    """The closed-form shear moment.

    Args:
        ref: Medium.

    Returns:
        sqrt(3)(lam+mu)/(6 pi mu (lam+2mu)).
    """
    return float(np.sqrt(3.0) * (ref.lam + ref.mu) / (6.0 * np.pi * ref.mu * (ref.lam + 2.0 * ref.mu)))


def b_real_space(ref: ReferenceMedium, a: float, n: int = 3000) -> float:
    """int_V d_y d_x G_xy dV by the divergence theorem, owing nothing to Gamma.

    Args:
        ref: Medium.
        a: Cube half-width.
        n: Gauss nodes per face axis.

    Returns:
        The moment.
    """
    kc = ref.lam + 2.0 * ref.mu
    b0 = (ref.lam + ref.mu) / (8.0 * np.pi * ref.mu * kc)
    t, w = np.polynomial.legendre.leggauss(n)
    xs, ws = a * t, a * w
    xg, zg = np.meshgrid(xs, xs, indexing="ij")
    wx, wz = np.meshgrid(ws, ws, indexing="ij")
    r2 = xg**2 + a**2 + zg**2
    # G_xy = b0 x y / r^3 for x != y, so d_x G_xy = b0 y (r^2 - 3 x^2) / r^5
    return 2.0 * float(np.sum(wx * wz * b0 * a * (r2 - 3.0 * xg**2) / np.sqrt(r2) ** 5))


def kz_batch(c: float, omega: float, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """k_{z,c} with the piecewise branch of (kzcDef), batched.

    Args:
        c: Wave speed.
        omega: Angular frequency.
        kx: Lateral wavenumbers, x.
        ky: Lateral wavenumbers, y.

    Returns:
        Complex, same shape.
    """
    arg = (omega / c) ** 2 - ky**2 - kx**2
    root = np.sqrt(np.abs(arg))
    return np.where(arg >= 0.0, root, 1j * root).astype(complex)


def dz_cols_batch(ref: ReferenceMedium, omega: float, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """The thesis eigen matrix, batched.  (Peigen), (SVeigen), (SHeigen).

    Args:
        ref: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumbers, x.
        ky: Lateral wavenumbers, y.

    Returns:
        Shape (N, 6, 6) complex.
    """
    rho, b2 = ref.rho, ref.beta**2
    zp = kz_batch(ref.alpha, omega, kx, ky)
    zs = kz_batch(ref.beta, omega, kx, ky)
    khh2 = (omega / ref.beta) ** 2 - ky**2
    out = np.zeros((kx.shape[0], 6, 6), dtype=complex)
    for col, s in ((0, 1.0), (3, -1.0)):
        out[:, 0, col] = s * 1j * zp
        out[:, 1, col] = 1j * kx
        out[:, 2, col] = 1j * ky
        out[:, 3, col] = rho * (2.0 * b2 * kx**2 + 2.0 * b2 * ky**2 - omega**2)
        out[:, 4, col] = -s * 2.0 * rho * b2 * kx * zp
        out[:, 5, col] = -s * 2.0 * rho * b2 * ky * zp
    for col, s in ((1, 1.0), (4, -1.0)):
        out[:, 0, col] = 1j * kx
        out[:, 1, col] = -s * 1j * zs
        out[:, 3, col] = -s * 2.0 * rho * b2 * kx * zs
        out[:, 4, col] = rho * (omega**2 - 2.0 * b2 * kx**2 - b2 * ky**2)
        out[:, 5, col] = -rho * b2 * kx * ky
    for col, s in ((2, 1.0), (5, -1.0)):
        out[:, 0, col] = -s * ky * zs
        out[:, 1, col] = -kx * ky
        out[:, 2, col] = khh2
        out[:, 3, col] = 2j * ky * rho * (b2 * kx**2 + b2 * ky**2 - omega**2)
        out[:, 4, col] = -s * 2j * rho * b2 * kx * ky * zs
        out[:, 5, col] = s * 1j * zs * rho * (omega**2 - 2.0 * b2 * ky**2)
    return out


def g_zint(
    ref: ReferenceMedium, omega: float, kx: np.ndarray, ky: np.ndarray, a: float, row: int, col: int
) -> np.ndarray:
    """int_{-a}^{a} G_{row,col}(k; -z') dz' with the receiver at the cube centre.

    G_ij = -Gamma[i, 3+j] in the thesis basis: a force f_j enters row 3+j with a
    minus sign and the displacement comes out in rows 0..2.

    Args:
        ref: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumbers, x.
        ky: Lateral wavenumbers, y.
        a: Cube half-width.
        row: Displacement component, 0..2 for (z, x, y).
        col: Force row, 3..5 for f_(z, x, y).

    Returns:
        Shape (N,) complex.
    """
    k = np.maximum(np.hypot(kx, ky), omega / ref.beta)
    s = np.sqrt(ref.mu * k)
    scl = np.stack([s, s, s, 1.0 / s, 1.0 / s, 1.0 / s], axis=1)
    dmat = dz_cols_batch(ref, omega, kx, ky) * scl[:, :, None]
    inv = np.linalg.inv(dmat)
    zp = kz_batch(ref.alpha, omega, kx, ky)
    zs = kz_batch(ref.beta, omega, kx, ky)
    lam6 = 1j * np.stack([zp, zs, zs, -zp, -zs, -zs], axis=1)
    tot = np.zeros(kx.shape[0], dtype=complex)
    # z' < 0 is below the receiver, so downgoing; z' > 0 upgoing, with the sign
    # of (FLdef)'s jump.
    for modes, sgn, lo, hi in ((range(3), 1.0, -a, 0.0), (range(3, 6), -1.0, 0.0, a)):
        for i in modes:
            e = lam6[:, i]
            val = np.where(np.abs(e) < 1e-300, hi - lo, (np.exp(-e * lo) - np.exp(-e * hi)) / e)
            tot += sgn * val * (dmat[:, row, i] / scl[:, row]) * (inv[:, i, col] * scl[:, col])
    return -tot


def sinc_axis(a: float, kmax: float, nper: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric Gauss panels cut at the sinc half-period pi/a.

    Args:
        a: Cube half-width.
        kmax: Cutoff.
        nper: Nodes per half-period.

    Returns:
        (nodes, weights).
    """
    half = np.pi / a
    edges = [0.0]
    while edges[-1] < kmax:
        edges.append(min(edges[-1] + half, kmax))
    ed = np.array(edges)
    ed = np.concatenate([-ed[::-1][:-1], ed])
    t, w = np.polynomial.legendre.leggauss(nper)
    kk, ww = [], []
    for lo, hi in zip(ed[:-1], ed[1:], strict=True):
        kk.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
        ww.append(0.5 * (hi - lo) * w)
    return np.concatenate(kk), np.concatenate(ww)


def b_first_order(
    ref: ReferenceMedium, omega: float, a: float, kmax: float, nper: int, flip: bool = False
) -> complex:
    """B from the first-order system, by the lateral k integral.

    Args:
        ref: Medium.
        omega: Angular frequency.
        a: Cube half-width.
        kmax: Cutoff.
        nper: Nodes per sinc half-period.
        flip: Negative control -- drop the two factors of i.

    Returns:
        The moment.
    """
    kv, kw = sinc_axis(a, kmax, nper)
    k1, k2 = np.meshgrid(kv, kv, indexing="ij")
    w1, w2 = np.meshgrid(kw, kw, indexing="ij")
    kx, ky, wt = k1.ravel(), k2.ravel(), (w1 * w2).ravel()

    def box(k: np.ndarray) -> np.ndarray:
        safe = np.where(np.abs(k) < 1e-12, 1.0, k)
        return np.where(np.abs(k) < 1e-12, 2.0 * a, 2.0 * np.sin(k * a) / safe)

    # (i k_x)(i k_y) = -k_x k_y.  Dropping the i's is the whole of the sign.
    deriv = (1.0 if flip else -1.0) * kx * ky
    tot = 0.0 + 0.0j
    for lo in range(0, kx.size, 400000):
        sl = slice(lo, lo + 400000)
        val = deriv[sl] * box(kx[sl]) * box(ky[sl]) * g_zint(ref, omega, kx[sl], ky[sl], a, 1, 5)
        tot += np.sum(wt[sl] * val)
    return complex(tot / (2.0 * np.pi) ** 2)


def c_asymptote(
    ref: ReferenceMedium, omega: float, a: float, theta: np.ndarray, row: int, col: int
) -> np.ndarray:
    """C(theta), from a two-point fit of k^2 g(k) = C + C1/k.

    Both radii sit where the thesis representation is accurate (see the valid
    range recorded by gate_thesis_spectral).  Measured, k^2 g is flat to five
    digits at k = 200, 400 and 800, so C is a genuine asymptote and the fitted
    C1 is at the level of round-off.

    Args:
        ref: Medium.
        omega: Angular frequency.
        a: Cube half-width.
        theta: Angles.
        row: Displacement component.
        col: Force row.

    Returns:
        Same shape as theta.
    """
    k1, k2 = 200.0, 400.0
    g1 = (k1**2 * g_zint(ref, omega, k1 * np.cos(theta), k1 * np.sin(theta), a, row, col)).real
    g2 = (k2**2 * g_zint(ref, omega, k2 * np.cos(theta), k2 * np.sin(theta), a, row, col)).real
    c1 = (g1 - g2) / (1.0 / k1 - 1.0 / k2)
    return g1 - c1 / k1


def moment_remainder(
    ref: ReferenceMedium, omega: float, a: float, kmax: float, nper: int, row: int, col: int
) -> float:
    """The k-space part of int_V d_x d_x G dV, with the asymptote subtracted.

    Regular at the origin, which is the point: -k_x^2 times -C/k^2 is
    +cos^2(theta) C, bounded.  So the subtraction needs no taper, no cutoff and
    no matching of one region to another -- and the result is independent of the
    box size to seven digits across a fourfold change, where the unsubtracted
    integral drifted by per cent.

    Args:
        ref: Medium.
        omega: Angular frequency.
        a: Cube half-width.
        kmax: Half-width of the Cartesian box.
        nper: Nodes per sinc half-period.
        row: Displacement component.
        col: Force row.

    Returns:
        The remainder.
    """
    kv, kw = sinc_axis(a, kmax, nper)
    k1g, k2g = np.meshgrid(kv, kv, indexing="ij")
    w1, w2 = np.meshgrid(kw, kw, indexing="ij")
    kx, ky, wt = k1g.ravel(), k2g.ravel(), (w1 * w2).ravel()

    def box(k: np.ndarray) -> np.ndarray:
        safe = np.where(np.abs(k) < 1e-12, 1.0, k)
        return np.where(np.abs(k) < 1e-12, 2.0 * a, 2.0 * np.sin(k * a) / safe)

    tot = 0.0
    for lo in range(0, kx.size, 300000):
        sl = slice(lo, lo + 300000)
        x, y, ww = kx[sl], ky[sl], wt[sl]
        ksq = x**2 + y**2
        gg = g_zint(ref, omega, x, y, a, row, col).real
        asym = c_asymptote(ref, omega, a, np.arctan2(y, x), row, col) / np.maximum(ksq, 1e-300)
        tot += float(np.sum(ww * (-(x**2) * box(x) * box(y) * (gg - asym))))
    return tot / (2.0 * np.pi) ** 2


def moment_local(
    ref: ReferenceMedium, omega: float, a: float, row: int, col: int, nmax: int = 16, nth: int = 4096
) -> float:
    """The local part, evaluated in REAL SPACE where it is finite.

    The asymptotic integrand is -k_x^2 F(k) C/k^2 = -cos^2(theta) C(theta) F(k),
    a function of DIRECTION only.  Its k-space integral is a pure pole with no
    finite part -- adding it back restores the divergence the subtraction
    removed -- so it is evaluated by representation instead:

      * n = 0 is a delta, contributing h_0 times 1_S(0) = h_0;
      * n != 0 is a principal-value dipole of degree -2, since the inverse
        transform of e^{i n theta} is (-i)^n |n| e^{i n phi} / 2 pi r^2, using
        int_0^inf k J_n(kr) dk = |n| / r^2.  Its integral over the square is
        int e^{i n phi} ln R(phi) dphi with R = a / max(|cos|, |sin|): the
        log-epsilon of the radial integral cancels because e^{i n phi} has zero
        angular mean.

    D4 symmetry admits n = 0, +-4, +-8, ...; measured, everything above n = 4
    contributes nothing, which is the check that the decomposition is right.

    Args:
        ref: Medium.
        omega: Angular frequency.
        a: Cube half-width.
        row: Displacement component.
        col: Force row.
        nmax: Highest harmonic retained.
        nth: Angular samples.

    Returns:
        The local term.
    """
    th = np.arange(nth) * 2.0 * np.pi / nth
    hh = -(np.cos(th) ** 2) * c_asymptote(ref, omega, a, th, row, col)
    rad = a / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
    hn = np.fft.fft(hh) / nth
    ln = np.fft.fft(np.log(rad)) / nth
    tot = float(hn[0].real)
    for n in list(range(1, nmax + 1)) + list(range(-nmax, 0)):
        lint = 2.0 * np.pi * ln[(-n) % nth]
        tot += float((hn[n % nth] * ((-1j) ** n) * abs(n) / (2.0 * np.pi) * lint).real)
    return tot


def moment_dxx(
    ref: ReferenceMedium, omega: float, a: float, row: int, col: int, kmax: float = 400.0, nmax: int = 16
) -> float:
    """int_V d_x d_x G_{row,col} dV, the two parts assembled.

    Args:
        ref: Medium.
        omega: Angular frequency.
        a: Cube half-width.
        row: Displacement component.
        col: Force row.
        kmax: Box half-width for the remainder.
        nmax: Highest harmonic in the local term.

    Returns:
        The moment.
    """
    return moment_remainder(ref, omega, a, kmax, 12, row, col) + moment_local(ref, omega, a, row, col, nmax)


def main() -> int:
    """Read B off the first-order system.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 76)
    print("  The cube's shear moment B, from d3 q = A q + d")
    print("=" * 76)
    ref = ReferenceMedium(5000.0, 3000.0, 2500.0)
    a = 0.5
    bc = b_closed(ref)

    print("")
    print("--- 1: the batched spectral machinery, against the scalar gate ----")
    kk1 = np.array([0.03, 0.7, 5.0, 60.0])
    kk2 = np.array([-0.01, 1.3, -2.0, 25.0])
    worst = 0.0
    for i in range(kk1.size):
        dmat, inv, scl, _ = dz_balanced(ref, 60.0, float(kk1[i]), float(kk2[i]))
        bat = dz_cols_batch(ref, 60.0, kk1[i : i + 1], kk2[i : i + 1])[0]
        # the scalar route carries the epsilon normalisation, the batched one
        # does not, so compare COLUMN DIRECTIONS rather than columns
        for c in range(6):
            # dz_balanced returns the BALANCED matrix; the batched one is raw.
            u = (dmat[:, c] / scl) / np.linalg.norm(dmat[:, c] / scl)
            v = bat[:, c] / np.linalg.norm(bat[:, c])
            worst = max(worst, float(1.0 - abs(np.vdot(u, v))))
    print(f"    worst 1 - |<scalar col | batched col>| = {worst:.3e}")
    report("the batched eigenvectors are the scalar ones", worst < 1e-12)

    print("")
    print("--- 2: the sign, pinned in real space -----------------------------")
    print("    int_V d_y d_x G_xy dV = surface integral of n_y (d_x G_xy), which")
    print("    returns the FULL distributional value, delta included.")
    bs = b_real_space(ref, a)
    rel_s = abs(bs - bc) / bc
    print(f"    real-space surface integral = {bs:+.10e}")
    print(f"    closed-form B               = {bc:+.10e}")
    print(f"    relative difference         = {rel_s:.3e}")
    report("itens(x,y,x,y) IS B, and the sign is POSITIVE", rel_s < 1e-7)

    print("")
    print("--- 3: B from the first-order system ------------------------------")
    print("    omega = 6 so that (k_S a)^2 = 1e-6 and the static form is the")
    print("    right comparison; the residual is quadrature, not frequency.")
    prev = None
    for kmax, nper in ((400.0, 8), (400.0, 12), (800.0, 12), (1200.0, 14)):
        got = b_first_order(ref, 6.0, a, kmax, nper)
        rel = abs(got.real - bc) / bc
        print(f"    kmax={kmax:7.0f} nper={nper:3d}   {got.real:+.10e}   rel {rel:.3e}")
        prev = got
    assert prev is not None
    rel_f = abs(prev.real - bc) / bc
    report("B from the first-order system matches the closed form", rel_f < 1e-4)
    # carried into section 5: the relations must be tested on FIRST-ORDER
    # numbers, so B's own derived value is used there, never the closed form
    bc_fo = prev.real

    print("")
    print("--- 4: the two factors of i are load-bearing ----------------------")
    bad = b_first_order(ref, 6.0, a, 400.0, 8, flip=True)
    print(f"    with (i k_x)(i k_y) dropped: {bad.real:+.10e}   ratio {bad.real / bc:+.6f}")
    report("NEGATIVE CONTROL: dropping them flips the sign exactly", abs(bad.real / bc + 1.0) < 1e-3)

    print("")
    print("--- 5: A and C, which DO carry the Eshelby delta ------------------")
    print("    itens(x,x,y,y) = A and itens(x,x,x,x) = A + 2B + C, both with")
    print("    lateral derivatives only.  Unlike B they sit in the delta_ij")
    print("    channels where the delta lives, so their k integrals diverge and")
    print("    need the asymptote taken out -- B, with i /= j, has no delta and")
    print("    converges outright.  That correspondence is a check in itself.")
    a_cl = bc - 1.0 / (3.0 * ref.mu)
    c_cl = -(5.0 - 2.0 * np.pi / np.sqrt(3.0)) * bc

    print("    the remainder converges in the BOX SIZE, with no tail at all:")
    rems = [moment_remainder(ref, 6.0, a, km, 12, 2, 5) for km in (100.0, 200.0, 400.0)]
    for km, r in zip((100.0, 200.0, 400.0), rems, strict=True):
        print(f"      kmax={km:6.0f}   {r:+.10e}")
    spread = max(abs(r - rems[-1]) for r in rems) / abs(rems[-1])
    print(f"      spread across a fourfold box change: {spread:.2e}")
    report("the subtracted remainder is cutoff-independent to seven digits", spread < 1e-6)

    print("    the local term saturates at n = 4, as D4 requires:")
    locs = [moment_local(ref, 6.0, a, 2, 5, nmax=n) for n in (0, 4, 8, 32)]
    for n, lv in zip((0, 4, 8, 32), locs, strict=True):
        print(f"      nmax={n:3d}   {lv:+.10e}")
    print(f"      n=0 alone is wrong by {abs(locs[0] - locs[-1]) / abs(locs[-1]):.1%}")
    hi_rel = abs(locs[2] - locs[3]) / abs(locs[3])
    print(f"      n = 8 against n = 32: {hi_rel:.2e}")
    report("nothing above n = 4 contributes, to 1e-7", hi_rel < 1e-7)
    report(
        "NEGATIVE CONTROL: the PV dipole is load-bearing, not decorative",
        abs(locs[0] - locs[1]) > 0.01 * abs(locs[1]),
    )

    a_fo = moment_dxx(ref, 6.0, a, 2, 5)
    s_fo = moment_dxx(ref, 6.0, a, 1, 4)
    c_fo = s_fo - a_fo - 2.0 * bc_fo
    print(f"    A = {a_fo:+.10e}   closed {a_cl:+.10e}   rel {abs(a_fo - a_cl) / abs(a_cl):.3e}")
    print(f"    C = {c_fo:+.10e}   closed {c_cl:+.10e}   rel {abs(c_fo - c_cl) / abs(c_cl):.3e}")
    report("A from the first-order system matches the closed form", abs(a_fo - a_cl) / abs(a_cl) < 2e-3)
    report("C from the first-order system matches the closed form", abs(c_fo - c_cl) / abs(c_cl) < 5e-3)

    print("")
    print("--- 6: the structural relations, TESTED not assumed ---------------")
    print("    Every number below came out of d3 q = A q + d; nothing is")
    print("    substituted from the closed forms.")
    rel_a = abs(a_fo - (bc_fo - 1.0 / (3.0 * ref.mu))) / abs(a_fo)
    rel_c = abs(c_fo - (-(5.0 - 2.0 * np.pi / np.sqrt(3.0)) * bc_fo)) / abs(c_fo)
    print(f"    A  vs  B - 1/3mu           {rel_a:.3e}")
    print(f"    C  vs  -(5 - 2pi/sqrt3) B  {rel_c:.3e}")
    report("A = B - 1/3mu holds between first-order quantities", rel_a < 2e-3)
    report("C = -(5 - 2pi/sqrt3) B holds between first-order quantities", rel_c < 5e-3)

    print("")
    print("=" * 76)
    ok = sum(1 for _, passed in _PASS if passed)
    print(f"  {ok}/{len(_PASS)} checks passed")
    for label, passed in _PASS:
        if not passed:
            print(f"    FAILED: {label}")
    print("=" * 76)
    return 0 if ok == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
