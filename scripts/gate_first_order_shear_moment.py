#!/usr/bin/env python3
"""The cube's shear moment B, from the first-order system.

This is the completion criterion of Task 1 in
docs/plans/2026-09-18-first-order-propagator-moment.md: derive the cube's
propagator moment from d3 q = A q + d rather than from the Kelvin tensor, and
read the shear moment off against

    B = sqrt(3) (lambda + mu) / (6 pi mu (lambda + 2 mu)).

WHICH REALISATION, AND WHY IT MATTERS
-------------------------------------
B multiplies (delta_ik delta_jm + delta_im delta_jk) in the moment tensor, so
itens(x, y, x, y) = B exactly.  That realisation uses ONLY LATERAL derivatives,
which become i k in the lateral Fourier domain with no z-derivative and hence no
A Gamma product:

    B = int_V d_y d_x G_xy dV
      = (2 pi)^-2 int d2k (i k_x)(i k_y) F(k) int_{-a}^{a} G_xy(k; -z') dz' .

Every other realisation of B involves d_z and would need A Gamma.

THE SIGN, PINNED
----------------
The two factors of i give (i k_x)(i k_y) = -k_x k_y, and dropping them costs
exactly a sign -- which is invisible against |B| and would have been reported as
success.  It is pinned here by an independent real-space evaluation that owes
nothing to the first-order machinery:

    int_V d_y d_x G_xy dV = closed surface integral of n_y (d_x G_xy) dA,

valid because the divergence theorem for a locally integrable F returns the FULL
distributional integral -- so any delta at the origin is included automatically,
and d_x G_xy goes as 1/r^2, integrable in three dimensions and non-singular on
faces that stand at distance a from the centre.  Only the y = +-a faces
contribute and their integrands add, the integrand being odd in y.

QUADRATURE
----------
CARTESIAN, on panels cut at the sinc half-period pi/a.  The integrand is
4 sin(k_x a) sin(k_y a) times the kernel -- a product of sincs separable in the
CARTESIAN components -- and on a polar grid it does not converge even at fixed
cutoff.  Read as cutoff dependence, that drift looks exactly like a logarithmic
divergence and invites a hunt for a missing Eshelby delta.  There is none; the
integral converges.

The normalisation factors epsilon of the thesis eigenvectors are not needed and
are not computed: epsilon scales column i of D_z and divides row i of its
inverse, so it cancels out of every projector.  Dropping it removes the one step
that cancels, and lets the whole construction batch.

Run:  conda run -n seismic python scripts/gate_first_order_shear_moment.py
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


def gxy_zint(ref: ReferenceMedium, omega: float, kx: np.ndarray, ky: np.ndarray, a: float) -> np.ndarray:
    """int_{-a}^{a} G_xy(k; -z') dz' with the receiver at the cube centre.

    G_ij = -Gamma[i, 3+j] in the thesis basis: a force f_j enters row 3+j with a
    minus sign and the displacement comes out in rows 0..2.

    Args:
        ref: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumbers, x.
        ky: Lateral wavenumbers, y.
        a: Cube half-width.

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
            tot += sgn * val * (dmat[:, 1, i] / scl[:, 1]) * (inv[:, i, 5] * scl[:, 5])
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
        val = deriv[sl] * box(kx[sl]) * box(ky[sl]) * gxy_zint(ref, omega, kx[sl], ky[sl], a)
        tot += np.sum(wt[sl] * val)
    return complex(tot / (2.0 * np.pi) ** 2)


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

    print("")
    print("--- 4: the two factors of i are load-bearing ----------------------")
    bad = b_first_order(ref, 6.0, a, 400.0, 8, flip=True)
    print(f"    with (i k_x)(i k_y) dropped: {bad.real:+.10e}   ratio {bad.real / bc:+.6f}")
    report("NEGATIVE CONTROL: dropping them flips the sign exactly", abs(bad.real / bc + 1.0) < 1e-3)

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
