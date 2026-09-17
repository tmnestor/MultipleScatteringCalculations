"""GATE: does the sinc^1 SPECTRAL average reproduce the real-space Gauss average?

`gate_sinc1_uv_convergence.py` cleared the UV obstruction for the single-average
form factor, but with one scope limit stated at the time: it weighted the
angular integral with 1, not with the strain kernel Gamma(k_hat). The
measure-zero argument it relies on -- that the slowly-decaying directions are
the coordinate AXES and the angular measure suppresses them -- could in
principle be undone by a direction-dependent weight that is large exactly there.
This closes that gap, and it is the blocking measurement: until the spectral
route is verified against something independent, the sinc^1 build is not
justified.

WHAT IS COMPUTED. In the static limit the elastostatic kernel is

    G~_ij(k)      = k^i k^j / ((lam+2mu) k^2) + (delta_ij - k^i k^j) / (mu k^2)
    Gdd~_ijkl(k)  = -k_k k_l G~_ij(k)
                  = -k^k k^l [ k^i k^j/(lam+2mu) + (delta_ij - k^i k^j)/mu ]

(hats denoting unit components), which is O(1) and DIRECTION-ONLY -- the k^2
cancels. That is precisely the "Gamma is O(1)" the previous gate assumed away.
The cell-averaged propagator is then

    <Gdd>_ijkl(R) = Int d^3k/(2pi)^3 [prod_m sinc(k_m h)]^n Gamma~_ijkl(k) e^{ik.R}

with n = 1 the SINGLE (source-cell) average and n = 2 the DOUBLE (Galerkin) one.

⚠ WHY THE STATIC LIMIT IS THE RIGHT TEST AND NOT A DODGE. The UV question is a
LARGE-k question, and at large k the omega^2 terms are subleading -- the kernel
tends to its static form regardless of frequency. So the static case exercises
exactly the behaviour in doubt, while avoiding the poles at |k| = omega/alpha,
omega/beta that force the subtracted formulation in
`Mathematica/InterVoxelPropagatorWS3.wl`. What it does NOT test is the dynamic
part, which is a separate question and is not claimed here.

⚠⚠ THE REAL-SPACE REFERENCE IS **NOT** `elastodynamic_greens_deriv` AT SMALL
OMEGA, AND THE FIRST VERSION OF THIS GATE FAILED BECAUSE IT WAS. That routine
carries P = 1/(4 pi rho omega^2), so phi and psi must cancel to O(omega^2) for a
finite static limit; at k_S r ~ 1e-6 that destroys ~12 of 16 digits. Measured:
the omega -> 0 "static" reference moved 262% between omega = 6e-3 and 6e-4 in a
component that must be omega-independent to 1e-12. The gate's control arm caught
it and aborted rather than reporting the n=1 number -- which is what the control
is for.

So the reference here is the ANALYTIC static Kelvin second derivative, validated
two independent ways:
  * against `elastodynamic_greens_deriv` on an omega PLATEAU -- agreement
    6.0e-8 at omega = 0.6, rising at large omega as the real dynamic
    corrections (1.5e-4 at omega = 60, matching (k_S r)^2) and at small omega as
    the cancellation (3.8e-2 at 6e-4, garbage by 6e-5);
  * by the Navier equation mu Gdd_ij,kk + (lam+mu) Gdd_ik,jk = 0 away from the
    source, which holds to 1.2e-15 relative to the term scale mu|Gdd|.

THE CONTROL IS THE WHOLE POINT OF RUNNING BOTH POWERS. n = 2 must reproduce the
real-space double average, which the package already validates (it is
`inter_voxel_propagator_9x9`, whose FACE/EDGE/CORNER constants came from this
same spectral integral). A sign slip, a factor of 2, or a wrong Fourier
convention in the kernel below shows up there FIRST, and if it does the n = 1
number is withheld rather than reported. Reporting a new measurement whose
machinery has not reproduced a known one is how a wrong result gets believed.

⚠ THE TEST OFFSET IS 2d, NOT THE FACE OFFSET d, and that cost a second control
failure before it was noticed. At face contact the DOUBLE average integrates the
tent function over [-d, d], so the source-field separation reaches ZERO and the
integrand is singular. The package's analytic O_h tables exist precisely because
"no quadrature converges" there, and tensor-product double-cube Gauss is a
RECORDED trap: it converges to a BIASED limit at face contact, which is why the
committed FACE/EDGE/CORNER constants are used instead. A Gauss reference at that
offset is a known-bad number, so the control was rejecting a correct spectral
value. At 2d the tent spans [d, 3d] and both averages are regular. (The SINGLE
average was never affected -- its source cell spans [d/2, 3d/2] and stays clear
of the origin at every offset.)

Run:  conda run -n seismic python scripts/gate_sinc1_vs_gauss_9x9.py
SI units (m, m/s, kg/m3, Pa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.resonance_tmatrix import elastodynamic_greens_deriv  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
MU = REF.rho * REF.beta**2
LAM = REF.rho * REF.alpha**2 - 2.0 * MU
NU = LAM / (2.0 * (LAM + MU))
D = 1.0  # cell side
H = 0.5 * D  # half-width

#: Where `elastodynamic_greens_deriv` is BOTH static and numerically clean --
#: measured, not assumed (see the plateau in the module docstring). Used only to
#: re-validate the analytic form below, never as the reference itself.
OMEGA_PLATEAU = 0.6


def kelvin_gdd(r_vec: np.ndarray) -> np.ndarray:
    """d_k d_l G_ij for the STATIC Kelvin tensor -- no 1/omega^2 cancellation.

    G_ij = A[(3-4nu) delta_ij / r + r_i r_j / r^3],  A = 1/(16 pi mu (1-nu)).
    """
    r_vec = np.asarray(r_vec, dtype=float)
    r = float(np.linalg.norm(r_vec))
    a = 1.0 / (16.0 * np.pi * MU * (1.0 - NU))
    d, x = np.eye(3), r_vec
    r3, r5, r7 = r**3, r**5, r**7

    term_a = (3.0 - 4.0 * NU) * (
        -np.einsum("ij,kl->ijkl", d, d) / r3 + 3.0 * np.einsum("ij,k,l->ijkl", d, x, x) / r5
    )
    term_b = (
        (np.einsum("ik,jl->ijkl", d, d) + np.einsum("jk,il->ijkl", d, d)) / r3
        - 3.0 * (np.einsum("ik,j,l->ijkl", d, x, x) + np.einsum("jk,i,l->ijkl", d, x, x)) / r5
        - 3.0
        * (
            np.einsum("il,j,k->ijkl", d, x, x)
            + np.einsum("jl,i,k->ijkl", d, x, x)
            + np.einsum("kl,i,j->ijkl", d, x, x)
        )
        / r5
        + 15.0 * np.einsum("i,j,k,l->ijkl", x, x, x, x) / r7
    )
    return a * (term_a + term_b)


#: Components compared. Chosen to span the tensor structures rather than to
#: flatter: a pure-axial one, a pure-shear one, and a cross term.
COMPONENTS = ((0, 0, 0, 0), (0, 1, 0, 1), (0, 0, 1, 1))


def gdd_avg_realspace(r_vec: np.ndarray, ng: int, *, double: bool) -> np.ndarray:
    """(1/V) Int_cell Gdd(R - u) du, by Gauss -- the trusted reference.

    Mirrors `slab_scattering._cell_averaged_propagator`'s node/weight
    construction: the double average is a single integral against the
    convolution of two cell indicators, i.e. a product of tent functions of
    half-width d, integrated on [-d,0] and [0,d] separately because the tent
    has a kink at 0.
    """
    x, w = np.polynomial.legendre.leggauss(ng)
    if double:
        nodes, wts = [], []
        for lo, hi in ((-D, 0.0), (0.0, D)):
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            for xi, wi in zip(x, w, strict=True):
                t = mid + half * xi
                nodes.append(t)
                wts.append(wi * half * (1.0 - abs(t) / D) / D)
    else:
        nodes = list(0.5 * D * x)
        wts = list(0.5 * w)

    acc = np.zeros((3, 3, 3, 3), dtype=float)
    for i, ui in enumerate(nodes):
        for j, uj in enumerate(nodes):
            for k, uk in enumerate(nodes):
                off = np.array([ui, uj, uk])
                acc += (wts[i] * wts[j] * wts[k]) * kelvin_gdd(r_vec - off)
    return acc.astype(complex)


def validate_kelvin() -> bool:
    """Re-check the analytic form here rather than trusting the docstring."""
    r_probe = np.array([1.0, 0.4, 0.2])
    ana = kelvin_gdd(r_probe)
    _, _, dyn = elastodynamic_greens_deriv(r_probe, OMEGA_PLATEAU, REF)
    plateau = float(np.max(np.abs(dyn.real - ana)) / np.max(np.abs(ana)))

    # Navier, away from the source. Normalise by the TERM scale mu|Gdd|, not by
    # |Gdd| -- the terms are O(mu |Gdd|) and the residual must be small against
    # THEM, which is the comparison that means anything.
    resid = MU * np.einsum("ijkk->ij", ana) + (LAM + MU) * np.einsum("ikjk->ij", ana)
    navier = float(np.max(np.abs(resid)) / (MU * np.max(np.abs(ana))))

    print("\n  analytic static Kelvin Gdd, re-validated here:")
    print(f"    vs elastodynamic_greens_deriv at omega = {OMEGA_PLATEAU}: {plateau:.2e}")
    print(f"    Navier residual / term scale:                    {navier:.2e}")
    ok = plateau < 1e-6 and navier < 1e-12
    print(f"    reference is sound: {ok}")
    return ok


def gdd_avg_spectral(
    r_vec: np.ndarray, power: int, kmax: float, dk: float, samples: int = 4
) -> dict[tuple[int, int, int, int], np.ndarray]:
    """Cumulative Int_0^K dk k^2 (1/2pi^2) <[prod sinc]^n Gamma~ e^{ik.R}>_Omega.

    Returns the cumulative integral on the k grid, per component, so the
    approach to the reference can be read rather than a single endpoint.

    Normalisation: Int d^3k/(2pi)^3 f = (1/(2pi)^3) Int dk k^2 Int dOmega f, and
    with Abar the angular MEAN that is (4pi/(2pi)^3) Int dk k^2 Abar
    = (1/(2 pi^2)) Int dk k^2 Abar.
    """
    kgrid = np.arange(dk, kmax + dk, dk)
    r_norm = max(float(np.linalg.norm(r_vec)), H)
    out = {c: np.zeros(len(kgrid), dtype=complex) for c in COMPONENTS}

    for idx, k in enumerate(kgrid):
        # Resolve BOTH oscillations: the form factor on 1/(k h) and the phase
        # e^{ik.R} on 1/(k|R|). The phase is the faster one whenever |R| > h.
        n_ang = max(60, int(samples * k * max(H, r_norm)))
        mu, w_mu = np.polynomial.legendre.leggauss(n_ang)
        phi = 2.0 * np.pi * np.arange(n_ang) / n_ang
        w_phi = 2.0 * np.pi / n_ang

        st = np.sqrt(1.0 - mu**2)
        nx = np.outer(st, np.cos(phi))
        ny = np.outer(st, np.sin(phi))
        nz = np.repeat(mu[:, None], n_ang, axis=1)
        nhat = (nx, ny, nz)

        ff = (
            np.sinc(k * nx * H / np.pi) * np.sinc(k * ny * H / np.pi) * np.sinc(k * nz * H / np.pi)
        ) ** power
        phase = np.exp(1j * k * (nx * r_vec[0] + ny * r_vec[1] + nz * r_vec[2]))
        wt = w_mu[:, None] * w_phi * ff * phase

        for c in COMPONENTS:
            i, j, kk, ll = c
            ninj = nhat[i] * nhat[j]
            long_part = ninj / (LAM + 2.0 * MU)
            trans_part = ((1.0 if i == j else 0.0) - ninj) / MU
            gam = -nhat[kk] * nhat[ll] * (long_part + trans_part)
            abar = float(np.sum((wt * gam).real)) + 1j * float(np.sum((wt * gam).imag))
            out[c][idx] = abar / (4.0 * np.pi)

    for c in COMPONENTS:
        out[c] = np.cumsum(kgrid**2 * out[c]) * dk / (2.0 * np.pi**2)
    return out


def run_arm(power: int, r_vec: np.ndarray, kmax: float, dk: float) -> tuple[float, float]:
    """Returns (relative error at kmax, relative error at kmax/2) for one power."""
    ref = gdd_avg_realspace(r_vec, 8, double=(power == 2))
    spec = gdd_avg_spectral(r_vec, power, kmax, dk)
    kgrid = np.arange(dk, kmax + dk, dk)

    label = {1: "n=1  SINGLE (collocation)", 2: "n=2  DOUBLE (control)"}[power]
    print(f"\n    {label}")
    print(f"      {'component':>12} {'real-space Gauss':>20} {'spectral (K=max)':>20} {'rel':>10}")
    errs_full, errs_half = [], []
    i_half = len(kgrid) // 2 - 1
    for c in COMPONENTS:
        r = ref[c].real
        s_full, s_half = spec[c][-1].real, spec[c][i_half].real
        scale = max(abs(r), 1e-300)
        errs_full.append(abs(s_full - r) / scale)
        errs_half.append(abs(s_half - r) / scale)
        print(f"      {str(c):>12} {r:20.10e} {s_full:20.10e} {errs_full[-1]:10.2e}")
    return max(errs_full), max(errs_half)


def main() -> int:
    print("=" * 78)
    print("GATE: sinc^1 SPECTRAL average vs the real-space Gauss average")
    print("=" * 78)
    # ⚠ NOT the face offset (D,0,0), and this cost a second control failure.
    # At face contact the DOUBLE average integrates the tent over [-d, d], so
    # the source-field separation reaches ZERO and the integrand is singular:
    # the library's own O_h tables exist precisely because "no quadrature
    # converges" there, and tensor-product double-cube Gauss is documented to
    # converge to a BIASED limit at face contact. Using it as a control
    # compares a correct spectral value against a known-bad reference.
    # At 2D the tent spans [d, 3d] and both averages are regular.
    r_vec = np.array([2.0 * D, 0.0, 0.0])
    kmax, dk = 120.0, 0.1
    print(f"\n  offset R = {r_vec}, d = {D}, K_max = {kmax:.0f}, dk = {dk}")
    print(f"  lam = {LAM:.4e}, mu = {MU:.4e}, nu = {NU:.6f}  (static kernel)")

    if not validate_kelvin():
        print("\n  ABORT -- the real-space reference itself does not validate.")
        return 1
    print("\n  The spectral integral converges to its target with a 1/K envelope")
    print("  (oscillatory tail), so the bar is set by the CONTROL arm, not by a")
    print("  tolerance chosen in advance -- and the control must clear it first.")

    err2_full, err2_half = run_arm(2, r_vec, kmax, dk)
    print(f"\n      control error: {err2_half:.2e} at K/2  ->  {err2_full:.2e} at K")
    control_ok = err2_full < 0.05 and err2_full < err2_half
    print(f"      CONTROL reproduces the validated double average: {control_ok}")
    if not control_ok:
        print("\n  ABORT -- the spectral kernel does not reproduce the known")
        print("  double average, so it has a sign, factor or convention error.")
        print("  The n=1 result is NOT reported: a new measurement whose")
        print("  machinery cannot reproduce a known one proves nothing.")
        return 1

    err1_full, err1_half = run_arm(1, r_vec, kmax, dk)
    print(f"\n      single error:  {err1_half:.2e} at K/2  ->  {err1_full:.2e} at K")

    # ⚠ THE CRITERION IS CONVERGENCE TO THE REFERENCE, NOT EQUAL ERROR AT EQUAL
    # K, and the first version of this gate got that wrong. It demanded
    # err(n=1) < 3 err(n=2) at the same K_max -- i.e. equal TRUNCATION error
    # from two integrals with deliberately different UV decay. sinc^2 falls as
    # 1/k^6 and sinc^1 as 1/k^3; that difference is the entire subject of this
    # gate, so requiring equal tails was a category error and would have
    # rejected a correct result. What must hold is that EACH arm converges to
    # ITS OWN reference as K grows, and that the residual is small.
    conv1, conv2 = err1_half / max(err1_full, 1e-300), err2_half / max(err2_full, 1e-300)
    print(f"\n  error reduction per doubling of K:  n=1 {conv1:.1f}x   n=2 {conv2:.1f}x")
    verdict = conv1 > 2.0 and conv2 > 2.0 and err1_full < 0.02
    print("\n" + "=" * 78)
    if verdict:
        print("PASS -- the sinc^1 spectral average reproduces the real-space")
        print("Gauss average WITH the strain kernel Gamma(k_hat) in place.  The")
        print("scope limit left open by gate_sinc1_uv_convergence.py is CLOSED:")
        print("the direction-dependent weight does not re-expose the axis")
        print("ridges.  So the spectral route is sound for the collocation")
        print("scheme, and the build is one exponent in machinery that exists.")
        print()
        print("⚠ n=1 carries a LARGER truncation error than n=2 at the same K,")
        print("and that is EXPECTED, not a defect: sinc^1 decays as 1/k^3 where")
        print("sinc^2 decays as 1/k^6, so the same K leaves a bigger tail.  It")
        print("is the practical cost of the single average -- a spectral")
        print("implementation needs a larger cutoff, or an analytic tail, to")
        print("reach the accuracy the double average gets for free.")
    else:
        print("FAIL -- the sinc^1 spectral average does not reproduce the")
        print("real-space reference at the accuracy the control achieves.  The")
        print("likely cause is the axis ridges the O(1) angular weight leaves")
        print("unsuppressed, which is exactly the risk this gate was written")
        print("for.  The single average then needs its static part handled")
        print("analytically, as the document does for the dynamic part.")
    print()
    print("⚠ SCOPE: static kernel, ONE non-contact offset (2d), three")
    print("components.  The DYNAMIC part (poles at |k| = omega/alpha,")
    print("omega/beta, and the subtracted formulation they force) is NOT")
    print("tested here, nor is the CONTACT shell, where the double average")
    print("is singular and the analytic O_h tables are required.")
    print("=" * 78)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
