"""GATE: does the SINGLE-average (sinc^1) spectral form factor converge in the UV?

THE PRIOR ART, and it is close. `docs/inter_voxel_propagator.tex` already builds
the volume-averaged propagator spectrally,

    P_ijkl(R,w) = Int d^3k/(2pi)^3 |f0(k)|^2 Gamma(k,w) e^{ik.R},
    f0(k) = a^3 sinc(k1 h) sinc(k2 h) sinc(k3 h) / (2 sqrt(pi)),   h = a/2

and `Mathematica/InterVoxelPropagatorWS3.wl` is the NIntegrate reference that
validated the committed FACE/EDGE/CORNER constants from exactly this integral,
carrying `Sinc[k1/2]^2 Sinc[k2/2]^2 Sinc[k3/2]^2`.

⚠ THAT IS |f0|^2 = sinc^SQUARED = the DOUBLE (Galerkin) average -- source cell
convolved with field cell. It is `inter_voxel_propagator_9x9`, i.e.
`contact_average='double'`. A COLLOCATION scheme needs one power less: the
source cell averaged against a POINT receiver, i.e. f0 itself, sinc^1.

WHY THIS GATE EXISTS. The document's UV argument is specific to the square:

    "The sinc^2 form factor provides decay |f0|^2 ~ 1/k^2 per axis, giving 1/k^6
     total.  But the 3D volume element k^2 dk and the angular measure contribute
     k^2, leaving the radial integral as Int dk/k^2, which -- combined with the
     O(1) kernel -- makes the 3D integral convergent."

With sinc^1 the decay is 1/k^3 rather than 1/k^6, and the same counting gives
Int dk/k -- LOGARITHMICALLY DIVERGENT. Yet <G> is manifestly finite in real
space for any offset outside the cell, so either the naive power counting is too
pessimistic (the angular measure suppresses the slow axis directions, which are
a set of measure zero) or the integral converges only conditionally and needs
the subtraction the document already applies.

Switching the exponent without settling this is exactly the kind of step that
looks like a one-character change and silently produces a divergent object.

WHAT IS MEASURED, and the first instrument was the WRONG one. The object is the
angular average of the form factor over the sphere |k| = k,

    S_n(k) = (1/4pi) Int dOmega  [prod_i sinc(k_i h)]^n,    n = 1, 2

since the static strain kernel Gamma(k_hat) is O(1) and direction-only, so the
UV behaviour turns on S_n alone. The obvious move is to fit S_n ~ k^-p and ask
for p > 3. That FAILS, and not for a numerical reason: sinc(k h) VANISHES at
k h = n pi, so S_n oscillates through zeros, and sampling it at fixed k
straddles them. The first version of this gate got p = -0.705 on one octave pair
-- a negative "decay rate" -- and still reported a plausible median of 3.189 over
the set. Two independent angular quadratures (200k-point Fibonacci, and
Gauss-Legendre x trapezoid) agree on the VALUES to five digits, so the values
were never the problem; the fit was.

THE RIGHT INSTRUMENT IS THE PARTIAL RADIAL INTEGRAL, AGAINST AN EXACT TARGET.
f0 is the Fourier transform of the NORMALISED cell indicator, so

    Int d^3k/(2pi)^3 f0(k) e^{ik.R} = indicator(R) / V,

and evaluating at R = 0 with V = d^3 = 1 gives, for both n = 1 and n = 2 (the
n = 2 case being the indicator autocorrelation at zero lag, also 1/V),

    Int_0^inf k^2 S_n(k) dk = (2 pi)^3 / (4 pi) = 2 pi^2.

A finite exact value IS the convergence proof. The numerics then only confirm
the approach to it, and the n=2 arm is the CONTROL: the document asserts that
integral converges, so if the measurement does not approach 2 pi^2 there, the
measurement is wrong and no n=1 verdict is given.

⚠ THE ACCEPTANCE CRITERION IS THE ENVELOPE, NOT A TOLERANCE AT THE LARGEST K,
and the control is what establishes that. I_n(K) approaches 2 pi^2 from
alternating sides with a 1/K envelope, so a fixed bar at one K measures where
the oscillation happened to be. The control's errors halve on each doubling of K
(measured ratios 2.14, 1.95, 1.99 -- an exact 1/K envelope), and that ratio is
CHECKED rather than assumed before any n=1 verdict is read.

Run:  conda run -n seismic python scripts/gate_sinc1_uv_convergence.py
"""

from __future__ import annotations

import numpy as np

#: Angular samples per oscillation of the form factor. The integrand oscillates
#: on the angular scale 1/(k h), so the quadrature must resolve THAT, not the
#: sphere. A first version of this gate used 200k Fibonacci points for every k,
#: giving ~0.0079 rad spacing against a 0.0078 rad oscillation at k = 256 -- and
#: produced a decay exponent of -0.705 on one octave pair, i.e. pure noise, while
#: the median over pairs still read a plausible 3.189. The resolution is now tied
#: to k and the pair spread is a hard gate.
#
#: Set to 4 after measuring, not guessed: a 200k-point Fibonacci rule and this
#: Gauss-Legendre rule at 20 samples/oscillation agree on S_n to FIVE digits, so
#: the angular integral is far easier than the oscillation scale suggests, and
#: 20 made the radial sweep below cost billions of point evaluations for no
#: accuracy. `agreement` in main() re-checks 4 against 16 rather than trusting
#: this comment.
SAMPLES_PER_OSCILLATION = 4


def s_n(k: float, n: int, h: float, samples: int = SAMPLES_PER_OSCILLATION) -> float:
    """(1/4pi) Int dOmega [prod_i sinc(k_i h)]^n, resolved at this k.

    Gauss-Legendre in cos(theta) and the trapezoid rule in phi: phi is periodic,
    so the trapezoid rule is spectrally accurate there, and both are tied to the
    oscillation scale 1/(k h).

    `samples` is a parameter rather than a module global so the resolution gate
    in main() can raise it for one comparison without mutating shared state.

    np.sinc(x) = sin(pi x)/(pi x), so the mathematical sinc(y) is np.sinc(y/pi).
    """
    n_ang = max(60, int(samples * k * h))
    mu, w_mu = np.polynomial.legendre.leggauss(n_ang)
    phi = 2.0 * np.pi * np.arange(n_ang) / n_ang
    w_phi = 2.0 * np.pi / n_ang

    st = np.sqrt(1.0 - mu**2)
    kx = k * np.outer(st, np.cos(phi))
    ky = k * np.outer(st, np.sin(phi))
    kz = k * np.repeat(mu[:, None], n_ang, axis=1)

    ff = np.sinc(kx * h / np.pi) * np.sinc(ky * h / np.pi) * np.sinc(kz * h / np.pi)
    integ = (ff**n) * w_mu[:, None] * w_phi
    return float(integ.sum() / (4.0 * np.pi))


def resolution(k: float, h: float) -> float:
    """Angular samples per oscillation actually used -- the honesty check."""
    n_ang = max(60, int(SAMPLES_PER_OSCILLATION * k * h))
    return n_ang / max(k * h, 1.0)


def main() -> int:
    h = 0.5  # half-width for a unit cell, d = 1

    print("=" * 78)
    print("GATE: UV convergence of the sinc^1 (single-average) form factor")
    print("=" * 78)
    print(f"\n  h = {h}; angular quadrature resolved to {SAMPLES_PER_OSCILLATION}")
    print("  samples per oscillation at each k (Gauss-Legendre x trapezoid)")
    print("  convergent  <=>  k^2 S_n(k) integrable  <=>  decay exponent p > 3")

    # ---- why a power-law fit is the WRONG instrument here -------------------
    print("\n  [why this is not a power-law fit]")
    ks = np.array([8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
    print(f"    {'k':>8} {'res':>7} {'S_1(k)':>15} {'S_2(k)':>15}")
    for k in ks:
        print(f"    {k:>8.0f} {resolution(k, h):7.1f} {s_n(k, 1, h):15.4e} {s_n(k, 2, h):15.4e}")
    print("\n    S_n(k) OSCILLATES and passes through zeros -- sinc(k h) vanishes")
    print("    at k h = n pi.  Sampling it at fixed k and fitting k^-p straddles")
    print("    those zeros and returns nonsense (an earlier version of this gate")
    print("    got p = -0.705 on one octave pair and a plausible-looking median")
    print("    of 3.189 over the set).  Two independent angular quadratures")
    print("    agree on these VALUES to five digits, so the values were never the")
    print("    problem -- the fit was.")

    # ---- the right instrument: the partial radial integral, against an EXACT
    # target. f0 is the transform of the normalised cell indicator, so
    #   Int d^3k/(2pi)^3 f0(k) e^{ik.R} = indicator(R)/V,
    # and at R = 0 with V = d^3 = 1 that gives, for BOTH n = 1 and n = 2
    # (n=2 is the indicator autocorrelation at zero lag, also 1/V),
    #   Int_0^inf k^2 S_n(k) dk = (2 pi)^3 / (4 pi) = 2 pi^2.
    # A finite exact value IS the convergence proof; the numerics only confirm
    # the approach to it, and any disagreement indicts the numerics.
    target = 2.0 * np.pi**2
    print(f"\n  [the right instrument] Int_0^K k^2 S_n(k) dk  ->  2 pi^2 = {target:.6f}")
    print("    exact, because f0 is the transform of the normalised cell")
    print("    indicator and the inverse transform at R = 0 is 1/V.")

    # Angular-resolution gate for the cheap rule actually used below.
    probe = 97.0  # off any sinc zero
    coarse = s_n(probe, 1, h, SAMPLES_PER_OSCILLATION)
    fine = s_n(probe, 1, h, 16)
    ang_err = abs(coarse - fine) / max(abs(fine), 1e-300)
    print(f"\n    angular gate at k={probe:.0f}: |S_1(4/osc) - S_1(16/osc)|/|S_1| = {ang_err:.2e}")
    if ang_err > 1e-6:
        print("    ABORT: the cheap angular rule is not converged.")
        return 1

    kmax, dk = 120.0, 0.1
    kgrid = np.arange(dk, kmax + dk, dk)
    vals1 = np.array([s_n(k, 1, h) for k in kgrid])
    vals2 = np.array([s_n(k, 2, h) for k in kgrid])
    integ1 = np.cumsum(kgrid**2 * vals1) * dk
    integ2 = np.cumsum(kgrid**2 * vals2) * dk

    print(f"\n    {'K':>8} {'I_1(K)':>13} {'err_1':>11} {'I_2(K)':>13} {'err_2':>11}")
    errs1, errs2 = [], []
    for K in (15.0, 30.0, 60.0, 120.0):
        i = int(round(K / dk)) - 1
        e1, e2 = abs(integ1[i] - target) / target, abs(integ2[i] - target) / target
        errs1.append(e1)
        errs2.append(e2)
        print(f"    {K:>8.0f} {integ1[i]:13.6f} {e1:11.2e} {integ2[i]:13.6f} {e2:11.2e}")

    # THE CRITERION IS THE ENVELOPE, NOT AN ABSOLUTE BAR AT THE LARGEST K, and
    # the control is what establishes which is appropriate. The integrand
    # oscillates, so I_n(K) approaches 2 pi^2 from alternating sides with a 1/K
    # envelope; a fixed tolerance at one K then measures where the oscillation
    # happened to be, not whether it converges. The control's own errors halve
    # on each doubling of K, which IS the 1/K envelope, and that ratio is
    # checked here rather than assumed.
    ratios2 = [errs2[i] / errs2[i + 1] for i in range(len(errs2) - 1)]
    print("\n  CONTROL (n=2, the validated double average)")
    print(f"    error ratio per doubling of K: {', '.join(f'{r:.2f}' for r in ratios2)}")
    print("    (2.0 would be an exact 1/K envelope -- the signature of a")
    print("     convergent oscillatory tail, not of a divergence)")
    control_ok = all(r > 1.5 for r in ratios2) and errs2[-1] < 0.05
    print(f"    control converging to its exact target: {control_ok}")
    if not control_ok:
        print("    FAIL -- the control does not approach its own exact target, so")
        print("    this quadrature cannot be trusted and no n=1 verdict is given.")
        return 1

    # n=1 oscillates in PHASE differently from n=2, so its per-K error is not
    # monotone. What must hold is that it stays bounded by the same envelope.
    env1 = max(errs1[-2:])
    verdict = env1 < 0.05 and env1 < errs1[0]
    print("\n  VERDICT (n=1, the collocation-consistent single average)")
    print(f"    error envelope over the last two K: {env1:.2e}  (vs {errs1[0]:.2e} at K=15)")
    print(f"    bounded and decreasing: {verdict}")
    print("=" * 78)
    if verdict:
        print("PASS -- the sinc^1 radial integral CONVERGES, to the exact value")
        print("2 pi^2.  The document's per-axis power counting (1/k^3, hence")
        print("Int dk/k) is too pessimistic: the slowly-decaying directions are")
        print("the coordinate AXES, a set of measure zero, and the angular")
        print("measure suppresses them.  So the existing spectral machinery can")
        print("carry the exponent 1 instead of 2 without becoming divergent.")
    else:
        print("FAIL -- the sinc^1 radial integral does not reach its exact")
        print("target, so the single average needs the static part handled")
        print("analytically before the spectral route is usable.")
    print()
    print("⚠ SCOPE, AND IT IS THE LIMIT THAT MATTERS.  This weights the angular")
    print("integral with 1, not with the strain kernel Gamma(k_hat).  That is")
    print("the same simplification the document's own UV argument makes (Gamma")
    print("is O(1) and direction-only), and it is what makes the exact 2 pi^2")
    print("target available -- but a direction-dependent weight could in")
    print("principle re-expose the axis ridges.  Verifying the full 9x9 integral")
    print("against the real-space Gauss reference is the next step, NOT")
    print("something this gate has done.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
