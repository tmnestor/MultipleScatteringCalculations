#!/usr/bin/env python3
"""The exact Mie scattered TRACTION, against two independent arbiters.

WHY THIS EXISTS
---------------
The R/T comparison on planes bounding the sphere needs the traction
tau_3 = sigma . zhat, because that -- with the displacement -- is the state the
impedance march works in, and defining both on the same state removes a
convention bridge.  This project's defects have lived in exactly such bridges.

The package had no full Mie stress anywhere: the boundary-condition matrices
carry sigma_rr and sigma_rtheta at arbitrary radius, which is what a spherical
boundary needs, but a plane of constant z also needs sigma_thetatheta and
sigma_phiphi.  ``Mathematica/MieSphericalWaves.wl`` derives the stress in
CARTESIAN -- avoiding curvilinear basis factors and any rotation -- and checks
it against Navier, ``div sigma + rho omega^2 u = 0``, which comes out exactly
zero once the inputs are exact rationals.
``sphere_scattering.mie_scattered_traction`` implements it in SPHERICAL
components and rotates.  Two different routes; this gate scores one against the
other.

THE TWO ARBITERS, AND WHY BOTH
------------------------------
  [1] The Mathematica reference.  Exact, symbolic, Navier-checked, and derived
      by a route sharing no algebra with the Python.  It is evaluated per
      potential family with unit coefficients, so it tests the field
      construction rather than the Mie coefficients.
  [2] Finite differences of the EXACT displacement, Richardson-extrapolated.
      The displacement is analytic away from the sphere, so this converges
      cleanly, and it exercises the ASSEMBLED Mie field -- coefficients and
      all -- which arbiter [1] does not.

Neither alone would do: [1] never sees the assembled sum, and [2] shares the
displacement evaluator with the thing under test, so a fault in that evaluator
would cancel.  Together they close both gaps.

Run:  conda run -n seismic python scripts/gate_mie_traction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.sphere_scattering import (  # noqa: E402
    MieResult,
    compute_elastic_mie,
    mie_scattered_displacement,
    mie_scattered_traction,
)

REFERENCE = ROOT / "Mathematica" / "MieSphericalWaves_reference.json"

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def rel(a: NDArray, b: NDArray) -> float:
    """Relative difference in the max norm.

    Args:
        a: First.
        b: Second.

    Returns:
        max|a-b| / max|b|.
    """
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def to_py(vec_xyz: list) -> NDArray:
    """Map a Mathematica (x, y, z) triple to the package's (z, x, y) ordering.

    The two orderings are the single most likely place for this comparison to
    go wrong silently, so the mapping is written once, here, and used for both
    the points and the field values.

    Args:
        vec_xyz: Length-3 sequence in (x, y, z) order.

    Returns:
        Shape (3,) in (z, x, y) order.
    """
    v = np.asarray(vec_xyz)
    return np.array([v[2], v[0], v[1]])


def to_py_complex(reim: list) -> NDArray:
    """Map Mathematica ``ReIm`` output in (x, y, z) to a complex (z, x, y) vector.

    Args:
        reim: Length-3 sequence of [re, im] pairs, in (x, y, z) order.

    Returns:
        Shape (3,) complex, in (z, x, y) order.
    """
    c = np.array([complex(p[0], p[1]) for p in reim])
    return np.array([c[2], c[0], c[1]])


def single_mode_mie(ref: ReferenceMedium, omega: float, n0: int, family: str, n_max: int) -> MieResult:
    """A MieResult carrying ONE potential family at ONE order, unit amplitude.

    This is how arbiter [1] is applied: it isolates the field construction from
    the Mie coefficient solve, so a disagreement points at one or the other
    rather than at both.

    Args:
        ref: Background medium.
        omega: Angular frequency.
        n0: The single order to switch on.
        family: "P" for the L-type potential, "SV" for the N-type.
        n_max: Array length.

    Returns:
        A MieResult with a single unit coefficient.
    """
    z = np.zeros(n_max + 1, dtype=complex)
    a_n, b_n = z.copy(), z.copy()
    if family == "P":
        a_n[n0] = 1.0
    else:
        b_n[n0] = 1.0
    return MieResult(
        a_n=a_n,
        b_n=b_n,
        c_n=z.copy(),
        a_n_sv=z.copy(),
        b_n_sv=z.copy(),
        n_max=n_max,
        omega=omega,
        radius=1.0,
        ref=ref,
        contrast=MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=0.0),
        ka_P=0.0,
        ka_S=0.0,
    )


def part1() -> None:
    """Against the Cartesian Mathematica derivation, family by family."""
    print("\n[1] against the Navier-checked Mathematica reference")
    if not REFERENCE.exists():
        report(f"reference dump present at {REFERENCE}", False)
        return
    data = json.loads(REFERENCE.read_text())
    lam, mu, rho = data["lam"], data["mu"], data["rho"]
    omega = data["omega"]
    alpha = float(np.sqrt((lam + 2.0 * mu) / rho))
    beta = float(np.sqrt(mu / rho))
    ref = ReferenceMedium(alpha=alpha, beta=beta, rho=rho)
    print(f"      lam={lam} mu={mu} rho={rho} omega={omega}")
    print(f"      alpha={alpha:.6f} beta={beta:.6f}  (kP={omega / alpha:.6f}, kS={omega / beta:.6f})")

    worst_u = {"P": 0.0, "SV": 0.0}
    worst_t = {"P": 0.0, "SV": 0.0}
    n_seen = 0
    for case in data["cases"]:
        # m = 1 and the SH family are not implemented in the axisymmetric
        # evaluator, which is P-incidence only; they are left for the
        # extension to a full R/T matrix.
        if case["m"] != 0 or case["family"] == "SH":
            continue
        fam, n0 = case["family"], case["n"]
        mie = single_mode_mie(ref, omega, n0, fam, n_max=max(4, n0))
        pt = to_py(case["point"]).reshape(1, 3)
        got_u = mie_scattered_displacement(mie, pt)[0]
        got_t = mie_scattered_traction(mie, pt)[0]
        want_u = to_py_complex(case["u"])
        want_t = to_py_complex(case["tau3"])
        worst_u[fam] = max(worst_u[fam], rel(got_u, want_u))
        worst_t[fam] = max(worst_t[fam], rel(got_t, want_t))
        n_seen += 1

    print(f"      compared {n_seen} cases (m=0, families P and SV)")
    for fam in ("P", "SV"):
        print(f"        {fam:3s}  displacement {worst_u[fam]:.3e}   traction {worst_t[fam]:.3e}")
        report(f"displacement matches the Cartesian derivation, {fam}", worst_u[fam] < 1e-10)
        report(f"TRACTION matches the Cartesian derivation, {fam}", worst_t[fam] < 1e-10)


def fd_traction(mie: MieResult, pt: NDArray, h: float) -> NDArray:
    """tau_3 by central differences of the exact displacement.

    Args:
        mie: The Mie solution.
        pt: Field point, shape (3,), ordered (z, x, y).
        h: Step.

    Returns:
        Shape (3,) complex.
    """
    lam, mu = mie.ref.lam, mie.ref.mu
    grad = np.zeros((3, 3), dtype=complex)  # grad[i, j] = d u_i / d x_j
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        up = mie_scattered_displacement(mie, (pt + e).reshape(1, 3))[0]
        um = mie_scattered_displacement(mie, (pt - e).reshape(1, 3))[0]
        grad[:, j] = (up - um) / (2.0 * h)
    div = np.trace(grad)
    sig = lam * div * np.eye(3) + mu * (grad + grad.T)
    return np.asarray(sig[:, 0])  # z is axis 0, so sigma . zhat is the z column


def part2() -> None:
    """Against finite differences of the exact displacement, on the ASSEMBLED field."""
    print("\n[2] against Richardson-extrapolated finite differences")
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    radius, ka = 10.0, 0.5
    omega = ka * ref.beta / radius
    mie = compute_elastic_mie(omega, radius, ref, contrast)

    pts = [
        np.array([14.0, 3.0, -2.0]),
        np.array([-16.0, 5.0, 4.0]),
        np.array([12.0, 0.02, 0.01]),  # essentially on the axis
        np.array([0.5, 13.0, 7.0]),  # near the equator
    ]
    print(f"      {'point':>26}  {'FD h':>8}  {'rel diff':>10}")
    worst = 0.0
    for pt in pts:
        exact = mie_scattered_traction(mie, pt.reshape(1, 3))[0]
        # Richardson: central differences are O(h^2), so (4 f(h/2) - f(h))/3
        # is O(h^4) and removes the leading truncation.
        f1 = fd_traction(mie, pt, 0.20)
        f2 = fd_traction(mie, pt, 0.10)
        rich = (4.0 * f2 - f1) / 3.0
        d = rel(rich, exact)
        worst = max(worst, d)
        print(f"      {str(pt):>26}  {'rich':>8}  {d:10.3e}")
    report("the traction matches finite differences of the exact displacement", worst < 1e-6)

    # And the FD must CONVERGE toward it, which distinguishes agreement from
    # coincidence at one step.
    pt = pts[0]
    exact = mie_scattered_traction(mie, pt.reshape(1, 3))[0]
    errs = [rel(fd_traction(mie, pt, h), exact) for h in (0.4, 0.2, 0.1)]
    orders = [float(np.log(errs[i] / errs[i + 1]) / np.log(2.0)) for i in range(len(errs) - 1)]
    print(f"      FD errors at h = 0.4, 0.2, 0.1: {'  '.join(f'{e:.2e}' for e in errs)}")
    print(f"      observed order: {'  '.join(f'{o:.2f}' for o in orders)}")
    report("finite differences converge to it at second order", 1.7 < orders[-1] < 2.3)


def part3() -> None:
    """Regularity on the axis, where the bounding plane is centred."""
    print("\n[3] regularity on the symmetry axis")
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    radius, ka = 10.0, 0.5
    omega = ka * ref.beta / radius
    mie = compute_elastic_mie(omega, radius, ref, contrast)

    # Approach the axis.  cot(theta) u_theta is the term that would blow up if
    # it were evaluated by dividing rather than through P_n^1/sin(theta).
    vals = []
    for eps in (1.0, 1e-2, 1e-4, 1e-6, 0.0):
        t = mie_scattered_traction(mie, np.array([[14.0, eps, 0.0]]))[0]
        vals.append(t)
        print(f"      x = {eps:<8g}  tau_3 = {np.abs(t)}")
    finite = all(bool(np.all(np.isfinite(v))) for v in vals)
    report("the traction is finite as the axis is approached", finite)
    report("and settles to a limit there", rel(vals[-2], vals[-1]) < 1e-6)

    # On the axis the transverse traction must vanish by symmetry: an
    # axisymmetric P field has no preferred azimuth.
    on_axis = vals[-1]
    print(f"      on-axis transverse components: {abs(on_axis[1]):.3e}, {abs(on_axis[2]):.3e}")
    report(
        "the transverse traction vanishes on the axis, as symmetry requires",
        max(abs(on_axis[1]), abs(on_axis[2])) < 1e-10 * max(abs(on_axis[0]), 1e-300),
    )


def main() -> int:
    """Run every part and summarise.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    print("=" * 78)
    print("  The exact Mie scattered traction tau_3 = sigma . zhat")
    print("=" * 78)
    for part in (part1, part2, part3):
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
