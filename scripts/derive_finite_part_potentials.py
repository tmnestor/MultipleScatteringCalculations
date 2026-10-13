"""The potentials Phi_j of the interface kernels as Hadamard finite parts, in closed form, for every j.

Twin: Mathematica/FinitePartPotentials.wl.  Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md
(12 October: K1 in space).  Paper: LatexPDFs/InterfaceCouplingIntegrals, Eq. (imagetransform).

THE DEFINITION.  The lateral transform of q^m e^{-q zeta} (zeta > 0) is, for m > -2 (Laplace-Hankel),

    (1/4 pi^2) int e^{i k.rho} q^m e^{-q zeta} d^2k = (1/2 pi) int_0^oo q^(m+1) e^{-q zeta} J0(q rho) dq
                                                    = Gamma(m + 2) R^(-m-2) P_(m+1)(zeta / R) / (2 pi).

For m = -1 - j, j >= 1, Gamma has a pole: Phi_j is Hadamard's finite part, the constant term of the Laurent
expansion in m, which is the analytic continuation in the power of q (the series' convention, as
``image_moments.box`` continues in the degree).  With n = j - 1, P_(-j+eps) = P_(n-eps) and

    d/dnu P_nu(x) at nu = n = P_n(x) [log((1 + x)/2) + 2 (psi(2n+1) - psi(n+1))]
                              + 2 sum_{k<n} (-1)^(n+k) (2k+1) / ((n-k)(n+k+1)) P_k(x),

    Phi_j = (-1)^n / (n! 2 pi) { [psi(j) - 2 (psi(2n+1) - psi(n+1)) - log((R + zeta)/2)] R^n P_n(zeta/R)
                                 - 2 sum_{k<n} (-1)^(n+k) (2k+1) / ((n-k)(n+k+1)) R^n P_k(zeta/R) }.

R^n P_k(zeta/R) is a polynomial in (R, zeta).  The pole's residue is the solid harmonic R^n P_n(zeta/R): the
finite part fixes it, with the constants psi(j) (Euler's gamma) and log 2, where a cutoff would leave it free.
Phi_0 = 1/(2 pi R) (no pole).  The paper's Phi1, Phi2 (K0) differ from these by multiples of 1 and of zeta,
which every K0 term annihilates by its lateral derivatives (n_lat >= j).

CHECKS (printed):
  1. the ladder d_zeta Phi_(j+1) = -Phi_j, symbolically;
  2. harmonicity, d_rho^2 + (1/rho) d_rho + d_zeta^2 = 0, symbolically;
  3. against the finite part computed from the integral itself at 40 digits (the head int_0^1 as the Taylor
     series of the entire e^{-q zeta} J0(q rho), sum over k != n of c_k / (k - n); the tail int_1^oo by quadrature).
     Measured 12 October 2026: 0 .. 8e-41 at zeta = 0.45 and 1.3 (rho = 0.7), j = 1 .. 4;
  4. Phi1, Phi2 against the paper's, the differences (log 2 - gamma)/2 pi and a multiple of zeta.
  5. THE LINE OF IMAGES IS ANOTHER CONVENTION.  int_zeta^oo (t - zeta)^n / n! Phi0(rho, t) dt diverges at
     t -> oo when the lateral derivatives are too few; its finite part there (dropping the powers of T and
     log T) differs from Phi_j by a polynomial H_j = Phi_j - FP_oo(line):
         H1 = -gamma/2pi,  H2 = gamma zeta/2pi,  H3 = ((2 gamma - 3) rho^2 - 4 gamma zeta^2)/16pi,
         H4 = -zeta ((6 gamma - 9) rho^2 - 4 gamma zeta^2)/48pi;
     for j >= 3 not harmonic (the dropped divergent terms are not harmonic on their own), so the line's finite
     part at infinity is not a potential.  Phi_j (the continuation in q) is the convention of the wavenumber
     remainder; moments built on the line of images must add the moments of d^alpha H_j.
Writes scripts/data/finite_part_potentials.json: Phi_j, j = 0 .. 6, in R, zeta, as sympy and as
Mathematica strings (the latter read by the twin).  Parse the sympy strings with
locals={'R': R, 'zeta': zeta}: a bare 'zeta' is sympy's Riemann zeta function.

Run:  PYTHONPATH=. python scripts/derive_finite_part_potentials.py
"""

import json
from pathlib import Path

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "data" / "finite_part_potentials.json"
J_MAX = 6

R, zeta, rho = sp.symbols("R zeta rho", positive=True)


def phi(j: int) -> sp.Expr:
    """Phi_j^FP as an expression in R and zeta (R = sqrt(rho^2 + zeta^2))."""
    if j == 0:
        return 1 / (2 * sp.pi * R)
    n = j - 1
    x = zeta / R
    const = sp.digamma(j) - 2 * (sp.digamma(2 * n + 1) - sp.digamma(n + 1))
    main = (const - sp.log((R + zeta) / 2)) * R**n * sp.legendre(n, x)
    rest = 2 * sum(
        sp.Integer(-1) ** (n + k)
        * sp.Rational(2 * k + 1, (n - k) * (n + k + 1))
        * R**n
        * sp.legendre(k, x)
        for k in range(n)
    )
    return sp.expand(
        sp.Integer(-1) ** n / (sp.factorial(n) * 2 * sp.pi) * (main - rest)
    )


def in_rho(e: sp.Expr) -> sp.Expr:
    return e.subs(R, sp.sqrt(rho**2 + zeta**2))


def fp_from_integral(j: int, rv: mp.mpf, zv: mp.mpf) -> mp.mpf:
    """(1/2 pi) FP int_0^oo q^-j e^{-q zeta} J0(q rho) dq from the integral, 40 digits."""
    f = lambda q: mp.exp(-q * zv) * mp.besselj(0, q * rv)
    c = mp.taylor(f, 0, 120)
    head = sum(c[k] / (k - j + 1) for k in range(121) if k != j - 1)
    tail = mp.quad(lambda q: f(q) / q**j, [1, 10, mp.inf])
    return (head + tail) / (2 * mp.pi)


def line_difference(j: int) -> sp.Expr:
    """H_j = Phi_j - (finite part at infinity of the line of images), in rho and zeta."""
    t, u = sp.symbols("t u", positive=True)
    n = j - 1
    line = sp.integrate(
        (t - zeta) ** n / sp.factorial(n) / sp.sqrt(rho**2 + t**2), (t, zeta, 1 / u)
    )
    ser = sp.series(sp.expand_log(line.rewrite(sp.log), force=True), u, 0, 1).removeO()
    finite = sum(
        a
        for a in sp.Add.make_args(sp.expand(sp.expand_log(ser, force=True)))
        if not a.has(u)
    )
    return sp.factor(
        sp.simplify(sp.expand_log(in_rho(phi(j)) - finite / (2 * sp.pi), force=True))
    )


def main() -> None:
    phis = [phi(j) for j in range(J_MAX + 1)]
    for j in range(J_MAX):
        lad = sp.simplify(sp.diff(in_rho(phis[j + 1]), zeta) + in_rho(phis[j]))
        e = in_rho(phis[j + 1])
        lap = sp.simplify(
            sp.diff(e, rho, 2) + sp.diff(e, rho) / rho + sp.diff(e, zeta, 2)
        )
        print(
            f"Phi{j + 1}: ladder d_zeta Phi{j + 1} + Phi{j} = {lad}, Laplacian = {lap}"
        )
    mp.mp.dps = 40
    for rv, zv in ((mp.mpf("0.7"), mp.mpf("0.45")), (mp.mpf("0.7"), mp.mpf("1.3"))):
        errs = []
        for j in range(1, 5):
            val = sp.N(
                phis[j].subs(
                    {
                        R: sp.sqrt(
                            sp.Float(str(rv), 45) ** 2 + sp.Float(str(zv), 45) ** 2
                        ),
                        zeta: sp.Float(str(zv), 45),
                    }
                ),
                40,
            )
            ref = fp_from_integral(j, rv, zv)
            errs.append(abs(mp.mpf(str(val)) - ref) / abs(ref))
        print(
            f"rho = {rv}, zeta = {zv}: closed form vs the integral's finite part, j = 1..4:",
            [mp.nstr(e, 2) for e in errs],
        )
    paper1 = -sp.log(R + zeta) / (2 * sp.pi)
    paper2 = (zeta * sp.log(R + zeta) - R) / (2 * sp.pi)
    print(
        "Phi1 - paper's Phi1 =",
        sp.simplify(sp.expand_log(phis[1] - paper1, force=True)),
    )
    print(
        "Phi2 - paper's Phi2 =",
        sp.simplify(sp.expand_log(phis[2] - paper2, force=True)),
    )
    for j in range(1, 5):
        h = line_difference(j)
        lap = sp.simplify(
            sp.diff(h, rho, 2) + sp.diff(h, rho) / rho + sp.diff(h, zeta, 2)
        )
        print(f"H{j} = Phi{j} - FP_oo(line of images) = {h}   Laplacian {lap}")
    OUT.write_text(
        json.dumps(
            {
                "variables": ["R", "zeta"],
                "definition": "Hadamard finite part (continuation in the power of q) of the lateral transform "
                "of q^(-1-j) e^(-q zeta)",
                "phi": [str(p) for p in phis],
                "phi_wl": [sp.mathematica_code(p) for p in phis],
            },
            indent=1,
        )
    )
    print(f"written {OUT.relative_to(ROOT)}")
    for j in range(1, 5):
        print(
            f"Phi{j} =",
            sp.collect(sp.expand(phis[j] * 2 * sp.pi), sp.log((R + zeta) / 2)),
            "/ (2 pi)",
        )


if __name__ == "__main__":
    main()
