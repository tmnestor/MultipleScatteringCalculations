"""Derive: Gauss-Legendre nodes and weights correct to 30 digits, n = 1 .. 100, stored as a table.

Why: ``numpy.polynomial.legendre.leggauss`` and ``scipy.special.roots_legendre`` return weights whose relative
errors grow with n -- measured against 30-digit mpmath: 4e-15 at n = 32, up to 7e-13 at n = 40 (in the small
weights near the ends), 6e-14 at n = 48.  Any Gauss rule built on them carries those errors into the integral.
This script computes the rules in mpmath (Newton's method on P_n at 40 digits, from the numpy nodes as starting
values; weights 2 / ((1 - x^2) P_n'(x)^2)) and checks each: the weights sum to 2, the rule integrates
x^(2k) exactly for 2k <= 2n - 1, and nodes and weights are symmetric.

Output: scripts/data/gauss_legendre.json, {"n": {"x": [...], "w": [...]}} with the nodes x >= 0 (and their
weights) as 30-digit strings; the rule is completed by symmetry (x -> -x).  Read by
``cubic_scattering.gauss_rules.gauss_legendre``.

Run:  PYTHONPATH=. python scripts/derive_gauss_legendre.py
"""

import json
from pathlib import Path

import mpmath as mp
from numpy.polynomial.legendre import leggauss

N_MAX = 100


def rule(n: int) -> tuple[list, list]:
    """All n nodes and weights at the working precision, ascending."""
    x0, _ = leggauss(n)
    xs, ws = [], []
    for t in x0:
        x = mp.mpf(float(t))
        for _ in range(8):
            p0, p1 = mp.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            x = x - p1 / dp
        p0, p1 = mp.mpf(1), x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


def main() -> None:
    mp.mp.dps = 40
    out = {}
    worst = mp.mpf(0)
    for n in range(1, N_MAX + 1):
        xs, ws = rule(n)
        # checks: sum of weights, even moments up to degree 2n - 1, symmetry
        err = abs(sum(ws) - 2)
        for k in range(1, n):
            err = max(
                err,
                abs(
                    sum(w * x ** (2 * k) for x, w in zip(xs, ws, strict=True))
                    - mp.mpf(2) / (2 * k + 1)
                ),
            )
        for i in range(n):
            err = max(err, abs(xs[i] + xs[n - 1 - i]), abs(ws[i] - ws[n - 1 - i]))
        worst = max(worst, err)
        half = [
            (x, w)
            for x, w in zip(xs, ws, strict=True)
            if x >= 0 or abs(x) < mp.mpf(10) ** -35
        ]
        out[str(n)] = {
            "x": [mp.nstr(abs(x), 30) for x, _ in half],
            "w": [mp.nstr(w, 30) for _, w in half],
        }
    print(f"n = 1 .. {N_MAX}: worst error of the checks {mp.nstr(worst, 3)}")
    path = Path(__file__).resolve().parent / "data" / "gauss_legendre.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print("written", path.name, f"({path.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
