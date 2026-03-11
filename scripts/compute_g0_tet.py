"""Compute g0_tet to high precision and save the value.

g0_tet = integral of 1/|x| over regular tetrahedron with vertices
(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1).
"""

import mpmath


def compute_g0_tet(dps: int = 50, maxdeg: int = 9) -> mpmath.mpf:
    """Compute g0_tet via face decomposition + Duffy transform."""
    mpmath.mp.dps = dps + 20

    A = [mpmath.mpf(1), mpmath.mpf(-1), mpmath.mpf(-1)]
    B = [mpmath.mpf(-1), mpmath.mpf(1), mpmath.mpf(-1)]
    C = [mpmath.mpf(-1), mpmath.mpf(-1), mpmath.mpf(1)]

    def integrand(s, v):
        u = v * (1 - s)
        w = (1 - s) * (1 - v)
        x0 = s * A[0] + u * B[0] + w * C[0]
        x1 = s * A[1] + u * B[1] + w * C[1]
        x2 = s * A[2] + u * B[2] + w * C[2]
        r = mpmath.sqrt(x0**2 + x1**2 + x2**2)
        return (1 - s) / r

    integral = mpmath.quad(
        integrand,
        [0, 1],
        [0, 1],
        method="gauss-legendre",
        maxdegree=maxdeg,
    )

    mpmath.mp.dps = dps
    return 8 * integral


def main():
    print("Computing g0_tet with convergence study")
    print("=" * 60)

    # Quick convergence at 20 dps
    print("\n20 dps convergence:")
    prev = None
    for md in range(4, 11):
        val = compute_g0_tet(dps=20, maxdeg=md)
        mpmath.mp.dps = 20
        s = mpmath.nstr(val, 18)
        if prev is not None:
            mpmath.mp.dps = 20
            d = mpmath.nstr(abs(val - prev), 5)
            print(f"  md={md:2d}: {s}  delta={d}")
        else:
            print(f"  md={md:2d}: {s}")
        prev = val

    # High precision at maxdeg=10
    print("\n50 dps, maxdeg=10:")
    g0 = compute_g0_tet(dps=50, maxdeg=10)
    mpmath.mp.dps = 50
    print(f"  g0_tet = {mpmath.nstr(g0, 48)}")
    print("  Expected: 4.389580813389239...")

    # Even higher
    print("\n60 dps, maxdeg=11:")
    g0b = compute_g0_tet(dps=60, maxdeg=11)
    mpmath.mp.dps = 60
    print(f"  g0_tet = {mpmath.nstr(g0b, 55)}")

    mpmath.mp.dps = 50
    print(f"\n  Difference md10-md11: {mpmath.nstr(abs(g0 - g0b), 10)}")

    # Save to file
    mpmath.mp.dps = 60
    val_str = mpmath.nstr(g0b, 55)
    from pathlib import Path

    out = Path(
        "/Users/tod/Desktop/MultipleScatteringCalculations/scripts/g0_tet_value.txt"
    )
    out.write_text(val_str + "\n")
    print(f"\n  Saved to g0_tet_value.txt: {val_str}")


if __name__ == "__main__":
    main()
