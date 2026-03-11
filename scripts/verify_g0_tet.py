"""Verify the closed-form expression for g0_tet.

RESULT: g0_tet = 4*sqrt(2)*asinh(sqrt(2)) - (2/3)*pi
              = 4*sqrt(2)*log(sqrt(2) + sqrt(3)) - (2/3)*pi

This is the integral of 1/|x| over the regular tetrahedron with vertices
(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1). Volume = 8/3.

Found via PSLQ: 3*g0 + 2*pi - 12*sqrt(2)*asinh(sqrt(2)) = 0
"""

import mpmath


def compute_g0_tet_numerical(dps: int = 50, maxdeg: int = 10) -> mpmath.mpf:
    """Compute g0_tet numerically via face decomposition."""
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
    mpmath.mp.dps = 60

    print("=" * 70)
    print("VERIFICATION: g0_tet = 4*sqrt(2)*asinh(sqrt(2)) - (2/3)*pi")
    print("            = 4*sqrt(2)*log(sqrt(2) + sqrt(3)) - (2/3)*pi")
    print("=" * 70)

    sqrt2 = mpmath.sqrt(2)
    sqrt3 = mpmath.sqrt(3)
    pi = mpmath.pi

    # Closed form
    g0_exact = 4 * sqrt2 * mpmath.asinh(sqrt2) - 2 * pi / 3
    g0_log = 4 * sqrt2 * mpmath.log(sqrt2 + sqrt3) - 2 * pi / 3

    print(f"\n  Closed form (asinh): {mpmath.nstr(g0_exact, 55)}")
    print(f"  Closed form (log):   {mpmath.nstr(g0_log, 55)}")
    print(f"  Difference:          {mpmath.nstr(g0_exact - g0_log, 10)}")

    # Numerical
    g0_num = compute_g0_tet_numerical(dps=60, maxdeg=10)
    mpmath.mp.dps = 60
    print(f"\n  Numerical (md=10):   {mpmath.nstr(g0_num, 55)}")
    print(f"  Exact - Numerical:   {mpmath.nstr(g0_exact - g0_num, 10)}")

    # Even higher precision numerical
    g0_num2 = compute_g0_tet_numerical(dps=60, maxdeg=11)
    mpmath.mp.dps = 60
    print(f"  Numerical (md=11):   {mpmath.nstr(g0_num2, 55)}")
    print(f"  Exact - Numerical:   {mpmath.nstr(g0_exact - g0_num2, 10)}")

    # Component values
    print(f"\n  sqrt(2)              = {mpmath.nstr(sqrt2, 50)}")
    print(f"  sqrt(3)              = {mpmath.nstr(sqrt3, 50)}")
    print(f"  asinh(sqrt(2))       = {mpmath.nstr(mpmath.asinh(sqrt2), 50)}")
    print(f"  log(sqrt(2)+sqrt(3)) = {mpmath.nstr(mpmath.log(sqrt2 + sqrt3), 50)}")
    print(f"  pi                   = {mpmath.nstr(pi, 50)}")

    print(
        f"\n  4*sqrt(2)*asinh(sqrt(2)) = {mpmath.nstr(4 * sqrt2 * mpmath.asinh(sqrt2), 50)}"
    )
    print(f"  (2/3)*pi                 = {mpmath.nstr(2 * pi / 3, 50)}")
    print(f"  Difference               = {mpmath.nstr(g0_exact, 50)}")

    # Alternative forms
    print("\n" + "=" * 70)
    print("Alternative expressions:")
    print("=" * 70)

    # asinh(sqrt2) = log(sqrt2 + sqrt3)
    # = log(sqrt2 + sqrt3) = log(sqrt(2) + sqrt(3))
    # Also: sqrt2 * log(sqrt2 + sqrt3) = sqrt2 * arccosh(sqrt3)? No.
    # arccosh(sqrt3) = log(sqrt3 + sqrt2) = asinh(sqrt2). Yes!
    # So g0 = 4 * sqrt2 * arccosh(sqrt3) - (2/3)*pi

    # (sqrt2+sqrt3)^2 = 2 + 2*sqrt6 + 3 = 5 + 2*sqrt6
    # So log(sqrt2+sqrt3) = (1/2)*log(5+2*sqrt6)
    # g0 = 2*sqrt2*log(5+2*sqrt6) - (2/3)*pi

    g0_v2 = 2 * sqrt2 * mpmath.log(5 + 2 * mpmath.sqrt(6)) - 2 * pi / 3
    print(f"\n  2*sqrt(2)*log(5+2*sqrt(6)) - (2/3)*pi = {mpmath.nstr(g0_v2, 50)}")
    print(f"  Match: {mpmath.nstr(g0_v2 - g0_exact, 10)}")

    # Using Pell: 5+2*sqrt6 = (sqrt2+sqrt3)^2
    # (5+2sqrt6)^2 = 49 + 20*sqrt6
    # (5+2sqrt6)^3 = 485 + 198*sqrt6  <-- the Pell number!
    g0_v3 = (
        mpmath.mpf(2) / 3 * sqrt2 * mpmath.log(485 + 198 * mpmath.sqrt(6)) - 2 * pi / 3
    )
    print(f"  (2/3)*sqrt(2)*log(485+198*sqrt(6)) - (2/3)*pi = {mpmath.nstr(g0_v3, 50)}")
    print(f"  Match: {mpmath.nstr(g0_v3 - g0_exact, 10)}")

    # Comparison with cube
    g0_cube = mpmath.mpf(4) / 3 * mpmath.log(70226 + 40545 * sqrt3) - 2 * pi
    print(
        f"\n  g0_cube = (4/3)*log(70226+40545*sqrt(3)) - 2*pi = {mpmath.nstr(g0_cube, 40)}"
    )
    print(
        f"  g0_tet  = 4*sqrt(2)*log(sqrt(2)+sqrt(3)) - (2/3)*pi = {mpmath.nstr(g0_exact, 40)}"
    )
    print(f"  Ratio g0_tet/g0_cube = {mpmath.nstr(g0_exact / g0_cube, 15)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  g0_tet = integral_{{tet}} 1/|x| dV

  Regular tetrahedron: vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
  Volume = 8/3, inscribed sphere radius = 1/sqrt(3)

  CLOSED FORM:

    g0_tet = 4*sqrt(2)*log(sqrt(2) + sqrt(3)) - (2/3)*pi

           = 4*sqrt(2)*asinh(sqrt(2)) - (2/3)*pi

           = 2*sqrt(2)*log(5 + 2*sqrt(6)) - (2/3)*pi

           = (2/3)*sqrt(2)*log(485 + 198*sqrt(6)) - (2/3)*pi

           = (2/3)*[sqrt(2)*log(485 + 198*sqrt(6)) - pi]

  Numerical value: {mpmath.nstr(g0_exact, 50)}

  Found via PSLQ integer relation: 3*g0 + 2*pi - 12*sqrt(2)*asinh(sqrt(2)) = 0

  Compare with cube:
    g0_cube = (4/3)*log(70226 + 40545*sqrt(3)) - 2*pi
""")


if __name__ == "__main__":
    main()
