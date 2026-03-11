"""Identify the constant g0_tet using high-precision quadrature + PSLQ.

g0_tet = integral of 1/|x| over the regular tetrahedron with vertices
(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1). Volume = 8/3.

RESULT: g0_tet = 4*sqrt(2)*asinh(sqrt(2)) - (2/3)*pi
              = 4*sqrt(2)*log(sqrt(2) + sqrt(3)) - (2/3)*pi

Found via PSLQ: 3*g0 + 2*pi - 12*sqrt(2)*asinh(sqrt(2)) = 0
"""

import mpmath


def compute_g0_tet(dps: int = 50, maxdeg: int = 10) -> mpmath.mpf:
    """Compute g0_tet via face decomposition + Duffy transform.

    Split tet into 4 sub-tets from origin to each face. By Td symmetry
    all 4 give equal contributions. Duffy transform removes the 1/r singularity.

    Parametrize sub-tet from origin to face {v1,v2,v3}:
      x(t,s,v) = t * [s*v1 + v*(1-s)*v2 + (1-s)*(1-v)*v3]
      t in [0,1], s in [0,1], v in [0,1]

    After integrating out t: g0_tet = 8 * int_[0,1]^2 (1-s)/R(s,v) ds dv
    where R = |s*v1 + v*(1-s)*v2 + (1-s)*(1-v)*v3|.
    """
    mpmath.mp.dps = dps + 20

    v1 = [mpmath.mpf(1), mpmath.mpf(-1), mpmath.mpf(-1)]
    v2 = [mpmath.mpf(-1), mpmath.mpf(1), mpmath.mpf(-1)]
    v3 = [mpmath.mpf(-1), mpmath.mpf(-1), mpmath.mpf(1)]

    def integrand(s, v):
        u = v * (1 - s)
        w = (1 - s) * (1 - v)
        x0 = s * v1[0] + u * v2[0] + w * v3[0]
        x1 = s * v1[1] + u * v2[1] + w * v3[1]
        x2 = s * v1[2] + u * v2[2] + w * v3[2]
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
    # ===================================================================
    # Phase 1: Compute g0_tet to high precision
    # ===================================================================
    print("=" * 72)
    print("Phase 1: Compute g0_tet via numerical quadrature")
    print("=" * 72)

    print("\nConvergence study (20 dps):")
    prev = None
    for md in range(4, 11):
        val = compute_g0_tet(dps=20, maxdeg=md)
        mpmath.mp.dps = 20
        s = mpmath.nstr(val, 18)
        if prev is not None:
            d = mpmath.nstr(abs(val - prev), 5)
            print(f"  maxdeg={md:2d}: {s}  delta={d}")
        else:
            print(f"  maxdeg={md:2d}: {s}")
        prev = val

    print("\nHigh-precision computation (50 dps, maxdeg=10):")
    g0 = compute_g0_tet(dps=50, maxdeg=10)
    mpmath.mp.dps = 50
    print(f"  g0_tet = {mpmath.nstr(g0, 48)}")

    # ===================================================================
    # Phase 2: PSLQ search
    # ===================================================================
    print("\n" + "=" * 72)
    print("Phase 2: PSLQ integer relation search")
    print("=" * 72)

    mpmath.mp.dps = 45

    sqrt2 = mpmath.sqrt(2)
    sqrt3 = mpmath.sqrt(3)
    sqrt6 = mpmath.sqrt(6)
    pi = mpmath.pi
    log2 = mpmath.log(2)
    log3 = mpmath.log(3)

    atan_sqrt2 = mpmath.atan(sqrt2)
    log_1p_sqrt2 = mpmath.log(1 + sqrt2)
    asinh_sqrt2 = mpmath.asinh(sqrt2)  # = log(sqrt2 + sqrt3)
    log_2p_sqrt3 = mpmath.log(2 + sqrt3)
    atan_2sqrt2 = mpmath.atan(2 * sqrt2)
    atan_sqrt2_over3 = mpmath.atan(sqrt2 / 3)
    atan_2sqrt2_over3 = mpmath.atan(2 * sqrt2 / 3)
    log_5p2sqrt6 = mpmath.log(5 + 2 * sqrt6)

    # Catalog of candidate basis elements
    catalog = {
        "atan(sqrt2)": atan_sqrt2,
        "log(1+sqrt2)": log_1p_sqrt2,
        "asinh(sqrt2)": asinh_sqrt2,
        "log(2+sqrt3)": log_2p_sqrt3,
        "sqrt2": sqrt2,
        "sqrt3": sqrt3,
        "sqrt6": sqrt6,
        "log2": log2,
        "log3": log3,
        "atan(2sqrt2)": atan_2sqrt2,
        "atan(sqrt2/3)": atan_sqrt2_over3,
        "atan(2sqrt2/3)": atan_2sqrt2_over3,
        "log(5+2sqrt6)": log_5p2sqrt6,
        "sqrt2*atan(sqrt2)": sqrt2 * atan_sqrt2,
        "sqrt3*atan(sqrt2)": sqrt3 * atan_sqrt2,
        "sqrt6*atan(sqrt2)": sqrt6 * atan_sqrt2,
        "sqrt2*log(1+sqrt2)": sqrt2 * log_1p_sqrt2,
        "sqrt3*log(1+sqrt2)": sqrt3 * log_1p_sqrt2,
        "sqrt6*log(1+sqrt2)": sqrt6 * log_1p_sqrt2,
        "sqrt2*asinh(sqrt2)": sqrt2 * asinh_sqrt2,
        "sqrt3*asinh(sqrt2)": sqrt3 * asinh_sqrt2,
        "sqrt6*asinh(sqrt2)": sqrt6 * asinh_sqrt2,
        "sqrt2*log2": sqrt2 * log2,
        "sqrt3*log2": sqrt3 * log2,
        "sqrt2*log3": sqrt2 * log3,
        "sqrt3*log3": sqrt3 * log3,
    }

    def report_relation(rel, names, vals):
        terms = []
        for c, n in zip(rel, names, strict=True):
            if c != 0:
                terms.append(f"{c}*{n}")
        print(f"  *** FOUND: {' + '.join(terms)} = 0")
        check = sum(c * v for c, v in zip(rel, vals, strict=True))
        print(f"      Verification: {mpmath.nstr(check, 15)}")
        if rel[0] != 0:
            g0_expr = (
                -sum(c * v for c, v in zip(rel[1:], vals[1:], strict=True)) / rel[0]
            )
            err = g0_expr - g0
            print(f"      g0 = {mpmath.nstr(g0_expr, 40)}")
            print(f"      Error: {mpmath.nstr(err, 10)}")

    # --- 4-element: {g0, 1, pi, X} ---
    print("\n--- 4-element PSLQ: {g0, 1, pi, X} ---")
    found_4 = False
    for name, val in catalog.items():
        rel = mpmath.pslq([g0, mpmath.mpf(1), pi, val], maxcoeff=10000)
        if rel:
            report_relation(rel, ["g0", "1", "pi", name], [g0, mpmath.mpf(1), pi, val])
            found_4 = True
    if not found_4:
        print("  No 4-element relations found.")

    # --- log(A+B*sqrtN) forms ---
    print("\n--- 4-element: log(A+B*sqrtN) forms ---")
    found_log = False
    for n_val, n_name in [(2, "sqrt2"), (3, "sqrt3"), (6, "sqrt6")]:
        sqrt_n = mpmath.sqrt(n_val)
        for a in range(1, 25):
            for b in range(1, 15):
                arg = a + b * sqrt_n
                if arg <= 0:
                    continue
                val = mpmath.log(arg)
                rel = mpmath.pslq([g0, mpmath.mpf(1), pi, val], maxcoeff=10000)
                if rel:
                    name = f"log({a}+{b}{n_name})"
                    report_relation(
                        rel, ["g0", "1", "pi", name], [g0, mpmath.mpf(1), pi, val]
                    )
                    found_log = True
    if not found_log:
        print("  No log forms found.")

    # --- 5-element: {g0, 1, pi, X, Y} ---
    print("\n--- 5-element PSLQ (selected pairs) ---")
    elem_list = list(catalog.items())
    found_5 = False
    n_tested = 0
    for i in range(len(elem_list)):
        for j in range(i + 1, len(elem_list)):
            n1, v1 = elem_list[i]
            n2, v2 = elem_list[j]
            n_tested += 1
            rel = mpmath.pslq([g0, mpmath.mpf(1), pi, v1, v2], maxcoeff=10000)
            if rel and rel[0] != 0:  # Only report if g0 is involved
                report_relation(
                    rel,
                    ["g0", "1", "pi", n1, n2],
                    [g0, mpmath.mpf(1), pi, v1, v2],
                )
                found_5 = True
    print(f"  Tested {n_tested} pairs.")
    if not found_5:
        print("  No 5-element relations involving g0.")

    # --- 6-element ---
    print("\n--- 6-element PSLQ (key combinations) ---")
    bases_6 = {
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),sqrt2}": [
            g0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            sqrt2,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),sqrt3}": [
            g0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            sqrt3,
        ],
        "{g0,1,pi,atan(sqrt2),asinh(sqrt2),sqrt2}": [
            g0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            asinh_sqrt2,
            sqrt2,
        ],
    }
    for name, basis in bases_6.items():
        rel = mpmath.pslq(basis, maxcoeff=10000)
        if rel:
            names_list = name.strip("{}").split(",")
            report_relation(rel, names_list, basis)
        else:
            print(f"  No relation: {name}")

    # --- 8-element ---
    print("\n--- 8-element PSLQ ---")
    bases_8 = {
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),sqrt2,sqrt3,sqrt6}": [
            g0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            sqrt2,
            sqrt3,
            sqrt6,
        ],
        "{g0,1,pi,sqrt2,sqrt3,sqrt6,atan(sqrt2),asinh(sqrt2)}": [
            g0,
            mpmath.mpf(1),
            pi,
            sqrt2,
            sqrt3,
            sqrt6,
            atan_sqrt2,
            asinh_sqrt2,
        ],
    }
    for name, basis in bases_8.items():
        rel = mpmath.pslq(basis, maxcoeff=10000)
        if rel:
            names_list = name.strip("{}").split(",")
            report_relation(rel, names_list, basis)
        else:
            print(f"  No relation: {name}")

    # --- Big kitchen-sink ---
    print("\n--- 10-element PSLQ (maxcoeff=500) ---")
    big = {
        "{g0,1,pi,sqrt2,sqrt3,sqrt6,atan,log1p,log2,log3}": [
            g0,
            mpmath.mpf(1),
            pi,
            sqrt2,
            sqrt3,
            sqrt6,
            atan_sqrt2,
            log_1p_sqrt2,
            log2,
            log3,
        ],
    }
    for name, basis in big.items():
        rel = mpmath.pslq(basis, maxcoeff=500)
        if rel:
            print(f"  FOUND {name}: {rel}")
        else:
            print(f"  No relation: {name}")

    # ===================================================================
    # Phase 3: Verification of the discovered identity
    # ===================================================================
    print("\n" + "=" * 72)
    print("Phase 3: Verification of g0_tet = 4*sqrt(2)*asinh(sqrt(2)) - (2/3)*pi")
    print("=" * 72)

    mpmath.mp.dps = 55

    g0_exact = 4 * sqrt2 * asinh_sqrt2 - 2 * pi / 3
    print(f"\n  Closed form:  {mpmath.nstr(g0_exact, 50)}")
    print(f"  Numerical:    {mpmath.nstr(g0, 50)}")
    print(f"  Difference:   {mpmath.nstr(g0_exact - g0, 10)}")

    # Alternative forms
    print("\n  Equivalent expressions:")
    print("    4*sqrt(2)*log(sqrt(2)+sqrt(3)) - (2/3)*pi")

    g0_v2 = 2 * sqrt2 * mpmath.log(5 + 2 * sqrt6) - 2 * pi / 3
    print(
        f"    2*sqrt(2)*log(5+2*sqrt(6)) - (2/3)*pi   [match: {mpmath.nstr(g0_v2 - g0_exact, 5)}]"
    )

    g0_v3 = mpmath.mpf(2) / 3 * sqrt2 * mpmath.log(485 + 198 * sqrt6) - 2 * pi / 3
    print(
        f"    (2/3)*sqrt(2)*log(485+198*sqrt(6)) - (2/3)*pi   [match: {mpmath.nstr(g0_v3 - g0_exact, 5)}]"
    )
    print("    (2/3)*[sqrt(2)*log(485+198*sqrt(6)) - pi]")

    # Comparison with cube
    g0_cube = mpmath.mpf(4) / 3 * mpmath.log(70226 + 40545 * sqrt3) - 2 * pi
    print(
        f"\n  g0_cube = (4/3)*log(70226+40545*sqrt(3)) - 2*pi = {mpmath.nstr(g0_cube, 30)}"
    )
    print(
        f"  g0_tet  = 4*sqrt(2)*asinh(sqrt(2)) - (2/3)*pi    = {mpmath.nstr(g0_exact, 30)}"
    )
    print(f"  Ratio g0_tet/g0_cube = {mpmath.nstr(g0_exact / g0_cube, 15)}")

    # mpmath.identify
    print("\n  mpmath.identify():")
    mpmath.mp.dps = 15
    for label, val in [
        ("g0", g0),
        ("g0/2", g0 / 2),
        ("g0/3", g0 / 3),
        ("g0/4", g0 / 4),
        ("g0/pi", g0 / pi),
    ]:
        result = mpmath.identify(val)
        print(f"    identify({label:8s}) = {result}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    mpmath.mp.dps = 50
    g0_exact = 4 * mpmath.sqrt(2) * mpmath.asinh(mpmath.sqrt(2)) - 2 * mpmath.pi / 3
    print(f"""
  g0_tet = integral_{{tet}} 1/|x| dV

  Regular tetrahedron with vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
  Volume = 8/3, inscribed sphere radius = 1/sqrt(3)

  CLOSED FORM (found via PSLQ: 3*g0 + 2*pi - 12*sqrt(2)*asinh(sqrt(2)) = 0):

    g0_tet = 4*sqrt(2)*asinh(sqrt(2)) - (2/3)*pi

           = 4*sqrt(2)*log(sqrt(2) + sqrt(3)) - (2/3)*pi

           = 2*sqrt(2)*log(5 + 2*sqrt(6)) - (2/3)*pi

           = (2/3)*[sqrt(2)*log(485 + 198*sqrt(6)) - pi]

  Numerical value: {mpmath.nstr(g0_exact, 48)}

  Note: asinh(sqrt(2)) = log(sqrt(2) + sqrt(3))
        (sqrt(2) + sqrt(3))^2 = 5 + 2*sqrt(6)
        (5 + 2*sqrt(6))^3 = 485 + 198*sqrt(6)    [Pell equation 485^2 - 6*198^2 = 1]
""")


if __name__ == "__main__":
    main()
