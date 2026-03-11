"""PSLQ integer relation search for g0_tet.

g0_tet = 4.389580813420834468509850515535369500627825617197338958...
"""

import mpmath

mpmath.mp.dps = 50

G0 = mpmath.mpf("4.389580813420834468509850515535369500627825617197338958")

# Precompute constants
sqrt2 = mpmath.sqrt(2)
sqrt3 = mpmath.sqrt(3)
sqrt6 = mpmath.sqrt(6)
pi = mpmath.pi
log2 = mpmath.log(2)
log3 = mpmath.log(3)

atan_sqrt2 = mpmath.atan(sqrt2)
log_1p_sqrt2 = mpmath.log(1 + sqrt2)  # = arsinh(1)
asinh_sqrt2 = mpmath.asinh(sqrt2)  # = log(sqrt2 + sqrt3)
log_2p_sqrt3 = mpmath.log(2 + sqrt3)  # = arccosh(2)
atan_2sqrt2 = mpmath.atan(2 * sqrt2)
atan_sqrt2_over3 = mpmath.atan(sqrt2 / 3)
atan_2sqrt2_over3 = mpmath.atan(2 * sqrt2 / 3)
log_5p2sqrt6 = mpmath.log(5 + 2 * sqrt6)  # = 2*asinh(sqrt2)
log_1p_sqrt3 = mpmath.log(1 + sqrt3)
log_1p_sqrt6 = mpmath.log(1 + sqrt6)

# Catalog of candidate basis elements
catalog = {
    "1": mpmath.mpf(1),
    "pi": pi,
    "sqrt2": sqrt2,
    "sqrt3": sqrt3,
    "sqrt6": sqrt6,
    "log2": log2,
    "log3": log3,
    "atan(sqrt2)": atan_sqrt2,
    "log(1+sqrt2)": log_1p_sqrt2,
    "asinh(sqrt2)": asinh_sqrt2,
    "log(2+sqrt3)": log_2p_sqrt3,
    "atan(2sqrt2)": atan_2sqrt2,
    "atan(sqrt2/3)": atan_sqrt2_over3,
    "atan(2sqrt2/3)": atan_2sqrt2_over3,
    "log(5+2sqrt6)": log_5p2sqrt6,
    "log(1+sqrt3)": log_1p_sqrt3,
    "log(1+sqrt6)": log_1p_sqrt6,
    # Products
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
    "sqrt6*log2": sqrt6 * log2,
    "sqrt2*log3": sqrt2 * log3,
    "sqrt3*log3": sqrt3 * log3,
    "sqrt6*log3": sqrt6 * log3,
    "sqrt2*pi": sqrt2 * pi,
    "sqrt3*pi": sqrt3 * pi,
    "sqrt6*pi": sqrt6 * pi,
}


def report(rel, names, vals):
    """Print a found relation."""
    terms = []
    for c, n in zip(rel, names, strict=True):
        if c != 0:
            terms.append(f"{c}*{n}")
    print(f"  *** FOUND: {' + '.join(terms)} = 0")
    check = sum(c * v for c, v in zip(rel, vals, strict=True))
    print(f"      Verification: {mpmath.nstr(check, 15)}")
    if rel[0] != 0:
        g0_expr = -sum(c * v for c, v in zip(rel[1:], vals[1:], strict=True)) / rel[0]
        err = g0_expr - G0
        print(f"      g0 = {mpmath.nstr(g0_expr, 40)}")
        print(f"      Error: {mpmath.nstr(err, 10)}")


def search_4():
    """4-element PSLQ: g0 = a + b*pi + c*X."""
    print("\n" + "=" * 70)
    print("4-element PSLQ: g0, 1, pi, X")
    print("=" * 70)
    found_any = False
    for name, val in catalog.items():
        if name == "1" or name == "pi":
            continue
        names = ["g0", "1", "pi", name]
        vals = [G0, mpmath.mpf(1), pi, val]
        rel = mpmath.pslq(vals, maxcoeff=10000)
        if rel:
            report(rel, names, vals)
            found_any = True
    if not found_any:
        print("  No 4-element relations found.")


def search_log_forms():
    """4-element: g0 = a + b*pi + c*log(A+B*sqrt(N))."""
    print("\n" + "=" * 70)
    print("4-element: log(A+B*sqrt(N)) forms")
    print("=" * 70)
    found_any = False
    for N, Nname in [(2, "sqrt2"), (3, "sqrt3"), (6, "sqrt6")]:
        sqrtN = mpmath.sqrt(N)
        for A in range(1, 30):
            for B in range(1, 20):
                arg = A + B * sqrtN
                if arg <= 0:
                    continue
                val = mpmath.log(arg)
                names = ["g0", "1", "pi", f"log({A}+{B}{Nname})"]
                vals = [G0, mpmath.mpf(1), pi, val]
                rel = mpmath.pslq(vals, maxcoeff=10000)
                if rel:
                    report(rel, names, vals)
                    found_any = True
    if not found_any:
        print("  No log forms found.")


def search_5_selective():
    """5-element PSLQ with carefully chosen pairs."""
    print("\n" + "=" * 70)
    print("5-element PSLQ: g0, 1, pi, X, Y")
    print("=" * 70)

    # Key transcendentals
    trans = {
        "atan(sqrt2)": atan_sqrt2,
        "log(1+sqrt2)": log_1p_sqrt2,
        "asinh(sqrt2)": asinh_sqrt2,
        "log(2+sqrt3)": log_2p_sqrt3,
        "log2": log2,
        "log3": log3,
        "atan(2sqrt2)": atan_2sqrt2,
        "atan(sqrt2/3)": atan_sqrt2_over3,
        "atan(2sqrt2/3)": atan_2sqrt2_over3,
        "log(5+2sqrt6)": log_5p2sqrt6,
        "log(1+sqrt3)": log_1p_sqrt3,
    }
    # Algebraic numbers
    alg = {
        "sqrt2": sqrt2,
        "sqrt3": sqrt3,
        "sqrt6": sqrt6,
    }
    # Products
    prods = {
        "sqrt2*atan(sqrt2)": sqrt2 * atan_sqrt2,
        "sqrt3*atan(sqrt2)": sqrt3 * atan_sqrt2,
        "sqrt6*atan(sqrt2)": sqrt6 * atan_sqrt2,
        "sqrt2*log(1+sqrt2)": sqrt2 * log_1p_sqrt2,
        "sqrt3*log(1+sqrt2)": sqrt3 * log_1p_sqrt2,
        "sqrt6*log(1+sqrt2)": sqrt6 * log_1p_sqrt2,
        "sqrt2*asinh(sqrt2)": sqrt2 * asinh_sqrt2,
        "sqrt3*asinh(sqrt2)": sqrt3 * asinh_sqrt2,
    }

    # Combine all candidate elements
    all_elems = {**trans, **alg, **prods}
    elem_names = list(all_elems.keys())
    elem_vals = [all_elems[n] for n in elem_names]

    found_any = False
    count = 0
    total = len(elem_names) * (len(elem_names) - 1) // 2
    print(f"  Testing {total} pairs...")
    for i in range(len(elem_names)):
        for j in range(i + 1, len(elem_names)):
            count += 1
            names = ["g0", "1", "pi", elem_names[i], elem_names[j]]
            vals = [G0, mpmath.mpf(1), pi, elem_vals[i], elem_vals[j]]
            rel = mpmath.pslq(vals, maxcoeff=10000)
            if rel:
                report(rel, names, vals)
                found_any = True
    print(f"  Tested {count} pairs.")
    if not found_any:
        print("  No 5-element relations found.")


def search_6_selected():
    """6-element PSLQ with selected triples."""
    print("\n" + "=" * 70)
    print("6-element PSLQ: g0, 1, pi, X, Y, Z")
    print("=" * 70)

    # The most likely elements given the geometry
    key_trans = [
        ("atan(sqrt2)", atan_sqrt2),
        ("log(1+sqrt2)", log_1p_sqrt2),
        ("asinh(sqrt2)", asinh_sqrt2),
        ("log2", log2),
        ("log3", log3),
        ("atan(2sqrt2)", atan_2sqrt2),
        ("atan(sqrt2/3)", atan_sqrt2_over3),
    ]
    key_alg = [
        ("sqrt2", sqrt2),
        ("sqrt3", sqrt3),
        ("sqrt6", sqrt6),
    ]

    found_any = False
    count = 0

    # Trans + Trans + Alg
    for i in range(len(key_trans)):
        for j in range(i + 1, len(key_trans)):
            for k in range(len(key_alg)):
                count += 1
                names = [
                    "g0",
                    "1",
                    "pi",
                    key_trans[i][0],
                    key_trans[j][0],
                    key_alg[k][0],
                ]
                vals = [
                    G0,
                    mpmath.mpf(1),
                    pi,
                    key_trans[i][1],
                    key_trans[j][1],
                    key_alg[k][1],
                ]
                rel = mpmath.pslq(vals, maxcoeff=10000)
                if rel:
                    report(rel, names, vals)
                    found_any = True

    # Trans + Alg + Alg
    for i in range(len(key_trans)):
        for j in range(len(key_alg)):
            for k in range(j + 1, len(key_alg)):
                count += 1
                names = ["g0", "1", "pi", key_trans[i][0], key_alg[j][0], key_alg[k][0]]
                vals = [
                    G0,
                    mpmath.mpf(1),
                    pi,
                    key_trans[i][1],
                    key_alg[j][1],
                    key_alg[k][1],
                ]
                rel = mpmath.pslq(vals, maxcoeff=10000)
                if rel:
                    report(rel, names, vals)
                    found_any = True

    print(f"  Tested {count} triples.")
    if not found_any:
        print("  No 6-element relations found.")


def search_7_8():
    """7 and 8-element PSLQ with carefully chosen bases."""
    print("\n" + "=" * 70)
    print("7-8 element PSLQ")
    print("=" * 70)

    bases = {
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),sqrt2,sqrt3}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            sqrt2,
            sqrt3,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),sqrt2,sqrt6}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            sqrt2,
            sqrt6,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),sqrt3,sqrt6}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            sqrt3,
            sqrt6,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),log2,sqrt2}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            log2,
            sqrt2,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),log2,log3}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            log2,
            log3,
        ],
        "{g0,1,pi,atan(sqrt2),asinh(sqrt2),sqrt2,sqrt3}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            asinh_sqrt2,
            sqrt2,
            sqrt3,
        ],
        "{g0,1,pi,atan(sqrt2),log2,log3,sqrt2}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log2,
            log3,
            sqrt2,
        ],
        "{g0,1,pi,log(1+sqrt2),log2,sqrt2,sqrt3}": [
            G0,
            mpmath.mpf(1),
            pi,
            log_1p_sqrt2,
            log2,
            sqrt2,
            sqrt3,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),sqrt2,sqrt3,sqrt6}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            sqrt2,
            sqrt3,
            sqrt6,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),log2,log3,sqrt2}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            log2,
            log3,
            sqrt2,
        ],
        "{g0,1,pi,atan(sqrt2),asinh(sqrt2),sqrt2,sqrt3,sqrt6}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            asinh_sqrt2,
            sqrt2,
            sqrt3,
            sqrt6,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),log2,sqrt2,sqrt3}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            log2,
            sqrt2,
            sqrt3,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),log3,sqrt2,sqrt3}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            log3,
            sqrt2,
            sqrt3,
        ],
        "{g0,1,pi,sqrt2,sqrt3,sqrt6,log(1+sqrt2),log2}": [
            G0,
            mpmath.mpf(1),
            pi,
            sqrt2,
            sqrt3,
            sqrt6,
            log_1p_sqrt2,
            log2,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),atan(2sqrt2),sqrt2}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            atan_2sqrt2,
            sqrt2,
        ],
        "{g0,1,pi,atan(sqrt2),log(1+sqrt2),atan(sqrt2/3),sqrt2}": [
            G0,
            mpmath.mpf(1),
            pi,
            atan_sqrt2,
            log_1p_sqrt2,
            atan_sqrt2_over3,
            sqrt2,
        ],
    }
    for name, basis in bases.items():
        rel = mpmath.pslq(basis, maxcoeff=10000)
        if rel:
            names_list = name.strip("{}").split(",")
            report(rel, names_list, basis)
        else:
            print(f"  No relation: {name}")


def search_kitchen_sink():
    """Large PSLQ bases."""
    print("\n" + "=" * 70)
    print("Large PSLQ (10+ elements, maxcoeff=500)")
    print("=" * 70)

    bases = {
        "FULL_A: {g0,1,pi,sqrt2,sqrt3,sqrt6,atan,log1sqrt2,log2,log3}": [
            G0,
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
        "FULL_B: {g0,1,pi,sqrt2,sqrt3,sqrt6,atan,asinh,log2,log3}": [
            G0,
            mpmath.mpf(1),
            pi,
            sqrt2,
            sqrt3,
            sqrt6,
            atan_sqrt2,
            asinh_sqrt2,
            log2,
            log3,
        ],
        "FULL_C: {g0,1,pi,sqrt2,sqrt3,sqrt6,atan,atan2,log1sqrt2,log2}": [
            G0,
            mpmath.mpf(1),
            pi,
            sqrt2,
            sqrt3,
            sqrt6,
            atan_sqrt2,
            atan_2sqrt2,
            log_1p_sqrt2,
            log2,
        ],
        "FULL_D: {g0,1,pi,sqrt2,sqrt3,sqrt6,atan,log1sqrt2,atan2,asinh}": [
            G0,
            mpmath.mpf(1),
            pi,
            sqrt2,
            sqrt3,
            sqrt6,
            atan_sqrt2,
            log_1p_sqrt2,
            atan_2sqrt2,
            asinh_sqrt2,
        ],
        "FULL_E: {g0,1,pi,sqrt2,sqrt3,sqrt6,atan,log1sqrt2,atan_s2o3,log2}": [
            G0,
            mpmath.mpf(1),
            pi,
            sqrt2,
            sqrt3,
            sqrt6,
            atan_sqrt2,
            log_1p_sqrt2,
            atan_sqrt2_over3,
            log2,
        ],
        "FULL_F: {g0,1,pi,sqrt2,sqrt3,sqrt6,atan,log1sqrt2,atan_2s2o3,log2}": [
            G0,
            mpmath.mpf(1),
            pi,
            sqrt2,
            sqrt3,
            sqrt6,
            atan_sqrt2,
            log_1p_sqrt2,
            atan_2sqrt2_over3,
            log2,
        ],
    }
    for name, basis in bases.items():
        rel = mpmath.pslq(basis, maxcoeff=500)
        if rel:
            print(f"  FOUND {name}: {rel}")
            check = sum(c * v for c, v in zip(rel, basis, strict=True))
            print(f"    Verification: {mpmath.nstr(check, 15)}")
        else:
            print(f"  No relation: {name}")


def search_ratios():
    """Try mpmath.identify on ratios and differences."""
    print("\n" + "=" * 70)
    print("mpmath.identify() and ratios")
    print("=" * 70)

    mpmath.mp.dps = 15
    g0 = G0

    for label, val in [
        ("g0", g0),
        ("g0/2", g0 / 2),
        ("g0/3", g0 / 3),
        ("g0/4", g0 / 4),
        ("g0/6", g0 / 6),
        ("g0/8", g0 / 8),
        ("g0*3/8", g0 * 3 / 8),
        ("g0/pi", g0 / pi),
        ("g0/sqrt2", g0 / sqrt2),
        ("g0/sqrt3", g0 / sqrt3),
        ("g0/sqrt6", g0 / sqrt6),
    ]:
        result = mpmath.identify(val)
        print(f"  identify({label:12s}) = {result}")

    # Differences
    print("\n  g0 - c*X residual identification:")
    mpmath.mp.dps = 15
    for c_num, c_den in [
        (0, 1),
        (1, 1),
        (-1, 1),
        (2, 1),
        (-2, 1),
        (1, 2),
        (-1, 2),
        (4, 3),
        (-4, 3),
        (1, 3),
        (2, 3),
        (-2, 3),
    ]:
        c = mpmath.mpf(c_num) / c_den
        for x_name, x_val in [
            ("pi", pi),
            ("atan(sqrt2)", atan_sqrt2),
            ("log(1+sqrt2)", log_1p_sqrt2),
            ("asinh(sqrt2)", asinh_sqrt2),
        ]:
            if c_num == 0 and x_name != "pi":
                continue
            diff = G0 - c * x_val
            result = mpmath.identify(diff)
            if result:
                print(f"    g0 - ({c_num}/{c_den})*{x_name} = {result}")


def main():
    print(f"g0_tet = {mpmath.nstr(G0, 50)}")
    print(f"Working precision: {mpmath.mp.dps} digits")

    search_4()
    search_log_forms()
    search_5_selective()
    search_6_selected()
    search_7_8()
    search_kitchen_sink()
    search_ratios()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
