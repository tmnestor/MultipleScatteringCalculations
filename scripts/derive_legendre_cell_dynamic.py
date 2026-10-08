#!/usr/bin/env python3
"""The single-site T-matrix of the linear Legendre cell, power by power in k_S h, in closed form.

For a cell of half-width h with a uniform contrast, T36 = F (M - K(0) E)^-1 M
(``graded_voxel.site.single_site_t36``). The self block is a power series in the wavenumber
(``blocks.near_block_series``),

    K(0) = sum_n (k_S h)^n K_n,
    K_n = (1 / 4 pi mu) sum_idx [ i^n / n! TA[idx] h^(5 - |idx|) U(n - 1, idx)
                                  + i^(n + 2) (1 - gamma^(n + 2)) / (n + 2)! TB[idx] h^(7 - |idx|) U(n + 1, idx) ],

gamma = beta / alpha, TA and TB the propagator's tables (``blocks.family_tables``), and U(m, idx) the
universal moments of the self cell, U_ac(m, idx) = int W_ac(s) d^idx r^m ds for h = 1. Every U is derived
here in exact arithmetic:

  odd m:  by the volume -> face -> edge reductions of ``derive_graded_voxel_site_blocks.py`` (r^m is
          singular or non-smooth at s = 0);
  even m: r^m is a polynomial and the integral over each box is elementary.

So K_n is exact for every n, its entries combinations with rational coefficients of a few constants. The
script checks:

  1. every exact U against ``blocks.universal_moment`` (master integrals at 40 digits, Gauss rules);
  2. the exact series summed to n = N against ``near_block_series`` and against ``near_block``
     (quadrature) at several k_S h, the difference falling as (k_S h)^(N + 1);
  3. the constants that occur, order by order;
  4. T36 from the exact series against ``single_site_t36`` with the quadrature self block;
  5. the reduced blocks of the dynamic T36 on the irreducible representations of the cube group.

Run:  python -u scripts/derive_legendre_cell_dynamic.py [N]     (N = highest power of k_S h, default 4)
Writes, in LatexPDFs/LegendreCellTMatrix/data: self_moments_exact.pkl (every U, exact; reused on later runs),
series_blocks.pkl (A_n and B_n, exact) and self_blocks_irreps.txt (the reduced blocks on the six constants)
"""

import itertools
import multiprocessing
import pickle
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import derive_graded_voxel_site_blocks as base  # noqa: E402

from cubic_scattering import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel.blocks import (  # noqa: E402
    family_tables,
    near_block,
    near_block_series,
    universal_moment,
)

OUT = ROOT / "LatexPDFs" / "LegendreCellTMatrix" / "data"
CACHE = OUT / "self_moments_exact.pkl"
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)


# ---------------------------------------------------------------------------
# even powers: r^m is a polynomial, and its box and face integrals are elementary
# ---------------------------------------------------------------------------


def _multinomial_terms(n: int):
    k = n // 2
    for i in range(k + 1):
        for j in range(k - i + 1):
            yield i, j, k - i - j, sp.factorial(k) / (sp.factorial(i) * sp.factorial(j) * sp.factorial(k - i - j))


@cache
def _poly_box(p: int, q: int, r: int, a: int, b: int, c: int, n: int) -> sp.Expr:
    """int_0^a int_0^b int_0^c x^p y^q z^r (x^2 + y^2 + z^2)^(n/2), n even >= 0."""
    tot = sp.Integer(0)
    for i, j, l, mult in _multinomial_terms(n):
        tot += (
            mult
            * sp.Integer(a) ** (p + 2 * i + 1) / (p + 2 * i + 1)
            * sp.Integer(b) ** (q + 2 * j + 1) / (q + 2 * j + 1)
            * sp.Integer(c) ** (r + 2 * l + 1) / (r + 2 * l + 1)
        )
    return tot


@cache
def _poly_face(q: int, r: int, a: int, b: int, c: int, n: int) -> sp.Expr:
    """int_0^b int_0^c y^q z^r (a^2 + y^2 + z^2)^(n/2), n even >= 0 (the face x = a)."""
    tot = sp.Integer(0)
    for i, j, l, mult in _multinomial_terms(n):
        tot += (
            mult
            * sp.Integer(a) ** (2 * i)
            * sp.Integer(b) ** (q + 2 * j + 1) / (q + 2 * j + 1)
            * sp.Integer(c) ** (r + 2 * l + 1) / (r + 2 * l + 1)
        )
    return tot


_odd_box, _odd_face = base.box_x, base.face_x


def _box(p, q, r, a, b, c, m):
    return _poly_box(p, q, r, a, b, c, m) if m % 2 == 0 else _odd_box(p, q, r, a, b, c, m)


def _face(q, r, a, b, c, m):
    return _poly_face(q, r, a, b, c, m) if m % 2 == 0 else _odd_face(q, r, a, b, c, m)


base.box_x, base.face_x = _box, _face  # moment_x looks them up at call time


# ---------------------------------------------------------------------------
# 1. the exact universal moments of the self cell
# ---------------------------------------------------------------------------


def needed_terms(n_max: int) -> list[tuple[int, tuple[int, ...]]]:
    """Every (m, idx) the self block needs through (k_S h)^n_max: TA at m = n - 1, TB at m = n + 1."""
    ta, tb = family_tables()
    terms = set()
    for n in range(n_max + 1):
        terms.update((n - 1, idx) for idx in ta)
        terms.update((n + 1, idx) for idx in tb)
    # an even power of r has no derivatives beyond its degree
    return sorted(t for t in terms if t[0] % 2 or len(t[1]) <= t[0])


def _exact_moment(term: tuple[int, tuple[int, ...]]):
    m, idx = term
    mat = [[sp.expand(base.moment_x(m, idx, a, c)) for c in range(4)] for a in range(4)]
    return term, mat


def exact_moments(n_max: int, workers: int = 4) -> dict[tuple[int, tuple[int, ...]], list[list[sp.Expr]]]:
    have: dict = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    todo = [t for t in needed_terms(n_max) if t not in have]
    if todo:
        t0 = time.time()
        with multiprocessing.Pool(workers) as pool:
            for k, (term, mat) in enumerate(pool.imap_unordered(_exact_moment, todo), 1):
                have[term] = mat
                if k % 20 == 0 or k == len(todo):
                    print(f"   {k}/{len(todo)} moments ({time.time() - t0:.0f} s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(have))
    return {t: have[t] for t in needed_terms(n_max)}


# ---------------------------------------------------------------------------
# exact arithmetic on combinations of constants: {monomial in the constants: rational}
# ---------------------------------------------------------------------------

Lin = dict  # {sympy monomial (product of constants, or 1): sympy Rational}


def lin(expr: sp.Expr) -> Lin:
    e = sp.expand(expr)
    out: Lin = {}
    for mono, coef in e.as_coefficients_dict().items():
        if coef != 0:
            out[mono] = out.get(mono, 0) + sp.Rational(coef)
    return {k: v for k, v in out.items() if v != 0}


def lin_add(acc: Lin, x: Lin, scale=1) -> None:
    for k, v in x.items():
        acc[k] = acc.get(k, 0) + scale * v
        if acc[k] == 0:
            del acc[k]


def lin_value(x: Lin, values: dict) -> float:
    return float(sum(float(v) * values[k] for k, v in x.items()))


def lin_expr(x: Lin) -> sp.Expr:
    return sp.Add(*[v * k for k, v in x.items()]) if x else sp.Integer(0)


# ---------------------------------------------------------------------------
# 2. the self block, power by power: 4 pi mu K_n = i^n [A_n / n! - (1 - gamma^(n+2)) B_n / (n+2)!]
#    at h = 1, as 36 x 36 arrays of Lin (index 9 a + i: test function a, component i)
# ---------------------------------------------------------------------------


def family_block(exact, m: int, table: dict) -> list[list[Lin]]:
    """sum_idx table[idx] U(m, idx), a 36 x 36 array of Lin (zero where U vanishes)."""
    out = [[{} for _ in range(36)] for _ in range(36)]
    for idx, coef in table.items():
        if (m, idx) not in exact:
            continue
        u = exact[(m, idx)]
        rat = [[sp.nsimplify(coef[i, j], rational=True) for j in range(9)] for i in range(9)]
        for a, c in itertools.product(range(4), repeat=2):
            ul = lin(u[a][c])
            if not ul:
                continue
            for i, j in itertools.product(range(9), repeat=2):
                if rat[i][j] != 0:
                    lin_add(out[9 * a + i][9 * c + j], ul, rat[i][j])
    return out


def series_blocks(exact, n_max: int) -> list[tuple[list[list[Lin]], list[list[Lin]]]]:
    """[(A_n, B_n)] for n = 0..n_max."""
    ta, tb = family_tables()
    return [(family_block(exact, n - 1, ta), family_block(exact, n + 1, tb)) for n in range(n_max + 1)]


def constants_of(blocks) -> list[sp.Expr]:
    found = set()
    for a_n, b_n in blocks:
        for mat in (a_n, b_n):
            for row in mat:
                for x in row:
                    found.update(x)
    return sorted(found, key=lambda e: (len(str(e)), str(e)))


def numeric_k(blocks, values: dict, kh: float, h: float, gamma: float, mu: float, n_max: int) -> np.ndarray:
    """K(0) = sum_n (k_S h)^n K_n at half-width h, shape (4, 4, 9, 9).

    An entry of row type r and column type c (0 displacement or force, 1 strain or moment) scales as
    h^(5 - r - c): K(h) = h^3 D K(1) D, D = diag(h on the displacement rows, 1 on the strain rows).
    """
    out = np.zeros((36, 36), dtype=complex)
    for n in range(n_max + 1):
        a_n, b_n = blocks[n]
        wa = 1j**n / float(sp.factorial(n))
        wb = -(1j**n) * (1.0 - gamma ** (n + 2)) / float(sp.factorial(n + 2))
        for p, q in itertools.product(range(36), repeat=2):
            v = 0.0
            if a_n[p][q]:
                v += wa * lin_value(a_n[p][q], values)
            if b_n[p][q]:
                v += wb * lin_value(b_n[p][q], values)
            if v:
                out[p, q] += kh**n * v
    d = np.tile(np.r_[np.full(3, h), np.ones(6)], 4)
    out = h**3 * d[:, None] * out * d[None, :] / (4.0 * np.pi * mu)
    return out.reshape(4, 9, 4, 9).transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# 5. the cube group: symmetry-adapted vectors, one partner row of each irrep
#    index 9 a + i: test function a (1, xi_0, xi_1, xi_2), component i (u_0..u_2, then Voigt
#    eps_00, eps_11, eps_22, 2 eps_12, 2 eps_02, 2 eps_01)
# ---------------------------------------------------------------------------

U0, U1, U2 = 0, 1, 2
E00, E11, E22, G12, G02, G01 = 3, 4, 5, 6, 7, 8
BASIS: dict[str, list[dict[tuple[int, int], int]]] = {
    # even: the displacement's first moments (xi_i u_j) and the uniform strain
    "A1g": [{(1, U0): 1, (2, U1): 1, (3, U2): 1}, {(0, E00): 1, (0, E11): 1, (0, E22): 1}],
    "Eg": [{(1, U0): 2, (2, U1): -1, (3, U2): -1}, {(0, E00): 2, (0, E11): -1, (0, E22): -1}],
    "T1g": [{(2, U2): 1, (3, U1): -1}],
    "T2g": [{(2, U2): 1, (3, U1): 1}, {(0, G12): 1}],
    # odd: the uniform displacement and the strain's first moments
    "T1u": [{(0, U0): 1}, {(1, E00): 1}, {(1, E11): 1, (1, E22): 1}, {(2, G01): 1, (3, G02): 1}],
    "T2u": [{(1, E11): 1, (1, E22): -1}, {(2, G01): 1, (3, G02): -1}],
    "A2u": [{(1, G12): 1, (2, G02): 1, (3, G01): 1}],
    "Eu": [{(2, G02): 1, (3, G01): -1}],
}
GRAM = [sp.Integer(8)] + [sp.Rational(8, 3)] * 3  # <L_a, L_a> at h = 1


def _vectors(name: str) -> sp.Matrix:
    cols = []
    for entries in BASIS[name]:
        v = sp.zeros(36, 1)
        for (a, i), val in entries.items():
            v[9 * a + i] = val
        cols.append(v)
    return sp.Matrix.hstack(*cols)


def _apply(mat: list[list[Lin]], v: sp.Matrix) -> list[Lin]:
    out = []
    for p in range(36):
        acc: Lin = {}
        for q in range(36):
            if v[q] != 0 and mat[p][q]:
                lin_add(acc, mat[p][q], v[q])
        out.append(acc)
    return out


def reduce_on_irreps(blocks) -> dict:
    """{irrep: V, M_Gamma, NA[n], NB[n] (exact, 4 pi mu M_Gamma^-1 V^+ A_n V and likewise for B_n), leak}."""
    out = {}
    for name in BASIS:
        v = _vectors(name)
        k = v.shape[1]
        left = (v.T * v).inv() * v.T  # V^+
        m_g = [sum(GRAM[p // 9] * v[p, j] ** 2 for p in range(36)) / sum(v[p, j] ** 2 for p in range(36))
               for j in range(k)]
        leak = 0
        na, nb = [], []
        for a_n, b_n in blocks:
            for mat, store in ((a_n, na), (b_n, nb)):
                cols = [_apply(mat, v[:, j]) for j in range(k)]
                red = [[{} for _ in range(k)] for _ in range(k)]
                for i, j in itertools.product(range(k), repeat=2):
                    for p in range(36):
                        if left[i, p] != 0 and cols[j][p]:
                            lin_add(red[i][j], cols[j][p], left[i, p] / m_g[i])
                # invariance: K V - V K_Gamma = 0 exactly (K_Gamma = M_Gamma N_Gamma)
                for j in range(k):
                    for p in range(36):
                        resid = dict(cols[j][p])
                        for i in range(k):
                            if v[p, i] != 0:
                                lin_add(resid, red[i][j], -v[p, i] * m_g[i])
                        leak = max(leak, len(resid))
                store.append(red)
        out[name] = {"V": np.array(v, dtype=float), "M": m_g, "NA": na, "NB": nb, "leak": leak}
    return out


def irrep_t(r: dict, values: dict, kh: float, gamma: float, mu: float, delta9: np.ndarray, n_max: int):
    """T_Gamma = M_Gamma Delta_Gamma (I - N_Gamma Delta_Gamma)^-1 at h = 1, N_Gamma from the series."""
    v = r["V"]
    k = v.shape[1]
    n_g = np.zeros((k, k), dtype=complex)
    for n in range(n_max + 1):
        wa = 1j**n / float(sp.factorial(n))
        wb = -(1j**n) * (1.0 - gamma ** (n + 2)) / float(sp.factorial(n + 2))
        for i, j in itertools.product(range(k), repeat=2):
            n_g[i, j] += kh**n * (wa * lin_value(r["NA"][n][i][j], values) + wb * lin_value(r["NB"][n][i][j], values))
    n_g /= 4.0 * np.pi * mu
    pinv = np.linalg.solve(v.T @ v, v.T)
    d_g = pinv @ np.kron(np.eye(4), delta9) @ v
    m_g = np.diag([float(x) for x in r["M"]])
    return m_g @ d_g @ np.linalg.inv(np.eye(k) - n_g @ d_g)


#: arsinh(1/sqrt2) = ln((1 + sqrt3)/sqrt2) = ln(2 + sqrt3) / 2: both logarithms are of fundamental units,
#: of Q(sqrt2) and of Q(sqrt3)
PI_BASIS = ["1/pi", "1", "sqrt2/pi", "sqrt3/pi", "ln(1+sqrt2)/pi", "ln(2+sqrt3)/pi"]


def _on_pi_basis(x: Lin) -> list[sp.Rational]:
    """x / (4 pi) as rational coefficients on PI_BASIS, arsinh(1/sqrt2) written as ln(2 + sqrt3) / 2."""
    l1, l2 = sp.log(1 + sp.sqrt(2)), sp.asinh(sp.sqrt(2) / 2)
    keys = {sp.Integer(1): 0, sp.pi: 1, sp.sqrt(2): 2, sp.sqrt(3): 3, l1: 4, l2: 5}
    out = [sp.Integer(0)] * 6
    for mono, coef in x.items():
        out[keys[mono]] += coef / 4 * (sp.Rational(1, 2) if mono == l2 else 1)
    return out


def write_blocks(reduced: dict, n_max: int) -> None:
    assert abs(float(sp.N(sp.asinh(sp.sqrt(2) / 2) - sp.log(2 + sp.sqrt(3)) / 2, 50))) < 1e-45
    lines = [
        "The single-site blocks of the linear Legendre cell, uniform contrast, power by power in k_S h.",
        "N_Gamma = (1/mu) sum_n (k_S h)^n i^n [ P_n / n! - (1 - gamma^(n+2)) Q_n / (n+2)! ],  gamma = beta/alpha,",
        "an entry of row type r and column type c (0 displacement, 1 strain) scaled by h^(2 - r - c) relative",
        f"to h = 1. Each entry of P_n and Q_n: rational coefficients on {PI_BASIS}.",
        "",
    ]
    for name, r in reduced.items():
        lines.append(f"{name}: vectors {BASIS[name]}  M_Gamma / h^3 = {[str(x) for x in r['M']]}")
        k = len(r["M"])
        for n in range(n_max + 1):
            for tag, store in (("P", r["NA"]), ("Q", r["NB"])):
                for i, j in itertools.product(range(k), repeat=2):
                    x = store[n][i][j]
                    if x:
                        val = float(lin_value(x, {c: float(sp.N(c, 30)) for c in x})) / (4 * np.pi)
                        lines.append(f"  {tag}_{n}[{i},{j}] = {[str(c) for c in _on_pi_basis(x)]}   ({val:+.12e})")
        lines.append("")
    (OUT / "self_blocks_irreps.txt").write_text("\n".join(lines) + "\n")
    print(f"   wrote {OUT / 'self_blocks_irreps.txt'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ok: list[bool] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        ok.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)

    # 1. exact moments against the package's
    exact = exact_moments(n_max)
    worst, where = 0.0, None
    for (m, idx), mat in exact.items():
        got = np.array([[float(sp.N(e, 30)) for e in row] for row in mat])
        num = universal_moment(m, idx, (0, 0, 0), 10, 4)[:, :4]
        err = float(np.abs(got - num).max() / max(np.abs(num).max(), 1e-300)) if np.abs(num).max() else float(
            np.abs(got).max()
        )
        if err > worst:
            worst, where = err, (m, idx)
    check(f"1. {len(exact)} exact self-cell moments, m = -1..{n_max + 1}, equal the package's", worst < 1e-12,
          f"{worst:.1e} at {where}")

    # 2. the series of the self block against the package's, at several k_S h
    blocks = series_blocks(exact, n_max)
    consts = constants_of(blocks)
    values = {c: float(sp.N(c, 40)) for c in consts}
    print(f"   constants in K_0..K_{n_max}: {consts}")
    h, gamma = 1.25, REF.beta / REF.alpha
    rows = []
    for kh in (0.02, 0.05, 0.1, 0.2):
        omega = kh / h * REF.beta
        k_ex = numeric_k(blocks, values, kh, h, gamma, REF.mu, n_max)
        k_ser = near_block_series((0, 0, 0), h, omega, REF, n_source=10, n_test=4)[:, :4]
        k_quad = near_block((0, 0, 0), h, omega, REF, n_q=14)[:, :4]
        scale = np.abs(k_ser).max()
        rows.append((kh, np.abs(k_ex - k_ser).max() / scale, np.abs(k_ex - k_quad).max() / scale,
                     np.abs(k_ser - k_quad).max() / scale))
    for kh, e_ser, e_quad, e_pq in rows:
        print(f"   k_S h = {kh:<5}: exact series to n = {n_max} against the package series {e_ser:.1e}, "
              f"against quadrature {e_quad:.1e} (package series against quadrature {e_pq:.1e})")
    slope = np.log(rows[-1][1] / rows[-2][1]) / np.log(rows[-1][0] / rows[-2][0])
    check(f"2a. the truncated exact series differs from the full one at order (k_S h)^{n_max + 1}",
          abs(slope - (n_max + 1)) < 0.3, f"slope {slope:.2f}")
    check("2b. at k_S h = 0.02 the truncated series is within its truncation of the package series",
          rows[0][1] < 1e-10 * 10 ** (5 - n_max), f"{rows[0][1]:.1e}")

    # 3. which constants occur at each order
    for n, (a_n, b_n) in enumerate(blocks):
        cs = constants_of([(a_n, b_n)])
        print(f"   K_{n}: {len(cs)} constants {cs}")

    # 4. T36 from the exact series against the single site with the quadrature self block
    from cubic_scattering.graded_voxel.site import contrast_operator, single_site_t36

    dlam, dmu, drho = 2.0e9, 1.0e9, 100.0
    t_rows = []
    for kh in (0.05, 0.1, 0.2):
        omega = kh / h * REF.beta
        delta = np.zeros((4, 9, 9), dtype=complex)
        delta[0] = contrast_operator(dlam, dmu, drho, omega)
        pad = np.zeros((4, 10, 9, 9), dtype=complex)
        pad[:, :4] = numeric_k(blocks, values, kh, h, gamma, REF.mu, n_max)
        t_ex = single_site_t36(h, delta, pad)
        t_quad = single_site_t36(h, delta, near_block((0, 0, 0), h, omega, REF, n_q=14))
        t_rows.append((kh, np.abs(t_ex - t_quad).max() / np.abs(t_quad).max()))
        print(f"   k_S h = {kh:<5}: T36 from the exact series against T36 by quadrature {t_rows[-1][1]:.1e}")
    slope = np.log(t_rows[-1][1] / t_rows[-2][1]) / np.log(t_rows[-1][0] / t_rows[-2][0])
    check(f"4. T36 from the series to (k_S h)^{n_max} carries an error of order (k_S h)^{n_max + 1}",
          abs(slope - (n_max + 1)) < 0.4, f"slope {slope:.2f}")
    with open(OUT / "series_blocks.pkl", "wb") as fh:
        pickle.dump({"n_max": n_max, "blocks": blocks, "constants": consts}, fh)

    # 5. the reduced blocks on the irreducible representations of O_h
    reduced = reduce_on_irreps(blocks)
    worst_leak = max(r["leak"] for r in reduced.values())
    check("5a. each irrep's span is invariant under every K_n (exactly)", worst_leak == 0, f"{worst_leak}")
    a1g = reduced["A1g"]["NA"][0][1][1], reduced["A1g"]["NB"][0][1][1]  # strain-strain, n = 0
    g = sp.Symbol("gamma", positive=True)
    static = sp.simplify(lin_expr(a1g[0]) - (1 - g**2) / 2 * lin_expr(a1g[1]))
    check("5b. static uniform dilatation: 4 pi mu N = -4 pi gamma^2 / 3, i.e. N = -1 / 3 (lambda + 2 mu)",
          sp.simplify(static + 4 * sp.pi * g**2 / 3) == 0, str(static))
    worst = 0.0
    for kh in (0.05, 0.1, 0.2):
        omega = kh * REF.beta  # h = 1
        delta = np.zeros((4, 9, 9), dtype=complex)
        delta[0] = contrast_operator(dlam, dmu, drho, omega)
        t36 = single_site_t36(1.0, delta, near_block((0, 0, 0), 1.0, omega, REF, n_q=14))
        err = max(
            np.abs(t36 @ r["V"] - r["V"] @ irrep_t(r, values, kh, gamma, REF.mu, delta[0], n_max)).max()
            for r in reduced.values()
        ) / np.abs(t36).max()
        if kh == 0.05:
            worst = err
        print(f"   k_S h = {kh:<5}: T36 against the irrep blocks of the series, worst over irreps {err:.1e}")
    check("5c. T36 V = V T_Gamma for every irrep, to the truncation of the series", worst < 1e-8, f"{worst:.1e}")
    # 6. the static dilatation in closed form is Eshelby's sphere result: T = 8 h^3 D / (1 + D / 3(lambda + 2 mu)),
    #    D = 3 dlambda + 2 dmu = 3 dK, i.e. 8 h^3 3 dK (K + 4 mu / 3) / (K' + 4 mu / 3)
    lam = REF.rho * (REF.alpha**2 - 2 * REF.beta**2)
    om = 1e-3
    delta = np.zeros((4, 9, 9), dtype=complex)
    delta[0] = contrast_operator(dlam, dmu, drho, om)
    t36 = single_site_t36(1.0, delta, near_block((0, 0, 0), 1.0, om, REF, n_q=14))
    v = reduced["A1g"]["V"]
    t_a1g = np.linalg.solve(v.T @ v, v.T @ t36 @ v)[1, 1].real
    d_bulk = 3 * dlam + 2 * dmu
    k_bulk = lam + 2 * REF.mu / 3
    eshelby = 8.0 * 3 * (d_bulk / 3) * (k_bulk + 4 * REF.mu / 3) / (k_bulk + d_bulk / 3 + 4 * REF.mu / 3)
    err = abs(t_a1g / eshelby - 1)
    check("6. static uniform dilatation of T36 equals Eshelby's sphere result in closed form", err < 1e-12,
          f"{err:.1e}")
    # 7. each block holds at most one displacement vector; its column of X = N Delta vanishes as k_S h -> 0, and
    #    the Schur complement on it, (a, b, c, S), reproduces the inverse of I - X
    worst_inv, worst_col = 0.0, 0.0
    for kh in (1e-6, 0.1, 0.3):
        d9 = contrast_operator(dlam, dmu, drho, kh * REF.beta)
        for name, r in reduced.items():
            k = len(r["M"])
            n_g = np.zeros((k, k), dtype=complex)
            for n in range(n_max + 1):
                wa = 1j**n / float(sp.factorial(n))
                wb = -(1j**n) * (1.0 - gamma ** (n + 2)) / float(sp.factorial(n + 2))
                n_g += kh**n * np.array([[wa * lin_value(r["NA"][n][i][j], values)
                                          + wb * lin_value(r["NB"][n][i][j], values) for j in range(k)]
                                         for i in range(k)])
            n_g /= 4.0 * np.pi * REF.mu
            v = r["V"]
            x = n_g @ np.linalg.solve(v.T @ v, v.T @ np.kron(np.eye(4), d9) @ v)
            disp = [i for i, vec in enumerate(BASIS[name]) if all(c < 3 for _, c in vec)]
            assert len(disp) <= 1
            if not disp or k == 1:
                continue
            if kh == 1e-6:
                worst_col = max(worst_col, np.abs(x[:, disp[0]]).max() / np.abs(x).max())
            o = disp + [i for i in range(k) if i not in disp]
            xo = x[np.ix_(o, o)]
            a, b, c = 1 - xo[0, 0], -xo[0, 1:], -xo[1:, 0]
            s_inv = np.linalg.inv(np.eye(k - 1) - xo[1:, 1:] - np.outer(c, b) / a)
            inv = np.empty((k, k), dtype=complex)
            inv[0, 0] = 1 / a + b @ s_inv @ c / a**2
            inv[0, 1:], inv[1:, 0], inv[1:, 1:] = -b @ s_inv / a, -s_inv @ c / a, s_inv
            back = np.empty_like(inv)
            back[np.ix_(o, o)] = inv
            direct = np.linalg.inv(np.eye(k) - x)
            worst_inv = max(worst_inv, np.abs(back - direct).max() / np.abs(direct).max())
    check("7. the displacement column of X vanishes statically; the Schur complement gives every inverse",
          worst_col < 1e-12 and worst_inv < 1e-13, f"column {worst_col:.1e}, inverse {worst_inv:.1e}")
    # 8. the uniform-shear channels E_g (dimension 2) and T_2g (dimension 3) average to the sphere's shear
    #    constant, exactly and for every Poisson ratio: (2 N_Eg + 3 N_T2g) / 5 = -(4 - 5 nu) / [15 mu (1 - nu)].
    #    Static 4 pi mu N = A_0 - (1 - g^2) B_0 / 2 on the uniform-strain vector, g = beta / alpha symbolic.
    g = sp.Symbol("g", positive=True)

    def static_n(name: str) -> sp.Expr:
        r = reduced[name]
        i = next(j for j, vec in enumerate(BASIS[name]) if all(a == 0 and c >= 3 for a, c in vec))
        return (lin_expr(r["NA"][0][i][i]) - (1 - g**2) / 2 * lin_expr(r["NB"][0][i][i])) / (4 * sp.pi)

    nu = (1 - 2 * g**2) / (2 * (1 - g**2))
    avg = (2 * static_n("Eg") + 3 * static_n("T2g")) / 5
    sphere = -(4 - 5 * nu) / (15 * (1 - nu))
    resid = sp.simplify(sp.expand_log(sp.expand(avg - sphere), force=True))
    print(f"   mu N_Eg = {float(static_n('Eg').subs(g, gamma)):.6f}, mu N_T2g = "
          f"{float(static_n('T2g').subs(g, gamma)):.6f}, average {float(avg.subs(g, gamma)):.7f}, "
          f"sphere {float(sphere.subs(g, gamma)):.7f}")
    check("8. (2 N_Eg + 3 N_T2g) / 5 equals the sphere's Eshelby shear constant, exactly for every nu",
          resid == 0, f"residual {resid}")
    write_blocks(reduced, n_max)

    print(f"{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
