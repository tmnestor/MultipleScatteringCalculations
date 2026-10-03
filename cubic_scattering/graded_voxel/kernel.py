"""The 9 x 9 point propagator [[G, C], [H, S]], vectorised and accurate at every k r.

The closed form of this object cancels catastrophically as k r -> 0 (eps / (k r)^2), so every point
propagator in the package is evaluated here: ``resonance_tmatrix.elastodynamic_greens_deriv`` and
``_propagator_block_9x9`` delegate to ``greens_tensors`` and ``kernel_9x9``.  Here

    G_ij = (1 / 4 pi mu) [ delta_ij g_b(r) + (1 / k_b^2) d_i d_j (g_b - g_a)(r) ],   g_k = e^{ikr} / r,

and every derivative of a radial function f is a combination of F_q = ((1/r) d/dr)^q f.  For
k_S r <= SERIES_LIMIT the F_q come from the power series g_k = sum_t (ik)^t r^(t-1) / t!, term by term
exactly; above it from closed forms generated once with sympy.  The Voigt contraction is the package's
``_voigt_contract``, probed once into linear maps, so the convention cannot drift.

The static part is the Kelvin tensor: the t = 0 term of the delta series (1/r) and the t = 2 term of the
derivative series (b2 d d r, b2 = -(1 - beta^2/alpha^2)/2).  ``static=False`` leaves the dynamic
remainder, which is at most weakly (1/r) singular.
"""

import math
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp
import yaml
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from ..resonance_tmatrix import _voigt_contract

SERIES_LIMIT = 0.5
N_SERIES = 24


@lru_cache(maxsize=1)
def voigt_maps() -> tuple[NDArray, NDArray, NDArray]:
    """The package's Voigt contraction as linear maps: C from Gd, H from Gd, S from Gdd."""
    mc, mh, ms = np.zeros((3, 6, 27)), np.zeros((6, 3, 27)), np.zeros((6, 6, 81))
    zero3, zero4 = np.zeros((3, 3, 3)), np.zeros((3, 3, 3, 3))
    for n in range(27):
        unit = np.zeros(27)
        unit[n] = 1.0
        c, h, _ = _voigt_contract(unit.reshape(3, 3, 3), zero4)
        mc[:, :, n], mh[:, :, n] = c.real, h.real
    for n in range(81):
        unit = np.zeros(81)
        unit[n] = 1.0
        _, _, s = _voigt_contract(zero3, unit.reshape(3, 3, 3, 3))
        ms[:, :, n] = s.real
    return mc, mh, ms


def static_b2(ref: ReferenceMedium) -> float:
    """Coefficient of d_i d_j r in the Kelvin tensor (times 1 / 4 pi mu)."""
    return -(1.0 - ref.beta**2 / ref.alpha**2) / 2.0


def falling(m: int, q: int) -> float:
    """m (m - 2) ... (m - 2q + 2): the coefficient of r^(m - 2q) in F_q of r^m."""
    out = 1.0
    for s in range(q):
        out *= m - 2 * s
    return out


def power_F(m: int, r: NDArray) -> list[NDArray]:
    """F_0..F_4 of r^m."""
    return [falling(m, q) * r ** float(m - 2 * q) for q in range(5)]


@lru_cache(maxsize=1)
def _closed_F_functions() -> list[Callable]:
    # k is complex in general (attenuated omega): only r carries an assumption
    r = sp.symbols("r", positive=True)
    k = sp.symbols("k")
    out = [sp.exp(sp.I * k * r) / r]
    for _ in range(4):
        out.append(sp.simplify(sp.diff(out[-1], r) / r))
    return [sp.lambdify((r, k), e, "numpy") for e in out]


def _helmholtz_F(k: complex, r: NDArray) -> list[NDArray]:
    return [np.asarray(fn(r, k), dtype=complex) * np.ones_like(r) for fn in _closed_F_functions()]


def radial_tensors(F: list[NDArray], X: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Derivative tensors of orders 0-4 of a radial function, from its F_0..F_4 at points X (N, 3)."""
    eye = np.eye(3)
    x = X
    xx = np.einsum("ni,nj->nij", x, x)
    d1 = x * F[1][:, None]
    d2 = eye[None] * F[1][:, None, None] + xx * F[2][:, None, None]
    sym3 = (
        np.einsum("ij,nk->nijk", eye, x)
        + np.einsum("ik,nj->nijk", eye, x)
        + np.einsum("jk,ni->nijk", eye, x)
    )
    d3 = sym3 * F[2][:, None, None, None] + np.einsum("nij,nk->nijk", xx, x) * F[3][:, None, None, None]
    dd = (
        np.einsum("ij,kl->ijkl", eye, eye)
        + np.einsum("ik,jl->ijkl", eye, eye)
        + np.einsum("il,jk->ijkl", eye, eye)
    )
    sym4 = (
        np.einsum("ij,nkl->nijkl", eye, xx)
        + np.einsum("ik,njl->nijkl", eye, xx)
        + np.einsum("il,njk->nijkl", eye, xx)
        + np.einsum("jk,nil->nijkl", eye, xx)
        + np.einsum("jl,nik->nijkl", eye, xx)
        + np.einsum("kl,nij->nijkl", eye, xx)
    )
    d4 = (
        dd[None] * F[2][:, None, None, None, None]
        + sym4 * F[3][:, None, None, None, None]
        + np.einsum("nij,nkl->nijkl", xx, xx) * F[4][:, None, None, None, None]
    )
    return F[0], d1, d2, d3, d4


def radial_component(F: list[NDArray], X: NDArray, idx: tuple[int, ...]) -> NDArray:
    """One component d_idx f of a radial function's derivative tensor (order <= 4), shape (N,).

    The same formulas as ``radial_tensors``, evaluated for fixed indices: a few vector operations instead
    of building the (N, 3, 3, 3, 3) tensor.
    """

    def dl(a: int, b: int) -> float:
        return 1.0 if a == b else 0.0

    x = X
    n = len(idx)
    if n == 0:
        return F[0]
    if n == 1:
        return x[:, idx[0]] * F[1]
    if n == 2:
        i, j = idx
        return dl(i, j) * F[1] + x[:, i] * x[:, j] * F[2]
    if n == 3:
        i, j, k = idx
        sym = dl(i, j) * x[:, k] + dl(i, k) * x[:, j] + dl(j, k) * x[:, i]
        return sym * F[2] + x[:, i] * x[:, j] * x[:, k] * F[3]
    i, j, k, m = idx
    pairs = dl(i, j) * dl(k, m) + dl(i, k) * dl(j, m) + dl(i, m) * dl(j, k)
    mixed = (
        dl(i, j) * x[:, k] * x[:, m]
        + dl(i, k) * x[:, j] * x[:, m]
        + dl(i, m) * x[:, j] * x[:, k]
        + dl(j, k) * x[:, i] * x[:, m]
        + dl(j, m) * x[:, i] * x[:, k]
        + dl(k, m) * x[:, i] * x[:, j]
    )
    return pairs * F[2] + mixed * F[3] + x[:, i] * x[:, j] * x[:, k] * x[:, m] * F[4]


def _accumulate(acc: list[NDArray], fa: list[NDArray], fb: list[NDArray], X: NDArray) -> None:
    """Add delta_ij f_a + d_i d_j f_b to [G, Gd, Gdd]."""
    eye = np.eye(3)
    d0, d1, d2, _, _ = radial_tensors(fa, X)
    acc[0] += np.einsum("ij,n->nij", eye, d0)
    acc[1] += np.einsum("ij,nk->nijk", eye, d1)
    acc[2] += np.einsum("ij,nkl->nijkl", eye, d2)
    _, _, e2, e3, e4 = radial_tensors(fb, X)
    acc[0] += e2
    acc[1] += e3
    acc[2] += e4


def _assemble(G: NDArray, Gd: NDArray, Gdd: NDArray) -> NDArray:
    mc, mh, ms = voigt_maps()
    n = G.shape[0]
    P = np.zeros((n, 9, 9), dtype=complex)
    gd, gdd = Gd.reshape(n, 27), Gdd.reshape(n, 81)
    P[:, :3, :3] = G
    P[:, :3, 3:] = np.einsum("iaz,pz->pia", mc, gd)
    P[:, 3:, :3] = np.einsum("aiz,pz->pai", mh, gd)
    P[:, 3:, 3:] = np.einsum("abz,pz->pab", ms, gdd)
    return P


CHUNK = 40_000

#: The numerics configuration: which implementation evaluates the point propagator.
NUMERICS_YAML = Path(__file__).resolve().parents[1] / "numerics.yml"
_BACKENDS = ("python", "fortran")


@lru_cache(maxsize=1)
def point_kernel_backend() -> str:
    """``point_kernel.backend`` from ``cubic_scattering/numerics.yml``: 'python' or 'fortran'.

    Read once per process. There is no default: a missing file, key or unknown value is an error.

    Raises:
        ValueError: with what is wrong, where to fix it, a valid example, and how to recover.
    """
    example = "point_kernel:\n  backend: fortran   # one of: python, fortran"
    if not NUMERICS_YAML.is_file():
        raise ValueError(
            "the numerics configuration file is missing.\n"
            f"  Where: {NUMERICS_YAML}\n"
            f"  Valid:\n{example}\n"
            "  Fix:   create that file with the block above."
        )
    cfg = yaml.safe_load(NUMERICS_YAML.read_text()) or {}
    section = cfg.get("point_kernel") if isinstance(cfg, dict) else None
    backend = section.get("backend") if isinstance(section, dict) else None
    if backend not in _BACKENDS:
        found = "missing" if backend is None else repr(backend)
        raise ValueError(
            f"point_kernel.backend is {found}; it must be one of {', '.join(_BACKENDS)}.\n"
            f"  Where: {NUMERICS_YAML}, key point_kernel.backend\n"
            f"  Valid:\n{example}\n"
            "  Fix:   set point_kernel.backend to python or fortran (fortran needs\n"
            "         conda run -n seismic python -m cubic_scattering.fortran.build)."
        )
    return str(backend)


def kernel_9x9(
    X: NDArray, omega: complex, ref: ReferenceMedium, *, static: bool = True, dynamic: bool = True
) -> NDArray:
    """The propagator at separations X = x - x' (N, 3), shape (N, 9, 9).

    Evaluated by the implementation that ``cubic_scattering/numerics.yml`` names
    (``point_kernel.backend``): ``kernel_9x9_python`` below, or the compiled transcription of it,
    ``kernel_fortran.kernel_9x9_fortran``. The two agree to round-off (``tests/test_kernel_fortran.py``).

    Raises:
        ValueError: at r = 0, where the propagator is a distribution (integrate it with
            ``graded_voxel.blocks`` instead); or if the configuration is invalid.
        ImportError: if the configuration asks for the compiled kernel and it has not been built.
    """
    if point_kernel_backend() == "fortran":
        from .kernel_fortran import kernel_9x9_fortran  # local: kernel_fortran imports this module

        return kernel_9x9_fortran(X, omega, ref, static=static, dynamic=dynamic)
    return kernel_9x9_python(X, omega, ref, static=static, dynamic=dynamic)


def kernel_9x9_python(
    X: NDArray, omega: complex, ref: ReferenceMedium, *, static: bool = True, dynamic: bool = True
) -> NDArray:
    """The propagator in NumPy: the reference implementation, shape (N, 9, 9).

    Evaluated in chunks of CHUNK points: the fourth-derivative tensors are (N, 3, 3, 3, 3) complex, and a
    whole 6-D Gauss grid at once would need gigabytes of temporaries.

    Raises:
        ValueError: at r = 0, where the propagator is a distribution.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if len(X) > CHUNK:
        return np.concatenate(
            [
                kernel_9x9_python(X[i : i + CHUNK], omega, ref, static=static, dynamic=dynamic)
                for i in range(0, len(X), CHUNK)
            ]
        )
    return _assemble(*greens_tensors(X, omega, ref, static=static, dynamic=dynamic))


def greens_tensors(
    X: NDArray, omega: complex, ref: ReferenceMedium, *, static: bool = True, dynamic: bool = True
) -> tuple[NDArray, NDArray, NDArray]:
    """G_ij, d_k G_ij and d_k d_l G_ij at separations X = x - x' (N, 3).

    Shapes (N, 3, 3), (N, 3, 3, 3), (N, 3, 3, 3, 3); the derivative indices come last, as in
    ``resonance_tmatrix.elastodynamic_greens_deriv``. Accurate to round-off at every k r (see the module
    docstring).

    Raises:
        ValueError: at r = 0, as ``kernel_9x9``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    r = np.linalg.norm(X, axis=1)
    if np.any(r == 0.0):
        raise ValueError(
            "greens_tensors: r = 0 requested.  The propagator is a distribution at the origin; its cell "
            "integrals come from graded_voxel.blocks.near_block, never from a point value."
        )
    ka, kb = omega / ref.alpha, omega / ref.beta
    n = len(X)
    acc = [
        np.zeros((n, 3, 3), complex),
        np.zeros((n, 3, 3, 3), complex),
        np.zeros((n, 3, 3, 3, 3), complex),
    ]
    small = abs(kb) * r <= SERIES_LIMIT  # abs: lattice_greens passes a complex (attenuated) omega
    if small.any():
        Xs, rs = X[small], r[small]
        part = [a[small] for a in acc]
        # the F_q are linear in the series terms: sum them first, build the tensors once
        fa = [np.zeros_like(rs, dtype=complex) for _ in range(5)]
        fb = [np.zeros_like(rs, dtype=complex) for _ in range(5)]
        for t in range(N_SERIES):
            a_t = (1j * kb) ** t / math.factorial(t)
            b_t = ((1j * kb) ** t - (1j * ka) ** t) / (math.factorial(t) * kb**2)
            use_a = static if t == 0 else dynamic
            use_b = t >= 2 and (static if t == 2 else dynamic)
            if not (use_a or use_b):
                continue
            F = power_F(t - 1, rs)
            for q in range(5):
                if use_a:
                    fa[q] += a_t * F[q]
                if use_b:
                    fb[q] += b_t * F[q]
        _accumulate(part, fa, fb, Xs)
        for a, p in zip(acc, part, strict=True):
            a[small] = p
    large = ~small
    if large.any():
        Xl, rl = X[large], r[large]
        part = [a[large] for a in acc]
        fa_tot = _helmholtz_F(kb, rl)
        fb_tot = [
            (fb - fa) / kb**2 for fb, fa in zip(_helmholtz_F(kb, rl), _helmholtz_F(ka, rl), strict=True)
        ]
        fa_st = power_F(-1, rl)
        fb_st = [static_b2(ref) * f for f in power_F(1, rl)]
        if static and dynamic:
            fa, fb = fa_tot, fb_tot
        elif static:
            fa, fb = fa_st, fb_st
        elif dynamic:
            fa = [x - y for x, y in zip(fa_tot, fa_st, strict=True)]
            fb = [x - y for x, y in zip(fb_tot, fb_st, strict=True)]
        else:
            fa = fb = [np.zeros_like(rl)] * 5
        _accumulate(part, fa, fb, Xl)
        for a, p in zip(acc, part, strict=True):
            a[large] = p
    scale = 1.0 / (4.0 * np.pi * ref.mu)
    return acc[0] * scale, acc[1] * scale, acc[2] * scale
