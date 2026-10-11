"""Gauss-Legendre rules correct to double precision, from a stored 30-digit table.

Use this, not ``numpy.polynomial.legendre.leggauss`` or ``scipy.special.roots_legendre``: their weights carry
relative errors that grow with n (4e-15 at n = 32, up to 7e-13 at n = 40, 6e-14 at n = 48, against 30-digit
mpmath), which limit any integral built on them to a few times 1e-15.  The table
(scripts/data/gauss_legendre.json, n = 1 .. 100) is produced and checked by scripts/derive_gauss_legendre.py;
other n are computed the same way in mpmath on demand.

Also: sum a tensor-product rule one axis at a time (``numpy.tensordot`` or successive matrix products), not in a
single ``numpy.einsum`` over all n^3 nodes, which accumulates every term in one running sum and loses about
2e-15 at n = 32 where three sums of n terms lose about 2e-16.
"""

import json
from functools import cache
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

_TABLE = (
    Path(__file__).resolve().parent.parent / "scripts" / "data" / "gauss_legendre.json"
)


@cache
def _table() -> dict:
    return json.loads(_TABLE.read_text()) if _TABLE.exists() else {}


def _compute(n: int) -> tuple[NDArray, NDArray]:
    """The rule in mpmath at 30 digits: Newton on P_n from the numpy nodes, weights 2 / ((1 - x^2) P_n'^2)."""
    x0, _ = leggauss(n)
    xs, ws = [], []
    with mp.workdps(30):
        for t in x0:
            x = mp.mpf(float(t))
            for _ in range(6):
                p0, p1 = mp.mpf(1), x
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                dp = n * (x * p1 - p0) / (x * x - 1)
                x = x - p1 / dp
            p0, p1 = mp.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            xs.append(float(x))
            ws.append(float(2 / ((1 - x * x) * dp * dp)))
    return np.array(xs), np.array(ws)


@cache
def gauss_legendre(n: int) -> tuple[NDArray, NDArray]:
    """Nodes (ascending) and weights of the n-point Gauss-Legendre rule on [-1, 1], correct to double precision.

    Args:
        n: Number of nodes, >= 1.

    Returns:
        (x, w), each shape (n,).

    Raises:
        ValueError: for n < 1.
    """
    if n < 1:
        raise ValueError(f"gauss_legendre: n must be >= 1, got {n}")
    entry = _table().get(str(n))
    if entry is None:
        return _compute(n)
    xp = np.array([float(v) for v in entry["x"]])
    wp = np.array([float(v) for v in entry["w"]])
    # the table holds x >= 0 (ascending); complete by symmetry, without duplicating a node at 0
    if xp[0] == 0.0:
        x = np.concatenate([-xp[:0:-1], xp])
        w = np.concatenate([wp[:0:-1], wp])
    else:
        x = np.concatenate([-xp[::-1], xp])
        w = np.concatenate([wp[::-1], wp])
    return x, w


def gauss_interval(n: int, lo: float, hi: float) -> tuple[NDArray, NDArray]:
    """The n-point rule mapped to [lo, hi]."""
    x, w = gauss_legendre(n)
    return lo + (hi - lo) * (x + 1) / 2, w * (hi - lo) / 2
