"""Tests for the analytic small-q head of the interface plane-wave integrals (``interface_lateral``).

Validates:
  1. The vertical moments as power series against ``interface_spectral._vertical_moments``
  2. The head at order 0, whose integrand is integrable at q = 0, against plain Gauss integration from 0
  3. The sampled head's order-0 case after the log-coefficient index fix (co[-1] was read for N = 0)
"""

import numpy as np
import pytest

from cubic_scattering import interface_lateral as il
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
KA, KB = isp._key(A), isp._key(B)


@pytest.mark.parametrize("zc,s", [(-H, 1), (-3 * H, 1), (H, -1)])
def test_vertical_series(zc, s):
    poly = np.array([0.3, -1.0, 0.5])
    ser = il._vertical_series(poly, zc, H, s, 3, 40)
    for qh in (0.3, 1.0):
        q = qh / H
        direct = isp._vertical_moments(poly, zc, H, s, np.array([q]), 3)[:, 0]
        val = ser @ (q ** np.arange(40))
        assert np.abs(val - direct).max() < 1e-12 * np.abs(direct).max()


@pytest.mark.parametrize(
    "kind,rec,src", [("R", (-1, 0, 0), (-1, 0, 0)), ("T", (1, 2, 2), (-1, 0, 0))]
)
def test_head_order0_against_plain(kind, rec, src):
    qs = il.qs_for(rec, src, H)
    plain = isp._static_integral(
        0, kind, rec, src, H, KA, KB, *isp._panels(0.0, qs * H, 0.05, 20), 10, 4
    )
    head = il.head(0, kind, rec, src, H, KA, KB, qs, 10, 4)
    assert np.abs(head - plain).max() < 1e-13 * np.abs(plain).max()


def test_sampled_head_order0_fixed():
    rec, src = (-1, 0, 0), (-1, 0, 0)
    qsh = 0.5
    plain = isp._static_integral(
        0, "R", rec, src, H, KA, KB, *isp._panels(0.0, qsh, 0.05, 20), 1, 1
    )
    sampled = isp.fp_low_integral(
        0, "R", rec, src, H, KA, KB, qsh * (1 + 1e-12), 1, 1, qsh=qsh, head="sampled"
    )
    # before the fix: wrong by O(1); the sampled head itself is good to ~2e-12 here
    assert np.abs(sampled - plain).max() < 1e-11 * np.abs(plain).max()
